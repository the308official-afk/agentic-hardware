from __future__ import annotations

import unittest

from agentic_kv.controller import (
    BackendCapabilities,
    ControllerEvent,
    ControllerPolicy,
    ControllerStateStore,
    EventType,
    GatewayAdmissionControlBackendAdapter,
    GatewayDemoteRestoreBackendAdapter,
    GatewayFullControllerBackendAdapter,
    GatewayPriorityBackendAdapter,
    GatewaySpeculativePreloadBackendAdapter,
    KVAction,
    PolicyConfig,
    SchedulerAction,
    SGLangTargetedKVPrefetchBackendAdapter,
    SessionPhase,
    build_harness_controller_signal,
)


class AgenticControllerTests(unittest.TestCase):
    def test_harness_controller_signal_normalizes_tool_wait_facts(self) -> None:
        signal = build_harness_controller_signal(
            {
                "harness": "claude_code",
                "mode": "controller_full",
                "pressure_level": "p3_high",
                "session_id": "task_1",
                "label": "task_1_replay_01",
                "phase": "tool_wait",
                "prefix_id": "task_1:prefix",
                "session_generation": 2,
                "task_index": 7,
                "tool_wait_profile": "agentic_mixed",
                "tool_wait_class": "moderate",
                "tool_wait_ms": 2000,
                "tool_wait_step": 1,
                "task_replay_steps": 2,
                "prompt_hash": "abc",
                "max_tokens": 8,
                "native_cache_profile": {"enabled": True, "cache_key_seed": "repo-session"},
            },
            monotonic_ms=100,
            expected_completion_ms=2100,
            deadline_after_completion_ms=50,
            eta_uncertainty_ms=500,
        )

        self.assertEqual(signal["schema_version"], "harness_controller_signal.v1")
        self.assertEqual(signal["task"]["session_id"], "task_1")
        self.assertEqual(signal["phase"]["work_class"], "target")
        self.assertTrue(signal["phase"]["user_waiting"])
        self.assertEqual(signal["tool"]["estimated_duration_ms"], 2000)
        self.assertEqual(signal["tool"]["expected_done_at_ms"], 2100)
        self.assertEqual(signal["tool_wait"]["profile"], "agentic_mixed")
        self.assertEqual(signal["tool_wait"]["step_index"], 1)
        self.assertEqual(signal["replay"]["deadline_after_tool_ms"], 50)
        self.assertTrue(signal["cache"]["stable_prefix"])
        self.assertEqual(signal["cache"]["cache_key"], "repo-session")
        self.assertFalse(signal["scheduling"]["safe_to_demote"])
        self.assertEqual(signal["competition"]["demotable_background_requests"], 0)
        self.assertTrue(signal["cost_feedback"]["allow_background_demote"])

    def test_harness_controller_signal_marks_background_as_demotable(self) -> None:
        signal = build_harness_controller_signal(
            {
                "harness": "hatcher",
                "mode": "controller_full",
                "session_id": "task_1_pressure_000",
                "label": "task_1_pressure_000_replay",
                "phase": "pressure_filler",
                "priority_label": "low",
                "tool_wait_ms": 500,
            }
        )

        self.assertEqual(signal["phase"]["work_class"], "background")
        self.assertFalse(signal["phase"]["user_waiting"])
        self.assertTrue(signal["scheduling"]["safe_to_demote"])
        self.assertTrue(signal["scheduling"]["preemptible"])
        self.assertEqual(signal["scheduling"]["urgency"], "background")

    def test_lifecycle_events_are_idempotent_and_generation_aware(self) -> None:
        store = ControllerStateStore()
        event = ControllerEvent(
            event_id="event-1",
            event=EventType.TOOL_STARTED,
            session_id="s1",
            prefix_id="p1",
            session_generation=2,
            monotonic_ms=10,
            expected_completion_ms=100,
            deadline_after_completion_ms=50,
        )

        state, accepted = store.apply_event(event)
        self.assertTrue(accepted)
        self.assertIs(state.phase, SessionPhase.TOOL_WAIT)

        duplicate, accepted = store.apply_event(event)
        self.assertFalse(accepted)
        self.assertEqual(duplicate, state)

        stale, accepted = store.apply_event(
            ControllerEvent(
                event_id="event-stale",
                event=EventType.TOOL_COMPLETED,
                session_id="s1",
                prefix_id="p-old",
                session_generation=1,
                monotonic_ms=50,
            )
        )
        self.assertFalse(accepted)
        self.assertEqual(stale, state)

    def test_eta_update_moves_tool_wait_into_prepare_and_keeps_metadata(self) -> None:
        store = ControllerStateStore()
        signal = build_harness_controller_signal(
            {
                "harness": "hatcher",
                "mode": "controller_full",
                "session_id": "s1",
                "phase": "tool_wait",
                "tool_wait_ms": 2000,
                "active_background_requests": 8,
                "demotable_background_requests": 8,
                "background_safe_to_demote": True,
            },
            expected_completion_ms=2000,
            deadline_after_completion_ms=50,
        )
        store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=2000,
                deadline_after_completion_ms=50,
                metadata={"harness_controller_signal": signal},
            )
        )
        prepare_state, accepted = store.apply_event(
            ControllerEvent(
                event_id="event-prepare",
                event=EventType.TOOL_ETA_UPDATED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=1500,
                expected_completion_ms=2000,
                deadline_after_completion_ms=50,
                metadata={"harness_controller_signal": signal},
            )
        )

        self.assertTrue(accepted)
        self.assertIs(prepare_state.phase, SessionPhase.PREPARE)
        self.assertEqual(
            prepare_state.metadata["harness_controller_signal"],
            signal,
        )

    def test_observe_only_policy_records_intent_without_active_mutation(self) -> None:
        store = ControllerStateStore()
        state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-1",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=1000,
                deadline_after_completion_ms=50,
                execution_priority=100,
            )
        )
        policy = ControllerPolicy(PolicyConfig(observe_only=True))
        decision = policy.plan(
            state,
            BackendCapabilities(kv_demote=True, priority_queue=True, observe_only=True),
            now_ms=0,
        )

        self.assertTrue(decision.observed_only)
        self.assertTrue(decision.commands)
        self.assertTrue(all(command.kv_action is KVAction.NONE for command in decision.commands))

    def test_active_ready_policy_emits_priority_command_when_supported(self) -> None:
        store = ControllerStateStore()
        state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-ready",
                event=EventType.TOOL_COMPLETED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=120,
                execution_priority=100,
            )
        )
        policy = ControllerPolicy(PolicyConfig(observe_only=False))
        decision = policy.plan(state, BackendCapabilities(priority_queue=True), now_ms=120)

        self.assertIs(decision.phase, SessionPhase.READY)
        self.assertFalse(decision.observed_only)
        self.assertIs(decision.commands[0].scheduler_action, SchedulerAction.SET_PRIORITY)
        self.assertEqual(decision.commands[0].priority, 100)

    def test_gateway_priority_adapter_acts_only_on_set_priority(self) -> None:
        store = ControllerStateStore()
        ready_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-ready",
                event=EventType.TOOL_COMPLETED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=120,
                execution_priority=100,
            )
        )
        wait_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s2",
                prefix_id="p2",
                monotonic_ms=0,
                expected_completion_ms=1000,
            )
        )
        policy = ControllerPolicy(PolicyConfig(observe_only=False))
        backend = GatewayPriorityBackendAdapter()

        ready_decision = policy.plan(ready_state, backend.capabilities(), now_ms=120)
        ready_result = backend.apply(ready_decision.commands[0])
        self.assertTrue(ready_result.acted)

        wait_decision = policy.plan(wait_state, backend.capabilities(), now_ms=0)
        wait_result = backend.apply(wait_decision.commands[0])
        self.assertFalse(wait_result.acted)

    def test_gateway_demote_restore_adapter_acts_on_demote_priority_and_release(self) -> None:
        store = ControllerStateStore()
        wait_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=250,
                execution_priority=100,
            )
        )
        policy = ControllerPolicy(
            PolicyConfig(
                observe_only=False,
                prepare_window_ms=0,
                min_demote_idle_ms=0,
                safety_margin_ms=0,
            )
        )
        backend = GatewayDemoteRestoreBackendAdapter()

        wait_decision = policy.plan(wait_state, backend.capabilities(), now_ms=0)
        demote_result = backend.apply(wait_decision.commands[0])
        self.assertIs(wait_decision.commands[0].kv_action, KVAction.DEMOTE)
        self.assertTrue(demote_result.acted)

        ready_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-ready",
                event=EventType.TOOL_COMPLETED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=250,
                execution_priority=100,
            )
        )
        ready_decision = policy.plan(ready_state, backend.capabilities(), now_ms=250)
        priority_result = backend.apply(ready_decision.commands[0])
        self.assertIs(ready_decision.commands[0].scheduler_action, SchedulerAction.SET_PRIORITY)
        self.assertTrue(priority_result.acted)

        finished_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-finished",
                event=EventType.SESSION_FINISHED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=300,
                execution_priority=100,
            )
        )
        finished_decision = policy.plan(finished_state, backend.capabilities(), now_ms=300)
        release_result = backend.apply(finished_decision.commands[0])
        self.assertIs(finished_decision.commands[0].kv_action, KVAction.RELEASE)
        self.assertTrue(release_result.acted)

    def test_gateway_speculative_preload_adapter_acts_on_prefetch(self) -> None:
        store = ControllerStateStore()
        state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=50,
                deadline_after_completion_ms=50,
            )
        )
        policy = ControllerPolicy(PolicyConfig(observe_only=False, prepare_window_ms=100))
        backend = GatewaySpeculativePreloadBackendAdapter()

        decision = policy.plan(state, backend.capabilities(), now_ms=0)
        result = backend.apply(decision.commands[0])

        self.assertFalse(decision.observed_only)
        self.assertIs(decision.commands[0].kv_action, KVAction.PREFETCH)
        self.assertTrue(result.acted)

    def test_gateway_admission_adapter_acts_on_prefetch_budget_priority_and_release(self) -> None:
        store = ControllerStateStore()
        wait_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=50,
                deadline_after_completion_ms=50,
                execution_priority=100,
            )
        )
        policy = ControllerPolicy(PolicyConfig(observe_only=False, prepare_window_ms=100))
        backend = GatewayAdmissionControlBackendAdapter()

        wait_decision = policy.plan(wait_state, backend.capabilities(), now_ms=0)
        wait_results = [backend.apply(command) for command in wait_decision.commands]
        self.assertIn(KVAction.PREFETCH, {command.kv_action for command in wait_decision.commands})
        self.assertIn(
            SchedulerAction.SET_BACKGROUND_PREFILL_BUDGET,
            {command.scheduler_action for command in wait_decision.commands},
        )
        self.assertTrue(all(result.acted for result in wait_results))

        ready_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-ready",
                event=EventType.TOOL_COMPLETED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=50,
                execution_priority=100,
            )
        )
        ready_decision = policy.plan(ready_state, backend.capabilities(), now_ms=50)
        ready_result = backend.apply(ready_decision.commands[0])
        self.assertIs(ready_decision.commands[0].scheduler_action, SchedulerAction.SET_PRIORITY)
        self.assertTrue(ready_result.acted)

        finished_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-finished",
                event=EventType.SESSION_FINISHED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=100,
                execution_priority=100,
            )
        )
        finished_decision = policy.plan(finished_state, backend.capabilities(), now_ms=100)
        release_result = backend.apply(finished_decision.commands[0])
        self.assertIs(finished_decision.commands[0].kv_action, KVAction.RELEASE)
        self.assertTrue(release_result.acted)

    def test_gateway_full_controller_adapter_combines_demote_priority_budget_and_release_without_prefetch(self) -> None:
        store = ControllerStateStore()
        wait_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=50,
                deadline_after_completion_ms=50,
                execution_priority=100,
            )
        )
        policy = ControllerPolicy(
            PolicyConfig(
                observe_only=False,
                prepare_window_ms=0,
                min_demote_idle_ms=0,
                safety_margin_ms=0,
            )
        )
        backend = GatewayFullControllerBackendAdapter()

        wait_decision = policy.plan(wait_state, backend.capabilities(), now_ms=0)
        wait_results = [backend.apply(command) for command in wait_decision.commands]
        self.assertIn(KVAction.DEMOTE, {command.kv_action for command in wait_decision.commands})
        self.assertTrue(all(result.acted for result in wait_results))

        prepare_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-prepare",
                event=EventType.TOOL_ETA_UPDATED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=50,
                expected_completion_ms=50,
                deadline_after_completion_ms=50,
                execution_priority=100,
            )
        )
        prepare_decision = policy.plan(prepare_state, backend.capabilities(), now_ms=50)
        self.assertNotIn(KVAction.PREFETCH, {command.kv_action for command in prepare_decision.commands})
        budget_command = next(
            command
            for command in prepare_decision.commands
            if command.scheduler_action is SchedulerAction.SET_BACKGROUND_PREFILL_BUDGET
        )
        self.assertTrue(backend.apply(budget_command).acted)

        ready_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-ready",
                event=EventType.TOOL_COMPLETED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=50,
                execution_priority=100,
            )
        )
        ready_decision = policy.plan(ready_state, backend.capabilities(), now_ms=50)
        ready_result = backend.apply(ready_decision.commands[0])
        self.assertIs(ready_decision.commands[0].scheduler_action, SchedulerAction.SET_PRIORITY)
        self.assertTrue(ready_result.acted)

        finished_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-finished",
                event=EventType.SESSION_FINISHED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=100,
                execution_priority=100,
            )
        )
        finished_decision = policy.plan(finished_state, backend.capabilities(), now_ms=100)
        finished_result = backend.apply(finished_decision.commands[0])
        self.assertIs(finished_decision.commands[0].kv_action, KVAction.RELEASE)
        self.assertTrue(finished_result.acted)

    def test_signal_aware_policy_waits_for_prepare_window_before_demoting(self) -> None:
        store = ControllerStateStore()
        signal = build_harness_controller_signal(
            {
                "harness": "hatcher",
                "mode": "controller_full",
                "session_id": "s1",
                "phase": "tool_wait",
                "priority_label": "high",
                "tool_wait_ms": 2000,
                "active_background_requests": 8,
                "demotable_background_requests": 8,
                "background_safe_to_demote": True,
            },
            expected_completion_ms=2000,
            deadline_after_completion_ms=50,
        )
        state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=2000,
                deadline_after_completion_ms=50,
                execution_priority=100,
                metadata={"harness_controller_signal": signal},
            )
        )
        policy = ControllerPolicy(
            PolicyConfig(observe_only=False, prepare_window_ms=0, signal_timing_enabled=True)
        )
        backend = GatewayFullControllerBackendAdapter()

        wait_decision = policy.plan(state, backend.capabilities(), now_ms=0)
        self.assertNotIn(KVAction.DEMOTE, {command.kv_action for command in wait_decision.commands})
        self.assertIn("waiting for prepare window", wait_decision.reason)

        prepare_state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-prepare",
                event=EventType.TOOL_ETA_UPDATED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=1500,
                expected_completion_ms=2000,
                deadline_after_completion_ms=50,
                execution_priority=100,
                metadata={"harness_controller_signal": signal},
            )
        )
        prepare_decision = policy.plan(prepare_state, backend.capabilities(), now_ms=1500)
        self.assertIn(KVAction.DEMOTE, {command.kv_action for command in prepare_decision.commands})

    def test_signal_aware_policy_skips_demote_for_short_waits(self) -> None:
        store = ControllerStateStore()
        signal = build_harness_controller_signal(
            {
                "harness": "hatcher",
                "mode": "controller_full",
                "session_id": "s1",
                "phase": "tool_wait",
                "priority_label": "high",
                "tool_wait_ms": 250,
                "active_background_requests": 8,
                "demotable_background_requests": 8,
                "background_safe_to_demote": True,
            },
            expected_completion_ms=250,
            deadline_after_completion_ms=50,
        )
        state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=250,
                deadline_after_completion_ms=50,
                execution_priority=100,
                metadata={"harness_controller_signal": signal},
            )
        )
        policy = ControllerPolicy(
            PolicyConfig(observe_only=False, prepare_window_ms=0, signal_timing_enabled=True)
        )
        backend = GatewayFullControllerBackendAdapter()

        decision = policy.plan(state, backend.capabilities(), now_ms=250)
        self.assertNotIn(KVAction.DEMOTE, {command.kv_action for command in decision.commands})
        self.assertIn("short tool wait", decision.reason)

    def test_targeted_kv_prefetch_adapter_records_unavailable_hook(self) -> None:
        store = ControllerStateStore()
        state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=50,
                deadline_after_completion_ms=50,
            )
        )
        policy = ControllerPolicy(PolicyConfig(observe_only=False, prepare_window_ms=100))
        backend = SGLangTargetedKVPrefetchBackendAdapter(direct_hook_available=False)

        decision = policy.plan(state, backend.capabilities(), now_ms=0)
        result = backend.apply(decision.commands[0])

        self.assertIs(decision.commands[0].kv_action, KVAction.PREFETCH)
        self.assertTrue(result.accepted)
        self.assertFalse(result.acted)
        self.assertIn("unavailable", result.reason)

    def test_targeted_kv_prefetch_adapter_acts_when_hook_available(self) -> None:
        store = ControllerStateStore()
        state, _ = store.apply_event(
            ControllerEvent(
                event_id="event-wait",
                event=EventType.TOOL_STARTED,
                session_id="s1",
                prefix_id="p1",
                monotonic_ms=0,
                expected_completion_ms=50,
                deadline_after_completion_ms=50,
            )
        )
        policy = ControllerPolicy(PolicyConfig(observe_only=False, prepare_window_ms=100))
        backend = SGLangTargetedKVPrefetchBackendAdapter(direct_hook_available=True)

        decision = policy.plan(state, backend.capabilities(), now_ms=0)
        result = backend.apply(decision.commands[0])

        self.assertIs(decision.commands[0].kv_action, KVAction.PREFETCH)
        self.assertTrue(result.accepted)
        self.assertTrue(result.acted)


if __name__ == "__main__":
    unittest.main()
