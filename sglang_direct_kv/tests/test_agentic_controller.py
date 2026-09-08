from __future__ import annotations

import unittest

from agentic_kv.controller import (
    BackendCapabilities,
    ControllerEvent,
    ControllerPolicy,
    ControllerStateStore,
    EventType,
    KVAction,
    PolicyConfig,
    SchedulerAction,
    SessionPhase,
)


class AgenticControllerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
