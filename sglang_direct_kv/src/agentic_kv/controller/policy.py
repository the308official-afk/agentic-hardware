from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .models import (
    BackendCapabilities,
    ControllerCommand,
    ControllerDecision,
    KVAction,
    SchedulerAction,
    SessionPhase,
)
from .state_store import SessionState
from .timing_estimator import TimingEstimator


@dataclass(frozen=True)
class PolicyConfig:
    safety_margin_ms: int = 6
    prepare_window_ms: int = 100
    min_demote_idle_ms: int = 250
    urgent_priority: int = 100
    background_prefill_budget_tokens: int = 1024
    command_ttl_ms: int = 100
    observe_only: bool = True


class ControllerPolicy:
    """Backend-neutral policy planner for agent replay preparation."""

    def __init__(self, config: PolicyConfig | None = None, timing: TimingEstimator | None = None) -> None:
        self.config = config or PolicyConfig()
        self.timing = timing or TimingEstimator()

    def plan(
        self,
        state: SessionState,
        capabilities: BackendCapabilities,
        *,
        now_ms: int,
    ) -> ControllerDecision:
        commands: list[ControllerCommand] = []
        reasons: list[str] = []
        active = not self.config.observe_only

        if state.phase is SessionPhase.TOOL_WAIT:
            commands.extend(self._plan_tool_wait(state, capabilities, now_ms, active, reasons))
        elif state.phase is SessionPhase.PREPARE:
            commands.extend(self._plan_prepare(state, capabilities, now_ms, active, reasons))
        elif state.phase is SessionPhase.READY:
            commands.extend(self._plan_ready(state, capabilities, now_ms, active, reasons))
        elif state.phase is SessionPhase.FINISHED:
            commands.extend(self._plan_finished(state, capabilities, now_ms, active, reasons))
        else:
            reasons.append("session is active; observe only")

        if not commands:
            commands.append(
                self._command(
                    state,
                    now_ms,
                    SchedulerAction.OBSERVE_ONLY,
                    KVAction.NONE,
                    "no supported active action for current state",
                )
            )

        return ControllerDecision(
            decision_id=self._id("decision", state.session_id, state.last_event_id, str(now_ms)),
            session_id=state.session_id,
            phase=state.phase,
            commands=tuple(commands),
            reason="; ".join(reasons) if reasons else "observe only",
            observed_only=not active or all(command.scheduler_action is SchedulerAction.OBSERVE_ONLY and command.kv_action is KVAction.NONE for command in commands),
        )

    def _plan_tool_wait(
        self,
        state: SessionState,
        capabilities: BackendCapabilities,
        now_ms: int,
        active: bool,
        reasons: list[str],
    ) -> list[ControllerCommand]:
        commands: list[ControllerCommand] = []
        expected_idle_ms = max(0, (state.expected_completion_ms or now_ms) - now_ms)
        time_to_completion_ms = expected_idle_ms
        if time_to_completion_ms <= self.config.prepare_window_ms:
            reasons.append("tool return is inside prepare window")
            if active and capabilities.kv_prefetch:
                commands.append(self._command(state, now_ms, SchedulerAction.OBSERVE_ONLY, KVAction.PREFETCH, "restore KV before replay"))
            if active and capabilities.background_prefill_budget:
                commands.append(
                    self._command(
                        state,
                        now_ms,
                        SchedulerAction.SET_BACKGROUND_PREFILL_BUDGET,
                        KVAction.NONE,
                        "reduce background prefill near replay deadline",
                        background_prefill_budget_tokens=self.config.background_prefill_budget_tokens,
                    )
                )
            if not commands:
                reasons.append("prepare actions not active or not supported")
            return commands
        restore_p99 = self.timing.p99("kv_restore_ms", self.timing.default_p99_ms)
        demote_p99 = self.timing.p99("kv_demote_ms", self.timing.default_p99_ms)
        threshold = demote_p99 + restore_p99 + self.config.safety_margin_ms
        if expected_idle_ms >= max(self.config.min_demote_idle_ms, threshold):
            reasons.append(f"tool wait has {expected_idle_ms} ms idle window")
            if active and capabilities.kv_demote:
                commands.append(self._command(state, now_ms, SchedulerAction.OBSERVE_ONLY, KVAction.DEMOTE, "idle tool wait makes HBM demotion worthwhile"))
            else:
                reasons.append("kv demote not active or not supported")
        else:
            reasons.append("tool wait too short for safe demotion")
        return commands

    def _plan_prepare(
        self,
        state: SessionState,
        capabilities: BackendCapabilities,
        now_ms: int,
        active: bool,
        reasons: list[str],
    ) -> list[ControllerCommand]:
        commands: list[ControllerCommand] = []
        reasons.append("session is in prepare window")
        if active and capabilities.kv_prefetch:
            commands.append(self._command(state, now_ms, SchedulerAction.OBSERVE_ONLY, KVAction.PREFETCH, "restore KV before replay"))
        if active and capabilities.background_prefill_budget:
            commands.append(
                self._command(
                    state,
                    now_ms,
                    SchedulerAction.SET_BACKGROUND_PREFILL_BUDGET,
                    KVAction.NONE,
                    "reduce background prefill near replay deadline",
                    background_prefill_budget_tokens=self.config.background_prefill_budget_tokens,
                )
            )
        if not commands:
            reasons.append("prepare actions not active or not supported")
        return commands

    def _plan_ready(
        self,
        state: SessionState,
        capabilities: BackendCapabilities,
        now_ms: int,
        active: bool,
        reasons: list[str],
    ) -> list[ControllerCommand]:
        reasons.append("tool completed; replay should run at next safe boundary")
        if active and capabilities.priority_queue:
            return [
                self._command(
                    state,
                    now_ms,
                    SchedulerAction.SET_PRIORITY,
                    KVAction.NONE,
                    "raise replay priority",
                    priority=max(self.config.urgent_priority, state.execution_priority),
                )
            ]
        reasons.append("priority queue not active or not supported")
        return []

    def _plan_finished(
        self,
        state: SessionState,
        capabilities: BackendCapabilities,
        now_ms: int,
        active: bool,
        reasons: list[str],
    ) -> list[ControllerCommand]:
        reasons.append("session finished; release tracked resources")
        if active and capabilities.kv_release:
            return [self._command(state, now_ms, SchedulerAction.OBSERVE_ONLY, KVAction.RELEASE, "release session KV")]
        reasons.append("kv release not active or not supported")
        return []

    def _command(
        self,
        state: SessionState,
        now_ms: int,
        scheduler_action: SchedulerAction,
        kv_action: KVAction,
        reason: str,
        *,
        priority: int | None = None,
        background_prefill_budget_tokens: int | None = None,
    ) -> ControllerCommand:
        return ControllerCommand(
            command_id=self._id("cmd", state.session_id, state.last_event_id, scheduler_action.value, kv_action.value),
            session_id=state.session_id,
            prefix_id=state.prefix_id,
            session_generation=state.session_generation,
            scheduler_action=scheduler_action,
            kv_action=kv_action,
            priority=priority,
            background_prefill_budget_tokens=background_prefill_budget_tokens,
            execute_at_ms=now_ms,
            expires_at_ms=now_ms + self.config.command_ttl_ms,
            reason=reason,
        )

    @staticmethod
    def _id(*parts: str) -> str:
        digest = hashlib.sha1(":".join(parts).encode("utf-8")).hexdigest()[:16]
        return f"{parts[0]}-{digest}"
