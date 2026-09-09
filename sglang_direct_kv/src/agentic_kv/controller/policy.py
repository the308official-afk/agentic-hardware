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
    short_wait_no_demote_ms: int = 500
    medium_wait_ms: int = 3_000
    long_wait_ms: int = 10_000
    medium_prepare_window_ms: int = 500
    long_prepare_window_ms: int = 1_000
    very_long_prepare_window_ms: int = 1_500
    max_background_slowdown_ratio: float = 2.0
    signal_timing_enabled: bool = False
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

        if self.config.signal_timing_enabled and self._uses_harness_signal(state):
            prepare_window_ms = self._prepare_window_ms_for_state(state)
            if time_to_completion_ms <= prepare_window_ms:
                reasons.append("tool return is inside timed prepare window")
                return self._prepare_window_commands(state, capabilities, now_ms, active, reasons)
            wait_ms = self._tool_wait_duration_ms(state)
            if wait_ms is not None and wait_ms < self.config.short_wait_no_demote_ms:
                reasons.append("short tool wait: use replay priority without early background demotion")
            else:
                reasons.append(f"waiting for prepare window; {time_to_completion_ms} ms until replay")
            return commands

        if time_to_completion_ms <= self.config.prepare_window_ms:
            reasons.append("tool return is inside prepare window")
            return self._prepare_window_commands(state, capabilities, now_ms, active, reasons)
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
        return self._prepare_window_commands(state, capabilities, now_ms, active, reasons)

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

    def _prepare_window_commands(
        self,
        state: SessionState,
        capabilities: BackendCapabilities,
        now_ms: int,
        active: bool,
        reasons: list[str],
    ) -> list[ControllerCommand]:
        commands: list[ControllerCommand] = []
        if active and capabilities.kv_demote and self._should_demote_background(state, reasons):
            commands.append(
                self._command(
                    state,
                    now_ms,
                    SchedulerAction.OBSERVE_ONLY,
                    KVAction.DEMOTE,
                    "demote background work during replay-critical window",
                )
            )
        if active and capabilities.kv_prefetch:
            commands.append(
                self._command(
                    state,
                    now_ms,
                    SchedulerAction.OBSERVE_ONLY,
                    KVAction.PREFETCH,
                    "restore KV before replay",
                )
            )
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

    def _should_demote_background(self, state: SessionState, reasons: list[str]) -> bool:
        signal = self._harness_signal(state)
        if not signal:
            return True
        if not self._signal_bool(signal, ("phase", "user_waiting"), default=False):
            reasons.append("not user-waiting work; skip background demotion")
            return False
        if not self._signal_bool(signal, ("replay", "likely"), default=False):
            reasons.append("replay is not marked likely; skip background demotion")
            return False
        wait_ms = self._tool_wait_duration_ms(state)
        if wait_ms is not None and wait_ms < self.config.short_wait_no_demote_ms:
            reasons.append("short tool wait: skip background demotion")
            return False
        if not self._signal_bool(signal, ("competition", "background_safe_to_demote"), default=False):
            reasons.append("no safe background demotion signal")
            return False
        demotable = self._signal_int(signal, ("competition", "demotable_background_requests"), default=0)
        if demotable is None or demotable <= 0:
            reasons.append("no demotable background requests")
            return False
        if not self._signal_bool(signal, ("cost_feedback", "allow_background_demote"), default=True):
            reasons.append("cost feedback disabled background demotion")
            return False
        slowdown = self._signal_float(signal, ("cost_feedback", "recent_background_slowdown_ratio"))
        if slowdown is not None and slowdown > self.config.max_background_slowdown_ratio:
            reasons.append(f"recent background slowdown {slowdown:.2f}x exceeds guardrail")
            return False
        reasons.append(f"{demotable} demotable background requests available")
        return True

    def _prepare_window_ms_for_state(self, state: SessionState) -> int:
        wait_ms = self._tool_wait_duration_ms(state)
        if wait_ms is None:
            return self.config.prepare_window_ms
        if wait_ms < self.config.short_wait_no_demote_ms:
            return min(wait_ms, self.config.prepare_window_ms)
        if wait_ms <= self.config.medium_wait_ms:
            return min(wait_ms, max(self.config.prepare_window_ms, self.config.medium_prepare_window_ms))
        if wait_ms <= self.config.long_wait_ms:
            return min(wait_ms, max(self.config.prepare_window_ms, self.config.long_prepare_window_ms))
        return min(wait_ms, max(self.config.prepare_window_ms, self.config.very_long_prepare_window_ms))

    def _tool_wait_duration_ms(self, state: SessionState) -> int | None:
        signal = self._harness_signal(state)
        if signal:
            wait_ms = self._signal_int(signal, ("tool_wait", "duration_ms"))
            if wait_ms is not None:
                return wait_ms
            return self._signal_int(signal, ("tool", "estimated_duration_ms"))
        if state.expected_completion_ms is None:
            return None
        return max(0, state.expected_completion_ms - state.last_event_ms)

    def _uses_harness_signal(self, state: SessionState) -> bool:
        signal = self._harness_signal(state)
        return bool(signal and signal.get("schema_version") == "harness_controller_signal.v1")

    @staticmethod
    def _harness_signal(state: SessionState) -> dict[str, object]:
        signal = state.metadata.get("harness_controller_signal")
        return signal if isinstance(signal, dict) else {}

    @staticmethod
    def _signal_value(signal: dict[str, object], path: tuple[str, ...]) -> object:
        current: object = signal
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _signal_bool(self, signal: dict[str, object], path: tuple[str, ...], *, default: bool) -> bool:
        value = self._signal_value(signal, path)
        if value in (None, ""):
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _signal_int(
        self,
        signal: dict[str, object],
        path: tuple[str, ...],
        *,
        default: int | None = None,
    ) -> int | None:
        value = self._signal_value(signal, path)
        try:
            if value in (None, ""):
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _signal_float(self, signal: dict[str, object], path: tuple[str, ...]) -> float | None:
        value = self._signal_value(signal, path)
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _id(*parts: str) -> str:
        digest = hashlib.sha1(":".join(parts).encode("utf-8")).hexdigest()[:16]
        return f"{parts[0]}-{digest}"
