from __future__ import annotations

from dataclasses import dataclass, field, replace

from .models import ControllerEvent, EventType, SessionPhase


@dataclass(frozen=True)
class SessionState:
    session_id: str
    prefix_id: str
    phase: SessionPhase = SessionPhase.ACTIVE
    session_generation: int = 0
    last_event_id: str = ""
    expected_completion_ms: int | None = None
    eta_uncertainty_ms: int | None = None
    deadline_after_completion_ms: int | None = None
    execution_priority: int = 0
    hbm_residency_priority: int = 0
    host_retention_priority: int = 0
    last_event_ms: int = 0
    seen_event_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def replay_deadline_ms(self) -> int | None:
        if self.expected_completion_ms is None or self.deadline_after_completion_ms is None:
            return None
        return self.expected_completion_ms + self.deadline_after_completion_ms


class ControllerStateStore:
    """Idempotent, generation-aware in-memory state store.

    The class intentionally has no SGLang imports. Production deployments can
    swap it for a persistent store that preserves the same event semantics.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def apply_event(self, event: ControllerEvent) -> tuple[SessionState, bool]:
        current = self._sessions.get(event.session_id)
        if current and event.event_id in current.seen_event_ids:
            return current, False
        if current and event.session_generation < current.session_generation:
            return current, False

        base = current or SessionState(
            session_id=event.session_id,
            prefix_id=event.prefix_id,
            session_generation=event.session_generation,
        )
        if event.session_generation > base.session_generation:
            base = SessionState(
                session_id=event.session_id,
                prefix_id=event.prefix_id,
                session_generation=event.session_generation,
            )

        phase = self._phase_after_event(base.phase, event.event)
        updated = replace(
            base,
            prefix_id=event.prefix_id or base.prefix_id,
            phase=phase,
            session_generation=event.session_generation,
            last_event_id=event.event_id,
            expected_completion_ms=event.expected_completion_ms
            if event.expected_completion_ms is not None
            else base.expected_completion_ms,
            eta_uncertainty_ms=event.eta_uncertainty_ms
            if event.eta_uncertainty_ms is not None
            else base.eta_uncertainty_ms,
            deadline_after_completion_ms=event.deadline_after_completion_ms
            if event.deadline_after_completion_ms is not None
            else base.deadline_after_completion_ms,
            execution_priority=event.execution_priority or base.execution_priority,
            hbm_residency_priority=event.hbm_residency_priority or base.hbm_residency_priority,
            host_retention_priority=event.host_retention_priority or base.host_retention_priority,
            last_event_ms=event.monotonic_ms,
            seen_event_ids=base.seen_event_ids | frozenset({event.event_id}),
        )
        self._sessions[event.session_id] = updated
        return updated, True

    @staticmethod
    def _phase_after_event(current: SessionPhase, event: EventType) -> SessionPhase:
        if current is SessionPhase.FINISHED:
            return current
        if event is EventType.TOOL_STARTED:
            return SessionPhase.TOOL_WAIT
        if event is EventType.TOOL_ETA_UPDATED:
            return SessionPhase.PREPARE if current is SessionPhase.PREPARE else current
        if event is EventType.TOOL_COMPLETED:
            return SessionPhase.READY
        if event is EventType.SESSION_FINISHED:
            return SessionPhase.FINISHED
        return current
