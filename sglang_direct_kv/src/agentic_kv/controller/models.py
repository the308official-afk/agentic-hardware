from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "agentic_controller.v1"


class EventType(str, Enum):
    TOOL_STARTED = "tool_started"
    TOOL_ETA_UPDATED = "tool_eta_updated"
    TOOL_COMPLETED = "tool_completed"
    SESSION_FINISHED = "session_finished"


class SessionPhase(str, Enum):
    ACTIVE = "active"
    TOOL_WAIT = "tool_wait"
    PREPARE = "prepare"
    READY = "ready"
    FINISHED = "finished"


class SchedulerAction(str, Enum):
    OBSERVE_ONLY = "observe_only"
    SET_PRIORITY = "set_priority"
    SET_BACKGROUND_PREFILL_BUDGET = "set_background_prefill_budget"


class KVAction(str, Enum):
    NONE = "none"
    DEMOTE = "demote"
    PREFETCH = "prefetch"
    RELEASE = "release"


@dataclass(frozen=True)
class ControllerEvent:
    event_id: str
    event: EventType
    session_id: str
    prefix_id: str
    monotonic_ms: int
    session_generation: int = 0
    expected_completion_ms: int | None = None
    eta_uncertainty_ms: int | None = None
    deadline_after_completion_ms: int | None = None
    execution_priority: int = 0
    hbm_residency_priority: int = 0
    host_retention_priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["event"] = self.event.value
        return row


@dataclass(frozen=True)
class BackendCapabilities:
    priority_queue: bool = False
    background_prefill_budget: bool = False
    safe_preemption: bool = False
    kv_demote: bool = False
    kv_prefetch: bool = False
    kv_release: bool = False
    live_metrics: bool = False
    observe_only: bool = True
    backend_name: str = "unknown"
    backend_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerCommand:
    command_id: str
    session_id: str
    prefix_id: str
    session_generation: int
    scheduler_action: SchedulerAction = SchedulerAction.OBSERVE_ONLY
    kv_action: KVAction = KVAction.NONE
    priority: int | None = None
    background_prefill_budget_tokens: int | None = None
    execute_at_ms: int | None = None
    expires_at_ms: int | None = None
    reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["scheduler_action"] = self.scheduler_action.value
        row["kv_action"] = self.kv_action.value
        return row


@dataclass(frozen=True)
class ControllerDecision:
    decision_id: str
    session_id: str
    phase: SessionPhase
    commands: tuple[ControllerCommand, ...]
    reason: str
    observed_only: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "session_id": self.session_id,
            "phase": self.phase.value,
            "commands": [command.to_dict() for command in self.commands],
            "reason": self.reason,
            "observed_only": self.observed_only,
        }
