from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import BackendCapabilities, ControllerCommand, SchedulerAction


@dataclass(frozen=True)
class BackendActionResult:
    command_id: str
    accepted: bool
    acted: bool
    reason: str
    backend_name: str = "observe_only"

    def to_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id,
            "accepted": self.accepted,
            "acted": self.acted,
            "reason": self.reason,
            "backend_name": self.backend_name,
        }


class BackendAdapter(Protocol):
    def capabilities(self) -> BackendCapabilities:
        ...

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        ...


class ObserveOnlyBackendAdapter:
    """Portable adapter that records intent without mutating SGLang."""

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            observe_only=True,
            live_metrics=True,
            backend_name="observe_only",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="observe-only adapter recorded command intent",
            backend_name=self._capabilities.backend_name,
        )


class GatewayPriorityBackendAdapter:
    """Adapter for scheduler-only mode.

    The actual SGLang mutation happens at the request boundary, where the
    gateway lowers this accepted controller decision into the request priority
    field. Keeping this adapter side-effect-free makes the controller portable.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            priority_queue=True,
            observe_only=False,
            backend_name="controller_scheduler_priority",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.scheduler_action is SchedulerAction.SET_PRIORITY and command.priority is not None:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller priority accepted for gateway lowering",
                backend_name=self._capabilities.backend_name,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="controller command recorded but not active in scheduler-only mode",
            backend_name=self._capabilities.backend_name,
        )
