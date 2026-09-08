from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol

from .models import BackendCapabilities, ControllerCommand, KVAction, SchedulerAction


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


class GatewayDemoteRestoreBackendAdapter:
    """Adapter for portable demote/restore experiments.

    Demotion and restoration are lowered at the gateway request boundary. This
    adapter records that the controller command is accepted, without depending
    on a private SGLang API for in-place queue manipulation.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            priority_queue=True,
            kv_demote=True,
            kv_release=True,
            live_metrics=True,
            observe_only=False,
            backend_name="controller_demote_restore",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action is KVAction.DEMOTE:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller demote accepted for gateway lowering of background traffic",
                backend_name=self._capabilities.backend_name,
            )
        if command.scheduler_action is SchedulerAction.SET_PRIORITY and command.priority is not None:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller replay priority accepted during demote/restore window",
                backend_name=self._capabilities.backend_name,
            )
        if command.kv_action is KVAction.RELEASE:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller restore/release accepted after replay",
                backend_name=self._capabilities.backend_name,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="controller command recorded but not active in demote/restore mode",
            backend_name=self._capabilities.backend_name,
        )


class GatewaySpeculativePreloadBackendAdapter:
    """Adapter for controller-driven gateway speculative KV preload.

    The controller remains backend-neutral: it emits a KV prefetch command, and
    the experiment gateway lowers that accepted command into the existing
    background warmup request path.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            kv_prefetch=True,
            observe_only=False,
            backend_name="controller_speculative_preload",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action.value == "prefetch":
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller KV prefetch accepted for gateway speculative preload",
                backend_name=self._capabilities.backend_name,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="controller command recorded but not active in preload-only mode",
            backend_name=self._capabilities.backend_name,
        )


class GatewayAdmissionControlBackendAdapter:
    """Adapter for controller-driven speculative work admission.

    The adapter accepts priority, prefetch, and background-budget commands. The
    actual admit/skip choice is made by the portable experiment driver using
    pressure knobs and the accepted command envelope.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            priority_queue=True,
            background_prefill_budget=True,
            kv_prefetch=True,
            kv_release=True,
            live_metrics=True,
            observe_only=False,
            backend_name="controller_admission_control",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action is KVAction.PREFETCH:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller prefetch accepted for admission-gated warmup",
                backend_name=self._capabilities.backend_name,
            )
        if command.scheduler_action is SchedulerAction.SET_BACKGROUND_PREFILL_BUDGET:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller background prefill budget accepted for admission gating",
                backend_name=self._capabilities.backend_name,
            )
        if command.scheduler_action is SchedulerAction.SET_PRIORITY and command.priority is not None:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller replay priority accepted for admission-control mode",
                backend_name=self._capabilities.backend_name,
            )
        if command.kv_action is KVAction.RELEASE:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller admission-control release recorded after replay",
                backend_name=self._capabilities.backend_name,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="controller command recorded but not active in admission-control mode",
            backend_name=self._capabilities.backend_name,
        )


class SGLangTargetedKVPrefetchBackendAdapter:
    """Adapter for a future direct SGLang host-to-device KV prefetch hook.

    This adapter intentionally separates controller intent from SGLang internals.
    If a stable direct hook is unavailable, the command is accepted and recorded
    but not reported as acted. That keeps Phase 4 portable across SGLang
    versions while still producing honest proof rows.
    """

    def __init__(
        self,
        capabilities: BackendCapabilities | None = None,
        *,
        direct_hook_available: bool | None = None,
    ) -> None:
        if direct_hook_available is None:
            direct_hook_available = os.environ.get("AGENTIC_KV_TARGETED_PREFETCH_HOOK", "").lower() in {
                "1",
                "true",
                "yes",
            }
        self.direct_hook_available = direct_hook_available
        self._capabilities = capabilities or BackendCapabilities(
            kv_prefetch=True,
            live_metrics=True,
            observe_only=False,
            backend_name="controller_targeted_kv_prefetch",
            backend_version="direct_hook_available=1" if direct_hook_available else "direct_hook_available=0",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action.value != "prefetch":
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=False,
                reason="controller command recorded but not active in targeted-prefetch mode",
                backend_name=self._capabilities.backend_name,
            )
        if not self.direct_hook_available:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=False,
                reason="targeted SGLang KV prefetch hook unavailable in this SGLang version",
                backend_name=self._capabilities.backend_name,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=True,
            reason="targeted SGLang KV prefetch hook accepted",
            backend_name=self._capabilities.backend_name,
        )
