"""Portable agent-aware controller primitives.

The controller package owns backend-neutral lifecycle state and policy. Backend
packages adapt its versioned commands to a specific SGLang release.
"""

from .backend import (
    BackendAdapter,
    BackendActionResult,
    GatewayAdmissionControlBackendAdapter,
    GatewayDemoteRestoreBackendAdapter,
    GatewayFullControllerBackendAdapter,
    GatewayPriorityBackendAdapter,
    GatewaySpeculativePreloadBackendAdapter,
    ObserveOnlyBackendAdapter,
    SGLangTargetedKVPrefetchBackendAdapter,
)
from .harness_signal import (
    HARNESS_CONTROLLER_SIGNAL_SCHEMA,
    HarnessControllerSignal,
    build_harness_controller_signal,
)
from .models import (
    BackendCapabilities,
    ControllerCommand,
    ControllerDecision,
    ControllerEvent,
    EventType,
    KVAction,
    SchedulerAction,
    SessionPhase,
)
from .policy import ControllerPolicy, PolicyConfig
from .state_store import ControllerStateStore, SessionState
from .timing_estimator import TimingEstimator

__all__ = [
    "BackendActionResult",
    "BackendAdapter",
    "BackendCapabilities",
    "ControllerCommand",
    "ControllerDecision",
    "ControllerEvent",
    "ControllerPolicy",
    "ControllerStateStore",
    "EventType",
    "GatewayAdmissionControlBackendAdapter",
    "GatewayDemoteRestoreBackendAdapter",
    "GatewayFullControllerBackendAdapter",
    "GatewayPriorityBackendAdapter",
    "GatewaySpeculativePreloadBackendAdapter",
    "HARNESS_CONTROLLER_SIGNAL_SCHEMA",
    "HarnessControllerSignal",
    "KVAction",
    "ObserveOnlyBackendAdapter",
    "PolicyConfig",
    "SchedulerAction",
    "SGLangTargetedKVPrefetchBackendAdapter",
    "SessionPhase",
    "SessionState",
    "TimingEstimator",
    "build_harness_controller_signal",
]
