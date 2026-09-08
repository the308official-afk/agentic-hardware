"""Portable agent-aware controller primitives.

The controller package owns backend-neutral lifecycle state and policy. Backend
packages adapt its versioned commands to a specific SGLang release.
"""

from .backend import (
    BackendAdapter,
    BackendActionResult,
    GatewayPriorityBackendAdapter,
    GatewaySpeculativePreloadBackendAdapter,
    ObserveOnlyBackendAdapter,
    SGLangTargetedKVPrefetchBackendAdapter,
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
    "GatewayPriorityBackendAdapter",
    "GatewaySpeculativePreloadBackendAdapter",
    "KVAction",
    "ObserveOnlyBackendAdapter",
    "PolicyConfig",
    "SchedulerAction",
    "SGLangTargetedKVPrefetchBackendAdapter",
    "SessionPhase",
    "SessionState",
    "TimingEstimator",
]
