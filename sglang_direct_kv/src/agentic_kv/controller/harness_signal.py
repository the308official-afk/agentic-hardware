from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


HARNESS_CONTROLLER_SIGNAL_SCHEMA = "harness_controller_signal.v1"


def _text(value: Any, default: str = "") -> str:
    if value in (None, "", [], {}):
        return default
    return str(value)


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TaskSignal:
    session_id: str
    request_id: str
    task_index: str = ""
    turn_id: str = ""
    prefix_id: str = ""
    session_generation: int = 0


@dataclass(frozen=True)
class PhaseSignal:
    name: str
    work_class: str
    user_waiting: bool
    manager_visible: bool = False
    interactive: bool = False


@dataclass(frozen=True)
class ToolSignal:
    name: str = ""
    type: str = ""
    started_at_ms: int | None = None
    estimated_duration_ms: int | None = None
    expected_done_at_ms: int | None = None
    eta_uncertainty_ms: int | None = None
    actual_duration_ms: int | None = None


@dataclass(frozen=True)
class ToolWaitSignal:
    profile: str = "fixed"
    wait_class: str = ""
    duration_ms: int | None = None
    step_index: int = 0
    total_steps: int = 1


@dataclass(frozen=True)
class ReplaySignal:
    likely: bool = False
    expected_count: int = 1
    deadline_after_tool_ms: int | None = None
    expected_request_id: str = ""


@dataclass(frozen=True)
class CacheSignal:
    stable_prefix: bool = False
    cache_key: str = ""
    reuse_scope: str = ""
    repo_context_hash: str = ""
    tool_schema_hash: str = ""
    conversation_prefix_hash: str = ""
    native_cache_signal_seen: bool = False
    native_cache_signal_source: str = ""


@dataclass(frozen=True)
class SchedulingSignal:
    urgency: str = "normal"
    priority: int | None = None
    latency_sensitivity: str = ""
    safe_to_demote: bool = False
    preemptible: bool = False
    can_delay_ms: int | None = None
    fairness_group: str = ""
    tenant_id: str = ""


@dataclass(frozen=True)
class CompetitionSignal:
    active_background_requests: int = 0
    demotable_background_requests: int = 0
    background_safe_to_demote: bool = False
    background_can_delay_ms: int | None = None
    concurrency: int | None = None


@dataclass(frozen=True)
class CostFeedbackSignal:
    recent_target_replay_debt_ms: int | None = None
    recent_background_ttft_ms: int | None = None
    recent_background_slowdown_ratio: float | None = None
    allow_background_demote: bool = True


@dataclass(frozen=True)
class OutputSignal:
    expected_output_tokens: int | None = None
    max_tokens: int | None = None
    streaming: bool = True


@dataclass(frozen=True)
class ResourceSignal:
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    uncached_tokens: int | None = None
    kv_bytes_estimate: int | None = None


@dataclass(frozen=True)
class HarnessControllerSignal:
    harness: str
    mode: str
    pressure_level: str
    task: TaskSignal
    phase: PhaseSignal
    tool: ToolSignal = field(default_factory=ToolSignal)
    tool_wait: ToolWaitSignal = field(default_factory=ToolWaitSignal)
    replay: ReplaySignal = field(default_factory=ReplaySignal)
    cache: CacheSignal = field(default_factory=CacheSignal)
    scheduling: SchedulingSignal = field(default_factory=SchedulingSignal)
    competition: CompetitionSignal = field(default_factory=CompetitionSignal)
    cost_feedback: CostFeedbackSignal = field(default_factory=CostFeedbackSignal)
    output: OutputSignal = field(default_factory=OutputSignal)
    resource: ResourceSignal = field(default_factory=ResourceSignal)
    native_signals: dict[str, Any] = field(default_factory=dict)
    schema_version: str = HARNESS_CONTROLLER_SIGNAL_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_work_class(meta: dict[str, Any]) -> str:
    explicit = _text(meta.get("work_class") or meta.get("request_group"))
    if explicit:
        return explicit
    phase = _text(meta.get("phase"))
    priority_label = _text(meta.get("priority_label")).lower()
    if phase == "speculative_prefill":
        return "speculative"
    if phase.startswith("pressure_filler") or priority_label in {"low", "background"}:
        return "background"
    return "target"


def infer_tool_type(meta: dict[str, Any]) -> str:
    explicit = _text(meta.get("tool_type"))
    if explicit:
        return explicit
    wait_class = _text(meta.get("tool_wait_class"))
    if wait_class in {"quick", "moderate", "slow", "pressure_fixed"}:
        return wait_class
    return "synthetic"


def infer_urgency(meta: dict[str, Any], work_class: str) -> str:
    priority_intent = meta.get("priority_intent")
    if isinstance(priority_intent, dict):
        priority_class = _text(priority_intent.get("class")).lower()
        if priority_class:
            return priority_class
    priority_label = _text(meta.get("priority_label")).lower()
    if priority_label in {"high", "urgent", "priority"}:
        return "urgent"
    if work_class in {"background", "filler", "speculative"}:
        return "background"
    return "normal"


def build_harness_controller_signal(
    meta: dict[str, Any],
    *,
    monotonic_ms: int | None = None,
    expected_completion_ms: int | None = None,
    deadline_after_completion_ms: int | None = None,
    eta_uncertainty_ms: int | None = None,
) -> dict[str, Any]:
    """Normalize harness/request facts for the portable controller.

    This function deliberately depends only on plain metadata. Harness adapters
    can add richer fields over time without coupling the controller to a
    specific client, runtime, or SGLang version.
    """

    existing = meta.get("harness_controller_signal")
    if isinstance(existing, dict) and existing.get("schema_version") == HARNESS_CONTROLLER_SIGNAL_SCHEMA:
        return existing

    session_id = _text(meta.get("session_id"), "harness_session")
    phase = _text(meta.get("phase"), "request")
    request_id = _text(meta.get("label") or meta.get("request_id"), f"{session_id}_{phase}")
    work_class = infer_work_class(meta)
    urgency = infer_urgency(meta, work_class)
    tool_wait_ms = _int_or_none(meta.get("tool_wait_ms"))
    expected_done = expected_completion_ms
    started_at = None
    if expected_done is not None and tool_wait_ms is not None:
        started_at = expected_done - tool_wait_ms
    elif phase == "tool_wait":
        started_at = monotonic_ms

    safe_to_demote = _bool(
        meta.get("safe_to_demote"),
        default=work_class in {"background", "filler", "speculative", "maintenance"},
    )
    native_cache_profile = meta.get("native_cache_profile") if isinstance(meta.get("native_cache_profile"), dict) else {}
    native_cache_seen = _bool(meta.get("harness_native_cache_signal_seen"), default=False)
    stable_prefix = _bool(
        meta.get("stable_prefix"),
        default=bool(meta.get("prefix_id")) or native_cache_seen or _bool(native_cache_profile.get("enabled")),
    )
    priority_value = (
        _int_or_none(meta.get("controller_sglang_priority"))
        or _int_or_none(meta.get("sglang_priority"))
        or _int_or_none(meta.get("high_priority"))
    )

    signal = HarnessControllerSignal(
        harness=_text(meta.get("harness"), "unknown"),
        mode=_text(meta.get("mode")),
        pressure_level=_text(meta.get("pressure_level")),
        task=TaskSignal(
            session_id=session_id,
            request_id=request_id,
            task_index=_text(meta.get("task_index")),
            turn_id=_text(meta.get("turn_id") or meta.get("tool_wait_step")),
            prefix_id=_text(meta.get("prefix_id") or session_id),
            session_generation=_int_or_none(meta.get("session_generation")) or 0,
        ),
        phase=PhaseSignal(
            name=phase,
            work_class=work_class,
            user_waiting=_bool(meta.get("user_waiting"), default=work_class == "target"),
            manager_visible=_bool(meta.get("manager_visible"), default=work_class == "target"),
            interactive=_bool(meta.get("interactive"), default=work_class == "target"),
        ),
        tool=ToolSignal(
            name=_text(meta.get("tool_name"), "synthetic_tool" if phase in {"tool_wait", "replay"} else ""),
            type=infer_tool_type(meta),
            started_at_ms=started_at,
            estimated_duration_ms=tool_wait_ms,
            expected_done_at_ms=expected_done,
            eta_uncertainty_ms=eta_uncertainty_ms,
            actual_duration_ms=_int_or_none(meta.get("actual_tool_ms")),
        ),
        tool_wait=ToolWaitSignal(
            profile=_text(meta.get("tool_wait_profile"), "fixed"),
            wait_class=_text(meta.get("tool_wait_class")),
            duration_ms=tool_wait_ms,
            step_index=_int_or_none(meta.get("tool_wait_step")) or 0,
            total_steps=_int_or_none(meta.get("task_replay_steps")) or 1,
        ),
        replay=ReplaySignal(
            likely=phase in {"tool_wait", "prepare", "replay"} or _bool(meta.get("replay_likely")),
            expected_count=_int_or_none(meta.get("task_replay_steps")) or 1,
            deadline_after_tool_ms=deadline_after_completion_ms or _int_or_none(meta.get("deadline_after_tool_ms")),
            expected_request_id=_text(meta.get("expected_replay_request_id") or meta.get("label")),
        ),
        cache=CacheSignal(
            stable_prefix=stable_prefix,
            cache_key=_text(meta.get("cache_key") or meta.get("gateway_cache_salt") or native_cache_profile.get("cache_key_seed")),
            reuse_scope=_text(meta.get("cache_reuse_scope"), "session" if stable_prefix else ""),
            repo_context_hash=_text(meta.get("repo_context_hash")),
            tool_schema_hash=_text(meta.get("tool_schema_hash")),
            conversation_prefix_hash=_text(meta.get("conversation_prefix_hash") or meta.get("prompt_hash")),
            native_cache_signal_seen=native_cache_seen,
            native_cache_signal_source=_text(meta.get("harness_native_cache_signal_source")),
        ),
        scheduling=SchedulingSignal(
            urgency=urgency,
            priority=priority_value,
            latency_sensitivity=_text(meta.get("latency_sensitivity") or meta.get("expected_inferred_priority")),
            safe_to_demote=safe_to_demote,
            preemptible=_bool(meta.get("preemptible"), default=safe_to_demote),
            can_delay_ms=_int_or_none(meta.get("can_delay_ms")) or (5000 if safe_to_demote else 0),
            fairness_group=_text(meta.get("fairness_group") or session_id),
            tenant_id=_text(meta.get("tenant_id")),
        ),
        competition=CompetitionSignal(
            active_background_requests=_int_or_none(meta.get("active_background_requests")) or 0,
            demotable_background_requests=_int_or_none(meta.get("demotable_background_requests")) or 0,
            background_safe_to_demote=_bool(meta.get("background_safe_to_demote"), default=False),
            background_can_delay_ms=_int_or_none(meta.get("background_can_delay_ms")),
            concurrency=_int_or_none(meta.get("concurrency")),
        ),
        cost_feedback=CostFeedbackSignal(
            recent_target_replay_debt_ms=_int_or_none(meta.get("recent_target_replay_debt_ms")),
            recent_background_ttft_ms=_int_or_none(meta.get("recent_background_ttft_ms")),
            recent_background_slowdown_ratio=_float_or_none(meta.get("recent_background_slowdown_ratio")),
            allow_background_demote=_bool(meta.get("cost_feedback_allow_background_demote"), default=True),
        ),
        output=OutputSignal(
            expected_output_tokens=_int_or_none(meta.get("expected_output_tokens") or meta.get("max_tokens")),
            max_tokens=_int_or_none(meta.get("max_tokens")),
            streaming=_bool(meta.get("streaming"), default=True),
        ),
        resource=ResourceSignal(
            prompt_tokens=_int_or_none(meta.get("prompt_tokens") or meta.get("target_prompt_tokens")),
            cached_tokens=_int_or_none(meta.get("cached_tokens")),
            uncached_tokens=_int_or_none(meta.get("uncached_tokens")),
            kv_bytes_estimate=_int_or_none(meta.get("kv_bytes_estimate")),
        ),
        native_signals={
            "priority_intent": meta.get("priority_intent", ""),
            "harness_input_priority_signal": meta.get("harness_input_priority_signal", ""),
            "harness_input_priority_signal_source": meta.get("harness_input_priority_signal_source", ""),
            "harness_emit_priority_signal": meta.get("harness_emit_priority_signal", ""),
            "harness_emit_priority_signal_source": meta.get("harness_emit_priority_signal_source", ""),
            "harness_native_cache_signal": meta.get("harness_native_cache_signal", ""),
            "harness_native_cache_signal_source": meta.get("harness_native_cache_signal_source", ""),
            "workflow_node": meta.get("workflow_node", ""),
            "workflow_node_goal": meta.get("workflow_node_goal", ""),
        },
    )
    return signal.to_dict()
