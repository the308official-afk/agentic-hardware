from __future__ import annotations

from typing import Any


EMPTY_VALUES = (None, "", [], {})


def first_value(*values: Any) -> str:
    for value in values:
        if value not in EMPTY_VALUES:
            return str(value)
    return ""


def nested_request(context: dict[str, Any]) -> dict[str, Any]:
    request = context.get("request")
    return request if isinstance(request, dict) else {}


def request_or_context_value(context: dict[str, Any], *keys: str) -> str:
    request = nested_request(context)
    values: list[Any] = []
    for key in keys:
        values.append(context.get(key))
        values.append(request.get(key))
    return first_value(*values)


def request_identity(context: dict[str, Any]) -> str:
    return request_or_context_value(
        context,
        "request_id",
        "agent_request_id",
        "rid",
        "agent_label",
        "label",
    )


def correlation_identity(context: dict[str, Any]) -> str:
    explicit = request_or_context_value(
        context,
        "correlation_id",
        "agent_correlation_id",
        "request_id",
        "agent_request_id",
        "rid",
    )
    if explicit:
        return explicit
    session_id = request_or_context_value(context, "agent_session_id", "session_id")
    phase = request_or_context_value(context, "agent_phase", "phase")
    label = request_or_context_value(context, "agent_label", "label")
    if session_id and phase and label:
        return f"{session_id}:{phase}:{label}"
    if session_id and phase:
        return f"{session_id}:{phase}"
    return session_id


def case_identity(context: dict[str, Any]) -> str:
    return request_or_context_value(
        context,
        "ledger_case_id",
        "case_id",
        "agent_case_id",
        "parent_run_id",
        "agent_parent_run_id",
    )


def gap_identity(context: dict[str, Any]) -> str:
    return request_or_context_value(context, "gap_id", "agent_gap_id", "gap_order")


def movement_kind_from_fields(
    *,
    direction: Any = "",
    movement: Any = "",
    event_type: Any = "",
    source_event: Any = "",
) -> str:
    direction_text = str(direction or "")
    movement_text = str(movement or "")
    event_text = str(event_type or "")
    source_text = str(source_event or "")
    haystack = " ".join((direction_text, movement_text, event_text, source_text)).lower()

    if movement_text.upper() == "H2D" or direction_text == "host_to_device" or "host_to_gpu_load" in haystack or "load_to_device" in haystack:
        return "H2D"
    if movement_text.upper() == "D2H" or direction_text == "device_to_host" or "gpu_to_host_write" in haystack or "backup_from_device" in haystack:
        return "D2H"
    if (
        movement_text in ("GPU evict", "GPU_EVICT")
        or direction_text == "device_evict"
        or "gpu_evict" in haystack
        or "gpu evict" in haystack
        or "evict_device" in haystack
    ):
        return "GPU_EVICT"
    if movement_text in ("host evict", "HOST_EVICT") or direction_text == "host_evict" or "evict_host" in haystack:
        return "HOST_EVICT"
    if "match_prefix" in haystack:
        return "CACHE_MATCH"
    if "recompute" in haystack:
        return "RECOMPUTE"
    return "UNKNOWN"


def movement_kind_from_row(row: dict[str, Any]) -> str:
    return movement_kind_from_fields(
        direction=row.get("direction", ""),
        movement=row.get("movement", "") or row.get("movement_kind", "") or row.get("kind", ""),
        event_type=row.get("event_type", ""),
        source_event=row.get("source_event", ""),
    )


def movement_kind_display(kind: str) -> str:
    return {
        "H2D": "H2D",
        "D2H": "D2H",
        "GPU_EVICT": "GPU evict",
        "HOST_EVICT": "host evict",
        "CACHE_MATCH": "cache match",
        "RECOMPUTE": "recompute",
        "UNKNOWN": "KV movement",
    }.get(kind, kind or "KV movement")


def movement_evidence_level(row: dict[str, Any]) -> str:
    has_start_end = row.get("copy_start_ms") not in EMPTY_VALUES and row.get("copy_end_ms") not in EMPTY_VALUES
    has_hook = row.get("source_event") not in EMPTY_VALUES
    has_host = row.get("host_index_signature") not in EMPTY_VALUES
    has_device = row.get("device_index_signature") not in EMPTY_VALUES
    has_node = row.get("node_id") not in EMPTY_VALUES
    if has_start_end and has_hook and has_host and has_device:
        return "DIRECT_EXACT_INDEXED"
    if has_start_end and has_hook and (has_host or has_device or has_node):
        return "DIRECT_PARTIAL_ID"
    if has_start_end and has_hook:
        return "DIRECT_TIMED"
    return "DERIVED_OR_INFERRED"


def correlation_source(row: dict[str, Any]) -> str:
    for key in ("correlation_id", "request_id", "agent_request_id", "agent_label", "session_id"):
        if row.get(key) not in EMPTY_VALUES:
            return key
    return "none"
