from __future__ import annotations

import re
from typing import Any, Iterable

from agentic_kv.evidence_schema import (
    case_identity,
    correlation_identity,
    correlation_source,
    gap_identity,
    movement_evidence_level,
    movement_kind_from_fields,
    request_identity,
    request_or_context_value,
)

from .events import KVEventType, NormalizedKVEvent


EVENT_MAP: dict[str, KVEventType] = {
    "hicache.write.end": KVEventType.WRITE_HOST,
    "hicache.evict_device.end": KVEventType.EVICT_GPU,
    "hicache.evict_host.end": KVEventType.EVICT_HOST,
    "hicache.load.end": KVEventType.LOAD_GPU,
    "hostpool.load_to_device_per_layer.end": KVEventType.LOAD_GPU,
    "hostpool.backup_from_device_all_layer.end": KVEventType.WRITE_HOST,
    "hiradix.init_load_back.end": KVEventType.LOAD_GPU,
    "hiradix.load_back.end": KVEventType.LOAD_GPU,
    "hiradix.match_prefix.end": KVEventType.MATCH_PREFIX,
}


def normalize_sglang_trace_events(trace_rows: Iterable[dict[str, Any]]) -> list[NormalizedKVEvent]:
    rows = list(trace_rows)
    base_ts = min((float(row["ts_ns"]) for row in rows if row.get("ts_ns")), default=0.0)
    events: list[NormalizedKVEvent] = []
    for row in rows:
        source_event = str(row.get("event") or "")
        event_type = EVENT_MAP.get(source_event)
        if event_type is None:
            continue
        context = context_from_trace_event(row)
        session_id = agent_session_from_context(context)
        if not session_id or "::live_prefetch::" in session_id:
            continue
        ledger_case_id = str(context.get("ledger_case_id") or row.get("ledger_case_id") or "")
        case_id = ledger_case_id or case_identity(context)
        if ledger_case_id:
            session_id = f"{ledger_case_id}::{session_id}"
        phase = agent_phase_from_context(context)
        token_start, token_end, token_count = event_range(event_type, context, row)
        duration = as_float(row.get("duration_ms"))
        ts = as_float(row.get("ts_ns"))
        time_ms = round((ts - base_ts) / 1_000_000.0, 3) if ts is not None and base_ts else None
        copy_start_ms = round(time_ms - duration, 3) if time_ms is not None and duration is not None else None
        host_start, host_end, host_count = index_range(context.get("host_indices"))
        device_start, device_end, device_count = index_range(context.get("device_indices"))
        request_id = request_identity(context)
        agent_request_id = request_or_context_value(context, "agent_request_id")
        correlation_id = correlation_identity(context)
        gap_id = gap_identity(context)
        movement_kind = movement_kind_from_fields(
            direction=context.get("direction", ""),
            movement="",
            event_type=event_type.value,
            source_event=source_event,
        )
        evidence_probe = {
            "copy_start_ms": copy_start_ms,
            "copy_end_ms": time_ms,
            "source_event": source_event,
            "host_index_signature": index_signature(context.get("host_indices")),
            "device_index_signature": index_signature(context.get("device_indices")),
            "node_id": str(context.get("node_id") or ""),
        }
        events.append(
            NormalizedKVEvent(
                event_type=event_type,
                session_id=session_id,
                phase=phase,
                time_ms=time_ms,
                duration_ms=duration,
                token_start=token_start,
                token_end=token_end,
                token_count=token_count,
                node_id=str(context.get("node_id") or ""),
                request_id=request_id,
                agent_request_id=agent_request_id,
                correlation_id=correlation_id,
                case_id=case_id,
                gap_id=gap_id,
                layer_id=str(context.get("layer_id") or ""),
                direction=str(context.get("direction") or ""),
                movement_kind=movement_kind,
                host_index_start=host_start,
                host_index_end=host_end,
                host_index_count=host_count,
                host_index_signature=index_signature(context.get("host_indices")),
                device_index_start=device_start,
                device_index_end=device_end,
                device_index_count=device_count,
                device_index_signature=index_signature(context.get("device_indices")),
                copy_start_ms=copy_start_ms,
                copy_end_ms=time_ms,
                source_event=source_event,
                confidence=event_confidence(event_type, context),
                evidence_level=movement_evidence_level(evidence_probe),
                exact_correlation_source=correlation_source(
                    {
                        "correlation_id": correlation_id,
                        "request_id": request_id,
                        "agent_request_id": agent_request_id,
                        "agent_label": request_or_context_value(context, "agent_label"),
                        "session_id": session_id,
                    }
                ),
                raw={
                    "event": source_event,
                    "phase": phase,
                    "direction": context.get("direction", ""),
                    "movement_kind": movement_kind,
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "case_id": case_id,
                    "gap_id": gap_id,
                },
            )
        )
    return sorted(events, key=lambda event: event.time_ms if event.time_ms is not None else -1.0)


def context_from_trace_event(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("kv_context")
    return context if isinstance(context, dict) else row


def nested_request(context: dict[str, Any]) -> dict[str, Any]:
    request = context.get("request")
    return request if isinstance(request, dict) else {}


def agent_session_from_context(context: dict[str, Any]) -> str:
    for key in ("agent_session_id", "session_id"):
        value = context.get(key)
        if value not in ("", None):
            return str(value)
    request = nested_request(context)
    for key in ("agent_session_id", "session_id"):
        value = request.get(key)
        if value not in ("", None):
            return str(value)
    for key in ("agent_sessions", "requests"):
        values = context.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            found = agent_session_from_context(item)
            if found:
                return found
    return ""


def agent_phase_from_context(context: dict[str, Any]) -> str:
    request = nested_request(context)
    return str(context.get("agent_phase") or request.get("agent_phase") or "unknown")


def event_confidence(event_type: KVEventType, context: dict[str, Any]) -> str:
    if index_signature(context.get("host_indices")) and index_signature(context.get("device_indices")):
        return "high"
    if context.get("node_id") not in ("", None):
        return "high"
    if event_type == KVEventType.MATCH_PREFIX:
        return "high"
    return "medium"


def event_range(
    event_type: KVEventType,
    context: dict[str, Any],
    row: dict[str, Any],
) -> tuple[int | None, int | None, int]:
    if event_type == KVEventType.WRITE_HOST:
        return index_range(context.get("device_indices"))
    if event_type == KVEventType.EVICT_GPU:
        return index_range(context.get("device_indices"))
    if event_type == KVEventType.EVICT_HOST:
        return index_range(context.get("host_indices"))
    if event_type == KVEventType.LOAD_GPU:
        host_range = index_range(context.get("host_indices"))
        if host_range[2]:
            return host_range
        device_range = index_range(context.get("device_indices"))
        if device_range[2]:
            return device_range
        host_hit = as_float(context.get("host_hit_length"))
        if host_hit is not None:
            return None, None, int(host_hit)
        return None, None, 0
    if event_type == KVEventType.MATCH_PREFIX:
        result = row.get("result")
        if isinstance(result, list) and result:
            return index_range(result[0])
    return None, None, 0


def index_range(value: Any) -> tuple[int | None, int | None, int]:
    if not isinstance(value, dict):
        return None, None, 0
    count = 0
    for key in ("index_count", "numel", "count"):
        parsed = as_float(value.get(key))
        if parsed is not None:
            count = int(parsed)
            break
    start = parse_int(value.get("min"))
    end = parse_int(value.get("max"))
    if start is None or end is None:
        text = str(value)
        match = re.search(r"(-?\d+)\.\.(-?\d+)", text)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
    if not count and start is not None and end is not None:
        count = max(0, end - start + 1)
    return start, end, count


def index_signature(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    count = 0
    for key in ("index_count", "numel", "count"):
        parsed = as_float(value.get(key))
        if parsed is not None:
            count = int(parsed)
            break
    digest = value.get("sha1_16")
    if count and digest:
        return f"{count}:{digest}"
    values = value.get("values")
    if isinstance(values, list):
        return f"{len(values)}:{','.join(str(item) for item in values)}"
    start = parse_int(value.get("min"))
    end = parse_int(value.get("max"))
    if count and start is not None and end is not None:
        return f"{count}:{start}..{end}"
    return ""


def parse_int(value: Any) -> int | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
