from __future__ import annotations

import re
from typing import Any, Iterable

from .events import KVEventType, NormalizedKVEvent


EVENT_MAP: dict[str, KVEventType] = {
    "hicache.write.end": KVEventType.WRITE_HOST,
    "hicache.evict_device.end": KVEventType.EVICT_GPU,
    "hicache.evict_host.end": KVEventType.EVICT_HOST,
    "hicache.load.end": KVEventType.LOAD_GPU,
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
        phase = agent_phase_from_context(context)
        token_start, token_end, token_count = event_range(event_type, context, row)
        duration = as_float(row.get("duration_ms"))
        ts = as_float(row.get("ts_ns"))
        time_ms = round((ts - base_ts) / 1_000_000.0, 3) if ts is not None and base_ts else None
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
                source_event=source_event,
                confidence=event_confidence(event_type, context),
                raw={"event": source_event, "phase": phase},
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

