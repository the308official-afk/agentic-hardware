from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


SCHEMA_NAME = "agentic.runtime_telemetry"
SCHEMA_VERSION = 1


def runtime_telemetry_enabled() -> bool:
    return (
        os.environ.get("AGENTIC_RUNTIME_TELEMETRY", "0") == "1"
        or os.environ.get("AGENTIC_RUNTIME_TELEMETRY_ENABLE", "0") == "1"
    )


def runtime_telemetry_path() -> Path | None:
    raw = os.environ.get("AGENTIC_RUNTIME_TELEMETRY_PATH", "").strip()
    if raw:
        return Path(raw)
    if not runtime_telemetry_enabled():
        return None
    trace_path = os.environ.get("AGENTIC_KV_TRACE_PATH", "").strip()
    if trace_path:
        return Path(trace_path).with_name("runtime_telemetry.jsonl")
    return Path("artifacts/runtime_telemetry.jsonl")


def _clean(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        cleaned = {str(key): _clean(item) for key, item in value.items()}
        return {key: item for key, item in cleaned.items() if item is not None}
    if isinstance(value, (list, tuple)):
        cleaned = [_clean(item) for item in value]
        return [item for item in cleaned if item is not None]
    return str(value)


def _first_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _request_identity(request: dict[str, Any]) -> dict[str, Any]:
    session_id = _first_value(request, ("agent_session_id", "session_id"))
    request_id = _first_value(
        request,
        (
            "agent_request_id",
            "request_id",
            "dynamo_hint_request_id",
            "rid",
            "agent_label",
        ),
    )
    return {
        "case_id": _first_value(request, ("agent_case_id", "case_id")),
        "gap_id": _first_value(request, ("agent_gap_id", "gap_id")),
        "session_id": session_id,
        "request_id": request_id,
        "request_phase": _first_value(request, ("agent_phase", "phase", "dynamo_hint_phase")),
        "request_label": _first_value(request, ("agent_label", "label")),
        "prompt_hash": _first_value(request, ("agent_prompt_hash", "prompt_hash")),
        "priority": _first_value(
            request,
            (
                "agent_priority",
                "dynamo_agent_priority",
                "dynamo_hint_priority",
                "priority",
                "sglang_priority",
            ),
        ),
        "correlation_id": _first_value(request, ("agent_correlation_id", "correlation_id")),
        "parent_run_id": _first_value(request, ("agent_parent_run_id", "parent_run_id")),
    }


def _request_from_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    request = context.get("request")
    if isinstance(request, dict):
        merged = dict(context)
        merged.update({key: value for key, value in request.items() if value not in (None, "", [], {})})
        return merged
    return context


def emit_runtime_event(
    event_type: str,
    *,
    phase: str = "point",
    backend: str | None = None,
    source_backend: str | None = None,
    source_hook: str = "",
    source_class: str = "",
    source_method: str = "",
    call_id: str = "",
    context: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    confidence: str = "direct_runtime_hook",
    error: str = "",
) -> dict[str, Any] | None:
    if not runtime_telemetry_enabled():
        return None
    path = runtime_telemetry_path()
    if path is None:
        return None

    request = _request_from_context(context)
    identity = _request_identity(request)
    event: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "event": "runtime_telemetry",
        "event_type": event_type,
        "event_phase": phase,
        "backend": backend or os.environ.get("AGENTIC_RUNTIME_TELEMETRY_BACKEND", "sglang"),
        "source_backend": source_backend or "runtime_hook",
        "source_hook": source_hook,
        "source_class": source_class,
        "source_method": source_method,
        "call_id": call_id,
        "ts_ns": time.time_ns(),
        "monotonic_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        "confidence": confidence,
        **identity,
    }
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    if error:
        event["error"] = error
    if payload:
        event["payload"] = payload

    event = _clean(event)
    if not isinstance(event, dict):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event

