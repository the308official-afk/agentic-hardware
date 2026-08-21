#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentic_kv.block_ledger import (
    block_lifecycle_by_gap_rows,
    block_lifecycle_focus_rows,
    block_lifecycle_verdict_counts,
    block_ledger_rows,
    build_block_ledger,
    exact_movement_rows,
    exact_movement_summary_rows,
    gap_lifecycle_summary_rows,
    ledger_summary_rows,
    normalize_sglang_trace_events,
    write_ledger_artifacts,
)
from agentic_kv.evidence_audit import audit_markdown, audit_report_data
from agentic_kv.evidence_schema import movement_kind_display, movement_kind_from_row
from build_live_agentbench_tool_gap_report import (
    build_expanded_gap_timeline_svg,
    build_local_timing_phase_timeline_svg,
    build_replay_execution_timeline_svg,
    display_ms,
    read_jsonl,
    table_html,
    timeline_kv_outcome,
    write_csv,
)
from build_live_paired_agentbench_report import (
    agent_session_from_context,
    css as master_css,
    context_from_trace_event,
    global_prefetch_margin_html as live_global_prefetch_margin_html,
    load_live_run as load_live_agentbench_run,
    mode_summary as live_mode_summary,
    movement_events_by_session,
    report_script,
    setup_diagram_svg,
    timeline_guide_html,
    toc_html,
)
from replay_path_classifier import (
    attach_replay_path_fields,
    bottleneck_summary_rows,
    build_replay_path_ledger,
    confidence_summary_rows,
    counterfactual_summary_rows,
    instrumentation_coverage_rows,
)


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: list[float]) -> float | str:
    return round(mean(values), 3) if values else ""


def rel_ms(ts_ns: Any, base_ts: float) -> float | str:
    value = as_float(ts_ns)
    if value is None:
        return ""
    return round((value / 1_000_000_000.0 - base_ts) * 1000.0, 3)


def has_events(value: Any) -> bool:
    try:
        return int(value or 0) > 0
    except (TypeError, ValueError):
        return False


def replay_path(row: dict[str, Any]) -> str:
    return replay_path_from_evidence(row)


def summarize_movement(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"start_ms": "", "end_ms": "", "duration_ms": "", "events": "0", "categories": ""}
    starts: list[float] = []
    ends: list[float] = []
    total = 0.0
    counted = 0
    categories: dict[str, int] = defaultdict(int)
    for event in events:
        ts = as_float(event.get("start_or_end_ms"))
        if ts is None:
            continue
        duration = as_float(event.get("duration_ms"))
        if duration is not None:
            starts.append(ts - duration)
            ends.append(ts)
            total += duration
            counted += 1
        else:
            starts.append(ts)
            ends.append(ts)
        categories[str(event.get("category") or event.get("event") or "event")] += 1
    if not starts or not ends:
        return {"start_ms": "", "end_ms": "", "duration_ms": "", "events": str(len(events)), "categories": ""}
    return {
        "start_ms": round(min(starts), 3),
        "end_ms": round(max(ends), 3),
        "duration_ms": round(total, 3) if counted else "",
        "events": str(len(events)),
        "categories": ", ".join(f"{key}:{value}" for key, value in sorted(categories.items())),
    }


def index_count(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    for key in ("index_count", "numel", "count"):
        parsed = as_float(value.get(key))
        if parsed is not None:
            return int(parsed)
    return None


def nested_request(context: dict[str, Any]) -> dict[str, Any]:
    request = context.get("request")
    return request if isinstance(request, dict) else {}


def request_token_count(request: dict[str, Any]) -> int | None:
    raw_ingest = as_float(request.get("ingest_input_tokens"))
    if raw_ingest is not None:
        return int(raw_ingest)
    for key in ("ingest_input_ids", "ingest_origin_input_ids", "input_ids", "origin_input_ids", "fill_ids"):
        count = index_count(request.get(key))
        if count is not None:
            return count
    return None


def active_request_token_count(request: dict[str, Any]) -> int | None:
    raw_active = as_float(request.get("active_input_tokens"))
    if raw_active is not None:
        return int(raw_active)
    for key in ("origin_input_ids", "fill_ids", "input_ids"):
        count = index_count(request.get(key))
        if count is not None:
            return count
    return None


def result_prefix_tokens(row: dict[str, Any]) -> int | None:
    result = row.get("result")
    if isinstance(result, list) and result:
        count = index_count(result[0])
        if count is not None:
            return count
    return None


def result_host_hit_tokens(row: dict[str, Any]) -> int | None:
    result = row.get("result")
    if isinstance(result, list) and len(result) > 3:
        value = as_float(result[3])
        if value is not None:
            return int(value)
    return None


def cache_events_by_session(trace_rows: list[dict[str, Any]], base_ts: float) -> dict[str, list[dict[str, Any]]]:
    cache_events = {
        "kv_telemetry.cache.end": "cache_telemetry",
        "hiradix.match_prefix.end": "match_prefix",
        "hiradix.ready_to_load_host_cache.end": "ready_to_load_host_cache",
        "hiradix.init_load_back.end": "init_load_back",
        "hiradix.load_back.end": "load_back",
        "hicache.load.end": "hicache_load",
        "hiradix.cache_finished_req.end": "cache_finished_req",
        "hiradix.cache_unfinished_req.end": "cache_unfinished_req",
    }
    by_session: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        event = str(row.get("event") or "")
        category = cache_events.get(event)
        if not category:
            continue
        context = context_from_trace_event(row)
        session_id = agent_session_from_context(context)
        if not session_id or "::live_prefetch::" in session_id:
            continue
        if event == "kv_telemetry.cache.end":
            item = {
                "source": "kv_cache_telemetry",
                "event": event,
                "category": str(row.get("category") or "cache_telemetry"),
                "agent_session_id": session_id,
                "agent_phase": row.get("agent_phase", ""),
                "duration_ms": row.get("duration_ms", ""),
                "start_or_end_ms": rel_ms(row.get("ts_ns"), base_ts),
                "input_tokens": row.get("input_tokens", ""),
                "active_input_tokens": row.get("active_input_tokens", ""),
                "ingest_input_tokens": row.get("ingest_input_tokens", ""),
                "scheduler_trimmed_tokens": row.get("scheduler_trimmed_tokens", ""),
                "cached_prefix_tokens": row.get("cached_prefix_tokens", ""),
                "host_hit_tokens": row.get("host_hit_tokens", ""),
                "host_load_tokens": row.get("host_load_tokens", ""),
                "device_load_tokens": row.get("device_load_tokens", ""),
                "cache_protected_tokens": row.get("cache_protected_tokens", ""),
                "kv_committed_tokens": row.get("kv_committed_tokens", ""),
                "request_id": row.get("request_id", ""),
            }
            by_session[session_id].append(item)
            continue
        request = nested_request(context)
        input_tokens = request_token_count(request)
        active_input_tokens = active_request_token_count(request)
        prefix_tokens = index_count(request.get("prefix_indices"))
        result_prefix = result_prefix_tokens(row)
        if result_prefix is not None:
            prefix_tokens = max(prefix_tokens or 0, result_prefix)
        scheduler_trimmed = as_float(request.get("scheduler_trimmed_tokens"))
        if scheduler_trimmed is None and input_tokens is not None and active_input_tokens is not None and input_tokens > active_input_tokens:
            scheduler_trimmed = input_tokens - active_input_tokens
        if scheduler_trimmed is not None:
            prefix_tokens = max(prefix_tokens or 0, int(scheduler_trimmed))
        host_hit = as_float(request.get("host_hit_length"))
        result_host_hit = result_host_hit_tokens(row)
        if result_host_hit is not None:
            host_hit = max(int(host_hit or 0), result_host_hit)
        host_indices = context.get("host_indices")
        device_indices = context.get("device_indices")
        item = {
            "source": "sglang_trace",
            "event": event,
            "category": category,
            "agent_session_id": session_id,
            "agent_phase": context.get("agent_phase") or request.get("agent_phase", ""),
            "duration_ms": row.get("duration_ms", ""),
            "start_or_end_ms": rel_ms(row.get("ts_ns"), base_ts),
            "input_tokens": input_tokens if input_tokens is not None else "",
            "active_input_tokens": active_input_tokens if active_input_tokens is not None else "",
            "ingest_input_tokens": request.get("ingest_input_tokens", ""),
            "scheduler_trimmed_tokens": int(scheduler_trimmed) if scheduler_trimmed is not None else "",
            "cached_prefix_tokens": prefix_tokens if prefix_tokens is not None else "",
            "host_hit_tokens": int(host_hit) if host_hit is not None else "",
            "host_load_tokens": index_count(host_indices) or "",
            "device_load_tokens": index_count(device_indices) or "",
            "cache_protected_tokens": request.get("cache_protected_len", ""),
            "kv_committed_tokens": request.get("kv_committed_len", ""),
            "request_id": request.get("rid", ""),
        }
        by_session[session_id].append(item)
    return by_session


def _agent_sessions_for_event(row: dict[str, Any]) -> list[str]:
    context = context_from_trace_event(row)
    sessions: list[str] = []
    direct = agent_session_from_context(context)
    if direct:
        sessions.append(direct)
    for key in ("agent_sessions", "requests"):
        values = context.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            session = agent_session_from_context(item)
            if session and session not in sessions:
                sessions.append(session)
    return sessions


def telemetry_events_by_session(
    trace_rows: list[dict[str, Any]],
    base_ts: float,
    event_name: str | set[str],
) -> dict[str, list[dict[str, Any]]]:
    event_names = {event_name} if isinstance(event_name, str) else event_name
    by_session: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        if row.get("event") not in event_names:
            continue
        for session_id in _agent_sessions_for_event(row):
            if "::live_prefetch::" in session_id:
                continue
            source_event = str(row.get("source_event") or "")
            phase = str(row.get("phase") or "") or str(row.get("event") or "").rsplit(".", 1)[-1]
            by_session[session_id].append(
                {
                    "event": row.get("event", ""),
                    "source_event": source_event,
                    "phase": phase,
                    "call_id": row.get("call_id", ""),
                    "category": row.get("category", ""),
                    "stage": row.get("stage", ""),
                    "stage_group": row.get("stage_group", ""),
                    "stage_order": row.get("stage_order", ""),
                    "exact_sglang_hook": row.get("exact_sglang_hook", ""),
                    "method": row.get("method", ""),
                    "duration_ms": row.get("duration_ms", ""),
                    "start_or_end_ms": rel_ms(row.get("ts_ns"), base_ts),
                    "request_count": row.get("request_count", ""),
                    "request_id": row.get("request_id", ""),
                    "scheduler_waiting_queue_len": row.get("scheduler_waiting_queue_len", ""),
                    "scheduler_running_queue_len": row.get("scheduler_running_queue_len", ""),
                    "scheduler_running_batch_request_count": row.get("scheduler_running_batch_request_count", ""),
                    "scheduler_running_batch_extend_num_tokens": row.get("scheduler_running_batch_extend_num_tokens", ""),
                    "scheduler_cur_batch_request_count": row.get("scheduler_cur_batch_request_count", ""),
                    "scheduler_cur_batch_extend_num_tokens": row.get("scheduler_cur_batch_extend_num_tokens", ""),
                    "forward_mode": row.get("forward_mode", ""),
                    "extend_num_tokens": row.get("extend_num_tokens", ""),
                    "seq_lens_sum": row.get("seq_lens_sum", ""),
                    "host_index_count": row.get("host_index_count", ""),
                    "device_index_count": row.get("device_index_count", ""),
                }
            )
    return by_session


def summarize_timed_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "event_count": 0,
            "start_ms": "",
            "end_ms": "",
            "duration_ms": "",
            "categories": "",
        }
    starts: list[float] = []
    ends: list[float] = []
    total = 0.0
    duration_count = 0
    categories: Counter[str] = Counter()
    for event in events:
        end = as_float(event.get("start_or_end_ms"))
        duration = as_float(event.get("duration_ms"))
        if end is None:
            continue
        if duration is not None:
            starts.append(end - duration)
            total += duration
            duration_count += 1
        else:
            starts.append(end)
        ends.append(end)
        categories[str(event.get("category") or event.get("method") or "event")] += 1
    return {
        "event_count": len(events),
        "start_ms": round(min(starts), 3) if starts else "",
        "end_ms": round(max(ends), 3) if ends else "",
        "duration_ms": round(total, 3) if duration_count else "",
        "categories": ", ".join(f"{key}:{value}" for key, value in sorted(categories.items())),
    }


def first_category_window(events: list[dict[str, Any]], categories: set[str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        if str(event.get("category") or "") not in categories:
            continue
        call_id = str(event.get("call_id") or "")
        phase = str(event.get("phase") or "")
        if call_id and phase in {"start", "end"}:
            grouped[call_id][phase] = event

    candidates: list[dict[str, Any]] = []
    for pair in grouped.values():
        start_event = pair.get("start")
        end_event = pair.get("end")
        if not start_event and not end_event:
            continue
        start = as_float((start_event or {}).get("start_or_end_ms"))
        end = as_float((end_event or {}).get("start_or_end_ms"))
        duration = as_float((end_event or {}).get("duration_ms"))
        if start is None and end is not None and duration is not None:
            start = end - duration
        if end is None and start is not None and duration is not None:
            end = start + duration
        if start is None and end is None:
            continue
        source = end_event or start_event or {}
        item = dict(source)
        item["start_ms"] = round(start if start is not None else end or 0.0, 3)
        item["end_ms"] = round(end if end is not None else start or 0.0, 3)
        item["duration_ms"] = duration if duration is not None else (
            round(item["end_ms"] - item["start_ms"], 3) if start is not None and end is not None else ""
        )
        item["timing_source"] = "explicit_start_end"
        candidates.append(item)

    for event in events:
        if str(event.get("category") or "") not in categories:
            continue
        if str(event.get("phase") or "") == "start":
            continue
        if str(event.get("call_id") or "") and str(event.get("call_id") or "") in grouped:
            continue
        end = as_float(event.get("start_or_end_ms"))
        if end is None:
            continue
        duration = as_float(event.get("duration_ms")) or 0.0
        item = dict(event)
        item["start_ms"] = round(end - duration, 3)
        item["end_ms"] = round(end, 3)
        item["timing_source"] = "end_minus_duration"
        candidates.append(item)
    if not candidates:
        return {"start_ms": "", "end_ms": "", "category": "", "duration_ms": "", "request_count": "", "request_id": "", "timing_source": ""}
    candidates.sort(key=lambda item: (as_float(item.get("start_ms")) or 0.0, as_float(item.get("end_ms")) or 0.0))
    first = candidates[0]
    return {
        "start_ms": first.get("start_ms", ""),
        "end_ms": first.get("end_ms", ""),
        "category": first.get("category", ""),
        "duration_ms": first.get("duration_ms", ""),
        "request_count": first.get("request_count", ""),
        "request_id": first.get("request_id", ""),
        "timing_source": first.get("timing_source", ""),
        "waiting_queue_len": first.get("scheduler_waiting_queue_len", ""),
        "running_queue_len": first.get("scheduler_running_queue_len", ""),
        "running_batch_request_count": first.get("scheduler_running_batch_request_count", ""),
        "running_batch_extend_num_tokens": first.get("scheduler_running_batch_extend_num_tokens", ""),
        "cur_batch_request_count": first.get("scheduler_cur_batch_request_count", ""),
        "cur_batch_extend_num_tokens": first.get("scheduler_cur_batch_extend_num_tokens", ""),
    }


def summarize_scheduler_queue_path(events: list[dict[str, Any]], replay_start_ms: Any, replay_due_ms: Any) -> dict[str, Any]:
    received = first_category_window(events, {"request_received", "sglang_receive"})
    queued = first_category_window(events, {"entered_scheduler_queue", "scheduler_queue_enter"})
    admitted = first_category_window(
        events,
        {
            "selected_for_prefill",
            "selected_to_run",
            "run_batch",
            "run_prebuilt_batch",
            "scheduler_select_prefill",
            "scheduler_select_run",
            "scheduler_run_batch",
            "scheduler_run_prebuilt_batch",
        },
    )
    request_start = as_float(replay_start_ms)
    due = as_float(replay_due_ms)
    submit_to_queue = ""
    queue_to_admit = ""
    admit_to_h2d = ""
    if request_start is not None and as_float(queued.get("start_ms")) is not None:
        submit_to_queue = round(float(queued["start_ms"]) - request_start, 3)
    if as_float(queued.get("end_ms")) is not None and as_float(admitted.get("start_ms")) is not None:
        queue_to_admit = round(float(admitted["start_ms"]) - float(queued["end_ms"]), 3)
    if as_float(admitted.get("start_ms")) is not None:
        admit_to_h2d = ""
    return {
        "received_start_ms": received["start_ms"],
        "received_end_ms": received["end_ms"],
        "queue_enter_start_ms": queued["start_ms"],
        "queue_enter_end_ms": queued["end_ms"],
        "admit_start_ms": admitted["start_ms"],
        "admit_end_ms": admitted["end_ms"],
        "admit_category": admitted["category"],
        "admit_timing_source": admitted.get("timing_source", ""),
        "queue_waiting_len": queued.get("waiting_queue_len", ""),
        "queue_running_len": queued.get("running_queue_len", ""),
        "admit_running_batch_requests": admitted.get("running_batch_request_count", "") or admitted.get("cur_batch_request_count", ""),
        "admit_running_batch_extend_tokens": admitted.get("running_batch_extend_num_tokens", "")
        or admitted.get("cur_batch_extend_num_tokens", ""),
        "request_start_lateness_ms": round(request_start - due, 3) if request_start is not None and due is not None else "",
        "submit_to_scheduler_queue_ms": submit_to_queue,
        "scheduler_queue_to_admit_ms": queue_to_admit,
        "admit_to_h2d_ms": admit_to_h2d,
    }


def max_numeric(events: list[dict[str, Any]], key: str) -> int | None:
    values: list[int] = []
    for event in events:
        value = as_float(event.get(key))
        if value is not None:
            values.append(int(value))
    return max(values) if values else None


def sum_duration(events: list[dict[str, Any]], category: str) -> float:
    total = 0.0
    for event in events:
        if event.get("category") != category:
            continue
        duration = as_float(event.get("duration_ms"))
        if duration is not None:
            total += duration
    return round(total, 3)


def count_category(events: list[dict[str, Any]], category: str) -> int:
    return sum(1 for event in events if event.get("category") == category)


def summarize_cache_path(events: list[dict[str, Any]], window_start_ms: Any = "", ttft_ms: Any = "") -> dict[str, Any]:
    if not events:
        return {
            "cache_event_count": 0,
            "input_tokens": "",
            "active_input_tokens": "",
            "scheduler_trimmed_tokens": "",
            "cached_prefix_tokens": "",
            "initial_cached_prefix_tokens": "",
            "final_cached_prefix_tokens": "",
            "new_prefill_tokens_est": "",
            "cache_hit_ratio_pct": "",
            "host_hit_tokens": "",
            "host_load_tokens": "",
            "cache_write_events": 0,
            "post_request_cache_write_events": 0,
            "progressive_cache_events": 0,
            "match_prefix_events": 0,
            "init_load_back_events": 0,
            "load_back_events": 0,
            "hicache_load_events": 0,
            "first_cache_event_delay_ms": "",
            "cache_work_end_to_first_token_ms": "",
            "cache_path_summary": "no SGLang cache events attributed",
        }
    compact_events = [event for event in events if event.get("source") == "kv_cache_telemetry"]
    events = compact_events or events
    input_tokens = max_numeric(events, "input_tokens")
    active_input_tokens = max_numeric(events, "active_input_tokens")
    prefix_categories = {"match_prefix", "ready_to_load_host_cache", "init_load_back", "load_back", "hicache_load"}
    prefix_events = [event for event in events if event.get("category") in prefix_categories]
    progressive_categories = {"cache_unfinished_req"}
    post_categories = {"cache_finished_req"}
    progressive_events = [event for event in events if event.get("category") in progressive_categories]
    post_events = [event for event in events if event.get("category") in post_categories]
    scheduler_trimmed_tokens = max_numeric(prefix_events, "scheduler_trimmed_tokens")
    prefix_counts = [
        int(value)
        for value in (as_float(event.get("cached_prefix_tokens")) for event in prefix_events)
        if value is not None
    ]
    initial_cached_prefix = prefix_counts[0] if prefix_counts else None
    cached_prefix = initial_cached_prefix
    final_cached_prefix = max_numeric(events, "cached_prefix_tokens")
    if scheduler_trimmed_tokens is not None:
        cached_prefix = max(cached_prefix or 0, scheduler_trimmed_tokens)
    host_hit = max_numeric(events, "host_hit_tokens")
    host_loads = [int(as_float(event.get("host_load_tokens")) or 0) for event in events if as_float(event.get("host_load_tokens")) is not None]
    host_load_tokens = max(host_loads) if host_loads else None
    new_prefill = ""
    hit_ratio = ""
    if input_tokens is not None and cached_prefix is not None:
        new_prefill = max(0, input_tokens - cached_prefix)
        hit_ratio = round(cached_prefix * 100.0 / input_tokens, 2) if input_tokens else 0.0
    event_times = [as_float(event.get("start_or_end_ms")) for event in events]
    event_times = [value for value in event_times if value is not None]
    first_delay = ""
    end_to_first = ""
    window_start = as_float(window_start_ms)
    first_token_at = None
    ttft = as_float(ttft_ms)
    if window_start is not None and event_times:
        first_delay = round(min(event_times) - window_start, 3)
        if ttft is not None:
            first_token_at = window_start + ttft
            before_first = [value for value in event_times if value <= first_token_at]
            if before_first:
                end_to_first = round(first_token_at - max(before_first), 3)
    load_tokens_text = f"host_load_tokens={host_load_tokens}" if host_load_tokens is not None else "host_load_tokens=0"
    prefix_text = f"cached_prefix={cached_prefix}/{input_tokens}" if input_tokens is not None and cached_prefix is not None else "cached_prefix=unknown"
    host_text = f"host_hit={host_hit}" if host_hit is not None else "host_hit=unknown"
    return {
        "cache_event_count": len(events),
        "input_tokens": input_tokens if input_tokens is not None else "",
        "active_input_tokens": active_input_tokens if active_input_tokens is not None else "",
        "scheduler_trimmed_tokens": scheduler_trimmed_tokens if scheduler_trimmed_tokens is not None else "",
        "cached_prefix_tokens": cached_prefix if cached_prefix is not None else "",
        "initial_cached_prefix_tokens": initial_cached_prefix if initial_cached_prefix is not None else "",
        "final_cached_prefix_tokens": final_cached_prefix if final_cached_prefix is not None else "",
        "new_prefill_tokens_est": new_prefill,
        "cache_hit_ratio_pct": hit_ratio,
        "host_hit_tokens": host_hit if host_hit is not None else "",
        "host_load_tokens": host_load_tokens if host_load_tokens is not None else "",
        "cache_write_events": count_category(events, "cache_finished_req") + count_category(events, "cache_unfinished_req"),
        "post_request_cache_write_events": len(post_events),
        "progressive_cache_events": len(progressive_events),
        "match_prefix_events": count_category(events, "match_prefix"),
        "ready_to_load_host_cache_events": count_category(events, "ready_to_load_host_cache"),
        "init_load_back_events": count_category(events, "init_load_back"),
        "load_back_events": count_category(events, "load_back"),
        "hicache_load_events": count_category(events, "hicache_load"),
        "match_prefix_total_ms": sum_duration(events, "match_prefix"),
        "init_load_back_total_ms": sum_duration(events, "init_load_back"),
        "hicache_load_total_ms": sum_duration(events, "hicache_load"),
        "first_cache_event_delay_ms": first_delay,
        "cache_work_end_to_first_token_ms": end_to_first,
        "cache_path_summary": (
            f"{prefix_text}; final_cached_prefix={final_cached_prefix if final_cached_prefix is not None else 'unknown'};"
            f" post_cache_writes={len(post_events)}; progressive_cache_events={len(progressive_events)};"
            f" {host_text}; {load_tokens_text}"
        ),
    }


def lifecycle_events_by_session(trace_rows: list[dict[str, Any]], base_ts: float) -> dict[str, list[dict[str, Any]]]:
    lifecycle_events = {
        "hicache.write.end": ("d2h_write", "device_to_host", "device_indices"),
        "hicache.evict_device.end": ("gpu_evict", "device_evict", "device_indices"),
        "hicache.evict_host.end": ("host_evict", "host_evict", "host_indices"),
        "hicache.load.end": ("h2d_load", "host_to_device", "host_indices"),
        "hiradix.init_load_back.end": ("init_load_back", "host_to_device", "host_indices"),
        "hiradix.load_back.end": ("load_back", "host_to_device", "host_indices"),
    }
    by_session: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        category_info = lifecycle_events.get(str(row.get("event") or ""))
        if not category_info:
            continue
        context = context_from_trace_event(row)
        session_id = agent_session_from_context(context)
        if not session_id or "::live_prefetch::" in session_id:
            continue
        category, direction, primary_index = category_info
        request = nested_request(context)
        primary_tokens = index_count(context.get(primary_index))
        if primary_tokens is None and category in {"init_load_back", "load_back"}:
            primary_tokens = int(as_float(context.get("host_hit_length")) or 0)
        if primary_tokens is None:
            primary_tokens = 0
        by_session[session_id].append(
            {
                "event": row.get("event", ""),
                "category": category,
                "direction": direction,
                "agent_phase": context.get("agent_phase") or request.get("agent_phase", "unknown"),
                "tokens": primary_tokens,
                "duration_ms": row.get("duration_ms", ""),
                "time_ms": rel_ms(row.get("ts_ns"), base_ts),
                "node_id": context.get("node_id", ""),
            }
        )
    return by_session


def summarize_kv_lifecycle(events: list[dict[str, Any]]) -> dict[str, Any]:
    totals: defaultdict[str, Counter[str]] = defaultdict(Counter)
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        phase = str(event.get("agent_phase") or "unknown")
        category = str(event.get("category") or "")
        tokens = int(as_float(event.get("tokens")) or 0)
        if category:
            totals[phase][category] += tokens
            counts[phase][category] += 1

    all_totals = Counter()
    for counter in totals.values():
        all_totals.update(counter)

    initial = totals.get("initial_turn", Counter())
    hint = totals.get("hint_prefetch", Counter())
    replay = totals.get("replay", Counter())
    replay_h2d_tokens = replay.get("h2d_load", 0) + replay.get("init_load_back", 0) + replay.get("load_back", 0)
    hint_h2d_tokens = hint.get("h2d_load", 0) + hint.get("init_load_back", 0) + hint.get("load_back", 0)
    host_write_tokens = initial.get("d2h_write", 0)
    gpu_evict_tokens = initial.get("gpu_evict", 0)
    host_evict_tokens = initial.get("host_evict", 0)

    if host_write_tokens and host_evict_tokens:
        verdict = "written_to_host_then_host_evicted"
        explanation = "KV reached host HiCache, but useful host KV was later evicted before replay could reload it."
    elif replay_h2d_tokens:
        verdict = "replay_loaded_from_host"
        explanation = "Replay used the host-to-device KV load-back path."
    elif hint_h2d_tokens:
        verdict = "prefetch_loaded_from_host"
        explanation = "The hint/prefetch path loaded KV from host before replay."
    elif host_write_tokens and gpu_evict_tokens:
        verdict = "written_to_host_gpu_evicted_no_replay_load"
        explanation = "KV was written to host and evicted from GPU, but replay did not show a load-back."
    elif host_write_tokens:
        verdict = "written_to_host_no_eviction"
        explanation = "KV was written to host; no target eviction/load-back was observed for this row."
    else:
        verdict = "no_lifecycle_movement_observed"
        explanation = "No lifecycle movement events were attributed to this row."

    return {
        "lifecycle_host_write_tokens": host_write_tokens or "",
        "lifecycle_gpu_evict_tokens": gpu_evict_tokens or "",
        "lifecycle_host_evict_tokens": host_evict_tokens or "",
        "lifecycle_hint_h2d_tokens": hint_h2d_tokens or "",
        "lifecycle_replay_h2d_tokens": replay_h2d_tokens or "",
        "lifecycle_total_d2h_write_tokens": all_totals.get("d2h_write", 0) or "",
        "lifecycle_total_h2d_load_tokens": (
            all_totals.get("h2d_load", 0) + all_totals.get("init_load_back", 0) + all_totals.get("load_back", 0)
        )
        or "",
        "lifecycle_verdict": verdict,
        "lifecycle_explanation": explanation,
        "lifecycle_event_counts": ", ".join(
            f"{phase}.{category}:{count}"
            for phase, counter in sorted(counts.items())
            for category, count in sorted(counter.items())
        ),
    }


def replay_path_from_evidence(row: dict[str, Any]) -> str:
    if has_events(row.get("replay_kv_h2d_events")):
        return "replay loaded KV"
    new_prefill = as_float(row.get("replay_new_prefill_tokens_est"))
    hit_ratio = as_float(row.get("replay_cache_hit_ratio_pct"))
    ttft = as_float(row.get("resume_ttft_ms"))
    if new_prefill is not None and new_prefill >= 128 and ttft is not None and ttft >= 1000:
        return "replay recompute/prefill suspected"
    if hit_ratio is not None and hit_ratio >= 90:
        return "mostly cache hit/resident"
    if ttft is not None and ttft >= 1000:
        return "scheduler/cache wait suspected"
    if ttft is None:
        return "unclear"
    return "likely cache hit/resident"


def per_gap_verdict(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or "")
    margin = as_float(row.get("prefetch_margin_ms"))
    replay_loaded = has_events(row.get("replay_kv_h2d_events"))
    hint_loaded = has_events(row.get("direct_kv_h2d_events"))
    replay_path_value = str(row.get("replay_path") or replay_path_from_evidence(row))
    hint_host_hit = as_float(row.get("hint_host_hit_tokens"))
    if mode == "no_prefetch":
        if replay_loaded:
            return "no_prefetch_replay_loaded_kv"
        if replay_path_value == "replay recompute/prefill suspected":
            return "no_prefetch_replay_recomputed"
        return "no_prefetch_cache_reused_or_scheduler_wait"
    if margin is None:
        return "prefetch_missing_or_unfinished"
    if margin < 0 and replay_loaded:
        return "prefetch_late_replay_loaded_kv"
    if margin < 0 and replay_path_value == "replay recompute/prefill suspected":
        return "prefetch_late_replay_recomputed"
    if margin < 0:
        return "prefetch_late_no_replay_h2d"
    if replay_loaded:
        return "prefetch_ready_but_replay_loaded_kv"
    if hint_loaded and replay_path_value in {"mostly cache hit/resident", "likely cache hit/resident"}:
        return "prefetch_success_cache_reused"
    if replay_path_value == "replay recompute/prefill suspected":
        return "prefetch_ready_but_replay_recomputed"
    if replay_path_value in {"mostly cache hit/resident", "likely cache hit/resident"}:
        if not hint_loaded and hint_host_hit == 0:
            return "prefetch_no_host_load_replay_cache_hit"
        return "prefetch_ready_replay_cache_hit"
    if not hint_loaded and hint_host_hit == 0:
        return "prefetch_ran_but_no_host_kv"
    return "prefetch_ready_no_replay_h2d"


def trace_request_windows(trace_rows: list[dict[str, Any]], base_ts: float) -> dict[tuple[str, str], dict[str, Any]]:
    windows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in trace_rows:
        if row.get("event") not in {"m27.request.submitted", "m27.request.start", "m27.request.end"}:
            continue
        session = str(row.get("session_id") or "")
        phase = str(row.get("phase") or "")
        if not session or not phase:
            continue
        item = windows.setdefault(
            (session, phase),
            {
                "session_id": session,
                "phase": phase,
                "mode": row.get("mode", ""),
                "label": row.get("label", ""),
                "prompt_hash": row.get("prompt_hash", ""),
            },
        )
        if row.get("event") == "m27.request.submitted":
            item["submitted_ms"] = rel_ms(row.get("ts_ns"), base_ts)
        elif row.get("event") == "m27.request.start":
            item["start_ms"] = rel_ms(row.get("ts_ns"), base_ts)
        else:
            item["end_ms"] = rel_ms(row.get("ts_ns"), base_ts)
            item["ttft_ms"] = row.get("ttft_ms", "")
            item["total_latency_ms"] = row.get("total_latency_ms", "")
    return windows


def first_event_by_session(trace_rows: list[dict[str, Any]], event_name: str, base_ts: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        if row.get("event") != event_name:
            continue
        session = str(row.get("session_id") or "")
        if not session or session in out:
            continue
        copied = dict(row)
        copied["ms"] = rel_ms(row.get("ts_ns"), base_ts)
        out[session] = copied
    return out


def latest_event_by_session(trace_rows: list[dict[str, Any]], event_name: str, base_ts: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        if row.get("event") != event_name:
            continue
        session = str(row.get("session_id") or "")
        if not session:
            continue
        copied = dict(row)
        copied["ms"] = rel_ms(row.get("ts_ns"), base_ts)
        out[session] = copied
    return out


def events_in_window(
    events_by_session: dict[str, list[dict[str, Any]]],
    session_id: str,
    start_ms: Any,
    end_ms: Any,
) -> list[dict[str, Any]]:
    start = as_float(start_ms)
    end = as_float(end_ms)
    if start is None or end is None:
        return []
    matched: list[dict[str, Any]] = []
    for event in events_by_session.get(session_id, []):
        ts = as_float(event.get("start_or_end_ms"))
        if ts is not None and start <= ts <= end:
            matched.append(event)
    return matched


def build_gaps_for_case(case_dir: Path, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
    telemetry_rows = read_jsonl(case_dir / "m27_copy_telemetry.jsonl")
    if not trace_rows:
        return [], []
    base_ts = min((float(row["ts_ns"]) / 1_000_000_000.0 for row in trace_rows if row.get("ts_ns")), default=0.0)
    requests = trace_request_windows(trace_rows, base_ts)
    session_meta = first_event_by_session(trace_rows, "m27.session.start", base_ts)
    tool_starts = first_event_by_session(trace_rows, "m27.tool_wait.start", base_ts)
    replay_due = first_event_by_session(trace_rows, "m27.replay.due", base_ts)
    pre_replay = first_event_by_session(trace_rows, "m27.pre_replay.checkpoint", base_ts)
    hint_submitted = first_event_by_session(trace_rows, "m27.hint.submitted", base_ts)
    prefetch_starts = first_event_by_session(trace_rows, "m27.prefetch.start", base_ts)
    prefetch_ends = latest_event_by_session(trace_rows, "m27.prefetch.end", base_ts)
    movement_by_session = movement_events_by_session(trace_rows, telemetry_rows, base_ts)
    cache_by_session = cache_events_by_session(trace_rows, base_ts)
    lifecycle_by_session = lifecycle_events_by_session(trace_rows, base_ts)
    scheduler_by_session = telemetry_events_by_session(
        trace_rows,
        base_ts,
        {"kv_telemetry.scheduler.start", "kv_telemetry.scheduler.end", "kv_telemetry.request_stage"},
    )
    prefill_by_session = telemetry_events_by_session(trace_rows, base_ts, "kv_telemetry.prefill.end")
    sessions = sorted(session_meta)
    gaps: list[dict[str, Any]] = []
    for idx, session in enumerate(sessions):
        meta = session_meta.get(session, {})
        current = requests.get((session, "initial_turn"), {})
        replay = requests.get((session, "replay"), {})
        tool = tool_starts.get(session, {})
        due = replay_due.get(session, {})
        checkpoint = pre_replay.get(session, {})
        hint = hint_submitted.get(session, {})
        p_start = prefetch_starts.get(session, {})
        p_end = prefetch_ends.get(session, {})
        if not current or not replay:
            continue
        due_ms = due.get("ms", "")
        prefetch_start_ms = p_start.get("ms", "")
        prefetch_end_ms = p_end.get("ms", "")
        hint_events = events_in_window(movement_by_session, session, prefetch_start_ms, prefetch_end_ms)
        replay_events = events_in_window(movement_by_session, session, replay.get("start_ms"), replay.get("end_ms"))
        hint_cache_events = events_in_window(cache_by_session, session, prefetch_start_ms, prefetch_end_ms)
        replay_cache_events = events_in_window(cache_by_session, session, replay.get("start_ms"), replay.get("end_ms"))
        replay_scheduler_events = events_in_window(scheduler_by_session, session, replay.get("start_ms"), replay.get("end_ms"))
        replay_prefill_events = events_in_window(prefill_by_session, session, replay.get("start_ms"), replay.get("end_ms"))
        hint_summary = summarize_movement(hint_events)
        replay_summary = summarize_movement(replay_events)
        hint_cache_summary = summarize_cache_path(hint_cache_events, prefetch_start_ms)
        replay_cache_summary = summarize_cache_path(replay_cache_events, replay.get("start_ms"), replay.get("ttft_ms", ""))
        lifecycle_summary = summarize_kv_lifecycle(lifecycle_by_session.get(session, []))
        replay_scheduler_summary = summarize_timed_events(replay_scheduler_events)
        replay_queue_summary = summarize_scheduler_queue_path(replay_scheduler_events, replay.get("start_ms", ""), due_ms)
        replay_prefill_summary = summarize_timed_events(replay_prefill_events)
        replay_prefill_end_ms = ""
        replay_ttft_ms = as_float(replay.get("ttft_ms", ""))
        replay_start_ms = as_float(replay.get("start_ms", ""))
        if replay_start_ms is not None and replay_ttft_ms is not None:
            replay_prefill_end_ms = round(replay_start_ms + replay_ttft_ms, 3)
        admit_to_h2d_ms = ""
        admit_start = as_float(replay_queue_summary.get("admit_start_ms"))
        replay_h2d_start = as_float(replay_summary.get("start_ms"))
        if admit_start is not None and replay_h2d_start is not None:
            admit_to_h2d_ms = round(replay_h2d_start - admit_start, 3)
        margin = ""
        if prefetch_end_ms not in ("", None) and due_ms not in ("", None):
            margin = round(float(due_ms) - float(prefetch_end_ms), 3)
        gap = {
            "session_id": session,
            "mode": mode,
            "task_index": meta.get("task_index", ""),
            "gap_order_in_task": idx,
            "tool_names": meta.get("tool_names", ""),
            "tool_call_count": 1 if meta.get("tool_names") else "",
            "tool_gap_ms": meta.get("tool_wait_ms", ""),
            "current_start_ms": current.get("start_ms", ""),
            "current_end_ms": current.get("end_ms", ""),
            "current_latency_ms": current.get("total_latency_ms", ""),
            "tool_gap_start_ms": tool.get("ms", ""),
            "tool_gap_end_ms": due_ms,
            "hint_submitted_ms": hint.get("ms", ""),
            "prefetch_start_ms": prefetch_start_ms,
            "prefetch_end_ms": prefetch_end_ms,
            "prefetch_duration_ms": (
                round(float(prefetch_end_ms) - float(prefetch_start_ms), 3)
                if prefetch_start_ms not in ("", None) and prefetch_end_ms not in ("", None)
                else ""
            ),
            "prefetch_margin_ms": margin,
            "prefetch_done_before_resume": 1 if isinstance(margin, (int, float)) and margin >= 0 else 0 if margin != "" else "",
            "prefetch_status": "done" if prefetch_end_ms not in ("", None) else "no_hint",
            "resume_start_ms": replay.get("start_ms", ""),
            "resume_submitted_ms": replay.get("submitted_ms", ""),
            "resume_end_ms": replay.get("end_ms", ""),
            "resume_latency_ms": replay.get("total_latency_ms", ""),
            "resume_ttft_ms": replay.get("ttft_ms", ""),
            "replay_prefill_start_ms": replay.get("start_ms", ""),
            "replay_prefill_end_ms": replay_prefill_end_ms,
            "replay_prefill_duration_ms": replay.get("ttft_ms", ""),
            "direct_kv_h2d_start_ms": hint_summary["start_ms"],
            "direct_kv_h2d_end_ms": hint_summary["end_ms"],
            "direct_kv_h2d_duration_ms": hint_summary["duration_ms"],
            "direct_kv_h2d_events": hint_summary["events"],
            "direct_kv_h2d_categories": hint_summary["categories"],
            "hint_cache_event_count": hint_cache_summary["cache_event_count"],
            "hint_input_tokens": hint_cache_summary["input_tokens"],
            "hint_active_input_tokens": hint_cache_summary["active_input_tokens"],
            "hint_scheduler_trimmed_tokens": hint_cache_summary["scheduler_trimmed_tokens"],
            "hint_cached_prefix_tokens": hint_cache_summary["cached_prefix_tokens"],
            "hint_initial_cached_prefix_tokens": hint_cache_summary["initial_cached_prefix_tokens"],
            "hint_final_cached_prefix_tokens": hint_cache_summary["final_cached_prefix_tokens"],
            "hint_new_prefill_tokens_est": hint_cache_summary["new_prefill_tokens_est"],
            "hint_cache_hit_ratio_pct": hint_cache_summary["cache_hit_ratio_pct"],
            "hint_host_hit_tokens": hint_cache_summary["host_hit_tokens"],
            "hint_host_load_tokens": hint_cache_summary["host_load_tokens"],
            "hint_progressive_cache_events": hint_cache_summary["progressive_cache_events"],
            "hint_post_request_cache_write_events": hint_cache_summary["post_request_cache_write_events"],
            "hint_match_prefix_events": hint_cache_summary["match_prefix_events"],
            "hint_init_load_back_events": hint_cache_summary["init_load_back_events"],
            "hint_load_back_events": hint_cache_summary["load_back_events"],
            "hint_hicache_load_events": hint_cache_summary["hicache_load_events"],
            "hint_cache_path_summary": hint_cache_summary["cache_path_summary"],
            "replay_kv_h2d_start_ms": replay_summary["start_ms"],
            "replay_kv_h2d_end_ms": replay_summary["end_ms"],
            "replay_kv_h2d_duration_ms": replay_summary["duration_ms"],
            "replay_kv_h2d_events": replay_summary["events"],
            "replay_kv_h2d_categories": replay_summary["categories"],
            "replay_cache_event_count": replay_cache_summary["cache_event_count"],
            "replay_input_tokens": replay_cache_summary["input_tokens"],
            "replay_active_input_tokens": replay_cache_summary["active_input_tokens"],
            "replay_scheduler_trimmed_tokens": replay_cache_summary["scheduler_trimmed_tokens"],
            "replay_cached_prefix_tokens": replay_cache_summary["cached_prefix_tokens"],
            "replay_initial_cached_prefix_tokens": replay_cache_summary["initial_cached_prefix_tokens"],
            "replay_final_cached_prefix_tokens": replay_cache_summary["final_cached_prefix_tokens"],
            "replay_new_prefill_tokens_est": replay_cache_summary["new_prefill_tokens_est"],
            "replay_cache_hit_ratio_pct": replay_cache_summary["cache_hit_ratio_pct"],
            "replay_host_hit_tokens": replay_cache_summary["host_hit_tokens"],
            "replay_host_load_tokens": replay_cache_summary["host_load_tokens"],
            "replay_cache_write_events": replay_cache_summary["cache_write_events"],
            "replay_progressive_cache_events": replay_cache_summary["progressive_cache_events"],
            "replay_post_request_cache_write_events": replay_cache_summary["post_request_cache_write_events"],
            "replay_match_prefix_events": replay_cache_summary["match_prefix_events"],
            "replay_ready_to_load_host_cache_events": replay_cache_summary["ready_to_load_host_cache_events"],
            "replay_init_load_back_events": replay_cache_summary["init_load_back_events"],
            "replay_load_back_events": replay_cache_summary["load_back_events"],
            "replay_hicache_load_events": replay_cache_summary["hicache_load_events"],
            "replay_match_prefix_total_ms": replay_cache_summary["match_prefix_total_ms"],
            "replay_init_load_back_total_ms": replay_cache_summary["init_load_back_total_ms"],
            "replay_hicache_load_total_ms": replay_cache_summary["hicache_load_total_ms"],
            "replay_first_cache_event_delay_ms": replay_cache_summary["first_cache_event_delay_ms"],
            "replay_cache_work_end_to_first_token_ms": replay_cache_summary["cache_work_end_to_first_token_ms"],
            "replay_cache_path_summary": replay_cache_summary["cache_path_summary"],
            "replay_scheduler_event_count": replay_scheduler_summary["event_count"],
            "replay_scheduler_start_ms": replay_scheduler_summary["start_ms"],
            "replay_scheduler_end_ms": replay_scheduler_summary["end_ms"],
            "replay_scheduler_total_ms": replay_scheduler_summary["duration_ms"],
            "replay_scheduler_categories": replay_scheduler_summary["categories"],
            "replay_sglang_receive_start_ms": replay_queue_summary["received_start_ms"],
            "replay_sglang_receive_end_ms": replay_queue_summary["received_end_ms"],
            "replay_scheduler_queue_enter_start_ms": replay_queue_summary["queue_enter_start_ms"],
            "replay_scheduler_queue_enter_end_ms": replay_queue_summary["queue_enter_end_ms"],
            "replay_scheduler_admit_start_ms": replay_queue_summary["admit_start_ms"],
            "replay_scheduler_admit_end_ms": replay_queue_summary["admit_end_ms"],
            "replay_scheduler_admit_category": replay_queue_summary["admit_category"],
            "replay_scheduler_admit_timing_source": replay_queue_summary["admit_timing_source"],
            "replay_scheduler_queue_waiting_len": replay_queue_summary["queue_waiting_len"],
            "replay_scheduler_queue_running_len": replay_queue_summary["queue_running_len"],
            "replay_scheduler_admit_running_batch_requests": replay_queue_summary["admit_running_batch_requests"],
            "replay_scheduler_admit_running_batch_extend_tokens": replay_queue_summary["admit_running_batch_extend_tokens"],
            "replay_request_start_lateness_ms": replay_queue_summary["request_start_lateness_ms"],
            "replay_submit_to_scheduler_queue_ms": replay_queue_summary["submit_to_scheduler_queue_ms"],
            "replay_scheduler_queue_to_admit_ms": replay_queue_summary["scheduler_queue_to_admit_ms"],
            "replay_scheduler_admit_to_h2d_ms": admit_to_h2d_ms,
            "replay_model_forward_event_count": replay_prefill_summary["event_count"],
            "replay_model_forward_start_ms": replay_prefill_summary["start_ms"],
            "replay_model_forward_end_ms": replay_prefill_summary["end_ms"],
            "replay_model_forward_total_ms": replay_prefill_summary["duration_ms"],
            "replay_model_forward_categories": replay_prefill_summary["categories"],
            "pre_replay_checkpoint_ms": checkpoint.get("ms", ""),
            "pre_replay_expected_reuse": checkpoint.get("expected_reuse", ""),
            "pre_replay_gpu_resident_tokens": checkpoint.get("gpu_resident_tokens", ""),
            "pre_replay_host_resident_tokens": checkpoint.get("host_resident_tokens", ""),
            "pre_replay_missing_tokens": checkpoint.get("missing_tokens", ""),
            "pre_replay_protected_tokens": checkpoint.get("protected_tokens", ""),
            **lifecycle_summary,
        }
        if has_events(gap["direct_kv_h2d_events"]) and has_events(gap["replay_kv_h2d_events"]):
            gap["movement_class"] = "hint and replay both moved KV"
        elif has_events(gap["direct_kv_h2d_events"]):
            gap["movement_class"] = "hint-side KV movement observed"
        elif has_events(gap["replay_kv_h2d_events"]):
            gap["movement_class"] = "replay loaded KV"
        else:
            gap["movement_class"] = "no visible HtoD"
        gap["replay_path"] = replay_path_from_evidence(gap)
        gap["per_gap_verdict"] = per_gap_verdict(gap)
        attach_replay_path_fields(gap)
        gaps.append(gap)
    return gaps, trace_rows


def mode_summary_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gap in gaps:
        by_mode[str(gap.get("mode") or "")].append(gap)
    rows: list[dict[str, Any]] = []
    for mode, items in sorted(by_mode.items()):
        margins = [float(row["prefetch_margin_ms"]) for row in items if row.get("prefetch_margin_ms") not in ("", None)]
        replay_ttfts = [float(row["resume_ttft_ms"]) for row in items if row.get("resume_ttft_ms") not in ("", None)]
        replay_paths = [str(row.get("replay_path") or replay_path_from_evidence(row)) for row in items]
        verdicts = [str(row.get("per_gap_verdict") or per_gap_verdict(row)) for row in items]
        rows.append(
            {
                "mode": mode,
                "controlled_gaps": len(items),
                "prefetch_attempts": len(margins),
                "late_prefetches": sum(1 for value in margins if value < 0),
                "median_prefetch_margin_ms": round(median(margins), 3) if margins else "",
                "avg_resume_ttft_ms": round(mean(replay_ttfts), 3) if replay_ttfts else "",
                "hint_h2d_gaps": sum(1 for row in items if has_events(row.get("direct_kv_h2d_events"))),
                "replay_h2d_gaps": sum(1 for row in items if has_events(row.get("replay_kv_h2d_events"))),
                "replay_recompute_or_wait_suspected_gaps": sum(
                    1 for value in replay_paths if value in {"replay recompute/prefill suspected", "scheduler/cache wait suspected"}
                ),
                "likely_cache_hit_or_resident_gaps": replay_paths.count("likely cache hit/resident"),
                "mostly_cache_hit_or_resident_gaps": replay_paths.count("mostly cache hit/resident"),
                "verdicts": ", ".join(f"{name}:{count}" for name, count in sorted(Counter(verdicts).items())),
            }
        )
    return rows


def timeline_mode_rank(row: dict[str, Any]) -> tuple[int, str]:
    mode = str(row.get("mode") or "")
    ranks = {
        "no_prefetch": 0,
        "direct_prefetch": 1,
        "request_warm": 2,
        "oracle_prefetch": 3,
        "oracle_direct_load": 3,
    }
    return ranks.get(mode, 9), mode


def selected_timeline_gaps(gaps: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    by_mode: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gaps:
        by_mode[timeline_mode_rank(row)].append(row)

    ordered: list[dict[str, Any]] = []
    for mode_key in sorted(by_mode):
        mode_rows = by_mode[mode_key]
        visible = [row for row in mode_rows if has_visible_kv_movement(row)]
        visible_ids = {id(row) for row in visible}
        fallback = [row for row in mode_rows if id(row) not in visible_ids]
        ordered.extend(visible + fallback)
    return ordered[:max_rows]


def has_visible_kv_movement(row: dict[str, Any]) -> bool:
    return has_events(row.get("direct_kv_h2d_events")) or has_events(row.get("replay_kv_h2d_events"))


def prefetch_attempt_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in gaps
        if str(row.get("mode") or "") == "direct_prefetch"
        and as_float(row.get("prefetch_margin_ms")) is not None
    ]


def focused_prefetch_gaps(
    gaps: list[dict[str, Any]], ready: bool, max_rows: int
) -> tuple[list[dict[str, Any]], int, int, int]:
    attempts = prefetch_attempt_gaps(gaps)
    matching = [
        row
        for row in attempts
        if bool(as_float(row.get("prefetch_margin_ms")) is not None and as_float(row.get("prefetch_margin_ms")) >= 0)
        == ready
    ]
    selected = [row for row in matching if has_visible_kv_movement(row)]
    if ready:
        matching.sort(key=lambda row: as_float(row.get("prefetch_margin_ms")) or 0.0, reverse=True)
        selected.sort(key=lambda row: as_float(row.get("prefetch_margin_ms")) or 0.0, reverse=True)
    else:
        matching.sort(key=lambda row: as_float(row.get("prefetch_margin_ms")) or 0.0)
        selected.sort(key=lambda row: as_float(row.get("prefetch_margin_ms")) or 0.0)
    selected_ids = {id(row) for row in selected}
    fallback = [row for row in matching if id(row) not in selected_ids]
    return (selected + fallback)[:max_rows], len(selected), len(matching), len(attempts)


def timeline_rows_with_labels(rows: list[dict[str, Any]], prefix: str = "G") -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        copied = dict(row)
        copied["timeline_label"] = f"{prefix}{idx:02d}"
        labeled.append(copied)
    return labeled


def case_fillers(row: dict[str, Any]) -> str:
    name = Path(str(row.get("case_dir") or "")).name
    if "_f" not in name:
        return ""
    return name.rsplit("_f", 1)[-1]


def timeline_mapping_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        output.append(
            {
                "row": row.get("timeline_label") or f"G{idx:02d}",
                "mode": row.get("mode", ""),
                "fillers": case_fillers(row),
                "task": row.get("task_index", ""),
                "gap": row.get("gap_order_in_task", ""),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                "replay_path": row.get("replay_path", replay_path_from_evidence(row)),
                "final_path": row.get("final_path", ""),
                "bottleneck_label": row.get("bottleneck_label", ""),
                "path_confidence": row.get("path_confidence", ""),
                "replay_cache_hit_pct": row.get("replay_cache_hit_ratio_pct", ""),
                "replay_new_prefill_tokens_est": row.get("replay_new_prefill_tokens_est", ""),
                "replay_final_cached_prefix_tokens": row.get("replay_final_cached_prefix_tokens", ""),
                "replay_progressive_cache_events": row.get("replay_progressive_cache_events", ""),
                "replay_post_request_cache_write_events": row.get("replay_post_request_cache_write_events", ""),
                "host_write_tokens": row.get("lifecycle_host_write_tokens", ""),
                "gpu_evict_tokens": row.get("lifecycle_gpu_evict_tokens", ""),
                "host_evict_tokens": row.get("lifecycle_host_evict_tokens", ""),
                "lifecycle": row.get("lifecycle_verdict", ""),
                "verdict": row.get("per_gap_verdict", ""),
                "movement": row.get("movement_class", ""),
            }
        )
    return output


def timeline_kv_outcome_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        outcome, _color, meaning = timeline_kv_outcome(row)
        hint_h2d = row.get("direct_kv_h2d_events", "")
        replay_h2d = row.get("replay_kv_h2d_events", "")
        recomputed = row.get("recomputed_tokens_est") or row.get("replay_new_prefill_tokens_est", "")
        output.append(
            {
                "row": row.get("timeline_label") or f"G{idx:02d}",
                "kv_outcome": outcome,
                "prefetch_timing": row.get("per_gap_verdict", ""),
                "hint_h2d_events": hint_h2d,
                "replay_h2d_events": replay_h2d,
                "recomputed_tokens_est": recomputed,
                "normal_prefill_or_wait_ms_est": row.get("prefill_compute_ms_est", ""),
                "ttft_ms": row.get("resume_ttft_ms", ""),
                "simple_meaning": meaning,
            }
        )
    return output


def kv_lifecycle_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        output.append(
            {
                "row": row.get("timeline_label") or f"G{idx:02d}",
                "mode": row.get("mode", ""),
                "lifecycle_verdict": row.get("lifecycle_verdict", ""),
                "lifecycle_explanation": row.get("lifecycle_explanation", ""),
                "fillers": case_fillers(row),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "simple_meaning": kv_lifecycle_simple_meaning(row),
                "host_write_tokens": row.get("lifecycle_host_write_tokens", ""),
                "gpu_evict_tokens": row.get("lifecycle_gpu_evict_tokens", ""),
                "host_evict_tokens": row.get("lifecycle_host_evict_tokens", ""),
                "hint_h2d_tokens": row.get("lifecycle_hint_h2d_tokens", ""),
                "replay_h2d_tokens": row.get("lifecycle_replay_h2d_tokens", ""),
                "replay_initial_match_tokens": row.get("replay_initial_cached_prefix_tokens", ""),
                "replay_final_cached_tokens": row.get("replay_final_cached_prefix_tokens", ""),
                "replay_new_prefill_tokens": row.get("replay_new_prefill_tokens_est", ""),
            }
        )
    return output


def kv_block_gap_table_rows(gaps: list[dict[str, Any]], kv_block_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return gap_lifecycle_summary_rows(gaps, kv_block_rows)


def kv_block_detail_rows(kv_block_rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    selected = kv_block_rows[:limit] if limit is not None else kv_block_rows
    columns = [
        "block_id",
        "session_id",
        "lifecycle_verdict",
        "lifecycle_explanation",
        "exact_attribution",
        "has_exact_host_device_indices",
        "loaded_by_hint",
        "loaded_by_replay",
        "lost_before_replay",
        "token_start",
        "token_end",
        "token_count",
        "node_id",
        "host_index_start",
        "host_index_end",
        "host_index_count",
        "device_index_start",
        "device_index_end",
        "device_index_count",
        "current_state",
        "first_write_host_ms",
        "first_evict_gpu_ms",
        "first_evict_host_ms",
        "first_load_gpu_ms",
        "last_load_gpu_ms",
        "write_host_events",
        "evict_gpu_events",
        "evict_host_events",
        "load_gpu_events",
        "hint_load_gpu_events",
        "replay_load_gpu_events",
        "confidence",
        "lifecycle_steps",
    ]
    return [{column: row.get(column, "") for column in columns} for row in selected]


def exact_block_lifecycle_rows_for_sample(
    gaps: list[dict[str, Any]],
    kv_block_rows: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = block_lifecycle_by_gap_rows(gaps, kv_block_rows)
    return block_lifecycle_focus_rows(rows, limit=limit)


def detailed_kv_lifecycle_table_rows(
    gaps: list[dict[str, Any]],
    kv_block_rows: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = block_lifecycle_by_gap_rows(gaps, kv_block_rows)
    if limit is not None:
        rows = rows[:limit]
    columns = [
        "row",
        "mode",
        "lifecycle_verdict",
        "lifecycle_explanation",
        "block_id",
        "node_id",
        "request_id",
        "correlation_id",
        "case_id",
        "gap_id",
        "token_range",
        "token_count",
        "first_write_host_ms",
        "first_evict_gpu_ms",
        "first_evict_host_ms",
        "h2d_start_ms",
        "h2d_end_ms",
        "h2d_duration_ms",
        "recompute_start_ms_est",
        "recompute_end_ms_est",
        "recompute_duration_ms_est",
        "replay_due_ms",
        "replay_start_ms",
        "first_token_ms",
        "replay_end_ms",
        "loaded_by_replay",
        "loaded_by_hint",
        "lost_before_replay",
        "write_host_events",
        "evict_gpu_events",
        "evict_host_events",
        "load_gpu_events",
        "replay_load_gpu_events",
        "hint_load_gpu_events",
        "exact_attribution",
        "confidence",
        "evidence_level",
        "exact_correlation_source",
        "host_index_signature",
        "device_index_signature",
        "lifecycle_steps",
        "evidence_summary",
    ]
    return [{column: row.get(column, "") for column in columns} for row in rows]


def detailed_kv_lifecycle_column_guide_rows() -> list[dict[str, str]]:
    return [
        {"column": "row", "meaning": "Timeline row, for example G00."},
        {"column": "mode", "meaning": "Experiment mode, such as no_prefetch or direct_prefetch."},
        {"column": "lifecycle_verdict", "meaning": "Short verdict for what happened to this KV block overall."},
        {"column": "lifecycle_explanation", "meaning": "Plain English explanation of the verdict."},
        {"column": "block_id", "meaning": "Stable logical ID assigned by our KV block ledger."},
        {"column": "node_id", "meaning": "SGLang radix/cache node ID if SGLang exposed one."},
        {"column": "request_id", "meaning": "Driver/SGLang request identity when the trace preserved it."},
        {"column": "correlation_id", "meaning": "Stable request-correlation ID carried from the workload driver when available."},
        {"column": "case_id", "meaning": "Controlled-case identity used to align movement events to a timeline row."},
        {"column": "gap_id", "meaning": "Tool-gap identity when the workload exposes one."},
        {"column": "token_range", "meaning": "Approximate token/index range covered by this logical KV block."},
        {"column": "token_count", "meaning": "Approximate number of KV indices/tokens in the block."},
        {"column": "first_write_host_ms", "meaning": "When this KV block was first backed up from GPU memory to host HiCache."},
        {"column": "first_evict_gpu_ms", "meaning": "When this block left GPU residency, if observed."},
        {"column": "first_evict_host_ms", "meaning": "When the host-side copy was lost, if observed."},
        {"column": "h2d_start_ms", "meaning": "When host-to-device reload of this block started."},
        {"column": "h2d_end_ms", "meaning": "When host-to-device reload of this block finished."},
        {"column": "h2d_duration_ms", "meaning": "How long the visible KV host-to-device movement took."},
        {"column": "recompute_start_ms_est", "meaning": "Estimated start of replay recompute/prefill when old KV had to be rebuilt."},
        {"column": "recompute_end_ms_est", "meaning": "Estimated end of replay recompute/prefill."},
        {"column": "recompute_duration_ms_est", "meaning": "Estimated time spent rebuilding missing prefix/KV work."},
        {"column": "replay_due_ms", "meaning": "When replay ideally needed the KV to already be ready."},
        {"column": "replay_start_ms", "meaning": "When the replay request actually started running."},
        {"column": "first_token_ms", "meaning": "When the replay produced its first output token."},
        {"column": "replay_end_ms", "meaning": "When the replay request finished."},
        {"column": "loaded_by_replay", "meaning": "1 means replay itself loaded this block from host to GPU."},
        {"column": "loaded_by_hint", "meaning": "1 means the hint/prefetch path loaded this block before replay."},
        {"column": "lost_before_replay", "meaning": "1 means the block was gone before replay could reuse it."},
        {"column": "write_host_events", "meaning": "Number of observed events backing this block up to host memory."},
        {"column": "evict_gpu_events", "meaning": "Number of observed GPU-residency eviction events for this block."},
        {"column": "evict_host_events", "meaning": "Number of observed host-cache eviction events for this block."},
        {"column": "load_gpu_events", "meaning": "Number of observed host-to-GPU load events for this block."},
        {"column": "replay_load_gpu_events", "meaning": "Number of host-to-GPU loads attributed to the replay path."},
        {"column": "hint_load_gpu_events", "meaning": "Number of host-to-GPU loads attributed to the hint/prefetch path."},
        {"column": "exact_attribution", "meaning": "What exact evidence was available, such as host and device index signatures."},
        {"column": "confidence", "meaning": "How strong the lifecycle evidence is."},
        {"column": "evidence_level", "meaning": "Whether the row is exact indexed evidence, partial direct evidence, timed evidence, or inferred."},
        {"column": "exact_correlation_source", "meaning": "Which field tied the movement/block back to a request or session."},
        {"column": "host_index_signature", "meaning": "Stable fingerprint of the host-side KV indices involved."},
        {"column": "device_index_signature", "meaning": "Stable fingerprint of the GPU-side KV indices involved."},
        {"column": "lifecycle_steps", "meaning": "Compact event history, read left to right."},
        {"column": "evidence_summary", "meaning": "Plain explanation of the evidence for this row."},
    ]


def exact_movement_table_rows(rows: list[dict[str, Any]], gaps: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    label_by_session = {
        str(gap.get("ledger_session_id") or gap.get("session_id") or ""): gap.get("timeline_label") or f"G{idx:02d}"
        for idx, gap in enumerate(gaps)
    }
    sampled_sessions = set(label_by_session)
    prioritized = [row for row in rows if str(row.get("session_id") or "") in sampled_sessions]
    if not prioritized:
        prioritized = rows
    selected = prioritized[:limit] if limit is not None else prioritized
    columns = [
        "row",
        "session_id",
        "phase",
        "movement",
        "movement_kind",
        "direction",
        "request_id",
        "correlation_id",
        "case_id",
        "gap_id",
        "copy_start_ms",
        "copy_end_ms",
        "duration_ms",
        "host_index_start",
        "host_index_end",
        "host_index_count",
        "device_index_start",
        "device_index_end",
        "device_index_count",
        "node_id",
        "layer_id",
        "source_event",
        "confidence",
        "evidence_level",
        "exact_correlation_source",
        "simple_meaning",
    ]
    output: list[dict[str, Any]] = []
    for row in selected:
        copied = dict(row)
        copied["row"] = label_by_session.get(str(row.get("session_id") or ""), "")
        output.append({column: copied.get(column, "") for column in columns})
    return output


def exact_attribution_explainer_rows() -> list[dict[str, str]]:
    return [
        {
            "evidence": "host_index_signature",
            "simple meaning": "Stable fingerprint of the host-side KV indices SGLang used.",
            "why it matters": "Lets us connect write/load/evict events that touch the same host KV block set.",
        },
        {
            "evidence": "device_index_signature",
            "simple meaning": "Stable fingerprint of the GPU-side KV indices SGLang used.",
            "why it matters": "Lets us see where host KV was loaded back into GPU KV storage.",
        },
        {
            "evidence": "copy_start_ms / copy_end_ms",
            "simple meaning": "The measured window around the SGLang KV movement function.",
            "why it matters": "This gives the closest SGLang-level timing for when KV movement was acted on and completed.",
        },
        {
            "evidence": "hostpool.load_to_device_per_layer",
            "simple meaning": "Lower-level host-pool load path used during host-to-GPU KV movement.",
            "why it matters": "This is closer to the actual H2D movement than the high-level HiCache load call.",
        },
        {
            "evidence": "loaded_by_replay",
            "simple meaning": "The replay request itself loaded this tracked block back to GPU.",
            "why it matters": "This tells us the movement happened on the critical user-visible resume path.",
        },
        {
            "evidence": "loaded_by_hint",
            "simple meaning": "The prefetch/hint path loaded this tracked block back to GPU.",
            "why it matters": "This tells us the hint path actually did useful KV movement before or during replay.",
        },
    ]


def kv_lifecycle_legend_rows() -> list[dict[str, str]]:
    return [
        {
            "field": "host_write_tokens",
            "simple meaning": "SGLang copied this session's KV from GPU memory into host HiCache.",
        },
        {
            "field": "gpu_evict_tokens",
            "simple meaning": "SGLang removed this session's KV from GPU memory.",
        },
        {
            "field": "host_evict_tokens",
            "simple meaning": "SGLang also removed this session's host-side HiCache copy.",
        },
        {
            "field": "hint_h2d_tokens",
            "simple meaning": "The prefetch attempt loaded KV from host back to GPU.",
        },
        {
            "field": "replay_h2d_tokens",
            "simple meaning": "The real replay request loaded KV from host back to GPU.",
        },
        {
            "field": "replay_initial_match_tokens",
            "simple meaning": "How much prefix/KV SGLang could reuse when replay first arrived.",
        },
        {
            "field": "replay_new_prefill_tokens",
            "simple meaning": "Estimated tokens replay had to rebuild/prefill because they were not immediately reusable.",
        },
    ]


def kv_lifecycle_simple_meaning(row: dict[str, Any]) -> str:
    host_write = as_float(row.get("lifecycle_host_write_tokens")) or 0
    gpu_evict = as_float(row.get("lifecycle_gpu_evict_tokens")) or 0
    host_evict = as_float(row.get("lifecycle_host_evict_tokens")) or 0
    hint_h2d = as_float(row.get("lifecycle_hint_h2d_tokens")) or 0
    replay_h2d = as_float(row.get("lifecycle_replay_h2d_tokens")) or 0
    initial_match = as_float(row.get("replay_initial_cached_prefix_tokens")) or 0
    new_prefill = as_float(row.get("replay_new_prefill_tokens_est")) or 0

    if host_write and gpu_evict and host_evict and not replay_h2d and new_prefill >= 128:
        return (
            "The target KV existed and was written to host, but pressure evicted it from both GPU and host. "
            "Replay could not reload the old KV, matched only a small prefix, and rebuilt/prefilled the missing tokens."
        )
    if replay_h2d:
        return (
            "Replay found useful KV in the host tier and loaded it back to GPU. "
            "This is the clean host-to-device load-back path."
        )
    if hint_h2d and not replay_h2d and initial_match >= 128:
        return (
            "The prefetch path loaded KV from host before replay. Replay did not need its own visible H2D load."
        )
    if host_write and gpu_evict and not host_evict and not replay_h2d:
        return (
            "The target KV was written to host and evicted from GPU, but replay did not show a host load-back. "
            "This means the host-backed path was not used for this replay."
        )
    if host_write and not gpu_evict and not host_evict and initial_match >= 128:
        return (
            "The target KV was written to host, but it also appears to have stayed reusable for replay. "
            "Replay mostly reused existing KV instead of loading or recomputing a large prefix."
        )
    if new_prefill >= 128:
        return (
            "Replay had to build many missing prefix tokens. No useful host-to-device load was visible for this row."
        )
    return str(row.get("lifecycle_explanation") or "No clear KV lifecycle path was visible for this row.")


def replay_attribution_rows(rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    selected = rows[:limit] if limit is not None else rows
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        output.append(
            {
                "row": row.get("timeline_label") or f"G{idx:02d}",
                "mode": row.get("mode", ""),
                "task": row.get("task_index", ""),
                "gap": row.get("gap_order_in_task", ""),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                "input_tokens": row.get("replay_input_tokens", ""),
                "active_input_tokens": row.get("replay_active_input_tokens", ""),
                "scheduler_trimmed_tokens": row.get("replay_scheduler_trimmed_tokens", ""),
                "cached_prefix_tokens": row.get("replay_cached_prefix_tokens", ""),
                "final_cached_prefix_tokens": row.get("replay_final_cached_prefix_tokens", ""),
                "cache_hit_pct": row.get("replay_cache_hit_ratio_pct", ""),
                "new_prefill_tokens_est": row.get("replay_new_prefill_tokens_est", ""),
                "host_hit_tokens": row.get("replay_host_hit_tokens", ""),
                "host_load_tokens": row.get("replay_host_load_tokens", ""),
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "cache_events": row.get("replay_cache_event_count", ""),
                "progressive_cache_events": row.get("replay_progressive_cache_events", ""),
                "post_request_cache_write_events": row.get("replay_post_request_cache_write_events", ""),
                "first_cache_event_delay_ms": row.get("replay_first_cache_event_delay_ms", ""),
                "scheduler_events": row.get("replay_scheduler_event_count", ""),
                "scheduler_total_ms": row.get("replay_scheduler_total_ms", ""),
                "model_forward_events": row.get("replay_model_forward_event_count", ""),
                "model_forward_total_ms": row.get("replay_model_forward_total_ms", ""),
                "replay_path": row.get("replay_path", ""),
                "final_path": row.get("final_path", ""),
                "bottleneck_label": row.get("bottleneck_label", ""),
                "path_confidence": row.get("path_confidence", ""),
                "verdict": row.get("per_gap_verdict", ""),
            }
        )
    return output


def verdict_summary_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for row in gaps:
        mode = str(row.get("mode") or "")
        verdict = str(row.get("per_gap_verdict") or per_gap_verdict(row))
        by_mode[mode][verdict] += 1
        totals[mode] += 1
    output: list[dict[str, Any]] = []
    for mode, verdicts in sorted(by_mode.items()):
        total = totals[mode]
        for verdict, count in sorted(verdicts.items()):
            output.append(
                {
                    "mode": mode,
                    "verdict": verdict,
                    "gaps": count,
                    "pct": round(count * 100.0 / total, 2) if total else 0.0,
                }
            )
    return output


def replay_path_proof_rows(ledger: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    selected = ledger[:limit] if limit is not None else ledger
    columns = [
        "row",
        "mode",
        "tool_gap_ms",
        "resume_ttft_ms",
        "final_path",
        "bottleneck_label",
        "confidence",
        "prefetch_outcome",
        "input_tokens",
        "active_input_tokens",
        "scheduler_trimmed_tokens",
        "matched_prefix_tokens",
        "final_cached_prefix_tokens",
        "unmatched_tokens",
        "host_load_tokens",
        "recomputed_tokens_est",
        "progressive_cache_events",
        "post_request_cache_write_events",
        "scheduler_wait_ms",
        "kv_prepare_ms",
        "model_forward_ms",
        "direct_h2d_events",
        "replay_h2d_events",
        "replay_scheduler_event_count",
        "replay_scheduler_total_ms",
        "replay_model_forward_event_count",
        "replay_model_forward_total_ms",
        "evidence_summary",
    ]
    return [{column: row.get(column, "") for column in columns} for row in selected]


def hardware_counterfactual_rows(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [
        "row",
        "mode",
        "tool_gap_ms",
        "prefetch_margin_ms",
        "observed_software_prefetch_duration_ms",
        "observed_copy_ms",
        "available_tool_gap_ms",
        "deadline_miss_ms",
        "counterfactual_verdict",
        "counterfactual_reason",
    ]
    return [{column: row.get(column, "") for column in columns} for row in ledger]


def request_id_coverage_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"events": 0, "with_agent_session": 0, "with_request_identity": 0}
    )
    for row in trace_rows:
        event = str(row.get("event") or "")
        if not event:
            continue
        if event.startswith("kv_telemetry.scheduler"):
            group = "scheduler"
        elif event.startswith("kv_telemetry.prefill") or event.startswith("worker."):
            group = "model_forward_prefill"
        elif event.startswith("kv_telemetry.copy") or event.startswith("hostpool."):
            group = "copy_load_back"
        elif event.startswith("kv_telemetry.cache") or event.startswith("hiradix.") or event.startswith("radix.") or event.startswith("hicache."):
            group = "cache_hicache"
        elif event.startswith("m27."):
            group = "driver"
        else:
            continue
        context = context_from_trace_event(row)
        sessions = _agent_sessions_for_event(row)
        request_id = (
            row.get("request_id")
            or row.get("agent_request_id")
            or row.get("correlation_id")
            or context.get("request_id")
            or context.get("agent_request_id")
            or context.get("agent_correlation_id")
        )
        req = context.get("request")
        if not request_id and isinstance(req, dict):
            request_id = (
                req.get("rid")
                or req.get("request_id")
                or req.get("agent_request_id")
                or req.get("agent_correlation_id")
                or req.get("agent_label")
            )
        item = grouped[group]
        item["events"] += 1
        if sessions or row.get("session_id") or row.get("agent_session_id"):
            item["with_agent_session"] += 1
        if request_id:
            item["with_request_identity"] += 1
    rows: list[dict[str, Any]] = []
    for group, counts in sorted(grouped.items()):
        total = counts["events"]
        rows.append(
            {
                "trace_area": group,
                "events": total,
                "with_agent_session": counts["with_agent_session"],
                "agent_session_coverage_pct": round(counts["with_agent_session"] * 100.0 / total, 2) if total else 0.0,
                "with_request_identity": counts["with_request_identity"],
                "request_identity_coverage_pct": round(counts["with_request_identity"] * 100.0 / total, 2)
                if total
                else 0.0,
            }
        )
    return rows


def observation_status(row: dict[str, Any]) -> tuple[str, str]:
    mode = str(row.get("mode") or "")
    margin = as_float(row.get("prefetch_margin_ms"))
    hint_h2d = has_events(row.get("direct_kv_h2d_events"))
    replay_h2d = has_events(row.get("replay_kv_h2d_events"))
    if mode == "no_prefetch":
        if replay_h2d:
            return "No prefetch; replay loaded KV", "#b45309"
        return "No prefetch; no visible H2D", "#64748b"
    if margin is None:
        if replay_h2d:
            return "No completed prefetch; replay loaded KV", "#b45309"
        return "No completed prefetch", "#64748b"
    if margin < 0:
        if replay_h2d:
            return "Late prefetch; replay loaded KV", "#b91c1c"
        return "Late prefetch", "#b91c1c"
    if hint_h2d and replay_h2d:
        return "Prefetch ready, but replay also loaded KV", "#b45309"
    if hint_h2d:
        return "Useful direct KV prefetch", "#166534"
    if replay_h2d:
        return "Prefetch finished, replay loaded KV", "#b45309"
    return "Prefetch finished; no visible H2D", "#166534"


def key_observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        mode = str(row.get("mode") or "")
        margin = as_float(row.get("prefetch_margin_ms"))
        gap_ms = as_float(row.get("tool_gap_ms"))
        ttft_ms = as_float(row.get("resume_ttft_ms"))
        hint_h2d = has_events(row.get("direct_kv_h2d_events"))
        replay_h2d = has_events(row.get("replay_kv_h2d_events"))
        path = row.get("replay_path") or replay_path_from_evidence(row)
        verdict = row.get("per_gap_verdict") or per_gap_verdict(row)
        cache_summary = row.get("replay_cache_path_summary") or ""
        lifecycle_summary = row.get("lifecycle_explanation") or ""
        status, _ = observation_status(row)
        ttft_note = (
            f" Replay waited {ttft_ms:.0f} ms before first token."
            if ttft_ms is not None
            else ""
        )

        if mode == "no_prefetch":
            if replay_h2d:
                what = "No hint was issued. When the resume request arrived, SGLang performed replay-side KV HtoD movement." + ttft_note
                why = "This is the baseline: the real request path had to handle KV movement at resume time."
            else:
                what = "No hint was issued, and this row did not show replay-side HtoD movement." + ttft_note
                if path == "recompute/scheduler wait suspected":
                    why = "The TTFT window is long even without cyan HtoD, so replay-side recompute, prefill, or scheduler waiting is suspected."
                else:
                    why = "The KV may already have been resident/reusable, or this row did not trigger observable host-to-device movement."
        elif margin is None:
            what = "A prefetch mode was selected, but the trace did not show a completed prefetch window for this row."
            why = "This is useful as a control/coverage warning, but it is weaker evidence than rows with measured margins."
        elif margin < 0:
            what = f"The prefetch attempt finished {abs(margin):.0f} ms after the resume request was already due." + ttft_note
            if replay_h2d:
                what += " The resume request also showed replay-side KV HtoD movement."
                why = "This is the failure case: the normal request path had to move KV because the hint path did not finish in time."
            elif path == "recompute/scheduler wait suspected":
                why = "The hint path was late and the TTFT window is long, so replay-side recompute, prefill, or scheduler waiting is suspected."
            else:
                why = "The hint path was late, so software prefetch did not meet the agent resume deadline."
        else:
            what = f"The prefetch attempt finished {margin:.0f} ms before the resume request was due." + ttft_note
            if hint_h2d and not replay_h2d:
                what += " Direct KV HtoD movement was visible on the hint side, with no replay-side HtoD in this row."
                why = "This is the clean success case: the hint appears to have prepared KV before the agent resumed."
            elif hint_h2d and replay_h2d:
                what += " But replay-side KV HtoD was also visible later."
                why = "This is an important hardware argument: moving KV early is not enough unless residency and reuse are also protected."
            elif replay_h2d:
                what += " Replay-side KV HtoD was still visible."
                why = "The software hint completed early, but the resume path still had to move KV, so reuse was not fully predictable."
            else:
                why = "The hint completed before replay, but this row did not show visible HtoD movement."

        output.append(
            {
                "row": label,
                "mode": mode,
                "status": status,
                "what happened": what,
                "why it matters": " ".join(
                    part
                    for part in (
                        why,
                        f"Lifecycle: {lifecycle_summary}" if lifecycle_summary else "",
                        f"Replay evidence: {cache_summary}" if cache_summary else "",
                    )
                    if part
                ),
                "tool_wait_ms": round(gap_ms, 3) if gap_ms is not None else "",
                "resume_ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else "",
                "replay_path": path,
                "verdict": verdict,
            }
        )
    return output


def load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def env_value(env: dict[str, Any], *keys: str) -> str:
    current: Any = env
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    if current in (None, ""):
        return ""
    return str(current)


def first_gpu(env: dict[str, Any]) -> dict[str, Any]:
    gpus = env.get("gpu", {}).get("gpus", []) if isinstance(env.get("gpu"), dict) else []
    if isinstance(gpus, list) and gpus and isinstance(gpus[0], dict):
        return gpus[0]
    return {}


def mib_to_gib(value: Any) -> str:
    try:
        return f"{float(value) / 1024.0:.2f} GiB"
    except (TypeError, ValueError):
        return ""


def runtime_config_rows(run_env: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    gpu = first_gpu(run_env)
    run_config = run_env.get("run_config", {}) if isinstance(run_env.get("run_config"), dict) else {}
    sglang_args = run_env.get("sglang", {}).get("server_args", {}) if isinstance(run_env.get("sglang"), dict) else {}
    sglang_runtime = run_env.get("sglang", {}).get("runtime", {}) if isinstance(run_env.get("sglang"), dict) else {}
    if not isinstance(sglang_args, dict):
        sglang_args = {}
    if not isinstance(sglang_runtime, dict):
        sglang_runtime = {}

    machine_rows = [
        {"item": "machine / cloud instance", "value": env_value(run_env, "cloud", "instance_type") or "not detected", "why it matters": "Tells us whether this is a small EC2 GPU box or a Grace Hopper-class machine."},
        {"item": "host RAM total", "value": env_value(run_env, "host_memory", "total_gib"), "why it matters": "Physical CPU memory available on the machine."},
        {"item": "host RAM available at capture", "value": env_value(run_env, "host_memory", "available_gib"), "why it matters": "Shows whether the machine itself was near RAM exhaustion."},
        {"item": "CPU", "value": env_value(run_env, "cpu", "Model name"), "why it matters": "Host-side KV movement and Python/SGLang control work run through this system."},
        {"item": "CPU cores", "value": env_value(run_env, "cpu", "CPU(s)"), "why it matters": "Useful context for scheduler/control-path overhead."},
    ]
    gpu_rows = [
        {"item": "GPU", "value": str(gpu.get("name") or ""), "why it matters": "The fast memory tier used by SGLang for active KV."},
        {"item": "GPU memory", "value": mib_to_gib(gpu.get("memory_total_mib")), "why it matters": "This is the device KV capacity before offload pressure starts."},
        {"item": "GPU memory type", "value": str(gpu.get("memory_type") or "unknown"), "why it matters": "A10G uses GDDR6; GH200/H100-class systems use HBM-class GPU memory."},
        {"item": "CUDA version", "value": env_value(run_env, "gpu", "cuda_version_from_nvidia_smi"), "why it matters": "CUDA/runtime version can affect copy and kernel behavior."},
        {"item": "driver version", "value": str(gpu.get("driver_version") or ""), "why it matters": "GPU software stack version used in the run."},
    ]
    model_rows = [
        {"item": "model", "value": env_value(run_env, "model") or str(run_config.get("MODEL") or ""), "why it matters": "Model size changes KV size and memory pressure."},
        {"item": "SGLang package", "value": env_value(run_env, "software", "sglang"), "why it matters": "Hooks and cache internals are version-sensitive."},
        {"item": "PyTorch package", "value": env_value(run_env, "software", "torch"), "why it matters": "Runtime and CUDA event behavior depends on the torch stack."},
        {"item": "dtype", "value": str(sglang_args.get("dtype") or ""), "why it matters": "Model dtype affects GPU memory use."},
        {"item": "KV cache dtype", "value": str(sglang_args.get("kv_cache_dtype") or ""), "why it matters": "KV dtype affects KV cache capacity and transfer size."},
        {"item": "context length", "value": str(sglang_runtime.get("context_len") or sglang_args.get("context_length") or ""), "why it matters": "Upper bound on prompt/history length in this run."},
        {"item": "tensor parallel size", "value": str(sglang_args.get("tp_size") or ""), "why it matters": "This run is usually single-GPU TP=1 on g5.2xlarge."},
    ]
    hicache_rows = [
        {"item": "HiCache enabled", "value": str(sglang_args.get("enable_hierarchical_cache") or ""), "why it matters": "Must be enabled for host-side KV caching."},
        {"item": "configured HiCache host KV shelf", "value": f"{sglang_args.get('hicache_size') or run_config.get('HICACHE_SIZE_GB') or ''} GB", "why it matters": "This is the host KV cache allocation, not the full physical RAM."},
        {"item": "physical host RAM", "value": env_value(run_env, "host_memory", "total_gib"), "why it matters": "The full machine RAM may be much larger than the HiCache shelf SGLang is allowed to use."},
        {"item": "HiCache backend", "value": str(sglang_args.get("hicache_io_backend") or ""), "why it matters": "Shows which host-cache movement backend SGLang used."},
        {"item": "HiCache write policy", "value": str(sglang_args.get("hicache_write_policy") or ""), "why it matters": "Controls how KV is written from GPU-side cache into host cache."},
        {"item": "HiCache memory layout", "value": str(sglang_args.get("hicache_mem_layout") or ""), "why it matters": "Relevant to how SGLang stores host-side KV."},
        {"item": "radix eviction policy", "value": str(sglang_args.get("radix_eviction_policy") or ""), "why it matters": "Explains why pressure tends to evict older/unprotected KV."},
        {"item": "max total tokens", "value": str(run_config.get("MAX_TOTAL_TOKENS") or sglang_args.get("max_total_tokens") or ""), "why it matters": "Caps how much token/KV capacity SGLang exposes to the run."},
        {"item": "mem fraction static", "value": str(run_config.get("MEM_FRACTION_STATIC") or sglang_args.get("mem_fraction_static") or ""), "why it matters": "Controls how much GPU memory SGLang reserves for static pools."},
    ]
    knob_rows = [
        {"item": "workload source", "value": str(run_config.get("WORKLOAD_SOURCE") or ""), "why it matters": "Synthetic is easier to reason about; real uses AgentBench/SWE-bench traces."},
        {"item": "modes", "value": str(run_config.get("MODES") or ""), "why it matters": "Shows whether this report includes no-prefetch, direct-prefetch, or both."},
        {"item": "filler list", "value": str(run_config.get("FILLER_LIST") or ""), "why it matters": "Filler requests create cache pressure during the tool-wait window."},
        {"item": "tool wait list", "value": str(run_config.get("TOOL_WAIT_LIST_MS") or ""), "why it matters": "How long the agent pauses before replay/resume arrives."},
        {"item": "target prompt tokens", "value": str(run_config.get("SYNTHETIC_PROMPT_TOKENS") or run_config.get("TARGET_PROMPT_TOKENS") or ""), "why it matters": "Approximate size of the KV we want to preserve or reload."},
        {"item": "filler prompt tokens", "value": str(run_config.get("FILLER_PROMPT_TOKENS") or ""), "why it matters": "Bigger filler prompts create more KV pressure."},
        {"item": "request concurrency", "value": str(run_config.get("REQUEST_CONCURRENCY") or ""), "why it matters": "Higher concurrency can increase scheduler and memory pressure."},
        {"item": "max prompt pairs", "value": str(run_config.get("MAX_PAIRS") or ""), "why it matters": "Controls the number of target first/replay pairs."},
    ]
    return {
        "machine": machine_rows,
        "gpu": gpu_rows,
        "model": model_rows,
        "hicache": hicache_rows,
        "knobs": knob_rows,
    }


def environment_html(run_env: dict[str, Any]) -> str:
    if not run_env:
        return """
        <h3>Machine And Runtime Configuration</h3>
        <p class="note">No <code>run_environment.json</code> file was attached to this report. Rebuild with <code>scripts/run_master_report.sh</code> to collect machine, GPU, model, and HiCache details automatically.</p>
        """
    rows = runtime_config_rows(run_env)
    return "\n".join(
        [
            "<h3>Machine And Runtime Configuration</h3>",
            "<p class=\"note\">Important: host memory in this report has two meanings. Physical host RAM is the machine's CPU memory. The HiCache host KV shelf is the smaller amount SGLang was configured to use for host-side KV cache.</p>",
            "<h4>Machine</h4>",
            table_html(rows["machine"], ["item", "value", "why it matters"]),
            "<h4>GPU</h4>",
            table_html(rows["gpu"], ["item", "value", "why it matters"]),
            "<h4>Model And Runtime</h4>",
            table_html(rows["model"], ["item", "value", "why it matters"]),
            "<h4>SGLang HiCache And Memory Knobs</h4>",
            table_html(rows["hicache"], ["item", "value", "why it matters"]),
            "<h4>Experiment Knobs</h4>",
            table_html(rows["knobs"], ["item", "value", "why it matters"]),
        ]
    )


def manager_setup_html(run_env: dict[str, Any] | None = None) -> str:
    setup_rows = [
        {"part": "1. Real prompt pair", "simple meaning": "Use two adjacent model turns from real AgentBench/DeepAgents traces."},
        {"part": "2. Controlled wait", "simple meaning": "Replay Turn A, then impose a chosen tool-wait window such as 100 ms or 500 ms."},
        {"part": "3. Cache pressure", "simple meaning": "Send filler requests during the wait to make KV residency harder."},
        {"part": "4. Direct KV hint", "simple meaning": "In prefetch modes, issue the marked direct-load request during the wait."},
        {"part": "5. Resume", "simple meaning": "Send Turn B at the scheduled resume time, even if the hint is still running."},
    ]
    return f"""
    <div class="setup-diagram">{setup_diagram_svg()}</div>
    <p>This milestone keeps real SWE-bench/DeepAgents prompt content, but controls the timing. That gives us a cleaner hardware-style experiment: same kind of agent text, known wait windows, known cache pressure, and direct SGLang KV-hook attempts.</p>
    {table_html(setup_rows, ["part", "simple meaning"])}
    {environment_html(run_env or {})}
    """


def metric_cards_html(mode_rows: list[dict[str, Any]]) -> str:
    by_mode = {str(row.get("mode") or ""): row for row in mode_rows}
    no_prefetch = by_mode.get("no_prefetch", {})
    direct = by_mode.get("direct_prefetch", {})
    cards = [
        ("controlled gaps", sum(int(row.get("controlled_gaps") or 0) for row in mode_rows)),
        ("no-prefetch avg TTFT", f"{no_prefetch.get('avg_resume_ttft_ms', '')} ms"),
        ("direct-prefetch avg TTFT", f"{direct.get('avg_resume_ttft_ms', '')} ms"),
        ("direct late prefetches", direct.get("late_prefetches", "")),
        ("direct H2D gaps", direct.get("hint_h2d_gaps", "")),
        ("suspected replay wait/recompute", sum(int(row.get("replay_recompute_or_wait_suspected_gaps") or 0) for row in mode_rows)),
        ("likely cache hit/resident", sum(int(row.get("likely_cache_hit_or_resident_gaps") or 0) for row in mode_rows)),
    ]
    if "oracle_prefetch" in by_mode:
        oracle = by_mode["oracle_prefetch"]
        cards.extend(
            [
                ("oracle-prefetch avg TTFT", f"{oracle.get('avg_resume_ttft_ms', '')} ms"),
                ("oracle late prefetches", oracle.get("late_prefetches", "")),
                ("oracle H2D gaps", oracle.get("hint_h2d_gaps", "")),
            ]
        )
    return "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{html.escape(str(label))}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
        for label, value in cards
    ) + "</div>"


def timeline_model_table_html() -> str:
    rows = [
        ("Initial model turn", "#2563eb", "First model request before tool wait"),
        ("Tool wait", "#d1d5db", "Agent/tool pause where prefetch could happen"),
        ("Direct prefetch attempt", "#a855f7", "Our hint/direct-load path tried to prepare KV"),
        ("Hint-side KV HtoD", "#16a34a", "Prefetch path actually loaded KV from host to GPU"),
        ("Replay-side KV HtoD", "#06b6d4", "Replay itself loaded KV from host to GPU"),
        ("Replay recompute", "#db2777", "Replay rebuilt missing old KV/prefix tokens"),
        ("Normal replay prefill", "#eab308", "Small remaining prefill/new-token processing before first token"),
        ("Replay decode", "#ef4444", "Generation after first token"),
        ("Replay due", "#111827", "Deadline when KV should ideally already be ready"),
    ]
    out = [
        '<div class="table-wrap"><table>',
        "<thead><tr><th>Timeline Element</th><th>Color Strip</th><th>Meaning</th></tr></thead><tbody>",
    ]
    for element, color, meaning in rows:
        if element == "Replay due":
            swatch = (
                '<span style="display:inline-block;width:76px;height:18px;background:#f8fafc;'
                'border:1px solid #cbd5e1;vertical-align:middle;position:relative;">'
                f'<span style="display:block;width:5px;height:18px;background:{color};margin:0 auto;"></span>'
                "</span>"
            )
        else:
            swatch = (
                f'<span style="display:inline-block;width:76px;height:18px;background:{color};'
                'border:1px solid rgba(15,23,42,0.22);border-radius:3px;vertical-align:middle;"></span>'
            )
        out.append(
            "<tr>"
            f"<td>{html.escape(element)}</td>"
            f"<td>{swatch}</td>"
            f"<td>{html.escape(meaning)}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


def h2d_finish_margin(row: dict[str, Any]) -> float | None:
    due = as_float(row.get("tool_gap_end_ms"))
    h2d_end = as_float(row.get("replay_kv_h2d_end_ms"))
    if due is None or h2d_end is None:
        return None
    return round(due - h2d_end, 3)


def relative_to_due(value: Any, due: float) -> float | str:
    numeric = as_float(value)
    if numeric is None:
        return ""
    return round(numeric - due, 3)


def replay_h2d_readiness_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(gaps):
        if str(row.get("mode") or "") != "no_prefetch":
            continue
        due = as_float(row.get("tool_gap_end_ms"))
        h2d_start = as_float(row.get("replay_kv_h2d_start_ms"))
        h2d_end = as_float(row.get("replay_kv_h2d_end_ms"))
        if due is None or h2d_start is None or h2d_end is None or not has_events(row.get("replay_kv_h2d_events")):
            continue
        resume_start = as_float(row.get("resume_start_ms"))
        resume_submitted = as_float(row.get("resume_submitted_ms"))
        event_duration = as_float(row.get("replay_kv_h2d_duration_ms"))
        start_delay = round(h2d_start - due, 3)
        wall_window = round(h2d_end - h2d_start, 3)
        finish_lateness = round(h2d_end - due, 3)
        finish_margin = round(due - h2d_end, 3)
        request_start_delay = round(resume_start - due, 3) if resume_start is not None else ""
        after_resume_start = round(h2d_start - resume_start, 3) if resume_start is not None else ""
        h2d_end_after_request_start = round(h2d_end - resume_start, 3) if resume_start is not None else ""
        filler_count = case_fillers(row)
        rows.append(
            {
                "order": len(rows),
                "session_id": row.get("session_id", ""),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", idx),
                "fillers": filler_count,
                "tool_gap_ms": row.get("tool_gap_ms", ""),
                "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                "replay_due_to_client_submit_ms": round(resume_submitted - due, 3) if resume_submitted is not None else "",
                "replay_due_to_request_start_ms": request_start_delay,
                "client_submit_to_request_start_ms": (
                    round(resume_start - resume_submitted, 3) if resume_start is not None and resume_submitted is not None else ""
                ),
                "replay_due_to_sglang_receive_ms": relative_to_due(row.get("replay_sglang_receive_start_ms"), due),
                "replay_due_to_scheduler_queue_ms": relative_to_due(row.get("replay_scheduler_queue_enter_start_ms"), due),
                "replay_due_to_scheduler_admit_ms": relative_to_due(row.get("replay_scheduler_admit_start_ms"), due),
                "scheduler_queue_waiting_len": row.get("replay_scheduler_queue_waiting_len", ""),
                "scheduler_queue_running_len": row.get("replay_scheduler_queue_running_len", ""),
                "scheduler_admit_running_batch_requests": row.get("replay_scheduler_admit_running_batch_requests", ""),
                "scheduler_admit_running_batch_extend_tokens": row.get("replay_scheduler_admit_running_batch_extend_tokens", ""),
                "replay_due_to_h2d_start_ms": start_delay,
                "scheduler_queue_to_admit_ms": row.get("replay_scheduler_queue_to_admit_ms", ""),
                "scheduler_admit_to_h2d_ms": row.get("replay_scheduler_admit_to_h2d_ms", ""),
                "request_start_to_h2d_start_ms": after_resume_start,
                "h2d_visible_wall_window_ms": wall_window,
                "h2d_event_duration_sum_ms": event_duration if event_duration is not None else "",
                "request_start_to_h2d_end_ms": h2d_end_after_request_start,
                "replay_due_to_h2d_end_ms": finish_lateness,
                "h2d_finish_margin_ms": finish_margin,
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "replay_h2d_tokens": row.get("lifecycle_replay_h2d_tokens") or row.get("replay_host_load_tokens", ""),
                "final_path": row.get("final_path", ""),
                "simple_meaning": (
                    f"H2D started {start_delay:.1f} ms after replay was due and finished "
                    f"{finish_lateness:.1f} ms after replay was due."
                    if finish_lateness >= 0
                    else f"H2D finished {abs(finish_lateness):.1f} ms before replay was due."
                ),
            }
        )
    return rows


def replay_queue_timing_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(gaps):
        if str(row.get("mode") or "") != "no_prefetch":
            continue
        due = as_float(row.get("tool_gap_end_ms"))
        if due is None:
            continue
        resume_submitted = as_float(row.get("resume_submitted_ms"))
        resume_start = as_float(row.get("resume_start_ms"))
        h2d_start = as_float(row.get("replay_kv_h2d_start_ms"))
        h2d_end = as_float(row.get("replay_kv_h2d_end_ms"))
        h2d_finish_margin_value = round(due - h2d_end, 3) if h2d_end is not None else ""
        h2d_start_delay = round(h2d_start - due, 3) if h2d_start is not None else ""
        h2d_end_delay = round(h2d_end - due, 3) if h2d_end is not None else ""
        request_start_to_h2d_start = round(h2d_start - resume_start, 3) if h2d_start is not None and resume_start is not None else ""
        request_start_to_h2d_end = round(h2d_end - resume_start, 3) if h2d_end is not None and resume_start is not None else ""
        simple_bits: list[str] = []
        if resume_submitted is not None:
            simple_bits.append(f"client submitted replay {resume_submitted - due:.1f} ms after due")
        if resume_start is not None:
            simple_bits.append(f"client request call started {resume_start - due:.1f} ms after due")
        queue_delay = row.get("replay_scheduler_queue_to_admit_ms", "")
        if queue_delay not in ("", None):
            simple_bits.append(f"SGLang queue-to-admit was {queue_delay} ms")
        if h2d_end is not None:
            simple_bits.append(f"H2D finished {h2d_end - due:.1f} ms after due")
        elif has_events(row.get("replay_kv_h2d_events")):
            simple_bits.append("H2D was observed but timing was incomplete")
        else:
            simple_bits.append("no replay-side H2D was observed")
        rows.append(
            {
                "order": len(rows),
                "session_id": row.get("session_id", ""),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", idx),
                "fillers": case_fillers(row),
                "tool_gap_ms": row.get("tool_gap_ms", ""),
                "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                "replay_due_to_client_submit_ms": round(resume_submitted - due, 3) if resume_submitted is not None else "",
                "replay_due_to_request_start_ms": round(resume_start - due, 3) if resume_start is not None else "",
                "client_submit_to_request_start_ms": (
                    round(resume_start - resume_submitted, 3) if resume_start is not None and resume_submitted is not None else ""
                ),
                "replay_due_to_sglang_receive_ms": relative_to_due(row.get("replay_sglang_receive_start_ms"), due),
                "replay_due_to_scheduler_queue_ms": relative_to_due(row.get("replay_scheduler_queue_enter_start_ms"), due),
                "replay_due_to_scheduler_admit_ms": relative_to_due(row.get("replay_scheduler_admit_start_ms"), due),
                "scheduler_queue_waiting_len": row.get("replay_scheduler_queue_waiting_len", ""),
                "scheduler_queue_running_len": row.get("replay_scheduler_queue_running_len", ""),
                "scheduler_admit_running_batch_requests": row.get("replay_scheduler_admit_running_batch_requests", ""),
                "scheduler_admit_running_batch_extend_tokens": row.get("replay_scheduler_admit_running_batch_extend_tokens", ""),
                "scheduler_queue_to_admit_ms": row.get("replay_scheduler_queue_to_admit_ms", ""),
                "scheduler_admit_to_h2d_ms": row.get("replay_scheduler_admit_to_h2d_ms", ""),
                "replay_due_to_h2d_start_ms": h2d_start_delay,
                "request_start_to_h2d_start_ms": request_start_to_h2d_start,
                "replay_due_to_h2d_end_ms": h2d_end_delay,
                "h2d_finish_margin_ms": h2d_finish_margin_value,
                "h2d_visible_wall_window_ms": round(h2d_end - h2d_start, 3) if h2d_start is not None and h2d_end is not None else "",
                "h2d_event_duration_sum_ms": row.get("replay_kv_h2d_duration_ms", ""),
                "request_start_to_h2d_end_ms": request_start_to_h2d_end,
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "replay_h2d_tokens": row.get("lifecycle_replay_h2d_tokens") or row.get("replay_host_load_tokens", ""),
                "final_path": row.get("final_path", ""),
                "simple_meaning": "; ".join(simple_bits),
            }
        )
    return rows


def replay_h2d_readiness_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    margins = [float(row["h2d_finish_margin_ms"]) for row in rows if row.get("h2d_finish_margin_ms") not in ("", None)]
    client_submit_delays = [float(row["replay_due_to_client_submit_ms"]) for row in rows if row.get("replay_due_to_client_submit_ms") not in ("", None)]
    request_start_delays = [float(row["replay_due_to_request_start_ms"]) for row in rows if row.get("replay_due_to_request_start_ms") not in ("", None)]
    sglang_receive_delays = [float(row["replay_due_to_sglang_receive_ms"]) for row in rows if row.get("replay_due_to_sglang_receive_ms") not in ("", None)]
    scheduler_queue_delays = [float(row["replay_due_to_scheduler_queue_ms"]) for row in rows if row.get("replay_due_to_scheduler_queue_ms") not in ("", None)]
    scheduler_admit_delays = [float(row["replay_due_to_scheduler_admit_ms"]) for row in rows if row.get("replay_due_to_scheduler_admit_ms") not in ("", None)]
    start_delays = [float(row["replay_due_to_h2d_start_ms"]) for row in rows if row.get("replay_due_to_h2d_start_ms") not in ("", None)]
    request_to_h2d_start = [float(row["request_start_to_h2d_start_ms"]) for row in rows if row.get("request_start_to_h2d_start_ms") not in ("", None)]
    queue_to_admit = [float(row["scheduler_queue_to_admit_ms"]) for row in rows if row.get("scheduler_queue_to_admit_ms") not in ("", None)]
    admit_to_h2d = [float(row["scheduler_admit_to_h2d_ms"]) for row in rows if row.get("scheduler_admit_to_h2d_ms") not in ("", None)]
    wall_windows = [float(row["h2d_visible_wall_window_ms"]) for row in rows if row.get("h2d_visible_wall_window_ms") not in ("", None)]
    event_durations = [float(row["h2d_event_duration_sum_ms"]) for row in rows if row.get("h2d_event_duration_sum_ms") not in ("", None)]
    late = [value for value in margins if value < 0]
    early = [value for value in margins if value >= 0]
    return [
        {
            "no_prefetch_replay_h2d_gaps": len(rows),
            "h2d_finished_after_replay_due": len(late),
            "late_pct": round(len(late) * 100.0 / len(rows), 2) if rows else "",
            "median_h2d_finish_margin_ms": round(median(margins), 3) if margins else "",
            "worst_h2d_lateness_ms": round(abs(min(late)), 3) if late else "",
            "best_early_margin_ms": round(max(early), 3) if early else "",
            "avg_due_to_client_submit_ms": avg(client_submit_delays),
            "avg_due_to_request_start_ms": avg(request_start_delays),
            "avg_due_to_sglang_receive_ms": avg(sglang_receive_delays),
            "avg_due_to_scheduler_queue_ms": avg(scheduler_queue_delays),
            "avg_due_to_scheduler_admit_ms": avg(scheduler_admit_delays),
            "avg_due_to_h2d_start_ms": avg(start_delays),
            "avg_request_start_to_h2d_start_ms": avg(request_to_h2d_start),
            "avg_scheduler_queue_to_admit_ms": avg(queue_to_admit),
            "avg_scheduler_admit_to_h2d_ms": avg(admit_to_h2d),
            "avg_h2d_visible_wall_window_ms": avg(wall_windows),
            "avg_h2d_event_duration_sum_ms": avg(event_durations),
        }
    ]


def replay_h2d_readiness_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [
        ("> +500 ms ready early", lambda value: value > 500),
        ("+100 to +500 ms ready early", lambda value: 100 < value <= 500),
        ("0 to +100 ms ready early", lambda value: 0 <= value <= 100),
        ("0 to -100 ms late", lambda value: -100 <= value < 0),
        ("-100 to -500 ms late", lambda value: -500 <= value < -100),
        ("< -500 ms late", lambda value: value < -500),
    ]
    total = len(rows)
    output: list[dict[str, Any]] = []
    for label, predicate in buckets:
        count = sum(1 for row in rows if predicate(float(row["h2d_finish_margin_ms"])))
        output.append(
            {
                "bucket": label,
                "replay_h2d_gaps": count,
                "pct": round(count * 100.0 / total, 2) if total else "",
            }
        )
    return output


def stage_confidence(value: Any, source: str) -> str:
    if as_float(value) is None:
        return "missing"
    return source


def positive_delta(start: Any, end: Any) -> float | str:
    start_value = as_float(start)
    end_value = as_float(end)
    if start_value is None or end_value is None:
        return ""
    return round(max(0.0, end_value - start_value), 3)


def first_token_ms(row: dict[str, Any]) -> float | None:
    start = as_float(row.get("resume_start_ms"))
    ttft = as_float(row.get("resume_ttft_ms"))
    if start is None or ttft is None:
        return None
    return start + ttft


def replay_first_cache_event_ms(row: dict[str, Any]) -> float | None:
    replay_start = as_float(row.get("resume_start_ms"))
    first_delay = as_float(row.get("replay_first_cache_event_delay_ms"))
    if replay_start is None or first_delay is None:
        return None
    return replay_start + first_delay


def replay_delay_segment_values(row: dict[str, Any]) -> dict[str, float]:
    due = as_float(row.get("tool_gap_end_ms"))
    submitted = as_float(row.get("resume_submitted_ms"))
    request_start = as_float(row.get("resume_start_ms"))
    sglang_receive = as_float(row.get("replay_sglang_receive_start_ms"))
    scheduler_queue = as_float(row.get("replay_scheduler_queue_enter_start_ms"))
    scheduler_admit = as_float(row.get("replay_scheduler_admit_start_ms"))
    cache_first = replay_first_cache_event_ms(row)
    h2d_start = as_float(row.get("replay_kv_h2d_start_ms"))
    h2d_end = as_float(row.get("replay_kv_h2d_end_ms"))
    first_token = first_token_ms(row)

    segments: dict[str, float] = {}

    def add(name: str, start: Any, end: Any) -> None:
        value = positive_delta(start, end)
        if isinstance(value, (int, float)) and value > 0:
            segments[name] = float(value)

    add("due_to_client_submit", due, submitted)
    add("client_submit_to_request_start", submitted, request_start)
    add("request_start_to_sglang_receive", request_start, sglang_receive)
    add("sglang_receive_to_scheduler_queue", sglang_receive, scheduler_queue)
    add("scheduler_queue_wait", scheduler_queue, scheduler_admit)
    add("scheduler_admit_to_cache_lookup", scheduler_admit, cache_first)
    add("cache_lookup_to_h2d_start", cache_first, h2d_start)
    add("scheduler_admit_to_h2d_start", scheduler_admit, h2d_start)
    add("h2d_copy", h2d_start, h2d_end)
    add("h2d_end_to_first_token", h2d_end, first_token)
    if "h2d_end_to_first_token" not in segments:
        add("request_start_to_first_token", request_start, first_token)
    return segments


def delay_source_name(segment: str) -> str:
    names = {
        "due_to_client_submit": "client submitted late",
        "client_submit_to_request_start": "client/workload dispatch dominated",
        "request_start_to_sglang_receive": "request transport/server receive dominated",
        "sglang_receive_to_scheduler_queue": "SGLang receive-to-queue dominated",
        "scheduler_queue_wait": "scheduler queue dominated",
        "scheduler_admit_to_cache_lookup": "scheduler/cache lookup dominated",
        "cache_lookup_to_h2d_start": "cache/load-back path dominated",
        "scheduler_admit_to_h2d_start": "scheduler admit-to-H2D dominated",
        "h2d_copy": "H2D copy dominated",
        "h2d_end_to_first_token": "post-H2D prefill/decode dominated",
        "request_start_to_first_token": "replay TTFT dominated",
    }
    return names.get(segment, segment.replace("_", " "))


def copy_verdict_for_delay_row(
    row: dict[str, Any],
    segments: dict[str, float],
    contention_by_row: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    label = str(row.get("timeline_label") or "")
    contention = contention_by_row.get(label, {})
    if str(contention.get("verdict") or "") == "blocked behind other H2D":
        return (
            "copy blocked behind other H2D",
            "Other exact H2D events were visible between replay due and this row's own H2D start.",
        )

    h2d_start_delay = None
    due = as_float(row.get("tool_gap_end_ms"))
    h2d_start = as_float(row.get("replay_kv_h2d_start_ms"))
    h2d_end = as_float(row.get("replay_kv_h2d_end_ms"))
    if due is not None and h2d_start is not None:
        h2d_start_delay = h2d_start - due
    h2d_duration = segments.get("h2d_copy", 0.0)

    if h2d_start is None or h2d_end is None or not has_events(row.get("replay_kv_h2d_events")):
        new_prefill = as_float(row.get("replay_new_prefill_tokens_est")) or 0.0
        if new_prefill >= 128:
            return (
                "no replay H2D; recompute/prefill path",
                "Replay did not show host-to-device KV movement, and cache counters suggest missing prefix work was rebuilt.",
            )
        return (
            "no visible replay H2D",
            "Replay did not show host-to-device KV movement in this row.",
        )
    if h2d_start_delay is not None and h2d_start_delay > 1000 and h2d_duration < max(1000.0, h2d_start_delay * 0.1):
        return (
            "copy issued late, copy was fast",
            "The target H2D request started long after replay due, but the visible copy window itself was short.",
        )
    if h2d_duration >= max(1000.0, (h2d_start_delay or 0.0) * 0.5):
        return (
            "copy issued on time or near-time, copy was slow",
            "The visible H2D copy window is a large part of the replay delay.",
        )
    return (
        "target H2D visible",
        "Replay-side host-to-device KV movement was visible and attributable.",
    )


def replay_delay_breakdown_rows(
    gaps: list[dict[str, Any]],
    h2d_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contention_summary = h2d_contention_summary_rows(select_h2d_contention_targets(gaps, max_targets=len(gaps)), h2d_events)
    contention_by_row = {str(row.get("target_row") or ""): row for row in contention_summary}
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(gaps):
        due = as_float(row.get("tool_gap_end_ms"))
        if due is None:
            continue
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        submitted = as_float(row.get("resume_submitted_ms"))
        request_start = as_float(row.get("resume_start_ms"))
        sglang_receive = as_float(row.get("replay_sglang_receive_start_ms"))
        scheduler_queue = as_float(row.get("replay_scheduler_queue_enter_start_ms"))
        scheduler_admit = as_float(row.get("replay_scheduler_admit_start_ms"))
        cache_first = replay_first_cache_event_ms(row)
        h2d_start = as_float(row.get("replay_kv_h2d_start_ms"))
        h2d_end = as_float(row.get("replay_kv_h2d_end_ms"))
        token_time = first_token_ms(row)
        segments = replay_delay_segment_values(row)
        dominant_segment = max(segments, key=segments.get) if segments else ""
        dominant_ms = round(segments[dominant_segment], 3) if dominant_segment else ""
        copy_verdict, copy_explanation = copy_verdict_for_delay_row(row, segments, contention_by_row)
        main_source = delay_source_name(dominant_segment) if dominant_segment else "unknown / missing instrumentation"

        exact_stages = [
            stage_confidence(sglang_receive, "exact"),
            stage_confidence(scheduler_queue, "exact"),
            stage_confidence(scheduler_admit, "exact"),
            stage_confidence(h2d_start, "exact"),
            stage_confidence(h2d_end, "exact"),
        ]
        measured_stages = [
            stage_confidence(submitted, "measured"),
            stage_confidence(request_start, "measured"),
            stage_confidence(token_time, "measured"),
        ]
        inferred_stages = [stage_confidence(cache_first, "inferred")]
        missing_count = exact_stages.count("missing") + measured_stages.count("missing") + inferred_stages.count("missing")
        if missing_count <= 1 and h2d_start is not None and h2d_end is not None:
            confidence = "high"
        elif h2d_start is not None or scheduler_admit is not None:
            confidence = "medium"
        else:
            confidence = "low"

        rows.append(
            {
                "row": label,
                "session_id": row.get("session_id", ""),
                "case_id": row.get("case_id", ""),
                "mode": row.get("mode", ""),
                "fillers": case_fillers(row),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", ""),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                "copy_verdict": copy_verdict,
                "main_delay_source": main_source,
                "dominant_delay_stage": dominant_segment,
                "dominant_delay_ms": dominant_ms,
                "copy_explanation": copy_explanation,
                "delay_confidence": confidence,
                "replay_due_ms": round(due, 3),
                "client_submit_relative_ms": round(submitted - due, 3) if submitted is not None else "",
                "request_start_relative_ms": round(request_start - due, 3) if request_start is not None else "",
                "sglang_receive_relative_ms": round(sglang_receive - due, 3) if sglang_receive is not None else "",
                "scheduler_queue_relative_ms": round(scheduler_queue - due, 3) if scheduler_queue is not None else "",
                "scheduler_admit_relative_ms": round(scheduler_admit - due, 3) if scheduler_admit is not None else "",
                "cache_first_event_relative_ms": round(cache_first - due, 3) if cache_first is not None else "",
                "h2d_start_relative_ms": round(h2d_start - due, 3) if h2d_start is not None else "",
                "h2d_end_relative_ms": round(h2d_end - due, 3) if h2d_end is not None else "",
                "first_token_relative_ms": round(token_time - due, 3) if token_time is not None else "",
                "due_to_client_submit_ms": segments.get("due_to_client_submit", ""),
                "client_submit_to_request_start_ms": segments.get("client_submit_to_request_start", ""),
                "request_start_to_sglang_receive_ms": segments.get("request_start_to_sglang_receive", ""),
                "sglang_receive_to_scheduler_queue_ms": segments.get("sglang_receive_to_scheduler_queue", ""),
                "scheduler_queue_wait_ms": segments.get("scheduler_queue_wait", ""),
                "scheduler_admit_to_cache_lookup_ms": segments.get("scheduler_admit_to_cache_lookup", ""),
                "cache_lookup_to_h2d_start_ms": segments.get("cache_lookup_to_h2d_start", ""),
                "scheduler_admit_to_h2d_start_ms": segments.get("scheduler_admit_to_h2d_start", ""),
                "h2d_duration_ms": segments.get("h2d_copy", ""),
                "h2d_end_to_first_token_ms": segments.get("h2d_end_to_first_token", ""),
                "request_start_to_first_token_ms": segments.get("request_start_to_first_token", ""),
                "scheduler_queue_waiting_len": row.get("replay_scheduler_queue_waiting_len", ""),
                "scheduler_queue_running_len": row.get("replay_scheduler_queue_running_len", ""),
                "scheduler_admit_running_batch_requests": row.get("replay_scheduler_admit_running_batch_requests", ""),
                "scheduler_admit_running_batch_extend_tokens": row.get("replay_scheduler_admit_running_batch_extend_tokens", ""),
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "replay_h2d_tokens": row.get("lifecycle_replay_h2d_tokens") or row.get("replay_host_load_tokens", ""),
                "recompute_tokens_est": row.get("replay_new_prefill_tokens_est", ""),
                "stage_confidence_summary": (
                    "client timestamps=measured; SGLang receive/scheduler/H2D hooks=exact when present; "
                    "first cache event and recompute windows=inferred; missing stages are left blank"
                ),
                "simple_meaning": (
                    f"{label}: {copy_verdict}. Dominant observed delay: {main_source}"
                    + (f" ({display_ms(dominant_ms)})." if dominant_ms != "" else ".")
                ),
            }
        )
    return rows


def replay_delay_verdict_rows(delay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verdict_counts = Counter(str(row.get("copy_verdict") or "") for row in delay_rows)
    source_counts = Counter(str(row.get("main_delay_source") or "") for row in delay_rows)
    rows: list[dict[str, Any]] = []
    for verdict, count in sorted(verdict_counts.items()):
        if not verdict:
            continue
        rows.append(
            {
                "type": "copy verdict",
                "label": verdict,
                "rows": count,
                "pct": round(count * 100.0 / len(delay_rows), 2) if delay_rows else "",
            }
        )
    for source, count in sorted(source_counts.items()):
        if not source:
            continue
        rows.append(
            {
                "type": "main delay source",
                "label": source,
                "rows": count,
                "pct": round(count * 100.0 / len(delay_rows), 2) if delay_rows else "",
            }
        )
    return rows


def trace_base_by_case(trace_rows: list[dict[str, Any]]) -> dict[str, float]:
    bases: dict[str, float] = {}
    for row in trace_rows:
        case_id = str(row.get("ledger_case_id") or "")
        ts = as_float(row.get("ts_ns"))
        if not case_id or ts is None:
            continue
        seconds = ts / 1_000_000_000.0
        if case_id not in bases or seconds < bases[case_id]:
            bases[case_id] = seconds
    return bases


def trace_local_ms(row: dict[str, Any], bases: dict[str, float]) -> float | None:
    case_id = str(row.get("ledger_case_id") or "")
    base = bases.get(case_id)
    ts = as_float(row.get("ts_ns"))
    if base is None or ts is None:
        return None
    return (ts / 1_000_000_000.0 - base) * 1000.0


def trace_scheduler_metric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    context = row.get("kv_context")
    if isinstance(context, dict):
        state = context.get("scheduler_state")
        if isinstance(state, dict):
            for key in keys:
                value = as_float(state.get(key))
                if value is not None:
                    return value
    return None


def replay_delay_running_context_rows(
    gaps: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    h2d_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bases = trace_base_by_case(trace_rows)
    by_case: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        case_id = str(row.get("ledger_case_id") or "")
        if case_id:
            by_case[case_id].append(row)

    output: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        due = as_float(gap.get("tool_gap_end_ms"))
        if due is None:
            continue
        label = str(gap.get("timeline_label") or f"G{idx:02d}")
        case_id = str(gap.get("case_id") or "")
        target_h2d_start = as_float(gap.get("replay_kv_h2d_start_ms"))
        token_time = first_token_ms(gap)
        window_end = target_h2d_start or token_time
        if window_end is None:
            continue
        target_session = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        exact_h2d_in_window = h2d_events_overlap_window(h2d_events, due, window_end, case_id=case_id)
        other_h2d = [
            event for event in exact_h2d_in_window if str(event.get("ledger_session_id") or "") != target_session
        ]
        target_h2d = [
            event for event in exact_h2d_in_window if str(event.get("ledger_session_id") or "") == target_session
        ]

        counts: Counter[str] = Counter()
        max_waiting: float | str = ""
        max_running: float | str = ""
        max_extend_tokens: float | str = ""
        max_request_count: float | str = ""
        for event in by_case.get(case_id, []):
            t = trace_local_ms(event, bases)
            if t is None or not (due <= t <= window_end):
                continue
            name = str(event.get("event") or "")
            if name.startswith("kv_telemetry.scheduler") or name.startswith("scheduler."):
                counts["scheduler_events"] += 1
            if name == "kv_telemetry.prefill.end" or name.startswith("worker.forward"):
                counts["model_forward_events"] += 1
            if name.endswith("process_batch_result_prefill.end"):
                counts["prefill_batch_events"] += 1
            if name.endswith("process_batch_result_decode.end"):
                counts["decode_batch_events"] += 1
            if (
                name.startswith("kv_telemetry.cache")
                or name.startswith("hiradix.")
                or name.startswith("hicache.")
            ):
                counts["cache_hicache_events"] += 1
            if name.startswith("hostpool.load_to_device_per_layer"):
                counts["raw_hostpool_h2d_events"] += 1
            if name.startswith("m27.request"):
                counts["client_request_events"] += 1

            waiting = trace_scheduler_metric(event, "scheduler_waiting_queue_len", "waiting_queue_len")
            running = trace_scheduler_metric(event, "scheduler_running_batch_request_count", "scheduler_cur_batch_request_count")
            extend_tokens = trace_scheduler_metric(event, "scheduler_running_batch_extend_num_tokens", "scheduler_cur_batch_extend_num_tokens")
            request_count = trace_scheduler_metric(event, "request_count")
            if waiting is not None:
                max_waiting = max(float(max_waiting or 0), waiting)
            if running is not None:
                max_running = max(float(max_running or 0), running)
            if extend_tokens is not None:
                max_extend_tokens = max(float(max_extend_tokens or 0), extend_tokens)
            if request_count is not None:
                max_request_count = max(float(max_request_count or 0), request_count)

        output.append(
            {
                "row": label,
                "case_id": case_id,
                "mode": gap.get("mode", ""),
                "fillers": case_fillers(gap),
                "tool_wait_ms": gap.get("tool_gap_ms", ""),
                "delay_window_start": "replay due",
                "delay_window_end": "target H2D start" if target_h2d_start is not None else "first token",
                "delay_window_ms": round(window_end - due, 3),
                "scheduler_events": counts["scheduler_events"],
                "model_forward_events": counts["model_forward_events"],
                "prefill_batch_events": counts["prefill_batch_events"],
                "decode_batch_events": counts["decode_batch_events"],
                "cache_hicache_events": counts["cache_hicache_events"],
                "raw_hostpool_h2d_events": counts["raw_hostpool_h2d_events"],
                "client_request_events": counts["client_request_events"],
                "exact_target_h2d_events_before_target_start": len(target_h2d),
                "exact_other_h2d_events_before_target_start": len(other_h2d),
                "max_scheduler_waiting_queue_len": int(max_waiting) if max_waiting != "" else "",
                "max_running_batch_requests": int(max_running) if max_running != "" else "",
                "max_running_batch_extend_tokens": int(max_extend_tokens) if max_extend_tokens != "" else "",
                "max_request_count_seen": int(max_request_count) if max_request_count != "" else "",
                "simple_meaning": (
                    f"Between replay due and the target H2D start, trace saw "
                    f"{counts['model_forward_events']} model-forward events, "
                    f"{counts['cache_hicache_events']} cache/HiCache events, "
                    f"{counts['scheduler_events']} scheduler events, and "
                    f"{len(other_h2d)} exact other H2D events in the same case."
                ),
            }
        )
    return output


def request_stage_trace_rows(
    gaps: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bases = trace_base_by_case(trace_rows)
    by_case: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        if row.get("event") != "kv_telemetry.request_stage":
            continue
        case_id = str(row.get("ledger_case_id") or "")
        if case_id:
            by_case[case_id].append(row)

    output: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        due = as_float(gap.get("tool_gap_end_ms"))
        if due is None:
            continue
        label = str(gap.get("timeline_label") or f"G{idx:02d}")
        case_id = str(gap.get("case_id") or "")
        session = str(gap.get("session_id") or "")
        first_token = first_token_ms(gap)
        resume_end = as_float(gap.get("resume_end_ms"))
        h2d_end = as_float(gap.get("replay_kv_h2d_end_ms"))
        window_end_candidates = [value for value in (first_token, h2d_end, resume_end) if value is not None]
        window_end = max(window_end_candidates) if window_end_candidates else due
        for event in by_case.get(case_id, []):
            t = trace_local_ms(event, bases)
            if t is None or t < due or t > window_end:
                continue
            sessions = _agent_sessions_for_event(event)
            if session not in sessions:
                continue
            output.append(
                {
                    "row": label,
                    "session_id": session,
                    "case_id": case_id,
                    "mode": gap.get("mode", ""),
                    "fillers": case_fillers(gap),
                    "tool_wait_ms": gap.get("tool_gap_ms", ""),
                    "stage": event.get("stage") or event.get("category", ""),
                    "stage_group": event.get("stage_group", ""),
                    "stage_order": event.get("stage_order", ""),
                    "phase": event.get("phase", ""),
                    "method": event.get("method", ""),
                    "class": event.get("class", ""),
                    "call_id": event.get("call_id", ""),
                    "time_relative_to_replay_due_ms": round(t - due, 3),
                    "absolute_case_ms": round(t, 3),
                    "duration_ms": event.get("duration_ms", ""),
                    "request_count": event.get("request_count", ""),
                    "request_id": event.get("request_id", ""),
                    "scheduler_waiting_queue_len": event.get("scheduler_waiting_queue_len", ""),
                    "scheduler_running_queue_len": event.get("scheduler_running_queue_len", ""),
                    "scheduler_running_batch_request_count": event.get("scheduler_running_batch_request_count", ""),
                    "scheduler_running_batch_extend_num_tokens": event.get("scheduler_running_batch_extend_num_tokens", ""),
                    "scheduler_cur_batch_request_count": event.get("scheduler_cur_batch_request_count", ""),
                    "scheduler_cur_batch_extend_num_tokens": event.get("scheduler_cur_batch_extend_num_tokens", ""),
                    "host_index_count": event.get("host_index_count", ""),
                    "device_index_count": event.get("device_index_count", ""),
                    "exact_sglang_hook": event.get("exact_sglang_hook", ""),
                    "simple_meaning": (
                        f"{label}: exact SGLang hook {event.get('stage') or event.get('category', '')} "
                        f"{event.get('phase', '')} occurred {display_ms(t - due)} after replay due."
                    ),
                }
            )
    output.sort(
        key=lambda row: (
            str(row.get("case_id") or ""),
            str(row.get("row") or ""),
            as_float(row.get("time_relative_to_replay_due_ms")) or 0.0,
            as_float(row.get("stage_order")) or 0.0,
            str(row.get("phase") or ""),
        )
    )
    return output


def h2d_activity_during_delay_rows(
    gaps: list[dict[str, Any]],
    h2d_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        due = as_float(gap.get("tool_gap_end_ms"))
        if due is None:
            continue
        label = str(gap.get("timeline_label") or f"G{idx:02d}")
        case_id = str(gap.get("case_id") or "")
        target_session = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        h2d_start = as_float(gap.get("replay_kv_h2d_start_ms"))
        first_token = first_token_ms(gap)
        window_end = h2d_start or first_token
        if window_end is None:
            continue
        for order, event in enumerate(h2d_events_overlap_window(h2d_events, due, window_end, case_id=case_id)):
            owner = str(event.get("ledger_session_id") or "")
            start = as_float(event.get("aligned_h2d_start_ms"))
            end = as_float(event.get("aligned_h2d_end_ms"))
            output.append(
                {
                    "row": label,
                    "case_id": case_id,
                    "mode": gap.get("mode", ""),
                    "fillers": case_fillers(gap),
                    "tool_wait_ms": gap.get("tool_gap_ms", ""),
                    "event_order": order,
                    "owner_kind": "target replay H2D" if owner == target_session else "other H2D in same case",
                    "owner_session_id": event.get("session_id", ""),
                    "owner_row": event.get("row", ""),
                    "phase": event.get("phase", ""),
                    "source_event": event.get("source_event", ""),
                    "node_id": event.get("node_id", ""),
                    "layer_id": event.get("layer_id", ""),
                    "block_key": event.get("block_key", ""),
                    "token_or_index_count": event.get("token_or_index_count", ""),
                    "start_relative_to_replay_due_ms": round(start - due, 3) if start is not None else "",
                    "end_relative_to_replay_due_ms": round(end - due, 3) if end is not None else "",
                    "duration_ms": event.get("h2d_duration_ms", ""),
                    "confidence": event.get("confidence", ""),
                    "simple_meaning": (
                        f"{label}: {event.get('phase', 'H2D')} from "
                        f"{'this replay' if owner == target_session else 'another session'} "
                        f"overlapped the delay window before target KV readiness."
                    ),
                }
            )
    return output


def delay_verdicts_by_gap_rows(delay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in delay_rows:
        rows.append(
            {
                "row": row.get("row", ""),
                "session_id": row.get("session_id", ""),
                "case_id": row.get("case_id", ""),
                "mode": row.get("mode", ""),
                "fillers": row.get("fillers", ""),
                "tool_wait_ms": row.get("tool_wait_ms", ""),
                "verdict": row.get("copy_verdict", ""),
                "main_delay_source": row.get("main_delay_source", ""),
                "dominant_delay_stage": row.get("dominant_delay_stage", ""),
                "dominant_delay_ms": row.get("dominant_delay_ms", ""),
                "delay_confidence": row.get("delay_confidence", ""),
                "evidence": row.get("copy_explanation", ""),
                "simple_meaning": row.get("simple_meaning", ""),
            }
        )
    return rows


def build_replay_delay_waterfall_svg(rows: list[dict[str, Any]], max_rows: int = 12) -> str:
    selected = rows[:max_rows]
    if not selected:
        return "<p>No replay delay rows were available for the waterfall.</p>"
    width = 1480
    left = 230
    right = 40
    top = 64
    row_h = 56
    legend_h = 72
    height = top + row_h * len(selected) + legend_h
    plot_w = width - left - right
    stages = [
        ("due_to_client_submit_ms", "submit", "#94a3b8"),
        ("client_submit_to_request_start_ms", "client dispatch", "#2563eb"),
        ("request_start_to_sglang_receive_ms", "receive", "#f97316"),
        ("sglang_receive_to_scheduler_queue_ms", "queue enter", "#ca8a04"),
        ("scheduler_queue_wait_ms", "sched wait", "#db2777"),
        ("scheduler_admit_to_cache_lookup_ms", "cache lookup", "#8b5cf6"),
        ("cache_lookup_to_h2d_start_ms", "load path", "#7c3aed"),
        ("h2d_duration_ms", "H2D", "#06b6d4"),
        ("h2d_end_to_first_token_ms", "post-H2D", "#eab308"),
        ("request_start_to_first_token_ms", "TTFT", "#eab308"),
    ]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Replay delay waterfall">',
        '<text x="12" y="26" font-size="18" font-weight="800" fill="#0f172a">Replay delay waterfall</text>',
        '<text x="12" y="46" font-size="12" fill="#475569">Each row uses its own local scale. Segment widths show what consumed that row&apos;s observed delay.</text>',
    ]
    for idx, row in enumerate(selected):
        y = top + idx * row_h
        band = "#ffffff" if idx % 2 == 0 else "#eef4fb"
        parts.append(f'<rect x="0" y="{y - 10}" width="{width}" height="{row_h}" fill="{band}"/>')
        label = str(row.get("row") or f"G{idx:02d}")
        verdict = str(row.get("copy_verdict") or "")
        source = str(row.get("main_delay_source") or "")
        parts.append(f'<text x="12" y="{y + 8}" font-size="15" font-weight="800" fill="#0f172a">{html.escape(label)}</text>')
        parts.append(f'<text x="12" y="{y + 25}" font-size="10" font-weight="800" fill="#b91c1c">{html.escape(verdict)}</text>')
        parts.append(f'<text x="12" y="{y + 41}" font-size="10" fill="#475569">{html.escape(source)}</text>')
        values = [(key, label_text, color, as_float(row.get(key)) or 0.0) for key, label_text, color in stages]
        # If we have a decomposed post-H2D segment, hide the coarse TTFT fallback.
        if any(key == "h2d_end_to_first_token_ms" and value > 0 for key, _, _, value in values):
            values = [item for item in values if item[0] != "request_start_to_first_token_ms"]
        values = [item for item in values if item[3] > 0]
        total = sum(value for _, _, _, value in values) or 1.0
        x = left
        for key, label_text, color, value in values:
            w = max(3.0, value / total * plot_w)
            title = f"{label} | {label_text}: {display_ms(value)} | {row.get('simple_meaning', '')}"
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="22" rx="4" fill="{color}" opacity="0.88">'
                f'<title>{html.escape(title)}</title></rect>'
            )
            if w > 76:
                parts.append(
                    f'<text x="{x + w / 2:.1f}" y="{y + 15:.1f}" text-anchor="middle" font-size="10" '
                    f'fill="#ffffff" font-weight="800">{html.escape(label_text)} {html.escape(display_ms(value))}</text>'
                )
            x += w
        parts.append(
            f'<text x="{left + plot_w + 8}" y="{y + 15}" font-size="10" fill="#475569">{html.escape(display_ms(total))} total shown</text>'
        )
    legend_y = height - 44
    lx = left
    for _key, label_text, color in stages[:9]:
        parts.append(f'<rect x="{lx:.1f}" y="{legend_y:.1f}" width="14" height="14" rx="3" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{lx + 20:.1f}" y="{legend_y + 12:.1f}" font-size="11" fill="#334155">{html.escape(label_text)}</text>')
        lx += 132
    parts.append("</svg>")
    return "\n".join(parts)


def replay_delay_breakdown_html(
    gaps: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    h2d_events: list[dict[str, Any]],
    max_rows: int = 18,
) -> str:
    delay_rows = replay_delay_breakdown_rows(gaps, h2d_events)
    if not delay_rows:
        return """
        <p>No replay delay breakdown rows were available for this report.</p>
        <p class="note">This usually means the selected run did not include replay due timestamps or replay requests.</p>
        """
    shown = delay_rows[:max_rows]
    return f"""
    <p>This section explains the missing time between replay due and replay-side KV readiness. It separates client dispatch, SGLang receive, scheduler queue/admit, cache lookup/load-back, H2D copy, and post-H2D first-token work.</p>
    <p class="note">The waterfall uses a local scale per row so long waits are readable. Long proof tables are moved to <strong>Evidence Tables / Raw Proof</strong> at the bottom of the report.</p>
    <h3>Delay Waterfall Timeline</h3>
    <div class="setup-diagram">{build_replay_delay_waterfall_svg(shown, max_rows=max_rows)}</div>
    <p class="note">For exact per-row values, open the bottom evidence section and look at <strong>Replay Delay Verdicts</strong>, <strong>Replay Delay Stage Trace</strong>, <strong>Stage Duration Table</strong>, and <strong>H2D Activity During The Delay Window</strong>.</p>
    """


def exact_h2d_source_rank(row: dict[str, Any]) -> int:
    source = str(row.get("source_event") or "")
    if "hostpool.load_to_device_per_layer" in source:
        return 0
    if source == "hicache.load.end":
        return 1
    if "load_back" in source:
        return 2
    return 3


def h2d_event_phase(row: dict[str, Any]) -> str:
    phase = str(row.get("phase") or "").lower()
    if "hint" in phase or "prefetch" in phase:
        return "hint"
    return "replay"


def aligned_h2d_activity_events(
    gaps: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return lower-level H2D copy events aligned to the same clock as gap rows.

    The exact movement table is produced from SGLang trace events and can carry a
    different local offset from the summarized gap rows. We align each
    session/phase by matching the first exact H2D copy start to the summarized
    replay/direct H2D start for the same session.
    """
    gap_by_session: dict[str, dict[str, Any]] = {}
    for idx, gap in enumerate(gaps):
        copied = dict(gap)
        copied.setdefault("timeline_label", f"G{idx:02d}")
        for key in (str(copied.get("ledger_session_id") or ""), str(copied.get("session_id") or "")):
            if key:
                gap_by_session[key] = copied

    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in exact_rows:
        if str(row.get("direction") or "") != "host_to_device" and str(row.get("movement") or "") != "host_to_gpu_load":
            continue
        session_id = str(row.get("session_id") or "")
        if session_id not in gap_by_session:
            continue
        if as_float(row.get("copy_start_ms")) is None or as_float(row.get("copy_end_ms")) is None:
            continue
        grouped[(session_id, h2d_event_phase(row))].append(row)

    selected_rows: list[dict[str, Any]] = []
    for (session_id, phase), rows in grouped.items():
        best_rank = min(exact_h2d_source_rank(row) for row in rows)
        # Prefer the closest SGLang-visible copy path to real H2D work. This
        # avoids double-counting the high-level wrapper events around the same
        # lower-level hostpool copy activity.
        selected_rows.extend(row for row in rows if exact_h2d_source_rank(row) == best_rank)

    min_start_by_group: dict[tuple[str, str], float] = {}
    for row in selected_rows:
        key = (str(row.get("session_id") or ""), h2d_event_phase(row))
        start = as_float(row.get("copy_start_ms"))
        if start is None:
            continue
        if key not in min_start_by_group or start < min_start_by_group[key]:
            min_start_by_group[key] = start

    output: list[dict[str, Any]] = []
    for row in selected_rows:
        session_id = str(row.get("session_id") or "")
        phase = h2d_event_phase(row)
        gap = gap_by_session.get(session_id)
        if not gap:
            continue
        anchor_key = "direct_kv_h2d_start_ms" if phase == "hint" else "replay_kv_h2d_start_ms"
        anchor_start = as_float(gap.get(anchor_key))
        raw_group_start = min_start_by_group.get((session_id, phase))
        if anchor_start is None or raw_group_start is None:
            continue
        raw_start = as_float(row.get("copy_start_ms"))
        raw_end = as_float(row.get("copy_end_ms"))
        if raw_start is None or raw_end is None:
            continue
        offset = anchor_start - raw_group_start
        start = round(raw_start + offset, 3)
        end = round(raw_end + offset, 3)
        due = as_float(gap.get("tool_gap_end_ms"))
        token_count = (
            as_float(row.get("token_or_index_count"))
            or as_float(row.get("host_index_count"))
            or as_float(row.get("device_index_count"))
            or 0.0
        )
        block_key = "|".join(
            str(row.get(key) or "")
            for key in ("session_id", "node_id", "host_index_signature", "device_index_signature")
        )
        output.append(
            {
                "row": gap.get("timeline_label") or "",
                "session_id": gap.get("session_id", ""),
                "ledger_session_id": session_id,
                "case_id": gap.get("case_id", ""),
                "mode": gap.get("mode", ""),
                "task_index": gap.get("task_index", ""),
                "gap_order_in_task": gap.get("gap_order_in_task", ""),
                "tool_gap_ms": gap.get("tool_gap_ms", ""),
                "fillers": case_fillers(gap),
                "phase": phase,
                "source_event": row.get("source_event", ""),
                "node_id": row.get("node_id", ""),
                "layer_id": row.get("layer_id", ""),
                "block_key": block_key,
                "token_or_index_count": round(token_count, 3),
                "aligned_h2d_start_ms": start,
                "aligned_h2d_end_ms": end,
                "h2d_duration_ms": round(max(0.0, end - start), 3),
                "relative_h2d_start_ms": round(start - due, 3) if due is not None else "",
                "relative_h2d_end_ms": round(end - due, 3) if due is not None else "",
                "raw_copy_start_ms": row.get("copy_start_ms", ""),
                "raw_copy_end_ms": row.get("copy_end_ms", ""),
                "raw_duration_ms": row.get("duration_ms", ""),
                "confidence": row.get("confidence", ""),
                "simple_meaning": row.get("simple_meaning", ""),
            }
        )
    output.sort(
        key=lambda row: (
            str(row.get("case_id") or ""),
            as_float(row.get("aligned_h2d_start_ms")) or 0.0,
            as_float(row.get("aligned_h2d_end_ms")) or 0.0,
            str(row.get("row") or ""),
        )
    )
    return output


def h2d_events_overlap_window(
    events: list[dict[str, Any]],
    start_ms: float,
    end_ms: float,
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    output = []
    for event in events:
        if case_id and str(event.get("case_id") or "") != case_id:
            continue
        start = as_float(event.get("aligned_h2d_start_ms"))
        end = as_float(event.get("aligned_h2d_end_ms"))
        if start is None or end is None:
            continue
        if start <= end_ms and end >= start_ms:
            output.append(event)
    return output


def peak_concurrent_h2d_events(events: list[dict[str, Any]]) -> int:
    points: list[tuple[float, int]] = []
    for event in events:
        start = as_float(event.get("aligned_h2d_start_ms"))
        end = as_float(event.get("aligned_h2d_end_ms"))
        if start is None or end is None:
            continue
        points.append((start, 1))
        points.append((max(start, end), -1))
    active = 0
    peak = 0
    for _, delta in sorted(points, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def exact_movement_case_id(row: dict[str, Any]) -> str:
    session_id = str(row.get("session_id") or "")
    if "::" in session_id:
        return session_id.split("::", 1)[0]
    return str(row.get("case_id") or "")


def exact_movement_display_session(row: dict[str, Any]) -> str:
    session_id = str(row.get("session_id") or "")
    if "::" in session_id:
        return session_id.split("::", 1)[1]
    return session_id


def exact_movement_kind(row: dict[str, Any]) -> str:
    return movement_kind_display(movement_kind_from_row(row))


def exact_movement_source_rank(row: dict[str, Any]) -> int:
    source = str(row.get("source_event") or "")
    kind = exact_movement_kind(row)
    if kind == "H2D":
        return exact_h2d_source_rank(row)
    if kind == "D2H":
        if "hostpool.backup_from_device_all_layer" in source:
            return 0
        if source == "hicache.write.end":
            return 1
        return 2
    if kind == "GPU evict":
        if source == "hicache.evict_device.end":
            return 0
        return 1
    return 2


def selected_exact_movement_rows(exact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer lower-level copy/eviction hooks when wrapper and lower-level events both exist."""
    grouped: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in exact_rows:
        session_id = str(row.get("session_id") or "")
        if not session_id:
            continue
        start = as_float(row.get("copy_start_ms"))
        end = as_float(row.get("copy_end_ms"))
        if start is None or end is None:
            continue
        key = (
            session_id,
            str(row.get("phase") or ""),
            str(row.get("movement") or ""),
            str(row.get("direction") or ""),
        )
        grouped[key].append(row)

    selected: list[dict[str, Any]] = []
    for rows in grouped.values():
        best_rank = min(exact_movement_source_rank(row) for row in rows)
        selected.extend(row for row in rows if exact_movement_source_rank(row) == best_rank)
    return selected


def exact_movement_case_offsets(
    gaps: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Return per-case raw-trace-to-gap-clock offsets using target H2D anchors."""
    gap_by_session: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        for key in (str(gap.get("ledger_session_id") or ""), str(gap.get("session_id") or "")):
            if key:
                gap_by_session[key] = gap

    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_exact_movement_rows(exact_rows):
        if exact_movement_kind(row) != "H2D":
            continue
        session_id = str(row.get("session_id") or "")
        if session_id not in gap_by_session:
            continue
        grouped[(session_id, h2d_event_phase(row))].append(row)

    offsets: defaultdict[str, list[float]] = defaultdict(list)
    for (session_id, phase), rows in grouped.items():
        raw_starts = [as_float(row.get("copy_start_ms")) for row in rows]
        raw_starts = [value for value in raw_starts if value is not None]
        if not raw_starts:
            continue
        gap = gap_by_session[session_id]
        anchor_key = "direct_kv_h2d_start_ms" if phase == "hint" else "replay_kv_h2d_start_ms"
        anchor_start = as_float(gap.get(anchor_key))
        if anchor_start is None:
            continue
        case_id = str(gap.get("case_id") or exact_movement_case_id(rows[0]))
        if case_id:
            offsets[case_id].append(anchor_start - min(raw_starts))

    return {case_id: median(values) for case_id, values in offsets.items() if values}


def all_aligned_kv_movement_events(
    gaps: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align all exact KV movement events, including filler pressure, to the case clock."""
    offsets = exact_movement_case_offsets(gaps, exact_rows)
    gap_by_ledger_session: dict[str, dict[str, Any]] = {
        str(gap.get("ledger_session_id") or ""): gap
        for gap in gaps
        if str(gap.get("ledger_session_id") or "")
    }

    output: list[dict[str, Any]] = []
    for row in selected_exact_movement_rows(exact_rows):
        case_id = exact_movement_case_id(row)
        offset = offsets.get(case_id)
        raw_start = as_float(row.get("copy_start_ms"))
        raw_end = as_float(row.get("copy_end_ms"))
        if offset is None or raw_start is None or raw_end is None:
            continue

        ledger_session_id = str(row.get("session_id") or "")
        display_session_id = exact_movement_display_session(row)
        owner_gap = gap_by_ledger_session.get(ledger_session_id)
        phase = str(row.get("phase") or "")
        if owner_gap:
            owner_kind = "target row"
            owner_row = str(owner_gap.get("timeline_label") or "")
        elif phase == "pressure_filler" or "_pressure_" in display_session_id:
            owner_kind = "pressure/filler"
            owner_row = ""
        else:
            owner_kind = "other session"
            owner_row = ""

        aligned_start = round(raw_start + offset, 3)
        aligned_end = round(raw_end + offset, 3)
        duration = max(0.0, aligned_end - aligned_start)
        block_key = "|".join(
            str(row.get(key) or "")
            for key in ("session_id", "node_id", "host_index_signature", "device_index_signature")
        )
        token_count = (
            as_float(row.get("token_or_index_count"))
            or as_float(row.get("host_index_count"))
            or as_float(row.get("device_index_count"))
            or 0.0
        )
        output.append(
            {
                "case_id": case_id,
                "owner_kind": owner_kind,
                "owner_row": owner_row,
                "session_id": display_session_id,
                "ledger_session_id": ledger_session_id,
                "phase": phase,
                "movement": row.get("movement", ""),
                "direction": row.get("direction", ""),
                "movement_kind": exact_movement_kind(row),
                "source_event": row.get("source_event", ""),
                "node_id": row.get("node_id", ""),
                "layer_id": row.get("layer_id", ""),
                "block_key": block_key,
                "token_or_index_count": round(token_count, 3),
                "aligned_start_ms": aligned_start,
                "aligned_end_ms": aligned_end,
                "duration_ms": round(duration, 3),
                "raw_copy_start_ms": row.get("copy_start_ms", ""),
                "raw_copy_end_ms": row.get("copy_end_ms", ""),
                "case_clock_offset_ms": round(offset, 3),
                "alignment_confidence": "case_h2d_anchor",
                "evidence_confidence": row.get("confidence", ""),
                "simple_meaning": row.get("simple_meaning", ""),
            }
        )
    output.sort(
        key=lambda row: (
            str(row.get("case_id") or ""),
            as_float(row.get("aligned_start_ms")) or 0.0,
            as_float(row.get("aligned_end_ms")) or 0.0,
            str(row.get("movement_kind") or ""),
            str(row.get("ledger_session_id") or ""),
        )
    )
    return output


def kv_events_overlap_window(
    events: list[dict[str, Any]],
    start_ms: float,
    end_ms: float,
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in events:
        if case_id and str(event.get("case_id") or "") != case_id:
            continue
        start = as_float(event.get("aligned_start_ms"))
        end = as_float(event.get("aligned_end_ms"))
        if start is None or end is None:
            continue
        if start <= end_ms and end >= start_ms:
            output.append(event)
    return output


def peak_concurrent_kv_events(events: list[dict[str, Any]]) -> int:
    points: list[tuple[float, int]] = []
    for event in events:
        start = as_float(event.get("aligned_start_ms"))
        end = as_float(event.get("aligned_end_ms"))
        if start is None or end is None:
            continue
        points.append((start, 1))
        points.append((max(start, end), -1))
    active = 0
    peak = 0
    for _, delta in sorted(points, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def client_dispatch_window_end(row: dict[str, Any]) -> tuple[float | None, str]:
    for key, label in (
        ("replay_sglang_receive_start_ms", "SGLang receive"),
        ("resume_start_ms", "client request call start"),
        ("replay_scheduler_queue_enter_start_ms", "scheduler queue"),
        ("replay_scheduler_admit_start_ms", "scheduler admit"),
        ("replay_kv_h2d_start_ms", "replay H2D start"),
    ):
        value = as_float(row.get(key))
        if value is not None:
            return value, label
    token_time = first_token_ms(row)
    if token_time is not None:
        return token_time, "first token"
    return None, ""


def client_dispatch_kv_movement_summary_rows(
    gaps: list[dict[str, Any]],
    all_kv_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        due = as_float(gap.get("tool_gap_end_ms"))
        if due is None:
            continue
        window_end, end_stage = client_dispatch_window_end(gap)
        if window_end is None or window_end <= due:
            continue
        label = str(gap.get("timeline_label") or f"G{idx:02d}")
        case_id = str(gap.get("case_id") or "")
        target_session = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        events = kv_events_overlap_window(all_kv_events, due, window_end, case_id=case_id)
        kind_counts = Counter(str(event.get("movement_kind") or "KV movement") for event in events)
        owner_counts = Counter(str(event.get("owner_kind") or "unknown") for event in events)
        target_events = [event for event in events if str(event.get("ledger_session_id") or "") == target_session]
        filler_events = [event for event in events if str(event.get("owner_kind") or "") == "pressure/filler"]
        token_sum = sum(as_float(event.get("token_or_index_count")) or 0.0 for event in events)
        duration_sum = sum(as_float(event.get("duration_ms")) or 0.0 for event in events)
        if not events:
            verdict = "no visible KV movement during dispatch"
            explanation = (
                "The exact SGLang KV movement hooks did not show H2D, D2H, or GPU eviction "
                "while this replay was waiting to reach SGLang."
            )
        elif kind_counts.get("H2D", 0) and kind_counts.get("D2H", 0):
            verdict = "mixed H2D/D2H movement during dispatch"
            explanation = (
                "Other KV loads and host backups were visible while this replay was still in the dispatch window."
            )
        elif kind_counts.get("H2D", 0):
            verdict = "H2D movement during dispatch"
            explanation = "Host-to-device KV loads were visible before this target replay reached SGLang."
        elif kind_counts.get("D2H", 0) or kind_counts.get("GPU evict", 0):
            verdict = "D2H/eviction movement during dispatch"
            explanation = "SGLang was backing up or evicting KV for other sessions while this replay was still waiting."
        else:
            verdict = "other KV movement during dispatch"
            explanation = "Some SGLang-visible KV movement happened during the dispatch window."
        rows.append(
            {
                "row": label,
                "session_id": gap.get("session_id", ""),
                "case_id": case_id,
                "mode": gap.get("mode", ""),
                "fillers": case_fillers(gap),
                "tool_wait_ms": gap.get("tool_gap_ms", ""),
                "client_dispatch_window": f"replay due -> {end_stage}",
                "dispatch_window_ms": round(window_end - due, 3),
                "all_kv_events": len(events),
                "h2d_events": kind_counts.get("H2D", 0),
                "d2h_events": kind_counts.get("D2H", 0),
                "gpu_evict_events": kind_counts.get("GPU evict", 0),
                "host_evict_events": kind_counts.get("host evict", 0),
                "target_row_events": len(target_events),
                "pressure_filler_events": len(filler_events),
                "other_session_events": owner_counts.get("other session", 0),
                "logical_blocks_touched": len({str(event.get("block_key") or "") for event in events if str(event.get("block_key") or "")}),
                "token_or_index_count_sum": round(token_sum, 3),
                "duration_sum_ms": round(duration_sum, 3),
                "peak_concurrent_kv_events": peak_concurrent_kv_events(events),
                "verdict": verdict,
                "simple_explanation": explanation,
            }
        )
    return rows


def client_dispatch_kv_movement_event_rows(
    gaps: list[dict[str, Any]],
    all_kv_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        due = as_float(gap.get("tool_gap_end_ms"))
        if due is None:
            continue
        window_end, end_stage = client_dispatch_window_end(gap)
        if window_end is None or window_end <= due:
            continue
        label = str(gap.get("timeline_label") or f"G{idx:02d}")
        case_id = str(gap.get("case_id") or "")
        target_session = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        events = kv_events_overlap_window(all_kv_events, due, window_end, case_id=case_id)
        events.sort(
            key=lambda event: (
                as_float(event.get("aligned_start_ms")) or 0.0,
                as_float(event.get("aligned_end_ms")) or 0.0,
                str(event.get("movement_kind") or ""),
            )
        )
        for order, event in enumerate(events):
            start = as_float(event.get("aligned_start_ms"))
            end = as_float(event.get("aligned_end_ms"))
            owner_session = str(event.get("ledger_session_id") or "")
            rows.append(
                {
                    "row": label,
                    "case_id": case_id,
                    "mode": gap.get("mode", ""),
                    "fillers": case_fillers(gap),
                    "tool_wait_ms": gap.get("tool_gap_ms", ""),
                    "client_dispatch_window": f"replay due -> {end_stage}",
                    "event_order": order,
                    "owner_kind": "target replay" if owner_session == target_session else event.get("owner_kind", ""),
                    "owner_row": event.get("owner_row", ""),
                    "owner_session_id": event.get("session_id", ""),
                    "phase": event.get("phase", ""),
                    "movement_kind": event.get("movement_kind", ""),
                    "movement": event.get("movement", ""),
                    "direction": event.get("direction", ""),
                    "source_event": event.get("source_event", ""),
                    "node_id": event.get("node_id", ""),
                    "layer_id": event.get("layer_id", ""),
                    "block_key": event.get("block_key", ""),
                    "token_or_index_count": event.get("token_or_index_count", ""),
                    "start_relative_to_replay_due_ms": round(start - due, 3) if start is not None else "",
                    "end_relative_to_replay_due_ms": round(end - due, 3) if end is not None else "",
                    "duration_ms": event.get("duration_ms", ""),
                    "alignment_confidence": event.get("alignment_confidence", ""),
                    "evidence_confidence": event.get("evidence_confidence", ""),
                    "simple_meaning": event.get("simple_meaning", ""),
                }
            )
    return rows


def pressure_level(event_count: int, peak: int, tokens: float, duration_sum: float) -> str:
    if event_count <= 0:
        return "none"
    if peak >= 8 or event_count >= 48 or tokens >= 50_000 or duration_sum >= 500:
        return "high"
    if peak >= 3 or event_count >= 12 or tokens >= 10_000 or duration_sum >= 100:
        return "medium"
    return "low"


def h2d_pressure_by_gap_rows(
    gaps: list[dict[str, Any]],
    h2d_events: list[dict[str, Any]],
    window_before_ms: float = 500.0,
    window_after_ms: float = 1000.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        due = as_float(gap.get("tool_gap_end_ms"))
        if due is None:
            continue
        label = str(gap.get("timeline_label") or f"G{idx:02d}")
        case_id = str(gap.get("case_id") or "")
        session_key = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        h2d_start = as_float(gap.get("replay_kv_h2d_start_ms")) or as_float(gap.get("direct_kv_h2d_start_ms"))
        h2d_end = as_float(gap.get("replay_kv_h2d_end_ms")) or as_float(gap.get("direct_kv_h2d_end_ms"))
        deadline_window_start = due - window_before_ms
        deadline_window_end = due + window_after_ms
        # If H2D happens much later than the deadline, a fixed +1000 ms window
        # hides the important work. Use a dynamic "deadline-to-ready" window for
        # pressure, and keep the fixed near-deadline count as a separate field.
        window_start = deadline_window_start
        window_end = max(deadline_window_end, h2d_end or deadline_window_end)
        deadline_nearby = h2d_events_overlap_window(h2d_events, deadline_window_start, deadline_window_end, case_id=case_id)
        nearby = h2d_events_overlap_window(h2d_events, window_start, window_end, case_id=case_id)
        own = [event for event in nearby if str(event.get("ledger_session_id") or "") == session_key]
        other = [event for event in nearby if str(event.get("ledger_session_id") or "") != session_key]
        token_sum = sum(as_float(event.get("token_or_index_count")) or 0.0 for event in nearby)
        duration_sum = sum(as_float(event.get("h2d_duration_ms")) or 0.0 for event in nearby)
        unique_blocks: dict[str, float] = {}
        for event in nearby:
            block_key = str(event.get("block_key") or "")
            if not block_key:
                continue
            unique_blocks[block_key] = max(
                unique_blocks.get(block_key, 0.0),
                as_float(event.get("token_or_index_count")) or 0.0,
            )
        peak = peak_concurrent_h2d_events(nearby)
        level = pressure_level(len(nearby), peak, token_sum, duration_sum)
        finish_margin = round(due - h2d_end, 3) if h2d_end is not None else ""
        rows.append(
            {
                "row": label,
                "session_id": gap.get("session_id", ""),
                "case_id": gap.get("case_id", ""),
                "mode": gap.get("mode", ""),
                "fillers": case_fillers(gap),
                "tool_wait_ms": gap.get("tool_gap_ms", ""),
                "replay_due_ms": round(due, 3),
                "deadline_window_start_relative_ms": -window_before_ms,
                "deadline_window_end_relative_ms": window_after_ms,
                "deadline_window_h2d_events": len(deadline_nearby),
                "window_start_relative_ms": round(window_start - due, 3),
                "window_end_relative_ms": round(window_end - due, 3),
                "nearby_h2d_events": len(nearby),
                "own_h2d_events": len(own),
                "other_h2d_events": len(other),
                "nearby_logical_blocks": len(unique_blocks),
                "nearby_h2d_event_tokens": round(token_sum, 3),
                "nearby_logical_block_tokens_est": round(sum(unique_blocks.values()), 3),
                "nearby_h2d_duration_sum_ms": round(duration_sum, 3),
                "peak_concurrent_h2d_events": peak,
                "pressure_level": level,
                "replay_h2d_start_relative_ms": round(h2d_start - due, 3) if h2d_start is not None else "",
                "replay_h2d_end_relative_ms": round(h2d_end - due, 3) if h2d_end is not None else "",
                "h2d_finish_margin_ms": finish_margin,
                "resume_ttft_ms": gap.get("resume_ttft_ms", ""),
                "replay_path": gap.get("replay_path", ""),
                "simple_meaning": (
                    f"{level} deadline-to-ready H2D pressure: {len(nearby)} copy events "
                    f"({len(own)} from this gap, {len(other)} from other gaps in the same case), "
                    f"peak concurrency {peak}, {token_sum:.0f} token/index movements. "
                    f"Fixed near-deadline window had {len(deadline_nearby)} H2D events."
                ),
            }
        )
    return rows


def h2d_activity_window_rows(h2d_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows = [
        ("before replay due", lambda start, end: end < 0),
        ("0-100 ms after due", lambda start, end: start < 100 and end >= 0),
        ("100-500 ms after due", lambda start, end: start < 500 and end >= 100),
        ("500-1000 ms after due", lambda start, end: start < 1000 and end >= 500),
        ("1-5 s after due", lambda start, end: start < 5000 and end >= 1000),
        ("5-10 s after due", lambda start, end: start < 10000 and end >= 5000),
        ("10-30 s after due", lambda start, end: start < 30000 and end >= 10000),
        (">30 s after due", lambda start, end: start >= 30000),
    ]
    output: list[dict[str, Any]] = []
    for label, predicate in windows:
        rows = []
        for event in h2d_events:
            start = as_float(event.get("relative_h2d_start_ms"))
            end = as_float(event.get("relative_h2d_end_ms"))
            if start is None or end is None:
                continue
            if predicate(start, end):
                rows.append(event)
        unique_sessions = {str(row.get("ledger_session_id") or "") for row in rows if str(row.get("ledger_session_id") or "")}
        unique_blocks = {str(row.get("block_key") or "") for row in rows if str(row.get("block_key") or "")}
        output.append(
            {
                "time_window_relative_to_replay_due": label,
                "h2d_events": len(rows),
                "sessions": len(unique_sessions),
                "logical_blocks_touched": len(unique_blocks),
                "token_or_index_count_sum": round(sum(as_float(row.get("token_or_index_count")) or 0.0 for row in rows), 3),
                "duration_sum_ms": round(sum(as_float(row.get("h2d_duration_ms")) or 0.0 for row in rows), 3),
                "peak_concurrent_h2d_events": peak_concurrent_h2d_events(rows),
            }
        )
    return output


def h2d_activity_window_bar_chart(rows: list[dict[str, Any]]) -> str:
    if not rows or not any((as_float(row.get("h2d_events")) or 0) > 0 for row in rows):
        return "<p>No exact H2D activity windows were available for the bandwidth-pressure chart.</p>"
    width = 1480
    height = 440
    left = 80
    right = 30
    top = 56
    bottom = 128
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_events = max(as_float(row.get("h2d_events")) or 0.0 for row in rows) or 1.0
    bar_w = plot_w / max(1, len(rows)) * 0.66
    gap_w = plot_w / max(1, len(rows))
    parts = [
        '<svg viewBox="0 0 1480 440" width="100%" role="img" aria-label="H2D activity window bar chart">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>',
        '<text x="80" y="32" font-size="13" fill="#334155" font-weight="700">H2D copy events grouped by time window around replay due</text>',
        '<text x="22" y="184" transform="rotate(-90 22 184)" font-size="13" font-weight="700" text-anchor="middle">H2D events</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = max_events * frac
        y = top + plot_h - frac * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#475569">{int(round(value))}</text>')
    for idx, row in enumerate(rows):
        count = as_float(row.get("h2d_events")) or 0.0
        x = left + idx * gap_w + (gap_w - bar_w) / 2
        h = 0 if max_events <= 0 else count / max_events * plot_h
        y = top + plot_h - h
        color = "#0f766e" if count else "#cbd5e1"
        title = (
            f"{row.get('time_window_relative_to_replay_due')}: events={row.get('h2d_events')}; "
            f"sessions={row.get('sessions')}; blocks={row.get('logical_blocks_touched')}; "
            f"tokens/indices={row.get('token_or_index_count_sum')}; duration_sum_ms={row.get('duration_sum_ms')}; "
            f"peak_concurrent={row.get('peak_concurrent_h2d_events')}"
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="5" fill="{color}" opacity="0.85">'
            f'<title>{html.escape(title)}</title></rect>'
        )
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{max(top + 14, y - 8):.1f}" text-anchor="middle" font-size="11" font-weight="700">{int(count)}</text>')
        label = str(row.get("time_window_relative_to_replay_due") or "")
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 20:.1f}" text-anchor="end" '
            f'transform="rotate(-28 {x + bar_w / 2:.1f} {top + plot_h + 20:.1f})" font-size="10" fill="#334155">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def select_h2d_contention_targets(
    gaps: list[dict[str, Any]],
    max_targets: int = 6,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        due = as_float(gap.get("tool_gap_end_ms"))
        h2d_end = as_float(gap.get("replay_kv_h2d_end_ms")) or as_float(gap.get("direct_kv_h2d_end_ms"))
        if due is None or h2d_end is None:
            continue
        copied = dict(gap)
        copied.setdefault("timeline_label", f"G{idx:02d}")
        copied["_contention_lateness_ms"] = h2d_end - due
        candidates.append(copied)
    candidates.sort(key=lambda row: as_float(row.get("_contention_lateness_ms")) or 0.0, reverse=True)
    return candidates[:max_targets]


def h2d_contention_event_rows(
    target_gaps: list[dict[str, Any]],
    h2d_events: list[dict[str, Any]],
    before_ms: float = 500.0,
    after_finish_ms: float = 500.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, gap in enumerate(target_gaps):
        target_row = str(gap.get("timeline_label") or f"C{target_index:02d}")
        due = as_float(gap.get("tool_gap_end_ms"))
        h2d_start = as_float(gap.get("replay_kv_h2d_start_ms")) or as_float(gap.get("direct_kv_h2d_start_ms"))
        h2d_end = as_float(gap.get("replay_kv_h2d_end_ms")) or as_float(gap.get("direct_kv_h2d_end_ms"))
        if due is None or h2d_end is None:
            continue
        target_session = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        case_id = str(gap.get("case_id") or "")
        window_start = due - before_ms
        window_end = max(due + 1000.0, h2d_end + after_finish_ms)
        window_events = h2d_events_overlap_window(h2d_events, window_start, window_end, case_id=case_id)
        window_events.sort(
            key=lambda event: (
                as_float(event.get("aligned_h2d_start_ms")) or 0.0,
                as_float(event.get("aligned_h2d_end_ms")) or 0.0,
                str(event.get("ledger_session_id") or ""),
            )
        )
        for order, event in enumerate(window_events):
            owner_session = str(event.get("ledger_session_id") or "")
            is_target = owner_session == target_session
            start = as_float(event.get("aligned_h2d_start_ms"))
            end = as_float(event.get("aligned_h2d_end_ms"))
            rows.append(
                {
                    "target_row": target_row,
                    "event_order": order,
                    "owner_row": event.get("row", ""),
                    "same_as_target": "yes" if is_target else "no",
                    "owner_kind": "target replay H2D" if is_target else "other H2D in same case",
                    "phase": event.get("phase", ""),
                    "source_event": event.get("source_event", ""),
                    "node_id": event.get("node_id", ""),
                    "layer_id": event.get("layer_id", ""),
                    "token_or_index_count": event.get("token_or_index_count", ""),
                    "start_relative_to_target_due_ms": round(start - due, 3) if start is not None else "",
                    "end_relative_to_target_due_ms": round(end - due, 3) if end is not None else "",
                    "duration_ms": event.get("h2d_duration_ms", ""),
                    "block_key": event.get("block_key", ""),
                    "confidence": event.get("confidence", ""),
                    "target_replay_due_ms": round(due, 3),
                    "target_h2d_start_relative_ms": round(h2d_start - due, 3) if h2d_start is not None else "",
                    "target_h2d_end_relative_ms": round(h2d_end - due, 3),
                    "case_id": case_id,
                    "target_session_id": gap.get("session_id", ""),
                    "owner_session_id": event.get("session_id", ""),
                }
            )
    return rows


def h2d_contention_summary_rows(
    target_gaps: list[dict[str, Any]],
    h2d_events: list[dict[str, Any]],
    before_ms: float = 500.0,
    after_finish_ms: float = 500.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, gap in enumerate(target_gaps):
        target_row = str(gap.get("timeline_label") or f"C{target_index:02d}")
        due = as_float(gap.get("tool_gap_end_ms"))
        h2d_start = as_float(gap.get("replay_kv_h2d_start_ms")) or as_float(gap.get("direct_kv_h2d_start_ms"))
        h2d_end = as_float(gap.get("replay_kv_h2d_end_ms")) or as_float(gap.get("direct_kv_h2d_end_ms"))
        if due is None or h2d_end is None:
            continue
        target_session = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        case_id = str(gap.get("case_id") or "")
        window_start = due - before_ms
        window_end = max(due + 1000.0, h2d_end + after_finish_ms)
        window_events = h2d_events_overlap_window(h2d_events, window_start, window_end, case_id=case_id)
        own = [event for event in window_events if str(event.get("ledger_session_id") or "") == target_session]
        other = [event for event in window_events if str(event.get("ledger_session_id") or "") != target_session]
        before_target = []
        if h2d_start is not None:
            before_target = [
                event
                for event in window_events
                if (as_float(event.get("aligned_h2d_end_ms")) or 0.0) > due
                and (as_float(event.get("aligned_h2d_start_ms")) or 0.0) < h2d_start
            ]
        other_before_target = [
            event for event in before_target if str(event.get("ledger_session_id") or "") != target_session
        ]
        if other_before_target:
            verdict = "blocked behind other H2D"
            explanation = (
                f"{len(other_before_target)} other H2D events overlapped the interval after replay due "
                "and before this row's own H2D started."
            )
        elif h2d_start is not None and h2d_start - due > 1000:
            verdict = "H2D path quiet before target"
            explanation = (
                "No other H2D events were visible before this row's own H2D started. "
                "The delay likely happened before SGLang reached the H2D copy path."
            )
        elif own:
            verdict = "target H2D visible"
            explanation = "The row's own H2D events were visible in the contention window."
        else:
            verdict = "no visible contention"
            explanation = "No H2D events were visible in the contention window."
        rows.append(
            {
                "target_row": target_row,
                "case_id": case_id,
                "mode": gap.get("mode", ""),
                "fillers": case_fillers(gap),
                "tool_wait_ms": gap.get("tool_gap_ms", ""),
                "target_h2d_lateness_ms": round(h2d_end - due, 3),
                "target_h2d_start_relative_ms": round(h2d_start - due, 3) if h2d_start is not None else "",
                "target_h2d_end_relative_ms": round(h2d_end - due, 3),
                "contention_window_start_ms": -before_ms,
                "contention_window_end_ms": round(window_end - due, 3),
                "all_h2d_events_in_window": len(window_events),
                "target_h2d_events": len(own),
                "other_h2d_events": len(other),
                "other_h2d_before_target_start": len(other_before_target),
                "peak_concurrent_h2d_events": peak_concurrent_h2d_events(window_events),
                "h2d_token_or_index_sum": round(sum(as_float(event.get("token_or_index_count")) or 0.0 for event in window_events), 3),
                "verdict": verdict,
                "simple_explanation": explanation,
            }
        )
    return rows


def build_per_gap_h2d_contention_svg(
    target_gaps: list[dict[str, Any]],
    h2d_events: list[dict[str, Any]],
    max_targets: int = 6,
) -> str:
    targets = target_gaps[:max_targets]
    if not targets:
        return "<p>No replay-H2D targets were available for the contention timeline.</p>"
    width = 1480
    left = 178
    right = 34
    top = 76
    target_h = 138
    lane_h = 18
    gap_h = 16
    height = top + len(targets) * target_h + 96
    plot_w = width - left - right

    def symlog(value: float, linear_width: float = 50.0) -> float:
        if value == 0:
            return 0.0
        return math.copysign(math.log1p(abs(value) / linear_width), value)

    windows: list[tuple[float, float]] = []
    target_data: list[tuple[dict[str, Any], float, float, list[dict[str, Any]]]] = []
    for gap in targets:
        due = as_float(gap.get("tool_gap_end_ms"))
        h2d_end = as_float(gap.get("replay_kv_h2d_end_ms")) or as_float(gap.get("direct_kv_h2d_end_ms"))
        if due is None or h2d_end is None:
            continue
        case_id = str(gap.get("case_id") or "")
        window_start = due - 500.0
        window_end = max(due + 1000.0, h2d_end + 500.0)
        events = h2d_events_overlap_window(h2d_events, window_start, window_end, case_id=case_id)
        events.sort(key=lambda event: as_float(event.get("aligned_h2d_start_ms")) or 0.0)
        rel_start = window_start - due
        rel_end = window_end - due
        windows.append((rel_start, rel_end))
        target_data.append((gap, rel_start, rel_end, events))
    if not target_data:
        return "<p>No contention windows could be built for the selected H2D targets.</p>"

    x_min = min(start for start, _ in windows)
    x_max = max(end for _, end in windows)
    x_min = min(x_min, -500.0)
    x_max = max(x_max, 1000.0)
    scaled_min = symlog(x_min)
    scaled_max = symlog(x_max)

    def x_pos(relative_ms: float) -> float:
        scaled = symlog(relative_ms)
        return left + (scaled - scaled_min) * plot_w / max(1e-9, scaled_max - scaled_min)

    def draw_bar(parts: list[str], x1: float, x2: float, y: float, color: str, label: str, title: str, opacity: float = 0.86) -> None:
        w = max(3.0, x2 - x1)
        parts.append(
            f'<rect x="{x1:.1f}" y="{y:.1f}" width="{w:.1f}" height="{lane_h:.1f}" rx="4" '
            f'fill="{color}" opacity="{opacity}"><title>{html.escape(title)}</title></rect>'
        )
        if w >= 48 and label:
            visible = label if len(label) <= int(w / 5.8) else label[: max(5, int(w / 5.8) - 3)] + "..."
            parts.append(
                f'<text x="{x1 + w / 2:.1f}" y="{y + 12.5:.1f}" text-anchor="middle" font-size="9" '
                f'fill="#ffffff" font-weight="800">{html.escape(visible)}</text>'
            )

    zero_x = x_pos(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Per-gap H2D contention timeline">',
        '<text x="12" y="26" font-size="18" font-weight="700" fill="#0f172a">Per-gap H2D contention timeline</text>',
        '<text x="12" y="48" font-size="12" fill="#475569">Each target row shows all exact H2D events in the same controlled case from replay due through the target KV-ready time.</text>',
        f'<line x1="{zero_x:.1f}" y1="{top - 20}" x2="{zero_x:.1f}" y2="{height - 64}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{zero_x + 4:.1f}" y="{top - 28}" font-size="11" font-weight="800">0 ms replay due</text>',
    ]
    ticks = [-500.0, -100.0, -10.0, 0.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 30000.0, 60000.0, 120000.0]
    for tick in ticks:
        if x_min <= tick <= x_max:
            x = x_pos(tick)
            parts.append(f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 70}" stroke="#e5e7eb"/>')
            label = f"{int(tick)} ms" if abs(tick) < 1000 else f"{tick / 1000:.0f} s"
            parts.append(f'<text x="{x:.1f}" y="{top - 18}" text-anchor="middle" font-size="10" fill="#475569">{html.escape(label)}</text>')

    for idx, (gap, rel_start, rel_end, events) in enumerate(target_data):
        y = top + idx * target_h
        band = "#ffffff" if idx % 2 == 0 else "#eef4fb"
        parts.append(f'<rect x="0" y="{y - 4:.1f}" width="{width}" height="{target_h - gap_h:.1f}" fill="{band}"/>')
        label = str(gap.get("timeline_label") or f"C{idx:02d}")
        due = as_float(gap.get("tool_gap_end_ms")) or 0.0
        h2d_start = as_float(gap.get("replay_kv_h2d_start_ms")) or as_float(gap.get("direct_kv_h2d_start_ms"))
        h2d_end = as_float(gap.get("replay_kv_h2d_end_ms")) or as_float(gap.get("direct_kv_h2d_end_ms"))
        target_session = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        summaries = h2d_contention_summary_rows([gap], h2d_events)
        verdict = summaries[0].get("verdict", "") if summaries else ""
        lateness = h2d_end - due if h2d_end is not None else 0.0
        parts.append(f'<text x="12" y="{y + 19:.1f}" font-size="15" font-weight="800" fill="#0f172a">{html.escape(label)}</text>')
        parts.append(f'<text x="12" y="{y + 39:.1f}" font-size="11" font-weight="800" fill="#b91c1c">late {display_ms(lateness)}</text>')
        parts.append(f'<text x="12" y="{y + 58:.1f}" font-size="10" fill="#475569">{html.escape(str(verdict))}</text>')
        parts.append(f'<text x="12" y="{y + 76:.1f}" font-size="10" fill="#475569">fillers {html.escape(case_fillers(gap))}; wait {html.escape(display_ms(gap.get("tool_gap_ms")))}</text>')

        lane_target_y = y + 18
        lane_other_y = y + 48
        lane_span_y = y + 78
        parts.append(f'<text x="{left - 12}" y="{lane_target_y + 13:.1f}" text-anchor="end" font-size="10" font-weight="700" fill="#334155">target H2D</text>')
        parts.append(f'<text x="{left - 12}" y="{lane_other_y + 13:.1f}" text-anchor="end" font-size="10" font-weight="700" fill="#334155">other H2D</text>')
        parts.append(f'<text x="{left - 12}" y="{lane_span_y + 13:.1f}" text-anchor="end" font-size="10" font-weight="700" fill="#334155">wait span</text>')
        parts.append(f'<line x1="{left}" y1="{lane_target_y + lane_h / 2:.1f}" x2="{left + plot_w}" y2="{lane_target_y + lane_h / 2:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<line x1="{left}" y1="{lane_other_y + lane_h / 2:.1f}" x2="{left + plot_w}" y2="{lane_other_y + lane_h / 2:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<line x1="{left}" y1="{lane_span_y + lane_h / 2:.1f}" x2="{left + plot_w}" y2="{lane_span_y + lane_h / 2:.1f}" stroke="#e2e8f0"/>')

        if h2d_start is not None and h2d_end is not None:
            draw_bar(
                parts,
                x_pos(h2d_start - due),
                x_pos(h2d_end - due),
                lane_span_y,
                "#f97316",
                "target wait",
                f"Target replay due to H2D finish: {h2d_end - due:.3f} ms",
                0.35,
            )

        other_lane_count = 0
        for event in events:
            start = as_float(event.get("aligned_h2d_start_ms"))
            end = as_float(event.get("aligned_h2d_end_ms"))
            if start is None or end is None:
                continue
            is_target = str(event.get("ledger_session_id") or "") == target_session
            rel_event_start = start - due
            rel_event_end = end - due
            y_event = lane_target_y if is_target else lane_other_y + (other_lane_count % 2) * 20
            if not is_target:
                other_lane_count += 1
            color = "#06b6d4" if is_target else "#64748b"
            if str(event.get("phase") or "") == "hint":
                color = "#16a34a"
            title = (
                f"{'target' if is_target else 'other'} H2D | owner={event.get('row', '')} | "
                f"phase={event.get('phase', '')} | source={event.get('source_event', '')} | "
                f"start={rel_event_start:.3f} ms | end={rel_event_end:.3f} ms | "
                f"duration={event.get('h2d_duration_ms', '')} ms | tokens={event.get('token_or_index_count', '')}"
            )
            draw_bar(
                parts,
                x_pos(rel_event_start),
                x_pos(rel_event_end),
                y_event,
                color,
                "target" if is_target else str(event.get("row") or "other"),
                title,
                0.88,
            )

    legend_y = height - 42
    legend = [
        ("target replay H2D", "#06b6d4"),
        ("other replay H2D", "#64748b"),
        ("hint/prefetch H2D", "#16a34a"),
        ("deadline-to-ready span", "#f97316"),
    ]
    lx = left
    for label, color in legend:
        parts.append(f'<rect x="{lx:.1f}" y="{legend_y:.1f}" width="14" height="14" rx="3" fill="{color}" opacity="0.85"/>')
        parts.append(f'<text x="{lx + 20:.1f}" y="{legend_y + 12:.1f}" font-size="12" fill="#334155">{html.escape(label)}</text>')
        lx += 210
    parts.append("</svg>")
    return "\n".join(parts)


def h2d_bandwidth_pressure_html(
    gaps: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
    max_detail_rows: int = 120,
) -> str:
    all_labeled = timeline_rows_with_labels(selected_timeline_gaps(gaps, len(gaps)))
    h2d_events = aligned_h2d_activity_events(all_labeled, exact_rows)
    pressure_rows = h2d_pressure_by_gap_rows(all_labeled, h2d_events)
    window_rows = h2d_activity_window_rows(h2d_events)
    contention_targets = select_h2d_contention_targets(all_labeled)
    contention_summary = h2d_contention_summary_rows(contention_targets, h2d_events)
    contention_events = h2d_contention_event_rows(contention_targets, h2d_events)
    if not h2d_events:
        return """
        <p>No exact H2D copy activity was available for the bandwidth-pressure section.</p>
        <p class="note">This usually means this run did not trigger host-to-device KV movement, or the run was generated before exact H2D tracing was enabled.</p>
        """
    summary = [
        {
            "aligned_h2d_events": len(h2d_events),
            "unique_sessions": len({str(row.get("ledger_session_id") or "") for row in h2d_events}),
            "unique_logical_blocks": len({str(row.get("block_key") or "") for row in h2d_events}),
            "token_or_index_count_sum": round(sum(as_float(row.get("token_or_index_count")) or 0.0 for row in h2d_events), 3),
            "duration_sum_ms": round(sum(as_float(row.get("h2d_duration_ms")) or 0.0 for row in h2d_events), 3),
            "high_pressure_gaps": sum(1 for row in pressure_rows if row.get("pressure_level") == "high"),
            "medium_pressure_gaps": sum(1 for row in pressure_rows if row.get("pressure_level") == "medium"),
            "low_pressure_gaps": sum(1 for row in pressure_rows if row.get("pressure_level") == "low"),
        }
    ]
    summary_cards = "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{html.escape(str(label))}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
        for label, value in [
            ("aligned H2D events", summary[0]["aligned_h2d_events"]),
            ("unique sessions", summary[0]["unique_sessions"]),
            ("unique logical blocks", summary[0]["unique_logical_blocks"]),
            ("H2D duration sum", f"{summary[0]['duration_sum_ms']} ms"),
            ("medium pressure gaps", summary[0]["medium_pressure_gaps"]),
            ("low pressure gaps", summary[0]["low_pressure_gaps"]),
        ]
    ) + "</div>"
    return f"""
    <p>This section answers: near each replay deadline, how busy was the KV host-to-device movement path?</p>
    <p class="note">The view is relative to replay due time. That avoids mixing separate controlled cases on one misleading absolute clock. Negative means before replay was due; positive means after replay was due.</p>
    {summary_cards}
    <h3>H2D Activity By Time Window</h3>
    <p>Each bar counts exact SGLang-visible H2D copy events in a window around replay due. This is the closest report-level view of memory-movement pressure without running a full hardware profiler.</p>
    <div class="setup-diagram">{h2d_activity_window_bar_chart(window_rows)}</div>
    <h3>Per-Gap Nearby H2D Pressure</h3>
    <p>The pressure window starts at <code>replay_due - 500 ms</code> and extends until either <code>replay_due + 1000 ms</code> or the observed H2D finish time, whichever is later. This shows the H2D traffic seen while the replay was waiting for KV readiness.</p>
    <p class="note"><code>deadline_window_h2d_events</code> keeps the original fixed near-deadline count from <code>-500 ms</code> to <code>+1000 ms</code>. <code>nearby_h2d_events</code> uses the wider deadline-to-ready window.</p>
    <h3>Per-Gap H2D Contention Timeline</h3>
    <p>This view picks the latest H2D rows and shows all exact H2D events in the same controlled case while that row was waiting for KV readiness.</p>
    <p class="note">If other H2D bars appear before the target row's own H2D starts, the target likely waited behind other movement. If the H2D lanes are quiet until the target starts, the delay likely happened before SGLang reached the copy path.</p>
    <div class="setup-diagram">{build_per_gap_h2d_contention_svg(contention_targets, h2d_events)}</div>
    <p class="note">Exact window rows, per-gap pressure rows, contention verdicts, contention events, and aligned H2D event rows are in <strong>Evidence Tables / Raw Proof</strong> at the bottom of the report.</p>
    """


def client_dispatch_movement_color(kind: str) -> str:
    return {
        "H2D": "#06b6d4",
        "D2H": "#f97316",
        "GPU evict": "#64748b",
        "host evict": "#991b1b",
    }.get(kind, "#8b5cf6")


def select_client_dispatch_kv_summary_rows(
    rows: list[dict[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    with_movement = [row for row in rows if int(as_float(row.get("all_kv_events")) or 0) > 0]
    if with_movement:
        selected = sorted(
            with_movement,
            key=lambda row: (
                -(as_float(row.get("all_kv_events")) or 0.0),
                -(as_float(row.get("h2d_events")) or 0.0),
                -(as_float(row.get("d2h_events")) or 0.0),
                str(row.get("row") or ""),
            ),
        )
        return selected[:max_rows]
    return sorted(
        rows,
        key=lambda row: (-(as_float(row.get("dispatch_window_ms")) or 0.0), str(row.get("row") or "")),
    )[:max_rows]


def build_client_dispatch_kv_movement_svg(
    summary_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    max_rows: int = 10,
) -> str:
    selected = select_client_dispatch_kv_summary_rows(summary_rows, max_rows)
    if not selected:
        return "<p>No client-dispatch windows were available for the KV movement chart.</p>"

    events_by_row: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in event_rows:
        events_by_row[str(event.get("row") or "")].append(event)

    max_time = 1000.0
    for row in selected:
        max_time = max(max_time, as_float(row.get("dispatch_window_ms")) or 0.0)
        for event in events_by_row.get(str(row.get("row") or ""), []):
            end = as_float(event.get("end_relative_to_replay_due_ms"))
            if end is not None:
                max_time = max(max_time, end)
    x_min = -100.0
    x_max = max(1000.0, max_time)
    width = 1480
    left = 230
    right = 44
    top = 96
    row_h = 104
    bottom = 84
    height = top + len(selected) * row_h + bottom
    plot_w = width - left - right

    def symlog(value: float, linear_width: float = 50.0) -> float:
        if value == 0:
            return 0.0
        return math.copysign(math.log1p(abs(value) / linear_width), value)

    scaled_min = symlog(x_min)
    scaled_max = symlog(x_max)

    def x_pos(value: float) -> float:
        scaled = symlog(value)
        return left + (scaled - scaled_min) * plot_w / max(1e-9, scaled_max - scaled_min)

    def draw_bar(parts: list[str], start: float, end: float, y: float, color: str, title: str, label: str = "") -> None:
        x1 = x_pos(start)
        x2 = x_pos(max(end, start))
        w = max(3.0, x2 - x1)
        parts.append(
            f'<rect x="{x1:.1f}" y="{y:.1f}" width="{w:.1f}" height="13" rx="4" fill="{color}" opacity="0.86">'
            f'<title>{html.escape(title)}</title></rect>'
        )
        if label and w >= 58:
            parts.append(
                f'<text x="{x1 + w / 2:.1f}" y="{y + 10:.1f}" text-anchor="middle" font-size="8" '
                f'font-weight="800" fill="#ffffff">{html.escape(label)}</text>'
            )

    zero_x = x_pos(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Client dispatch KV movement timeline">',
        '<text x="12" y="28" font-size="18" font-weight="800" fill="#0f172a">All KV movement during client dispatch</text>',
        '<text x="12" y="50" font-size="12" fill="#475569">Each row shows exact SGLang-visible KV movement while the target replay was waiting to reach SGLang.</text>',
        f'<line x1="{zero_x:.1f}" y1="{top - 28}" x2="{zero_x:.1f}" y2="{height - 58}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{zero_x + 4:.1f}" y="{top - 36}" font-size="11" font-weight="800" fill="#111827">0 ms replay due</text>',
    ]
    ticks = [-100.0, -10.0, 0.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 30000.0, 60000.0, 120000.0]
    if all(abs(x_max - tick) > 1 for tick in ticks):
        ticks.append(x_max)
    for tick in sorted(tick for tick in ticks if x_min <= tick <= x_max):
        x = x_pos(tick)
        label = f"{int(tick)} ms" if abs(tick) < 1000 else f"{tick / 1000:.0f} s"
        parts.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 62}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{top - 24}" text-anchor="middle" font-size="10" fill="#475569">{html.escape(label)}</text>')

    lane_names = [("H2D", "H2D"), ("D2H", "D2H"), ("GPU evict", "evict")]
    for idx, row in enumerate(selected):
        y = top + idx * row_h
        band = "#ffffff" if idx % 2 == 0 else "#eef4fb"
        row_id = str(row.get("row") or f"C{idx:02d}")
        dispatch_ms = as_float(row.get("dispatch_window_ms")) or 0.0
        parts.append(f'<rect x="0" y="{y - 6:.1f}" width="{width}" height="{row_h - 10}" fill="{band}"/>')
        parts.append(f'<text x="12" y="{y + 15:.1f}" font-size="15" font-weight="800" fill="#0f172a">{html.escape(row_id)}</text>')
        parts.append(f'<text x="12" y="{y + 34:.1f}" font-size="11" font-weight="800" fill="#334155">{html.escape(str(row.get("verdict") or ""))}</text>')
        parts.append(f'<text x="12" y="{y + 53:.1f}" font-size="10" fill="#475569">dispatch {html.escape(display_ms(dispatch_ms))}; fillers {html.escape(str(row.get("fillers") or ""))}</text>')
        parts.append(f'<text x="12" y="{y + 71:.1f}" font-size="10" fill="#475569">H2D {row.get("h2d_events", 0)} | D2H {row.get("d2h_events", 0)} | evict {row.get("gpu_evict_events", 0)}</text>')

        dispatch_x1 = x_pos(0.0)
        dispatch_x2 = x_pos(dispatch_ms)
        dispatch_w = max(3.0, dispatch_x2 - dispatch_x1)
        parts.append(
            f'<rect x="{dispatch_x1:.1f}" y="{y + 5:.1f}" width="{dispatch_w:.1f}" height="76" rx="5" '
            f'fill="#dbeafe" opacity="0.38"><title>Client dispatch window: replay due to {html.escape(str(row.get("client_dispatch_window") or ""))}; {dispatch_ms:.3f} ms</title></rect>'
        )
        for lane_idx, (kind, lane_label) in enumerate(lane_names):
            lane_y = y + 12 + lane_idx * 23
            parts.append(f'<text x="{left - 10}" y="{lane_y + 10:.1f}" text-anchor="end" font-size="10" font-weight="700" fill="#334155">{lane_label}</text>')
            parts.append(f'<line x1="{left}" y1="{lane_y + 7:.1f}" x2="{left + plot_w}" y2="{lane_y + 7:.1f}" stroke="#dbe4ee"/>')

        for event in events_by_row.get(row_id, []):
            start = as_float(event.get("start_relative_to_replay_due_ms"))
            end = as_float(event.get("end_relative_to_replay_due_ms"))
            if start is None or end is None:
                continue
            kind = str(event.get("movement_kind") or "KV movement")
            lane_index = 0 if kind == "H2D" else 1 if kind == "D2H" else 2
            lane_y = y + 12 + lane_index * 23
            color = client_dispatch_movement_color(kind)
            owner = str(event.get("owner_kind") or "")
            label = "filler" if owner == "pressure/filler" else "target" if owner == "target replay" else ""
            title = (
                f"{row_id} | {kind} | owner={owner} | session={event.get('owner_session_id', '')} | "
                f"source={event.get('source_event', '')} | start={start:.3f} ms | end={end:.3f} ms | "
                f"duration={event.get('duration_ms', '')} ms | tokens/indices={event.get('token_or_index_count', '')}"
            )
            draw_bar(parts, start, end, lane_y, color, title, label)

    legend_y = height - 42
    lx = left
    legend = [
        ("client dispatch window", "#dbeafe"),
        ("H2D", client_dispatch_movement_color("H2D")),
        ("D2H/write host", client_dispatch_movement_color("D2H")),
        ("GPU evict", client_dispatch_movement_color("GPU evict")),
    ]
    for label, color in legend:
        parts.append(f'<rect x="{lx:.1f}" y="{legend_y:.1f}" width="15" height="15" rx="3" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{lx + 22:.1f}" y="{legend_y + 12:.1f}" font-size="12" fill="#334155">{html.escape(label)}</text>')
        lx += 230
    parts.append("</svg>")
    return "\n".join(parts)


def client_dispatch_kv_movement_html(
    summary_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    max_rows: int = 10,
) -> str:
    if not summary_rows:
        return """
        <p>No client-dispatch windows were available for this report.</p>
        <p class="note">This usually means the run did not include replay due timestamps and request-stage timestamps.</p>
        """
    windows_with_movement = [row for row in summary_rows if int(as_float(row.get("all_kv_events")) or 0) > 0]
    h2d_windows = [row for row in summary_rows if int(as_float(row.get("h2d_events")) or 0) > 0]
    d2h_windows = [row for row in summary_rows if int(as_float(row.get("d2h_events")) or 0) > 0]
    evict_windows = [row for row in summary_rows if int(as_float(row.get("gpu_evict_events")) or 0) > 0]
    total_events = sum(int(as_float(row.get("all_kv_events")) or 0) for row in summary_rows)
    pressure_events = sum(int(as_float(row.get("pressure_filler_events")) or 0) for row in summary_rows)
    cards = [
        ("dispatch windows", len(summary_rows)),
        ("windows with KV movement", len(windows_with_movement)),
        ("windows with H2D", len(h2d_windows)),
        ("windows with D2H", len(d2h_windows)),
        ("windows with GPU evict", len(evict_windows)),
        ("pressure/filler events", pressure_events),
        ("all KV events in dispatch", total_events),
    ]
    cards_html = "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{html.escape(str(label))}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
        for label, value in cards
    ) + "</div>"
    return f"""
    <p>This section checks the long <strong>client/workload dispatch</strong> interval. It asks: while the target replay was waiting to reach SGLang, were other sessions already moving, backing up, or evicting KV?</p>
    <p class="note">This uses exact SGLang KV movement hooks aligned onto each controlled case clock. It includes target rows and pressure/filler sessions. It is still SGLang-visible movement, not a full PCIe/CUPTI hardware trace.</p>
    {cards_html}
    <h3>Client Dispatch KV Movement Timeline</h3>
    <p>The blue background is the target replay's dispatch window. Colored bars show exact KV movement from any session in the same controlled case during that window.</p>
    <div class="setup-diagram">{build_client_dispatch_kv_movement_svg(summary_rows, event_rows, max_rows=max_rows)}</div>
    <p class="note">The full aligned event rows and per-gap summaries are in <strong>Evidence Tables / Raw Proof</strong> at the bottom of the report.</p>
    """


def h2d_symlog_value(value: float, linear_width: float = 50.0) -> float:
    if value == 0:
        return 0.0
    return math.copysign(math.log1p(abs(value) / linear_width), value)


def h2d_symlog_tick_values(min_margin: float, max_margin: float) -> list[float]:
    candidates = [
        -100000.0,
        -50000.0,
        -10000.0,
        -5000.0,
        -1000.0,
        -500.0,
        -100.0,
        -50.0,
        0.0,
        50.0,
        100.0,
        500.0,
        1000.0,
        5000.0,
        10000.0,
    ]
    ticks = [value for value in candidates if min_margin <= value <= max_margin]
    for value in (min_margin, max_margin):
        if all(abs(value - tick) > 1 for tick in ticks):
            ticks.append(value)
    return sorted(ticks)


def filler_palette(rows: list[dict[str, Any]]) -> dict[str, str]:
    palette = [
        "#1d4ed8",  # blue
        "#f97316",  # orange
        "#16a34a",  # green
        "#db2777",  # magenta
        "#0891b2",  # cyan
        "#dc2626",  # red
        "#ca8a04",  # yellow-brown
        "#334155",  # slate
    ]
    fillers = sorted({str(row.get("fillers") or "unknown") for row in rows}, key=lambda value: (as_float(value) is None, as_float(value) or 0.0, value))
    return {filler: palette[idx % len(palette)] for idx, filler in enumerate(fillers)}


def build_replay_h2d_readiness_dot_plot(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No no-prefetch replay-side H2D rows were available for the readiness plot.</p>"
    width = 1480
    height = 540
    left = 96
    right = 40
    top = 64
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    margins = [float(row["h2d_finish_margin_ms"]) for row in rows]
    min_margin = min(margins)
    max_margin = max(margins)
    pad = max(50.0, (max_margin - min_margin) * 0.08)
    y_min = min(min_margin - pad, -50.0)
    y_max = max(max_margin + pad, 50.0)
    scaled_min = h2d_symlog_value(y_min)
    scaled_max = h2d_symlog_value(y_max)
    colors = filler_palette(rows)

    def x_pos(index: int) -> float:
        if len(rows) <= 1:
            return left + plot_w / 2
        return left + index * plot_w / (len(rows) - 1)

    def y_pos(value: float) -> float:
        scaled = h2d_symlog_value(value)
        return top + (scaled_max - scaled) * plot_h / (scaled_max - scaled_min)

    zero_y = y_pos(0.0)
    parts = [
        '<svg viewBox="0 0 1480 540" width="100%" role="img" aria-label="Global replay H2D readiness dot plot">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{left + plot_w - 8}" y="{zero_y - 8:.1f}" text-anchor="end" font-size="12" font-weight="700">0 ms replay due</text>',
        '<text x="20" y="280" transform="rotate(-90 20 280)" text-anchor="middle" font-size="13" font-weight="700">H2D finish margin ms (symlog)</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="13" font-weight="700">no-prefetch replay gap order</text>',
        '<text x="104" y="36" font-size="13" fill="#166534" font-weight="700">above line = replay-side KV H2D finished before due</text>',
        '<text x="470" y="36" font-size="13" fill="#b91c1c" font-weight="700">below line = replay waited for H2D after due</text>',
    ]

    seen_ticks: set[int] = set()
    for value in h2d_symlog_tick_values(y_min, y_max):
        rounded = int(round(value))
        if rounded in seen_ticks:
            continue
        seen_ticks.add(rounded)
        y = y_pos(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{rounded} ms</text>')

    x_tick_step = max(1, len(rows) // 10)
    for index in range(0, len(rows), x_tick_step):
        x = x_pos(index)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="10">{index}</text>')

    for index, row in enumerate(rows):
        margin = float(row["h2d_finish_margin_ms"])
        duration = as_float(row.get("h2d_visible_wall_window_ms"))
        radius = max(4.5, min(10.0, 4.5 + (duration or 0.0) / 120.0))
        filler = str(row.get("fillers") or "unknown")
        color = colors.get(filler, "#64748b")
        stroke = "#166534" if margin >= 0 else "#991b1b"
        x = x_pos(index)
        y = y_pos(margin)
        title = (
            f"{row.get('session_id')} | fillers={filler} | tool_gap={row.get('tool_gap_ms')} ms | "
            f"H2D finish margin={margin:.3f} ms | due->H2D start={row.get('replay_due_to_h2d_start_ms')} ms | "
            f"H2D wall={row.get('h2d_visible_wall_window_ms')} ms | events={row.get('replay_h2d_events')} | "
            f"TTFT={row.get('resume_ttft_ms')} ms"
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" opacity="0.86" stroke="{stroke}" stroke-width="2">'
            f'<title>{html.escape(title)}</title></circle>'
        )

    lx = left
    ly = height - 62
    for filler, color in colors.items():
        parts.append(f'<circle cx="{lx}" cy="{ly}" r="6" fill="{color}" stroke="#334155" stroke-width="1"/>')
        parts.append(f'<text x="{lx + 12}" y="{ly + 4}" font-size="12">fillers {html.escape(filler)}</text>')
        lx += 140
    parts.append("</svg>")
    return "\n".join(parts)


def build_replay_request_vs_h2d_timeline_plot(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No no-prefetch replay-side H2D rows were available for the request-vs-H2D plot.</p>"
    width = 1480
    height = 690
    left = 130
    right = 50
    top = 108
    bottom = 166
    plot_w = width - left - right
    plot_h = height - top - bottom
    marker_group_pad = 72
    marker_plot_left = left + marker_group_pad
    marker_plot_w = plot_w - 2 * marker_group_pad
    numeric_keys = [
        "replay_due_to_client_submit_ms",
        "replay_due_to_request_start_ms",
        "replay_due_to_sglang_receive_ms",
        "replay_due_to_scheduler_queue_ms",
        "replay_due_to_scheduler_admit_ms",
        "replay_due_to_h2d_start_ms",
        "replay_due_to_h2d_end_ms",
    ]
    values: list[float] = []
    for row in rows:
        for key in numeric_keys:
            value = as_float(row.get(key))
            if value is not None:
                values.append(value)
    if not values:
        return "<p>No request/H2D timing values were available for the request-vs-H2D plot.</p>"
    min_value = min(values)
    max_value = max(values)
    pad = max(50.0, (max_value - min_value) * 0.08)
    y_min = min(min_value - pad, -50.0)
    y_max = max(max_value + pad, 50.0)
    scaled_min = h2d_symlog_value(y_min)
    scaled_max = h2d_symlog_value(y_max)

    def x_pos(index: int) -> float:
        if len(rows) <= 1:
            return marker_plot_left + marker_plot_w / 2
        return marker_plot_left + index * marker_plot_w / (len(rows) - 1)

    def y_pos(value: float) -> float:
        scaled = h2d_symlog_value(value)
        return top + (scaled - scaled_min) * plot_h / (scaled_max - scaled_min)

    def circle(x: float, y: float, color: str, title: str) -> str:
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" opacity="0.95" '
            f'stroke="#ffffff" stroke-width="2"><title>{html.escape(title)}</title></circle>'
        )

    def square(x: float, y: float, color: str, title: str) -> str:
        return (
            f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" rx="2" fill="{color}" '
            f'opacity="0.95" stroke="#ffffff" stroke-width="2"><title>{html.escape(title)}</title></rect>'
        )

    def triangle(x: float, y: float, color: str, title: str) -> str:
        points = f"{x:.1f},{y - 9:.1f} {x - 9:.1f},{y + 8:.1f} {x + 9:.1f},{y + 8:.1f}"
        return (
            f'<polygon points="{points}" fill="{color}" opacity="0.95" stroke="#ffffff" '
            f'stroke-width="2"><title>{html.escape(title)}</title></polygon>'
        )

    def diamond(x: float, y: float, color: str, title: str) -> str:
        points = f"{x:.1f},{y - 8:.1f} {x + 8:.1f},{y:.1f} {x:.1f},{y + 8:.1f} {x - 8:.1f},{y:.1f}"
        return (
            f'<polygon points="{points}" fill="{color}" opacity="0.95" stroke="#ffffff" '
            f'stroke-width="2"><title>{html.escape(title)}</title></polygon>'
        )

    zero_y = y_pos(0.0)
    parts = [
        '<svg viewBox="0 0 1480 690" width="100%" role="img" aria-label="Replay request versus H2D start timeline plot">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{left + plot_w - 10}" y="{max(top + 16, zero_y - 10):.1f}" text-anchor="end" font-size="12" font-weight="700">0 ms replay due</text>',
        '<text x="30" y="316" transform="rotate(-90 30 316)" text-anchor="middle" font-size="13" font-weight="700">time relative to replay due (symlog)</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 50}" text-anchor="middle" font-size="13" font-weight="700">no-prefetch replay gap order</text>',
        '<text x="130" y="42" font-size="13" fill="#334155" font-weight="700">top = before replay due; bottom = after replay due</text>',
        '<text x="130" y="64" font-size="12" fill="#64748b">Marker groups are spread sideways per gap so stages do not cover each other. Gray lines connect the stages in request order.</text>',
    ]

    seen_ticks: set[int] = set()
    labeled_tick_ys: list[float] = []
    for value in h2d_symlog_tick_values(y_min, y_max):
        rounded = int(round(value))
        if rounded in seen_ticks:
            continue
        seen_ticks.add(rounded)
        y = y_pos(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        if rounded == 0:
            continue
        if any(abs(y - prev_y) < 18 for prev_y in labeled_tick_ys):
            continue
        labeled_tick_ys.append(y)
        parts.append(f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#334155">{rounded} ms</text>')

    x_tick_step = max(1, len(rows) // 10)
    for index in range(0, len(rows), x_tick_step):
        x = x_pos(index)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="10" fill="#475569">{index}</text>')

    for index, row in enumerate(rows):
        x = x_pos(index)
        title_prefix = (
            f"{row.get('session_id')} | fillers={row.get('fillers')} | tool_gap={row.get('tool_gap_ms')} ms | "
            f"TTFT={row.get('resume_ttft_ms')} ms"
        )
        marker_specs = [
            ("client submit", "replay_due_to_client_submit_ms", -42, "#7c3aed", "diamond"),
            ("client request call start", "replay_due_to_request_start_ms", -28, "#2563eb", "circle"),
            ("SGLang receive", "replay_due_to_sglang_receive_ms", -14, "#f97316", "diamond"),
            ("scheduler queue", "replay_due_to_scheduler_queue_ms", 0, "#ca8a04", "square"),
            ("scheduler admit", "replay_due_to_scheduler_admit_ms", 14, "#db2777", "triangle"),
            ("H2D start", "replay_due_to_h2d_start_ms", 28, "#0f766e", "triangle"),
            ("H2D finish", "replay_due_to_h2d_end_ms", 42, "#06b6d4", "square"),
        ]
        visible_points: list[tuple[float, float, str]] = []
        connectors: list[str] = []
        markers: list[str] = []
        for label, key, offset, color, shape in marker_specs:
            value = as_float(row.get(key))
            if value is None:
                continue
            marker_x = x + offset
            marker_y = y_pos(value)
            visible_points.append((marker_x, marker_y, color))
            timing = f"{value:.3f} ms after due" if value >= 0 else f"{abs(value):.3f} ms before due"
            title = f"{title_prefix} | {label}={timing}"
            if shape == "circle":
                markers.append(circle(marker_x, marker_y, color, title))
            elif shape == "triangle":
                markers.append(triangle(marker_x, marker_y, color, title))
            elif shape == "square":
                markers.append(square(marker_x, marker_y, color, title))
            else:
                markers.append(diamond(marker_x, marker_y, color, title))
        for (x1, y1, _), (x2, y2, _) in zip(visible_points, visible_points[1:]):
            connectors.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="1.5" opacity="0.55"/>',
            )
        parts.extend(connectors)
        parts.extend(markers)

    legend = [
        ("client submit", "#7c3aed", "diamond"),
        ("request call start", "#2563eb", "circle"),
        ("SGLang receive", "#f97316", "diamond"),
        ("scheduler queue", "#ca8a04", "square"),
        ("scheduler admit", "#db2777", "triangle"),
        ("H2D start", "#0f766e", "triangle"),
        ("H2D finish", "#06b6d4", "square"),
    ]
    lx = left
    ly = height - 104
    for label, color, kind in legend:
        if kind == "circle":
            parts.append(f'<circle cx="{lx}" cy="{ly}" r="6" fill="{color}"/>')
        elif kind == "triangle":
            points = f"{lx},{ly - 7} {lx - 7},{ly + 6} {lx + 7},{ly + 6}"
            parts.append(f'<polygon points="{points}" fill="{color}"/>')
        elif kind == "square":
            parts.append(f'<rect x="{lx - 6}" y="{ly - 6}" width="12" height="12" rx="2" fill="{color}"/>')
        elif kind == "diamond":
            points = f"{lx},{ly - 7} {lx + 7},{ly} {lx},{ly + 7} {lx - 7},{ly}"
            parts.append(f'<polygon points="{points}" fill="{color}"/>')
        else:
            parts.append(f'<line x1="{lx - 8}" y1="{ly}" x2="{lx + 8}" y2="{ly}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{lx + 14}" y="{ly + 4}" font-size="12">{html.escape(label)}</text>')
        lx += 188
    parts.append(f'<rect x="{left}" y="{height - 78}" width="{plot_w}" height="44" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>')
    parts.append(f'<text x="{left + 14}" y="{height - 51}" font-size="12" fill="#475569">Read one gap left to right: client submits replay, Python request begins, SGLang receives it, scheduler queues/admits it, then visible KV H2D starts and finishes.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def global_replay_h2d_readiness_html(gaps: list[dict[str, Any]]) -> str:
    h2d_rows = replay_h2d_readiness_rows(gaps)
    queue_rows = replay_queue_timing_rows(gaps)
    if not h2d_rows and not queue_rows:
        return """
        <p>No no-prefetch replay rows were available for queue/H2D timing in this report.</p>
        """
    summary = replay_h2d_readiness_summary(h2d_rows) if h2d_rows else []
    buckets = replay_h2d_readiness_bucket_rows(h2d_rows) if h2d_rows else []
    detail = queue_rows[:80]
    summary_cards = ""
    if summary:
        summary_cards = "<div class=\"cards\">" + "\n".join(
            f"<div class=\"card\"><div class=\"label\">{html.escape(str(label))}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
            for label, value in [
                ("H2D rows", summary[0].get("h2d_rows", "")),
                ("finished before due", summary[0].get("finished_before_due", "")),
                ("late H2D rows", summary[0].get("late_h2d_rows", "")),
                ("median finish margin", f"{summary[0].get('median_finish_margin_ms', '')} ms"),
                ("worst lateness", f"{summary[0].get('worst_lateness_ms', '')} ms"),
            ]
        ) + "</div>"
    h2d_plot_html = (
        f"""
    <h3>Replay H2D Readiness Dot Plot</h3>
    <div class="setup-diagram">{build_replay_h2d_readiness_dot_plot(h2d_rows)}</div>
        """
        if h2d_rows
        else """
    <h3>Replay H2D Readiness Dot Plot</h3>
    <p>No replay-side H2D movement was observed in these no-prefetch rows. The queue timeline above still shows replay submission and scheduler timing where available.</p>
        """
    )
    return f"""
    <p>This no-prefetch chart answers a different question from prefetch margin: when replay needed KV, how late did the replay-side host-to-device KV load finish?</p>
    <p class="note">The dot value is <code>replay_due_time - replay_h2d_finish_time</code>. Positive means the KV load finished before the replay deadline. Negative means the replay deadline passed first, so the model turn had to wait for KV readiness.</p>
    <p class="note">The timing is split into concrete queue stages: replay due, client submit, SGLang receive, scheduler queue, scheduler admit, H2D start, and H2D finish. This separates queue delay from the visible host-to-device movement window.</p>
    {summary_cards}
    <h3>Replay Queue Timeline vs H2D Start</h3>
    <p>This chart checks whether the replay request was submitted late, queued inside SGLang, delayed before scheduler admission, or delayed before visible KV H2D movement began.</p>
    <p class="note">Each gap has stage markers drawn side-by-side so they do not hide each other. Their vertical position still shows the real timing relative to replay due. Missing markers mean that stage was not present in the trace, usually because the run was generated before scheduler tracing was enabled.</p>
    <div class="setup-diagram">{build_replay_request_vs_h2d_timeline_plot(queue_rows)}</div>
    {h2d_plot_html}
    <p class="note">Readiness buckets and exact queue timing rows are in <strong>Evidence Tables / Raw Proof</strong> at the bottom of the report.</p>
    """


def global_readiness_section_title(gaps: list[dict[str, Any]]) -> str:
    has_prefetch_margins = any(as_float(row.get("prefetch_margin_ms")) is not None for row in gaps)
    has_no_prefetch_h2d = any(
        str(row.get("mode") or "") == "no_prefetch" and has_events(row.get("replay_kv_h2d_events"))
        for row in gaps
    )
    if has_no_prefetch_h2d and not has_prefetch_margins:
        return "Global Replay H2D Readiness"
    if has_no_prefetch_h2d:
        return "Global KV Readiness"
    return "Global Prefetch Margin"


def global_readiness_html(gaps: list[dict[str, Any]]) -> str:
    has_prefetch_margins = any(as_float(row.get("prefetch_margin_ms")) is not None for row in gaps)
    has_no_prefetch_h2d = any(
        str(row.get("mode") or "") == "no_prefetch" and has_events(row.get("replay_kv_h2d_events"))
        for row in gaps
    )
    sections: list[str] = []
    if has_no_prefetch_h2d:
        sections.append("<h3>No-Prefetch Replay H2D Readiness</h3>")
        sections.append(global_replay_h2d_readiness_html(gaps))
    if has_prefetch_margins:
        sections.append("<h3>Direct-Prefetch Margin</h3>")
        sections.append(live_global_prefetch_margin_html(gaps))
    if not sections:
        return "<p>No prefetch-margin or replay-side H2D readiness rows were available for this run.</p>"
    return "\n".join(sections)


def code_block(text: str) -> str:
    return f"<pre><code>{html.escape(text.strip())}</code></pre>"


def load_run_config_for_result_root(result_root: Path) -> dict[str, str]:
    candidates = [
        result_root / "run_config.env",
        result_root / "manifest.json",
    ]
    if len(result_root.parents) >= 3:
        results_root = result_root.parents[2]
        candidates.extend(
            [
                results_root / "reports" / result_root.name / "run_config.env",
                results_root / "reports" / result_root.name / "manifest.json",
            ]
        )

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".json":
            data = load_json(path)
            knobs = data.get("pressure_knobs", {}) if isinstance(data, dict) else {}
            config = {str(k).upper(): str(v) for k, v in knobs.items() if v is not None}
            if isinstance(data, dict):
                for key in ["report_label", "experiment_kind", "model", "pressure_profile", "workload_source"]:
                    value = data.get(key)
                    if value is not None:
                        config[key.upper()] = str(value)
            if config:
                return config
            continue

        config: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
        if config:
            return config
    return {}


def shell_value(value: str) -> str:
    if value == "":
        return '""'
    if any(ch.isspace() for ch in value) or any(ch in value for ch in ['"', "'", "$", "\\"]):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def command_block_lines(env_pairs: list[tuple[str, str]], model: str) -> str:
    lines = [
        "cd ~/agentic_hardware/sglang_direct_kv",
        "source .venv/bin/activate",
        "",
    ]
    lines.extend(f"{key}={shell_value(value)} \\" for key, value in env_pairs)
    lines.extend(
        [
            "bash scripts/run_master_report.sh \\",
            f"  {model}",
        ]
    )
    return "\n".join(lines)


def reproduce_controlled_replay_html(result_root: Path) -> str:
    run_config = load_run_config_for_result_root(result_root)
    label = run_config.get("REPORT_LABEL") or result_root.name or "controlled_demo_1"
    model = run_config.get("MODEL") or "Qwen/Qwen2.5-Coder-7B-Instruct"
    trace_index = run_config.get("TRACE_INDEX_CSV") or "~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv"
    if trace_index.startswith("/home/ec2-user/"):
        trace_index = "~/" + trace_index.removeprefix("/home/ec2-user/")

    run_master = command_block_lines(
        [
            ("AGENTIC_KV_TRACE_SCHEDULER", run_config.get("AGENTIC_KV_TRACE_SCHEDULER") or "1"),
            ("EXPERIMENT_KIND", run_config.get("EXPERIMENT_KIND") or "controlled"),
            ("REPORT_LABEL", label),
            ("PRESSURE_PROFILE", run_config.get("PRESSURE_PROFILE") or "custom"),
            ("UPDATE_LATEST", run_config.get("UPDATE_LATEST") or "1"),
            ("MAX_TIMELINE_GAPS", run_config.get("MAX_TIMELINE_GAPS") or "24"),
            ("MAX_PAIRS", run_config.get("MAX_PAIRS") or "2"),
            ("MODES", run_config.get("MODES") or "no_prefetch"),
            ("TOOL_WAIT_LIST_MS", run_config.get("TOOL_WAIT_LIST_MS") or "500"),
            ("FILLER_LIST", run_config.get("FILLER_LIST") or "8 12 16 24 32"),
            ("REQUEST_CONCURRENCY", run_config.get("REQUEST_CONCURRENCY") or "4"),
            ("FILLER_PROMPT_TOKENS", run_config.get("FILLER_PROMPT_TOKENS") or "1024"),
            ("MAX_TOTAL_TOKENS", run_config.get("MAX_TOTAL_TOKENS") or "12288"),
            ("HICACHE_SIZE_GB", run_config.get("HICACHE_SIZE_GB") or "16"),
            ("MEM_FRACTION_STATIC", run_config.get("MEM_FRACTION_STATIC") or "0.72"),
            ("TRACE_INDEX_CSV", trace_index),
        ],
        model,
    )
    build_only = command_block_lines(
        [
            ("BUILD_ONLY", "1"),
            ("EXPERIMENT_KIND", "controlled"),
            ("REPORT_LABEL", f"{label}_rebuild"),
            ("UPDATE_LATEST", "0"),
            ("CONTROLLED_ROOT", f"artifacts/results/runs/controlled/{label}"),
        ],
        model,
    )
    dry_run = command_block_lines(
        [
            ("DRY_RUN", "1"),
            ("EXPERIMENT_KIND", "controlled"),
            ("REPORT_LABEL", label),
        ],
        model,
    )
    knob_rows = [
        {"knob": "EXPERIMENT_KIND", "meaning": "controlled, live, or both"},
        {"knob": "PRESSURE_PROFILE", "meaning": "custom, low, medium, high, or extreme"},
        {"knob": "REPORT_LABEL", "meaning": "folder name for this run under artifacts/results/reports/"},
        {"knob": "UPDATE_LATEST", "meaning": "1 replaces artifacts/results/latest_master_report.html"},
        {"knob": "BUILD_ONLY", "meaning": "1 rebuilds HTML from existing run folders"},
        {"knob": "DRY_RUN", "meaning": "1 prints what would run without launching SGLang"},
        {"knob": "MAX_TIMELINE_GAPS", "meaning": "number of rows shown in each timeline"},
        {"knob": "START_INDEX / END_INDEX", "meaning": "AgentBench task range for live runs"},
        {"knob": "WORKLOAD_JSONL / TRACE_INDEX_CSV", "meaning": "real prompt-pair source for controlled replay"},
    ]
    pressure_rows = [
        {"profile": "low", "controlled pressure": "4 prompt pairs, fillers 8/16, concurrency 2", "live pressure": "tasks 0-3, max steps 6"},
        {"profile": "medium", "controlled pressure": "8 prompt pairs, fillers 16/32, concurrency 4", "live pressure": "tasks 0-15, max steps 10"},
        {"profile": "high", "controlled pressure": "16 prompt pairs, fillers 32/64/128, concurrency 8", "live pressure": "tasks 0-31, max steps 10"},
        {"profile": "extreme", "controlled pressure": "24 prompt pairs, fillers 64/128/192, concurrency 12", "live pressure": "tasks 0-63, max steps 15"},
        {"profile": "custom", "controlled pressure": "use only the knobs you explicitly set", "live pressure": "use only the knobs you explicitly set"},
    ]
    return "\n".join(
        [
            "<p>Use one script to generate this master report. It can run the controlled replay experiment, the live AgentBench direct-prefetch experiment, or both.</p>",
            "<h3>Run The Full Master Report</h3>",
            code_block(run_master),
            "<p>Output:</p>",
            code_block(f"artifacts/results/latest_master_report.html\nartifacts/results/reports/{label}/master_report.html\nartifacts/results/latest_manifest.json"),
            "<h3>Rebuild From Existing Runs</h3>",
            "<p>Use this when experiments already ran and you only want to regenerate the HTML, tables, timelines, and global prefetch-margin dot charts.</p>",
            code_block(build_only),
            "<h3>Preview Before Running</h3>",
            code_block(dry_run),
            "<h3>Important Knobs</h3>",
            table_html(knob_rows, ["knob", "meaning"]),
            "<h3>Pressure Profiles</h3>",
            "<p>Use <code>PRESSURE_PROFILE</code> when you want to deliberately increase cache pressure and request pressure. Manual environment variables still override the profile defaults.</p>",
            table_html(pressure_rows, ["profile", "controlled pressure", "live pressure"]),
        ]
    )


def _rows_matching_label(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("row") or "") == label]


def _first_matching_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    matching = _rows_matching_label(rows, label)
    return matching[0] if matching else {}


def _card_value(value: Any, suffix: str = "") -> str:
    if value in ("", None):
        return "not observed"
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return f"{text}{suffix}"


def _delay_summary_text(delay_row: dict[str, Any]) -> str:
    if not delay_row:
        return "No replay delay breakdown was available for this row."
    verdict = str(delay_row.get("copy_verdict") or "delay path observed")
    source = str(delay_row.get("main_delay_source") or "unknown delay source")
    simple = str(delay_row.get("simple_meaning") or "")
    if simple:
        return f"{verdict}: {source}. {simple}"
    return f"{verdict}: {source}."


def _dispatch_summary_text(dispatch_row: dict[str, Any]) -> str:
    if not dispatch_row:
        return "No client-dispatch window was available for this row."
    verdict = str(dispatch_row.get("verdict") or "dispatch window observed")
    events = dispatch_row.get("all_kv_events", "")
    h2d = dispatch_row.get("h2d_events", "")
    d2h = dispatch_row.get("d2h_events", "")
    evict = dispatch_row.get("gpu_evict_events", "")
    return f"{verdict}. During dispatch: {events} KV events, {h2d} H2D, {d2h} D2H, {evict} GPU evictions."


def _block_summary_text(block_rows: list[dict[str, Any]]) -> str:
    if not block_rows:
        return "No block-ledger rows were available for this gap."
    verdicts = Counter(str(row.get("lifecycle_verdict") or "unknown") for row in block_rows)
    pieces = [f"{count} {verdict}" for verdict, count in sorted(verdicts.items())]
    exact = sum(1 for row in block_rows if str(row.get("evidence_level") or "").startswith("exact"))
    return f"{len(block_rows)} logical KV blocks tracked; " + ", ".join(pieces) + f"; {exact} exact-index rows."


def _forensic_evidence_table_rows(
    gap: dict[str, Any],
    delay_row: dict[str, Any],
    queue_row: dict[str, Any],
    dispatch_row: dict[str, Any],
    block_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    label = str(gap.get("timeline_label") or "")
    return [
        {
            "evidence": "gap setup",
            "what it says": (
                f"{label} is mode={gap.get('mode', '')}, fillers={case_fillers(gap)}, "
                f"tool wait={gap.get('tool_gap_ms', '')} ms, task={gap.get('task_index', '')}, "
                f"gap={gap.get('gap_order_in_task', '')}."
            ),
        },
        {
            "evidence": "replay path",
            "what it says": (
                f"path={gap.get('replay_path') or replay_path_from_evidence(gap)}; "
                f"final_path={gap.get('final_path', '')}; TTFT={gap.get('resume_ttft_ms', '')} ms."
            ),
        },
        {
            "evidence": "queue timing",
            "what it says": queue_row.get("simple_meaning", "No no-prefetch queue row was available."),
        },
        {
            "evidence": "delay breakdown",
            "what it says": _delay_summary_text(delay_row),
        },
        {
            "evidence": "client-dispatch KV movement",
            "what it says": _dispatch_summary_text(dispatch_row),
        },
        {
            "evidence": "block ledger",
            "what it says": _block_summary_text(block_rows),
        },
    ]


def per_gap_forensic_view_html(
    gaps: list[dict[str, Any]],
    kv_block_rows: list[dict[str, Any]],
    h2d_pressure_rows: list[dict[str, Any]],
    replay_delay_rows: list[dict[str, Any]],
    replay_queue_rows: list[dict[str, Any]],
    client_dispatch_summary_rows: list[dict[str, Any]],
    client_dispatch_event_rows: list[dict[str, Any]],
    exact_kv_movement_rows: list[dict[str, Any]],
    max_cases: int,
) -> str:
    if not gaps:
        return "<p>No timeline rows were available for per-gap forensics.</p>"

    cases: list[str] = []
    for gap in gaps[:max_cases]:
        label = str(gap.get("timeline_label") or f"G{len(cases):02d}")
        block_rows = detailed_kv_lifecycle_table_rows([gap], kv_block_rows, limit=8)
        delay_row = _first_matching_label(replay_delay_rows, label)
        queue_row = _first_matching_label(replay_queue_rows, label)
        dispatch_row = _first_matching_label(client_dispatch_summary_rows, label)
        dispatch_events = _rows_matching_label(client_dispatch_event_rows, label)
        pressure_rows = _rows_matching_label(h2d_pressure_rows, label)
        exact_rows = exact_movement_table_rows(exact_kv_movement_rows, [gap], limit=12)
        status, status_color = observation_status(gap)
        summary_cards = [
            ("mode", gap.get("mode", "")),
            ("fillers", case_fillers(gap)),
            ("tool wait", _card_value(gap.get("tool_gap_ms"), " ms")),
            ("TTFT", _card_value(gap.get("resume_ttft_ms"), " ms")),
            ("replay path", gap.get("replay_path") or replay_path_from_evidence(gap)),
            ("verdict", gap.get("per_gap_verdict") or per_gap_verdict(gap)),
        ]
        cards_html = "<div class=\"cards\">" + "\n".join(
            f"<div class=\"card\"><div class=\"label\">{html.escape(str(name))}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
            for name, value in summary_cards
        ) + "</div>"
        evidence_rows = _forensic_evidence_table_rows(gap, delay_row, queue_row, dispatch_row, block_rows)
        cases.append(
            f"""
      <details class="forensic-case">
        <summary><h3>{html.escape(label)}: <span style="color:{html.escape(status_color)}">{html.escape(status)}</span></h3></summary>
        {cards_html}
        <p class="note">This case file shows the same row across the main lifecycle timeline, queue timing, client-dispatch KV movement, and delay waterfall. Full proof rows stay in the appendix.</p>
        <h4>1. Readable KV Lifecycle Timeline</h4>
        <div class="setup-diagram">{build_local_timing_phase_timeline_svg([gap], 1, show_prefetch_legend=True, kv_block_lifecycle_rows=block_rows, h2d_pressure_rows=pressure_rows, show_block_lifecycle_strip=False, show_h2d_pressure_strip=False)}</div>
        <h4>2. Replay Queue Timeline vs H2D Start</h4>
        <div class="setup-diagram">{build_replay_request_vs_h2d_timeline_plot([queue_row] if queue_row else [])}</div>
        <h4>3. Client Dispatch KV Movement</h4>
        <div class="setup-diagram">{build_client_dispatch_kv_movement_svg([dispatch_row] if dispatch_row else [], dispatch_events, max_rows=1)}</div>
        <h4>4. Replay Delay Waterfall</h4>
        <div class="setup-diagram">{build_replay_delay_waterfall_svg([delay_row] if delay_row else [], max_rows=1)}</div>
        <h4>5. Short Evidence Summary</h4>
        {table_html(evidence_rows, ["evidence", "what it says"])}
        <h4>6. Sample Exact KV/Block Proof Rows</h4>
        {table_html(block_rows, ["row", "mode", "lifecycle_verdict", "block_id", "node_id", "token_range", "h2d_start_ms", "h2d_end_ms", "h2d_duration_ms", "exact_attribution", "evidence_level"], limit=8)}
        {table_html(exact_rows, ["row", "phase", "movement_kind", "direction", "copy_start_ms", "copy_end_ms", "duration_ms", "node_id", "source_event", "confidence", "evidence_level"], limit=12)}
      </details>
            """
        )

    return f"""
    <p>This section is a per-row case file. Open one row, for example <code>G00</code>, to see the same gap across all major views without jumping around the report.</p>
    <p class="note">Use this when presenting: start with the readable lifecycle timeline, then open the queue, dispatch, and waterfall views underneath to explain why the replay was ready, late, loaded from host, or recomputed.</p>
    {"".join(cases)}
    """


def _relative_ms(row: dict[str, Any], key: str, due: float) -> float | None:
    value = as_float(row.get(key))
    return value - due if value is not None else None


def _relative_span(
    row: dict[str, Any],
    start_key: str,
    end_key: str,
    due: float,
) -> tuple[float, float] | None:
    start = _relative_ms(row, start_key, due)
    end = _relative_ms(row, end_key, due)
    if start is None or end is None or end <= start:
        return None
    return (start, end)


def _event_relative_span(event: dict[str, Any], due: float) -> tuple[float, float] | None:
    start = as_float(event.get("aligned_start_ms"))
    end = as_float(event.get("aligned_end_ms"))
    if start is None or end is None or end <= start:
        return None
    return (start - due, end - due)


def unified_stack_kv_events_for_gap(
    gap: dict[str, Any],
    all_kv_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    due = as_float(gap.get("tool_gap_end_ms"))
    if due is None:
        return []
    case_id = str(gap.get("case_id") or "")
    window_start = due + min(-500.0, (_relative_ms(gap, "current_start_ms", due) or -500.0))
    window_end_candidates = [
        as_float(gap.get("resume_end_ms")),
        as_float(gap.get("replay_kv_h2d_end_ms")),
        as_float(gap.get("direct_kv_h2d_end_ms")),
        as_float(gap.get("resume_start_ms")),
    ]
    window_end = max(value for value in window_end_candidates if value is not None) if any(
        value is not None for value in window_end_candidates
    ) else due + 1000.0
    output: list[dict[str, Any]] = []
    for event in all_kv_events:
        if case_id and str(event.get("case_id") or "") != case_id:
            continue
        start = as_float(event.get("aligned_start_ms"))
        end = as_float(event.get("aligned_end_ms"))
        if start is None or end is None:
            continue
        if start <= window_end and end >= window_start:
            output.append(event)
    return output


def unified_stack_axis_values(
    gaps: list[dict[str, Any]],
    all_kv_events: list[dict[str, Any]],
) -> list[float]:
    values: list[float] = [0.0]
    for gap in gaps:
        due = as_float(gap.get("tool_gap_end_ms"))
        if due is None:
            continue
        for key in [
            "current_start_ms",
            "current_end_ms",
            "tool_gap_start_ms",
            "tool_gap_end_ms",
            "hint_submitted_ms",
            "prefetch_start_ms",
            "prefetch_end_ms",
            "resume_submitted_ms",
            "resume_start_ms",
            "resume_end_ms",
            "replay_sglang_receive_start_ms",
            "replay_sglang_receive_end_ms",
            "replay_scheduler_queue_enter_start_ms",
            "replay_scheduler_admit_start_ms",
            "replay_kv_h2d_start_ms",
            "replay_kv_h2d_end_ms",
            "replay_prefill_start_ms",
            "replay_prefill_end_ms",
            "replay_model_forward_start_ms",
            "replay_model_forward_end_ms",
        ]:
            value = _relative_ms(gap, key, due)
            if value is not None:
                values.append(value)
        token_time = first_token_ms(gap)
        if token_time is not None:
            values.append(token_time - due)
        for event in unified_stack_kv_events_for_gap(gap, all_kv_events):
            span = _event_relative_span(event, due)
            if span:
                values.extend(span)
    return values


def unified_stack_color(kind: str) -> str:
    return {
        "initial": "#2563eb",
        "tool_wait": "#cbd5e1",
        "deadline": "#111827",
        "client_dispatch": "#312e81",
        "sglang_receive": "#f59e0b",
        "scheduler": "#7c3aed",
        "load_path": "#7c3aed",
        "prefetch": "#a855f7",
        "hint_h2d": "#16a34a",
        "h2d": "#06b6d4",
        "d2h": "#f97316",
        "evict": "#334155",
        "host_evict": "#7f1d1d",
        "recompute": "#c026d3",
        "prefill": "#eab308",
        "decode": "#dc2626",
        "marker": "#475569",
    }.get(kind, "#64748b")


def build_unified_per_gap_stack_timeline_svg(
    gaps: list[dict[str, Any]],
    all_kv_events: list[dict[str, Any]],
    max_rows: int,
) -> str:
    rows = gaps[:max_rows]
    if not rows:
        return "<p>No rows were available for the unified forensic stack timeline.</p>"
    axis_values = unified_stack_axis_values(rows, all_kv_events)
    x_min = min(axis_values) if axis_values else -1000.0
    x_max = max(axis_values) if axis_values else 1000.0
    if x_min >= 0:
        x_min = -500.0
    if x_max <= 0:
        x_max = 1000.0
    pad = max(50.0, (x_max - x_min) * 0.03)
    x_min -= pad
    x_max += pad

    width = 1680
    left = 250
    right = 56
    top = 132
    row_h = 166
    bottom = 96
    plot_w = width - left - right
    height = top + len(rows) * row_h + bottom
    scaled_min = h2d_symlog_value(x_min)
    scaled_max = h2d_symlog_value(x_max)

    def x_pos(value: float) -> float:
        scaled = h2d_symlog_value(value)
        return left + (scaled - scaled_min) * plot_w / max(1e-9, scaled_max - scaled_min)

    def draw_span(
        parts: list[str],
        start: float,
        end: float,
        y: float,
        h: float,
        color: str,
        label: str,
        title: str,
        opacity: float = 0.86,
    ) -> None:
        clipped_start = max(x_min, min(x_max, start))
        clipped_end = max(x_min, min(x_max, end))
        if clipped_end <= clipped_start:
            return
        x1 = x_pos(clipped_start)
        x2 = x_pos(clipped_end)
        w = max(3.0, x2 - x1)
        parts.append(
            f'<rect x="{x1:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{color}" opacity="{opacity}">'
            f'<title>{html.escape(title)}</title></rect>'
        )
        if label and w >= 68:
            parts.append(
                f'<text x="{x1 + w / 2:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" font-size="9" '
                f'font-weight="800" fill="#0f172a">{html.escape(label)}</text>'
            )

    def draw_marker(parts: list[str], value: float, y1: float, y2: float, color: str, title: str) -> None:
        if value < x_min or value > x_max:
            return
        x = x_pos(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2.2">'
            f'<title>{html.escape(title)}</title></line>'
        )

    zero_x = x_pos(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Unified per-gap forensic stack timeline">',
        '<text x="12" y="30" font-size="20" font-weight="800" fill="#0f172a">Unified per-gap forensic stack timeline</text>',
        '<text x="12" y="54" font-size="12" fill="#475569">All lanes use one shared replay-relative time axis. Negative means before replay due; positive means after replay due.</text>',
        f'<line x1="{zero_x:.1f}" y1="{top - 30}" x2="{zero_x:.1f}" y2="{height - 68}" stroke="#111827" stroke-width="2.5"/>',
        f'<text x="{zero_x + 6:.1f}" y="{top - 40}" font-size="12" font-weight="800">0 ms replay due</text>',
    ]
    ticks = h2d_symlog_tick_values(x_min, x_max)
    for value in ticks:
        x = x_pos(value)
        label = f"{int(value)} ms" if abs(value) < 1000 else f"{value / 1000:.0f} s"
        parts.append(f'<line x1="{x:.1f}" y1="{top - 22}" x2="{x:.1f}" y2="{height - 70}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{top - 28}" text-anchor="middle" font-size="10" fill="#475569">{html.escape(label)}</text>')

    lane_offsets = {
        "flow": 16,
        "request": 46,
        "kv": 78,
        "replay": 112,
        "verdict": 142,
    }
    lane_names = [
        ("flow", "flow"),
        ("request", "request"),
        ("kv", "KV move"),
        ("replay", "replay"),
        ("verdict", "verdict"),
    ]

    for idx, row in enumerate(rows):
        due = as_float(row.get("tool_gap_end_ms"))
        if due is None:
            continue
        y = top + idx * row_h
        band = "#ffffff" if idx % 2 == 0 else "#eef4fb"
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        status, status_color = observation_status(row)
        parts.append(f'<rect x="0" y="{y - 8:.1f}" width="{width}" height="{row_h - 10}" fill="{band}"/>')
        parts.append(f'<text x="12" y="{y + 14:.1f}" font-size="15" font-weight="800">{html.escape(label)}</text>')
        parts.append(f'<text x="12" y="{y + 34:.1f}" font-size="10" font-weight="800" fill="{status_color}">{html.escape(str(row.get("per_gap_verdict") or status))}</text>')
        parts.append(f'<text x="12" y="{y + 52:.1f}" font-size="10" fill="#475569">mode {html.escape(str(row.get("mode") or ""))}; fillers {html.escape(case_fillers(row))}; wait {html.escape(str(row.get("tool_gap_ms") or ""))} ms</text>')
        parts.append(f'<text x="12" y="{y + 70:.1f}" font-size="10" fill="#475569">{html.escape(str(row.get("replay_path") or replay_path_from_evidence(row)))}</text>')
        for lane_key, lane_label in lane_names:
            lane_y = y + lane_offsets[lane_key]
            parts.append(f'<text x="{left - 10}" y="{lane_y + 9:.1f}" text-anchor="end" font-size="10" font-weight="700" fill="#334155">{html.escape(lane_label)}</text>')
            parts.append(f'<line x1="{left}" y1="{lane_y + 5:.1f}" x2="{left + plot_w}" y2="{lane_y + 5:.1f}" stroke="#dbe4ee"/>')

        flow_y = y + lane_offsets["flow"]
        for span, color, span_label, title in [
            (_relative_span(row, "current_start_ms", "current_end_ms", due), unified_stack_color("initial"), "initial", "initial model turn"),
            (_relative_span(row, "tool_gap_start_ms", "tool_gap_end_ms", due), unified_stack_color("tool_wait"), "tool wait", "agent/tool wait window"),
            (_relative_span(row, "resume_start_ms", "resume_end_ms", due), unified_stack_color("decode"), "resume", "resume request wall time"),
        ]:
            if span:
                draw_span(parts, span[0], span[1], flow_y, 14, color, span_label, f"{label} | {title}: {display_ms(span[1] - span[0])}", opacity=0.78)

        request_y = y + lane_offsets["request"]
        request_spans = [
            (_relative_span(row, "resume_submitted_ms", "resume_start_ms", due), unified_stack_color("client_dispatch"), "client dispatch", "replay submitted but Python/client request call had not started"),
            (_relative_span(row, "replay_sglang_receive_start_ms", "replay_sglang_receive_end_ms", due), unified_stack_color("sglang_receive"), "receive", "SGLang receive stage"),
            (_relative_span(row, "replay_scheduler_queue_enter_start_ms", "replay_scheduler_admit_start_ms", due), unified_stack_color("scheduler"), "scheduler", "scheduler queue/admit wait"),
            (_relative_span(row, "replay_scheduler_admit_start_ms", "replay_kv_h2d_start_ms", due), unified_stack_color("load_path"), "load path", "scheduler admit to visible replay H2D start"),
        ]
        for span, color, span_label, title in request_spans:
            if span:
                draw_span(parts, span[0], span[1], request_y, 14, color, span_label, f"{label} | {title}: {display_ms(span[1] - span[0])}")
        for marker_key, marker_label, color in [
            ("resume_submitted_ms", "client submit", "#7c3aed"),
            ("resume_start_ms", "request start", "#2563eb"),
            ("replay_sglang_receive_start_ms", "SGLang receive", "#f97316"),
            ("replay_scheduler_admit_start_ms", "scheduler admit", "#db2777"),
        ]:
            marker = _relative_ms(row, marker_key, due)
            if marker is not None:
                draw_marker(parts, marker, request_y - 4, request_y + 18, color, f"{label} | {marker_label}: {display_ms(marker)} relative to due")

        prefetch_span = _relative_span(row, "prefetch_start_ms", "prefetch_end_ms", due)
        if prefetch_span:
            draw_span(parts, prefetch_span[0], prefetch_span[1], request_y + 17, 11, unified_stack_color("prefetch"), "prefetch", f"{label} | direct prefetch attempt: {display_ms(prefetch_span[1] - prefetch_span[0])}", opacity=0.72)
        hint_span = _relative_span(row, "direct_kv_h2d_start_ms", "direct_kv_h2d_end_ms", due)
        if hint_span:
            draw_span(parts, hint_span[0], hint_span[1], request_y + 30, 10, unified_stack_color("hint_h2d"), "hint H2D", f"{label} | hint-side KV H2D: {display_ms(hint_span[1] - hint_span[0])}")

        kv_y = y + lane_offsets["kv"]
        kind_y = {
            "H2D": kv_y - 2,
            "D2H": kv_y + 8,
            "GPU evict": kv_y + 18,
            "host evict": kv_y + 28,
        }
        kind_color = {
            "H2D": unified_stack_color("h2d"),
            "D2H": unified_stack_color("d2h"),
            "GPU evict": unified_stack_color("evict"),
            "host evict": unified_stack_color("host_evict"),
        }
        event_counts: Counter[str] = Counter()
        target_session = str(row.get("ledger_session_id") or row.get("session_id") or "")
        for event in unified_stack_kv_events_for_gap(row, all_kv_events)[:260]:
            span = _event_relative_span(event, due)
            if not span:
                continue
            kind = str(event.get("movement_kind") or "KV movement")
            if kind not in kind_y:
                continue
            event_counts[kind] += 1
            owner = str(event.get("ledger_session_id") or "")
            opacity = 0.92 if owner == target_session else 0.52
            bar_h = 8 if owner == target_session else 6
            title = (
                f"{label} | {kind} | owner={event.get('owner_kind', '')} | phase={event.get('phase', '')} | "
                f"tokens/idx={event.get('token_or_index_count', '')} | {display_ms(span[0])} -> {display_ms(span[1])} relative to due"
            )
            draw_span(parts, span[0], span[1], kind_y[kind], bar_h, kind_color[kind], "", title, opacity=opacity)
        counts_text = " | ".join(f"{kind} {event_counts.get(kind, 0)}" for kind in ["H2D", "D2H", "GPU evict", "host evict"])
        parts.append(f'<text x="{left + 8}" y="{kv_y + 43:.1f}" font-size="9" fill="#475569">{html.escape(counts_text)}</text>')

        replay_y = y + lane_offsets["replay"]
        replay_h2d = _relative_span(row, "replay_kv_h2d_start_ms", "replay_kv_h2d_end_ms", due)
        if replay_h2d:
            draw_span(parts, replay_h2d[0], replay_h2d[1], replay_y - 3, 10, unified_stack_color("h2d"), "replay H2D", f"{label} | replay-side KV H2D: {display_ms(replay_h2d[1] - replay_h2d[0])}")
        first_token = first_token_ms(row)
        replay_start = as_float(row.get("resume_start_ms"))
        if first_token is not None and replay_start is not None:
            prefill_start_rel = replay_start - due
            first_token_rel = first_token - due
            recompute_tokens = as_float(row.get("recomputed_tokens_est")) or as_float(row.get("replay_new_prefill_tokens_est")) or 0.0
            if recompute_tokens >= 128:
                draw_span(parts, prefill_start_rel, first_token_rel, replay_y + 10, 10, unified_stack_color("recompute"), "recompute/prefill", f"{label} | estimated replay recompute/prefill until first token: {display_ms(first_token_rel - prefill_start_rel)}", opacity=0.82)
            else:
                draw_span(parts, prefill_start_rel, first_token_rel, replay_y + 10, 10, unified_stack_color("prefill"), "TTFT", f"{label} | remaining time to first token: {display_ms(first_token_rel - prefill_start_rel)}", opacity=0.82)
            draw_marker(parts, first_token_rel, replay_y - 5, replay_y + 30, unified_stack_color("prefill"), f"{label} | first token: {display_ms(first_token_rel)} relative to due")
        decode_span = None
        if first_token is not None and as_float(row.get("resume_end_ms")) is not None:
            decode_span = (first_token - due, (as_float(row.get("resume_end_ms")) or first_token) - due)
        if decode_span and decode_span[1] > decode_span[0]:
            draw_span(parts, decode_span[0], decode_span[1], replay_y + 24, 10, unified_stack_color("decode"), "decode", f"{label} | decode after first token: {display_ms(decode_span[1] - decode_span[0])}", opacity=0.78)

        verdict_y = y + lane_offsets["verdict"]
        verdict = str(row.get("lifecycle_verdict") or row.get("final_path") or row.get("per_gap_verdict") or "")
        explanation = str(row.get("lifecycle_explanation") or row.get("replay_cache_path_summary") or "")
        parts.append(f'<text x="{left + 6}" y="{verdict_y + 8:.1f}" font-size="10" font-weight="800" fill="#0f172a">{html.escape(verdict[:90])}</text>')
        if explanation:
            parts.append(f'<text x="{left + 6}" y="{verdict_y + 24:.1f}" font-size="9" fill="#475569">{html.escape(explanation[:170])}</text>')

    legend_y = height - 48
    lx = left
    legend = [
        ("initial", unified_stack_color("initial")),
        ("tool wait", unified_stack_color("tool_wait")),
        ("client dispatch", unified_stack_color("client_dispatch")),
        ("scheduler/load path", unified_stack_color("scheduler")),
        ("H2D", unified_stack_color("h2d")),
        ("D2H", unified_stack_color("d2h")),
        ("evict", unified_stack_color("evict")),
        ("recompute", unified_stack_color("recompute")),
        ("decode", unified_stack_color("decode")),
    ]
    for legend_label, color in legend:
        parts.append(f'<rect x="{lx:.1f}" y="{legend_y:.1f}" width="13" height="13" rx="3" fill="{color}" opacity="0.86"/>')
        parts.append(f'<text x="{lx + 18:.1f}" y="{legend_y + 11:.1f}" font-size="11" fill="#334155">{html.escape(legend_label)}</text>')
        lx += 148
    parts.append("</svg>")
    return "\n".join(parts)


def build_unified_per_gap_stack_timeline_svg_v2(
    gaps: list[dict[str, Any]],
    all_kv_events: list[dict[str, Any]],
    max_rows: int,
) -> str:
    rows = gaps[:max_rows]
    if not rows:
        return "<p>No rows were available for the unified forensic stack timeline.</p>"
    axis_values = unified_stack_axis_values(rows, all_kv_events)
    x_min = min(axis_values) if axis_values else -1000.0
    x_max = max(axis_values) if axis_values else 1000.0
    if x_min >= 0:
        x_min = -500.0
    if x_max <= 0:
        x_max = 1000.0
    pad = max(50.0, (x_max - x_min) * 0.03)
    x_min -= pad
    x_max += pad

    width = 1780
    left = 280
    right = 56
    top = 150
    row_h = 244
    bottom = 96
    plot_w = width - left - right
    height = top + len(rows) * row_h + bottom
    scaled_min = h2d_symlog_value(x_min)
    scaled_max = h2d_symlog_value(x_max)

    def overview_x(value: float) -> float:
        scaled = h2d_symlog_value(value)
        return left + (scaled - scaled_min) * plot_w / max(1e-9, scaled_max - scaled_min)

    def zoom_bounds(row: dict[str, Any], events: list[dict[str, Any]], due: float) -> tuple[float, float] | None:
        values: list[float] = []
        for event in events:
            span = _event_relative_span(event, due)
            if span:
                values.extend(span)
        for span in [
            _relative_span(row, "direct_kv_h2d_start_ms", "direct_kv_h2d_end_ms", due),
            _relative_span(row, "replay_kv_h2d_start_ms", "replay_kv_h2d_end_ms", due),
        ]:
            if span:
                values.extend(span)
        if not values:
            return None
        start = min(values)
        end = max(values)
        span_ms = max(1.0, end - start)
        padding = max(25.0, span_ms * 0.06)
        return start - padding, end + padding

    def zoom_x(value: float, z_min: float, z_max: float) -> float:
        return left + (value - z_min) * plot_w / max(1e-9, z_max - z_min)

    def draw_span(
        parts: list[str],
        x_fn: Callable[[float], float],
        axis_min: float,
        axis_max: float,
        start: float,
        end: float,
        y: float,
        h: float,
        color: str,
        label: str,
        title: str,
        opacity: float = 0.86,
        min_w: float = 3.0,
        break_long: bool = False,
    ) -> None:
        clipped_start = max(axis_min, min(axis_max, start))
        clipped_end = max(axis_min, min(axis_max, end))
        if clipped_end <= clipped_start:
            return
        x1 = x_fn(clipped_start)
        x2 = x_fn(clipped_end)
        w = max(min_w, x2 - x1)
        parts.append(
            f'<rect x="{x1:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{color}" opacity="{opacity}">'
            f'<title>{html.escape(title)}</title></rect>'
        )
        if break_long and w >= 220:
            bx = x1 + w * 0.52
            parts.append(
                f'<rect x="{bx - 18:.1f}" y="{y - 1:.1f}" width="36" height="{h + 2:.1f}" rx="5" fill="#ffffff" opacity="0.86"/>'
            )
            parts.append(
                f'<text x="{bx:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" font-size="11" '
                f'font-weight="900" fill="{color}">...</text>'
            )
        if label and w >= 74:
            text_x = x1 + w / 2
            parts.append(
                f'<text x="{text_x:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" font-size="9" '
                f'font-weight="800" fill="#0f172a">{html.escape(label)}</text>'
            )

    def draw_overview_marker(parts: list[str], value: float, y1: float, y2: float, color: str, title: str) -> None:
        if value < x_min or value > x_max:
            return
        x = overview_x(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2.2">'
            f'<title>{html.escape(title)}</title></line>'
        )

    zero_x = overview_x(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Unified per-gap forensic stack timeline with per-gap KV zoom">',
        '<text x="12" y="30" font-size="20" font-weight="800" fill="#0f172a">Unified per-gap forensic stack timeline</text>',
        '<text x="12" y="54" font-size="12" fill="#475569">Each gap has a compact overview plus an expanded KV activity zoom. The overview uses replay-relative symlog time; each zoom strip uses its own local linear time.</text>',
        '<text x="12" y="76" font-size="12" fill="#475569">Long overview bars are intentionally compressed and marked with “...” when they would otherwise dominate the row.</text>',
        f'<line x1="{zero_x:.1f}" y1="{top - 32}" x2="{zero_x:.1f}" y2="{height - 70}" stroke="#111827" stroke-width="2.4"/>',
        f'<text x="{zero_x + 6:.1f}" y="{top - 42}" font-size="12" font-weight="800">0 ms replay due</text>',
    ]
    ticks = h2d_symlog_tick_values(x_min, x_max)
    for value in ticks:
        x = overview_x(value)
        label = f"{int(value)} ms" if abs(value) < 1000 else f"{value / 1000:.0f} s"
        parts.append(f'<line x1="{x:.1f}" y1="{top - 24}" x2="{x:.1f}" y2="{height - 70}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{top - 30}" text-anchor="middle" font-size="10" fill="#475569">{html.escape(label)}</text>')

    for idx, row in enumerate(rows):
        due = as_float(row.get("tool_gap_end_ms"))
        if due is None:
            continue
        y = top + idx * row_h
        band = "#ffffff" if idx % 2 == 0 else "#eef4fb"
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        status, status_color = observation_status(row)
        row_events = unified_stack_kv_events_for_gap(row, all_kv_events)
        target_session = str(row.get("ledger_session_id") or row.get("session_id") or "")
        event_counts: Counter[str] = Counter(str(event.get("movement_kind") or "") for event in row_events)
        zoom = zoom_bounds(row, row_events, due)

        parts.append(f'<rect x="0" y="{y - 8:.1f}" width="{width}" height="{row_h - 10}" fill="{band}"/>')
        parts.append(f'<text x="12" y="{y + 14:.1f}" font-size="15" font-weight="900">{html.escape(label)}</text>')
        parts.append(f'<text x="12" y="{y + 34:.1f}" font-size="10" font-weight="900" fill="{status_color}">{html.escape(str(row.get("per_gap_verdict") or status))}</text>')
        parts.append(f'<text x="12" y="{y + 52:.1f}" font-size="10" fill="#475569">mode {html.escape(str(row.get("mode") or ""))}; fillers {html.escape(case_fillers(row))}; wait {html.escape(str(row.get("tool_gap_ms") or ""))} ms</text>')
        parts.append(
            f'<text x="12" y="{y + 70:.1f}" font-size="10" fill="#475569">'
            f'H2D {event_counts.get("H2D", 0)} | D2H {event_counts.get("D2H", 0)} | evict {event_counts.get("GPU evict", 0)}'
            f'</text>'
        )
        parts.append(f'<text x="12" y="{y + 88:.1f}" font-size="10" fill="#475569">{html.escape(str(row.get("replay_path") or replay_path_from_evidence(row))[:48])}</text>')

        overview_lanes = [
            ("overview", y + 18),
            ("request", y + 48),
            ("replay", y + 78),
        ]
        for lane_label, lane_y in overview_lanes:
            parts.append(f'<text x="{left - 10}" y="{lane_y + 9:.1f}" text-anchor="end" font-size="10" font-weight="800" fill="#334155">{html.escape(lane_label)}</text>')
            parts.append(f'<line x1="{left}" y1="{lane_y + 5:.1f}" x2="{left + plot_w}" y2="{lane_y + 5:.1f}" stroke="#dbe4ee"/>')

        overview_y = y + 18
        for span, color, span_label, title, opacity in [
            (_relative_span(row, "current_start_ms", "current_end_ms", due), unified_stack_color("initial"), "initial", "initial model turn", 0.84),
            (_relative_span(row, "tool_gap_start_ms", "tool_gap_end_ms", due), unified_stack_color("tool_wait"), "tool wait", "agent/tool wait window", 0.78),
            (_relative_span(row, "resume_start_ms", "resume_end_ms", due), unified_stack_color("decode"), "resume", "resume request wall time", 0.70),
        ]:
            if span:
                draw_span(parts, overview_x, x_min, x_max, span[0], span[1], overview_y, 14, color, span_label, f"{label} | {title}: {display_ms(span[1] - span[0])}", opacity=opacity, break_long=True)

        request_y = y + 48
        request_spans = [
            (_relative_span(row, "resume_submitted_ms", "resume_start_ms", due), unified_stack_color("client_dispatch"), "client dispatch", "client dispatch: replay submitted but client call had not started"),
            (_relative_span(row, "replay_sglang_receive_start_ms", "replay_sglang_receive_end_ms", due), unified_stack_color("sglang_receive"), "receive", "SGLang receive stage"),
            (_relative_span(row, "replay_scheduler_queue_enter_start_ms", "replay_scheduler_admit_start_ms", due), unified_stack_color("scheduler"), "scheduler", "scheduler queue/admit wait"),
            (_relative_span(row, "replay_scheduler_admit_start_ms", "replay_kv_h2d_start_ms", due), unified_stack_color("load_path"), "load path", "scheduler admit to visible replay H2D start"),
        ]
        for span, color, span_label, title in request_spans:
            if span:
                draw_span(parts, overview_x, x_min, x_max, span[0], span[1], request_y, 14, color, span_label, f"{label} | {title}: {display_ms(span[1] - span[0])}", break_long=True)

        replay_y = y + 78
        replay_h2d = _relative_span(row, "replay_kv_h2d_start_ms", "replay_kv_h2d_end_ms", due)
        if replay_h2d:
            draw_span(parts, overview_x, x_min, x_max, replay_h2d[0], replay_h2d[1], replay_y - 2, 9, unified_stack_color("h2d"), "", f"{label} | replay-side KV H2D: {display_ms(replay_h2d[1] - replay_h2d[0])}", min_w=7)
        first_token = first_token_ms(row)
        replay_start = as_float(row.get("resume_start_ms"))
        if first_token is not None and replay_start is not None:
            prefill_start_rel = replay_start - due
            first_token_rel = first_token - due
            recompute_tokens = as_float(row.get("recomputed_tokens_est")) or as_float(row.get("replay_new_prefill_tokens_est")) or 0.0
            replay_color = unified_stack_color("recompute") if recompute_tokens >= 128 else unified_stack_color("prefill")
            replay_label = "recompute/TTFT" if recompute_tokens >= 128 else "TTFT"
            draw_span(parts, overview_x, x_min, x_max, prefill_start_rel, first_token_rel, replay_y + 11, 10, replay_color, replay_label, f"{label} | replay pre-first-token path: {display_ms(first_token_rel - prefill_start_rel)}", opacity=0.82, break_long=True)
            draw_overview_marker(parts, first_token_rel, replay_y - 5, replay_y + 30, unified_stack_color("prefill"), f"{label} | first token: {display_ms(first_token_rel)} relative to due")
        if first_token is not None and as_float(row.get("resume_end_ms")) is not None:
            decode_span = (first_token - due, (as_float(row.get("resume_end_ms")) or first_token) - due)
            if decode_span[1] > decode_span[0]:
                draw_span(parts, overview_x, x_min, x_max, decode_span[0], decode_span[1], replay_y + 25, 9, unified_stack_color("decode"), "decode", f"{label} | decode after first token: {display_ms(decode_span[1] - decode_span[0])}", opacity=0.78, break_long=True)

        zoom_title_y = y + 116
        parts.append(f'<text x="{left - 10}" y="{zoom_title_y + 9:.1f}" text-anchor="end" font-size="10" font-weight="900" fill="#334155">KV zoom</text>')
        if zoom is None:
            parts.append(f'<text x="{left + 8}" y="{zoom_title_y + 9:.1f}" font-size="10" fill="#64748b">No SGLang-visible KV movement in this gap window.</text>')
        else:
            z_min, z_max = zoom
            z_span = max(1.0, z_max - z_min)
            zoom_label = f"expanded KV burst: {display_ms(z_min)} -> {display_ms(z_max)} relative to replay due"
            parts.append(f'<text x="{left + 8}" y="{zoom_title_y - 4:.1f}" font-size="10" font-weight="800" fill="#475569">{html.escape(zoom_label)}</text>')
            for tick_value in [z_min, z_min + z_span * 0.25, z_min + z_span * 0.5, z_min + z_span * 0.75, z_max]:
                tx = zoom_x(tick_value, z_min, z_max)
                parts.append(f'<line x1="{tx:.1f}" y1="{zoom_title_y + 14:.1f}" x2="{tx:.1f}" y2="{zoom_title_y + 91:.1f}" stroke="#e5e7eb"/>')
                parts.append(f'<text x="{tx:.1f}" y="{zoom_title_y + 108:.1f}" text-anchor="middle" font-size="9" fill="#64748b">{html.escape(display_ms(tick_value))}</text>')
            if z_min <= 0 <= z_max:
                zx = zoom_x(0.0, z_min, z_max)
                parts.append(f'<line x1="{zx:.1f}" y1="{zoom_title_y + 14:.1f}" x2="{zx:.1f}" y2="{zoom_title_y + 91:.1f}" stroke="#111827" stroke-width="1.6"/>')
                parts.append(f'<text x="{zx + 4:.1f}" y="{zoom_title_y + 26:.1f}" font-size="9" font-weight="800">due</text>')
            zoom_lanes = {
                "H2D": (zoom_title_y + 20, unified_stack_color("h2d")),
                "D2H": (zoom_title_y + 42, unified_stack_color("d2h")),
                "GPU evict": (zoom_title_y + 64, unified_stack_color("evict")),
                "host evict": (zoom_title_y + 82, unified_stack_color("host_evict")),
            }
            for lane_name, (lane_y, _) in zoom_lanes.items():
                parts.append(f'<text x="{left - 10}" y="{lane_y + 8:.1f}" text-anchor="end" font-size="9" font-weight="800" fill="#334155">{html.escape(lane_name.replace("GPU ", ""))}</text>')
                parts.append(f'<line x1="{left}" y1="{lane_y + 5:.1f}" x2="{left + plot_w}" y2="{lane_y + 5:.1f}" stroke="#dbe4ee"/>')
            for event in row_events[:420]:
                span = _event_relative_span(event, due)
                if not span:
                    continue
                kind = str(event.get("movement_kind") or "")
                if kind not in zoom_lanes:
                    continue
                lane_y, color = zoom_lanes[kind]
                owner = str(event.get("ledger_session_id") or "")
                target = owner == target_session
                title = (
                    f"{label} | expanded KV burst | {kind} | owner={event.get('owner_kind', '')} | "
                    f"session={event.get('ledger_session_id', '')} | block={event.get('logical_block_id', '')} | "
                    f"tokens/idx={event.get('token_or_index_count', '')} | {display_ms(span[0])} -> {display_ms(span[1])} relative to due"
                )
                draw_span(
                    parts,
                    lambda value, z_min=z_min, z_max=z_max: zoom_x(value, z_min, z_max),
                    z_min,
                    z_max,
                    span[0],
                    span[1],
                    lane_y,
                    8 if target else 6,
                    color,
                    "",
                    title,
                    opacity=0.94 if target else 0.48,
                    min_w=5.0 if target else 3.5,
                )

        verdict_y = y + row_h - 28
        verdict = str(row.get("lifecycle_verdict") or row.get("final_path") or row.get("per_gap_verdict") or "")
        explanation = str(row.get("lifecycle_explanation") or row.get("replay_cache_path_summary") or "")
        parts.append(f'<text x="{left}" y="{verdict_y:.1f}" font-size="10" font-weight="900" fill="#0f172a">verdict: {html.escape(verdict[:92])}</text>')
        if explanation:
            parts.append(f'<text x="{left}" y="{verdict_y + 16:.1f}" font-size="9" fill="#475569">{html.escape(explanation[:190])}</text>')

    legend_y = height - 48
    lx = left
    legend = [
        ("initial", unified_stack_color("initial")),
        ("tool wait", unified_stack_color("tool_wait")),
        ("client dispatch", unified_stack_color("client_dispatch")),
        ("scheduler/load path", unified_stack_color("scheduler")),
        ("H2D", unified_stack_color("h2d")),
        ("D2H", unified_stack_color("d2h")),
        ("evict", unified_stack_color("evict")),
        ("recompute", unified_stack_color("recompute")),
        ("decode", unified_stack_color("decode")),
    ]
    for legend_label, color in legend:
        parts.append(f'<rect x="{lx:.1f}" y="{legend_y:.1f}" width="13" height="13" rx="3" fill="{color}" opacity="0.86"/>')
        parts.append(f'<text x="{lx + 18:.1f}" y="{legend_y + 11:.1f}" font-size="11" fill="#334155">{html.escape(legend_label)}</text>')
        lx += 150
    parts.append("</svg>")
    return "\n".join(parts)


def unified_per_gap_forensic_stack_html(
    gaps: list[dict[str, Any]],
    all_kv_events: list[dict[str, Any]],
    max_rows: int,
) -> str:
    if not gaps:
        return "<p>No timeline rows were available for the unified forensic stack.</p>"
    return f"""
    <p>This is a preview of a merged per-gap view. Each gap has a compact overview plus a local zoom of the dense KV movement burst.</p>
    <p class="note">Use the overview to see the big timing story. Use the expanded KV zoom under each row to inspect the small H2D, D2H, and eviction bars that otherwise get crushed at the far right.</p>
    <div class="setup-diagram">{build_unified_per_gap_stack_timeline_svg_v2(gaps, all_kv_events, max_rows)}</div>
    <p class="note">Target-row movement is drawn thicker and more opaque. Pressure/filler or other-session movement is thinner and faded. The zoom strip uses a local linear scale per gap, while the overview remains replay-relative symlog time.</p>
    """


def live_direct_prefetch_html(live_run: dict[str, Any] | None, max_timeline_gaps: int) -> str:
    if not live_run:
        return """
  <details id="live-direct" class="section-card theme-profiled">
    <summary><h2>Live AgentBench Direct Prefetch</h2></summary>
    <p>No live direct-prefetch run was attached to this report.</p>
  </details>
"""
    gaps = live_run.get("gaps", [])
    mode_rows = [live_mode_summary(live_run)]
    interesting = timeline_rows_with_labels(selected_timeline_gaps(gaps, max_timeline_gaps), prefix="L")
    summary = mode_rows[0] if mode_rows else {}
    live_cards = [
        ("live analyzed requests", summary.get("analyzed_model_requests", "")),
        ("live tool calls", summary.get("total_tool_calls", "")),
        ("live tool gaps", summary.get("observed_tool_gaps", "")),
        ("late live prefetches", summary.get("late_prefetch_attempts", "")),
        ("avg live prefetch duration", f"{summary.get('avg_prefetch_duration_ms', '')} ms"),
        ("hint H2D gaps", sum(1 for row in gaps if has_events(row.get("direct_kv_h2d_events")))),
        ("replay H2D gaps", sum(1 for row in gaps if has_events(row.get("replay_kv_h2d_events")))),
    ]
    cards_html = "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{html.escape(str(label))}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
        for label, value in live_cards
    ) + "</div>"
    live_setup_rows = [
        {
            "part": "Request source",
            "simple meaning": "Real SWE-bench / DeepAgents task execution creates live model turns and real tool calls.",
        },
        {
            "part": "Hint trigger",
            "simple meaning": "When the live proxy sees a model turn produce tool calls, it emits a hint for that session.",
        },
        {
            "part": "Direct prefetch attempt",
            "simple meaning": "The controller sends a marked direct-load request so SGLang can exercise its KV load-back path.",
        },
        {
            "part": "Resume",
            "simple meaning": "The real agent continues after the tool work; the report checks whether prefetch finished before that resume.",
        },
    ]
    detail_columns = [
        "session_id",
        "task_index",
        "gap_order_in_task",
        "tool_names",
        "tool_gap_ms",
        "prefetch_duration_ms",
        "prefetch_margin_ms",
        "resume_latency_ms",
        "direct_kv_h2d_events",
        "replay_kv_h2d_events",
    ]
    return f"""
  <details id="live-direct" class="section-card theme-profiled">
    <summary><h2>Live AgentBench Direct Prefetch</h2></summary>
    <p class="note">This section is the real live workload check. It uses only direct prefetch mode: real AgentBench/DeepAgents tool calls create live gaps, and the controller tries to trigger direct SGLang KV load-back during those gaps.</p>
    <h3>Live Summary</h3>
    {cards_html}
    <h3>How This Live Run Works</h3>
    {table_html(live_setup_rows, ["part", "simple meaning"])}
    <h3>Global Live Prefetch Margin</h3>
    <p>Positive margin means the live direct-prefetch path finished before the real agent resumed. Negative margin means the agent resumed first.</p>
    {live_global_prefetch_margin_html(gaps)}
    <h3>Live Direct-Prefetch Timeline</h3>
    <p class="note">Rows with green or cyan bars are shown first. Green is hint-side direct KV HtoD evidence; cyan is replay-side HtoD evidence from the real resume request.</p>
    {timeline_model_table_html()}
    {build_expanded_gap_timeline_svg(interesting, max_timeline_gaps, show_prefetch_legend=True, scale="symlog")}
    <h3>Live Row Map</h3>
    {table_html(timeline_mapping_rows(interesting))}
    <h3>Live Key Observations</h3>
    {table_html(key_observation_rows(interesting), ["row", "mode", "status", "what happened", "why it matters", "tool_wait_ms", "resume_ttft_ms"])}
    <h3>Live Direct KV Evidence</h3>
    {table_html(gaps, detail_columns, limit=200)}
  </details>
"""


def render_html(
    gaps: list[dict[str, Any]],
    result_root: Path,
    max_timeline_gaps: int,
    live_run: dict[str, Any] | None = None,
    request_coverage: list[dict[str, Any]] | None = None,
    kv_block_rows: list[dict[str, Any]] | None = None,
    exact_kv_movement_rows: list[dict[str, Any]] | None = None,
    trace_rows: list[dict[str, Any]] | None = None,
    run_environment: dict[str, Any] | None = None,
) -> str:
    mode_rows = mode_summary_rows(gaps)
    ledger = build_replay_path_ledger(gaps)
    request_coverage = request_coverage or []
    kv_block_rows = kv_block_rows or []
    exact_kv_movement_rows = exact_kv_movement_rows or []
    trace_rows = trace_rows or []
    run_environment = run_environment or {}
    all_timeline_rows = timeline_rows_with_labels(selected_timeline_gaps(gaps, len(gaps)))
    interesting = timeline_rows_with_labels(selected_timeline_gaps(gaps, max_timeline_gaps))
    interesting_block_lifecycle_rows = block_lifecycle_by_gap_rows(interesting, kv_block_rows)
    all_h2d_activity_events = aligned_h2d_activity_events(all_timeline_rows, exact_kv_movement_rows)
    all_kv_movement_events = all_aligned_kv_movement_events(all_timeline_rows, exact_kv_movement_rows)
    client_dispatch_kv_summary_rows = client_dispatch_kv_movement_summary_rows(
        all_timeline_rows, all_kv_movement_events
    )
    client_dispatch_kv_event_rows = client_dispatch_kv_movement_event_rows(
        all_timeline_rows, all_kv_movement_events
    )
    all_h2d_pressure_rows = h2d_pressure_by_gap_rows(all_timeline_rows, all_h2d_activity_events)
    interesting_labels = {str(row.get("timeline_label") or "") for row in interesting}
    interesting_h2d_pressure_rows = [
        row for row in all_h2d_pressure_rows if str(row.get("row") or "") in interesting_labels
    ]
    replay_delay_rows = replay_delay_breakdown_rows(all_timeline_rows, all_h2d_activity_events)
    replay_delay_verdict_table_rows = replay_delay_verdict_rows(replay_delay_rows)
    replay_delay_stage_rows = request_stage_trace_rows(all_timeline_rows, trace_rows)
    replay_delay_h2d_rows = h2d_activity_during_delay_rows(all_timeline_rows, all_h2d_activity_events)
    replay_delay_running_rows = replay_delay_running_context_rows(all_timeline_rows, trace_rows, all_h2d_activity_events)
    replay_h2d_readiness_table_rows = replay_h2d_readiness_rows(gaps)
    replay_h2d_readiness_bucket_table_rows = replay_h2d_readiness_bucket_rows(replay_h2d_readiness_table_rows)
    replay_queue_table_rows = replay_queue_timing_rows(gaps)
    evidence_audit = audit_report_data(
        {
            "gaps": gaps,
            "exact_kv_movement_attribution": exact_kv_movement_rows,
            "kv_block_ledger": kv_block_rows,
            "replay_delay_stage_trace": replay_delay_stage_rows,
            "replay_queue_timing": replay_queue_table_rows,
            "client_dispatch_kv_movement_summary": client_dispatch_kv_summary_rows,
            "client_dispatch_kv_movement_events": client_dispatch_kv_event_rows,
        }
    )
    h2d_activity_window_table_rows = h2d_activity_window_rows(all_h2d_activity_events)
    h2d_contention_targets = select_h2d_contention_targets(all_timeline_rows)
    h2d_contention_summary_table_rows = h2d_contention_summary_rows(
        h2d_contention_targets, all_h2d_activity_events
    )
    h2d_contention_event_table_rows = h2d_contention_event_rows(
        h2d_contention_targets, all_h2d_activity_events
    )
    global_title = global_readiness_section_title(gaps)
    gap_columns = [
        "session_id",
        "mode",
        "task_index",
        "tool_names",
        "tool_gap_ms",
        "prefetch_duration_ms",
        "prefetch_margin_ms",
        "resume_ttft_ms",
        "replay_path",
        "per_gap_verdict",
        "final_path",
        "bottleneck_label",
        "path_confidence",
        "prefetch_outcome",
        "resume_submitted_ms",
        "resume_start_ms",
        "replay_sglang_receive_start_ms",
        "replay_scheduler_queue_enter_start_ms",
        "replay_scheduler_admit_start_ms",
        "replay_request_start_lateness_ms",
        "replay_submit_to_scheduler_queue_ms",
        "replay_scheduler_queue_to_admit_ms",
        "replay_scheduler_admit_to_h2d_ms",
        "scheduler_wait_ms",
        "kv_prepare_ms",
        "host_load_ms",
        "prefill_compute_ms_est",
        "recomputed_tokens_est",
        "gpu_resident_hit_tokens",
        "replay_input_tokens",
        "replay_active_input_tokens",
        "replay_scheduler_trimmed_tokens",
        "replay_cached_prefix_tokens",
        "replay_final_cached_prefix_tokens",
        "replay_cache_hit_ratio_pct",
        "replay_new_prefill_tokens_est",
        "lifecycle_host_write_tokens",
        "lifecycle_gpu_evict_tokens",
        "lifecycle_host_evict_tokens",
        "lifecycle_hint_h2d_tokens",
        "lifecycle_replay_h2d_tokens",
        "lifecycle_verdict",
        "replay_progressive_cache_events",
        "replay_post_request_cache_write_events",
        "movement_class",
        "direct_kv_h2d_events",
        "replay_kv_h2d_events",
    ]
    toc = [
        ("summary", "Summary"),
        ("setup", "Experiment Setup"),
        ("global-prefetch", global_title),
        ("h2d-pressure", "KV H2D Bandwidth Pressure"),
        ("delay-breakdown", "Replay Delay Breakdown"),
        ("client-dispatch-kv", "Client Dispatch KV Movement"),
        ("timeline-guide", "How To Read Timelines"),
        ("readable-phase-timeline", "Readable KV Lifecycle Timeline"),
        ("unified-forensic-stack", "Unified Forensic Stack Timeline"),
        ("per-gap-forensics", "Per-Gap Forensic View"),
        ("observations", "Key Observations"),
        ("evidence-audit", "Instrumentation Evidence Audit"),
        ("appendix", "Evidence Tables / Raw Proof"),
        ("reproduce", "Reproduce This Report"),
    ]
    if live_run:
        toc.insert(6, ("live-direct", "Live Direct Prefetch"))
    live_section = live_direct_prefetch_html(live_run, max_timeline_gaps) if live_run else ""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Real-Prompt Controlled Replay Master Report</title>
  <style>{master_css()}</style>
</head>
<body>
<main>
  <section>
    <h1>Real-Prompt Controlled Replay Master Report</h1>
    <p>This is the controlled real-prompt version of the master report. The prompts come from real SWE-bench / DeepAgents traces, while the experiment controls tool-wait time, cache pressure, and direct KV prefetch timing.</p>
    <p class="note">Use this report for the clean hardware-story experiment: real prompt content, known resume deadlines, known pressure levels, direct SGLang KV-hook attempts, and replay-side KV movement evidence.</p>
    <h2>Table of Contents</h2>
    {toc_html(toc)}
  </section>

  <details id="summary" class="section-card theme-summary">
    <summary><h2>Summary</h2></summary>
    <p>This section gives the headline numbers across no-prefetch and direct-prefetch modes. The replay-before-first-token window is inferred from TTFT and is now split into evidence colors: cyan for replay-side host KV load, magenta for recompute/rebuild, and gold for remaining prefill or wait.</p>
    {metric_cards_html(mode_rows)}
  </details>

  <details id="setup" class="section-card theme-setup">
    <summary><h2>Experiment Setup And Manager Summary</h2></summary>
    {manager_setup_html(run_environment)}
  </details>

  <details id="global-prefetch" class="section-card theme-global">
    <summary><h2>{html.escape(global_title)}</h2></summary>
    <p>For no-prefetch rows, this section measures replay-side KV H2D readiness. For direct-prefetch rows, it also reports the normal prefetch margin.</p>
    {global_readiness_html(gaps)}
  </details>

  <details id="h2d-pressure" class="section-card theme-directkv">
    <summary><h2>KV H2D Bandwidth Pressure</h2></summary>
    <p>This section shows how much host-to-device KV movement was happening near replay deadlines. It helps explain whether a late replay was isolated or happened while the KV movement path was already busy.</p>
    {h2d_bandwidth_pressure_html(gaps, exact_kv_movement_rows)}
  </details>

  <details id="delay-breakdown" class="section-card theme-profiled">
    <summary><h2>Replay Delay Breakdown</h2></summary>
    <p>This section answers the next question: if replay-side H2D started late, where did the time go before the copy began?</p>
    {replay_delay_breakdown_html(all_timeline_rows, trace_rows, all_h2d_activity_events, max_rows=max_timeline_gaps)}
  </details>

  <details id="client-dispatch-kv" class="section-card theme-directkv">
    <summary><h2>Client Dispatch KV Movement</h2></summary>
    {client_dispatch_kv_movement_html(client_dispatch_kv_summary_rows, client_dispatch_kv_event_rows, max_rows=max_timeline_gaps)}
  </details>

  <details id="timeline-guide" class="section-card theme-guide">
    <summary><h2>How To Read The Timelines</h2></summary>
    {timeline_guide_html(profiled_available=True)}
  </details>

  <details id="readable-phase-timeline" class="section-card theme-clean">
    <summary><h2>Readable KV Lifecycle Timeline</h2></summary>
    <p class="note">This is the main timeline. Each row is one controlled replay gap such as <code>G00</code>. The columns are local views, so the bars are stretched for readability while the printed durations preserve the measured timing.</p>
    <p class="note">Cyan and green bars now carry exact logical KV block attribution from the ledger. Hover over those bars to see block IDs, node IDs, token ranges, H2D start/end times, durations, and evidence confidence. Magenta/gold replay work remains explicitly marked as estimated.</p>
    <p class="note">Detailed block lifecycle counts and nearby H2D pressure rows are kept in <strong>Evidence Tables / Raw Proof</strong> at the bottom so this timeline stays easy to scan.</p>
    {build_local_timing_phase_timeline_svg(interesting, max_timeline_gaps, show_prefetch_legend=True, kv_block_lifecycle_rows=interesting_block_lifecycle_rows, h2d_pressure_rows=interesting_h2d_pressure_rows, show_block_lifecycle_strip=False, show_h2d_pressure_strip=False)}
  </details>

  <details id="unified-forensic-stack" class="section-card theme-profiled">
    <summary><h2>Unified Forensic Stack Timeline</h2></summary>
    {unified_per_gap_forensic_stack_html(interesting, all_kv_movement_events, max_timeline_gaps)}
  </details>

  <details id="per-gap-forensics" class="section-card theme-profiled">
    <summary><h2>Per-Gap Forensic View</h2></summary>
    {per_gap_forensic_view_html(interesting, kv_block_rows, all_h2d_pressure_rows, replay_delay_rows, replay_queue_table_rows, client_dispatch_kv_summary_rows, client_dispatch_kv_event_rows, exact_kv_movement_rows, max_timeline_gaps)}
  </details>

  {live_section}

  <details id="observations" class="section-card theme-observations">
    <summary><h2>Key Observations Per Gap/Session</h2></summary>
    <p>This section translates the timeline rows into plain English. It uses the same compact row names as the chart, so <code>G00</code> here means the same <code>G00</code> in the timeline.</p>
    {table_html(key_observation_rows(interesting), ["row", "mode", "status", "what happened", "why it matters", "tool_wait_ms", "resume_ttft_ms", "replay_path", "verdict"])}
  </details>

  <details id="evidence-audit" class="section-card theme-profiled">
    <summary><h2>Instrumentation Evidence Audit</h2></summary>
    <p>This section audits whether the report is backed by direct SGLang instrumentation, derived values, or inference. It is meant to keep the report honest: exact KV bars should trace to real hooks and IDs; inferred values should be labeled as inferred.</p>
    <h3>Audit Summary</h3>
    {table_html(evidence_audit["summary"])}
    <h3>Audit Matrix</h3>
    {table_html(evidence_audit["matrix"])}
    <h3>Chart Evidence Inventory</h3>
    {table_html(evidence_audit["chart_inventory"])}
    <h3>Artifact Inventory</h3>
    {table_html(evidence_audit["artifact_inventory"])}
  </details>

  <details id="appendix" class="section-card theme-appendix">
    <summary><h2>Evidence Tables / Raw Proof</h2></summary>
    <p class="note">These long tables are grouped here so the main report stays chart-first. Use this section when you want to audit the exact measured values behind the charts.</p>
    <h3>Detailed KV Block Lifecycle Column Guide</h3>
    {table_html(detailed_kv_lifecycle_column_guide_rows(), ["column", "meaning"])}
    <h3>Detailed KV Block Lifecycle Rows</h3>
    <p class="note">The H2D timing columns come from SGLang-visible KV movement hooks. Recompute timing is labeled <code>_est</code> because it is inferred from replay prefill/TTFT counters rather than from a physical block-level recompute event.</p>
    {table_html(detailed_kv_lifecycle_table_rows(gaps, kv_block_rows), limit=1000)}
    <h3>Replay H2D Readiness Rows</h3>
    {table_html(replay_h2d_readiness_table_rows)}
    <h3>Replay H2D Readiness Buckets</h3>
    {table_html(replay_h2d_readiness_bucket_table_rows)}
    <h3>Replay Queue Timing Rows</h3>
    {table_html(replay_queue_table_rows, limit=1000)}
    <h3>Replay Delay Verdicts</h3>
    {table_html(replay_delay_verdict_table_rows)}
    <h3>Replay Delay Stage Trace</h3>
    <p class="note">These rows come from direct SGLang method hooks emitted as <code>kv_telemetry.request_stage</code>. They are not parsed from server logs.</p>
    {table_html(replay_delay_stage_rows, limit=2000)}
    <h3>Replay Delay Stage Duration Rows</h3>
    {table_html(replay_delay_rows)}
    <h3>H2D Activity During The Delay Window</h3>
    {table_html(replay_delay_h2d_rows, limit=2000)}
    <h3>What Was Running During The Delay</h3>
    {table_html(replay_delay_running_rows)}
    <h3>All Aligned KV Movement Rows</h3>
    {table_html(all_kv_movement_events, limit=2000)}
    <h3>Client Dispatch KV Movement Summary</h3>
    {table_html(client_dispatch_kv_summary_rows)}
    <h3>Client Dispatch KV Movement Events</h3>
    {table_html(client_dispatch_kv_event_rows, limit=2000)}
    <h3>H2D Activity Window Rows</h3>
    {table_html(h2d_activity_window_table_rows)}
    <h3>Per-Gap H2D Pressure Rows</h3>
    {table_html(all_h2d_pressure_rows)}
    <h3>Per-Gap H2D Contention Verdict Rows</h3>
    {table_html(h2d_contention_summary_table_rows)}
    <h3>Per-Gap H2D Contention Event Rows</h3>
    {table_html(h2d_contention_event_table_rows, limit=2000)}
    <h3>Mode Summary</h3>
    {table_html(mode_rows)}
    <h3>Replay Path Proof Rows</h3>
    {table_html(replay_path_proof_rows(ledger), limit=250)}
    <h3>Replay Attribution Rows</h3>
    {table_html(replay_attribution_rows(gaps), limit=200)}
    <h3>Exact Movement Summary</h3>
    {table_html(exact_movement_summary_rows(exact_kv_movement_rows))}
    <h3>Exact Movement Rows</h3>
    {table_html(exact_movement_table_rows(exact_kv_movement_rows, interesting, limit=300))}
    <h3>KV Block Ledger Summary</h3>
    {table_html(ledger_summary_rows(kv_block_rows))}
    <h3>Per-Gap Block Summary</h3>
    {table_html(kv_block_gap_table_rows(interesting, kv_block_rows))}
    <h3>Request-ID Plumbing Audit</h3>
    {table_html(request_coverage)}
    <h3>Full Gap Details</h3>
    {table_html(gaps, gap_columns)}
  </details>

  <details id="reproduce" class="section-card theme-reproduce">
    <summary><h2>Reproduce This Report</h2></summary>
    {reproduce_controlled_replay_html(result_root)}
  </details>
</main>
{report_script()}
</body>
</html>
"""


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_cases(root: Path) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir():
            continue
        trace = child / "m27_trace.jsonl"
        if trace.exists():
            mode = child.name.split("_tw", 1)[0]
            cases.append((mode, child))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Milestone 27 controlled replay report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--live-direct-root", type=Path)
    parser.add_argument("--max-timeline-gaps", type=int, default=18)
    parser.add_argument("--run-environment-json", type=Path)
    args = parser.parse_args()

    all_gaps: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    for mode, case_dir in discover_cases(args.root):
        gaps, trace_rows = build_gaps_for_case(case_dir, mode)
        case_id = case_dir.name
        for gap in gaps:
            gap["case_id"] = case_id
            gap["case_dir"] = str(case_dir)
            gap["ledger_session_id"] = f"{case_id}::{gap.get('session_id', '')}"
        for row in trace_rows:
            row["ledger_case_id"] = case_id
            context = row.get("kv_context")
            if isinstance(context, dict):
                context["ledger_case_id"] = case_id
        all_gaps.extend(gaps)
        all_trace_rows.extend(trace_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = build_replay_path_ledger(all_gaps)
    normalized_kv_events = normalize_sglang_trace_events(all_trace_rows)
    kv_ledger = build_block_ledger(normalized_kv_events)
    kv_block_rows = block_ledger_rows(kv_ledger)
    kv_block_lifecycle_by_gap = block_lifecycle_by_gap_rows(all_gaps, kv_block_rows)
    exact_kv_rows = exact_movement_rows(normalized_kv_events)
    request_coverage = request_id_coverage_rows(all_trace_rows)
    h2d_readiness = replay_h2d_readiness_rows(all_gaps)
    queue_timing = replay_queue_timing_rows(all_gaps)
    all_labeled_gaps = timeline_rows_with_labels(selected_timeline_gaps(all_gaps, len(all_gaps)))
    h2d_activity_events = aligned_h2d_activity_events(all_labeled_gaps, exact_kv_rows)
    all_kv_movement_events = all_aligned_kv_movement_events(all_labeled_gaps, exact_kv_rows)
    client_dispatch_kv_summary = client_dispatch_kv_movement_summary_rows(
        all_labeled_gaps, all_kv_movement_events
    )
    client_dispatch_kv_events = client_dispatch_kv_movement_event_rows(
        all_labeled_gaps, all_kv_movement_events
    )
    h2d_pressure_rows = h2d_pressure_by_gap_rows(all_labeled_gaps, h2d_activity_events)
    h2d_activity_windows = h2d_activity_window_rows(h2d_activity_events)
    h2d_contention_targets = select_h2d_contention_targets(all_labeled_gaps)
    h2d_contention_summary = h2d_contention_summary_rows(h2d_contention_targets, h2d_activity_events)
    h2d_contention_events = h2d_contention_event_rows(h2d_contention_targets, h2d_activity_events)
    replay_delay_breakdown = replay_delay_breakdown_rows(all_labeled_gaps, h2d_activity_events)
    replay_delay_verdicts = replay_delay_verdict_rows(replay_delay_breakdown)
    replay_delay_running_context = replay_delay_running_context_rows(all_labeled_gaps, all_trace_rows, h2d_activity_events)
    replay_delay_stage_trace = request_stage_trace_rows(all_labeled_gaps, all_trace_rows)
    replay_delay_h2d_activity = h2d_activity_during_delay_rows(all_labeled_gaps, h2d_activity_events)
    replay_delay_gap_verdicts = delay_verdicts_by_gap_rows(replay_delay_breakdown)
    evidence_audit = audit_report_data(
        {
            "gaps": all_gaps,
            "exact_kv_movement_attribution": exact_kv_rows,
            "kv_block_ledger": kv_block_rows,
            "replay_delay_stage_trace": replay_delay_stage_trace,
            "replay_queue_timing": queue_timing,
            "client_dispatch_kv_movement_summary": client_dispatch_kv_summary,
            "client_dispatch_kv_movement_events": client_dispatch_kv_events,
        }
    )
    write_csv(args.out_dir / "controlled_replay_gaps.csv", all_gaps)
    write_csv(args.out_dir / "replay_path_ledger.csv", ledger)
    write_csv(args.out_dir / "replay_h2d_readiness.csv", h2d_readiness)
    write_csv(args.out_dir / "replay_queue_timing.csv", queue_timing)
    write_csv(args.out_dir / "replay_delay_breakdown.csv", replay_delay_breakdown)
    write_csv(args.out_dir / "replay_delay_verdicts.csv", replay_delay_verdicts)
    write_csv(args.out_dir / "replay_delay_running_context.csv", replay_delay_running_context)
    write_csv(args.out_dir / "replay_delay_stage_trace.csv", replay_delay_stage_trace)
    write_csv(args.out_dir / "replay_delay_h2d_activity.csv", replay_delay_h2d_activity)
    write_csv(args.out_dir / "replay_delay_gap_verdicts.csv", replay_delay_gap_verdicts)
    write_csv(args.out_dir / "h2d_activity_events.csv", h2d_activity_events)
    write_csv(args.out_dir / "all_aligned_kv_movement_events.csv", all_kv_movement_events)
    write_csv(args.out_dir / "client_dispatch_kv_movement_summary.csv", client_dispatch_kv_summary)
    write_csv(args.out_dir / "client_dispatch_kv_movement_events.csv", client_dispatch_kv_events)
    write_csv(args.out_dir / "h2d_pressure_by_gap.csv", h2d_pressure_rows)
    write_csv(args.out_dir / "h2d_activity_windows.csv", h2d_activity_windows)
    write_csv(args.out_dir / "h2d_contention_by_gap.csv", h2d_contention_summary)
    write_csv(args.out_dir / "h2d_contention_events.csv", h2d_contention_events)
    write_csv(args.out_dir / "hardware_counterfactual.csv", hardware_counterfactual_rows(ledger))
    write_csv(args.out_dir / "instrumentation_coverage.csv", instrumentation_coverage_rows(all_gaps, ledger))
    write_csv(args.out_dir / "request_id_coverage_report.csv", request_coverage)
    write_csv(args.out_dir / "exact_kv_movement_attribution.csv", exact_kv_rows)
    write_csv(args.out_dir / "exact_kv_movement_summary.csv", exact_movement_summary_rows(exact_kv_rows))
    write_csv(args.out_dir / "kv_block_lifecycle_by_gap.csv", kv_block_lifecycle_by_gap)
    write_csv(args.out_dir / "kv_block_lifecycle_verdict_counts.csv", block_lifecycle_verdict_counts(kv_block_lifecycle_by_gap))
    write_csv(args.out_dir / "instrumentation_evidence_audit_summary.csv", evidence_audit["summary"])
    write_csv(args.out_dir / "instrumentation_evidence_audit_matrix.csv", evidence_audit["matrix"])
    write_csv(args.out_dir / "instrumentation_chart_inventory.csv", evidence_audit["chart_inventory"])
    write_csv(args.out_dir / "instrumentation_artifact_inventory.csv", evidence_audit["artifact_inventory"])
    (args.out_dir / "instrumentation_evidence_audit.md").write_text(
        audit_markdown(evidence_audit),
        encoding="utf-8",
    )
    write_ledger_artifacts(args.out_dir, kv_block_rows, all_gaps)
    write_json(
        args.out_dir / "controlled_replay_report.json",
        {
            "gaps": all_gaps,
            "summary": mode_summary_rows(all_gaps),
            "replay_path_ledger": ledger,
            "replay_h2d_readiness": h2d_readiness,
            "replay_queue_timing": queue_timing,
            "replay_delay_breakdown": replay_delay_breakdown,
            "replay_delay_verdicts": replay_delay_verdicts,
            "replay_delay_running_context": replay_delay_running_context,
            "replay_delay_stage_trace": replay_delay_stage_trace,
            "replay_delay_h2d_activity": replay_delay_h2d_activity,
            "replay_delay_gap_verdicts": replay_delay_gap_verdicts,
            "replay_h2d_readiness_summary": replay_h2d_readiness_summary(h2d_readiness),
            "h2d_activity_events": h2d_activity_events,
            "all_aligned_kv_movement_events": all_kv_movement_events,
            "client_dispatch_kv_movement_summary": client_dispatch_kv_summary,
            "client_dispatch_kv_movement_events": client_dispatch_kv_events,
            "h2d_pressure_by_gap": h2d_pressure_rows,
            "h2d_activity_windows": h2d_activity_windows,
            "h2d_contention_by_gap": h2d_contention_summary,
            "h2d_contention_events": h2d_contention_events,
            "exact_kv_movement_attribution": exact_kv_rows,
            "exact_kv_movement_summary": exact_movement_summary_rows(exact_kv_rows),
            "kv_block_ledger": kv_block_rows,
            "kv_block_lifecycle_summary": ledger_summary_rows(kv_block_rows),
            "kv_block_gap_summary": gap_lifecycle_summary_rows(all_gaps, kv_block_rows),
            "kv_block_lifecycle_by_gap": kv_block_lifecycle_by_gap,
            "kv_block_lifecycle_verdict_counts": block_lifecycle_verdict_counts(kv_block_lifecycle_by_gap),
            "bottleneck_summary": bottleneck_summary_rows(ledger),
            "confidence_summary": confidence_summary_rows(ledger),
            "counterfactual_summary": counterfactual_summary_rows(ledger),
            "instrumentation_coverage": instrumentation_coverage_rows(all_gaps, ledger),
            "request_id_coverage": request_coverage,
            "instrumentation_evidence_audit_summary": evidence_audit["summary"],
            "instrumentation_evidence_audit_matrix": evidence_audit["matrix"],
            "instrumentation_chart_inventory": evidence_audit["chart_inventory"],
            "instrumentation_artifact_inventory": evidence_audit["artifact_inventory"],
        },
    )
    live_run = None
    if args.live_direct_root:
        live_run = load_live_agentbench_run(args.live_direct_root, "live_direct_prefetch", include_preflight=False)

    html_text = render_html(
        all_gaps,
        args.root,
        args.max_timeline_gaps,
        live_run=live_run,
        request_coverage=request_coverage,
        kv_block_rows=kv_block_rows,
        exact_kv_movement_rows=exact_kv_rows,
        trace_rows=all_trace_rows,
        run_environment=load_json(args.run_environment_json),
    )
    report_path = args.out_dir / "controlled_replay_report.html"
    report_path.write_text(html_text, encoding="utf-8")

    if args.latest_root:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, args.latest_root / "latest_controlled_replay_report.html")
        shutil.copy2(report_path, args.latest_root / "latest_master_report.html")

    print(f"Wrote Milestone 27 report to {report_path}")


if __name__ == "__main__":
    main()
