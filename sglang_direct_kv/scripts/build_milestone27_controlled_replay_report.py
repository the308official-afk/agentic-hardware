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
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentic_kv.block_ledger import (
    block_ledger_rows,
    build_block_ledger,
    gap_lifecycle_summary_rows,
    ledger_summary_rows,
    normalize_sglang_trace_events,
    write_ledger_artifacts,
)
from build_live_agentbench_tool_gap_report import (
    build_expanded_gap_timeline_svg,
    build_local_timing_phase_timeline_svg,
    build_replay_execution_timeline_svg,
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
    event_name: str,
) -> dict[str, list[dict[str, Any]]]:
    by_session: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        if row.get("event") != event_name:
            continue
        for session_id in _agent_sessions_for_event(row):
            if "::live_prefetch::" in session_id:
                continue
            by_session[session_id].append(
                {
                    "event": row.get("event", ""),
                    "source_event": row.get("source_event", ""),
                    "category": row.get("category", ""),
                    "method": row.get("method", ""),
                    "duration_ms": row.get("duration_ms", ""),
                    "start_or_end_ms": rel_ms(row.get("ts_ns"), base_ts),
                    "request_count": row.get("request_count", ""),
                    "request_id": row.get("request_id", ""),
                    "forward_mode": row.get("forward_mode", ""),
                    "extend_num_tokens": row.get("extend_num_tokens", ""),
                    "seq_lens_sum": row.get("seq_lens_sum", ""),
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
        if row.get("event") not in {"m27.request.start", "m27.request.end"}:
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
        if row.get("event") == "m27.request.start":
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
    scheduler_by_session = telemetry_events_by_session(trace_rows, base_ts, "kv_telemetry.scheduler.end")
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
        replay_prefill_summary = summarize_timed_events(replay_prefill_events)
        replay_prefill_end_ms = ""
        replay_ttft_ms = as_float(replay.get("ttft_ms", ""))
        replay_start_ms = as_float(replay.get("start_ms", ""))
        if replay_start_ms is not None and replay_ttft_ms is not None:
            replay_prefill_end_ms = round(replay_start_ms + replay_ttft_ms, 3)
        margin = ""
        due_ms = due.get("ms", "")
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
        "token_start",
        "token_end",
        "token_count",
        "node_id",
        "current_state",
        "write_host_events",
        "evict_gpu_events",
        "evict_host_events",
        "load_gpu_events",
        "lost_before_replay",
        "confidence",
    ]
    return [{column: row.get(column, "") for column in columns} for row in selected]


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
    grouped: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"events": 0, "with_agent_session": 0, "with_request_id": 0})
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
        request_id = row.get("request_id") or context.get("request_id")
        req = context.get("request")
        if not request_id and isinstance(req, dict):
            request_id = req.get("rid") or req.get("request_id")
        item = grouped[group]
        item["events"] += 1
        if sessions or row.get("session_id") or row.get("agent_session_id"):
            item["with_agent_session"] += 1
        if request_id:
            item["with_request_id"] += 1
    rows: list[dict[str, Any]] = []
    for group, counts in sorted(grouped.items()):
        total = counts["events"]
        rows.append(
            {
                "trace_area": group,
                "events": total,
                "with_agent_session": counts["with_agent_session"],
                "agent_session_coverage_pct": round(counts["with_agent_session"] * 100.0 / total, 2) if total else 0.0,
                "with_request_id": counts["with_request_id"],
                "request_id_coverage_pct": round(counts["with_request_id"] * 100.0 / total, 2) if total else 0.0,
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
                "replay_due_to_request_start_ms": request_start_delay,
                "replay_due_to_h2d_start_ms": start_delay,
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


def replay_h2d_readiness_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    margins = [float(row["h2d_finish_margin_ms"]) for row in rows if row.get("h2d_finish_margin_ms") not in ("", None)]
    request_start_delays = [float(row["replay_due_to_request_start_ms"]) for row in rows if row.get("replay_due_to_request_start_ms") not in ("", None)]
    start_delays = [float(row["replay_due_to_h2d_start_ms"]) for row in rows if row.get("replay_due_to_h2d_start_ms") not in ("", None)]
    request_to_h2d_start = [float(row["request_start_to_h2d_start_ms"]) for row in rows if row.get("request_start_to_h2d_start_ms") not in ("", None)]
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
            "avg_due_to_request_start_ms": avg(request_start_delays),
            "avg_due_to_h2d_start_ms": avg(start_delays),
            "avg_request_start_to_h2d_start_ms": avg(request_to_h2d_start),
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
    height = 560
    left = 96
    right = 40
    top = 70
    bottom = 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    numeric_keys = [
        "replay_due_to_request_start_ms",
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
            return left + plot_w / 2
        return left + index * plot_w / (len(rows) - 1)

    def y_pos(value: float) -> float:
        scaled = h2d_symlog_value(value)
        return top + (scaled - scaled_min) * plot_h / (scaled_max - scaled_min)

    def circle(x: float, y: float, color: str, title: str) -> str:
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" opacity="0.9" '
            f'stroke="#ffffff" stroke-width="1.5"><title>{html.escape(title)}</title></circle>'
        )

    def square(x: float, y: float, color: str, title: str) -> str:
        return (
            f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2" fill="{color}" '
            f'opacity="0.92" stroke="#ffffff" stroke-width="1.5"><title>{html.escape(title)}</title></rect>'
        )

    def triangle(x: float, y: float, color: str, title: str) -> str:
        points = f"{x:.1f},{y - 7:.1f} {x - 7:.1f},{y + 6:.1f} {x + 7:.1f},{y + 6:.1f}"
        return (
            f'<polygon points="{points}" fill="{color}" opacity="0.92" stroke="#ffffff" '
            f'stroke-width="1.5"><title>{html.escape(title)}</title></polygon>'
        )

    zero_y = y_pos(0.0)
    parts = [
        '<svg viewBox="0 0 1480 560" width="100%" role="img" aria-label="Replay request versus H2D start timeline plot">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{left + plot_w - 8}" y="{zero_y - 8:.1f}" text-anchor="end" font-size="12" font-weight="700">0 ms replay due</text>',
        '<text x="20" y="288" transform="rotate(-90 20 288)" text-anchor="middle" font-size="13" font-weight="700">time relative to replay due ms (symlog)</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="13" font-weight="700">no-prefetch replay gap order</text>',
        '<text x="104" y="36" font-size="13" fill="#334155" font-weight="700">higher = earlier/before due; lower = later/after due</text>',
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
        x = x_pos(index)
        request_start = as_float(row.get("replay_due_to_request_start_ms"))
        h2d_start = as_float(row.get("replay_due_to_h2d_start_ms"))
        h2d_end = as_float(row.get("replay_due_to_h2d_end_ms"))
        if request_start is not None and h2d_start is not None:
            y1 = y_pos(request_start)
            y2 = y_pos(h2d_start)
            parts.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="2" opacity="0.75"/>')
        if h2d_start is not None and h2d_end is not None:
            y1 = y_pos(h2d_start)
            y2 = y_pos(h2d_end)
            parts.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="#06b6d4" stroke-width="4" opacity="0.75"/>')
        title_prefix = (
            f"{row.get('session_id')} | fillers={row.get('fillers')} | tool_gap={row.get('tool_gap_ms')} ms | "
            f"TTFT={row.get('resume_ttft_ms')} ms"
        )
        if request_start is not None:
            parts.append(circle(x, y_pos(request_start), "#2563eb", f"{title_prefix} | replay request start={request_start:.3f} ms after due"))
        if h2d_start is not None:
            parts.append(triangle(x, y_pos(h2d_start), "#0891b2", f"{title_prefix} | H2D start={h2d_start:.3f} ms after due"))
        if h2d_end is not None:
            parts.append(square(x, y_pos(h2d_end), "#06b6d4", f"{title_prefix} | H2D finish={h2d_end:.3f} ms after due"))

    legend = [
        ("replay request start", "#2563eb", "circle"),
        ("H2D start", "#0891b2", "triangle"),
        ("H2D finish", "#06b6d4", "square"),
        ("request-to-H2D wait", "#94a3b8", "line"),
        ("visible H2D window", "#06b6d4", "line"),
    ]
    lx = left
    ly = height - 68
    for label, color, kind in legend:
        if kind == "circle":
            parts.append(f'<circle cx="{lx}" cy="{ly}" r="6" fill="{color}"/>')
        elif kind == "triangle":
            points = f"{lx},{ly - 7} {lx - 7},{ly + 6} {lx + 7},{ly + 6}"
            parts.append(f'<polygon points="{points}" fill="{color}"/>')
        elif kind == "square":
            parts.append(f'<rect x="{lx - 6}" y="{ly - 6}" width="12" height="12" rx="2" fill="{color}"/>')
        else:
            parts.append(f'<line x1="{lx - 8}" y1="{ly}" x2="{lx + 8}" y2="{ly}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{lx + 14}" y="{ly + 4}" font-size="12">{html.escape(label)}</text>')
        lx += 205
    parts.append("</svg>")
    return "\n".join(parts)


def global_replay_h2d_readiness_html(gaps: list[dict[str, Any]]) -> str:
    rows = replay_h2d_readiness_rows(gaps)
    if not rows:
        return """
        <p>No no-prefetch replay-side H2D movement was observed in this report.</p>
        <p class="note">That can mean the KV was still resident, replay recomputed instead of loading from host, or this run did not include no-prefetch rows with attributed H2D telemetry.</p>
        """
    summary = replay_h2d_readiness_summary(rows)
    buckets = replay_h2d_readiness_bucket_rows(rows)
    detail = rows[:80]
    return f"""
    <p>This no-prefetch chart answers a different question from prefetch margin: when replay needed KV, how late did the replay-side host-to-device KV load finish?</p>
    <p class="note">The dot value is <code>replay_due_time - replay_h2d_finish_time</code>. Positive means the KV load finished before the replay deadline. Negative means the replay deadline passed first, so the model turn had to wait for KV readiness.</p>
    <p class="note">The timing is split into three concrete pieces: <code>replay due -> H2D start</code>, <code>H2D start -> H2D end</code>, and <code>replay due -> H2D end</code>. This separates waiting before movement from the visible host-to-device movement window.</p>
    {table_html(summary)}
    <h3>Replay Request vs H2D Start</h3>
    <p>This chart checks whether the replay request itself was issued late, or whether the request arrived and then waited before visible KV H2D movement began.</p>
    <div class="setup-diagram">{build_replay_request_vs_h2d_timeline_plot(rows)}</div>
    <h3>Replay H2D Readiness Dot Plot</h3>
    <div class="setup-diagram">{build_replay_h2d_readiness_dot_plot(rows)}</div>
    <h3>Readiness Buckets</h3>
    {table_html(buckets)}
    <h3>Timing Split Behind The Plot</h3>
    {table_html(detail, ["order", "session_id", "fillers", "tool_gap_ms", "resume_ttft_ms", "replay_due_to_request_start_ms", "replay_due_to_h2d_start_ms", "request_start_to_h2d_start_ms", "h2d_visible_wall_window_ms", "h2d_event_duration_sum_ms", "request_start_to_h2d_end_ms", "replay_due_to_h2d_end_ms", "h2d_finish_margin_ms", "replay_h2d_events", "replay_h2d_tokens", "final_path", "simple_meaning"])}
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


def reproduce_controlled_replay_html(result_root: Path) -> str:
    run_master = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

EXPERIMENT_KIND=controlled \
REPORT_LABEL=controlled_demo_1 \
PRESSURE_PROFILE=high \
UPDATE_LATEST=1 \
MAX_TIMELINE_GAPS=18 \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
MODES="no_prefetch direct_prefetch" \
TOOL_WAIT_LIST_MS="100 250 500 1000" \
FILLER_LIST="32 64 128" \
REQUEST_CONCURRENCY=8 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.75 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
"""
    build_only = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

BUILD_ONLY=1 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL=controlled_demo_1_rebuild \
UPDATE_LATEST=0 \
CONTROLLED_ROOT=artifacts/results/runs/controlled/controlled_demo_1 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
"""
    dry_run = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

DRY_RUN=1 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL=controlled_demo_1 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
"""
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
            code_block("artifacts/results/latest_master_report.html\nartifacts/results/reports/controlled_demo_1/master_report.html\nartifacts/results/latest_manifest.json"),
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
    run_environment: dict[str, Any] | None = None,
) -> str:
    mode_rows = mode_summary_rows(gaps)
    ledger = build_replay_path_ledger(gaps)
    request_coverage = request_coverage or []
    kv_block_rows = kv_block_rows or []
    run_environment = run_environment or {}
    interesting = timeline_rows_with_labels(selected_timeline_gaps(gaps, max_timeline_gaps))
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
        ("timeline-guide", "How To Read Timelines"),
        ("replay-proof", "Replay Path Proof"),
        ("bottlenecks", "Bottleneck Breakdown"),
        ("counterfactual", "Hardware Opportunity"),
        ("replay-attribution", "Replay Path Attribution"),
        ("timelines", "Mixed Timeline Sample"),
        ("readable-phase-timeline", "Readable Phase Timeline"),
        ("kv-lifecycle", "KV Lifecycle Evidence"),
        ("kv-block-ledger", "KV Block Ledger"),
        ("replay-execution-timeline", "Replay Execution Timeline"),
        ("observations", "Key Observations"),
        ("performance", "Mode Tables"),
        ("direct-kv", "Direct KV Evidence"),
        ("appendix", "Gap Details"),
        ("reproduce", "Reproduce This Report"),
    ]
    if live_run:
        toc.insert(7, ("live-direct", "Live Direct Prefetch"))
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

  <details id="timeline-guide" class="section-card theme-guide">
    <summary><h2>How To Read The Timelines</h2></summary>
    {timeline_guide_html(profiled_available=True)}
  </details>

  <details id="replay-proof" class="section-card theme-directkv">
    <summary><h2>Replay Path Proof Table</h2></summary>
    <p>This is the main evidence ledger. Each row explains what happened when the replay request resumed: whether it reused cache, loaded KV from host to GPU, recomputed missing tokens, or mostly waited in the scheduler/request path.</p>
    <p class="note">Confidence matters. High confidence means the row has direct SGLang evidence plus HtoD movement evidence. Medium confidence means SGLang counters support the label. Low confidence means the label still depends mostly on TTFT and timeline shape.</p>
    <p class="note"><code>matched_prefix_tokens</code> means the first cache match visible when replay started. <code>final_cached_prefix_tokens</code> means how much prefix existed later after replay/cache work progressed. A large difference between them is evidence of replay-side prefill/recompute work, not a clean cache hit.</p>
    {table_html(replay_path_proof_rows(ledger), limit=250)}
  </details>

  <details id="bottlenecks" class="section-card theme-observations">
    <summary><h2>Bottleneck Breakdown</h2></summary>
    <p>This section groups the replay rows by the bottleneck label used in the proof table.</p>
    <h3>Bottleneck Summary</h3>
    {table_html(bottleneck_summary_rows(ledger))}
    <h3>Confidence Summary</h3>
    {table_html(confidence_summary_rows(ledger))}
    <h3>Instrumentation Coverage</h3>
    {table_html(instrumentation_coverage_rows(gaps, ledger))}
    <h3>Request-ID Plumbing Audit</h3>
    <p>This table checks whether the agent/session identity reached each SGLang area. Higher coverage means stronger attribution.</p>
    {table_html(request_coverage)}
  </details>

  <details id="counterfactual" class="section-card theme-deductions">
    <summary><h2>Counterfactual Hardware Opportunity</h2></summary>
    <p>This section asks a narrow question: when software prefetch was late, was the measured copy work small enough that a deadline-aware, priority-aware hardware path might plausibly have finished it inside the tool gap?</p>
    <h3>Counterfactual Summary</h3>
    {table_html(counterfactual_summary_rows(ledger))}
    <h3>Per-Row Counterfactual</h3>
    {table_html(hardware_counterfactual_rows(ledger), limit=250)}
  </details>

  <details id="replay-attribution" class="section-card theme-directkv">
    <summary><h2>Replay Path Attribution</h2></summary>
    <p>This section turns the segmented TTFT window into stronger evidence. For each replay, it reports SGLang prefix/cache counters observed inside the replay window: prompt tokens, cached prefix tokens, estimated new prefill tokens, host-hit tokens, host-load tokens, and replay-side HtoD events.</p>
    <p class="note">The verdict is evidence-backed but still conservative. Initial cached-prefix tokens show what was reusable when replay began. Final cached-prefix tokens show what existed later after replay work. A cyan bar plus host-load tokens is stronger proof that replay loaded KV from host to GPU.</p>
    <h3>Verdict Summary</h3>
    {table_html(verdict_summary_rows(gaps))}
    <h3>Replay Attribution Rows</h3>
    {table_html(replay_attribution_rows(gaps), limit=200)}
  </details>

  <details id="timelines" class="section-card theme-clean">
    <summary><h2>Mixed Timeline Sample / Deadline Timeline</h2></summary>
    <p class="note">This is the deadline view. The black line is when replay was due. This view is best for seeing whether the purple prefetch attempt finished before the deadline.</p>
    <h3>Timeline Model</h3>
    {timeline_model_table_html()}
    {build_expanded_gap_timeline_svg(interesting, max_timeline_gaps, show_prefetch_legend=True, scale="symlog")}
    <h3>KV Outcome For Timeline Rows</h3>
    <p>This table uses the same row names as the timeline. It explains whether replay reused KV, loaded KV from host, recomputed missing prefix tokens, or had a late/wasted prefetch.</p>
    {table_html(timeline_kv_outcome_rows(interesting))}
    <h3>Timeline Row Map</h3>
    <p>This table maps the compact row names in the chart back to the full experiment details.</p>
    {table_html(timeline_mapping_rows(interesting))}
  </details>

  <details id="readable-phase-timeline" class="section-card theme-clean">
    <summary><h2>Readable Phase Timeline</h2></summary>
    <p class="note">This view keeps readable phase columns, but each column has its own local timing. In simple words: the bars inside <code>prefetch</code> are positioned relative to the prefetch attempt, and the bars inside <code>replay path</code> are positioned relative to the replay request.</p>
    <p class="note">Replay work is split into separate lanes: cyan means replay-side KV HtoD, magenta means recompute/rebuild, gold means remaining before-first-token work, and red means decode after first token.</p>
    <h3>Timeline Model</h3>
    {timeline_model_table_html()}
    {build_local_timing_phase_timeline_svg(interesting, max_timeline_gaps, show_prefetch_legend=True)}
  </details>

  <details id="kv-lifecycle" class="section-card theme-directkv">
    <summary><h2>KV Lifecycle Evidence</h2></summary>
    <p>This section follows the KV lifecycle for the same rows shown in the timeline. It answers five simple questions for each tool gap:</p>
    <ol>
      <li>Did SGLang write this session's KV to host HiCache?</li>
      <li>Was that KV evicted from GPU memory?</li>
      <li>Was the host copy also evicted?</li>
      <li>Did prefetch or replay load the old KV back from host to GPU?</li>
      <li>If not, did replay have to rebuild/prefill missing KV?</li>
    </ol>
    <p class="note">The <code>simple_meaning</code> column is the fastest way to read this table. Example: if host write, GPU eviction, and host eviction are all nonzero, but replay H2D is zero and replay only matched a tiny prefix, then the old KV was lost and replay rebuilt/prefilled.</p>
    <h3>Column Legend</h3>
    {table_html(kv_lifecycle_legend_rows())}
    <h3>Per-Row KV Lifecycle</h3>
    {table_html(kv_lifecycle_evidence_rows(interesting))}
  </details>

  <details id="kv-block-ledger" class="section-card theme-directkv">
    <summary><h2>KV Block Ledger</h2></summary>
    <p>This section tracks logical KV blocks across SGLang cache events. It is more detailed than the timeline: each block has a stable ledger row showing whether it was written to host, evicted from GPU, evicted from host, or loaded back.</p>
    <p class="note">This is logical block tracking, not a physical GPU page snooper. The ledger uses SGLang node IDs when available and nearby token-range matching when node IDs are missing.</p>
    <h3>Block Ledger Summary</h3>
    {table_html(ledger_summary_rows(kv_block_rows))}
    <h3>Per-Gap Block Summary</h3>
    {table_html(kv_block_gap_table_rows(interesting, kv_block_rows))}
    <h3>Per-Block Ledger Rows</h3>
    {table_html(kv_block_detail_rows(kv_block_rows, limit=500))}
  </details>

  <details id="replay-execution-timeline" class="section-card theme-clean">
    <summary><h2>Replay Execution Timeline</h2></summary>
    <p class="note">This is the replay view. Each row is aligned at the actual resume request start. Cyan, magenta, and gold show the before-first-token work; red shows decode after first token.</p>
    <h3>Timeline Model</h3>
    {timeline_model_table_html()}
    {build_replay_execution_timeline_svg(interesting, max_timeline_gaps, show_prefetch_legend=True)}
    <h3>Replay Timeline Row Map</h3>
    {table_html(timeline_mapping_rows(interesting))}
  </details>

  {live_section}

  <details id="observations" class="section-card theme-observations">
    <summary><h2>Key Observations Per Gap/Session</h2></summary>
    <p>This section translates the timeline rows into plain English. It uses the same compact row names as the chart, so <code>G00</code> here means the same <code>G00</code> in the timeline.</p>
    {table_html(key_observation_rows(interesting), ["row", "mode", "status", "what happened", "why it matters", "tool_wait_ms", "resume_ttft_ms", "replay_path", "verdict"])}
  </details>

  <details id="performance" class="section-card theme-clean-table">
    <summary><h2>Mode Tables</h2></summary>
    <h3>Mode Summary</h3>
    {table_html(mode_rows)}
  </details>

  <details id="direct-kv" class="section-card theme-directkv">
    <summary><h2>Direct KV Load Evidence</h2></summary>
    <p>Green bars and <code>direct_kv_h2d_*</code> columns come from SGLang-level KV movement hooks and lightweight copy telemetry during the prefetch attempt. Cyan/replay columns show KV movement performed by the real resume request.</p>
    {table_html(gaps, ["session_id", "mode", "tool_gap_ms", "prefetch_margin_ms", "resume_ttft_ms", "final_path", "bottleneck_label", "path_confidence", "prefetch_outcome", "movement_class", "direct_kv_h2d_events", "direct_kv_h2d_duration_ms", "replay_kv_h2d_events", "replay_kv_h2d_duration_ms", "replay_input_tokens", "replay_active_input_tokens", "replay_scheduler_trimmed_tokens", "replay_cached_prefix_tokens", "replay_final_cached_prefix_tokens", "replay_cache_hit_ratio_pct", "replay_new_prefill_tokens_est", "replay_progressive_cache_events", "replay_post_request_cache_write_events", "replay_host_hit_tokens", "replay_host_load_tokens", "scheduler_wait_ms", "kv_prepare_ms", "model_forward_ms", "replay_scheduler_event_count", "replay_scheduler_total_ms", "replay_model_forward_event_count", "replay_model_forward_total_ms"])}
  </details>

  <details id="appendix" class="section-card theme-appendix">
    <summary><h2>Gap Details</h2></summary>
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
        for gap in gaps:
            gap["case_dir"] = str(case_dir)
        all_gaps.extend(gaps)
        all_trace_rows.extend(trace_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = build_replay_path_ledger(all_gaps)
    normalized_kv_events = normalize_sglang_trace_events(all_trace_rows)
    kv_ledger = build_block_ledger(normalized_kv_events)
    kv_block_rows = block_ledger_rows(kv_ledger)
    request_coverage = request_id_coverage_rows(all_trace_rows)
    h2d_readiness = replay_h2d_readiness_rows(all_gaps)
    write_csv(args.out_dir / "controlled_replay_gaps.csv", all_gaps)
    write_csv(args.out_dir / "replay_path_ledger.csv", ledger)
    write_csv(args.out_dir / "replay_h2d_readiness.csv", h2d_readiness)
    write_csv(args.out_dir / "hardware_counterfactual.csv", hardware_counterfactual_rows(ledger))
    write_csv(args.out_dir / "instrumentation_coverage.csv", instrumentation_coverage_rows(all_gaps, ledger))
    write_csv(args.out_dir / "request_id_coverage_report.csv", request_coverage)
    write_ledger_artifacts(args.out_dir, kv_block_rows, all_gaps)
    write_json(
        args.out_dir / "controlled_replay_report.json",
        {
            "gaps": all_gaps,
            "summary": mode_summary_rows(all_gaps),
            "replay_path_ledger": ledger,
            "replay_h2d_readiness": h2d_readiness,
            "replay_h2d_readiness_summary": replay_h2d_readiness_summary(h2d_readiness),
            "kv_block_ledger": kv_block_rows,
            "kv_block_lifecycle_summary": ledger_summary_rows(kv_block_rows),
            "kv_block_gap_summary": gap_lifecycle_summary_rows(all_gaps, kv_block_rows),
            "bottleneck_summary": bottleneck_summary_rows(ledger),
            "confidence_summary": confidence_summary_rows(ledger),
            "counterfactual_summary": counterfactual_summary_rows(ledger),
            "instrumentation_coverage": instrumentation_coverage_rows(all_gaps, ledger),
            "request_id_coverage": request_coverage,
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
