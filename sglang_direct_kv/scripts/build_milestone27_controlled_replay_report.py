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
    replay_phase_segments,
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

PREFETCH_MODE_NAMES = {
    "direct_prefetch",
    "priority_direct_prefetch",
    "deadline_priority_prefetch",
}
HARDWARE_BYPASS_LEVELS: list[tuple[str, str, float]] = [
    ("best_case", "Best case: measured H2D only", 0.0),
    ("realistic", "Realistic: measured H2D + 50 ms", 50.0),
    ("conservative", "Conservative: measured H2D + 150 ms", 150.0),
]


def canonical_mode(mode: Any) -> str:
    raw = str(mode or "")
    if raw.startswith("projected_hardware"):
        return "projected_hardware_bypass"
    if raw.startswith("deadline_priority_prefetch"):
        return "deadline_priority_prefetch"
    if raw.startswith("dynamo_priority_hints"):
        return "dynamo_priority_hints"
    if raw.startswith("priority_direct_prefetch"):
        return "priority_direct_prefetch"
    if raw.startswith("direct_prefetch"):
        return "direct_prefetch"
    if raw.startswith("no_prefetch"):
        return "no_prefetch"
    if raw.startswith("oracle_direct_load"):
        return "oracle_direct_load"
    if raw.startswith("oracle_prefetch"):
        return "oracle_prefetch"
    return raw


def display_mode(mode: Any) -> str:
    labels = {
        "no_prefetch": "No prefetch",
        "direct_prefetch": "Direct prefetch",
        "dynamo_priority_hints": "Dynamo priority hints only",
        "priority_direct_prefetch": "Priority direct prefetch",
        "deadline_priority_prefetch": "Deadline priority prefetch",
        "projected_hardware_bypass": "Projected hardware bypass",
        "oracle_direct_load": "Oracle direct load",
        "oracle_prefetch": "Oracle prefetch",
    }
    return labels.get(canonical_mode(mode), str(mode or "unknown"))


def mode_badge_style(mode: Any) -> tuple[str, str]:
    return {
        "no_prefetch": ("#334155", "#f1f5f9"),
        "direct_prefetch": ("#7c3aed", "#f3e8ff"),
        "dynamo_priority_hints": ("#b45309", "#fffbeb"),
        "priority_direct_prefetch": ("#0891b2", "#ecfeff"),
        "deadline_priority_prefetch": ("#16a34a", "#dcfce7"),
        "projected_hardware_bypass": ("#0f766e", "#f0fdfa"),
        "oracle_direct_load": ("#ea580c", "#ffedd5"),
        "oracle_prefetch": ("#ea580c", "#ffedd5"),
    }.get(canonical_mode(mode), ("#475569", "#f8fafc"))


def mode_row_background_style(mode: Any, row_index: int = 0) -> tuple[str, str, float]:
    """Return a subtle row tint, left accent color, and opacity for mode grouping."""
    canonical = canonical_mode(mode)
    base = {
        "no_prefetch": ("#f8fafc", "#64748b", 0.92),
        "direct_prefetch": ("#f0f9ff", "#0284c7", 0.88),
        "dynamo_priority_hints": ("#fffbeb", "#f59e0b", 0.90),
        "priority_direct_prefetch": ("#ecfeff", "#0891b2", 0.88),
        "deadline_priority_prefetch": ("#f0fdf4", "#16a34a", 0.88),
        "projected_hardware_bypass": ("#ecfdf5", "#0f766e", 0.92),
        "oracle_direct_load": ("#fff7ed", "#ea580c", 0.88),
        "oracle_prefetch": ("#fff7ed", "#ea580c", 0.88),
    }.get(canonical)
    if base:
        return base
    return ("#ffffff" if row_index % 2 == 0 else "#eef4fb", "#cbd5e1", 0.92)


def display_verdict(verdict: Any) -> str:
    labels = {
        "no_prefetch_replay_loaded_kv": "No prefetch; replay loaded KV",
        "no_prefetch_replay_recomputed": "No prefetch; replay recomputed",
        "no_prefetch_cache_reused_or_scheduler_wait": "No prefetch; cache reused or scheduler wait",
        "prefetch_late_replay_loaded_kv": "Prefetch late; replay loaded KV",
        "prefetch_late_replay_recomputed": "Prefetch late; replay recomputed",
        "prefetch_late_no_replay_h2d": "Prefetch late; no replay H2D",
        "prefetch_ready_but_replay_loaded_kv": "Prefetch ready, but replay still loaded KV",
        "prefetch_success_cache_reused": "Prefetch success; replay reused cache",
        "prefetch_ready_but_replay_recomputed": "Prefetch ready, but replay recomputed",
        "prefetch_no_host_load_replay_cache_hit": "Prefetch found no host KV; replay cache hit",
        "prefetch_ready_replay_cache_hit": "Prefetch ready; replay cache hit",
        "prefetch_ran_but_no_host_kv": "Prefetch ran, but no host KV was available",
        "prefetch_ready_no_replay_h2d": "Prefetch ready; no replay H2D",
        "prefetch_missing_or_unfinished": "Prefetch missing or unfinished",
        "PROJECTED HARDWARE - not measured": "Projected hardware - not measured",
        "projected_hardware_bypass": "Projected hardware bypass",
        "true_kv_prefetch_success": "True KV prefetch success",
        "hint_completed_early_but_no_kv_load_seen": "Hint early, but no KV load seen",
        "hint_completed_early_but_replay_recomputed": "Hint early, but replay recomputed",
        "hint_loaded_kv_but_evicted_before_replay": "Hint loaded KV, but residency was lost",
        "hint_loaded_kv_but_replay_reloaded": "Hint loaded KV, but replay reloaded KV",
        "hint_late": "Hint late",
        "no_reuse_evidence": "No strong reuse evidence",
        "no_prefetch_baseline": "No prefetch baseline",
    }
    return labels.get(str(verdict or ""), str(verdict or "unknown"))


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


def replay_recompute_seen(row: dict[str, Any]) -> bool:
    """Return true when replay did visible prefill/recompute work after the hint."""
    for key in (
        "replay_runtime_prefill_attributed_tokens",
        "replay_new_prefill_tokens_est",
        "recomputed_tokens_est",
        "replay_prefill_recompute_event_count",
    ):
        value = as_float(row.get(key))
        if value is not None and value > 0:
            return True
    path = str(row.get("replay_path") or replay_path_from_evidence(row))
    return path == "replay prefill/recompute path suspected"


def prefetch_truth_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Classify useful KV prefetch more strictly than request-completion timing."""
    mode = canonical_mode(row.get("mode"))
    if mode == "no_prefetch":
        return {
            "prefetch_truth_verdict": "no_prefetch_baseline",
            "prefetch_truth_short_label": "NO PREFETCH",
            "prefetch_truth_explanation": "No prefetch hint was issued for this baseline row.",
            "prefetch_truth_confidence": "exact",
            "hint_completed_before_replay": "",
            "hint_h2d_seen": "",
            "replay_reloaded_after_hint": "",
            "replay_recomputed_after_hint": "",
            "true_kv_prefetch_success": "",
        }
    if mode == "projected_hardware_bypass":
        return {
            "prefetch_truth_verdict": "projected_hardware_bypass",
            "prefetch_truth_short_label": "PROJECTED, NOT MEASURED",
            "prefetch_truth_explanation": "This is a projected hardware counterfactual, not a measured SGLang prefetch outcome.",
            "prefetch_truth_confidence": "projected",
            "hint_completed_before_replay": "",
            "hint_h2d_seen": "",
            "replay_reloaded_after_hint": "",
            "replay_recomputed_after_hint": "",
            "true_kv_prefetch_success": "",
        }
    if mode == "dynamo_priority_hints":
        return {
            "prefetch_truth_verdict": "priority_hints_only_no_direct_prefetch",
            "prefetch_truth_short_label": "PRIORITY HINTS ONLY",
            "prefetch_truth_explanation": (
                "This mode sends Dynamo-style priority metadata and an SGLang priority value, "
                "but it does not issue the direct KV prefetch hook."
            ),
            "prefetch_truth_confidence": "exact_driver_mode",
            "hint_completed_before_replay": "",
            "hint_h2d_seen": "",
            "replay_reloaded_after_hint": "",
            "replay_recomputed_after_hint": "",
            "true_kv_prefetch_success": "",
        }

    margin = as_float(row.get("prefetch_margin_ms"))
    hint_completed = margin is not None
    hint_early = bool(margin is not None and margin >= 0)
    hint_h2d = has_events(row.get("direct_kv_h2d_events")) or has_events(row.get("lifecycle_hint_h2d_tokens"))
    replay_h2d = has_events(row.get("replay_kv_h2d_events")) or has_events(row.get("lifecycle_replay_h2d_tokens"))
    replay_recompute = replay_recompute_seen(row)
    lifecycle = str(row.get("lifecycle_verdict") or "")
    host_lost = "host_evicted" in lifecycle or "missing" in lifecycle
    gpu_evicted_no_load = "gpu_evicted_no_replay_load" in lifecycle

    if not hint_completed:
        verdict = "no_reuse_evidence"
        label = "NO COMPLETED HINT"
        explanation = "The trace did not show the hint request finishing, so we cannot claim useful prefetch."
        confidence = "exact"
        success = 0
    elif not hint_early:
        verdict = "hint_late"
        label = "HINT LATE"
        explanation = f"The hint request finished {abs(margin):.1f} ms after replay was due."
        confidence = "exact"
        success = 0
    elif hint_h2d and (host_lost or gpu_evicted_no_load):
        verdict = "hint_loaded_kv_but_evicted_before_replay"
        label = "HINT LOADED; RESIDENCY LOST"
        explanation = "The hint loaded KV, but lifecycle evidence says useful residency was lost before replay could use it."
        confidence = "direct_lifecycle_evidence"
        success = 0
    elif hint_h2d and replay_h2d:
        verdict = "hint_loaded_kv_but_replay_reloaded"
        label = "HINT LOADED; REPLAY RELOADED"
        explanation = "The hint loaded KV, but replay still performed host-to-device KV loading, so reuse was not proven."
        confidence = "direct_h2d_evidence"
        success = 0
    elif hint_early and replay_recompute:
        verdict = "hint_completed_early_but_replay_recomputed"
        label = "HINT EARLY; REPLAY RECOMPUTED"
        explanation = "The hint request completed before replay, but replay still did prefill/recompute work."
        confidence = "direct_or_estimated_replay_prefill_evidence"
        success = 0
    elif hint_early and not hint_h2d:
        verdict = "hint_completed_early_but_no_kv_load_seen"
        label = "HINT EARLY; NO KV LOAD SEEN"
        explanation = "The hint request completed before replay, but no hint-side host-to-device KV movement was observed."
        confidence = "exact_request_timing_no_h2d_evidence"
        success = 0
    elif hint_early and hint_h2d and not replay_h2d and not replay_recompute:
        verdict = "true_kv_prefetch_success"
        label = "TRUE KV PREFETCH"
        explanation = "The hint loaded KV before replay, and replay showed no reload/recompute evidence for that row."
        confidence = "strong_direct_evidence"
        success = 1
    else:
        verdict = "no_reuse_evidence"
        label = "NO STRONG REUSE EVIDENCE"
        explanation = "The trace is not strong enough to prove that the hinted KV became useful replay residency."
        confidence = "insufficient"
        success = 0

    return {
        "prefetch_truth_verdict": verdict,
        "prefetch_truth_short_label": label,
        "prefetch_truth_explanation": explanation,
        "prefetch_truth_confidence": confidence,
        "hint_completed_before_replay": 1 if hint_early else 0 if hint_completed else "",
        "hint_h2d_seen": 1 if hint_h2d else 0,
        "replay_reloaded_after_hint": 1 if replay_h2d else 0,
        "replay_recomputed_after_hint": 1 if replay_recompute else 0,
        "true_kv_prefetch_success": success,
    }


def attach_prefetch_truth_fields(row: dict[str, Any]) -> None:
    row.update(prefetch_truth_fields(row))


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
    for key in ("request_prefill_attribution", "batch_request_prefill_attribution"):
        values = row.get(key)
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
                    "batch_object_id": row.get("batch_object_id", ""),
                    "batch_uncached_token_sum": row.get("batch_uncached_token_sum", ""),
                    "batch_full_token_sum": row.get("batch_full_token_sum", ""),
                    "batch_cached_prefix_token_sum": row.get("batch_cached_prefix_token_sum", ""),
                    "batch_requests_with_uncached_tokens": row.get("batch_requests_with_uncached_tokens", ""),
                    "batch_uncached_token_ranges_sample": row.get("batch_uncached_token_ranges_sample", ""),
                    "request_prefill_attribution": row.get("request_prefill_attribution", ""),
                    "batch_request_prefill_attribution": row.get("batch_request_prefill_attribution", ""),
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


def _prefill_attribution_items(event: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("request_prefill_attribution", "batch_request_prefill_attribution"):
        value = event.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def summarize_prefill_runtime_attribution(events: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
    """Summarize SGLang-runtime evidence for uncached replay prefill/recompute.

    This is stronger than the old report-only estimate because it uses request
    token attribution carried by the SGLang model-forward hooks. It is still
    runtime-level evidence, not per-CUDA-kernel hardware profiling.
    """

    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        call_id = str(event.get("call_id") or "")
        if not call_id:
            continue
        phase = str(event.get("phase") or "")
        if phase not in {"start", "end"}:
            continue
        paired[call_id][phase] = event

    candidates: list[dict[str, Any]] = []
    for call_id, pair in paired.items():
        start_event = pair.get("start") or {}
        end_event = pair.get("end") or {}
        source = end_event or start_event
        start = as_float(start_event.get("start_or_end_ms"))
        end = as_float(end_event.get("start_or_end_ms"))
        duration = as_float(end_event.get("duration_ms")) or as_float(start_event.get("duration_ms"))
        if start is None and end is not None and duration is not None:
            start = end - duration
        if end is None and start is not None and duration is not None:
            end = start + duration
        if start is None or end is None:
            continue

        matched_items = [
            item
            for item in [*_prefill_attribution_items(start_event), *_prefill_attribution_items(end_event)]
            if agent_session_from_context(item) == session_id
        ]
        if not matched_items:
            continue
        best = max(
            matched_items,
            key=lambda item: as_float(item.get("prefill_uncached_token_count")) or 0.0,
        )
        uncached = as_float(best.get("prefill_uncached_token_count"))
        if uncached is None or uncached <= 0:
            continue
        candidates.append(
            {
                "call_id": call_id,
                "start_ms": round(start, 3),
                "end_ms": round(end, 3),
                "duration_ms": round(end - start, 3),
                "method": source.get("method", ""),
                "category": source.get("category", ""),
                "batch_object_id": source.get("batch_object_id", ""),
                "batch_request_count": source.get("request_count", ""),
                "batch_extend_num_tokens": source.get("extend_num_tokens", ""),
                "batch_uncached_token_sum": source.get("batch_uncached_token_sum", ""),
                "batch_full_token_sum": source.get("batch_full_token_sum", ""),
                "batch_cached_prefix_token_sum": source.get("batch_cached_prefix_token_sum", ""),
                "request_uncached_tokens": int(uncached),
                "request_full_input_tokens": best.get("prefill_full_input_tokens", ""),
                "request_active_input_tokens": best.get("prefill_active_input_tokens", ""),
                "request_cached_prefix_tokens": best.get("prefill_cached_prefix_tokens", ""),
                "request_uncached_token_start": best.get("prefill_uncached_token_start", ""),
                "request_uncached_token_end": best.get("prefill_uncached_token_end", ""),
                "request_uncached_token_range": best.get("prefill_token_range", ""),
            }
        )

    if not candidates:
        return {
            "event_count": 0,
            "start_ms": "",
            "end_ms": "",
            "duration_ms": "",
            "tokens": "",
            "token_range": "",
            "batch_id": "",
            "batch_request_count": "",
            "batch_uncached_token_sum": "",
            "evidence": "no request-attributed model-forward batch observed",
            "confidence": "fallback_estimate",
        }

    candidates.sort(key=lambda item: (as_float(item.get("start_ms")) or 0.0, -(as_float(item.get("request_uncached_tokens")) or 0.0)))
    first = candidates[0]
    total_tokens = sum(int(as_float(item.get("request_uncached_tokens")) or 0) for item in candidates)
    ranges = [str(item.get("request_uncached_token_range")) for item in candidates if item.get("request_uncached_token_range")]
    return {
        "event_count": len(candidates),
        "start_ms": first["start_ms"],
        "end_ms": max(item["end_ms"] for item in candidates),
        "duration_ms": round(sum(float(item["duration_ms"]) for item in candidates), 3),
        "tokens": total_tokens,
        "token_range": ", ".join(ranges[:4]),
        "batch_id": first.get("batch_object_id", ""),
        "batch_request_count": first.get("batch_request_count", ""),
        "batch_uncached_token_sum": first.get("batch_uncached_token_sum", ""),
        "batch_extend_num_tokens": first.get("batch_extend_num_tokens", ""),
        "request_full_input_tokens": first.get("request_full_input_tokens", ""),
        "request_cached_prefix_tokens": first.get("request_cached_prefix_tokens", ""),
        "evidence": (
            f"SGLang model-forward hook attributed {int(total_tokens)} uncached tokens "
            f"for this replay session"
        ),
        "confidence": "runtime_attributed",
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
        return "replay prefill/recompute path suspected"
    if hit_ratio is not None and hit_ratio >= 90:
        return "mostly cache hit/resident"
    if ttft is not None and ttft >= 1000:
        return "scheduler/cache wait suspected"
    if ttft is None:
        return "unclear"
    return "likely cache hit/resident"


def per_gap_verdict(row: dict[str, Any]) -> str:
    mode = canonical_mode(row.get("mode"))
    margin = as_float(row.get("prefetch_margin_ms"))
    replay_loaded = has_events(row.get("replay_kv_h2d_events"))
    hint_loaded = has_events(row.get("direct_kv_h2d_events"))
    replay_path_value = str(row.get("replay_path") or replay_path_from_evidence(row))
    hint_host_hit = as_float(row.get("hint_host_hit_tokens"))
    if mode == "no_prefetch":
        if replay_loaded:
            return "no_prefetch_replay_loaded_kv"
        if replay_path_value == "replay prefill/recompute path suspected":
            return "no_prefetch_replay_recomputed"
        return "no_prefetch_cache_reused_or_scheduler_wait"
    if margin is None:
        return "prefetch_missing_or_unfinished"
    if margin < 0 and replay_loaded:
        return "prefetch_late_replay_loaded_kv"
    if margin < 0 and replay_path_value == "replay prefill/recompute path suspected":
        return "prefetch_late_replay_recomputed"
    if margin < 0:
        return "prefetch_late_no_replay_h2d"
    if replay_loaded:
        return "prefetch_ready_but_replay_loaded_kv"
    if hint_loaded and replay_path_value in {"mostly cache hit/resident", "likely cache hit/resident"}:
        return "prefetch_success_cache_reused"
    if replay_path_value == "replay prefill/recompute path suspected":
        return "prefetch_ready_but_replay_recomputed"
    if replay_path_value in {"mostly cache hit/resident", "likely cache hit/resident"}:
        if not hint_loaded and hint_host_hit == 0:
            return "prefetch_no_host_load_replay_cache_hit"
        return "prefetch_ready_replay_cache_hit"
    if not hint_loaded and hint_host_hit == 0:
        return "prefetch_ran_but_no_host_kv"
    return "prefetch_ready_no_replay_h2d"


def trace_request_windows(trace_rows: list[dict[str, Any]], base_ts: float) -> dict[tuple[str, str], dict[str, Any]]:
    bridge_fields = [
        "hint_source",
        "dynamo_agent_priority",
        "sglang_priority",
        "deadline_offset_ms",
        "priority_translation",
        "priority_policy",
    ]
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
        for field in bridge_fields:
            if row.get(field) not in ("", None):
                item[field] = row.get(field)
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
    prefill_by_session = telemetry_events_by_session(
        trace_rows,
        base_ts,
        {"kv_telemetry.prefill.start", "kv_telemetry.prefill.end"},
    )
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
        replay_prefill_attribution = summarize_prefill_runtime_attribution(replay_prefill_events, session)
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
            "replay_prefill_recompute_start_ms": replay_prefill_summary["start_ms"],
            "replay_prefill_recompute_end_ms": replay_prefill_summary["end_ms"],
            "replay_prefill_recompute_duration_ms": replay_prefill_summary["duration_ms"],
            "replay_prefill_recompute_event_count": replay_prefill_summary["event_count"],
            "replay_prefill_recompute_timing_source": (
                "exact_model_forward_hook" if replay_prefill_summary["event_count"] else "fallback_request_ttft_window"
            ),
            "replay_runtime_prefill_attribution_events": replay_prefill_attribution["event_count"],
            "replay_runtime_prefill_attribution_start_ms": replay_prefill_attribution["start_ms"],
            "replay_runtime_prefill_attribution_end_ms": replay_prefill_attribution["end_ms"],
            "replay_runtime_prefill_attribution_duration_ms": replay_prefill_attribution["duration_ms"],
            "replay_runtime_prefill_attributed_tokens": replay_prefill_attribution["tokens"],
            "replay_runtime_prefill_token_range": replay_prefill_attribution["token_range"],
            "replay_runtime_prefill_batch_id": replay_prefill_attribution["batch_id"],
            "replay_runtime_prefill_batch_request_count": replay_prefill_attribution["batch_request_count"],
            "replay_runtime_prefill_batch_uncached_token_sum": replay_prefill_attribution["batch_uncached_token_sum"],
            "replay_runtime_prefill_batch_extend_num_tokens": replay_prefill_attribution.get("batch_extend_num_tokens", ""),
            "replay_runtime_prefill_request_full_input_tokens": replay_prefill_attribution.get("request_full_input_tokens", ""),
            "replay_runtime_prefill_request_cached_prefix_tokens": replay_prefill_attribution.get("request_cached_prefix_tokens", ""),
            "replay_runtime_prefill_evidence": replay_prefill_attribution["evidence"],
            "replay_runtime_prefill_confidence": replay_prefill_attribution["confidence"],
            "pre_replay_checkpoint_ms": checkpoint.get("ms", ""),
            "pre_replay_expected_reuse": checkpoint.get("expected_reuse", ""),
            "pre_replay_gpu_resident_tokens": checkpoint.get("gpu_resident_tokens", ""),
            "pre_replay_host_resident_tokens": checkpoint.get("host_resident_tokens", ""),
            "pre_replay_missing_tokens": checkpoint.get("missing_tokens", ""),
            "pre_replay_protected_tokens": checkpoint.get("protected_tokens", ""),
            "hint_source": p_start.get("hint_source") or hint.get("hint_source") or replay.get("hint_source", ""),
            "hint_dynamo_agent_priority": p_start.get("dynamo_agent_priority") or hint.get("dynamo_agent_priority", ""),
            "hint_sglang_priority": p_start.get("sglang_priority") or hint.get("sglang_priority", ""),
            "hint_priority_translation": p_start.get("priority_translation") or hint.get("priority_translation", ""),
            "replay_hint_source": replay.get("hint_source", ""),
            "replay_dynamo_agent_priority": replay.get("dynamo_agent_priority", ""),
            "replay_sglang_priority": replay.get("sglang_priority", ""),
            "replay_priority_translation": replay.get("priority_translation", ""),
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
        attach_prefetch_truth_fields(gap)
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
        truth_verdicts = [str(row.get("prefetch_truth_verdict") or "") for row in items if row.get("prefetch_truth_verdict")]
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
                    1 for value in replay_paths if value in {"replay prefill/recompute path suspected", "scheduler/cache wait suspected"}
                ),
                "likely_cache_hit_or_resident_gaps": replay_paths.count("likely cache hit/resident"),
                "mostly_cache_hit_or_resident_gaps": replay_paths.count("mostly cache hit/resident"),
                "true_kv_prefetch_successes": sum(
                    1 for row in items if str(row.get("true_kv_prefetch_success") or "") == "1"
                ),
                "hint_completed_before_replay": sum(
                    1 for row in items if str(row.get("hint_completed_before_replay") or "") == "1"
                ),
                "verdicts": ", ".join(f"{name}:{count}" for name, count in sorted(Counter(verdicts).items())),
                "prefetch_truth_verdicts": ", ".join(
                    f"{name}:{count}" for name, count in sorted(Counter(truth_verdicts).items())
                ),
            }
        )
    return rows


def prefetch_truth_table_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(timeline_rows_with_labels(selected_timeline_gaps(gaps, len(gaps)))):
        mode = canonical_mode(row.get("mode"))
        if mode == "projected_hardware_bypass":
            continue
        rows.append(
            {
                "row": row.get("timeline_label") or f"G{idx:02d}",
                "mode_key": mode,
                "mode": display_mode(mode),
                "case_id": row.get("case_id", ""),
                "session_id": row.get("session_id", ""),
                "task": row.get("task_index", ""),
                "gap": row.get("gap_order_in_task", ""),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "fillers": case_fillers(row),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "hint_completed_before_replay": row.get("hint_completed_before_replay", ""),
                "hint_h2d_seen": row.get("hint_h2d_seen", ""),
                "replay_reloaded_after_hint": row.get("replay_reloaded_after_hint", ""),
                "replay_recomputed_after_hint": row.get("replay_recomputed_after_hint", ""),
                "true_kv_prefetch_success": row.get("true_kv_prefetch_success", ""),
                "prefetch_truth_verdict": row.get("prefetch_truth_verdict", ""),
                "prefetch_truth_explanation": row.get("prefetch_truth_explanation", ""),
                "prefetch_truth_confidence": row.get("prefetch_truth_confidence", ""),
                "direct_kv_h2d_events": row.get("direct_kv_h2d_events", ""),
                "replay_kv_h2d_events": row.get("replay_kv_h2d_events", ""),
                "replay_runtime_prefill_attributed_tokens": row.get("replay_runtime_prefill_attributed_tokens", ""),
                "lifecycle_verdict": row.get("lifecycle_verdict", ""),
            }
        )
    return rows


def prefetch_truth_summary_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = prefetch_truth_table_rows(gaps)
    by_mode: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("mode") or "")].append(row)
    output: list[dict[str, Any]] = []
    for mode, items in sorted(by_mode.items()):
        verdicts = Counter(str(row.get("prefetch_truth_verdict") or "") for row in items)
        attempts = [row for row in items if row.get("prefetch_margin_ms") not in ("", None)]
        true_successes = sum(1 for row in items if str(row.get("true_kv_prefetch_success") or "") == "1")
        output.append(
            {
                "mode": mode,
                "rows": len(items),
                "prefetch_attempts": len(attempts),
                "hint_completed_before_replay": sum(
                    1 for row in items if str(row.get("hint_completed_before_replay") or "") == "1"
                ),
                "true_kv_prefetch_successes": true_successes,
                "true_success_pct": round(true_successes * 100.0 / len(attempts), 2) if attempts else "",
                "truth_verdicts": ", ".join(f"{name}:{count}" for name, count in sorted(verdicts.items()) if name),
            }
        )
    return output


def prefetch_truth_metric_cards(gaps: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    rows = [
        row
        for row in prefetch_truth_table_rows(gaps)
        if str(row.get("mode_key") or "") != "no_prefetch"
    ]
    attempts = [row for row in rows if row.get("prefetch_margin_ms") not in ("", None)]
    early = [row for row in rows if str(row.get("hint_completed_before_replay") or "") == "1"]
    successes = [row for row in rows if str(row.get("true_kv_prefetch_success") or "") == "1"]
    replay_reloaded = [row for row in rows if str(row.get("replay_reloaded_after_hint") or "") == "1"]
    replay_recomputed = [row for row in rows if str(row.get("replay_recomputed_after_hint") or "") == "1"]
    early_not_proven = max(0, len(early) - len(successes))
    return [
        ("prefetch attempts", str(len(attempts)), "Measured software hint/direct-load attempts"),
        ("hints finished before replay", str(len(early)), "The purple path completed before the replay deadline"),
        ("true KV prefetch successes", str(len(successes)), "Strict successes with useful KV residency/reuse evidence"),
        ("early but not proven useful", str(early_not_proven), "Hint finished early, but replay still reloaded/recomputed or no KV load was seen"),
        ("replay reloaded after hint", str(len(replay_reloaded)), "Replay still performed KV H2D after a prefetch-mode hint"),
        ("replay recomputed after hint", str(len(replay_recomputed)), "Replay still did prefill/recompute work after a prefetch-mode hint"),
    ]


def timeline_mode_rank(row: dict[str, Any]) -> tuple[int, str]:
    mode = canonical_mode(row.get("mode"))
    ranks = {
        "no_prefetch": 0,
        "direct_prefetch": 1,
        "dynamo_priority_hints": 2,
        "deadline_priority_prefetch": 3,
        "priority_direct_prefetch": 4,
        "oracle_prefetch": 4,
        "oracle_direct_load": 4,
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
        if canonical_mode(row.get("mode")) in PREFETCH_MODE_NAMES
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


def scenario_compare_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("task_index") or ""),
        str(row.get("gap_order_in_task") or ""),
        str(row.get("tool_gap_ms") or ""),
        str(case_fillers(row) or ""),
    )


def scenario_compare_sort_key(key: tuple[str, str, str, str]) -> tuple[int, int, float, float, str]:
    task, gap, wait, fillers = key
    task_rank = int(task) if str(task).isdigit() else 10**9
    gap_rank = int(gap) if str(gap).isdigit() else 10**9
    wait_rank = as_float(wait)
    filler_rank = as_float(fillers)
    return (
        task_rank,
        gap_rank,
        wait_rank if wait_rank is not None else 10**12,
        filler_rank if filler_rank is not None else 10**12,
        "|".join(key),
    )


def mode_short_label(mode: Any) -> str:
    labels = {
        "no_prefetch": "NP",
        "direct_prefetch": "DP",
        "dynamo_priority_hints": "DH",
        "deadline_priority_prefetch": "DLP",
        "projected_hardware_bypass": "HW",
        "priority_direct_prefetch": "PDP",
        "oracle_direct_load": "ODL",
        "oracle_prefetch": "OP",
    }
    return labels.get(canonical_mode(mode), str(mode or "M")[:3].upper())


def projected_hardware_timeline_row(
    source_row: dict[str, Any],
    scenario_label: str,
) -> dict[str, Any] | None:
    projection = next(
        (
            row
            for row in projected_hardware_bypass_rows([source_row])
            if row.get("hardware_projection") == "realistic"
        ),
        None,
    )
    if not projection:
        return None
    projected_start = as_float(projection.get("projected_hardware_start_ms"))
    projected_end = as_float(projection.get("projected_hardware_end_ms"))
    projected_duration = as_float(projection.get("projected_hardware_duration_ms"))
    projected_margin = as_float(projection.get("projected_hardware_margin_ms"))
    if projected_start is None or projected_end is None:
        return None

    copied = dict(source_row)
    copied["mode"] = "projected_hardware_bypass"
    copied["comparison_scenario"] = scenario_label
    copied["timeline_label"] = f"{scenario_label}-HW"
    copied["is_projected_hardware_row"] = 1
    copied["source_measured_mode"] = display_mode(source_row.get("mode"))
    copied["per_gap_verdict"] = "PROJECTED HARDWARE - not measured"
    copied["final_path"] = "projected_hardware_bypass"
    copied["lifecycle_verdict"] = "PROJECTED HARDWARE - not measured"
    copied["lifecycle_explanation"] = (
        "Projected row only. It estimates a hardware KV movement path using measured H2D duration plus "
        "50 ms control overhead; it is not a measured SGLang request."
    )
    copied["direct_kv_h2d_start_ms"] = projected_start
    copied["direct_kv_h2d_end_ms"] = projected_end
    copied["direct_kv_h2d_duration_ms"] = projected_duration if projected_duration is not None else projected_end - projected_start
    copied["direct_kv_h2d_events"] = projection.get("measured_h2d_events", "")
    copied["prefetch_start_ms"] = projected_start
    copied["prefetch_end_ms"] = projected_end
    copied["prefetch_duration_ms"] = copied["direct_kv_h2d_duration_ms"]
    copied["prefetch_margin_ms"] = projected_margin if projected_margin is not None else ""
    copied["projected_hardware_margin_ms"] = projection.get("projected_hardware_margin_ms", "")
    copied["projected_hardware_duration_ms"] = projection.get("projected_hardware_duration_ms", "")
    copied["projected_hardware_start_ms"] = projection.get("projected_hardware_start_ms", "")
    copied["projected_hardware_end_ms"] = projection.get("projected_hardware_end_ms", "")
    copied["hardware_projection_label"] = projection.get("hardware_projection_label", "")
    copied["measured_h2d_source"] = projection.get("measured_h2d_source", "")
    copied["resume_start_ms"] = ""
    copied["resume_end_ms"] = ""
    copied["resume_ttft_ms"] = ""
    copied["replay_sglang_receive_start_ms"] = ""
    copied["replay_sglang_receive_end_ms"] = ""
    copied["replay_scheduler_queue_enter_start_ms"] = ""
    copied["replay_scheduler_queue_enter_end_ms"] = ""
    copied["replay_scheduler_admit_start_ms"] = ""
    copied["replay_scheduler_admit_end_ms"] = ""
    copied["replay_model_forward_start_ms"] = ""
    copied["replay_model_forward_end_ms"] = ""
    copied["replay_runtime_prefill_attribution_start_ms"] = ""
    copied["replay_runtime_prefill_attribution_end_ms"] = ""
    copied["replay_runtime_prefill_attributed_tokens"] = ""
    copied["replay_runtime_prefill_confidence"] = ""
    copied["replay_kv_h2d_start_ms"] = ""
    copied["replay_kv_h2d_end_ms"] = ""
    copied["replay_kv_h2d_events"] = ""
    copied["recomputed_tokens_est"] = ""
    copied["replay_new_prefill_tokens_est"] = ""
    return copied


def grouped_mode_comparison_rows(gaps: list[dict[str, Any]], max_scenarios: int) -> list[dict[str, Any]]:
    scenario_to_modes: defaultdict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in gaps:
        mode = canonical_mode(row.get("mode"))
        if mode not in {"no_prefetch", "direct_prefetch", "dynamo_priority_hints"}:
            continue
        scenario_to_modes[scenario_compare_key(row)][mode] = row

    complete_or_partial = [
        (key, rows_by_mode)
        for key, rows_by_mode in scenario_to_modes.items()
        if len(rows_by_mode) >= 2
    ]
    complete_or_partial.sort(key=lambda item: scenario_compare_sort_key(item[0]))

    mode_order = ["no_prefetch", "direct_prefetch", "dynamo_priority_hints"]
    output: list[dict[str, Any]] = []
    for scenario_idx, (_key, rows_by_mode) in enumerate(complete_or_partial[:max_scenarios]):
        scenario_label = f"C{scenario_idx:02d}"
        for mode in mode_order:
            row = rows_by_mode.get(mode)
            if not row:
                continue
            copied = dict(row)
            copied["comparison_scenario"] = scenario_label
            copied["timeline_label"] = f"{scenario_label}-{mode_short_label(mode)}"
            output.append(copied)
        projection_source = (
            rows_by_mode.get("dynamo_priority_hints")
            or rows_by_mode.get("direct_prefetch")
            or rows_by_mode.get("no_prefetch")
        )
        if projection_source:
            projected = projected_hardware_timeline_row(projection_source, scenario_label)
            if projected:
                output.append(projected)
    return output


def mode_comparison_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        margin = as_float(row.get("prefetch_margin_ms"))
        output.append(
            {
                "scenario": row.get("comparison_scenario", ""),
                "row": row.get("timeline_label", ""),
                "mode": display_mode(row.get("mode")),
                "task": row.get("task_index", ""),
                "gap": row.get("gap_order_in_task", ""),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "prefetch_margin_ms": round(margin, 3) if margin is not None else "",
                "hint_h2d_events": row.get("direct_kv_h2d_events", ""),
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                "verdict": display_verdict(row.get("per_gap_verdict", "")),
                "lifecycle": row.get("lifecycle_verdict", ""),
            }
        )
    return output


def dynamo_priority_hint_translation_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps:
        if canonical_mode(row.get("mode")) != "dynamo_priority_hints":
            continue
        rows.append(
            {
                "row": row.get("timeline_label", ""),
                "case_id": row.get("case_id", ""),
                "session_id": row.get("session_id", ""),
                "task": row.get("task_index", ""),
                "gap": row.get("gap_order_in_task", ""),
                "fillers": case_fillers(row),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "hint_source": row.get("hint_source", ""),
                "hint_dynamo_agent_priority": row.get("hint_dynamo_agent_priority", ""),
                "hint_sglang_priority": row.get("hint_sglang_priority", ""),
                "hint_priority_translation": row.get("hint_priority_translation", ""),
                "replay_dynamo_agent_priority": row.get("replay_dynamo_agent_priority", ""),
                "replay_sglang_priority": row.get("replay_sglang_priority", ""),
                "replay_priority_translation": row.get("replay_priority_translation", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "direct_kv_h2d_events": row.get("direct_kv_h2d_events", ""),
                "replay_kv_h2d_events": row.get("replay_kv_h2d_events", ""),
                "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                "verdict": display_verdict(row.get("per_gap_verdict", "")),
            }
        )
    return rows


def measured_kv_h2d_for_projection(row: dict[str, Any]) -> dict[str, Any]:
    direct_duration = as_float(row.get("direct_kv_h2d_duration_ms"))
    replay_duration = as_float(row.get("replay_kv_h2d_duration_ms"))
    if direct_duration is not None and has_events(row.get("direct_kv_h2d_events")):
        return {
            "source": "hint-side direct KV H2D",
            "duration_ms": direct_duration,
            "events": row.get("direct_kv_h2d_events", ""),
            "tokens": row.get("lifecycle_hint_h2d_tokens") or row.get("hint_host_load_tokens", ""),
            "actual_ready_end_ms": as_float(row.get("direct_kv_h2d_end_ms")),
        }
    if replay_duration is not None and has_events(row.get("replay_kv_h2d_events")):
        return {
            "source": "replay-side KV H2D",
            "duration_ms": replay_duration,
            "events": row.get("replay_kv_h2d_events", ""),
            "tokens": row.get("lifecycle_replay_h2d_tokens") or row.get("replay_host_load_tokens", ""),
            "actual_ready_end_ms": as_float(row.get("replay_kv_h2d_end_ms")),
        }
    return {"source": "", "duration_ms": None, "events": "", "tokens": "", "actual_ready_end_ms": None}


def actual_software_ready_margin_ms(row: dict[str, Any], measured: dict[str, Any]) -> float | None:
    due = as_float(row.get("tool_gap_end_ms"))
    if due is None:
        return None
    prefetch_margin = as_float(row.get("prefetch_margin_ms"))
    if canonical_mode(row.get("mode")) in PREFETCH_MODE_NAMES and prefetch_margin is not None:
        return prefetch_margin
    actual_ready_end = as_float(measured.get("actual_ready_end_ms"))
    if actual_ready_end is not None:
        return round(due - actual_ready_end, 3)
    first_token = first_token_ms(row)
    if first_token is not None:
        return round(due - first_token, 3)
    return None


def projected_hardware_bypass_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(gaps):
        if canonical_mode(row.get("mode")) == "projected_hardware_bypass":
            continue
        measured = measured_kv_h2d_for_projection(row)
        measured_duration = as_float(measured.get("duration_ms"))
        if measured_duration is None:
            continue
        due = as_float(row.get("tool_gap_end_ms"))
        if due is None:
            continue
        tool_start = as_float(row.get("tool_gap_start_ms"))
        tool_wait = as_float(row.get("tool_gap_ms"))
        if tool_start is None and tool_wait is not None:
            tool_start = due - tool_wait
        if tool_start is None:
            continue
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        software_margin = actual_software_ready_margin_ms(row, measured)
        software_lateness = max(0.0, -software_margin) if software_margin is not None else 0.0
        ttft = as_float(row.get("resume_ttft_ms"))
        for level, level_label, overhead_ms in HARDWARE_BYPASS_LEVELS:
            projected_duration = measured_duration + overhead_ms
            projected_end = tool_start + projected_duration
            projected_margin = due - projected_end
            projected_lateness = max(0.0, -projected_margin)
            estimated_saved = max(0.0, software_lateness - projected_lateness)
            if ttft is not None:
                estimated_saved = min(ttft, estimated_saved)
            rows.append(
                {
                    "row": label,
                    "mode": display_mode(row.get("mode")),
                    "scenario": row.get("comparison_scenario", ""),
                    "task": row.get("task_index", ""),
                    "gap": row.get("gap_order_in_task", ""),
                    "tool_wait_ms": row.get("tool_gap_ms", ""),
                    "measured_h2d_source": measured.get("source", ""),
                    "measured_h2d_events": measured.get("events", ""),
                    "measured_h2d_tokens_or_indices": measured.get("tokens", ""),
                    "measured_h2d_duration_ms": round(measured_duration, 3),
                    "software_ready_margin_ms": round(software_margin, 3) if software_margin is not None else "",
                    "hardware_projection": level,
                    "hardware_projection_label": level_label,
                    "hardware_overhead_ms": round(overhead_ms, 3),
                    "projected_hardware_duration_ms": round(projected_duration, 3),
                    "projected_hardware_start_ms": round(tool_start, 3),
                    "projected_hardware_end_ms": round(projected_end, 3),
                    "projected_hardware_margin_ms": round(projected_margin, 3),
                    "would_meet_deadline": 1 if projected_margin >= 0 else 0,
                    "estimated_ttft_saved_ms": round(estimated_saved, 3),
                    "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                    "simple_meaning": (
                        "Projected hardware would finish before replay."
                        if projected_margin >= 0
                        else "Projected hardware would still miss replay, but by less if software was later."
                    ),
                }
            )
    return rows


def projected_hardware_bypass_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_level: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[str(row.get("hardware_projection") or "")].append(row)
    output: list[dict[str, Any]] = []
    for level, level_label, _overhead in HARDWARE_BYPASS_LEVELS:
        items = by_level.get(level, [])
        margins = [float(row["projected_hardware_margin_ms"]) for row in items if row.get("projected_hardware_margin_ms") not in ("", None)]
        saved = [float(row["estimated_ttft_saved_ms"]) for row in items if row.get("estimated_ttft_saved_ms") not in ("", None)]
        h2d = [float(row["measured_h2d_duration_ms"]) for row in items if row.get("measured_h2d_duration_ms") not in ("", None)]
        hits = sum(1 for row in items if str(row.get("would_meet_deadline")) == "1")
        output.append(
            {
                "projection": level_label,
                "gaps_with_measured_h2d": len(items),
                "projected_deadline_hits": hits,
                "projected_hit_rate_pct": round(hits * 100.0 / len(items), 2) if items else "",
                "median_projected_margin_ms": round(median(margins), 3) if margins else "",
                "worst_projected_lateness_ms": round(abs(min([m for m in margins if m < 0])), 3) if any(m < 0 for m in margins) else "",
                "avg_measured_h2d_ms": avg(h2d),
                "avg_estimated_ttft_saved_ms": avg(saved),
                "total_estimated_ttft_saved_ms": round(sum(saved), 3) if saved else "",
            }
        )
    return output


def projected_hardware_bypass_cards_html(summary_rows: list[dict[str, Any]]) -> str:
    by_projection = {str(row.get("projection") or ""): row for row in summary_rows}
    cards: list[tuple[str, str]] = []
    for _level, label, _overhead in HARDWARE_BYPASS_LEVELS:
        row = by_projection.get(label, {})
        cards.append((f"{label} hit rate", f"{row.get('projected_hit_rate_pct', '')}%"))
    if summary_rows:
        cards.insert(0, ("gaps with measured KV H2D", str(summary_rows[0].get("gaps_with_measured_h2d", ""))))
    return "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{html.escape(label)}</div><div class=\"value\">{html.escape(value)}</div></div>"
        for label, value in cards
    ) + "</div>"


def build_projected_hardware_bypass_margin_plot(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No measured KV H2D rows were available for projected hardware-bypass analysis.</p>"
    rows_by_gap: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        gap = str(row.get("row") or "")
        if gap:
            rows_by_gap[gap][str(row.get("hardware_projection") or "")] = row
    gap_labels = sorted(rows_by_gap, key=lambda value: (len(value), value))
    software_values: list[float] = []
    projected_values: list[float] = []
    for gap in gap_labels:
        any_row = next(iter(rows_by_gap[gap].values()))
        software_margin = as_float(any_row.get("software_ready_margin_ms"))
        if software_margin is not None:
            software_values.append(software_margin)
        for level, _label, _overhead in HARDWARE_BYPASS_LEVELS:
            value = as_float(rows_by_gap[gap].get(level, {}).get("projected_hardware_margin_ms"))
            if value is not None:
                projected_values.append(value)
    margins = software_values + projected_values
    if not margins:
        return "<p>No projection margin values were available.</p>"

    width = 1480
    height = 580
    left = 98
    right = 48
    top = 74
    bottom = 118
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_margin = min(margins)
    max_margin = max(margins)
    pad = max(50.0, (max_margin - min_margin) * 0.08)
    y_min = min(min_margin - pad, -50.0)
    y_max = max(max_margin + pad, 50.0)
    scaled_min = h2d_symlog_value(y_min)
    scaled_max = h2d_symlog_value(y_max)

    def x_pos(index: int) -> float:
        if len(gap_labels) <= 1:
            return left + plot_w / 2
        return left + index * plot_w / (len(gap_labels) - 1)

    def y_pos(value: float) -> float:
        scaled = h2d_symlog_value(value)
        return top + (scaled_max - scaled) * plot_h / (scaled_max - scaled_min)

    zero_y = y_pos(0.0)
    colors = {
        "software": "#7c3aed",
        "best_case": "#16a34a",
        "realistic": "#0891b2",
        "conservative": "#f97316",
    }
    offsets = {
        "software": -30,
        "best_case": -10,
        "realistic": 10,
        "conservative": 30,
    }
    parts = [
        '<svg viewBox="0 0 1480 580" width="100%" role="img" aria-label="Projected hardware bypass deadline margin plot">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{left + plot_w - 8}" y="{zero_y - 8:.1f}" text-anchor="end" font-size="12" font-weight="700">0 ms replay due</text>',
        '<text x="22" y="292" transform="rotate(-90 22 292)" text-anchor="middle" font-size="13" font-weight="700">KV ready margin ms (symlog)</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 44}" text-anchor="middle" font-size="13" font-weight="700">gap / scenario order</text>',
        '<text x="106" y="38" font-size="13" fill="#166534" font-weight="700">above line = KV ready before replay</text>',
        '<text x="430" y="38" font-size="13" fill="#b91c1c" font-weight="700">below line = KV ready after replay</text>',
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

    x_tick_step = max(1, len(gap_labels) // 12)
    for index, gap in enumerate(gap_labels):
        x = x_pos(index)
        if index % x_tick_step == 0 or index == len(gap_labels) - 1:
            parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#94a3b8"/>')
            parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="10">{html.escape(gap)}</text>')

    for index, gap in enumerate(gap_labels):
        x = x_pos(index)
        by_level = rows_by_gap[gap]
        any_row = next(iter(by_level.values()))
        software_margin = as_float(any_row.get("software_ready_margin_ms"))
        if software_margin is not None:
            y = y_pos(software_margin)
            title = f"{gap} | observed software ready margin={software_margin:.3f} ms | mode={any_row.get('mode')}"
            parts.append(
                f'<circle cx="{x + offsets["software"]:.1f}" cy="{y:.1f}" r="6" fill="{colors["software"]}" opacity="0.88" stroke="#ffffff" stroke-width="1.5"><title>{html.escape(title)}</title></circle>'
            )
        for level, label, _overhead in HARDWARE_BYPASS_LEVELS:
            projection = by_level.get(level)
            if not projection:
                continue
            margin = as_float(projection.get("projected_hardware_margin_ms"))
            if margin is None:
                continue
            y = y_pos(margin)
            title = (
                f"{gap} | {label} | projected margin={margin:.3f} ms | "
                f"measured H2D={projection.get('measured_h2d_duration_ms')} ms | "
                f"overhead={projection.get('hardware_overhead_ms')} ms | "
                f"deadline_met={projection.get('would_meet_deadline')}"
            )
            parts.append(
                f'<rect x="{x + offsets[level] - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2" fill="{colors[level]}" opacity="0.88" stroke="#ffffff" stroke-width="1.5"><title>{html.escape(title)}</title></rect>'
            )

    legend = [
        ("observed software ready", colors["software"], "circle"),
        ("best-case hardware", colors["best_case"], "square"),
        ("realistic hardware", colors["realistic"], "square"),
        ("conservative hardware", colors["conservative"], "square"),
    ]
    lx = left
    ly = height - 84
    for label, color, kind in legend:
        if kind == "circle":
            parts.append(f'<circle cx="{lx}" cy="{ly}" r="6" fill="{color}"/>')
        else:
            parts.append(f'<rect x="{lx - 6}" y="{ly - 6}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{ly + 4}" font-size="12">{html.escape(label)}</text>')
        lx += 270
    parts.append("</svg>")
    return "\n".join(parts)


def projected_hardware_bypass_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return """
        <p>No measured KV H2D rows were available for projected hardware-bypass analysis.</p>
        """
    summary_rows = projected_hardware_bypass_summary_rows(rows)
    return f"""
    <p>This is a what-if estimate, not a real hardware run. It uses the measured KV H2D copy duration from the actual experiment, then asks what would happen if a low-overhead hardware/runtime path could start that same copy at tool-wait start and protect the KV until replay.</p>
    <p class="note">This does not assume memory copying is free. It keeps the measured copy time and adds three overhead settings: best case adds 0 ms, realistic adds 50 ms, and conservative adds 150 ms.</p>
    {projected_hardware_bypass_cards_html(summary_rows)}
    <h3>Projected Deadline Margin</h3>
    <p>Positive margin means the projected hardware path would finish before replay was due. Negative margin means even the projected hardware path would still be late.</p>
    <div class="setup-diagram">{build_projected_hardware_bypass_margin_plot(rows)}</div>
    <p class="note">The long per-gap projection table is in <strong>Evidence Tables / Raw Proof</strong>. Use it to audit the exact measured H2D duration, overhead assumption, projected finish time, and estimated TTFT saved for each row.</p>
    """


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
        {"column": "recompute_start_ms_est", "meaning": "Start of the replay prefill/recompute window. Exact when model-forward hooks are present; otherwise a fallback estimate."},
        {"column": "recompute_end_ms_est", "meaning": "End of the replay prefill/recompute window. Exact when model-forward hooks are present; otherwise a fallback estimate."},
        {"column": "recompute_duration_ms_est", "meaning": "Measured or estimated time spent in before-first-token replay work, including prefill and possible missing-KV rebuild."},
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
                "runtime_prefill_attribution": row.get("replay_runtime_prefill_confidence", ""),
                "runtime_prefill_tokens": row.get("replay_runtime_prefill_attributed_tokens", ""),
                "runtime_prefill_token_range": row.get("replay_runtime_prefill_token_range", ""),
                "runtime_prefill_batch": row.get("replay_runtime_prefill_batch_id", ""),
                "runtime_prefill_batch_requests": row.get("replay_runtime_prefill_batch_request_count", ""),
                "runtime_prefill_evidence": row.get("replay_runtime_prefill_evidence", ""),
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
        mode = canonical_mode(row.get("mode"))
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
                    "mode_label": display_mode(mode),
                    "verdict": verdict,
                    "verdict_label": display_verdict(verdict),
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
        "replay_runtime_prefill_confidence",
        "replay_runtime_prefill_attributed_tokens",
        "replay_runtime_prefill_token_range",
        "replay_runtime_prefill_batch_id",
        "replay_runtime_prefill_batch_request_count",
        "replay_runtime_prefill_evidence",
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
    mode = canonical_mode(row.get("mode"))
    truth = str(row.get("prefetch_truth_verdict") or "")
    if mode in PREFETCH_MODE_NAMES and truth:
        truth_styles = {
            "true_kv_prefetch_success": ("True KV prefetch success", "#166534"),
            "hint_completed_early_but_no_kv_load_seen": ("Hint early; no KV load seen", "#b45309"),
            "hint_completed_early_but_replay_recomputed": ("Hint early; replay recomputed", "#b45309"),
            "hint_loaded_kv_but_evicted_before_replay": ("Hint loaded; residency lost", "#b91c1c"),
            "hint_loaded_kv_but_replay_reloaded": ("Hint loaded; replay reloaded", "#b45309"),
            "hint_late": ("Hint late", "#b91c1c"),
            "no_reuse_evidence": ("No strong reuse evidence", "#64748b"),
        }
        if truth in truth_styles:
            return truth_styles[truth]
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
        mode = canonical_mode(row.get("mode"))
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
        truth_verdict = str(row.get("prefetch_truth_verdict") or "")
        truth_label = str(row.get("prefetch_truth_short_label") or "")
        truth_explanation = str(row.get("prefetch_truth_explanation") or "")
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
                "prefetch_truth": truth_label or display_verdict(truth_verdict),
                "prefetch_truth_explanation": truth_explanation,
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
    by_mode = {canonical_mode(row.get("mode")): row for row in mode_rows}
    no_prefetch = by_mode.get("no_prefetch", {})
    direct = by_mode.get("direct_prefetch", {})
    dynamo = by_mode.get("dynamo_priority_hints", {})
    deadline = by_mode.get("deadline_priority_prefetch", {})
    priority = by_mode.get("priority_direct_prefetch", {})
    cards = [
        ("controlled gaps", sum(int(row.get("controlled_gaps") or 0) for row in mode_rows)),
        ("no-prefetch avg TTFT", f"{no_prefetch.get('avg_resume_ttft_ms', '')} ms"),
        ("direct-prefetch avg TTFT", f"{direct.get('avg_resume_ttft_ms', '')} ms"),
        ("direct late prefetches", direct.get("late_prefetches", "")),
        ("direct H2D gaps", direct.get("hint_h2d_gaps", "")),
        ("suspected replay wait/recompute", sum(int(row.get("replay_recompute_or_wait_suspected_gaps") or 0) for row in mode_rows)),
        ("likely cache hit/resident", sum(int(row.get("likely_cache_hit_or_resident_gaps") or 0) for row in mode_rows)),
    ]
    if deadline:
        cards.extend(
            [
                ("deadline-prefetch avg TTFT", f"{deadline.get('avg_resume_ttft_ms', '')} ms"),
                ("deadline late prefetches", deadline.get("late_prefetches", "")),
                ("deadline H2D gaps", deadline.get("hint_h2d_gaps", "")),
            ]
        )
    if dynamo:
        cards.extend(
            [
                ("Dynamo-hint avg TTFT", f"{dynamo.get('avg_resume_ttft_ms', '')} ms"),
                ("Dynamo-hint late prefetches", dynamo.get("late_prefetches", "")),
            ]
        )
    if priority:
        cards.extend(
            [
                ("priority-prefetch avg TTFT", f"{priority.get('avg_resume_ttft_ms', '')} ms"),
                ("priority late prefetches", priority.get("late_prefetches", "")),
                ("priority H2D gaps", priority.get("hint_h2d_gaps", "")),
            ]
        )
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
        (
            "Direct prefetch attempt",
            "#a855f7",
            "The software baseline calls our direct SGLang KV load hook during the tool wait. This is separate from Dynamo priority hints.",
        ),
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
        if canonical_mode(row.get("mode")) != "no_prefetch":
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
        if canonical_mode(row.get("mode")) != "no_prefetch":
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


def mode_kv_ready_timing(row: dict[str, Any]) -> tuple[float | None, str, str, str]:
    due = as_float(row.get("tool_gap_end_ms"))
    if due is None:
        return None, "", "", "missing replay due"
    mode = canonical_mode(row.get("mode"))
    if mode == "no_prefetch":
        h2d_end = as_float(row.get("replay_kv_h2d_end_ms"))
        if h2d_end is not None and has_events(row.get("replay_kv_h2d_events")):
            return (
                h2d_end,
                "measured",
                "replay-side KV H2D",
                "No prefetch: readiness is when replay-side KV H2D finished.",
            )
        return None, "", "", "No replay-side KV H2D was observed."
    if mode == "dynamo_priority_hints":
        h2d_end = as_float(row.get("replay_kv_h2d_end_ms"))
        if h2d_end is not None and has_events(row.get("replay_kv_h2d_events")):
            return (
                h2d_end,
                "measured",
                "replay-side KV H2D",
                (
                    "Dynamo priority hints only: no direct KV prefetch hook is counted. "
                    "Readiness is when the prioritized replay path finished replay-side KV H2D."
                ),
            )
        return None, "", "", "Dynamo priority hints only: no replay-side KV H2D was observed."
    if mode in PREFETCH_MODE_NAMES:
        mode_label = display_mode(mode)
        direct_end = as_float(row.get("direct_kv_h2d_end_ms"))
        replay_start = as_float(row.get("resume_start_ms"))
        if (
            direct_end is not None
            and has_events(row.get("direct_kv_h2d_events"))
            and (replay_start is None or direct_end <= replay_start)
        ):
            return (
                direct_end,
                "measured",
                "hint-side direct KV H2D",
                f"{mode_label}: readiness is when hint-side direct KV H2D finished before replay started.",
            )
        replay_end = as_float(row.get("replay_kv_h2d_end_ms"))
        if replay_end is not None and has_events(row.get("replay_kv_h2d_events")):
            return (
                replay_end,
                "measured fallback",
                "replay-side KV H2D",
                f"{mode_label}: useful readiness falls back to replay-side KV H2D.",
            )
        if direct_end is not None and has_events(row.get("direct_kv_h2d_events")):
            return (
                direct_end,
                "measured late hint-side H2D",
                "hint-side direct KV H2D",
                f"{mode_label}: hint-side KV H2D finished after replay started, so this is counted as late readiness.",
            )
        prefetch_margin = as_float(row.get("prefetch_margin_ms"))
        if prefetch_margin is not None:
            return (
                due - prefetch_margin,
                "measured request path",
                "prefetch request completion",
                f"{mode_label} had no visible H2D, so this uses the measured prefetch request completion margin.",
            )
        return None, "", "", "No hint-side or replay-side KV H2D was observed."
    return None, "", "", f"Mode {mode} is not included in this readiness comparison."


def mode_readiness_margin(row: dict[str, Any]) -> tuple[float | None, str, str, str]:
    due = as_float(row.get("tool_gap_end_ms"))
    ready, evidence_kind, source, meaning = mode_kv_ready_timing(row)
    if due is None or ready is None:
        return None, evidence_kind, source, meaning
    return round(due - ready, 3), evidence_kind, source, meaning


def global_kv_readiness_by_mode_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_to_modes: defaultdict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in gaps:
        mode = canonical_mode(row.get("mode"))
        if mode in {"no_prefetch", "direct_prefetch", "dynamo_priority_hints"}:
            scenario_to_modes[scenario_compare_key(row)][mode] = row

    scenarios = [
        (key, rows_by_mode)
        for key, rows_by_mode in scenario_to_modes.items()
        if any(mode in rows_by_mode for mode in ("no_prefetch", "direct_prefetch", "dynamo_priority_hints"))
    ]
    scenarios.sort(key=lambda item: scenario_compare_sort_key(item[0]))

    output: list[dict[str, Any]] = []
    for scenario_idx, (key, rows_by_mode) in enumerate(scenarios):
        scenario_label = f"C{scenario_idx:02d}"
        for mode in ("no_prefetch", "direct_prefetch", "dynamo_priority_hints"):
            row = rows_by_mode.get(mode)
            if not row:
                continue
            margin, evidence_kind, source, meaning = mode_readiness_margin(row)
            due = as_float(row.get("tool_gap_end_ms"))
            replay_start = as_float(row.get("resume_start_ms"))
            resume_ttft = as_float(row.get("resume_ttft_ms"))
            first_token = replay_start + resume_ttft if replay_start is not None and resume_ttft is not None else None
            kv_ready, _, _, _ = mode_kv_ready_timing(row)
            replay_start_relative = (
                round(replay_start - due, 3) if due is not None and replay_start is not None else ""
            )
            first_token_relative = round(first_token - due, 3) if due is not None and first_token is not None else ""
            first_token_margin = round(due - first_token, 3) if due is not None and first_token is not None else ""
            if margin is None and replay_start_relative == "" and first_token_relative == "":
                continue
            kv_ready_relative = round(kv_ready - due, 3) if due is not None and kv_ready is not None else ""
            replay_start_to_kv_ready = (
                round(kv_ready - replay_start, 3) if kv_ready is not None and replay_start is not None else ""
            )
            first_token_to_kv_ready = (
                round(kv_ready - first_token, 3) if kv_ready is not None and first_token is not None else ""
            )
            output.append(
                {
                    "scenario": scenario_label,
                    "mode": display_mode(mode),
                    "mode_key": mode,
                    "task": row.get("task_index", key[0]),
                    "gap": row.get("gap_order_in_task", key[1]),
                    "tool_wait_ms": row.get("tool_gap_ms", key[2]),
                    "fillers": case_fillers(row) or key[3],
                    "kv_ready_margin_ms": margin if margin is not None else "",
                    "kv_ready_relative_ms": kv_ready_relative,
                    "replay_start_relative_ms": replay_start_relative,
                    "first_token_relative_ms": first_token_relative,
                    "first_token_margin_ms": first_token_margin,
                    "replay_start_to_kv_ready_ms": replay_start_to_kv_ready,
                    "first_token_to_kv_ready_ms": first_token_to_kv_ready,
                    "ready_before_replay_due": "" if margin is None else 1 if margin >= 0 else 0,
                    "replay_started_before_or_at_due": (
                        1 if replay_start_relative not in ("", None) and float(replay_start_relative) <= 0 else 0
                    ),
                    "first_token_before_or_at_due": (
                        1 if first_token_relative not in ("", None) and float(first_token_relative) <= 0 else 0
                    ),
                    "evidence_kind": evidence_kind,
                    "readiness_source": source,
                    "measured_or_projected": "measured",
                    "simple_meaning": meaning,
                }
            )

        projection_source = None
        realistic = None
        for candidate_mode in ("direct_prefetch", "no_prefetch", "dynamo_priority_hints"):
            candidate = rows_by_mode.get(candidate_mode)
            if not candidate:
                continue
            projections = projected_hardware_bypass_rows([candidate])
            realistic = next((row for row in projections if row.get("hardware_projection") == "realistic"), None)
            if realistic:
                projection_source = candidate
                break
        if projection_source and realistic:
            margin = as_float(realistic.get("projected_hardware_margin_ms"))
            if margin is not None:
                due = as_float(projection_source.get("tool_gap_end_ms"))
                replay_start = as_float(projection_source.get("resume_start_ms"))
                resume_ttft = as_float(projection_source.get("resume_ttft_ms"))
                first_token = replay_start + resume_ttft if replay_start is not None and resume_ttft is not None else None
                projected_ready = as_float(realistic.get("projected_hardware_end_ms"))
                replay_start_relative = (
                    round(replay_start - due, 3) if due is not None and replay_start is not None else ""
                )
                first_token_relative = round(first_token - due, 3) if due is not None and first_token is not None else ""
                first_token_margin = round(due - first_token, 3) if due is not None and first_token is not None else ""
                kv_ready_relative = (
                    round(projected_ready - due, 3)
                    if due is not None and projected_ready is not None
                    else round(-margin, 3)
                )
                replay_start_to_kv_ready = (
                    round(projected_ready - replay_start, 3)
                    if projected_ready is not None and replay_start is not None
                    else ""
                )
                first_token_to_kv_ready = (
                    round(projected_ready - first_token, 3)
                    if projected_ready is not None and first_token is not None
                    else ""
                )
                output.append(
                    {
                        "scenario": scenario_label,
                        "mode": "Projected hardware bypass",
                        "mode_key": "projected_hardware_bypass",
                        "task": realistic.get("task", key[0]),
                        "gap": realistic.get("gap", key[1]),
                        "tool_wait_ms": realistic.get("tool_wait_ms", key[2]),
                        "fillers": case_fillers(projection_source) or key[3],
                        "kv_ready_margin_ms": round(margin, 3),
                        "kv_ready_relative_ms": kv_ready_relative,
                        "replay_start_relative_ms": replay_start_relative,
                        "first_token_relative_ms": first_token_relative,
                        "first_token_margin_ms": first_token_margin,
                        "replay_start_to_kv_ready_ms": replay_start_to_kv_ready,
                        "first_token_to_kv_ready_ms": first_token_to_kv_ready,
                        "ready_before_replay_due": 1 if margin >= 0 else 0,
                        "replay_started_before_or_at_due": (
                            1
                            if replay_start_relative not in ("", None)
                            and float(replay_start_relative) <= 0
                            else 0
                        ),
                        "first_token_before_or_at_due": (
                            1
                            if first_token_relative not in ("", None)
                            and float(first_token_relative) <= 0
                            else 0
                        ),
                        "evidence_kind": "projected",
                        "readiness_source": "projected hardware H2D",
                        "measured_or_projected": "projected, not measured",
                        "simple_meaning": (
                            "Projected hardware bypass: assumes the urgent KV copy starts at the tool-wait boundary "
                            "and pays measured H2D time plus 50 ms overhead. The measured H2D duration is taken "
                            "from the best available measured H2D source for the same scenario."
                        ),
                    }
                )
    return output


def global_kv_readiness_by_mode_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode_order = ["no_prefetch", "direct_prefetch", "dynamo_priority_hints", "projected_hardware_bypass"]
    output: list[dict[str, Any]] = []
    for mode in mode_order:
        items = [row for row in rows if canonical_mode(row.get("mode_key")) == mode]
        if not items:
            continue
        margins = [float(row["kv_ready_margin_ms"]) for row in items if row.get("kv_ready_margin_ms") not in ("", None)]
        ready = sum(1 for row in items if str(row.get("ready_before_replay_due")) == "1")
        late_margins = [value for value in margins if value < 0]
        output.append(
            {
                "mode": display_mode(mode),
                "dots": len(items),
                "ready_before_replay_due": ready,
                "late": len(items) - ready,
                "ready_pct": round(ready * 100.0 / len(items), 2) if items else "",
                "median_margin_ms": round(median(margins), 3) if margins else "",
                "worst_lateness_ms": round(abs(min(late_margins)), 3) if late_margins else "",
                "evidence": (
                    "projected, not measured"
                    if mode == "projected_hardware_bypass"
                    else "measured SGLang KV movement/request timing"
                ),
            }
        )
    return output


def build_global_kv_readiness_by_mode_dot_plot(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No mode-comparison KV readiness rows were available for this run.</p>"
    width = 1480
    height = 560
    left = 96
    right = 48
    top = 78
    bottom = 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    margins = [float(row["kv_ready_margin_ms"]) for row in rows if row.get("kv_ready_margin_ms") not in ("", None)]
    if not margins:
        return "<p>No KV readiness margins were available for this run.</p>"
    min_margin = min(margins)
    max_margin = max(margins)
    pad = max(50.0, (max_margin - min_margin) * 0.08)
    y_min = min(min_margin - pad, -50.0)
    y_max = max(max_margin + pad, 50.0)
    scaled_min = h2d_symlog_value(y_min)
    scaled_max = h2d_symlog_value(y_max)
    scenario_labels = sorted({str(row.get("scenario") or "") for row in rows}, key=lambda value: (len(value), value))
    scenario_index = {label: idx for idx, label in enumerate(scenario_labels)}
    mode_offsets = {
        "no_prefetch": -27.0,
        "direct_prefetch": -9.0,
        "dynamo_priority_hints": 9.0,
        "projected_hardware_bypass": 27.0,
    }
    mode_styles = {
        "no_prefetch": ("#2563eb", "circle", "NP"),
        "direct_prefetch": ("#7c3aed", "square", "DP"),
        "dynamo_priority_hints": ("#f59e0b", "diamond", "DH"),
        "projected_hardware_bypass": ("#0f766e", "hollow", "HW"),
    }

    def x_pos(label: str, mode_key: str) -> float:
        if len(scenario_labels) <= 1:
            base = left + plot_w / 2
        else:
            base = left + scenario_index[label] * plot_w / (len(scenario_labels) - 1)
        return base + mode_offsets.get(mode_key, 0.0)

    def y_pos(value: float) -> float:
        scaled = h2d_symlog_value(value)
        return top + (scaled_max - scaled) * plot_h / (scaled_max - scaled_min)

    def dot_svg(x: float, y: float, color: str, kind: str, title: str) -> str:
        escaped_title = html.escape(title)
        if kind == "square":
            return f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2" fill="{color}" opacity="0.9"><title>{escaped_title}</title></rect>'
        if kind == "diamond":
            points = f"{x:.1f},{y - 8:.1f} {x + 8:.1f},{y:.1f} {x:.1f},{y + 8:.1f} {x - 8:.1f},{y:.1f}"
            return f'<polygon points="{points}" fill="{color}" opacity="0.95"><title>{escaped_title}</title></polygon>'
        if kind == "hollow":
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#ffffff" stroke="{color}" stroke-width="3" stroke-dasharray="3 2"><title>{escaped_title}</title></circle>'
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{color}" opacity="0.9"><title>{escaped_title}</title></circle>'

    zero_y = y_pos(0.0)
    parts = [
        '<svg viewBox="0 0 1480 560" width="100%" role="img" aria-label="Global KV readiness by mode dot plot">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{left + plot_w - 8}" y="{zero_y - 8:.1f}" text-anchor="end" font-size="12" font-weight="700">0 ms replay due</text>',
        '<text x="22" y="280" transform="rotate(-90 22 280)" text-anchor="middle" font-size="13" font-weight="700">KV ready margin ms (symlog)</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 40}" text-anchor="middle" font-size="13" font-weight="700">controlled scenario order</text>',
        '<text x="104" y="36" font-size="13" fill="#166534" font-weight="700">above line = KV ready before replay due</text>',
        '<text x="470" y="36" font-size="13" fill="#b91c1c" font-weight="700">below line = KV became ready after replay was due</text>',
        '<text x="104" y="56" font-size="12" fill="#475569">HW dots are projected from measured H2D duration plus 50 ms overhead; NP, DP, and DH dots are measured.</text>',
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

    x_tick_step = max(1, len(scenario_labels) // 12)
    for index, label in enumerate(scenario_labels):
        if index % x_tick_step != 0 and index != len(scenario_labels) - 1:
            continue
        x = x_pos(label, "")
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="10">{html.escape(label)}</text>')

    for row in rows:
        margin = as_float(row.get("kv_ready_margin_ms"))
        scenario = str(row.get("scenario") or "")
        mode_key = canonical_mode(row.get("mode_key"))
        if margin is None or scenario not in scenario_index:
            continue
        color, kind, short = mode_styles.get(mode_key, ("#64748b", "circle", "?"))
        x = x_pos(scenario, mode_key)
        y = y_pos(margin)
        title = (
            f"{scenario} {display_mode(mode_key)} | margin={margin:.3f} ms | "
            f"source={row.get('readiness_source')} | fillers={row.get('fillers')} | "
            f"tool_wait={row.get('tool_wait_ms')} ms | {row.get('measured_or_projected')}"
        )
        parts.append(dot_svg(x, y, color, kind, title))
        parts.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="8" fill="{color}" font-weight="700">{short}</text>')

    lx = left
    ly = height - 82
    legend_items = [
        ("No prefetch", "no_prefetch"),
        ("Direct prefetch", "direct_prefetch"),
        ("Dynamo priority hints only", "dynamo_priority_hints"),
        ("Projected HW bypass", "projected_hardware_bypass"),
    ]
    for label, mode_key in legend_items:
        color, kind, short = mode_styles[mode_key]
        parts.append(dot_svg(lx, ly, color, kind, label))
        parts.append(f'<text x="{lx + 14}" y="{ly + 4}" font-size="12">{html.escape(short)} = {html.escape(label)}</text>')
        lx += 260
    parts.append("</svg>")
    return "\n".join(parts)


def global_replay_start_by_mode_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode_order = ["no_prefetch", "direct_prefetch", "dynamo_priority_hints", "projected_hardware_bypass"]
    output: list[dict[str, Any]] = []
    for mode in mode_order:
        items = [row for row in rows if canonical_mode(row.get("mode_key")) == mode]
        values = [
            float(row["replay_start_relative_ms"])
            for row in items
            if row.get("replay_start_relative_ms") not in ("", None)
        ]
        if not values:
            continue
        late = [value for value in values if value > 0]
        on_time = [value for value in values if value <= 0]
        output.append(
            {
                "mode": display_mode(mode),
                "dots": len(values),
                "replay_started_on_or_before_due": len(on_time),
                "replay_started_late": len(late),
                "late_pct": round(len(late) * 100.0 / len(values), 2) if values else "",
                "median_replay_start_relative_ms": round(median(values), 3),
                "worst_replay_start_lateness_ms": round(max(late), 3) if late else "",
            }
        )
    return output


def global_first_token_by_mode_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode_order = ["no_prefetch", "direct_prefetch", "dynamo_priority_hints", "projected_hardware_bypass"]
    output: list[dict[str, Any]] = []
    for mode in mode_order:
        items = [row for row in rows if canonical_mode(row.get("mode_key")) == mode]
        values = [
            float(row["first_token_relative_ms"])
            for row in items
            if row.get("first_token_relative_ms") not in ("", None)
        ]
        if not values:
            continue
        late = [value for value in values if value > 0]
        on_time = [value for value in values if value <= 0]
        output.append(
            {
                "mode": display_mode(mode),
                "circles": len(values),
                "first_token_on_or_before_due": len(on_time),
                "first_token_late": len(late),
                "late_pct": round(len(late) * 100.0 / len(values), 2) if values else "",
                "median_first_token_relative_ms": round(median(values), 3),
                "worst_first_token_lateness_ms": round(max(late), 3) if late else "",
            }
        )
    return output


def global_replay_vs_kv_status_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode_order = ["no_prefetch", "direct_prefetch", "dynamo_priority_hints", "projected_hardware_bypass"]
    output: list[dict[str, Any]] = []
    for mode in mode_order:
        items = [row for row in rows if canonical_mode(row.get("mode_key")) == mode]
        if not items:
            continue
        replay_dots = sum(1 for row in items if row.get("first_token_relative_ms") not in ("", None))
        kv_squares = sum(1 for row in items if row.get("kv_ready_margin_ms") not in ("", None))
        if mode == "dynamo_priority_hints":
            meaning = "first-token timing is measured; KV square appears only when replay-side H2D was measured"
        elif mode == "projected_hardware_bypass":
            meaning = "projected KV-ready square, not measured hardware"
        elif mode == "direct_prefetch":
            meaning = "measured direct-prefetch or replay-side KV readiness when observed"
        else:
            meaning = "measured replay start and replay-side KV readiness when observed"
        output.append(
            {
                "mode": display_mode(mode),
                "first_token_circles": replay_dots,
                "kv_ready_squares": kv_squares,
                "missing_kv_ready_squares": max(0, replay_dots - kv_squares),
                "meaning": meaning,
            }
        )
    return output


def build_global_replay_vs_kv_readiness_plot(rows: list[dict[str, Any]]) -> str:
    usable = [
        row
        for row in rows
        if row.get("first_token_relative_ms") not in ("", None)
        or row.get("kv_ready_relative_ms") not in ("", None)
    ]
    if not usable:
        return "<p>No first-token or KV-ready timing rows were available for this run.</p>"
    width = 1500
    height = 720
    left = 118
    right = 68
    top = 92
    bottom = 210
    plot_w = width - left - right
    plot_h = height - top - bottom
    values: list[float] = []
    for row in usable:
        first_token_relative = as_float(row.get("first_token_relative_ms"))
        kv_ready_margin = as_float(row.get("kv_ready_margin_ms"))
        kv_ready_relative = as_float(row.get("kv_ready_relative_ms"))
        if first_token_relative is not None:
            values.append(first_token_relative)
        if kv_ready_relative is not None:
            values.append(kv_ready_relative)
        elif kv_ready_margin is not None:
            values.append(-kv_ready_margin)
    if not values:
        return "<p>No first-token or KV-ready timing values were available for this run.</p>"
    v_min = min(values)
    v_max = max(values)
    pad = max(50.0, (v_max - v_min) * 0.08)
    y_min = min(v_min - pad, -50.0)
    y_max = max(v_max + pad, 50.0)
    scaled_min = h2d_symlog_value(y_min)
    scaled_max = h2d_symlog_value(y_max)
    scenario_labels = sorted({str(row.get("scenario") or "") for row in usable}, key=lambda value: (len(value), value))
    scenario_index = {label: idx for idx, label in enumerate(scenario_labels)}
    mode_offsets = {
        "no_prefetch": -30.0,
        "direct_prefetch": -10.0,
        "dynamo_priority_hints": 10.0,
        "projected_hardware_bypass": 30.0,
    }
    mode_colors = {
        "no_prefetch": "#2563eb",
        "direct_prefetch": "#7c3aed",
        "dynamo_priority_hints": "#f59e0b",
        "projected_hardware_bypass": "#0f766e",
    }
    max_offset = max(abs(value) for value in mode_offsets.values())
    base_left = left + max_offset + 18
    base_right = left + plot_w - max_offset - 18

    def x_pos(label: str, mode_key: str) -> float:
        if len(scenario_labels) <= 1:
            base = left + plot_w / 2
        else:
            base = base_left + scenario_index[label] * (base_right - base_left) / (len(scenario_labels) - 1)
        return base + mode_offsets.get(mode_key, 0.0)

    def scenario_x(label: str) -> float:
        if len(scenario_labels) <= 1:
            return left + plot_w / 2
        return base_left + scenario_index[label] * (base_right - base_left) / (len(scenario_labels) - 1)

    def y_pos(value: float) -> float:
        scaled = h2d_symlog_value(value)
        return top + (scaled_max - scaled) * plot_h / (scaled_max - scaled_min)

    zero_y = y_pos(0.0)
    parts = [
        '<svg viewBox="0 0 1500 720" width="100%" role="img" aria-label="Replay first token versus KV readiness by mode">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="10" fill="#ffffff" stroke="#e2e8f0"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{left + plot_w - 8}" y="{zero_y - 8:.1f}" text-anchor="end" font-size="12" font-weight="700">0 ms deadline</text>',
        '<text x="26" y="300" transform="rotate(-90 26 300)" text-anchor="middle" font-size="13" font-weight="700">lateness vs deadline ms (symlog)</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 46}" text-anchor="middle" font-size="13" font-weight="700">controlled scenario order</text>',
        '<text x="104" y="34" font-size="13" fill="#111827" font-weight="700">above 0 ms = late; below 0 ms = early</text>',
        '<text x="104" y="56" font-size="12" fill="#475569">Circle = first replay token. Square = KV ready. Higher means later.</text>',
    ]
    seen_ticks: set[int] = set()
    seen_tick_y: list[float] = []
    for value in h2d_symlog_tick_values(y_min, y_max):
        rounded = int(round(value))
        if rounded in seen_ticks:
            continue
        y = y_pos(value)
        if any(abs(y - prior_y) < 18 for prior_y in seen_tick_y):
            continue
        seen_ticks.add(rounded)
        seen_tick_y.append(y)
        parts.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{rounded} ms</text>')

    x_tick_step = max(1, len(scenario_labels) // 12)
    for index, label in enumerate(scenario_labels):
        if index % x_tick_step != 0 and index != len(scenario_labels) - 1:
            continue
        x = scenario_x(label)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="10">{html.escape(label)}</text>')

    for row in usable:
        scenario = str(row.get("scenario") or "")
        mode_key = canonical_mode(row.get("mode_key"))
        if scenario not in scenario_index:
            continue
        color = mode_colors.get(mode_key, "#64748b")
        x = x_pos(scenario, mode_key)
        first_token_rel = as_float(row.get("first_token_relative_ms"))
        kv_margin = as_float(row.get("kv_ready_margin_ms"))
        kv_rel = as_float(row.get("kv_ready_relative_ms"))
        kv_lateness = kv_rel if kv_rel is not None else -kv_margin if kv_margin is not None else None
        title_base = (
            f"{scenario} {display_mode(mode_key)} | fillers={row.get('fillers')} | "
            f"tool_wait={row.get('tool_wait_ms')} ms"
        )
        if first_token_rel is not None:
            y = y_pos(first_token_rel)
            title = (
                f"{title_base} | first token lateness={first_token_rel:.3f} ms "
                f"(positive=late, negative=early)"
            )
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.2" fill="{color}" opacity="0.92" stroke="#ffffff" stroke-width="1.5"><title>{html.escape(title)}</title></circle>')
        if kv_lateness is not None:
            y = y_pos(kv_lateness)
            title = (
                f"{title_base} | KV ready lateness={kv_lateness:.3f} ms "
                f"(positive=late, negative=early) | "
                f"source={row.get('readiness_source')} | {row.get('measured_or_projected')}"
            )
            parts.append(
                f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2" '
                f'fill="{color}" opacity="0.55" stroke="{color}" stroke-width="2"><title>{html.escape(title)}</title></rect>'
            )

    legend_y = top + plot_h + 62
    legend_box_x = left
    legend_box_y = top + plot_h + 38
    legend_box_w = plot_w
    legend_box_h = 104
    parts.append(
        f'<rect x="{legend_box_x}" y="{legend_box_y}" width="{legend_box_w}" height="{legend_box_h}" '
        'rx="10" fill="#f8fafc" stroke="#e2e8f0"/>'
    )
    lx = left + 24
    parts.append(f'<text x="{lx}" y="{legend_y}" font-size="12" fill="#475569" font-weight="700">Marker shape</text>')
    parts.append(f'<circle cx="{lx + 118:.1f}" cy="{legend_y - 4:.1f}" r="6.2" fill="#0f172a" opacity="0.85"/>')
    parts.append(f'<text x="{lx + 136}" y="{legend_y}" font-size="12" fill="#334155">first replay token</text>')
    parts.append(f'<rect x="{lx + 300:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" rx="2" fill="#0f172a" opacity="0.45" stroke="#0f172a" stroke-width="2"/>')
    parts.append(f'<text x="{lx + 320}" y="{legend_y}" font-size="12" fill="#334155">KV ready</text>')
    legend_modes = [
        ("NP", "No prefetch", "no_prefetch"),
        ("DP", "Direct prefetch", "direct_prefetch"),
        ("DH", "Dynamo priority hints only", "dynamo_priority_hints"),
        ("HW", "Projected hardware bypass", "projected_hardware_bypass"),
    ]
    lx = left + 24
    ly = legend_y + 42
    parts.append(f'<text x="{lx}" y="{ly + 4}" font-size="12" fill="#475569" font-weight="700">Mode color</text>')
    lx += 118
    for short, label, mode_key in legend_modes:
        color = mode_colors[mode_key]
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="6.5" fill="{color}"/>')
        parts.append(f'<rect x="{lx + 20:.1f}" y="{ly - 6:.1f}" width="12" height="12" rx="2" fill="{color}" opacity="0.55" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx + 42}" y="{ly + 4}" font-size="12">{html.escape(short)} = {html.escape(label)}</text>')
        lx += 290
    parts.append("</svg>")
    return "\n".join(parts)


def global_kv_readiness_by_mode_html(gaps: list[dict[str, Any]]) -> str:
    rows = global_kv_readiness_by_mode_rows(gaps)
    if not rows:
        return "<p>No mode-comparison KV readiness rows were available for this run.</p>"
    return f"""
    <h3>Replay First Token vs KV Ready</h3>
    <p class="note">Circle = first replay token. Square = KV ready. Above zero is late; below zero is early.</p>
    <div class="setup-diagram">{build_global_replay_vs_kv_readiness_plot(rows)}</div>
    """


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
                "no replay H2D; prefill/recompute path",
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
                "request_id": row.get("request_id", ""),
                "agent_request_id": row.get("agent_request_id", ""),
                "correlation_id": row.get("correlation_id", ""),
                "block_key": block_key,
                "host_index_start": row.get("host_index_start", ""),
                "host_index_end": row.get("host_index_end", ""),
                "host_index_count": row.get("host_index_count", ""),
                "host_index_signature": row.get("host_index_signature", ""),
                "device_index_start": row.get("device_index_start", ""),
                "device_index_end": row.get("device_index_end", ""),
                "device_index_count": row.get("device_index_count", ""),
                "device_index_signature": row.get("device_index_signature", ""),
                "token_or_index_count": round(token_count, 3),
                "aligned_start_ms": aligned_start,
                "aligned_end_ms": aligned_end,
                "duration_ms": round(duration, 3),
                "raw_copy_start_ms": row.get("copy_start_ms", ""),
                "raw_copy_end_ms": row.get("copy_end_ms", ""),
                "case_clock_offset_ms": round(offset, 3),
                "alignment_confidence": "case_h2d_anchor",
                "evidence_confidence": row.get("confidence", ""),
                "evidence_level": row.get("evidence_level", ""),
                "exact_correlation_source": row.get("exact_correlation_source", ""),
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


def _trace_row_case_id(row: dict[str, Any]) -> str:
    context = row.get("kv_context")
    if isinstance(context, dict):
        nested = context.get("request")
        if isinstance(nested, dict):
            return str(
                row.get("ledger_case_id")
                or context.get("ledger_case_id")
                or context.get("agent_case_id")
                or nested.get("agent_case_id")
                or ""
            )
        return str(row.get("ledger_case_id") or context.get("ledger_case_id") or context.get("agent_case_id") or "")
    return str(row.get("ledger_case_id") or row.get("agent_case_id") or "")


def _trace_row_agent_context(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("kv_context")
    if isinstance(context, dict):
        request = context.get("request")
        if isinstance(request, dict):
            merged = dict(request)
            merged.update(context)
            return merged
        return context
    return row


def _pool_state_from_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("kv_pool_usage_pct") not in (None, "", [], {}):
        return {
            "source": row.get("kv_pool_source", ""),
            "object_type": row.get("kv_pool_object_type", ""),
            "total_slots": row.get("kv_pool_total_slots", ""),
            "free_slots": row.get("kv_pool_free_slots", ""),
            "used_slots": row.get("kv_pool_used_slots", ""),
            "usage_pct": row.get("kv_pool_usage_pct", ""),
            "page_size": row.get("kv_pool_page_size", ""),
        }
    context = row.get("kv_context")
    if isinstance(context, dict):
        state = context.get("kv_pool_state")
        if isinstance(state, dict):
            return state
    return {}


def kv_pool_samples_from_trace(gaps: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize direct SGLang GPU KV-pool samples into the case clock."""

    base_ns_by_case: dict[str, float] = {}
    for row in trace_rows:
        case_id = _trace_row_case_id(row)
        ts = as_float(row.get("ts_ns"))
        if not case_id or ts is None:
            continue
        base_ns_by_case[case_id] = min(base_ns_by_case.get(case_id, ts), ts)

    output: list[dict[str, Any]] = []
    for row in trace_rows:
        state = _pool_state_from_trace_row(row)
        usage = as_float(state.get("usage_pct"))
        total = as_float(state.get("total_slots"))
        if usage is None or total is None:
            continue
        case_id = _trace_row_case_id(row)
        base_ns = base_ns_by_case.get(case_id)
        ts = as_float(row.get("ts_ns"))
        if not case_id or base_ns is None or ts is None:
            continue
        context = _trace_row_agent_context(row)
        output.append(
            {
                "case_id": case_id,
                "aligned_ms": round((ts - base_ns) / 1_000_000.0, 3),
                "event": row.get("event", ""),
                "source_event": row.get("source_event", ""),
                "method": row.get("method", ""),
                "stage": row.get("stage", row.get("category", "")),
                "phase": row.get("phase", ""),
                "agent_session_id": context.get("agent_session_id", ""),
                "agent_phase": context.get("agent_phase", ""),
                "agent_gap_id": context.get("agent_gap_id", ""),
                "request_id": row.get("request_id", context.get("request_id", "")),
                "kv_pool_usage_pct": round(usage, 3),
                "kv_pool_total_slots": round(total, 3),
                "kv_pool_used_slots": state.get("used_slots", ""),
                "kv_pool_free_slots": state.get("free_slots", ""),
                "kv_pool_page_size": state.get("page_size", ""),
                "kv_pool_source": state.get("source", ""),
                "kv_pool_object_type": state.get("object_type", ""),
                "pool_pressure": kv_pool_pressure_label(usage),
                "evidence": "direct_sglang_pool_sample",
            }
        )
    output.sort(key=lambda item: (str(item.get("case_id") or ""), as_float(item.get("aligned_ms")) or 0.0))
    return output


def kv_pool_pressure_label(usage_pct: float | None) -> str:
    if usage_pct is None:
        return "unknown"
    if usage_pct >= 95.0:
        return "very high"
    if usage_pct >= 85.0:
        return "high"
    if usage_pct >= 65.0:
        return "medium"
    return "low"


def kv_pool_heat_color(usage_pct: float | None) -> str:
    if usage_pct is None:
        return "#cbd5e1"
    if usage_pct >= 99.0:
        return "#991b1b"
    if usage_pct >= 95.0:
        return "#ef4444"
    if usage_pct >= 85.0:
        return "#f97316"
    if usage_pct >= 65.0:
        return "#eab308"
    return "#22c55e"


def nearest_kv_pool_sample(
    samples: list[dict[str, Any]],
    target_ms: Any,
    max_distance_ms: float = 5000.0,
) -> dict[str, Any] | None:
    target = as_float(target_ms)
    if target is None or not samples:
        return None
    best: dict[str, Any] | None = None
    best_distance = float("inf")
    for sample in samples:
        aligned = as_float(sample.get("aligned_ms"))
        if aligned is None:
            continue
        distance = abs(aligned - target)
        if distance < best_distance:
            best = sample
            best_distance = distance
    if best is None or best_distance > max_distance_ms:
        return None
    out = dict(best)
    out["distance_from_target_ms"] = round(best_distance, 3)
    return out


def kv_pool_histogram_bins(
    samples: list[dict[str, Any]],
    due_ms: Any,
    start_ms: Any,
    end_ms: Any,
    bin_count: int = 36,
) -> list[dict[str, Any]]:
    due = as_float(due_ms)
    start = as_float(start_ms)
    end = as_float(end_ms)
    if due is None or start is None or end is None or end <= start:
        return []
    bin_count = max(4, min(80, bin_count))
    span = end - start
    buckets: list[list[float]] = [[] for _ in range(bin_count)]
    for sample in samples:
        aligned = as_float(sample.get("aligned_ms"))
        usage = as_float(sample.get("kv_pool_usage_pct"))
        if aligned is None or usage is None or aligned < start or aligned > end:
            continue
        idx = min(bin_count - 1, max(0, int((aligned - start) * bin_count / max(1e-9, span))))
        buckets[idx].append(usage)
    bins: list[dict[str, Any]] = []
    for idx, values in enumerate(buckets):
        b_start = start + span * idx / bin_count
        b_end = start + span * (idx + 1) / bin_count
        if values:
            avg_usage = mean(values)
            max_usage = max(values)
            bins.append(
                {
                    "start_rel_ms": round(b_start - due, 3),
                    "end_rel_ms": round(b_end - due, 3),
                    "avg_usage_pct": round(avg_usage, 3),
                    "max_usage_pct": round(max_usage, 3),
                    "samples": len(values),
                    "pressure": kv_pool_pressure_label(max_usage),
                }
            )
        else:
            bins.append(
                {
                    "start_rel_ms": round(b_start - due, 3),
                    "end_rel_ms": round(b_end - due, 3),
                    "avg_usage_pct": "",
                    "max_usage_pct": "",
                    "samples": 0,
                    "pressure": "missing",
                }
            )
    return bins


def kv_pool_window_stats(samples: list[dict[str, Any]], start_ms: Any, end_ms: Any) -> dict[str, Any]:
    start = as_float(start_ms)
    end = as_float(end_ms)
    if start is None or end is None or end < start:
        return {"samples": 0}
    values = [
        as_float(sample.get("kv_pool_usage_pct"))
        for sample in samples
        if (as_float(sample.get("aligned_ms")) is not None and start <= (as_float(sample.get("aligned_ms")) or 0.0) <= end)
    ]
    values = [value for value in values if value is not None]
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "min_usage_pct": round(min(values), 3),
        "max_usage_pct": round(max(values), 3),
        "avg_usage_pct": round(mean(values), 3),
    }


def kv_pool_residency_by_gap_rows(
    gaps: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples_by_case: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_case[str(sample.get("case_id") or "")].append(sample)

    rows: list[dict[str, Any]] = []
    for gap in gaps:
        case_samples = samples_by_case.get(str(gap.get("case_id") or ""), [])
        due = as_float(gap.get("tool_gap_end_ms"))
        replay_h2d_start = as_float(gap.get("replay_kv_h2d_start_ms"))
        direct_h2d_start = as_float(gap.get("direct_kv_h2d_start_ms"))
        wait_stats = kv_pool_window_stats(case_samples, gap.get("tool_gap_start_ms"), gap.get("tool_gap_end_ms"))
        before_replay_h2d_stats = kv_pool_window_stats(
            case_samples,
            gap.get("tool_gap_end_ms"),
            replay_h2d_start if replay_h2d_start is not None else gap.get("resume_start_ms"),
        )
        dispatch_stats = kv_pool_window_stats(case_samples, gap.get("resume_submitted_ms"), gap.get("resume_start_ms"))
        pool_window_start_candidates = [
            as_float(gap.get("tool_gap_start_ms")),
            as_float(gap.get("prefetch_start_ms")),
            due,
        ]
        pool_window_end_candidates = [
            as_float(gap.get("resume_end_ms")),
            first_token_ms(gap),
            as_float(gap.get("replay_kv_h2d_end_ms")),
            as_float(gap.get("direct_kv_h2d_end_ms")),
            as_float(gap.get("resume_start_ms")),
        ]
        pool_window_start_values = [value for value in pool_window_start_candidates if value is not None]
        pool_window_end_values = [value for value in pool_window_end_candidates if value is not None]
        if due is not None:
            pool_window_start = min(pool_window_start_values) if pool_window_start_values else due - 500.0
            pool_window_end = max(pool_window_end_values) if pool_window_end_values else due + 1000.0
            pool_pad = max(50.0, (pool_window_end - pool_window_start) * 0.04)
            pool_window_start -= pool_pad
            pool_window_end += pool_pad
        else:
            pool_window_start = None
            pool_window_end = None
        pool_histogram = kv_pool_histogram_bins(
            case_samples,
            due,
            pool_window_start,
            pool_window_end,
        )

        checkpoint_map = {
            "at_due": gap.get("tool_gap_end_ms"),
            "at_prefetch_h2d_start": direct_h2d_start,
            "at_replay_submit": gap.get("resume_submitted_ms"),
            "at_replay_start": gap.get("resume_start_ms"),
            "at_replay_h2d_start": replay_h2d_start,
            "at_replay_h2d_end": gap.get("replay_kv_h2d_end_ms"),
            "at_first_token": first_token_ms(gap),
        }
        checkpoint_values: dict[str, Any] = {}
        checkpoint_sources: dict[str, Any] = {}
        for name, target in checkpoint_map.items():
            sample = nearest_kv_pool_sample(case_samples, target)
            checkpoint_values[f"pool_usage_pct_{name}"] = sample.get("kv_pool_usage_pct", "") if sample else ""
            checkpoint_sources[f"pool_sample_distance_ms_{name}"] = sample.get("distance_from_target_ms", "") if sample else ""

        usage_candidates = [
            as_float(wait_stats.get("max_usage_pct")),
            as_float(before_replay_h2d_stats.get("max_usage_pct")),
            as_float(dispatch_stats.get("max_usage_pct")),
            *(as_float(value) for value in checkpoint_values.values()),
        ]
        max_usage = max([value for value in usage_candidates if value is not None] + [0.0])
        verdict = kv_pool_pressure_label(max_usage)
        rows.append(
            {
                "row": gap.get("timeline_label", ""),
                "case_id": gap.get("case_id", ""),
                "mode": gap.get("mode", ""),
                "fillers": case_fillers(gap),
                "tool_wait_ms": gap.get("tool_gap_ms", ""),
                "replay_path": gap.get("replay_path", ""),
                "kv_pool_verdict": verdict,
                "max_observed_usage_pct": round(max_usage, 3) if max_usage else "",
                "samples_in_case": len(case_samples),
                "samples_during_tool_wait": wait_stats.get("samples", 0),
                "max_usage_during_tool_wait_pct": wait_stats.get("max_usage_pct", ""),
                "samples_during_client_dispatch": dispatch_stats.get("samples", 0),
                "max_usage_during_client_dispatch_pct": dispatch_stats.get("max_usage_pct", ""),
                "samples_due_to_replay_h2d": before_replay_h2d_stats.get("samples", 0),
                "max_usage_due_to_replay_h2d_pct": before_replay_h2d_stats.get("max_usage_pct", ""),
                "pool_timeline_start_rel_ms": (
                    round(pool_window_start - due, 3)
                    if pool_window_start is not None and due is not None
                    else ""
                ),
                "pool_timeline_end_rel_ms": (
                    round(pool_window_end - due, 3)
                    if pool_window_end is not None and due is not None
                    else ""
                ),
                "pool_timeline_bins_json": json.dumps(pool_histogram, separators=(",", ":")),
                **checkpoint_values,
                **checkpoint_sources,
                "evidence": "direct_sglang_kv_pool_state" if case_samples else "missing_pool_samples",
                "simple_meaning": (
                    f"SGLang KV pool looked {verdict} near this replay."
                    if case_samples
                    else "No direct SGLang KV-pool sample was available. Rerun with AGENTIC_KV_TRACE_KV_POOL=1."
                ),
            }
        )
    return rows


def gpu_kv_residency_summary_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No GPU KV residency rows were available.</p>"
    observed = [row for row in rows if row.get("evidence") == "direct_sglang_kv_pool_state"]
    counts = Counter(str(row.get("kv_pool_verdict") or "unknown") for row in observed)
    cards = [
        ("gaps with pool samples", f"{len(observed)} / {len(rows)}"),
        ("very high pressure gaps", counts.get("very high", 0)),
        ("high pressure gaps", counts.get("high", 0)),
        ("medium pressure gaps", counts.get("medium", 0)),
    ]
    cards_html = "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{html.escape(label)}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
        for label, value in cards
    ) + "</div>"
    return f"""
    <p>This chart uses direct SGLang KV-pool samples. It answers: was the GPU KV pool already full or nearly full around replay/H2D time?</p>
    {cards_html}
    <div class="setup-diagram">{build_gpu_kv_residency_svg(rows)}</div>
    <p class="note">Thresholds: medium is 65-85%, high is 85-95%, very high is 95%+. These are KV-pool occupancy signals from SGLang internals, not total GPU memory from <code>nvidia-smi</code>.</p>
    """


def build_gpu_kv_residency_svg(rows: list[dict[str, Any]], max_rows: int = 48) -> str:
    plotted = rows[:max_rows]
    if not plotted:
        return "<p>No GPU KV residency rows were available.</p>"
    width = 1500
    height = 520
    left = 90
    right = 40
    top = 45
    bottom = 75
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_for(idx: int) -> float:
        return left + (idx + 0.5) * plot_w / max(1, len(plotted))

    def y_for(value: float) -> float:
        return top + (100.0 - max(0.0, min(100.0, value))) * plot_h / 100.0

    colors = {
        "due": "#111827",
        "replay_start": "#2563eb",
        "h2d_start": "#0891b2",
        "h2d_end": "#14b8a6",
        "prefetch": "#16a34a",
    }
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="GPU KV pool residency around replay">',
        '<text x="12" y="24" font-size="18" font-weight="900" fill="#0f172a">GPU KV pool residency around replay</text>',
    ]
    for threshold, label, fill in [(95, "very high", "#fee2e2"), (85, "high", "#ffedd5"), (65, "medium", "#fef9c3")]:
        y = y_for(threshold)
        parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{max(0, y - top):.1f}" fill="{fill}" opacity="0.32"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#94a3b8" stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{left + plot_w + 6}" y="{y + 4:.1f}" font-size="10" fill="#475569">{label} {threshold}%</text>')
    for tick in [0, 25, 50, 75, 100]:
        y = y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#475569">{tick}%</text>')
    parts.append(f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" font-size="12" font-weight="800" fill="#0f172a">SGLang KV pool usage</text>')
    marker_defs = [
        ("pool_usage_pct_at_due", "due", "circle"),
        ("pool_usage_pct_at_replay_start", "replay_start", "square"),
        ("pool_usage_pct_at_prefetch_h2d_start", "prefetch", "diamond"),
        ("pool_usage_pct_at_replay_h2d_start", "h2d_start", "triangle"),
        ("pool_usage_pct_at_replay_h2d_end", "h2d_end", "square"),
    ]
    for idx, row in enumerate(plotted):
        x = x_for(idx)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f1f5f9"/>')
        for key, name, shape in marker_defs:
            value = as_float(row.get(key))
            if value is None:
                continue
            y = y_for(value)
            color = colors[name]
            title = f"{row.get('row')} | {name}: {value:.1f}% | {row.get('kv_pool_verdict', '')}"
            if shape == "circle":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{html.escape(title)}</title></circle>')
            elif shape == "triangle":
                parts.append(f'<path d="M {x:.1f} {y - 6:.1f} L {x - 6:.1f} {y + 5:.1f} L {x + 6:.1f} {y + 5:.1f} Z" fill="{color}"><title>{html.escape(title)}</title></path>')
            elif shape == "diamond":
                parts.append(f'<path d="M {x:.1f} {y - 6:.1f} L {x + 6:.1f} {y:.1f} L {x:.1f} {y + 6:.1f} L {x - 6:.1f} {y:.1f} Z" fill="{color}"><title>{html.escape(title)}</title></path>')
            else:
                parts.append(f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2" fill="{color}"><title>{html.escape(title)}</title></rect>')
        if idx % max(1, len(plotted) // 12) == 0:
            parts.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" font-size="9" fill="#475569">{html.escape(str(row.get("row") or idx))}</text>')
    legend_x = left
    legend_y = height - 28
    for idx, (_, name, shape) in enumerate(marker_defs):
        x = legend_x + idx * 230
        color = colors[name]
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="{legend_y}" font-size="11" fill="#334155">{html.escape(name.replace("_", " "))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


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
    if global_kv_readiness_by_mode_rows(gaps):
        return "Global KV Readiness By Mode"
    if any(
        canonical_mode(row.get("mode")) == "no_prefetch" and has_events(row.get("replay_kv_h2d_events"))
        for row in gaps
    ):
        return "Global Replay H2D Readiness"
    return "Global Prefetch Margin"


def global_readiness_html(gaps: list[dict[str, Any]]) -> str:
    has_prefetch_margins = any(as_float(row.get("prefetch_margin_ms")) is not None for row in gaps)
    has_no_prefetch_h2d = any(
        canonical_mode(row.get("mode")) == "no_prefetch" and has_events(row.get("replay_kv_h2d_events"))
        for row in gaps
    )
    sections: list[str] = []
    mode_rows = global_kv_readiness_by_mode_rows(gaps)
    if mode_rows:
        sections.append(global_kv_readiness_by_mode_html(gaps))
    if has_no_prefetch_h2d and not mode_rows:
        sections.append("<h3>No-Prefetch Replay Queue And H2D Detail</h3>")
        sections.append(global_replay_h2d_readiness_html(gaps))
    if has_prefetch_margins and not mode_rows:
        sections.append("<h3>Direct-Prefetch Margin</h3>")
        sections.append(live_global_prefetch_margin_html(gaps))
    if not sections:
        return "<p>No prefetch-margin or replay-side H2D readiness rows were available for this run.</p>"
    return "\n".join(sections)


def prefetch_truth_check_html(gaps: list[dict[str, Any]]) -> str:
    cards = prefetch_truth_metric_cards(gaps)
    if not cards:
        return "<p>No measured prefetch-mode rows were available for this run.</p>"
    summary_rows = prefetch_truth_summary_rows(gaps)
    return f"""
    <p>This section is the guardrail against overclaiming. A purple prefetch bar only means the software hint/direct-load path ran. It does <strong>not</strong> automatically mean the right KV was useful at replay time.</p>
    <p class="note">A row counts as a true KV prefetch success only when the hint completed before replay, hint-side KV H2D/residency evidence exists, and replay did not reload or recompute the same useful context. Everything else is labeled as an early hint, late hint, replay reload, replay recompute, or insufficient reuse evidence.</p>
    <div class="metric-grid">
      {''.join(f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(value)}</div><div class="metric-sub">{html.escape(subtitle)}</div></div>' for label, value, subtitle in cards)}
    </div>
    <h3>Truth Verdict Summary</h3>
    {table_html(summary_rows)}
    <p class="note">The full per-gap truth table is in <strong>Evidence Tables / Raw Proof</strong> as <code>prefetch_truth_table.csv</code>.</p>
    """


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
            ("AGENTIC_KV_TRACE_KV_POOL", run_config.get("AGENTIC_KV_TRACE_KV_POOL") or "1"),
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


def compact_quantity(value: Any, suffix: str = "") -> str:
    number = as_float(value)
    if number is None:
        return ""
    abs_number = abs(number)
    if abs_number >= 1_000_000:
        text = f"{number / 1_000_000:.1f}M"
    elif abs_number >= 1_000:
        text = f"{number / 1_000:.1f}k"
    elif float(number).is_integer():
        text = str(int(number))
    else:
        text = f"{number:.1f}"
    text = text.replace(".0M", "M").replace(".0k", "k")
    return f"{text} {suffix}".strip()


def compact_tokens(value: Any) -> str:
    return compact_quantity(value, "idx")


def compact_token_count(value: Any) -> str:
    return compact_quantity(value, "tok")


def short_bar_label(parts: list[str], max_chars: int = 42) -> str:
    label = " / ".join(part for part in parts if part)
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1].rstrip() + "..."


def target_replay_h2d_summary(
    row: dict[str, Any],
    row_events: list[dict[str, Any]],
    target_session: str,
) -> dict[str, Any]:
    replay_start = as_float(row.get("replay_kv_h2d_start_ms"))
    replay_end = as_float(row.get("replay_kv_h2d_end_ms"))
    selected: list[dict[str, Any]] = []
    for event in row_events:
        if str(event.get("movement_kind") or "") != "H2D":
            continue
        if target_session and str(event.get("ledger_session_id") or "") != target_session:
            continue
        event_start = as_float(event.get("aligned_start_ms"))
        event_end = as_float(event.get("aligned_end_ms"))
        if event_start is None or event_end is None:
            continue
        if replay_start is not None and replay_end is not None:
            if event_end < replay_start - 8.0 or event_start > replay_end + 8.0:
                continue
        selected.append(event)

    block_keys = {
        str(event.get("block_key") or event.get("logical_block_id") or "")
        for event in selected
        if str(event.get("block_key") or event.get("logical_block_id") or "")
    }
    token_count = sum(as_float(event.get("token_or_index_count")) or 0.0 for event in selected)
    duration = None
    if replay_start is not None and replay_end is not None and replay_end > replay_start:
        duration = replay_end - replay_start
    elif selected:
        start_values = [as_float(event.get("aligned_start_ms")) for event in selected]
        end_values = [as_float(event.get("aligned_end_ms")) for event in selected]
        start_values = [value for value in start_values if value is not None]
        end_values = [value for value in end_values if value is not None]
        if start_values and end_values:
            duration = max(end_values) - min(start_values)
    return {
        "events": len(selected),
        "blocks": len(block_keys) or len(selected),
        "tokens": token_count,
        "duration_ms": duration,
        "events_rows": selected,
    }


def kv_event_tooltip(label: str, event: dict[str, Any], span: tuple[float, float], target: bool) -> str:
    fields = [
        f"{label} expanded KV burst",
        f"kind={event.get('movement_kind', '')}",
        f"target_row={'yes' if target else 'no'}",
        f"owner={event.get('owner_kind', '')}",
        f"phase={event.get('phase', '')}",
        f"session={event.get('ledger_session_id', '')}",
        f"request={event.get('request_id', '')}",
        f"node={event.get('node_id', '')}",
        f"host_idx={event.get('host_index_start', '')}..{event.get('host_index_end', '')}",
        f"device_idx={event.get('device_index_start', '')}..{event.get('device_index_end', '')}",
        f"count={event.get('token_or_index_count', '')}",
        f"duration={display_ms(as_float(event.get('duration_ms')) or max(0.0, span[1] - span[0]))}",
        f"relative={display_ms(span[0])}->{display_ms(span[1])}",
        f"source={event.get('source_event', '')}",
        f"evidence={event.get('evidence_level', '') or event.get('evidence_confidence', '')}",
        f"correlation={event.get('exact_correlation_source', '')}",
    ]
    meaning = str(event.get("simple_meaning") or "")
    if meaning:
        fields.append(meaning)
    return " | ".join(part for part in fields if not part.endswith("="))


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
            "direct_kv_h2d_start_ms",
            "direct_kv_h2d_end_ms",
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
        "projected_hardware": "#0f766e",
        "marker": "#475569",
    }.get(kind, "#64748b")


def unified_stack_legend_table_html() -> str:
    rows = [
        (
            "Initial model turn",
            unified_stack_color("initial"),
            "The first model request before the agent/tool wait begins.",
        ),
        (
            "Tool wait",
            unified_stack_color("tool_wait"),
            "The pause while a tool is running. This is the opportunity window where prefetch could help.",
        ),
        (
            "Client dispatch",
            unified_stack_color("client_dispatch"),
            "The replay request has become due, but it is still in the driver/client path before useful SGLang work starts.",
        ),
        (
            "Scheduler / load path",
            unified_stack_color("scheduler"),
            "SGLang receive, scheduler queue/admit, cache lookup, or load-back decision work before useful model/KV work.",
        ),
        (
            "Direct prefetch attempt",
            unified_stack_color("prefetch"),
            "The direct SGLang KV load-back hook path. This is not prompt warming and is separate from priority-hint-only mode.",
        ),
        (
            "Hint-side KV H2D",
            unified_stack_color("hint_h2d"),
            "Host-to-device KV movement caused by the direct prefetch/hint path before or around replay.",
        ),
        (
            "Replay KV H2D",
            unified_stack_color("h2d"),
            "KV cache data loaded from host memory back into GPU memory. In the replay zoom, this is the replay request's own KV load-back.",
        ),
        (
            "D2H / offload",
            unified_stack_color("d2h"),
            "Device-to-host KV movement: KV is backed up or offloaded from GPU memory to host memory.",
        ),
        (
            "Evict",
            unified_stack_color("evict"),
            "KV leaves GPU residency. A host copy may still exist unless host eviction also happens.",
        ),
        (
            "Prefill / recompute",
            unified_stack_color("recompute"),
            "Model-forward work before the first output token. This can include recomputing missing KV or processing uncached replay prompt tokens.",
        ),
        (
            "Remaining before-first-token time",
            unified_stack_color("prefill"),
            "Leftover time before the first output token after visible H2D and prefill/recompute are separated. This can include batching, handoff, runtime overhead, or other unclassified wait.",
        ),
        (
            "Decode / token generation",
            unified_stack_color("decode"),
            "Generation after the first output token is produced.",
        ),
    ]
    body = []
    for name, color, meaning in rows:
        body.append(
            "<tr>"
            f'<td><span style="display:inline-block;width:54px;height:16px;border-radius:4px;'
            f'background:{html.escape(color)};border:1px solid #cbd5e1"></span></td>'
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(meaning)}</td>"
            "</tr>"
        )
    return (
        '<table class="compact-table">'
        "<thead><tr><th>Color</th><th>Timeline element</th><th>Simple meaning</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


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
        raw_mode = str(row.get("mode") or "")
        truth_label = str(row.get("prefetch_truth_short_label") or "")
        if canonical_mode(raw_mode) in PREFETCH_MODE_NAMES and truth_label:
            verdict_label = truth_label
        else:
            verdict_label = display_verdict(row.get("per_gap_verdict") or status)
        is_projected_hardware_row = canonical_mode(raw_mode) == "projected_hardware_bypass"
        mode_label = display_mode(raw_mode)
        parts.append(f'<text x="12" y="{y + 34:.1f}" font-size="10" font-weight="800" fill="{status_color}">{html.escape(verdict_label)}</text>')
        parts.append(f'<text x="12" y="{y + 52:.1f}" font-size="10" fill="#475569">mode {html.escape(mode_label)}; fillers {html.escape(case_fillers(row))}; wait {html.escape(str(row.get("tool_gap_ms") or ""))} ms</text>')
        if raw_mode and raw_mode != canonical_mode(raw_mode):
            parts.append(f'<text x="12" y="{y + 70:.1f}" font-size="9" fill="#94a3b8">run {html.escape(raw_mode[:50])}</text>')
            path_y = y + 86
        else:
            path_y = y + 70
        parts.append(f'<text x="12" y="{path_y:.1f}" font-size="10" fill="#475569">{html.escape(str(row.get("replay_path") or replay_path_from_evidence(row)))}</text>')
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
            prefill_recompute_start = as_float(row.get("replay_prefill_recompute_start_ms"))
            prefill_start_rel = (prefill_recompute_start - due) if prefill_recompute_start is not None else replay_start - due
            first_token_rel = first_token - due
            recompute_tokens = as_float(row.get("recomputed_tokens_est")) or as_float(row.get("replay_new_prefill_tokens_est")) or 0.0
            if recompute_tokens >= 128:
                draw_span(parts, prefill_start_rel, first_token_rel, replay_y + 10, 10, unified_stack_color("recompute"), "prefill/recompute", f"{label} | replay prefill/recompute window until first token: {display_ms(first_token_rel - prefill_start_rel)}", opacity=0.82)
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
        ("KV H2D", unified_stack_color("h2d")),
        ("D2H/offload", unified_stack_color("d2h")),
        ("evict", unified_stack_color("evict")),
        ("prefill/recompute", unified_stack_color("recompute")),
        ("remaining before-token", unified_stack_color("prefill")),
        ("decode", unified_stack_color("decode")),
    ]
    if any(canonical_mode(row.get("mode")) == "projected_hardware_bypass" for row in rows):
        legend.append(("projected hardware path", unified_stack_color("projected_hardware")))
    if projected_hardware_rows:
        legend.append(("projected hardware ready", unified_stack_color("projected_hardware")))
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
    kv_pool_residency_rows: list[dict[str, Any]] | None = None,
    projected_hardware_rows: list[dict[str, Any]] | None = None,
    compact_projected_rows: bool = False,
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

    width = 1880
    left = 330
    right = 56
    top = 194
    measured_row_h = 1200
    projected_row_h = 250
    scenario_gap_h = 44 if any(str(row.get("comparison_scenario") or "") for row in rows) else 0
    bottom = 116
    plot_w = width - left - right
    row_heights = [
        projected_row_h
        if compact_projected_rows and canonical_mode(row.get("mode")) == "projected_hardware_bypass"
        else measured_row_h
        for row in rows
    ]
    row_starts: list[int] = []
    cursor_y = top
    previous_scenario = ""
    for idx, row_height in enumerate(row_heights):
        scenario = str(rows[idx].get("comparison_scenario") or "")
        if idx > 0 and scenario_gap_h and scenario and scenario != previous_scenario:
            cursor_y += scenario_gap_h
        row_starts.append(cursor_y)
        cursor_y += row_height
        previous_scenario = scenario
    height = cursor_y + bottom
    scaled_min = h2d_symlog_value(x_min)
    scaled_max = h2d_symlog_value(x_max)
    main_bar_h = 22
    kv_target_bar_h = 16
    kv_other_bar_h = 12
    min_visible_bar_w = 7.0
    min_visible_event_w = 5.0
    kv_pool_by_label = {
        str(row.get("row") or ""): row for row in (kv_pool_residency_rows or []) if str(row.get("row") or "")
    }
    projected_by_label: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for projected in projected_hardware_rows or []:
        row_label = str(projected.get("row") or "")
        projection_kind = str(projected.get("hardware_projection") or "")
        if row_label and projection_kind:
            projected_by_label[row_label][projection_kind] = projected

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

    def replay_zoom_bounds(row: dict[str, Any], due: float) -> tuple[float, float] | None:
        values: list[float] = []
        for span in [
            _relative_span(row, "resume_start_ms", "resume_end_ms", due),
            _relative_span(row, "replay_kv_h2d_start_ms", "replay_kv_h2d_end_ms", due),
            _relative_span(row, "replay_prefill_start_ms", "replay_prefill_end_ms", due),
            _relative_span(row, "replay_model_forward_start_ms", "replay_model_forward_end_ms", due),
        ]:
            if span:
                values.extend(span)
        first_token = first_token_ms(row)
        if first_token is not None:
            values.append(first_token - due)
        replay_start = _relative_ms(row, "resume_start_ms", due)
        if replay_start is not None:
            values.append(replay_start)
        if not values:
            return None
        start = min(values)
        end = max(values)
        span_ms = max(1.0, end - start)
        padding = max(20.0, span_ms * 0.08)
        return start - padding, end + padding

    def zoom_x(value: float, z_min: float, z_max: float) -> float:
        return left + (value - z_min) * plot_w / max(1e-9, z_max - z_min)

    def local_zoom_x(value: float, z_min: float, z_max: float, x0: float, w: float) -> float:
        return x0 + (value - z_min) * w / max(1e-9, z_max - z_min)

    def draw_axis_break(
        parts: list[str],
        x: float,
        y1: float,
        y2: float,
        label_text: str,
    ) -> None:
        mid = (y1 + y2) / 2
        parts.append(
            f'<path d="M {x - 12:.1f} {y1:.1f} C {x - 18:.1f} {mid - 28:.1f}, {x - 4:.1f} {mid - 16:.1f}, {x - 10:.1f} {mid:.1f} '
            f'C {x - 16:.1f} {mid + 18:.1f}, {x - 4:.1f} {mid + 30:.1f}, {x - 10:.1f} {y2:.1f}" '
            f'stroke="#111827" stroke-width="2.4" fill="none" opacity="0.9"/>'
        )
        parts.append(
            f'<path d="M {x + 12:.1f} {y1:.1f} C {x + 6:.1f} {mid - 28:.1f}, {x + 20:.1f} {mid - 16:.1f}, {x + 14:.1f} {mid:.1f} '
            f'C {x + 8:.1f} {mid + 18:.1f}, {x + 20:.1f} {mid + 30:.1f}, {x + 14:.1f} {y2:.1f}" '
            f'stroke="#111827" stroke-width="2.4" fill="none" opacity="0.9"/>'
        )
        parts.append(f'<text x="{x:.1f}" y="{mid + 4:.1f}" text-anchor="middle" font-size="13" font-weight="900" fill="#111827">...</text>')
        if label_text:
            parts.append(
                f'<text x="{x:.1f}" y="{y2 + 18:.1f}" text-anchor="middle" font-size="8" '
                f'font-weight="800" fill="#475569">{html.escape(label_text)}</text>'
            )

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
        min_w: float = min_visible_bar_w,
        break_long: bool = False,
        label_min_w: float = 118.0,
        font_size: int = 10,
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
        if label and w >= label_min_w:
            text_x = x1 + w / 2
            text_fill = "#0f172a" if color in {
                unified_stack_color("tool_wait"),
                unified_stack_color("h2d"),
                unified_stack_color("prefill"),
                unified_stack_color("sglang_receive"),
            } else "#ffffff"
            if opacity < 0.74:
                text_fill = "#0f172a"
            parts.append(
                f'<text x="{text_x:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" font-size="{font_size}" '
                f'font-weight="900" fill="{text_fill}">{html.escape(label)}</text>'
            )

    def draw_small_bar_callout(
        parts: list[str],
        x1: float,
        x2: float,
        y: float,
        h: float,
        label: str,
        title: str,
        color: str,
        fill_color: str = "#ecfeff",
        text_color: str = "#155e75",
    ) -> None:
        if not label:
            return
        text_w = min(310.0, max(110.0, len(label) * 5.8 + 22.0))
        plot_right = left + plot_w
        if x2 + text_w + 12.0 <= plot_right:
            rect_x = x2 + 10.0
            text_x = rect_x + text_w / 2
            line_x1 = x2
            line_x2 = rect_x
        else:
            rect_x = max(left, x1 - text_w - 10.0)
            text_x = rect_x + text_w / 2
            line_x1 = rect_x + text_w
            line_x2 = x1
        rect_y = y - 1.0
        center_y = y + h / 2
        parts.append(
            f'<line x1="{line_x1:.1f}" y1="{center_y:.1f}" x2="{line_x2:.1f}" y2="{center_y:.1f}" '
            f'stroke="{color}" stroke-width="1.4" opacity="0.72"><title>{html.escape(title)}</title></line>'
        )
        parts.append(
            f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{text_w:.1f}" height="{h + 2:.1f}" rx="5" '
            f'fill="{fill_color}" stroke="{color}" stroke-width="1.2" opacity="0.96"><title>{html.escape(title)}</title></rect>'
        )
        parts.append(
            f'<text x="{text_x:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" font-size="9" '
            f'font-weight="900" fill="{text_color}">{html.escape(label)}</text>'
        )

    def draw_overview_span(
        parts: list[str],
        start: float,
        end: float,
        y: float,
        h: float,
        color: str,
        label: str,
        title: str,
        opacity: float = 0.86,
        min_w: float = min_visible_bar_w,
        break_long: bool = True,
        label_min_w: float = 118.0,
        font_size: int = 10,
        callout_fill: str = "#ffffff",
        callout_text: str = "#0f172a",
    ) -> None:
        clipped_start = max(x_min, min(x_max, start))
        clipped_end = max(x_min, min(x_max, end))
        if clipped_end <= clipped_start:
            return
        x1 = overview_x(clipped_start)
        x2 = overview_x(clipped_end)
        w = max(min_w, x2 - x1)
        draw_span(
            parts,
            overview_x,
            x_min,
            x_max,
            start,
            end,
            y,
            h,
            color,
            label,
            title,
            opacity=opacity,
            min_w=min_w,
            break_long=break_long,
            label_min_w=label_min_w,
            font_size=font_size,
        )
        if label and w < label_min_w:
            draw_small_bar_callout(
                parts,
                x1,
                x1 + w,
                y,
                h,
                label,
                title,
                color,
                fill_color=callout_fill,
                text_color=callout_text,
            )

    def draw_overview_marker(parts: list[str], value: float, y1: float, y2: float, color: str, title: str) -> None:
        if value < x_min or value > x_max:
            return
        x = overview_x(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2.2">'
            f'<title>{html.escape(title)}</title></line>'
        )

    def draw_overview_local_axis(parts: list[str], row_y: float) -> None:
        """Draw compact replay-relative timestamps for the per-row overview lanes."""
        axis_y = row_y + 10.0
        label_y = row_y + 3.0
        last_label_x = -10**9
        parts.append(
            f'<line x1="{left:.1f}" y1="{axis_y:.1f}" x2="{left + plot_w:.1f}" y2="{axis_y:.1f}" '
            f'stroke="#cbd5e1" stroke-width="0.9" opacity="0.72"/>'
        )
        for value in ticks:
            if value < x_min or value > x_max:
                continue
            x = overview_x(value)
            if x - last_label_x < 72:
                continue
            label = f"{int(value)} ms" if abs(value) < 1000 else f"{value / 1000:.0f} s"
            parts.append(
                f'<line x1="{x:.1f}" y1="{axis_y - 4:.1f}" x2="{x:.1f}" y2="{axis_y + 4:.1f}" '
                f'stroke="#94a3b8" stroke-width="0.8" opacity="0.8"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="8" '
                f'font-weight="800" fill="#64748b">{html.escape(label)}</text>'
            )
            last_label_x = x

    zero_x = overview_x(0.0)
    legend = [
        ("initial", unified_stack_color("initial")),
        ("tool wait", unified_stack_color("tool_wait")),
        ("client dispatch", unified_stack_color("client_dispatch")),
        ("scheduler/load path", unified_stack_color("scheduler")),
        ("direct prefetch", unified_stack_color("prefetch")),
        ("hint-side KV H2D", unified_stack_color("hint_h2d")),
        ("KV H2D", unified_stack_color("h2d")),
        ("D2H/offload", unified_stack_color("d2h")),
        ("evict", unified_stack_color("evict")),
        ("prefill/recompute", unified_stack_color("recompute")),
        ("remaining before-token", unified_stack_color("prefill")),
        ("decode", unified_stack_color("decode")),
    ]

    def append_legend(parts: list[str], x_start: float, y_pos: float) -> None:
        col_w = 285
        row_gap = 24
        for idx, (legend_label, color) in enumerate(legend):
            lx = x_start + (idx % 5) * col_w
            ly = y_pos + (idx // 5) * row_gap
            parts.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="14" height="14" rx="3" fill="{color}" opacity="0.88"/>')
            parts.append(f'<text x="{lx + 20:.1f}" y="{ly + 12:.1f}" font-size="11" fill="#334155">{html.escape(legend_label)}</text>')

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Unified per-gap forensic stack timeline with per-gap KV zoom">',
        '<text x="12" y="30" font-size="20" font-weight="800" fill="#0f172a">Unified per-gap forensic stack timeline</text>',
        '<text x="12" y="54" font-size="12" fill="#475569">Each gap has a compact overview, an expanded KV activity zoom, an expanded replay zoom, a GPU KV-pool residency zoom, and a KV readiness deadline zoom.</text>',
        '<text x="12" y="76" font-size="12" fill="#475569">The overview shows the broad timing story; the expanded zoom lanes give dense KV, replay, pool pressure, and deadline activity room to breathe.</text>',
        f'<line x1="{zero_x:.1f}" y1="{top - 32}" x2="{zero_x:.1f}" y2="{height - 70}" stroke="#111827" stroke-width="2.4"/>',
        f'<text x="{zero_x + 6:.1f}" y="{top - 42}" font-size="12" font-weight="800">0 ms replay due</text>',
    ]
    append_legend(parts, left, 102)
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
        y = row_starts[idx]
        row_h = row_heights[idx]
        raw_mode = str(row.get("mode") or "")
        scenario = str(row.get("comparison_scenario") or "")
        previous_scenario_for_row = str(rows[idx - 1].get("comparison_scenario") or "") if idx > 0 else ""
        if idx > 0 and scenario_gap_h and scenario and scenario != previous_scenario_for_row:
            sep_y = y - scenario_gap_h / 2.0
            parts.append(
                f'<line x1="18" y1="{sep_y:.1f}" x2="{width - 24}" y2="{sep_y:.1f}" '
                f'stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="8 8" opacity="0.65"/>'
            )
            parts.append(
                f'<rect x="18" y="{sep_y - 13:.1f}" width="92" height="24" rx="12" '
                f'fill="#ffffff" stroke="#cbd5e1" opacity="0.98"/>'
            )
            parts.append(
                f'<text x="64" y="{sep_y + 4:.1f}" text-anchor="middle" font-size="11" '
                f'font-weight="900" fill="#334155">{html.escape(scenario)}</text>'
            )
        band, mode_accent, band_opacity = mode_row_background_style(raw_mode, idx)
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        kv_pool_row = kv_pool_by_label.get(label, {})
        status, status_color = observation_status(row)
        row_events = unified_stack_kv_events_for_gap(row, all_kv_events)
        target_session = str(row.get("ledger_session_id") or row.get("session_id") or "")
        event_counts: Counter[str] = Counter(str(event.get("movement_kind") or "") for event in row_events)
        replay_h2d_summary = target_replay_h2d_summary(row, row_events, target_session)
        replay_start_abs_for_summary = as_float(row.get("resume_start_ms"))
        first_token_abs_for_summary = first_token_ms(row)
        ttft_ms = (
            first_token_abs_for_summary - replay_start_abs_for_summary
            if first_token_abs_for_summary is not None and replay_start_abs_for_summary is not None
            else None
        )
        dispatch_start_abs = as_float(row.get("resume_submitted_ms"))
        dispatch_ms = (
            replay_start_abs_for_summary - dispatch_start_abs
            if replay_start_abs_for_summary is not None and dispatch_start_abs is not None
            else None
        )
        recompute_tokens_for_summary = (
            as_float(row.get("recomputed_tokens_est"))
            or as_float(row.get("replay_new_prefill_tokens_est"))
            or 0.0
        )
        zoom = zoom_bounds(row, row_events, due)
        replay_zoom = replay_zoom_bounds(row, due)

        parts.append(
            f'<rect x="0" y="{y - 10:.1f}" width="{width}" height="{row_h - 12}" '
            f'fill="{band}" opacity="{band_opacity:.2f}"/>'
        )
        parts.append(
            f'<rect x="0" y="{y - 10:.1f}" width="9" height="{row_h - 12}" '
            f'fill="{mode_accent}" opacity="0.92"/>'
        )
        parts.append(f'<rect x="{left - 2:.1f}" y="{y + 10:.1f}" width="{plot_w + 4:.1f}" height="196" rx="8" fill="#ffffff" opacity="0.38"/>')
        if not (compact_projected_rows and canonical_mode(row.get("mode")) == "projected_hardware_bypass"):
            parts.append(f'<rect x="{left - 2:.1f}" y="{y + 228:.1f}" width="{plot_w + 4:.1f}" height="170" rx="8" fill="#f8fafc" opacity="0.80"/>')
            parts.append(f'<rect x="{left - 2:.1f}" y="{y + 428:.1f}" width="{plot_w + 4:.1f}" height="344" rx="8" fill="#fff7ed" opacity="0.42"/>')
            parts.append(f'<rect x="{left - 2:.1f}" y="{y + 792:.1f}" width="{plot_w + 4:.1f}" height="150" rx="8" fill="#f0fdf4" opacity="0.70"/>')
            parts.append(f'<rect x="{left - 2:.1f}" y="{y + 962:.1f}" width="{plot_w + 4:.1f}" height="160" rx="8" fill="#f8fafc" opacity="0.92"/>')
        parts.append(f'<text x="16" y="{y + 18:.1f}" font-size="16" font-weight="900">{html.escape(label)}</text>')
        is_projected_hardware_row = canonical_mode(raw_mode) == "projected_hardware_bypass"
        truth_label = str(row.get("prefetch_truth_short_label") or "")
        if canonical_mode(raw_mode) in PREFETCH_MODE_NAMES and truth_label:
            verdict_label = truth_label
        else:
            verdict_label = display_verdict(row.get("per_gap_verdict") or status)
        mode_label = display_mode(raw_mode)
        parts.append(f'<text x="16" y="{y + 42:.1f}" font-size="10" font-weight="900" fill="{status_color}">{html.escape(verdict_label)}</text>')
        mode_fg, mode_bg = mode_badge_style(raw_mode)
        badge_w = max(96.0, min(240.0, len(mode_label) * 7.0 + 24.0))
        parts.append(
            f'<rect x="16" y="{y + 52:.1f}" width="{badge_w:.1f}" height="20" rx="6" '
            f'fill="{mode_bg}" stroke="{mode_fg}" stroke-width="1.2" opacity="0.98"/>'
        )
        parts.append(
            f'<text x="{16 + badge_w / 2:.1f}" y="{y + 66:.1f}" text-anchor="middle" '
            f'font-size="10" font-weight="900" fill="{mode_fg}">mode: {html.escape(mode_label)}</text>'
        )
        if is_projected_hardware_row:
            source_mode = str(row.get("source_measured_mode") or "")
            source_note = f"based on {source_mode}" if source_mode else "projected row"
            parts.append(
                f'<text x="16" y="{y + 78:.1f}" font-size="9" font-weight="900" fill="#0f766e">'
                f'PROJECTED, NOT MEASURED - {html.escape(source_note)}</text>'
            )
        parts.append(f'<text x="16" y="{y + 84:.1f}" font-size="10" fill="#475569">fillers {html.escape(case_fillers(row))} | wait {html.escape(str(row.get("tool_gap_ms") or ""))} ms</text>')
        parts.append(
            f'<text x="16" y="{y + 108:.1f}" font-size="10" fill="#475569">'
            f'H2D {event_counts.get("H2D", 0)} | D2H {event_counts.get("D2H", 0)} | evict {event_counts.get("GPU evict", 0)}'
            f'</text>'
        )
        replay_h2d_left = (
            f"replay H2D {replay_h2d_summary['blocks']} blk / {compact_tokens(replay_h2d_summary['tokens'])}"
            if replay_h2d_summary["events"]
            else "replay H2D none"
        )
        parts.append(f'<text x="16" y="{y + 132:.1f}" font-size="10" fill="#475569">{html.escape(replay_h2d_left[:54])}</text>')
        replay_work_parts = []
        prefetch_start_abs_for_summary = as_float(row.get("prefetch_start_ms"))
        prefetch_end_abs_for_summary = as_float(row.get("prefetch_end_ms"))
        prefetch_margin_for_summary = as_float(row.get("prefetch_margin_ms"))
        if prefetch_start_abs_for_summary is not None and prefetch_end_abs_for_summary is not None:
            prefetch_summary = f"prefetch {display_ms(prefetch_end_abs_for_summary - prefetch_start_abs_for_summary)}"
            if prefetch_margin_for_summary is not None:
                prefetch_summary += f" | margin {display_ms(prefetch_margin_for_summary)}"
            replay_work_parts.append(prefetch_summary)
        if recompute_tokens_for_summary >= 128:
            replay_work_parts.append(f"recompute {compact_token_count(recompute_tokens_for_summary)}")
        if ttft_ms is not None:
            replay_work_parts.append(f"TTFT {display_ms(ttft_ms)}")
        if dispatch_ms is not None:
            replay_work_parts.append(f"dispatch {display_ms(dispatch_ms)}")
        if replay_work_parts:
            parts.append(
                f'<text x="16" y="{y + 156:.1f}" font-size="10" fill="#475569">'
                f'{html.escape(" | ".join(replay_work_parts)[:56])}</text>'
            )
        pool_text = ""
        if kv_pool_row and kv_pool_row.get("evidence") == "direct_sglang_kv_pool_state":
            max_pool = kv_pool_row.get("max_observed_usage_pct", "")
            pool_verdict = kv_pool_row.get("kv_pool_verdict", "")
            h2d_pool = (
                kv_pool_row.get("pool_usage_pct_at_replay_h2d_start")
                or kv_pool_row.get("pool_usage_pct_at_prefetch_h2d_start")
                or ""
            )
            pool_text = f"GPU KV pool max {max_pool}%"
            if h2d_pool not in ("", None):
                pool_text += f" | H2D {h2d_pool}%"
            if pool_verdict:
                pool_text += f" | {pool_verdict}"
        else:
            pool_text = "GPU KV pool: no sample"
        parts.append(f'<text x="16" y="{y + 180:.1f}" font-size="10" fill="#475569">{html.escape(pool_text[:58])}</text>')
        parts.append(
            f'<text x="16" y="{y + 204:.1f}" font-size="10" fill="#475569">'
            f'{html.escape(str(row.get("replay_path") or replay_path_from_evidence(row))[:54])}</text>'
        )

        overview_lanes = [
            ("overview", y + 28),
            ("prefetch path", y + 64),
            ("request path", y + 100),
            ("replay summary", y + 136),
        ]
        draw_overview_local_axis(parts, y + 12)
        for lane_label, lane_y in overview_lanes:
            parts.append(f'<text x="{left - 10}" y="{lane_y + 9:.1f}" text-anchor="end" font-size="10" font-weight="800" fill="#334155">{html.escape(lane_label)}</text>')
            parts.append(f'<line x1="{left}" y1="{lane_y + 5:.1f}" x2="{left + plot_w}" y2="{lane_y + 5:.1f}" stroke="#dbe4ee"/>')

        overview_y = y + 28
        for span, color, span_label, title, opacity in [
            (_relative_span(row, "current_start_ms", "current_end_ms", due), unified_stack_color("initial"), "initial", "initial model turn", 0.84),
            (_relative_span(row, "tool_gap_start_ms", "tool_gap_end_ms", due), unified_stack_color("tool_wait"), "tool wait", "agent/tool wait window", 0.78),
            (_relative_span(row, "resume_start_ms", "resume_end_ms", due), unified_stack_color("decode"), "resume", "resume request wall time", 0.70),
        ]:
            if span:
                draw_overview_span(parts, span[0], span[1], overview_y - 4, main_bar_h, color, span_label, f"{label} | {title}: {display_ms(span[1] - span[0])}", opacity=opacity)

        prefetch_y = y + 64
        prefetch_span = _relative_span(row, "prefetch_start_ms", "prefetch_end_ms", due)
        if prefetch_span:
            prefetch_duration = prefetch_span[1] - prefetch_span[0]
            prefetch_margin = as_float(row.get("prefetch_margin_ms"))
            margin_text = f" | margin={display_ms(prefetch_margin)}" if prefetch_margin is not None else ""
            prefetch_color = unified_stack_color("projected_hardware") if is_projected_hardware_row else unified_stack_color("prefetch")
            prefetch_label = "projected HW path" if is_projected_hardware_row else "direct prefetch"
            prefetch_title = (
                f"{label} | PROJECTED, NOT MEASURED hardware KV path: {display_ms(prefetch_duration)}{margin_text}"
                if is_projected_hardware_row
                else f"{label} | direct KV prefetch attempt: {display_ms(prefetch_duration)}{margin_text}"
            )
            draw_overview_span(
                parts,
                prefetch_span[0],
                prefetch_span[1],
                prefetch_y - 4,
                main_bar_h,
                prefetch_color,
                prefetch_label,
                prefetch_title,
                opacity=0.74,
                break_long=True,
                label_min_w=118.0,
                callout_fill="#f5f3ff",
                callout_text="#4c1d95",
            )
        hint_h2d_span = _relative_span(row, "direct_kv_h2d_start_ms", "direct_kv_h2d_end_ms", due)
        if hint_h2d_span:
            hint_duration = hint_h2d_span[1] - hint_h2d_span[0]
            hint_color = unified_stack_color("projected_hardware") if is_projected_hardware_row else unified_stack_color("hint_h2d")
            hint_label = "projected HW H2D" if is_projected_hardware_row else "hint H2D"
            hint_title = (
                f"{label} | PROJECTED, NOT MEASURED hardware KV H2D: {display_ms(hint_duration)} | "
                f"estimated from measured source={row.get('measured_h2d_source', '')}"
                if is_projected_hardware_row
                else f"{label} | hint-side direct KV H2D: {display_ms(hint_duration)} | events={row.get('direct_kv_h2d_events', '')}"
            )
            draw_overview_span(
                parts,
                hint_h2d_span[0],
                hint_h2d_span[1],
                prefetch_y + 20,
                max(12.0, main_bar_h - 6),
                hint_color,
                hint_label,
                hint_title,
                opacity=0.92,
                label_min_w=78.0,
                font_size=9,
                callout_fill="#ecfdf5",
                callout_text="#14532d",
            )

        if compact_projected_rows and is_projected_hardware_row:
            projected_margin = as_float(row.get("projected_hardware_margin_ms") or row.get("prefetch_margin_ms"))
            projected_duration = as_float(row.get("projected_hardware_duration_ms") or row.get("direct_kv_h2d_duration_ms"))
            measured_source = str(row.get("measured_h2d_source") or "measured KV H2D duration")
            margin_label = (
                f"projected ready {display_ms(projected_margin)} before replay"
                if projected_margin is not None and projected_margin >= 0
                else f"projected late {display_ms(abs(projected_margin))}"
                if projected_margin is not None
                else "projected readiness unknown"
            )
            margin_color = "#16a34a" if projected_margin is not None and projected_margin >= 0 else "#dc2626"
            parts.append(
                f'<text x="{left + 8:.1f}" y="{y + 160:.1f}" font-size="11" font-weight="900" '
                f'fill="#0f766e">PROJECTED, NOT MEASURED</text>'
            )
            parts.append(
                f'<text x="{left + 8:.1f}" y="{y + 180:.1f}" font-size="10" fill="#475569">'
                f'Hardware bypass estimate uses {html.escape(measured_source)} plus fixed control overhead; detailed measured SGLang lanes are intentionally hidden.</text>'
            )
            parts.append(
                f'<text x="{left + 8:.1f}" y="{y + 202:.1f}" font-size="11" font-weight="900" '
                f'fill="{margin_color}">{html.escape(margin_label)}</text>'
            )
            if projected_duration is not None:
                parts.append(
                    f'<text x="{left + 320:.1f}" y="{y + 202:.1f}" font-size="10" fill="#475569">'
                    f'projected KV movement duration: {html.escape(display_ms(projected_duration))}</text>'
                )
            continue

        request_y = y + 100
        request_spans = [
            (_relative_span(row, "resume_submitted_ms", "resume_start_ms", due), unified_stack_color("client_dispatch"), "client dispatch", "client dispatch: replay submitted but client call had not started"),
            (_relative_span(row, "replay_sglang_receive_start_ms", "replay_sglang_receive_end_ms", due), unified_stack_color("sglang_receive"), "receive", "SGLang receive stage"),
            (_relative_span(row, "replay_scheduler_queue_enter_start_ms", "replay_scheduler_admit_start_ms", due), unified_stack_color("scheduler"), "scheduler", "scheduler queue/admit wait"),
            (_relative_span(row, "replay_scheduler_admit_start_ms", "replay_kv_h2d_start_ms", due), unified_stack_color("load_path"), "load path", "scheduler admit to visible replay H2D start"),
        ]
        for span, color, span_label, title in request_spans:
            if span:
                draw_overview_span(parts, span[0], span[1], request_y - 4, main_bar_h, color, span_label, f"{label} | {title}: {display_ms(span[1] - span[0])}")

        replay_y = y + 136
        replay_h2d = _relative_span(row, "replay_kv_h2d_start_ms", "replay_kv_h2d_end_ms", due)
        if replay_h2d:
            draw_overview_span(
                parts,
                replay_h2d[0],
                replay_h2d[1],
                replay_y - 6,
                main_bar_h,
                unified_stack_color("h2d"),
                "replay H2D",
                f"{label} | replay-side KV H2D: {display_ms(replay_h2d[1] - replay_h2d[0])}",
                label_min_w=82.0,
                callout_fill="#ecfeff",
                callout_text="#155e75",
            )
        first_token = first_token_ms(row)
        replay_start = as_float(row.get("resume_start_ms"))
        if first_token is not None and replay_start is not None:
            prefill_start_rel = replay_start - due
            first_token_rel = first_token - due
            runtime_attr_tokens = as_float(row.get("replay_runtime_prefill_attributed_tokens"))
            runtime_attr_start_abs = as_float(row.get("replay_runtime_prefill_attribution_start_ms"))
            runtime_attr_end_abs = as_float(row.get("replay_runtime_prefill_attribution_end_ms"))
            runtime_attr_confidence = str(row.get("replay_runtime_prefill_confidence") or "")
            has_runtime_prefill_timing = (
                runtime_attr_confidence == "runtime_attributed"
                and runtime_attr_start_abs is not None
                and runtime_attr_end_abs is not None
                and runtime_attr_end_abs > runtime_attr_start_abs
            )
            recompute_tokens = runtime_attr_tokens or as_float(row.get("recomputed_tokens_est")) or as_float(row.get("replay_new_prefill_tokens_est")) or 0.0
            segments = replay_phase_segments(row)
            recompute_ms = segments.get("recompute", 0.0)
            cursor_rel = prefill_start_rel
            if (has_runtime_prefill_timing or recompute_tokens > 0) and (recompute_ms > 0 or has_runtime_prefill_timing):
                if has_runtime_prefill_timing:
                    cursor_rel = runtime_attr_start_abs - due
                    recompute_end_rel = runtime_attr_end_abs - due
                    timing_source = "runtime_attributed_model_forward_batch"
                else:
                    recompute_end_rel = min(first_token_rel, cursor_rel + recompute_ms)
                    timing_source = "fallback_replay_counter_estimate"
                draw_overview_span(
                    parts,
                    cursor_rel,
                    recompute_end_rel,
                    replay_y + 18,
                    main_bar_h,
                    unified_stack_color("recompute"),
                    "prefill/recompute",
                    f"{label} | replay prefill/recompute attribution: {display_ms(recompute_end_rel - cursor_rel)} | tokens={compact_token_count(recompute_tokens)} | timing={timing_source}",
                    opacity=0.82,
                    break_long=True,
                    callout_fill="#fdf4ff",
                    callout_text="#86198f",
                )
                cursor_rel = recompute_end_rel
            replay_h2d_end_rel = replay_h2d[1] if replay_h2d else None
            remaining_start_rel = max(candidate for candidate in [cursor_rel, replay_h2d_end_rel] if candidate is not None)
            if first_token_rel > remaining_start_rel:
                draw_overview_span(
                    parts,
                    remaining_start_rel,
                    first_token_rel,
                    replay_y + 46,
                    main_bar_h,
                    unified_stack_color("prefill"),
                    "remaining before-token",
                    f"{label} | remaining before-first-token work: {display_ms(first_token_rel - remaining_start_rel)}",
                    opacity=0.84,
                    break_long=True,
                    callout_fill="#fefce8",
                    callout_text="#854d0e",
                )
            elif recompute_tokens <= 0 and not has_runtime_prefill_timing:
                draw_overview_span(
                    parts,
                    prefill_start_rel,
                    first_token_rel,
                    replay_y + 46,
                    main_bar_h,
                    unified_stack_color("prefill"),
                    "TTFT",
                    f"{label} | time to first token: {display_ms(first_token_rel - prefill_start_rel)}",
                    opacity=0.84,
                    break_long=True,
                    callout_fill="#fefce8",
                    callout_text="#854d0e",
                )
            draw_overview_marker(parts, first_token_rel, replay_y - 5, replay_y + 30, unified_stack_color("prefill"), f"{label} | first token: {display_ms(first_token_rel)} relative to due")
        if first_token is not None and as_float(row.get("resume_end_ms")) is not None:
            decode_span = (first_token - due, (as_float(row.get("resume_end_ms")) or first_token) - due)
            if decode_span[1] > decode_span[0]:
                draw_overview_span(parts, decode_span[0], decode_span[1], replay_y + 74, main_bar_h, unified_stack_color("decode"), "decode", f"{label} | decode after first token: {display_ms(decode_span[1] - decode_span[0])}", opacity=0.78)

        zoom_title_y = y + 252
        parts.append(f'<text x="{left - 10}" y="{zoom_title_y + 9:.1f}" text-anchor="end" font-size="10" font-weight="900" fill="#334155">KV zoom</text>')
        if zoom is None:
            parts.append(f'<text x="{left + 8}" y="{zoom_title_y + 9:.1f}" font-size="10" fill="#64748b">No SGLang-visible KV movement in this gap window.</text>')
        else:
            z_min, z_max = zoom
            z_span = max(1.0, z_max - z_min)
            zoom_label = f"expanded KV burst: {display_ms(z_min)} -> {display_ms(z_max)} relative to replay due"
            parts.append(f'<text x="{left + 8}" y="{zoom_title_y - 10:.1f}" font-size="10" font-weight="800" fill="#475569">KV zoom: expanded memory movement region</text>')
            parts.append(f'<text x="{left + 8}" y="{zoom_title_y + 6:.1f}" font-size="10" fill="#64748b">{html.escape(zoom_label)}</text>')
            for tick_value in [z_min, z_min + z_span * 0.25, z_min + z_span * 0.5, z_min + z_span * 0.75, z_max]:
                tx = zoom_x(tick_value, z_min, z_max)
                parts.append(f'<line x1="{tx:.1f}" y1="{zoom_title_y + 30:.1f}" x2="{tx:.1f}" y2="{zoom_title_y + 142:.1f}" stroke="#e5e7eb"/>')
                parts.append(f'<text x="{tx:.1f}" y="{zoom_title_y + 160:.1f}" text-anchor="middle" font-size="9" fill="#64748b">{html.escape(display_ms(tick_value))}</text>')
            if z_min <= 0 <= z_max:
                zx = zoom_x(0.0, z_min, z_max)
                parts.append(f'<line x1="{zx:.1f}" y1="{zoom_title_y + 30:.1f}" x2="{zx:.1f}" y2="{zoom_title_y + 142:.1f}" stroke="#111827" stroke-width="1.6"/>')
                parts.append(f'<text x="{zx + 4:.1f}" y="{zoom_title_y + 42:.1f}" font-size="9" font-weight="800">due</text>')
            zoom_lanes = {
                "H2D": (zoom_title_y + 42, unified_stack_color("h2d")),
                "D2H": (zoom_title_y + 72, unified_stack_color("d2h")),
                "GPU evict": (zoom_title_y + 102, unified_stack_color("evict")),
                "host evict": (zoom_title_y + 132, unified_stack_color("host_evict")),
            }
            hint_h2d_relative_span = _relative_span(row, "direct_kv_h2d_start_ms", "direct_kv_h2d_end_ms", due)
            drew_hint_h2d_in_kv_zoom = False
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
                hint_side_target_h2d = (
                    target
                    and kind == "H2D"
                    and hint_h2d_relative_span is not None
                    and span[0] <= hint_h2d_relative_span[1]
                    and span[1] >= hint_h2d_relative_span[0]
                )
                if hint_side_target_h2d:
                    color = unified_stack_color("hint_h2d")
                    drew_hint_h2d_in_kv_zoom = True
                title = kv_event_tooltip(label, event, span, target)
                event_label = ""
                if target and kind == "H2D":
                    event_label = short_bar_label(
                        [
                            "hint H2D" if hint_side_target_h2d else "H2D",
                            compact_tokens(event.get("token_or_index_count")),
                            display_ms(as_float(event.get("duration_ms")) or (span[1] - span[0])),
                        ],
                        max_chars=30,
                    )
                draw_span(
                    parts,
                    lambda value, z_min=z_min, z_max=z_max: zoom_x(value, z_min, z_max),
                    z_min,
                    z_max,
                    span[0],
                    span[1],
                    lane_y - (kv_target_bar_h if target else kv_other_bar_h) / 2 + 5,
                    kv_target_bar_h if target else kv_other_bar_h,
                    color,
                    event_label,
                    title,
                    opacity=0.94 if target else 0.48,
                    min_w=min_visible_event_w,
                    label_min_w=88.0,
                    font_size=9,
                )
            if hint_h2d_relative_span and not drew_hint_h2d_in_kv_zoom:
                hint_duration = hint_h2d_relative_span[1] - hint_h2d_relative_span[0]
                fallback_label = short_bar_label(
                    [
                        "hint H2D",
                        f"{row.get('direct_kv_h2d_events', '')} evt" if row.get("direct_kv_h2d_events", "") not in ("", None) else "",
                        display_ms(hint_duration),
                    ],
                    max_chars=30,
                )
                fallback_title = (
                    f"{label} | hint-side direct KV H2D from row-level summary | "
                    f"events={row.get('direct_kv_h2d_events', '')} | duration={display_ms(hint_duration)} | "
                    f"start={display_ms(hint_h2d_relative_span[0])} relative to due | "
                    f"end={display_ms(hint_h2d_relative_span[1])} relative to due"
                )
                draw_span(
                    parts,
                    lambda value, z_min=z_min, z_max=z_max: zoom_x(value, z_min, z_max),
                    z_min,
                    z_max,
                    hint_h2d_relative_span[0],
                    hint_h2d_relative_span[1],
                    zoom_lanes["H2D"][0] - kv_target_bar_h / 2 + 5,
                    kv_target_bar_h,
                    unified_stack_color("hint_h2d"),
                    fallback_label,
                    fallback_title,
                    opacity=0.96,
                    min_w=min_visible_event_w,
                    label_min_w=88.0,
                    font_size=9,
                )

        replay_zoom_title_y = y + 452
        parts.append(f'<text x="{left - 10}" y="{replay_zoom_title_y + 9:.1f}" text-anchor="end" font-size="10" font-weight="900" fill="#334155">replay zoom</text>')
        if replay_zoom is None:
            parts.append(f'<text x="{left + 8}" y="{replay_zoom_title_y + 9:.1f}" font-size="10" fill="#64748b">No replay timing was available for this gap.</text>')
        else:
            base_rz_min, base_rz_max = replay_zoom
            rz_min, rz_max = base_rz_min, base_rz_max
            rz_span = max(1.0, rz_max - rz_min)
            has_prefetch_window = False
            if hint_h2d_span is not None:
                # Use a broken axis only when the prefetch copy really happened
                # before the replay region. If the two windows overlap, drawing
                # them side-by-side makes the labels look like time moves
                # backwards, so we keep one chronological replay window.
                prefetch_gap_before_replay = base_rz_min - hint_h2d_span[1]
                has_prefetch_window = prefetch_gap_before_replay > 50.0
                if not has_prefetch_window:
                    hint_duration = max(1.0, hint_h2d_span[1] - hint_h2d_span[0])
                    hint_pad = max(20.0, hint_duration * 0.10)
                    rz_min = min(base_rz_min, hint_h2d_span[0] - hint_pad)
                    rz_max = max(base_rz_max, hint_h2d_span[1] + hint_pad)
                    rz_span = max(1.0, rz_max - rz_min)
            if has_prefetch_window:
                prefetch_window_start = hint_h2d_span[0]
                prefetch_window_end = hint_h2d_span[1]
                prefetch_window_span = max(1.0, prefetch_window_end - prefetch_window_start)
                hz_min = prefetch_window_start - max(20.0, prefetch_window_span * 0.10)
                hz_max = prefetch_window_end + max(20.0, prefetch_window_span * 0.10)
                left_window_w = plot_w * 0.27
                break_w = 74.0
                right_window_x = left + left_window_w + break_w
                right_window_w = plot_w - left_window_w - break_w
                break_x = left + left_window_w + break_w / 2
                skipped_start = hz_max
                skipped_end = rz_min
                skipped_text = ""
                if skipped_end > skipped_start:
                    skipped_text = f"skipped {display_ms(skipped_end - skipped_start)}"
                else:
                    skipped_text = "no skipped gap"

                def prefetch_replay_x(value: float, z_min: float = hz_min, z_max: float = hz_max) -> float:
                    return local_zoom_x(value, z_min, z_max, left, left_window_w)

                def replay_x(value: float, z_min: float = rz_min, z_max: float = rz_max) -> float:
                    return local_zoom_x(value, z_min, z_max, right_window_x, right_window_w)

                replay_zoom_label = (
                    f"two-window replay region: prefetch {display_ms(hz_min)} -> {display_ms(hz_max)}, "
                    f"then replay {display_ms(rz_min)} -> {display_ms(rz_max)} relative to replay due"
                )
                parts.append(f'<text x="{left + 8}" y="{replay_zoom_title_y - 10:.1f}" font-size="10" font-weight="800" fill="#475569">Replay zoom: broken-axis prefetch + replay execution region</text>')
                parts.append(f'<text x="{left + 8}" y="{replay_zoom_title_y + 6:.1f}" font-size="10" fill="#64748b">{html.escape(replay_zoom_label)}</text>')
                parts.append(f'<text x="{left + left_window_w / 2:.1f}" y="{replay_zoom_title_y + 26:.1f}" text-anchor="middle" font-size="9" font-weight="900" fill="#166534">prefetch window</text>')
                parts.append(f'<text x="{right_window_x + right_window_w / 2:.1f}" y="{replay_zoom_title_y + 26:.1f}" text-anchor="middle" font-size="9" font-weight="900" fill="#334155">replay window</text>')
                draw_axis_break(parts, break_x, replay_zoom_title_y + 36, replay_zoom_title_y + 276, skipped_text)
                for tick_value in [hz_min, (hz_min + hz_max) / 2, hz_max]:
                    tx = prefetch_replay_x(tick_value)
                    parts.append(f'<line x1="{tx:.1f}" y1="{replay_zoom_title_y + 32:.1f}" x2="{tx:.1f}" y2="{replay_zoom_title_y + 260:.1f}" stroke="#e5e7eb"/>')
                    parts.append(f'<text x="{tx:.1f}" y="{replay_zoom_title_y + 280:.1f}" text-anchor="middle" font-size="9" fill="#64748b">{html.escape(display_ms(tick_value))}</text>')
                for tick_value in [rz_min, rz_min + rz_span * 0.25, rz_min + rz_span * 0.5, rz_min + rz_span * 0.75, rz_max]:
                    tx = replay_x(tick_value)
                    parts.append(f'<line x1="{tx:.1f}" y1="{replay_zoom_title_y + 32:.1f}" x2="{tx:.1f}" y2="{replay_zoom_title_y + 260:.1f}" stroke="#e5e7eb"/>')
                    parts.append(f'<text x="{tx:.1f}" y="{replay_zoom_title_y + 280:.1f}" text-anchor="middle" font-size="9" fill="#64748b">{html.escape(display_ms(tick_value))}</text>')
                if rz_min <= 0 <= rz_max:
                    zx = replay_x(0.0)
                    parts.append(f'<line x1="{zx:.1f}" y1="{replay_zoom_title_y + 32:.1f}" x2="{zx:.1f}" y2="{replay_zoom_title_y + 260:.1f}" stroke="#111827" stroke-width="1.6"/>')
                    parts.append(f'<text x="{zx + 4:.1f}" y="{replay_zoom_title_y + 44:.1f}" font-size="9" font-weight="800">due</text>')
            else:
                right_window_x = left
                right_window_w = plot_w

                def replay_x(value: float, z_min: float = rz_min, z_max: float = rz_max) -> float:
                    return local_zoom_x(value, z_min, z_max, right_window_x, right_window_w)

                prefetch_replay_x = replay_x
                hz_min = rz_min
                hz_max = rz_max
                replay_zoom_label = f"expanded replay region: {display_ms(rz_min)} -> {display_ms(rz_max)} relative to replay due"
                replay_zoom_title = "Replay zoom: expanded replay execution region"
                if hint_h2d_span is not None:
                    replay_zoom_title = "Replay zoom: chronological prefetch + replay execution region"
                    replay_zoom_label = (
                        f"single-window replay region: prefetch/replay overlap or arrive close together; "
                        f"shown chronologically from {display_ms(rz_min)} -> {display_ms(rz_max)} relative to replay due"
                    )
                parts.append(f'<text x="{left + 8}" y="{replay_zoom_title_y - 10:.1f}" font-size="10" font-weight="800" fill="#475569">{html.escape(replay_zoom_title)}</text>')
                parts.append(f'<text x="{left + 8}" y="{replay_zoom_title_y + 6:.1f}" font-size="10" fill="#64748b">{html.escape(replay_zoom_label)}</text>')
                for tick_value in [rz_min, rz_min + rz_span * 0.25, rz_min + rz_span * 0.5, rz_min + rz_span * 0.75, rz_max]:
                    tx = replay_x(tick_value)
                    parts.append(f'<line x1="{tx:.1f}" y1="{replay_zoom_title_y + 32:.1f}" x2="{tx:.1f}" y2="{replay_zoom_title_y + 260:.1f}" stroke="#e5e7eb"/>')
                    parts.append(f'<text x="{tx:.1f}" y="{replay_zoom_title_y + 280:.1f}" text-anchor="middle" font-size="9" fill="#64748b">{html.escape(display_ms(tick_value))}</text>')
                if rz_min <= 0 <= rz_max:
                    zx = replay_x(0.0)
                    parts.append(f'<line x1="{zx:.1f}" y1="{replay_zoom_title_y + 32:.1f}" x2="{zx:.1f}" y2="{replay_zoom_title_y + 260:.1f}" stroke="#111827" stroke-width="1.6"/>')
                    parts.append(f'<text x="{zx + 4:.1f}" y="{replay_zoom_title_y + 44:.1f}" font-size="9" font-weight="800">due</text>')

            replay_zoom_lanes = [
                ("replay request", replay_zoom_title_y + 52),
                ("prefetch KV H2D", replay_zoom_title_y + 90),
                ("replay KV H2D", replay_zoom_title_y + 128),
                ("prefill/recompute", replay_zoom_title_y + 166),
                ("remaining before-token", replay_zoom_title_y + 204),
                ("decode", replay_zoom_title_y + 242),
            ]
            for lane_name, lane_y in replay_zoom_lanes:
                parts.append(f'<text x="{left - 10}" y="{lane_y + 8:.1f}" text-anchor="end" font-size="9" font-weight="800" fill="#334155">{html.escape(lane_name)}</text>')
                parts.append(f'<line x1="{left}" y1="{lane_y + 5:.1f}" x2="{left + plot_w}" y2="{lane_y + 5:.1f}" stroke="#dbe4ee"/>')

            if hint_h2d_span:
                hint_duration = hint_h2d_span[1] - hint_h2d_span[0]
                hint_label = short_bar_label(
                    [
                        "prefetch KV H2D",
                        f"{row.get('direct_kv_h2d_events', '')} evt" if row.get("direct_kv_h2d_events", "") not in ("", None) else "",
                        display_ms(hint_duration),
                    ],
                    max_chars=42,
                )
                hint_title = (
                    f"{label} | prefetch-side KV H2D shown in the replay zoom's left window | "
                    f"events={row.get('direct_kv_h2d_events', '')} | duration={display_ms(hint_duration)} | "
                    f"start={display_ms(hint_h2d_span[0])} relative to due | end={display_ms(hint_h2d_span[1])} relative to due"
                )
                draw_span(
                    parts,
                    prefetch_replay_x,
                    hz_min,
                    hz_max,
                    hint_h2d_span[0],
                    hint_h2d_span[1],
                    replay_zoom_title_y + 79,
                    main_bar_h,
                    unified_stack_color("hint_h2d"),
                    hint_label,
                    hint_title,
                    opacity=0.92,
                    min_w=min_visible_bar_w,
                    label_min_w=82.0,
                    font_size=9,
                )
                hint_x1 = prefetch_replay_x(max(hz_min, min(hz_max, hint_h2d_span[0])))
                hint_x2 = prefetch_replay_x(max(hz_min, min(hz_max, hint_h2d_span[1])))
                if max(min_visible_bar_w, hint_x2 - hint_x1) < 82.0:
                    draw_small_bar_callout(
                        parts,
                        hint_x1,
                        hint_x2,
                        replay_zoom_title_y + 79,
                        main_bar_h,
                        hint_label,
                        hint_title,
                        unified_stack_color("hint_h2d"),
                        fill_color="#f0fdf4",
                        text_color="#166534",
                    )

            replay_request_span = _relative_span(row, "resume_start_ms", "resume_end_ms", due)
            if replay_request_span:
                replay_request_duration = replay_request_span[1] - replay_request_span[0]
                draw_span(
                    parts,
                    replay_x,
                    rz_min,
                    rz_max,
                    replay_request_span[0],
                    replay_request_span[1],
                    replay_zoom_title_y + 41,
                    main_bar_h,
                    unified_stack_color("decode"),
                    short_bar_label(["replay request", display_ms(replay_request_duration)], max_chars=38),
                    f"{label} | replay request wall time: {display_ms(replay_request_duration)} | start={display_ms(replay_request_span[0])} relative to due | end={display_ms(replay_request_span[1])} relative to due",
                    opacity=0.58,
                    min_w=min_visible_bar_w,
                    label_min_w=118.0,
                )

            replay_h2d_span = _relative_span(row, "replay_kv_h2d_start_ms", "replay_kv_h2d_end_ms", due)
            if replay_h2d_span:
                replay_h2d_duration = replay_h2d_span[1] - replay_h2d_span[0]
                replay_h2d_label = short_bar_label(
                    [
                        f"replay KV H2D: {replay_h2d_summary['blocks']} blk" if replay_h2d_summary["events"] else "replay KV H2D",
                        compact_tokens(replay_h2d_summary["tokens"]) if replay_h2d_summary["tokens"] else "",
                        display_ms(replay_h2d_duration),
                    ],
                    max_chars=42,
                )
                replay_h2d_title = (
                    f"{label} | replay-side KV H2D | events={replay_h2d_summary['events']} | "
                    f"blocks={replay_h2d_summary['blocks']} | indices={compact_tokens(replay_h2d_summary['tokens'])} | "
                    f"duration={display_ms(replay_h2d_duration)} | start={display_ms(replay_h2d_span[0])} relative to due | "
                    f"end={display_ms(replay_h2d_span[1])} relative to due"
                )
                draw_span(
                    parts,
                    replay_x,
                    rz_min,
                    rz_max,
                    replay_h2d_span[0],
                    replay_h2d_span[1],
                    replay_zoom_title_y + 117,
                    main_bar_h,
                    unified_stack_color("h2d"),
                    replay_h2d_label,
                    replay_h2d_title,
                    opacity=0.92,
                    min_w=min_visible_bar_w,
                    label_min_w=82.0,
                    font_size=9,
                )
                clipped_h2d_start = max(rz_min, min(rz_max, replay_h2d_span[0]))
                clipped_h2d_end = max(rz_min, min(rz_max, replay_h2d_span[1]))
                if clipped_h2d_end > clipped_h2d_start:
                    h2d_x1 = replay_x(clipped_h2d_start)
                    h2d_x2 = replay_x(clipped_h2d_end)
                    if max(6.0, h2d_x2 - h2d_x1) < 82.0:
                        draw_small_bar_callout(
                            parts,
                            h2d_x1,
                            h2d_x2,
                            replay_zoom_title_y + 117,
                            main_bar_h,
                            replay_h2d_label,
                            replay_h2d_title,
                            unified_stack_color("h2d"),
                        )

            replay_start_abs = as_float(row.get("resume_start_ms"))
            first_token_abs = first_token_ms(row)
            if replay_start_abs is not None and first_token_abs is not None:
                pre_token_start_abs = as_float(row.get("replay_prefill_start_ms")) or replay_start_abs
                pre_token_start = pre_token_start_abs - due
                first_token_rel = first_token_abs - due
                runtime_attr_tokens = as_float(row.get("replay_runtime_prefill_attributed_tokens"))
                runtime_attr_start_abs = as_float(row.get("replay_runtime_prefill_attribution_start_ms"))
                runtime_attr_end_abs = as_float(row.get("replay_runtime_prefill_attribution_end_ms"))
                runtime_attr_confidence = str(row.get("replay_runtime_prefill_confidence") or "")
                has_runtime_prefill_timing = (
                    runtime_attr_confidence == "runtime_attributed"
                    and runtime_attr_start_abs is not None
                    and runtime_attr_end_abs is not None
                    and runtime_attr_end_abs > runtime_attr_start_abs
                )
                recompute_tokens = runtime_attr_tokens or as_float(row.get("recomputed_tokens_est")) or as_float(row.get("replay_new_prefill_tokens_est")) or 0.0
                cached_prefix = compact_token_count(row.get("replay_cached_prefix_tokens"))
                input_tokens = compact_token_count(row.get("replay_input_tokens"))
                timing_source = str(row.get("replay_prefill_recompute_timing_source") or "fallback_request_ttft_window")
                segments = replay_phase_segments(row)
                recompute_ms = segments.get("recompute", 0.0)
                normal_prefill_ms = segments.get("normal_prefill", 0.0)
                cursor_rel = pre_token_start
                if (has_runtime_prefill_timing or recompute_tokens > 0) and (recompute_ms > 0 or has_runtime_prefill_timing):
                    if has_runtime_prefill_timing:
                        cursor_rel = runtime_attr_start_abs - due
                        recompute_end_rel = runtime_attr_end_abs - due
                        timing_source = "runtime_attributed_model_forward_batch"
                    else:
                        recompute_end_rel = min(first_token_rel, cursor_rel + recompute_ms)
                    recompute_label = short_bar_label(
                        [
                            "prefill/recompute" if runtime_attr_confidence == "runtime_attributed" else "prefill/recompute est.",
                            compact_token_count(recompute_tokens),
                            display_ms(recompute_end_rel - cursor_rel),
                        ],
                        max_chars=44,
                    )
                    token_range = str(row.get("replay_runtime_prefill_token_range") or "")
                    batch_id = str(row.get("replay_runtime_prefill_batch_id") or "")
                    batch_requests = str(row.get("replay_runtime_prefill_batch_request_count") or "")
                    recompute_title = (
                        f"{label} | replay prefill/recompute attribution | timing_source={timing_source} | "
                        f"duration={display_ms(recompute_end_rel - cursor_rel)} | rebuilt_or_new_prefill={compact_token_count(recompute_tokens)} | "
                        f"input={input_tokens} | cached_prefix={cached_prefix} | token_range={token_range or 'unknown'} | "
                        f"batch={batch_id or 'unknown'} | batch_requests={batch_requests or 'unknown'} | "
                        f"note={'request-attributed SGLang model-forward evidence' if runtime_attr_confidence == 'runtime_attributed' else 'fallback estimate derived from replay counters and model-forward timing'}"
                    )
                    draw_span(
                        parts,
                        replay_x,
                        rz_min,
                        rz_max,
                        cursor_rel,
                        recompute_end_rel,
                        replay_zoom_title_y + 155,
                        main_bar_h,
                        unified_stack_color("recompute"),
                        recompute_label,
                        recompute_title,
                        opacity=0.88,
                        min_w=min_visible_bar_w,
                        label_min_w=82.0,
                        font_size=9,
                    )
                    cursor_rel = max(cursor_rel, recompute_end_rel)
                else:
                    missing_title = f"{label} | no visible prefill/recompute attribution segment"
                    draw_span(
                        parts,
                        replay_x,
                        rz_min,
                        rz_max,
                        pre_token_start,
                        min(first_token_rel, pre_token_start + 1.0),
                        replay_zoom_title_y + 155,
                        main_bar_h,
                        "#f8fafc",
                        "",
                        missing_title,
                        opacity=0.40,
                        min_w=min_visible_event_w,
                        label_min_w=9999.0,
                    )
                replay_h2d_end_rel = None
                if replay_h2d_span:
                    replay_h2d_end_rel = replay_h2d_span[1]
                remaining_start_candidates = [cursor_rel]
                if replay_h2d_end_rel is not None:
                    remaining_start_candidates.append(replay_h2d_end_rel)
                remaining_start_rel = max(remaining_start_candidates)
                if first_token_rel > remaining_start_rel:
                    prefill_end_rel = first_token_rel
                    prefill_label = short_bar_label(
                        ["remaining before-token", display_ms(prefill_end_rel - remaining_start_rel)],
                        max_chars=42,
                    )
                    prefill_title = (
                        f"{label} | remaining before-first-token replay work | duration={display_ms(prefill_end_rel - remaining_start_rel)} | "
                        f"start={display_ms(remaining_start_rel)} relative to due | end={display_ms(prefill_end_rel)} relative to due | "
                        f"note=leftover TTFT after visible H2D and prefill/recompute attribution are separated"
                    )
                    draw_span(
                        parts,
                        replay_x,
                        rz_min,
                        rz_max,
                        remaining_start_rel,
                        prefill_end_rel,
                        replay_zoom_title_y + 193,
                        main_bar_h,
                        unified_stack_color("prefill"),
                        prefill_label,
                        prefill_title,
                        opacity=0.88,
                        min_w=min_visible_bar_w,
                        label_min_w=82.0,
                        font_size=9,
                    )
                    clipped_prefill_start = max(rz_min, min(rz_max, remaining_start_rel))
                    clipped_prefill_end = max(rz_min, min(rz_max, prefill_end_rel))
                    if clipped_prefill_end > clipped_prefill_start:
                        prefill_x1 = replay_x(clipped_prefill_start)
                        prefill_x2 = replay_x(clipped_prefill_end)
                        if max(min_visible_bar_w, prefill_x2 - prefill_x1) < 82.0:
                            draw_small_bar_callout(
                                parts,
                                prefill_x1,
                                prefill_x2,
                                replay_zoom_title_y + 193,
                                main_bar_h,
                                prefill_label,
                                prefill_title,
                                unified_stack_color("prefill"),
                                fill_color="#fef3c7",
                                text_color="#78350f",
                            )
                else:
                    draw_span(
                        parts,
                        replay_x,
                        rz_min,
                        rz_max,
                        remaining_start_rel,
                        min(first_token_rel, remaining_start_rel + 1.0),
                        replay_zoom_title_y + 193,
                        main_bar_h,
                        "#f8fafc",
                        "",
                        f"{label} | no remaining before-first-token TTFT segment after H2D/recompute split",
                        opacity=0.40,
                        min_w=min_visible_event_w,
                        label_min_w=9999.0,
                    )
                first_token_x = replay_x(first_token_rel)
                parts.append(f'<line x1="{first_token_x:.1f}" y1="{replay_zoom_title_y + 150:.1f}" x2="{first_token_x:.1f}" y2="{replay_zoom_title_y + 260:.1f}" stroke="#eab308" stroke-width="1.8"><title>{html.escape(label)} | first token: {html.escape(display_ms(first_token_rel))} relative to due</title></line>')
                parts.append(f'<text x="{first_token_x + 5:.1f}" y="{replay_zoom_title_y + 154:.1f}" font-size="8" font-weight="800" fill="#92400e">first token</text>')

            if first_token_abs is not None and as_float(row.get("resume_end_ms")) is not None:
                decode_start = first_token_abs - due
                decode_end = (as_float(row.get("resume_end_ms")) or first_token_abs) - due
                if decode_end > decode_start:
                    decode_duration = decode_end - decode_start
                    draw_span(
                        parts,
                        replay_x,
                        rz_min,
                        rz_max,
                        decode_start,
                        decode_end,
                        replay_zoom_title_y + 231,
                        main_bar_h,
                        unified_stack_color("decode"),
                        short_bar_label(["decode", display_ms(decode_duration)], max_chars=36),
                        f"{label} | decode after first token | duration={display_ms(decode_duration)} | start={display_ms(decode_start)} relative to due | end={display_ms(decode_end)} relative to due",
                        opacity=0.86,
                        min_w=min_visible_bar_w,
                        label_min_w=82.0,
                        font_size=9,
                    )

        pool_zoom_title_y = y + 818
        parts.append(f'<text x="{left - 10}" y="{pool_zoom_title_y + 9:.1f}" text-anchor="end" font-size="10" font-weight="900" fill="#334155">GPU pool zoom</text>')
        pool_bins: list[dict[str, Any]] = []
        if kv_pool_row and kv_pool_row.get("pool_timeline_bins_json"):
            try:
                parsed_pool_bins = json.loads(str(kv_pool_row.get("pool_timeline_bins_json") or "[]"))
                if isinstance(parsed_pool_bins, list):
                    pool_bins = [item for item in parsed_pool_bins if isinstance(item, dict)]
            except json.JSONDecodeError:
                pool_bins = []
        pool_bins_with_samples = [
            item for item in pool_bins if as_float(item.get("max_usage_pct")) is not None and as_float(item.get("samples")) not in (None, 0.0)
        ]
        if not pool_bins_with_samples:
            parts.append(
                f'<text x="{left + 8}" y="{pool_zoom_title_y - 10:.1f}" font-size="10" font-weight="800" fill="#475569">'
                f'GPU KV-pool residency zoom</text>'
            )
            parts.append(
                f'<text x="{left + 8}" y="{pool_zoom_title_y + 8:.1f}" font-size="10" fill="#64748b">'
                f'No direct SGLang KV-pool samples were available for this row. Rerun with AGENTIC_KV_TRACE_KV_POOL=1.</text>'
            )
            parts.append(f'<line x1="{left}" y1="{pool_zoom_title_y + 72:.1f}" x2="{left + plot_w}" y2="{pool_zoom_title_y + 72:.1f}" stroke="#dbe4ee"/>')
        else:
            pool_min = min(as_float(item.get("start_rel_ms")) or 0.0 for item in pool_bins)
            pool_max = max(as_float(item.get("end_rel_ms")) or 0.0 for item in pool_bins)
            pool_span = max(1.0, pool_max - pool_min)
            pool_max_usage = max(as_float(item.get("max_usage_pct")) or 0.0 for item in pool_bins_with_samples)
            pool_avg_values = [as_float(item.get("avg_usage_pct")) for item in pool_bins_with_samples]
            pool_avg_usage = mean([value for value in pool_avg_values if value is not None]) if pool_avg_values else 0.0
            pool_verdict = str(kv_pool_row.get("kv_pool_verdict") or kv_pool_pressure_label(pool_max_usage))

            def pool_x(value: float, z_min: float = pool_min, z_max: float = pool_max) -> float:
                return local_zoom_x(value, z_min, z_max, left, plot_w)

            chart_top = pool_zoom_title_y + 42
            chart_h = 72.0
            chart_bottom = chart_top + chart_h
            parts.append(
                f'<text x="{left + 8}" y="{pool_zoom_title_y - 10:.1f}" font-size="10" font-weight="800" fill="#475569">'
                f'GPU KV-pool residency zoom</text>'
            )
            parts.append(
                f'<text x="{left + 8}" y="{pool_zoom_title_y + 8:.1f}" font-size="10" fill="#64748b">'
                f'SGLang KV-pool occupancy while this gap was active. max {pool_max_usage:.1f}%, avg {pool_avg_usage:.1f}%, pressure {html.escape(pool_verdict)}.</text>'
            )
            for pct, dash, color, threshold_label in [
                (65.0, "3 4", "#ca8a04", "65%"),
                (85.0, "4 4", "#ea580c", "85%"),
                (95.0, "5 4", "#dc2626", "95% near full"),
            ]:
                py = chart_bottom - chart_h * pct / 100.0
                parts.append(
                    f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" '
                    f'stroke="{color}" stroke-width="1" stroke-dasharray="{dash}" opacity="0.45"/>'
                )
                parts.append(
                    f'<text x="{left - 8}" y="{py + 3:.1f}" text-anchor="end" font-size="8" '
                    f'font-weight="800" fill="{color}">{threshold_label}</text>'
                )
            parts.append(f'<line x1="{left}" y1="{chart_bottom:.1f}" x2="{left + plot_w}" y2="{chart_bottom:.1f}" stroke="#dbe4ee"/>')
            for tick_value in [pool_min, pool_min + pool_span * 0.25, pool_min + pool_span * 0.5, pool_min + pool_span * 0.75, pool_max]:
                tx = pool_x(tick_value)
                parts.append(f'<line x1="{tx:.1f}" y1="{chart_top:.1f}" x2="{tx:.1f}" y2="{chart_bottom + 9:.1f}" stroke="#e5e7eb"/>')
                parts.append(f'<text x="{tx:.1f}" y="{chart_bottom + 25:.1f}" text-anchor="middle" font-size="8" fill="#64748b">{html.escape(display_ms(tick_value))}</text>')
            if pool_min <= 0 <= pool_max:
                zx = pool_x(0.0)
                parts.append(f'<line x1="{zx:.1f}" y1="{chart_top:.1f}" x2="{zx:.1f}" y2="{chart_bottom + 9:.1f}" stroke="#111827" stroke-width="1.4"/>')
                parts.append(f'<text x="{zx + 4:.1f}" y="{chart_top + 11:.1f}" font-size="8" font-weight="900" fill="#111827">due</text>')
            peak_labeled = False
            for item in pool_bins:
                usage = as_float(item.get("max_usage_pct"))
                samples = as_float(item.get("samples")) or 0.0
                start_rel = as_float(item.get("start_rel_ms"))
                end_rel = as_float(item.get("end_rel_ms"))
                if usage is None or samples <= 0 or start_rel is None or end_rel is None or end_rel <= start_rel:
                    continue
                x1 = pool_x(start_rel)
                x2 = pool_x(end_rel)
                w = max(2.0, x2 - x1 - 1.0)
                bar_h = max(3.0, chart_h * min(100.0, max(0.0, usage)) / 100.0)
                bar_y = chart_bottom - bar_h
                color = kv_pool_heat_color(usage)
                title = (
                    f"{label} | GPU KV-pool bin | time={display_ms(start_rel)} -> {display_ms(end_rel)} relative to due | "
                    f"max_usage={usage:.3f}% | avg_usage={item.get('avg_usage_pct', '')}% | samples={int(samples)} | pressure={item.get('pressure', '')}"
                )
                parts.append(
                    f'<rect x="{x1:.1f}" y="{bar_y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="2" '
                    f'fill="{color}" opacity="0.82"><title>{html.escape(title)}</title></rect>'
                )
                label_text = f"{usage:.0f}%"
                is_peak = abs(usage - pool_max_usage) < 0.05 and not peak_labeled
                should_label = w >= 18.0 and (usage >= 65.0 or is_peak or len(pool_bins_with_samples) <= 24)
                if should_label:
                    text_color = "#ffffff" if usage >= 85.0 else "#111827"
                    if bar_h >= 16.0:
                        parts.append(
                            f'<text x="{x1 + w / 2:.1f}" y="{bar_y + min(bar_h - 4.0, 13.0):.1f}" '
                            f'text-anchor="middle" font-size="7" font-weight="900" fill="{text_color}">{label_text}</text>'
                        )
                    else:
                        parts.append(
                            f'<text x="{x1 + w / 2:.1f}" y="{bar_y - 3.0:.1f}" '
                            f'text-anchor="middle" font-size="7" font-weight="900" fill="#334155">{label_text}</text>'
                        )
                if is_peak:
                    peak_labeled = True
                    peak_label_y = max(chart_top + 8.0, bar_y - 12.0)
                    parts.append(
                        f'<text x="{x1 + w / 2:.1f}" y="{peak_label_y:.1f}" text-anchor="middle" '
                        f'font-size="8" font-weight="900" fill="#991b1b">peak {label_text}</text>'
                    )

        deadline_zoom_title_y = y + 988
        parts.append(f'<text x="{left - 10}" y="{deadline_zoom_title_y + 9:.1f}" text-anchor="end" font-size="10" font-weight="900" fill="#334155">deadline zoom</text>')
        hint_deadline_span = _relative_span(row, "direct_kv_h2d_start_ms", "direct_kv_h2d_end_ms", due)
        replay_deadline_span = _relative_span(row, "replay_kv_h2d_start_ms", "replay_kv_h2d_end_ms", due)
        projected_realistic = projected_by_label.get(label, {}).get("realistic")
        projected_span = None
        if projected_realistic:
            projected_start_abs = as_float(projected_realistic.get("projected_hardware_start_ms"))
            projected_end_abs = as_float(projected_realistic.get("projected_hardware_end_ms"))
            if projected_start_abs is not None and projected_end_abs is not None and projected_end_abs > projected_start_abs:
                projected_span = (projected_start_abs - due, projected_end_abs - due)
        useful_span = hint_deadline_span or replay_deadline_span
        useful_kind = "prefetch" if hint_deadline_span else "replay"
        if useful_span is None:
            parts.append(
                f'<text x="{left + 8}" y="{deadline_zoom_title_y - 10:.1f}" font-size="10" font-weight="800" fill="#475569">'
                f'KV readiness deadline zoom</text>'
            )
            parts.append(
                f'<text x="{left + 8}" y="{deadline_zoom_title_y + 8:.1f}" font-size="10" fill="#64748b">'
                f'No useful host-to-device KV movement was observed for this row.</text>'
            )
            parts.append(f'<line x1="{left}" y1="{deadline_zoom_title_y + 50:.1f}" x2="{left + plot_w}" y2="{deadline_zoom_title_y + 50:.1f}" stroke="#dbe4ee"/>')
            parts.append(
                f'<text x="{left + plot_w / 2:.1f}" y="{deadline_zoom_title_y + 56:.1f}" text-anchor="middle" '
                f'font-size="11" font-weight="900" fill="#64748b">no visible KV H2D before first-token path</text>'
            )
        else:
            ready_ms = useful_span[1]
            start_ms = useful_span[0]
            useful_duration = max(0.0, useful_span[1] - useful_span[0])
            if ready_ms <= 0:
                verdict_text = f"ready {display_ms(abs(ready_ms))} before due"
                verdict_color = "#15803d"
                gap_color = "#16a34a"
                gap_start, gap_end = ready_ms, 0.0
            else:
                verdict_text = f"late by {display_ms(ready_ms)}"
                verdict_color = "#dc2626"
                gap_color = "#dc2626"
                gap_start, gap_end = 0.0, ready_ms
            dz_values = [0.0, start_ms, ready_ms]
            if projected_span:
                dz_values.extend(projected_span)
            if first_token is not None:
                dz_values.append(first_token - due)
            dz_min = min(dz_values)
            dz_max = max(dz_values)
            dz_span = max(1.0, dz_max - dz_min)
            dz_pad = max(20.0, dz_span * 0.08)
            dz_min -= dz_pad
            dz_max += dz_pad

            def deadline_x(value: float, z_min: float = dz_min, z_max: float = dz_max) -> float:
                return local_zoom_x(value, z_min, z_max, left, plot_w)

            h2d_label_prefix = "prefetch KV H2D" if useful_kind == "prefetch" else "replay KV H2D"
            h2d_color = unified_stack_color("hint_h2d") if useful_kind == "prefetch" else unified_stack_color("h2d")
            event_count = row.get("direct_kv_h2d_events") if useful_kind == "prefetch" else replay_h2d_summary["events"]
            block_count = "" if useful_kind == "prefetch" else f"{replay_h2d_summary['blocks']} blk"
            token_count = "" if useful_kind == "prefetch" else compact_tokens(replay_h2d_summary["tokens"])
            h2d_label = short_bar_label(
                [
                    h2d_label_prefix,
                    f"{event_count} evt" if useful_kind == "prefetch" and event_count not in ("", None) else block_count,
                    token_count,
                    display_ms(useful_duration),
                ],
                max_chars=46,
            )
            h2d_title = (
                f"{label} | deadline zoom useful KV movement | kind={useful_kind} | "
                f"start={display_ms(start_ms)} relative to due | end={display_ms(ready_ms)} relative to due | "
                f"duration={display_ms(useful_duration)} | verdict={verdict_text}"
            )
            parts.append(
                f'<text x="{left + 8}" y="{deadline_zoom_title_y - 10:.1f}" font-size="10" font-weight="800" fill="#475569">'
                f'KV readiness deadline zoom</text>'
            )
            parts.append(
                f'<text x="{left + 8}" y="{deadline_zoom_title_y + 8:.1f}" font-size="10" fill="#64748b">'
                f'Shows the gap from replay due to useful KV readiness. {html.escape(verdict_text)}.</text>'
            )
            for tick_value in [dz_min, dz_min + dz_span * 0.25, dz_min + dz_span * 0.5, dz_min + dz_span * 0.75, dz_max]:
                tx = deadline_x(tick_value)
                parts.append(f'<line x1="{tx:.1f}" y1="{deadline_zoom_title_y + 30:.1f}" x2="{tx:.1f}" y2="{deadline_zoom_title_y + 122:.1f}" stroke="#e5e7eb"/>')
                parts.append(f'<text x="{tx:.1f}" y="{deadline_zoom_title_y + 140:.1f}" text-anchor="middle" font-size="9" fill="#64748b">{html.escape(display_ms(tick_value))}</text>')
            if dz_min <= 0 <= dz_max:
                zx = deadline_x(0.0)
                parts.append(f'<line x1="{zx:.1f}" y1="{deadline_zoom_title_y + 30:.1f}" x2="{zx:.1f}" y2="{deadline_zoom_title_y + 122:.1f}" stroke="#111827" stroke-width="1.8"/>')
                parts.append(f'<text x="{zx + 5:.1f}" y="{deadline_zoom_title_y + 43:.1f}" font-size="9" font-weight="900" fill="#111827">due</text>')
            deadline_lanes = [
                ("readiness gap", deadline_zoom_title_y + 55),
                ("useful KV H2D", deadline_zoom_title_y + 94),
            ]
            if projected_span:
                deadline_lanes.append(("projected HW", deadline_zoom_title_y + 126))
            for lane_name, lane_y in deadline_lanes:
                parts.append(f'<text x="{left - 10}" y="{lane_y + 8:.1f}" text-anchor="end" font-size="9" font-weight="800" fill="#334155">{html.escape(lane_name)}</text>')
                parts.append(f'<line x1="{left}" y1="{lane_y + 5:.1f}" x2="{left + plot_w}" y2="{lane_y + 5:.1f}" stroke="#dbe4ee"/>')
            gap_x1 = deadline_x(max(dz_min, min(dz_max, gap_start)))
            gap_x2 = deadline_x(max(dz_min, min(dz_max, gap_end)))
            if abs(gap_x2 - gap_x1) >= 2.0:
                parts.append(
                    f'<line x1="{gap_x1:.1f}" y1="{deadline_zoom_title_y + 60:.1f}" x2="{gap_x2:.1f}" y2="{deadline_zoom_title_y + 60:.1f}" '
                    f'stroke="{gap_color}" stroke-width="2.2" stroke-dasharray="6 5"><title>{html.escape(label)} | {html.escape(verdict_text)}</title></line>'
                )
                gap_label_x = min(max((gap_x1 + gap_x2) / 2, left + 70), left + plot_w - 70)
                parts.append(
                    f'<text x="{gap_label_x:.1f}" y="{deadline_zoom_title_y + 52:.1f}" text-anchor="middle" font-size="9" '
                    f'font-weight="900" fill="{verdict_color}">{html.escape(verdict_text)}</text>'
                )
            draw_span(
                parts,
                deadline_x,
                dz_min,
                dz_max,
                useful_span[0],
                useful_span[1],
                deadline_zoom_title_y + 83,
                main_bar_h,
                h2d_color,
                h2d_label,
                h2d_title,
                opacity=0.94,
                min_w=min_visible_bar_w,
                label_min_w=96.0,
                font_size=9,
            )
            clipped_deadline_start = max(dz_min, min(dz_max, useful_span[0]))
            clipped_deadline_end = max(dz_min, min(dz_max, useful_span[1]))
            if clipped_deadline_end > clipped_deadline_start:
                h2d_x1 = deadline_x(clipped_deadline_start)
                h2d_x2 = deadline_x(clipped_deadline_end)
                if max(min_visible_bar_w, h2d_x2 - h2d_x1) < 96.0:
                    draw_small_bar_callout(
                        parts,
                        h2d_x1,
                        h2d_x2,
                        deadline_zoom_title_y + 83,
                        main_bar_h,
                        h2d_label,
                        h2d_title,
                        h2d_color,
                        fill_color="#f0fdf4" if useful_kind == "prefetch" else "#ecfeff",
                        text_color="#166534" if useful_kind == "prefetch" else "#155e75",
                    )
            ready_x = deadline_x(max(dz_min, min(dz_max, ready_ms)))
            parts.append(
                f'<circle cx="{ready_x:.1f}" cy="{deadline_zoom_title_y + 60:.1f}" r="4.5" fill="{verdict_color}">'
                f'<title>{html.escape(label)} | KV ready marker: {html.escape(verdict_text)}</title></circle>'
            )
            if projected_span:
                projected_start, projected_end = projected_span
                projected_margin = as_float(projected_realistic.get("projected_hardware_margin_ms")) if projected_realistic else None
                projected_duration = as_float(projected_realistic.get("projected_hardware_duration_ms")) if projected_realistic else None
                projected_color = unified_stack_color("projected_hardware")
                projected_label = short_bar_label(
                    [
                        "projected, not measured",
                        f"ready {display_ms(projected_margin)}" if projected_margin is not None and projected_margin >= 0 else "",
                        display_ms(projected_duration) if projected_duration is not None else "",
                    ],
                    max_chars=52,
                )
                px1 = deadline_x(max(dz_min, min(dz_max, projected_start)))
                px2 = deadline_x(max(dz_min, min(dz_max, projected_end)))
                pw = max(min_visible_bar_w, px2 - px1)
                projected_title = (
                    f"{label} | PROJECTED, NOT MEASURED hardware bypass | realistic projection | "
                    f"start={display_ms(projected_start)} relative to due | end={display_ms(projected_end)} relative to due | "
                    f"duration={display_ms(projected_duration) if projected_duration is not None else 'unknown'} | "
                    f"margin={display_ms(projected_margin) if projected_margin is not None else 'unknown'} | "
                    f"source={projected_realistic.get('measured_h2d_source', '') if projected_realistic else ''}"
                )
                parts.append(
                    f'<rect x="{px1:.1f}" y="{deadline_zoom_title_y + 115:.1f}" width="{pw:.1f}" height="{main_bar_h:.1f}" '
                    f'rx="4" fill="#ffffff" stroke="{projected_color}" stroke-width="2" stroke-dasharray="7 5" opacity="0.96">'
                    f'<title>{html.escape(projected_title)}</title></rect>'
                )
                if pw >= 160:
                    parts.append(
                        f'<text x="{px1 + pw / 2:.1f}" y="{deadline_zoom_title_y + 129:.1f}" text-anchor="middle" '
                        f'font-size="9" font-weight="900" fill="{projected_color}">{html.escape(projected_label)}</text>'
                    )
                else:
                    draw_small_bar_callout(
                        parts,
                        px1,
                        px1 + pw,
                        deadline_zoom_title_y + 115,
                        main_bar_h,
                        projected_label,
                        projected_title,
                        projected_color,
                        fill_color="#f0fdfa",
                        text_color="#115e59",
                    )

        verdict_y = y + row_h - 28
        verdict = str(row.get("lifecycle_verdict") or row.get("final_path") or row.get("per_gap_verdict") or "")
        explanation = str(row.get("lifecycle_explanation") or row.get("replay_cache_path_summary") or "")
        parts.append(f'<text x="{left}" y="{verdict_y:.1f}" font-size="10" font-weight="900" fill="#0f172a">verdict: {html.escape(verdict[:92])}</text>')
        if explanation:
            parts.append(f'<text x="{left}" y="{verdict_y + 16:.1f}" font-size="9" fill="#475569">{html.escape(explanation[:190])}</text>')

    append_legend(parts, left, height - 48)
    parts.append("</svg>")
    return "\n".join(parts)


def unified_per_gap_forensic_stack_html(
    gaps: list[dict[str, Any]],
    all_kv_events: list[dict[str, Any]],
    max_rows: int,
    kv_pool_residency_rows: list[dict[str, Any]] | None = None,
) -> str:
    if not gaps:
        return "<p>No timeline rows were available for the unified forensic stack.</p>"
    return f"""
    <p>This is a preview of a merged per-gap view. Each gap has a compact overview, a local zoom of the dense KV movement burst, a local zoom of replay execution, a GPU KV-pool residency zoom, and a deadline zoom for KV readiness.</p>
    <p class="note">Use the overview to see the big timing story. Use the expanded KV zoom to inspect H2D, D2H, and eviction bars. Use the replay zoom to inspect prefetch KV H2D, replay KV H2D, prefill/recompute, remaining before-first-token time, first-token timing, and decode. Use the GPU pool zoom to see whether SGLang's KV pool was nearly full around the replay. Use the deadline zoom to see whether useful KV H2D finished before or after replay was due.</p>
    <h3>Legend / How To Read This Timeline</h3>
    {unified_stack_legend_table_html()}
    <p class="note">When prefetch-side H2D exists, the replay zoom becomes a two-window broken-axis view: the left window shows the earlier green prefetch KV H2D, the right window shows replay execution, and the break marker shows the long elapsed time compressed between them.</p>
    <p class="note">In the GPU pool zoom, green/yellow means lower KV-pool pressure, orange/red means high pressure, and dark red means the KV pool was effectively full. This is direct SGLang KV memory-pool telemetry, not a coarse NVIDIA-SMI whole-GPU memory estimate.</p>
    <p class="note">In the deadline zoom, the dashed line is the readiness gap. Green means KV became ready before replay was due; red means useful KV H2D completed late.</p>
    <p class="note">The magenta <strong>prefill/recompute</strong> bar is model-forward work before the first output token. It may include recomputing missing KV or processing uncached replay prompt tokens. The gold <strong>remaining before-first-token time</strong> is leftover time after visible H2D and prefill/recompute are separated out.</p>
    <p class="note">Rendering rule: every instrumented event is drawn, even when it is very small. Tiny events use a minimum visual width so they remain visible; hover text keeps the exact measured duration.</p>
    <div class="setup-diagram">{build_unified_per_gap_stack_timeline_svg_v2(gaps, all_kv_events, max_rows, kv_pool_residency_rows)}</div>
    <p class="note">Target-row movement is drawn thicker and more opaque. Pressure/filler or other-session movement is thinner and faded. The zoom strip uses a local linear scale per gap, while the overview remains replay-relative symlog time.</p>
    """


def grouped_mode_comparison_timeline_html(
    rows: list[dict[str, Any]],
    all_kv_events: list[dict[str, Any]],
    kv_pool_residency_rows: list[dict[str, Any]] | None = None,
    projected_hardware_rows: list[dict[str, Any]] | None = None,
) -> str:
    if not rows:
        return """
        <p>No grouped mode comparison rows were available. This section appears when the same task/gap scenario exists in no prefetch and direct prefetch.</p>
        """
    scenario_count = len({str(row.get("comparison_scenario") or "") for row in rows})
    mode_key_items = []
    for mode in ("no_prefetch", "direct_prefetch", "dynamo_priority_hints", "projected_hardware_bypass"):
        fg, badge_bg = mode_badge_style(mode)
        row_bg, accent, opacity = mode_row_background_style(mode)
        mode_key_items.append(
            f'<span class="pill" style="background:{row_bg}; border:1px solid {accent}; color:{fg};">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:{accent};margin-right:6px;"></span>'
            f'{html.escape(display_mode(mode))}</span>'
        )
    mode_key_html = '<div class="toc-pills" style="margin-top:10px;">' + "".join(mode_key_items) + "</div>"
    return f"""
    <p>This view groups the same controlled scenario across modes. For example, <code>C00-NP</code>, <code>C00-DP</code>, <code>C00-DH</code>, and <code>C00-HW</code> are the same task/gap setup shown under no prefetch, direct prefetch, Dynamo priority hints only, and projected hardware bypass.</p>
    <p class="note">Mode order is always: no prefetch, direct prefetch, Dynamo priority hints only, then projected hardware bypass. Dynamo priority hints only sends priority metadata and an SGLang priority value; it does not issue our direct KV prefetch hook. The projected hardware row is <strong>not measured</strong>; it estimates where a low-overhead hardware KV movement path could have completed using the measured KV H2D duration plus a small fixed hardware-control overhead. Projected rows show only the compact projection overview, not detailed measured SGLang lanes.</p>
    <h3>Legend / How To Read This Timeline</h3>
    {unified_stack_legend_table_html()}
    <p class="note">Rows are lightly tinted by mode, with a stronger color strip on the far left of each row.</p>
    {mode_key_html}
    <div class="cards">
      <div class="card"><div class="label">scenarios compared</div><div class="value">{scenario_count}</div></div>
      <div class="card"><div class="label">timeline rows</div><div class="value">{len(rows)}</div></div>
      <div class="card"><div class="label">modes shown</div><div class="value">NP / DP / DH / HW</div></div>
    </div>
    <p class="note">The scenario row map and exact per-row numbers are in <strong>Evidence Tables / Raw Proof</strong> at the bottom of the report.</p>
    <div class="setup-diagram">{build_unified_per_gap_stack_timeline_svg_v2(rows, all_kv_events, len(rows), kv_pool_residency_rows, compact_projected_rows=True)}</div>
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
            "simple meaning": "The controller sends a marked direct-load request for the hinted session. Priority-hint-only mode does not do this.",
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
    kv_pool_sample_rows = kv_pool_samples_from_trace(all_timeline_rows, trace_rows)
    kv_pool_residency_rows = kv_pool_residency_by_gap_rows(all_timeline_rows, kv_pool_sample_rows)
    interesting_kv_pool_residency_rows = [
        row for row in kv_pool_residency_rows if str(row.get("row") or "") in interesting_labels
    ]
    grouped_comparison_rows = grouped_mode_comparison_rows(gaps, max_timeline_gaps)
    grouped_kv_pool_residency_rows = kv_pool_residency_by_gap_rows(grouped_comparison_rows, kv_pool_sample_rows)
    grouped_hardware_projection_rows = projected_hardware_bypass_rows(grouped_comparison_rows)
    dynamo_priority_rows = dynamo_priority_hint_translation_rows(gaps)
    prefetch_truth_summary_table_rows = prefetch_truth_summary_rows(gaps)
    prefetch_truth_table = prefetch_truth_table_rows(gaps)
    global_mode_readiness_table_rows = global_kv_readiness_by_mode_rows(gaps)
    global_mode_readiness_summary_table_rows = global_kv_readiness_by_mode_summary_rows(
        global_mode_readiness_table_rows
    )
    global_replay_start_summary_table_rows = global_replay_start_by_mode_summary_rows(
        global_mode_readiness_table_rows
    )
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
    hardware_bypass_projection_rows = projected_hardware_bypass_rows(all_timeline_rows)
    hardware_bypass_projection_summary_rows = projected_hardware_bypass_summary_rows(
        hardware_bypass_projection_rows
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
        "prefetch_truth_verdict",
        "prefetch_truth_explanation",
        "prefetch_truth_confidence",
        "hint_completed_before_replay",
        "hint_h2d_seen",
        "replay_reloaded_after_hint",
        "replay_recomputed_after_hint",
        "true_kv_prefetch_success",
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
        ("prefetch-truth", "Prefetch Truth Check"),
        ("hardware-bypass", "Projected Hardware Bypass Benefit"),
        ("h2d-pressure", "KV H2D Bandwidth Pressure"),
        ("gpu-kv-residency", "GPU KV Pool Residency"),
        ("delay-breakdown", "Replay Delay Breakdown"),
        ("client-dispatch-kv", "Client Dispatch KV Movement"),
        ("timeline-guide", "How To Read Timelines"),
        ("readable-phase-timeline", "Readable KV Lifecycle Timeline"),
        ("unified-forensic-stack", "Unified Forensic Stack Timeline"),
        ("grouped-mode-comparison", "Grouped Mode Comparison"),
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
    <p>This section gives the headline numbers across no-prefetch and direct-prefetch modes. The replay-before-first-token window is split into evidence colors: cyan for replay-side host KV load, magenta for estimated recompute/rebuild, and gold for remaining prefill or TTFT work.</p>
    {metric_cards_html(mode_rows)}
  </details>

  <details id="setup" class="section-card theme-setup">
    <summary><h2>Experiment Setup And Manager Summary</h2></summary>
    {manager_setup_html(run_environment)}
  </details>

  <details id="global-prefetch" class="section-card theme-global" open>
    <summary><h2>{html.escape(global_title)}</h2></summary>
    <p>This section compares when the first replay token appeared and when useful KV became ready.</p>
    {global_readiness_html(gaps)}
  </details>

  <details id="prefetch-truth" class="section-card theme-observations">
    <summary><h2>Prefetch Truth Check</h2></summary>
    {prefetch_truth_check_html(gaps)}
  </details>

  <details id="hardware-bypass" class="section-card theme-directkv">
    <summary><h2>Projected Hardware Bypass Benefit</h2></summary>
    {projected_hardware_bypass_html(hardware_bypass_projection_rows)}
  </details>

  <details id="h2d-pressure" class="section-card theme-directkv">
    <summary><h2>KV H2D Bandwidth Pressure</h2></summary>
    <p>This section shows how much host-to-device KV movement was happening near replay deadlines. It helps explain whether a late replay was isolated or happened while the KV movement path was already busy.</p>
    {h2d_bandwidth_pressure_html(gaps, exact_kv_movement_rows)}
  </details>

  <details id="gpu-kv-residency" class="section-card theme-directkv">
    <summary><h2>GPU KV Pool Residency</h2></summary>
    <p>This section uses direct SGLang KV-pool state, not total GPU memory, to show whether the KV cache pool was near full around replay and H2D events.</p>
    {gpu_kv_residency_summary_html(kv_pool_residency_rows)}
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
    {unified_per_gap_forensic_stack_html(interesting, all_kv_movement_events, max_timeline_gaps, interesting_kv_pool_residency_rows)}
  </details>

  <details id="grouped-mode-comparison" class="section-card theme-profiled" open>
    <summary><h2>Grouped Mode Comparison Timeline</h2></summary>
    {grouped_mode_comparison_timeline_html(grouped_comparison_rows, all_kv_movement_events, grouped_kv_pool_residency_rows, grouped_hardware_projection_rows)}
  </details>

  {live_section}

  <details id="observations" class="section-card theme-observations">
    <summary><h2>Key Observations Per Gap/Session</h2></summary>
    <p>This section translates the timeline rows into plain English. It uses the same compact row names as the chart, so <code>G00</code> here means the same <code>G00</code> in the timeline.</p>
    {table_html(key_observation_rows(interesting), ["row", "mode", "status", "prefetch_truth", "what happened", "why it matters", "prefetch_truth_explanation", "tool_wait_ms", "resume_ttft_ms", "replay_path", "verdict"])}
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
    <h3>Prefetch Truth Summary</h3>
    <p class="note">This table distinguishes early hint completion from true useful KV prefetch. A true success needs KV residency/reuse evidence, not just an early purple bar.</p>
    {table_html(prefetch_truth_summary_table_rows)}
    <h3>Prefetch Truth Rows</h3>
    {table_html(prefetch_truth_table, limit=1000)}
    <h3>Detailed KV Block Lifecycle Rows</h3>
    <p class="note">The H2D timing columns come from SGLang-visible KV movement hooks. Recompute timing is labeled <code>_est</code> because it is inferred from replay prefill/TTFT counters rather than from a physical block-level recompute event.</p>
    {table_html(detailed_kv_lifecycle_table_rows(gaps, kv_block_rows), limit=1000)}
    <h3>Replay H2D Readiness Rows</h3>
    {table_html(replay_h2d_readiness_table_rows)}
    <h3>Replay H2D Readiness Buckets</h3>
    {table_html(replay_h2d_readiness_bucket_table_rows)}
    <h3>Global KV Readiness By Mode Summary</h3>
    {table_html(global_mode_readiness_summary_table_rows)}
    <h3>Global Replay Start By Mode Summary</h3>
    {table_html(global_replay_start_summary_table_rows)}
    <h3>Global KV Readiness By Mode Rows</h3>
    {table_html(global_mode_readiness_table_rows, limit=1000)}
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
    <h3>GPU KV Pool Residency By Gap</h3>
    <p class="note">These rows come from direct SGLang KV memory pool samples. They are not derived from <code>nvidia-smi</code>.</p>
    {table_html(kv_pool_residency_rows, limit=1000)}
    <h3>Raw GPU KV Pool Samples</h3>
    {table_html(kv_pool_sample_rows, limit=2000)}
    <h3>All Aligned KV Movement Rows</h3>
    {table_html(all_kv_movement_events, limit=2000)}
    <h3>Client Dispatch KV Movement Summary</h3>
    {table_html(client_dispatch_kv_summary_rows)}
    <h3>Client Dispatch KV Movement Events</h3>
    {table_html(client_dispatch_kv_event_rows, limit=2000)}
    <h3>H2D Activity Window Rows</h3>
    {table_html(h2d_activity_window_table_rows)}
    <h3>Projected Hardware Bypass Summary</h3>
    {table_html(hardware_bypass_projection_summary_rows)}
    <h3>Projected Hardware Bypass Per-Gap Rows</h3>
    {table_html(hardware_bypass_projection_rows, limit=2000)}
    <h3>Per-Gap H2D Pressure Rows</h3>
    {table_html(all_h2d_pressure_rows)}
    <h3>Per-Gap H2D Contention Verdict Rows</h3>
    {table_html(h2d_contention_summary_table_rows)}
    <h3>Per-Gap H2D Contention Event Rows</h3>
    {table_html(h2d_contention_event_table_rows, limit=2000)}
    <h3>Mode Summary</h3>
    {table_html(mode_rows)}
    <h3>Dynamo Priority Hint Translation Rows</h3>
    <p class="note">These rows show the bridge used by <code>dynamo_priority_hints</code>: the emitted <code>custom_params.nvext.agent_hints</code> priority and the translated SGLang <code>priority</code> integer sent on the OpenAI-compatible request.</p>
    {table_html(dynamo_priority_rows, limit=1000)}
    <h3>Grouped Mode Comparison Rows</h3>
    <p class="note">This table maps compact grouped timeline labels such as <code>C00-NP</code>, <code>C00-DP</code>, and <code>C00-HW</code> back to their exact mode, task, gap, wait time, prefetch margin, H2D counts, and verdict.</p>
    {table_html(mode_comparison_summary_rows(grouped_comparison_rows), limit=1000)}
    <h3>Grouped Projected Hardware Bypass Overlay Rows</h3>
    <p class="note">These are projected rows used only for the dashed teal overlay in the grouped comparison timeline. They are not measured events.</p>
    {table_html(grouped_hardware_projection_rows, limit=1000)}
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
    hardware_bypass_projection = projected_hardware_bypass_rows(all_labeled_gaps)
    hardware_bypass_projection_summary = projected_hardware_bypass_summary_rows(hardware_bypass_projection)
    global_mode_readiness = global_kv_readiness_by_mode_rows(all_gaps)
    global_mode_readiness_summary = global_kv_readiness_by_mode_summary_rows(global_mode_readiness)
    global_replay_start_summary = global_replay_start_by_mode_summary_rows(global_mode_readiness)
    dynamo_priority_rows = dynamo_priority_hint_translation_rows(all_gaps)
    prefetch_truth_table = prefetch_truth_table_rows(all_gaps)
    prefetch_truth_summary = prefetch_truth_summary_rows(all_gaps)
    replay_delay_breakdown = replay_delay_breakdown_rows(all_labeled_gaps, h2d_activity_events)
    replay_delay_verdicts = replay_delay_verdict_rows(replay_delay_breakdown)
    replay_delay_running_context = replay_delay_running_context_rows(all_labeled_gaps, all_trace_rows, h2d_activity_events)
    replay_delay_stage_trace = request_stage_trace_rows(all_labeled_gaps, all_trace_rows)
    replay_delay_h2d_activity = h2d_activity_during_delay_rows(all_labeled_gaps, h2d_activity_events)
    replay_delay_gap_verdicts = delay_verdicts_by_gap_rows(replay_delay_breakdown)
    kv_pool_sample_rows = kv_pool_samples_from_trace(all_labeled_gaps, all_trace_rows)
    kv_pool_residency_rows = kv_pool_residency_by_gap_rows(all_labeled_gaps, kv_pool_sample_rows)
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
    write_csv(args.out_dir / "kv_pool_samples.csv", kv_pool_sample_rows)
    write_csv(args.out_dir / "kv_pool_residency_by_gap.csv", kv_pool_residency_rows)
    write_csv(args.out_dir / "h2d_activity_events.csv", h2d_activity_events)
    write_csv(args.out_dir / "all_aligned_kv_movement_events.csv", all_kv_movement_events)
    write_csv(args.out_dir / "client_dispatch_kv_movement_summary.csv", client_dispatch_kv_summary)
    write_csv(args.out_dir / "client_dispatch_kv_movement_events.csv", client_dispatch_kv_events)
    write_csv(args.out_dir / "h2d_pressure_by_gap.csv", h2d_pressure_rows)
    write_csv(args.out_dir / "h2d_activity_windows.csv", h2d_activity_windows)
    write_csv(args.out_dir / "h2d_contention_by_gap.csv", h2d_contention_summary)
    write_csv(args.out_dir / "h2d_contention_events.csv", h2d_contention_events)
    write_csv(args.out_dir / "projected_hardware_bypass.csv", hardware_bypass_projection)
    write_csv(args.out_dir / "projected_hardware_bypass_summary.csv", hardware_bypass_projection_summary)
    write_csv(args.out_dir / "global_kv_readiness_by_mode.csv", global_mode_readiness)
    write_csv(args.out_dir / "global_kv_readiness_by_mode_summary.csv", global_mode_readiness_summary)
    write_csv(args.out_dir / "global_replay_start_by_mode_summary.csv", global_replay_start_summary)
    write_csv(args.out_dir / "dynamo_priority_hint_translation.csv", dynamo_priority_rows)
    write_csv(args.out_dir / "prefetch_truth_table.csv", prefetch_truth_table)
    write_csv(args.out_dir / "prefetch_truth_summary.csv", prefetch_truth_summary)
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
            "kv_pool_samples": kv_pool_sample_rows,
            "kv_pool_residency_by_gap": kv_pool_residency_rows,
            "replay_h2d_readiness_summary": replay_h2d_readiness_summary(h2d_readiness),
            "h2d_activity_events": h2d_activity_events,
            "all_aligned_kv_movement_events": all_kv_movement_events,
            "client_dispatch_kv_movement_summary": client_dispatch_kv_summary,
            "client_dispatch_kv_movement_events": client_dispatch_kv_events,
            "h2d_pressure_by_gap": h2d_pressure_rows,
            "h2d_activity_windows": h2d_activity_windows,
            "h2d_contention_by_gap": h2d_contention_summary,
            "h2d_contention_events": h2d_contention_events,
            "projected_hardware_bypass": hardware_bypass_projection,
            "projected_hardware_bypass_summary": hardware_bypass_projection_summary,
            "global_kv_readiness_by_mode": global_mode_readiness,
            "global_kv_readiness_by_mode_summary": global_mode_readiness_summary,
            "global_replay_start_by_mode_summary": global_replay_start_summary,
            "dynamo_priority_hint_translation": dynamo_priority_rows,
            "prefetch_truth_table": prefetch_truth_table,
            "prefetch_truth_summary": prefetch_truth_summary,
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
