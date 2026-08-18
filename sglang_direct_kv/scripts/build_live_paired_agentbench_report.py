#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from build_live_agentbench_tool_gap_report import (
    augment_gaps_with_prefetch,
    build_expanded_gap_timeline_svg,
    build_tool_gaps,
    is_preflight_request,
    maybe_float,
    normalize_requests,
    read_csv,
    read_jsonl,
    table_html,
    write_csv,
)


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def short(value: Any, limit: int = 120) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def avg(values: list[float]) -> float | str:
    return round(mean(values), 3) if values else ""


def med(values: list[float]) -> float | str:
    return round(median(values), 3) if values else ""


def pct(delta: float | None, baseline: float | None) -> float | str:
    if delta is None or baseline in (None, 0):
        return ""
    return round(delta * 100.0 / baseline, 2)


def has_events(value: Any) -> bool:
    try:
        return int(value or 0) > 0
    except (TypeError, ValueError):
        return False


def delta_ms(later: Any, earlier: Any) -> float | str:
    later_value = as_float(later)
    earlier_value = as_float(earlier)
    if later_value is None or earlier_value is None:
        return ""
    return round(later_value - earlier_value, 3)


def task_index_map(task_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run_id") or ""): row for row in task_rows if row.get("run_id")}


def add_task_and_pair_fields(
    requests: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    mode: str,
) -> None:
    by_run = task_index_map(task_rows)
    request_by_ordinal = {str(row.get("ordinal") or ""): row for row in requests}
    for request in requests:
        run_id = str(request.get("parent_run_id") or "")
        task = by_run.get(run_id, {})
        request["mode"] = mode
        request["task_index"] = task.get("task_index", "")
        request["task_status"] = task.get("status", "")

    per_task_counter: defaultdict[str, int] = defaultdict(int)
    for gap in sorted(gaps, key=lambda row: float(row.get("current_start_ms") or 0.0)):
        run_id = str(gap.get("parent_run_id") or "")
        task = by_run.get(run_id, {})
        task_index = str(task.get("task_index", ""))
        task_key = task_index if task_index else run_id
        gap_order = per_task_counter[task_key]
        per_task_counter[task_key] += 1
        gap["mode"] = mode
        gap["task_index"] = task_index
        gap["task_status"] = task.get("status", "")
        gap["gap_order_in_task"] = gap_order
        gap["pair_key"] = f"task_{task_key}:gap_{gap_order}"
        current_request = request_by_ordinal.get(str(gap.get("from_proxy_ordinal") or ""), {})
        resume_request = request_by_ordinal.get(str(gap.get("to_proxy_ordinal") or ""), {})
        gap["current_agent_session_id"] = current_request.get("agent_session_id") or current_request.get("context_request_id") or ""
        gap["resume_agent_session_id"] = resume_request.get("agent_session_id") or resume_request.get("context_request_id") or ""
        gap["resume_context_request_id"] = resume_request.get("context_request_id") or ""


def context_from_trace_event(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("kv_context")
    if isinstance(context, dict):
        return context
    return row


def agent_session_from_context(context: dict[str, Any]) -> str:
    for key in ("agent_session_id", "session_id"):
        value = context.get(key)
        if value not in ("", None):
            return str(value)
    req = context.get("request")
    if isinstance(req, dict):
        for key in ("agent_session_id", "session_id"):
            value = req.get(key)
            if value not in ("", None):
                return str(value)
    sessions = context.get("agent_sessions")
    if isinstance(sessions, list):
        for item in sessions:
            if isinstance(item, dict):
                value = item.get("agent_session_id")
                if value not in ("", None):
                    return str(value)
    return ""


def relative_ms_from_ns(ts_ns: Any, base_ts: float) -> float | str:
    value = as_float(ts_ns)
    if value is None:
        return ""
    return round((value / 1_000_000_000.0 - base_ts) * 1000.0, 3)


def movement_events_by_session(
    trace_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    base_ts: float,
) -> dict[str, list[dict[str, Any]]]:
    direct_categories = {
        "hiradix.init_load_back.end": "init_load_back",
        "hiradix.load_back.end": "load_back",
        "hicache.load.end": "hicache_load",
        "hicache.start_loading.end": "hicache_start_loading",
        "hostpool.load_to_device_per_layer.end": "hostpool_h2d",
    }
    by_session: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        event = str(row.get("event") or "")
        category = direct_categories.get(event)
        if not category:
            continue
        context = context_from_trace_event(row)
        if context.get("direction") != "host_to_device":
            continue
        session_id = agent_session_from_context(context)
        if not session_id or "::live_prefetch::" in session_id:
            continue
        by_session[session_id].append(
            {
                "source": "sglang_trace",
                "event": event,
                "category": category,
                "agent_session_id": session_id,
                "direction": "host_to_device",
                "duration_ms": row.get("duration_ms", ""),
                "start_or_end_ms": relative_ms_from_ns(row.get("ts_ns"), base_ts),
            }
        )

    for row in telemetry_rows:
        event = str(row.get("event") or "")
        if event not in {"kv_telemetry.copy.start", "kv_telemetry.copy.end"}:
            continue
        if str(row.get("direction") or "") != "host_to_device":
            continue
        session_id = str(row.get("agent_session_id") or "")
        if not session_id or "::live_prefetch::" in session_id:
            continue
        by_session[session_id].append(
            {
                "source": "kv_copy_telemetry",
                "event": event,
                "category": "telemetry_h2d",
                "agent_session_id": session_id,
                "direction": "host_to_device",
                "duration_ms": row.get("duration_ms", ""),
                "start_or_end_ms": relative_ms_from_ns(row.get("ts_ns"), base_ts),
            }
        )
    return by_session


def summarize_movement_window(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "start_ms": "",
            "end_ms": "",
            "duration_ms": "",
            "events": "0",
            "categories": "",
        }
    telemetry_events = [event for event in events if event.get("category") == "telemetry_h2d"]
    chosen = telemetry_events or events
    starts: list[float] = []
    ends: list[float] = []
    total_duration = 0.0
    duration_count = 0
    categories: Counter[str] = Counter()
    for event in chosen:
        ts = as_float(event.get("start_or_end_ms"))
        if ts is None:
            continue
        duration = as_float(event.get("duration_ms"))
        if duration is not None:
            starts.append(ts - duration)
            ends.append(ts)
            total_duration += duration
            duration_count += 1
        else:
            starts.append(ts)
            ends.append(ts)
        categories[str(event.get("category") or "")] += 1
    if not starts or not ends:
        return {
            "start_ms": "",
            "end_ms": "",
            "duration_ms": "",
            "events": str(len(chosen)),
            "categories": "",
        }
    return {
        "start_ms": round(min(starts), 3),
        "end_ms": round(max(ends), 3),
        "duration_ms": round(total_duration, 3) if duration_count else "",
        "events": str(len(chosen)),
        "categories": ", ".join(f"{name}:{count}" for name, count in categories.items()),
    }


def attach_replay_kv_windows(
    gaps: list[dict[str, Any]],
    events_by_session: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    replay_evidence: list[dict[str, Any]] = []
    for gap in gaps:
        session_id = str(gap.get("resume_agent_session_id") or "")
        replay_start = as_float(gap.get("resume_start_ms"))
        replay_end = as_float(gap.get("resume_end_ms"))
        matched: list[dict[str, Any]] = []
        if session_id and replay_start is not None and replay_end is not None:
            for event in events_by_session.get(session_id, []):
                ts = as_float(event.get("start_or_end_ms"))
                if ts is None:
                    continue
                if replay_start <= ts <= replay_end:
                    matched.append(event)
        summary = summarize_movement_window(matched)
        gap["replay_kv_h2d_start_ms"] = summary["start_ms"]
        gap["replay_kv_h2d_end_ms"] = summary["end_ms"]
        gap["replay_kv_h2d_duration_ms"] = summary["duration_ms"]
        gap["replay_kv_h2d_events"] = summary["events"]
        gap["replay_kv_h2d_categories"] = summary["categories"]
        hint_end = as_float(gap.get("prefetch_end_ms"))
        replay_kv_start = as_float(summary["start_ms"])
        if replay_kv_start is not None and hint_end is not None:
            gap["replay_kv_started_before_hint_done"] = "yes" if replay_kv_start < hint_end else "no"
        else:
            gap["replay_kv_started_before_hint_done"] = ""
        replay_evidence.append(
            {
                "session_id": gap.get("session_id", ""),
                "task_index": gap.get("task_index", ""),
                "gap_order_in_task": gap.get("gap_order_in_task", ""),
                "tools": gap.get("tool_names", ""),
                "resume_agent_session_id": session_id,
                "replay_kv_h2d_observed": "yes" if matched else "no",
                "replay_kv_h2d_start_ms": gap.get("replay_kv_h2d_start_ms", ""),
                "replay_kv_h2d_end_ms": gap.get("replay_kv_h2d_end_ms", ""),
                "replay_kv_h2d_duration_ms": gap.get("replay_kv_h2d_duration_ms", ""),
                "replay_kv_h2d_events": gap.get("replay_kv_h2d_events", ""),
                "replay_kv_h2d_categories": gap.get("replay_kv_h2d_categories", ""),
                "hint_prefetch_end_ms": gap.get("prefetch_end_ms", ""),
                "replay_kv_started_before_hint_done": gap.get("replay_kv_started_before_hint_done", ""),
                "interpretation": (
                    "real replay request did host-to-device KV work while serving this resume"
                    if matched
                    else "no replay-side host-to-device KV movement was attributed inside this replay window"
                ),
            }
        )
    return replay_evidence


def attach_direct_kv_windows(
    gaps: list[dict[str, Any]],
    direct_kv_events: list[dict[str, Any]],
    direct_kv_evidence: list[dict[str, Any]],
) -> None:
    evidence_by_hint = {str(row.get("hint_id") or ""): row for row in direct_kv_evidence if row.get("hint_id")}
    events_by_hint: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in direct_kv_events:
        hint_id = str(event.get("hint_id") or "")
        if not hint_id:
            continue
        if event.get("direction") != "host_to_device":
            continue
        category = str(event.get("category") or "")
        if category not in {
            "telemetry_h2d",
            "hostpool_h2d",
            "hicache_load",
            "load_back",
            "init_load_back",
            "hicache_start_loading",
        }:
            continue
        events_by_hint[hint_id].append(event)

    for gap in gaps:
        hint_id = str(gap.get("hint_id") or "")
        evidence = evidence_by_hint.get(hint_id, {})
        if evidence:
            gap["direct_kv_load_observed"] = evidence.get("direct_kv_load_observed", "")
            gap["direct_kv_init_load_back_events"] = evidence.get("init_load_back_events", "")
            gap["direct_kv_load_back_events"] = evidence.get("load_back_events", "")
            gap["direct_kv_hicache_load_events"] = evidence.get("hicache_load_events", "")
            gap["direct_kv_telemetry_h2d_events"] = evidence.get("telemetry_h2d_events", "")
        hint_events = events_by_hint.get(hint_id, [])
        if not hint_events:
            gap["direct_kv_h2d_start_ms"] = ""
            gap["direct_kv_h2d_end_ms"] = ""
            gap["direct_kv_h2d_duration_ms"] = ""
            gap["direct_kv_h2d_events"] = "0"
            continue

        telemetry_events = [event for event in hint_events if event.get("category") == "telemetry_h2d"]
        chosen = telemetry_events or hint_events
        starts: list[float] = []
        ends: list[float] = []
        total_duration = 0.0
        duration_count = 0
        categories: Counter[str] = Counter()
        for event in chosen:
            ts = as_float(event.get("start_or_end_ms"))
            if ts is None:
                continue
            duration = as_float(event.get("duration_ms"))
            if duration is not None:
                starts.append(ts - duration)
                ends.append(ts)
                total_duration += duration
                duration_count += 1
            else:
                starts.append(ts)
                ends.append(ts)
            categories[str(event.get("category") or "")] += 1

        if not starts or not ends:
            gap["direct_kv_h2d_start_ms"] = ""
            gap["direct_kv_h2d_end_ms"] = ""
            gap["direct_kv_h2d_duration_ms"] = ""
            gap["direct_kv_h2d_events"] = str(len(chosen))
            continue

        gap["direct_kv_h2d_start_ms"] = round(min(starts), 3)
        gap["direct_kv_h2d_end_ms"] = round(max(ends), 3)
        gap["direct_kv_h2d_duration_ms"] = round(total_duration, 3) if duration_count else ""
        gap["direct_kv_h2d_events"] = str(len(chosen))
        gap["direct_kv_h2d_categories"] = ", ".join(f"{name}:{count}" for name, count in categories.items())


def load_live_run(root: Path, mode: str, include_preflight: bool) -> dict[str, Any]:
    proxy_jsonl = root / "tool_normalizer_proxy.jsonl"
    task_index_csv = root / "exp6_direct_sglang_task_index.csv"
    hint_log = root / "live_hint_events.jsonl"
    controller_log = root / "live_prefetch_controller.jsonl"
    direct_kv_report_dir = root / "live_direct_kv_load_report"
    direct_kv_summary_csv = direct_kv_report_dir / "live_direct_kv_load_summary.csv"
    direct_kv_evidence_csv = direct_kv_report_dir / "live_direct_kv_load_evidence.csv"
    direct_kv_events_csv = direct_kv_report_dir / "live_direct_kv_load_events.csv"
    trace_jsonl = root / "live_direct_kv_trace.jsonl"
    telemetry_jsonl = root / "live_direct_kv_copy_telemetry.jsonl"

    raw_rows = read_jsonl(proxy_jsonl)
    hint_rows = read_jsonl(hint_log)
    controller_rows = read_jsonl(controller_log)
    trace_rows = read_jsonl(trace_jsonl)
    telemetry_rows = read_jsonl(telemetry_jsonl)
    direct_kv_summary = read_csv(direct_kv_summary_csv)
    direct_kv_evidence = read_csv(direct_kv_evidence_csv)
    direct_kv_events = read_csv(direct_kv_events_csv)
    task_rows = read_csv(task_index_csv)
    all_requests = normalize_requests(raw_rows)
    if include_preflight:
        requests = all_requests
        excluded_preflight = 0
    else:
        requests = [row for row in all_requests if not is_preflight_request(row)]
        excluded_preflight = len(all_requests) - len(requests)

    gaps = build_tool_gaps(requests)
    base_ts = min((float(row["start_ts"]) for row in all_requests), default=0.0)
    gaps = augment_gaps_with_prefetch(gaps, hint_rows, controller_rows, base_ts)
    add_task_and_pair_fields(requests, gaps, task_rows, mode)
    attach_direct_kv_windows(gaps, direct_kv_events, direct_kv_evidence)
    replay_kv_evidence = attach_replay_kv_windows(
        gaps,
        movement_events_by_session(trace_rows, telemetry_rows, base_ts),
    )

    return {
        "mode": mode,
        "root": str(root),
        "proxy_jsonl": str(proxy_jsonl),
        "task_index_csv": str(task_index_csv),
        "hint_log": str(hint_log) if hint_log.exists() else "",
        "controller_log": str(controller_log) if controller_log.exists() else "",
        "trace_jsonl": str(trace_jsonl) if trace_jsonl.exists() else "",
        "telemetry_jsonl": str(telemetry_jsonl) if telemetry_jsonl.exists() else "",
        "direct_kv_summary": direct_kv_summary,
        "direct_kv_evidence": direct_kv_evidence,
        "direct_kv_events": direct_kv_events,
        "replay_kv_evidence": replay_kv_evidence,
        "direct_kv_report_html": str(direct_kv_report_dir / "live_direct_kv_load_report.html")
        if (direct_kv_report_dir / "live_direct_kv_load_report.html").exists()
        else "",
        "raw_request_count": len(raw_rows),
        "captured_model_requests": len(all_requests),
        "excluded_preflight_requests": excluded_preflight,
        "requests": requests,
        "gaps": gaps,
        "task_rows": task_rows,
        "hint_rows": hint_rows,
        "controller_rows": controller_rows,
    }


def mode_summary(run: dict[str, Any]) -> dict[str, Any]:
    requests = run["requests"]
    gaps = run["gaps"]
    hint_rows = run["hint_rows"]
    controller_rows = run["controller_rows"]
    gap_values = [float(row.get("tool_gap_ms") or 0.0) for row in gaps]
    resume_values = [float(row.get("resume_latency_ms") or 0.0) for row in gaps if row.get("resume_latency_ms") not in ("", None)]
    prefetch_attempts = [
        row
        for row in gaps
        if row.get("prefetch_start_ms") not in ("", None) or row.get("prefetch_end_ms") not in ("", None)
    ]
    margins = [float(row["prefetch_margin_ms"]) for row in gaps if row.get("prefetch_margin_ms") not in ("", None)]
    durations = [
        float(row["prefetch_duration_ms"])
        for row in gaps
        if row.get("prefetch_duration_ms") not in ("", None)
    ]
    tool_counter: Counter[str] = Counter()
    for request in requests:
        for name in str(request.get("tool_names") or "").split(","):
            if name:
                tool_counter[name] += 1
    return {
        "mode": run["mode"],
        "tasks_in_index": len(run["task_rows"]),
        "captured_model_requests": run["captured_model_requests"],
        "analyzed_model_requests": len(requests),
        "excluded_preflight_requests": run["excluded_preflight_requests"],
        "requests_with_tools": sum(1 for row in requests if int(row.get("tool_count") or 0) > 0),
        "total_tool_calls": sum(int(row.get("tool_count") or 0) for row in requests),
        "observed_tool_gaps": len(gaps),
        "avg_tool_gap_ms": avg(gap_values),
        "median_tool_gap_ms": med(gap_values),
        "avg_resume_request_latency_ms": avg(resume_values),
        "median_resume_request_latency_ms": med(resume_values),
        "live_hints_submitted": len(hint_rows),
        "controller_events": len(controller_rows),
        "prefetch_attempts_matched_to_gaps": len(prefetch_attempts),
        "prefetch_done_before_resume": sum(1 for row in prefetch_attempts if row.get("prefetch_done_before_resume") == 1),
        "late_prefetch_attempts": sum(1 for row in prefetch_attempts if as_float(row.get("prefetch_margin_ms")) is not None and float(row["prefetch_margin_ms"]) < 0),
        "avg_prefetch_duration_ms": avg(durations),
        "avg_prefetch_margin_ms": avg(margins),
        "top_tools": ", ".join(f"{name}: {count}" for name, count in tool_counter.most_common(8)),
    }


def pair_gaps(no_gaps: list[dict[str, Any]], prefetch_gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    no_by_key = {str(row.get("pair_key") or ""): row for row in no_gaps}
    prefetch_by_key = {str(row.get("pair_key") or ""): row for row in prefetch_gaps}
    keys = sorted(
        set(no_by_key) | set(prefetch_by_key),
        key=lambda key: (
            int(key.split(":")[0].replace("task_", "")) if key.startswith("task_") and key.split(":")[0].replace("task_", "").isdigit() else 10**9,
            int(key.split(":gap_")[-1]) if ":gap_" in key and key.split(":gap_")[-1].isdigit() else 10**9,
            key,
        ),
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        base = no_by_key.get(key, {})
        pref = prefetch_by_key.get(key, {})
        no_latency = as_float(base.get("resume_latency_ms"))
        pref_latency = as_float(pref.get("resume_latency_ms"))
        delta = round(no_latency - pref_latency, 3) if no_latency is not None and pref_latency is not None else None
        margin = as_float(pref.get("prefetch_margin_ms"))
        if delta is None:
            perf = "unpaired"
        elif delta > 0:
            perf = "prefetch run faster"
        elif delta < 0:
            perf = "prefetch run slower"
        else:
            perf = "same"
        if margin is None:
            prefetch_verdict = "no matched hint"
        elif margin >= 0:
            prefetch_verdict = "prefetch done before resume"
        else:
            prefetch_verdict = "prefetch late"
        rows.append(
            {
                "pair_key": key,
                "task_index": pref.get("task_index") or base.get("task_index") or "",
                "gap_order_in_task": pref.get("gap_order_in_task") if pref else base.get("gap_order_in_task", ""),
                "tool_names": pref.get("tool_names") or base.get("tool_names") or "",
                "phases": f"{pref.get('from_phase') or base.get('from_phase') or ''} -> {pref.get('to_phase') or base.get('to_phase') or ''}",
                "no_prefetch_tool_gap_ms": base.get("tool_gap_ms", ""),
                "prefetch_tool_gap_ms": pref.get("tool_gap_ms", ""),
                "no_prefetch_resume_request_latency_ms": base.get("resume_latency_ms", ""),
                "prefetch_resume_request_latency_ms": pref.get("resume_latency_ms", ""),
                "prefetch_gain_ms": delta if delta is not None else "",
                "prefetch_gain_pct": pct(delta, no_latency),
                "performance_verdict": perf,
                "prefetch_status": pref.get("prefetch_status", ""),
                "prefetch_duration_ms": pref.get("prefetch_duration_ms", ""),
                "prefetch_margin_ms": pref.get("prefetch_margin_ms", ""),
                "prefetch_verdict": prefetch_verdict,
                "prefetch_resume_overlap_ms": pref.get("prefetch_resume_overlap_ms", ""),
                "hint_side_h2d_events": pref.get("direct_kv_h2d_events", ""),
                "replay_side_h2d_events": pref.get("replay_kv_h2d_events", ""),
                "replay_kv_started_before_hint_done": pref.get("replay_kv_started_before_hint_done", ""),
                "pairing_note": "paired by same SWE-bench task index and same gap order; live runs can still diverge",
                "no_prefetch_preview": short(base.get("resume_preview", "")),
                "prefetch_preview": short(pref.get("resume_preview", "")),
            }
        )
    return rows


def paired_summary(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired = [
        row
        for row in pair_rows
        if row.get("no_prefetch_resume_request_latency_ms") not in ("", None)
        and row.get("prefetch_resume_request_latency_ms") not in ("", None)
    ]
    gains = [float(row["prefetch_gain_ms"]) for row in paired if row.get("prefetch_gain_ms") not in ("", None)]
    no_latencies = [float(row["no_prefetch_resume_request_latency_ms"]) for row in paired]
    pre_latencies = [float(row["prefetch_resume_request_latency_ms"]) for row in paired]
    late = [row for row in pair_rows if row.get("prefetch_verdict") == "prefetch late"]
    ready = [row for row in pair_rows if row.get("prefetch_verdict") == "prefetch done before resume"]
    return [
        {
            "paired_tool_gaps": len(paired),
            "avg_no_prefetch_resume_request_latency_ms": avg(no_latencies),
            "avg_prefetch_resume_request_latency_ms": avg(pre_latencies),
            "avg_prefetch_gain_ms": avg(gains),
            "median_prefetch_gain_ms": med(gains),
            "prefetch_faster_pairs": sum(1 for row in paired if float(row["prefetch_gain_ms"]) > 0),
            "prefetch_slower_pairs": sum(1 for row in paired if float(row["prefetch_gain_ms"]) < 0),
            "prefetch_done_before_resume_pairs": len(ready),
            "prefetch_late_pairs": len(late),
        }
    ]


def key_deductions(mode_rows: list[dict[str, Any]], pair_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode = {row["mode"]: row for row in mode_rows}
    no = by_mode.get("no_prefetch", {})
    pref = by_mode.get("live_prefetch", {})
    paired = pair_summary_rows[0] if pair_summary_rows else {}
    return [
        {
            "finding": "This report uses real live SWE-bench / Deep Agents traffic.",
            "evidence": f"no-prefetch analyzed requests={no.get('analyzed_model_requests', '')}; live-prefetch analyzed requests={pref.get('analyzed_model_requests', '')}.",
            "why_it_matters": "The request stream is no longer synthetic; tool calls, tool gaps, and resumes come from the real harness talking to SGLang.",
        },
        {
            "finding": "Tool gaps are real prefetch opportunities.",
            "evidence": f"no-prefetch gaps={no.get('observed_tool_gaps', '')}; live-prefetch gaps={pref.get('observed_tool_gaps', '')}; avg prefetch-run gap={pref.get('avg_tool_gap_ms', '')} ms.",
            "why_it_matters": "These gaps are where a runtime would issue KV residency hints after a tool call.",
        },
        {
            "finding": "Best-effort software prefetch can miss short live windows.",
            "evidence": f"live-prefetch matched attempts={pref.get('prefetch_attempts_matched_to_gaps', '')}; late attempts={pref.get('late_prefetch_attempts', '')}; avg prefetch duration={pref.get('avg_prefetch_duration_ms', '')} ms.",
            "why_it_matters": "If the controller/SGLang path takes longer than the tool gap, the next agent turn still arrives before the prefetch path finishes.",
        },
        {
            "finding": "Prefetch can help or hurt depending on timing and interference.",
            "evidence": f"paired gaps={paired.get('paired_tool_gaps', '')}; faster pairs={paired.get('prefetch_faster_pairs', '')}; slower pairs={paired.get('prefetch_slower_pairs', '')}; avg gain={paired.get('avg_prefetch_gain_ms', '')} ms.",
            "why_it_matters": "This supports the hardware story: policy hints alone are not enough; movement needs deadline-aware scheduling, protection, and telemetry.",
        },
        {
            "finding": "Live pairing is useful, but not perfectly deterministic.",
            "evidence": "Pairs are matched by SWE-bench task index plus gap order inside that task.",
            "why_it_matters": "The two live agent runs can choose slightly different actions, so exact per-gap comparisons should be read alongside aggregate trends.",
        },
    ]


def timeline_summary(gaps: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps[:limit]:
        margin = as_float(row.get("prefetch_margin_ms"))
        overlap = as_float(row.get("prefetch_resume_overlap_ms")) or 0.0
        if margin is None:
            status = "no matched hint"
        elif margin >= 0:
            status = "prefetch finished before resume"
        else:
            status = "prefetch late"
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", ""),
                "tools": row.get("tool_names", ""),
                "tool_gap_ms": row.get("tool_gap_ms", ""),
                "prefetch_status": row.get("prefetch_status", ""),
                "prefetch_duration_ms": row.get("prefetch_duration_ms", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "prefetch_resume_overlap_ms": round(overlap, 3),
                "hint_h2d_events": row.get("direct_kv_h2d_events", ""),
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "replay_kv_started_before_hint_done": row.get("replay_kv_started_before_hint_done", ""),
                "timeline_verdict": status,
            }
        )
    return rows


def checkpoint_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps:
        margin = as_float(row.get("prefetch_margin_ms"))
        has_hint = row.get("prefetch_status") not in ("", "no_hint", None)
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "task_index": row.get("task_index", ""),
                "hint_was_submitted": "yes" if row.get("hint_id") else "no",
                "controller_started_prefetch": "yes" if row.get("prefetch_start_ms") not in ("", None) else "no",
                "controller_finished_prefetch": "yes" if row.get("prefetch_end_ms") not in ("", None) else "no",
                "prefetch_finished_before_resume": "yes" if margin is not None and margin >= 0 else "no" if has_hint else "",
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "hint_side_h2d_observed": "yes" if row.get("direct_kv_h2d_events") not in ("", "0", None) else "no",
                "replay_side_h2d_observed": "yes" if row.get("replay_kv_h2d_events") not in ("", "0", None) else "no",
                "replay_kv_started_before_hint_done": row.get("replay_kv_started_before_hint_done", ""),
                "resume_request_latency_ms": row.get("resume_latency_ms", ""),
            }
        )
    return rows


def session_observations(gaps: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps[:limit]:
        margin = as_float(row.get("prefetch_margin_ms"))
        if margin is None:
            status = "No matched hint"
            what = "This live gap did not have a matched controller prefetch attempt."
            deduction = "Good for observing the tool gap, but not an intervention example."
        elif margin >= 0:
            status = "Useful prefetch timing"
            what = f"The live software prefetch request completed {margin:.3f} ms before the real resume request."
            deduction = "The hint path met this deadline, so this is a success timing example."
        else:
            status = "Late prefetch"
            what = f"The live software prefetch request completed {abs(margin):.3f} ms after the real resume request started."
            deduction = "The runtime had the semantic hint, but the ordinary software/SGLang path did not finish in time."
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "status": status,
                "what_happened": what,
                "deduction_and_evidence": (
                    f"{deduction} tool gap={row.get('tool_gap_ms')} ms; "
                    f"prefetch duration={row.get('prefetch_duration_ms')} ms; "
                    f"resume request latency={row.get('resume_latency_ms')} ms; "
                    f"tools={row.get('tool_names')}; "
                    f"hint-side HtoD events={row.get('direct_kv_h2d_events', '')}; "
                    f"replay-side HtoD events={row.get('replay_kv_h2d_events', '')}."
                ),
            }
        )
    return rows


def interesting_kv_gaps(gaps: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    movement_rows = [
        row
        for row in gaps
        if has_events(row.get("direct_kv_h2d_events")) or has_events(row.get("replay_kv_h2d_events"))
    ]
    if len(movement_rows) >= limit:
        return movement_rows[:limit]
    seen = {id(row) for row in movement_rows}
    filler = [row for row in gaps if id(row) not in seen]
    return [*movement_rows, *filler[: max(0, limit - len(movement_rows))]]


def kv_movement_classification(row: dict[str, Any]) -> str:
    has_green = has_events(row.get("direct_kv_h2d_events"))
    has_cyan = has_events(row.get("replay_kv_h2d_events"))
    if has_green and has_cyan:
        return "cyan + green"
    if has_cyan:
        return "cyan only"
    if has_green:
        return "green only"
    return "no visible HtoD"


def cyan_green_diagnosis(row: dict[str, Any]) -> str:
    has_green = has_events(row.get("direct_kv_h2d_events"))
    has_cyan = has_events(row.get("replay_kv_h2d_events"))
    replay_h2d_start = as_float(row.get("replay_kv_h2d_start_ms"))
    direct_h2d_start = as_float(row.get("direct_kv_h2d_start_ms"))
    prefetch_start = as_float(row.get("prefetch_start_ms"))
    prefetch_end = as_float(row.get("prefetch_end_ms"))
    resume_start = as_float(row.get("resume_start_ms"))

    if has_cyan and not has_green:
        if prefetch_start is None:
            return "Replay needed KV movement, but no matched prefetch attempt started for this gap."
        if replay_h2d_start is not None and prefetch_start > replay_h2d_start:
            return "Replay-side KV movement began before the prefetch attempt started. The replay got there first."
        if replay_h2d_start is not None and prefetch_end is not None and prefetch_end > replay_h2d_start:
            return "The prefetch attempt was still running when replay-side KV movement began. Replay served itself before the hint path produced visible HtoD."
        return "Replay-side KV movement was visible, but the hint path did not produce attributed HtoD. Check hook matching, host KV availability, or already-resident KV."

    if has_green and not has_cyan:
        if resume_start is not None and prefetch_end is not None and prefetch_end <= resume_start:
            return "The hint path moved KV before the resume request, and the replay did not need visible replay-side HtoD."
        if resume_start is not None and direct_h2d_start is not None and direct_h2d_start > resume_start:
            return "The hint path moved KV after replay had already started; replay-side HtoD was not separately observed."
        return "The hint path produced visible direct KV HtoD; replay-side HtoD was not observed."

    if has_green and has_cyan:
        if replay_h2d_start is not None and direct_h2d_start is not None and replay_h2d_start < direct_h2d_start:
            return "Replay-side KV movement started before hint-side HtoD. This suggests the prefetch was too late for this resume."
        if replay_h2d_start is not None and prefetch_end is not None and replay_h2d_start < prefetch_end:
            return "Replay-side KV movement started before the full hint path finished. The two paths overlapped."
        return "Both hint-side and replay-side KV movement were observed. Inspect timing to see which path acted first."

    if row.get("prefetch_status") not in ("", "no_hint", None):
        return "The hint path ran, but no host-to-device KV movement was attributed to either hint or replay for this gap."
    return "No visible HtoD movement was attributed for this gap."


def kv_movement_diagnostic_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps:
        classification = kv_movement_classification(row)
        if classification == "no visible HtoD":
            continue
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", ""),
                "tools": row.get("tool_names", ""),
                "movement_class": classification,
                "tool_gap_ms": row.get("tool_gap_ms", ""),
                "hint_submitted_ms": row.get("hint_submitted_ms", ""),
                "prefetch_start_ms": row.get("prefetch_start_ms", ""),
                "prefetch_start_delay_ms": delta_ms(row.get("prefetch_start_ms"), row.get("hint_submitted_ms")),
                "prefetch_end_ms": row.get("prefetch_end_ms", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "hint_h2d_start_ms": row.get("direct_kv_h2d_start_ms", ""),
                "hint_h2d_end_ms": row.get("direct_kv_h2d_end_ms", ""),
                "hint_h2d_events": row.get("direct_kv_h2d_events", ""),
                "replay_start_ms": row.get("resume_start_ms", ""),
                "replay_h2d_start_ms": row.get("replay_kv_h2d_start_ms", ""),
                "replay_h2d_end_ms": row.get("replay_kv_h2d_end_ms", ""),
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "replay_h2d_before_hint_done": row.get("replay_kv_started_before_hint_done", ""),
                "diagnosis": cyan_green_diagnosis(row),
            }
        )
    return rows


def direct_hook_decision_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps:
        if row.get("prefetch_status") in ("", "no_hint", None):
            continue
        init_events = int(row.get("direct_kv_init_load_back_events") or 0)
        load_back_events = int(row.get("direct_kv_load_back_events") or 0)
        hicache_load_events = int(row.get("direct_kv_hicache_load_events") or 0)
        h2d_events = int(row.get("direct_kv_h2d_events") or 0)
        if h2d_events:
            decision = "direct load produced HtoD"
        elif init_events or load_back_events or hicache_load_events:
            decision = "load path entered but no HtoD attributed"
        else:
            decision = "no visible load-back path"
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "hint_id": row.get("hint_id", ""),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", ""),
                "tools": row.get("tool_names", ""),
                "prefetch_status": row.get("prefetch_status", ""),
                "prefetch_duration_ms": row.get("prefetch_duration_ms", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "init_load_back_events": init_events,
                "load_back_events": load_back_events,
                "hicache_load_events": hicache_load_events,
                "hint_h2d_events": h2d_events,
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "decision": decision,
                "likely_reason_if_no_hint_h2d": cyan_green_diagnosis(row) if not h2d_events else "",
            }
        )
    return rows


def request_details(run: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run["requests"][:limit]:
        rows.append(
            {
                "mode": run["mode"],
                "task_index": row.get("task_index", ""),
                "parent_run_id": row.get("parent_run_id", ""),
                "phase": row.get("phase", ""),
                "message_count": row.get("message_count", ""),
                "tool_count": row.get("tool_count", ""),
                "tool_names": row.get("tool_names", ""),
                "elapsed_ms": row.get("elapsed_ms", ""),
                "context_present": row.get("request_context_present", ""),
                "preview": short(row.get("content_preview", ""), 160),
            }
        )
    return rows


def render_markdown(
    mode_rows: list[dict[str, Any]],
    pair_summary_rows: list[dict[str, Any]],
    deductions: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Live Paired AgentBench Report",
        "",
        "This report compares a live no-prefetch AgentBench/SGLang run against a live prefetch-intervention run.",
        "",
        "## Performance Summary",
        "",
    ]
    for row in mode_rows:
        lines.append(
            f"- `{row['mode']}`: requests={row['analyzed_model_requests']}, tool_calls={row['total_tool_calls']}, "
            f"tool_gaps={row['observed_tool_gaps']}, avg_resume_latency_ms={row['avg_resume_request_latency_ms']}"
        )
    if pair_summary_rows:
        row = pair_summary_rows[0]
        lines.extend(
            [
                "",
                "## Paired Summary",
                "",
                f"- paired_tool_gaps: `{row.get('paired_tool_gaps')}`",
                f"- avg_prefetch_gain_ms: `{row.get('avg_prefetch_gain_ms')}`",
                f"- prefetch_faster_pairs: `{row.get('prefetch_faster_pairs')}`",
                f"- prefetch_slower_pairs: `{row.get('prefetch_slower_pairs')}`",
            ]
        )
    lines.extend(["", "## Key Deductions", ""])
    for row in deductions:
        lines.append(f"- **{row['finding']}** {row['evidence']} {row['why_it_matters']}")
    lines.extend(["", "## Paired Session Evidence", ""])
    lines.append("| pair_key | task | tools | no_prefetch_ms | live_prefetch_ms | gain_ms | prefetch_margin_ms | verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in pair_rows:
        lines.append(
            f"| {row.get('pair_key','')} | {row.get('task_index','')} | {row.get('tool_names','')} | "
            f"{row.get('no_prefetch_resume_request_latency_ms','')} | {row.get('prefetch_resume_request_latency_ms','')} | "
            f"{row.get('prefetch_gain_ms','')} | {row.get('prefetch_margin_ms','')} | {row.get('performance_verdict','')} |"
        )
    return "\n".join(lines) + "\n"


def css() -> str:
    return """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
    main { max-width: 1760px; margin: 0 auto; padding: 24px; }
    section, details.section-card { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px 24px; margin: 18px 0; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
    h1 { font-size: 34px; margin: 0 0 8px; }
    h2 { font-size: 26px; margin: 0 0 12px; }
    h3 { font-size: 18px; margin: 18px 0 8px; }
    p { color: #334155; line-height: 1.45; }
    a { color: #2563eb; text-decoration: none; }
    .note { background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 14px; color: #1e3a8a; }
    .warn { background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; color: #7c2d12; }
    .theme-summary { --theme: #1e3a8a; --theme-bg: #eff6ff; }
    .theme-setup { --theme: #2563eb; --theme-bg: #eff6ff; }
    .theme-guide { --theme: #475569; --theme-bg: #f1f5f9; }
    .theme-global { --theme: #dc2626; --theme-bg: #fef2f2; }
    .theme-clean { --theme: #15803d; --theme-bg: #f0fdf4; }
    .theme-clean-table { --theme: #65a30d; --theme-bg: #f7fee7; }
    .theme-profiled { --theme: #7e22ce; --theme-bg: #faf5ff; }
    .theme-directkv { --theme: #0e7490; --theme-bg: #ecfeff; }
    .theme-deductions { --theme: #b45309; --theme-bg: #fffbeb; }
    .theme-checkpoints { --theme: #ea580c; --theme-bg: #fff7ed; }
    .theme-observations { --theme: #0f766e; --theme-bg: #f0fdfa; }
    .theme-paired { --theme: #be123c; --theme-bg: #fff1f2; }
    .theme-reproduce { --theme: #0891b2; --theme-bg: #ecfeff; }
    .theme-appendix { --theme: #64748b; --theme-bg: #f8fafc; }
    section[class*="theme-"], details.section-card[class*="theme-"] { border-top: 5px solid var(--theme); }
    section[class*="theme-"] > h2, details.section-card summary h2 { border-left: 8px solid var(--theme); padding-left: 10px; color: var(--theme); }
    details.section-card summary { cursor: pointer; list-style: none; display: flex; align-items: center; gap: 8px; }
    details.section-card summary::-webkit-details-marker { display: none; }
    details.section-card summary h2 { margin: 0; }
    details.section-card summary h2::before { content: "▶"; display: inline-block; color: var(--theme); font-size: 18px; margin-right: 8px; transform: translateY(-1px); }
    details.section-card[open] summary h2::before { content: "▼"; }
    .section-color-legend { color: #475569; font-size: 14px; margin: 8px 0 12px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .timeline-stack { display: grid; grid-template-columns: 1fr; gap: 26px; }
    .timeline-panel { min-width: 0; }
    .setup-diagram { margin: 12px 0 18px; }
    .toc { display: flex; flex-wrap: wrap; gap: 10px 16px; }
    .toc a { background: var(--theme-bg, #f1f5f9); border: 1px solid #e2e8f0; border-left: 7px solid var(--theme, #64748b); border-radius: 6px; padding: 7px 10px; color: #0f172a; font-weight: 650; }
    .toc-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
    .toc-actions button { border: 1px solid #cbd5e1; background: #fff; color: #0f172a; border-radius: 6px; padding: 7px 10px; font-weight: 650; cursor: pointer; }
    .toc-actions button:hover { background: #f8fafc; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #f8fafc; }
    .card .label { color: #64748b; font-size: 13px; }
    .card .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th { text-align: left; background: #f1f5f9; color: #111827; padding: 9px 10px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; }
    td { padding: 9px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }
    code { background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }
    pre { white-space: pre-wrap; background: #0f172a; color: #e5e7eb; border-radius: 8px; padding: 12px; overflow: auto; }
    pre code { background: transparent; color: inherit; padding: 0; }
    svg text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }
    @media (max-width: 1000px) { .grid, .cards { grid-template-columns: 1fr; } }
    """


def metric_cards(mode_rows: list[dict[str, Any]], pair_summary_rows: list[dict[str, Any]]) -> str:
    by_mode = {row["mode"]: row for row in mode_rows}
    pref = by_mode.get("live_prefetch", {})
    paired = pair_summary_rows[0] if pair_summary_rows else {}
    cards = [
        ("paired tool gaps", paired.get("paired_tool_gaps", "")),
        ("avg live prefetch gain", f"{paired.get('avg_prefetch_gain_ms', '')} ms"),
        ("late prefetch pairs", paired.get("prefetch_late_pairs", "")),
        ("live tool calls", pref.get("total_tool_calls", "")),
    ]
    return "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{fmt(label)}</div><div class=\"value\">{fmt(value)}</div></div>"
        for label, value in cards
    ) + "</div>"


def setup_diagram_svg() -> str:
    boxes = [
        (50, 65, 190, 74, "Agent Task", "SWE-bench / DeepAgents"),
        (280, 65, 190, 74, "First Turn", "model builds KV"),
        (510, 65, 190, 74, "Tool Call", "read_file / run_tests"),
        (740, 65, 190, 74, "Tool Wait Gap", "prefetch opportunity"),
        (970, 65, 190, 74, "Hint / Prefetch", "try to prepare KV"),
        (1200, 65, 190, 74, "Resume Turn", "measure latency"),
    ]
    arrows = [
        (240, 102, 280, 102),
        (470, 102, 510, 102),
        (700, 102, 740, 102),
        (930, 102, 970, 102),
        (1160, 102, 1200, 102),
    ]
    parts = [
        '<svg viewBox="0 0 1440 210" width="100%" role="img" aria-label="Simple experiment setup flow diagram">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#334155"/>',
        "</marker>",
        "</defs>",
        '<rect x="20" y="25" width="1400" height="150" rx="10" fill="#f8fafc" stroke="#e5e7eb"/>',
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    for idx, (x, y, w, h, title, subtitle) in enumerate(boxes):
        fill = "#fff7ed" if idx in (3, 4) else "#eff6ff" if idx == 5 else "#ffffff"
        stroke = "#ea580c" if idx in (3, 4) else "#2563eb" if idx == 5 else "#cbd5e1"
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
                f'<text x="{x + w / 2}" y="{y + 27}" text-anchor="middle" font-size="15" font-weight="700">{html.escape(title)}</text>',
                f'<text x="{x + w / 2}" y="{y + 48}" text-anchor="middle" font-size="12" fill="#475569">{html.escape(subtitle)}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="720" y="192" text-anchor="middle" font-size="13" fill="#475569">Core question: was the right KV ready before the resume turn arrived?</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def experiment_setup_html(mode_rows: list[dict[str, Any]], pair_summary_rows: list[dict[str, Any]]) -> str:
    by_mode = {row["mode"]: row for row in mode_rows}
    no_prefetch = by_mode.get("no_prefetch", {})
    live_prefetch = by_mode.get("live_prefetch", {})
    paired = pair_summary_rows[0] if pair_summary_rows else {}
    setup_rows = [
        {"part": "1. Request source", "simple meaning": "Real SWE-bench / DeepAgents tasks create model turns.", "example": "SWE-bench task -> DeepAgents -> SGLang model request"},
        {"part": "2. Tool wait window", "simple meaning": "The agent calls a tool, then pauses. That pause is the chance to prefetch KV.", "example": "read_file(), grep(), or run_tests() is running"},
        {"part": "3. Resume request", "simple meaning": "The tool returns, then the agent asks the model to continue.", "example": "tests failed with error X; what should I do next?"},
    ]
    mode_rows_simple = [
        {"mode": "No prefetch", "what happens": "The system waits until the resume turn arrives, then SGLang handles KV reuse/load normally."},
        {"mode": "Live prefetch", "what happens": "During the tool wait, the hint path tries to prepare KV before the resume turn arrives."},
    ]
    metric_rows = [
        {"metric": "tool gap", "meaning": "Time between a tool-call response and the next model request from the same live agent run."},
        {"metric": "prefetch duration", "meaning": "How long the live software prefetch/controller path took to complete."},
        {"metric": "prefetch margin", "meaning": "Whether prefetch finished before or after the real resume request boundary."},
        {"metric": "resume request latency", "meaning": "End-to-end latency of the next model request after the tool gap."},
        {"metric": "late prefetch count", "meaning": "How often the hint path missed the available tool-gap window."},
    ]
    observation_rows = [
        {"observation": "The traffic was live agent traffic, not synthetic prompts.", "evidence": f"{no_prefetch.get('analyzed_model_requests', '')} no-prefetch requests and {live_prefetch.get('analyzed_model_requests', '')} live-prefetch requests were analyzed."},
        {"observation": "The run produced real tool-call gaps.", "evidence": f"{no_prefetch.get('observed_tool_gaps', '')} no-prefetch gaps and {live_prefetch.get('observed_tool_gaps', '')} live-prefetch gaps were observed."},
        {"observation": "Tool gaps were often very short.", "evidence": f"Median live-prefetch tool gap was {live_prefetch.get('median_tool_gap_ms', '')} ms."},
        {"observation": "Software prefetch was usually too slow for those windows.", "evidence": f"{live_prefetch.get('late_prefetch_attempts', '')} late prefetch attempts; average prefetch duration was {live_prefetch.get('avg_prefetch_duration_ms', '')} ms."},
        {"observation": "Hints alone did not guarantee a win.", "evidence": f"Paired gaps={paired.get('paired_tool_gaps', '')}; faster pairs={paired.get('prefetch_faster_pairs', '')}; slower pairs={paired.get('prefetch_slower_pairs', '')}."},
    ]
    return f"""
    <div class="setup-diagram">{setup_diagram_svg()}</div>
    <h3>Simple Setup</h3>
    {table_html(setup_rows, ["part", "simple meaning", "example"])}
    <h3>Modes Compared</h3>
    {table_html(mode_rows_simple, ["mode", "what happens"])}
    <h3>What Was Measured</h3>
    {table_html(metric_rows, ["metric", "meaning"])}
    <h3>What Was Observed</h3>
    {table_html(observation_rows, ["observation", "evidence"])}
    <h3>Why This Supports The Hardware Proposal</h3>
    <p>Current GPU/runtime data-movement paths can move memory, but they do not know that a transfer is urgent KV for a soon-resuming agent session. This experiment shows that when prefetch is routed through ordinary software and SGLang request paths, it can miss short live tool gaps. A hint-aware hardware/runtime path could make these movements more predictable by prioritizing urgent KV, protecting prefetched KV, and exposing telemetry for late or wasted prefetches.</p>
    """


def timeline_guide_html(profiled_available: bool) -> str:
    step_rows = [
        {"step": "1. Ask model what to do", "timeline color": "blue bar", "simple meaning": "model turn before the tool call"},
        {"step": "2. Model says to call a tool", "timeline color": "end of blue bar", "simple meaning": "the model turn hands work to a tool"},
        {"step": "3. Tool runs", "timeline color": "gray bar", "simple meaning": "read_file(), grep(), run_tests(), etc. is happening"},
        {"step": "4. Agent waits", "timeline color": "gray bar", "simple meaning": "this wait is the prefetch opportunity"},
        {"step": "5. Tool returns", "timeline color": "black vertical line", "simple meaning": "the next model turn is due"},
        {"step": "6. Agent asks model again", "timeline color": "red bar", "simple meaning": "resume request after the tool result"},
        {"step": "During steps 3/4, if prefetch is enabled", "timeline color": "purple bar", "simple meaning": "our software prefetch attempt runs during the wait"},
        {"step": "Inside the purple window, if KV moved", "timeline color": "green bar", "simple meaning": "direct SGLang KV host-to-device movement was observed for that hint"},
        {"step": "Inside the red replay request, if KV moved", "timeline color": "cyan bar", "simple meaning": "the real replay request itself performed host-to-device KV movement"},
    ]
    rows = [
        {
            "color": "blue",
            "meaning": "Initial model turn",
            "simple description": "The agent asks the model what to do next. This request may produce a tool call such as read_file(), grep(), edit_file(), or execute().",
        },
        {
            "color": "gray",
            "meaning": "Tool wait window",
            "simple description": "The tool or harness is running, so the model is idle for this session. This pause is the opportunity to prepare that session's KV before the agent resumes.",
        },
        {
            "color": "purple",
            "meaning": "Prefetch attempt window",
            "simple description": "This includes detecting that the tool call finished, creating a hint for that agent/session, calling our direct SGLang KV hook, letting SGLang check whether host-side KV exists, and if needed, asking SGLang to move KV back to GPU memory.",
        },
        {
            "color": "green",
            "meaning": "Direct KV host-to-device movement",
            "simple description": "This is evidence that KV load/copy work was observed for the hint. In simple words: this is the closest timeline bar to 'KV pages were brought back toward GPU memory'.",
        },
        {
            "color": "cyan",
            "meaning": "Replay-side KV host-to-device movement",
            "simple description": "This is evidence from the normal replay request path. It means the resumed request itself had to move KV from host-side memory toward GPU memory while serving the real agent turn.",
        },
        {
            "color": "black",
            "meaning": "Replay due boundary",
            "simple description": "This is when the tool result is ready and the next model turn should be able to start. If purple or green finishes after this line, the prefetch path was late for that resume.",
        },
        {
            "color": "red",
            "meaning": "Replay request",
            "simple description": "The agent asks the model to continue after the tool result. This is the request we want to speed up by having the right KV ready before it arrives.",
        },
        {
            "color": "yellow",
            "meaning": "First token marker",
            "simple description": "Synthetic reports may show this as the point where the replay request starts producing output. It helps separate request arrival from first-token latency.",
        },
    ]
    note = "Purple means the prefetch attempt ran. Green means that attempt had attributed direct KV host-to-device movement. Purple without green means the hint ran, but no matching HtoD load was observed for that displayed gap."
    return f"""
    <p class="note">{html.escape(note)}</p>
    <p class="note">The timeline uses a symlog full-replay view. It compresses very long replay requests so red bars can extend to their true end while the important activity around <code>0 ms</code> remains visible.</p>
    <h3>One Row In Plain English</h3>
    <p>Each timeline row is one tool-wait episode: model turn, tool call, wait, optional prefetch, then model resume.</p>
    {table_html(step_rows, ["step", "timeline color", "simple meaning"])}
    <h3>Color Legend</h3>
    {table_html(rows, ["color", "meaning", "simple description"])}
    """


def prefetch_margin_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps:
        margin = as_float(row.get("prefetch_margin_ms"))
        if margin is None:
            continue
        duration = as_float(row.get("prefetch_duration_ms"))
        if margin >= 50:
            verdict = "early"
        elif margin >= 0:
            verdict = "barely early"
        elif margin > -50:
            verdict = "near miss"
        else:
            verdict = "late"
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", ""),
                "tools": row.get("tool_names", ""),
                "tool_gap_ms": as_float(row.get("tool_gap_ms")),
                "prefetch_duration_ms": duration,
                "prefetch_margin_ms": margin,
                "verdict": verdict,
            }
        )
    return rows


def prefetch_margin_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    margins = [float(row["prefetch_margin_ms"]) for row in rows]
    durations = [
        float(row["prefetch_duration_ms"])
        for row in rows
        if row.get("prefetch_duration_ms") is not None
    ]
    early = [value for value in margins if value >= 0]
    late = [value for value in margins if value < 0]
    return [
        {
            "prefetch_attempts": len(rows),
            "finished_before_resume": len(early),
            "late": len(late),
            "late_pct": round(len(late) * 100.0 / len(rows), 2) if rows else "",
            "median_margin_ms": round(median(margins), 3) if margins else "",
            "worst_lateness_ms": round(abs(min(late)), 3) if late else "",
            "best_early_margin_ms": round(max(early), 3) if early else "",
            "avg_prefetch_duration_ms": avg(durations),
        }
    ]


def prefetch_margin_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [
        ("> +500 ms early", lambda value: value > 500),
        ("+100 to +500 ms early", lambda value: 100 < value <= 500),
        ("0 to +100 ms early", lambda value: 0 <= value <= 100),
        ("0 to -100 ms late", lambda value: -100 <= value < 0),
        ("-100 to -500 ms late", lambda value: -500 <= value < -100),
        ("< -500 ms late", lambda value: value < -500),
    ]
    total = len(rows)
    output: list[dict[str, Any]] = []
    for label, predicate in buckets:
        count = sum(1 for row in rows if predicate(float(row["prefetch_margin_ms"])))
        output.append(
            {
                "bucket": label,
                "sessions": count,
                "pct": round(count * 100.0 / total, 2) if total else "",
            }
        )
    return output


def symlog_value(value: float, linear_width: float = 50.0) -> float:
    if value == 0:
        return 0.0
    return math.copysign(math.log1p(abs(value) / linear_width), value)


def symlog_tick_values(min_margin: float, max_margin: float) -> list[float]:
    candidates = [
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


def build_prefetch_margin_dot_plot(rows: list[dict[str, Any]], scale: str = "linear") -> str:
    if not rows:
        return "<p>No matched prefetch attempts were available for the global margin plot.</p>"
    if scale not in {"linear", "symlog"}:
        raise ValueError(f"unsupported margin plot scale: {scale}")
    width = 1480
    height = 520
    left = 86
    right = 34
    top = 56
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    margins = [float(row["prefetch_margin_ms"]) for row in rows]
    min_margin = min(margins)
    max_margin = max(margins)
    pad = max(50.0, (max_margin - min_margin) * 0.08)
    y_min = min(min_margin - pad, -50.0)
    y_max = max(max_margin + pad, 50.0)
    if scale == "symlog":
        y_min = min(y_min, -50.0)
        y_max = max(y_max, 50.0)
        scaled_min = symlog_value(y_min)
        scaled_max = symlog_value(y_max)
    else:
        scaled_min = y_min
        scaled_max = y_max

    def x_pos(index: int) -> float:
        if len(rows) <= 1:
            return left + plot_w / 2
        return left + index * plot_w / (len(rows) - 1)

    def y_pos(value: float) -> float:
        scaled = symlog_value(value) if scale == "symlog" else value
        return top + (scaled_max - scaled) * plot_h / (scaled_max - scaled_min)

    zero_y = y_pos(0.0)
    scale_label = "symlog" if scale == "symlog" else "linear"
    axis_label = "prefetch margin ms (symlog)" if scale == "symlog" else "prefetch margin ms"
    parts = [
        f'<svg viewBox="0 0 1480 520" width="100%" role="img" aria-label="Global prefetch margin dot plot {scale_label} view">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{left + plot_w - 8}" y="{zero_y - 8:.1f}" text-anchor="end" font-size="12" font-weight="700">0 ms deadline</text>',
        f'<text x="18" y="265" transform="rotate(-90 18 265)" text-anchor="middle" font-size="13" font-weight="700">{axis_label}</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="13" font-weight="700">live tool-gap order</text>',
        '<text x="94" y="34" font-size="13" fill="#166534" font-weight="700">above line = finished before resume</text>',
        '<text x="340" y="34" font-size="13" fill="#b91c1c" font-weight="700">below line = late prefetch</text>',
    ]

    tick_values = symlog_tick_values(y_min, y_max) if scale == "symlog" else [y_min, y_min / 2, 0.0, y_max / 2, y_max]
    seen_ticks: set[int] = set()
    for value in tick_values:
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
        margin = float(row["prefetch_margin_ms"])
        duration = row.get("prefetch_duration_ms")
        color = "#16a34a" if margin >= 50 else "#84cc16" if margin >= 0 else "#f97316" if margin > -50 else "#dc2626"
        radius = 5
        if isinstance(duration, (int, float)):
            radius = max(4, min(9, 4 + float(duration) / 350.0))
        x = x_pos(index)
        y = y_pos(margin)
        title = (
            f"{row.get('session_id')} | task={row.get('task_index')} gap={row.get('gap_order_in_task')} | "
            f"margin={margin:.3f} ms | duration={duration if duration is not None else 'n/a'} ms | "
            f"tool_gap={row.get('tool_gap_ms')} ms | tools={row.get('tools')}"
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" opacity="0.86" stroke="#ffffff" stroke-width="1.4">'
            f'<title>{html.escape(title)}</title></circle>'
        )

    legend = [
        ("early", "#16a34a"),
        ("barely early", "#84cc16"),
        ("near miss", "#f97316"),
        ("late", "#dc2626"),
    ]
    lx = left
    ly = height - 50
    for label, color in legend:
        parts.append(f'<circle cx="{lx}" cy="{ly}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{lx + 12}" y="{ly + 4}" font-size="12">{html.escape(label)}</text>')
        lx += 145
    parts.append("</svg>")
    return "\n".join(parts)


def global_prefetch_margin_html(gaps: list[dict[str, Any]]) -> str:
    rows = prefetch_margin_rows(gaps)
    summary = prefetch_margin_summary(rows)
    buckets = prefetch_margin_bucket_rows(rows)
    detail = rows[:40]
    return f"""
    <p>This chart compresses every matched live prefetch attempt into one dot. The y-axis is the prefetch margin: positive means the prefetch finished before the resume request; negative means it finished after the agent already resumed.</p>
    {table_html(summary)}
    <h3>Symlog View</h3>
    <p>This view compresses the y-axis with a symmetric log-style scale so both small near-deadline misses and very large late prefetches are easy to see. Above zero means early; below zero means late.</p>
    <div class="setup-diagram">{build_prefetch_margin_dot_plot(rows, scale="symlog")}</div>
    <h3>Margin Buckets</h3>
    {table_html(buckets)}
    <h3>First 40 Points Behind The Plot</h3>
    {table_html(detail, ["session_id", "task_index", "gap_order_in_task", "tools", "tool_gap_ms", "prefetch_duration_ms", "prefetch_margin_ms", "verdict"])}
    """


def per_task_prefetch_breakdown(run: dict[str, Any]) -> list[dict[str, Any]]:
    gaps_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run["gaps"]:
        gaps_by_task[str(row.get("task_index", ""))].append(row)

    rows: list[dict[str, Any]] = []
    for task in run.get("task_rows", []):
        task_index = str(task.get("task_index", ""))
        gaps = gaps_by_task.get(task_index, [])
        tools: Counter[str] = Counter()
        for gap in gaps:
            for name in str(gap.get("tool_names") or "").split(","):
                name = name.strip()
                if name:
                    tools[name] += 1
        tool_summary = ", ".join(f"{name}:{count}" for name, count in tools.most_common()) or "-"
        status = str(task.get("status", ""))
        rows.append(
            {
                "task_index": task_index,
                "task_status": status,
                "status_meaning": "completed" if status == "0" else "failed/no complete result" if status else "",
                "tool_gaps_prefetch_attempts": len(gaps),
                "tool_calls_in_gaps": sum(int(gap.get("tool_call_count") or 0) for gap in gaps),
                "tools": tool_summary,
            }
        )
    return rows


SECTION_THEMES = {
    "summary": "theme-summary",
    "setup": "theme-setup",
    "timeline-guide": "theme-guide",
    "global-prefetch": "theme-global",
    "timelines": "theme-clean",
    "live-direct": "theme-profiled",
    "performance": "theme-clean-table",
    "profiled": "theme-profiled",
    "direct-kv": "theme-directkv",
    "deductions": "theme-deductions",
    "checkpoints": "theme-checkpoints",
    "observations": "theme-observations",
    "paired": "theme-paired",
    "reproduce": "theme-reproduce",
    "appendix": "theme-appendix",
}


def toc_html(items: list[tuple[str, str]]) -> str:
    links = "".join(
        f'<a class="{SECTION_THEMES.get(anchor, "theme-appendix")}" href="#{anchor}">{html.escape(label)}</a>'
        for anchor, label in items
    )
    return (
        '<p class="section-color-legend">Colors group sections by evidence type: setup, clean performance, '
        'profiled mechanism, interpretation, and appendix.</p>'
        f'<div class="toc">{links}</div>'
        '<div class="toc-actions"><button type="button" data-action="expand-all">Expand All</button>'
        '<button type="button" data-action="collapse-all">Collapse All</button></div>'
    )


def report_script() -> str:
    return """
<script>
document.addEventListener("DOMContentLoaded", function () {
  const cards = Array.from(document.querySelectorAll("details.section-card"));
  document.querySelectorAll(".toc a[href^='#']").forEach(function (link) {
    link.addEventListener("click", function () {
      const id = link.getAttribute("href").slice(1);
      const target = document.getElementById(id);
      if (target && target.tagName.toLowerCase() === "details") {
        target.open = true;
      }
    });
  });
  document.querySelectorAll("[data-action='expand-all']").forEach(function (button) {
    button.addEventListener("click", function () {
      cards.forEach(function (card) { card.open = true; });
    });
  });
  document.querySelectorAll("[data-action='collapse-all']").forEach(function (button) {
    button.addEventListener("click", function () {
      cards.forEach(function (card) { card.open = false; });
    });
  });
});
</script>
"""


def code_block(text: str) -> str:
    return f"<pre><code>{html.escape(text.strip())}</code></pre>"


def reproduce_live_report_html() -> str:
    run_new = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=manager_demo_1 \
UPDATE_LATEST=0 \
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
START_INDEX=0 \
END_INDEX=15 \
MAX_STEPS=10 \
AGENTBENCH_ROOT=~/kv_cache_offloading \
bash scripts/run_labeled_live_master_report.sh
"""
    build_existing = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=manager_demo_1_rebuild \
UPDATE_LATEST=0 \
NO_PREFETCH_ROOT=artifacts/results/<run>/no_prefetch_live \
PREFETCH_ROOT=artifacts/results/<run>/live_prefetch \
bash scripts/build_labeled_live_master_report.sh
"""
    update_latest = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=manager_demo_latest \
UPDATE_LATEST=1 \
START_INDEX=0 \
END_INDEX=15 \
MAX_STEPS=10 \
bash scripts/run_labeled_live_master_report.sh Qwen/Qwen2.5-Coder-7B-Instruct
"""
    return "\n".join(
        [
            "<p>This section gives copy-paste commands for reproducing the real SWE-bench / DeepAgents master report without overwriting the current latest report by default.</p>",
            "<h3>Run A New Labeled Live Experiment</h3>",
            code_block(run_new),
            "<p>Output:</p>",
            code_block("artifacts/results/labeled/live/manager_demo_1/master_report.html"),
            "<h3>Rebuild A Labeled Report From Existing Runs</h3>",
            "<p>Use this when the no-prefetch and live-prefetch folders already exist and you only want to regenerate the HTML/report tables.</p>",
            code_block(build_existing),
            "<h3>Deliberately Refresh The Latest Master Report</h3>",
            "<p>Only use this when you want to replace <code>artifacts/results/latest_master_report.html</code>.</p>",
            code_block(update_latest),
        ]
    )


def direct_kv_evidence_html(
    pref_run: dict[str, Any],
    movement_diagnostics: list[dict[str, Any]],
    hook_decisions: list[dict[str, Any]],
) -> str:
    summary = pref_run.get("direct_kv_summary") or []
    evidence = pref_run.get("direct_kv_evidence") or []
    events = pref_run.get("direct_kv_events") or []
    replay_evidence = pref_run.get("replay_kv_evidence") or []
    report_html = pref_run.get("direct_kv_report_html") or ""
    if not summary and not evidence and not replay_evidence:
        return "\n".join(
            [
                '<p class="warn">No Milestone 26 direct-KV evidence files were found for this prefetch run.</p>',
                "<p>Run <code>scripts/run_milestone26_live_direct_kv_load_intervention.sh</code> or <code>scripts/run_milestone26_live_paired_direct_kv_report.sh</code> to enable SGLang KV trace hooks and generate this section.</p>",
            ]
        )
    return "\n".join(
        [
            "<p>This section checks whether the live tool-call hint triggered SGLang's real direct KV load-back path. It looks for <code>hiradix.init_load_back</code>, <code>hiradix.load_back</code>, <code>hicache.load</code>, and host-to-device copy telemetry attributed to each live hint.</p>",
            f"<p>Detailed report: <code>{fmt(report_html)}</code></p>" if report_html else "",
            "<h3>Direct KV Load Summary</h3>",
            table_html(summary),
            "<h3>KV Movement Diagnostic Table</h3>",
            "<p>This table focuses on gaps with either green hint-side HtoD or cyan replay-side HtoD. It explains whether replay got there first, the hint produced useful HtoD, or both paths moved KV.</p>",
            table_html(
                movement_diagnostics,
                [
                    "session_id",
                    "task_index",
                    "gap_order_in_task",
                    "tools",
                    "movement_class",
                    "tool_gap_ms",
                    "prefetch_start_delay_ms",
                    "prefetch_margin_ms",
                    "hint_h2d_events",
                    "replay_h2d_events",
                    "replay_h2d_before_hint_done",
                    "diagnosis",
                ],
                limit=120,
            ),
            "<h3>Direct Hook Decision Table</h3>",
            "<p>This table checks what the direct hook appeared to do for each matched hint. It helps separate true late prefetches from possible hook/attribution issues.</p>",
            table_html(hook_decisions, limit=120),
            "<h3>Per-Hint Direct KV Evidence</h3>",
            table_html(evidence, limit=80),
            "<h3>Matched KV/Copy Events</h3>",
            table_html(events, limit=80),
            "<h3>Replay-Side KV Demand Evidence</h3>",
            "<p>This table checks the real replay request path, not the hint path. If replay-side HtoD is observed, the resumed request itself performed host-to-device KV work during serving.</p>",
            table_html(replay_evidence, limit=80),
        ]
    )


def render_html(
    no_run: dict[str, Any],
    pref_run: dict[str, Any],
    mode_rows: list[dict[str, Any]],
    pair_summary_rows: list[dict[str, Any]],
    deductions: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    max_timeline_gaps: int,
) -> str:
    pref_gaps = pref_run["gaps"]
    no_gaps = no_run["gaps"]
    interesting_pref_gaps = interesting_kv_gaps(pref_gaps, max_timeline_gaps)
    checkpoint = checkpoint_rows(pref_gaps)
    observations = session_observations(interesting_pref_gaps, max_timeline_gaps)
    timeline_rows = timeline_summary(interesting_pref_gaps, max_timeline_gaps)
    movement_diagnostics = kv_movement_diagnostic_rows(pref_gaps)
    hook_decisions = direct_hook_decision_rows(pref_gaps)
    detail_rows = [
        {
            "mode": "no_prefetch",
            "root": no_run["root"],
            "proxy_jsonl": no_run["proxy_jsonl"],
            "task_index_csv": no_run["task_index_csv"],
        },
        {
            "mode": "live_prefetch",
            "root": pref_run["root"],
            "proxy_jsonl": pref_run["proxy_jsonl"],
            "task_index_csv": pref_run["task_index_csv"],
            "hint_log": pref_run["hint_log"],
            "controller_log": pref_run["controller_log"],
        },
    ]
    toc = [
        ("summary", "Summary"),
        ("setup", "Experiment Setup"),
        ("timeline-guide", "How To Read Timelines"),
        ("global-prefetch", "Global Prefetch Margin"),
        ("timelines", "Clean Performance Timelines"),
        ("performance", "Clean Performance Tables"),
        ("profiled", "Profiled Mechanism Evidence"),
        ("direct-kv", "Direct KV Load Evidence"),
        ("deductions", "Key Deductions"),
        ("checkpoints", "Prefetch Checkpoints"),
        ("observations", "Key Observations Per Session"),
        ("paired", "Paired Session Evidence"),
        ("reproduce", "Reproduce This Report"),
        ("appendix", "Appendix"),
    ]
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Paired AgentBench Report</title>
  <style>{css()}</style>
</head>
<body>
<main>
  <section>
    <h1>Live Paired AgentBench Report</h1>
    <p>This is the live version of the paired evidence report. The request generator is real SWE-bench / Deep Agents traffic; the backend is direct SGLang; the comparison is no-prefetch versus live software prefetch intervention.</p>
    <p class="note">Important metric note: this report uses full resume request latency captured by the proxy, not streaming TTFT. It still shows whether the next real agent turn became faster or slower after the prefetch intervention.</p>
    <h2>Table of Contents</h2>
    {toc_html(toc)}
  </section>

  <details id="summary" class="section-card theme-summary" open>
    <summary><h2>Summary</h2></summary>
    <p>This report uses real SWE-bench / DeepAgents traffic. It answers whether the live software hint path finishes before the next real agent turn resumes.</p>
    {metric_cards(mode_rows, pair_summary_rows)}
  </details>

  <details id="setup" class="section-card theme-setup">
    <summary><h2>Experiment Setup And Manager Summary</h2></summary>
    <p>This section is intended for slide-building: it shows the live request path, the hint/prefetch path, how the experiment was conducted, and the main evidence collected.</p>
    {experiment_setup_html(mode_rows, pair_summary_rows)}
  </details>

  <details id="timeline-guide" class="section-card theme-guide">
    <summary><h2>How To Read The Timelines</h2></summary>
    <p>The timelines are the primary visual evidence. Tables below each timeline provide the exact numbers behind the picture.</p>
    {timeline_guide_html(profiled_available=False)}
  </details>

  <details id="global-prefetch" class="section-card theme-global">
    <summary><h2>Global Prefetch Margin</h2></summary>
    <p>This is the high-level view across all live prefetch attempts. It answers whether most hints finished early enough or arrived late.</p>
    {global_prefetch_margin_html(pref_gaps)}
    <h3>Per-Task Prefetch Breakdown</h3>
    <p>This table explains where the plotted dots came from. One launched task can produce many tool gaps, and each matched live-prefetch tool gap becomes one dot in the global chart.</p>
    {table_html(per_task_prefetch_breakdown(pref_run))}
  </details>

  <details id="timelines" class="section-card theme-clean">
    <summary><h2>A. Clean Performance Timelines</h2></summary>
    <p class="note">Profiler is off. Use this section for live request-flow and latency claims.</p>
    <p>Blue is a live model turn that emitted tool calls. Gray is the observed tool/harness gap. Red is the next live model turn. Purple appears only in the live-prefetch run and shows the software prefetch request. Green appears when that hint has attributed direct KV host-to-device movement. Cyan appears when the real replay request itself performed replay-side KV host-to-device movement.</p>
    <h3>Symlog Full Replay View</h3>
    <p>The symlog view compresses large distances far from <code>0 ms</code>, but keeps small timings near replay due visible. Use this when you want to see the full red replay duration together with the purple, green, and cyan KV activity.</p>
    <p class="note">The live-prefetch timeline below prioritizes gaps with visible KV movement: cyan replay-side HtoD or green hint-side HtoD. This keeps the main visual focused on the agentic phases where host-to-device KV block movement actually appeared.</p>
    <div class="timeline-stack">
      <div class="timeline-panel">
        <h3>No Prefetch</h3>
        {build_expanded_gap_timeline_svg(no_gaps, max_timeline_gaps, show_prefetch_legend=False, scale="symlog")}
      </div>
      <div class="timeline-panel">
        <h3>Live Prefetch Intervention: KV Movement Prioritized</h3>
        {build_expanded_gap_timeline_svg(interesting_pref_gaps, max_timeline_gaps, scale="symlog")}
      </div>
    </div>
  </details>

  <details id="performance" class="section-card theme-clean-table">
    <summary><h2>A.1 Clean Performance Tables</h2></summary>
    <p>These tables provide the exact request counts, tool-gap counts, latency values, and paired aggregate numbers behind the clean timelines.</p>
    <h3>By Mode</h3>
    {table_html(mode_rows)}
    <h3>Paired Aggregate</h3>
    {table_html(pair_summary_rows)}
  </details>

  <details id="profiled" class="section-card theme-profiled">
    <summary><h2>B. Profiled Mechanism Evidence</h2></summary>
    <p class="warn">Not available yet for the real SWE-bench / DeepAgents master report. This report currently shows clean live request timing and live hint-controller timing, but it does not yet include torch-profiler CUDA HtoD attribution for live SWE-bench traffic.</p>
    <p>After we add the live profiled attribution run, this section will show dark-green CUDA HtoD copy bars, KV/copy telemetry, replay reload evidence, and checkpoint tables for the real traffic path.</p>
  </details>

  <details id="direct-kv" class="section-card theme-directkv">
    <summary><h2>B.1 Direct KV Load Evidence</h2></summary>
    {direct_kv_evidence_html(pref_run, movement_diagnostics, hook_decisions)}
  </details>

  <details id="deductions" class="section-card theme-deductions">
    <summary><h2>Key Deductions</h2></summary>
    {table_html(deductions, ["finding", "evidence", "why_it_matters"])}
  </details>

  <details id="checkpoints" class="section-card theme-checkpoints">
    <summary><h2>Prefetch Checkpoints</h2></summary>
    <p>These checkpoints make the live intervention path explicit: hint submitted, controller started, controller finished, and whether it finished before resume.</p>
    {table_html(checkpoint[:max_timeline_gaps])}
  </details>

  <details id="observations" class="section-card theme-observations">
    <summary><h2>Key Observations Per Session</h2></summary>
    {table_html(observations, ["session_id", "status", "what_happened", "deduction_and_evidence"])}
  </details>

  <details id="paired" class="section-card theme-paired">
    <summary><h2>Paired Session Evidence</h2></summary>
    {table_html(pair_rows)}
  </details>

  <details id="reproduce" class="section-card theme-reproduce">
    <summary><h2>Reproduce This Report</h2></summary>
    {reproduce_live_report_html()}
  </details>

  <details id="appendix" class="section-card theme-appendix">
    <summary><h2>Appendix: Detailed Evidence</h2></summary>
    <h3>Timeline Summary</h3>
    {table_html(timeline_rows)}
    <h3>Input Runs</h3>
    {table_html(detail_rows)}
    <h3>Live Request Details</h3>
    {table_html(request_details(pref_run, 80))}
  </details>
</main>
{report_script()}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired live AgentBench/SGLang report.")
    parser.add_argument("--no-prefetch-root", type=Path, required=True)
    parser.add_argument("--prefetch-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--max-timeline-gaps", type=int, default=16)
    parser.add_argument("--include-preflight", action="store_true")
    args = parser.parse_args()

    no_run = load_live_run(args.no_prefetch_root, "no_prefetch", args.include_preflight)
    pref_run = load_live_run(args.prefetch_root, "live_prefetch", args.include_preflight)
    mode_rows = [mode_summary(no_run), mode_summary(pref_run)]
    pair_rows = pair_gaps(no_run["gaps"], pref_run["gaps"])
    pair_summary_rows = paired_summary(pair_rows)
    deductions = key_deductions(mode_rows, pair_summary_rows)
    movement_diagnostics = kv_movement_diagnostic_rows(pref_run["gaps"])
    hook_decisions = direct_hook_decision_rows(pref_run["gaps"])
    interesting_pref_gaps = interesting_kv_gaps(pref_run["gaps"], 10_000)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "live_paired_mode_summary.csv", mode_rows)
    write_csv(args.out_dir / "live_paired_summary.csv", pair_summary_rows)
    write_csv(args.out_dir / "live_paired_key_deductions.csv", deductions)
    write_csv(args.out_dir / "live_paired_session_evidence.csv", pair_rows)
    write_csv(args.out_dir / "live_prefetch_checkpoint_results.csv", checkpoint_rows(pref_run["gaps"]))
    write_csv(args.out_dir / "live_prefetch_timeline_summary.csv", timeline_summary(pref_run["gaps"], 10_000))
    write_csv(args.out_dir / "live_prefetch_session_observations.csv", session_observations(pref_run["gaps"], 10_000))
    write_csv(args.out_dir / "live_prefetch_request_details.csv", request_details(pref_run, 10_000))
    write_csv(args.out_dir / "live_prefetch_kv_movement_gaps.csv", interesting_pref_gaps)
    write_csv(args.out_dir / "live_prefetch_kv_movement_diagnostics.csv", movement_diagnostics)
    write_csv(args.out_dir / "live_prefetch_direct_hook_decisions.csv", hook_decisions)
    write_csv(args.out_dir / "live_no_prefetch_tool_gaps.csv", no_run["gaps"])
    write_csv(args.out_dir / "live_prefetch_tool_gaps.csv", pref_run["gaps"])
    write_csv(args.out_dir / "live_direct_kv_load_summary.csv", pref_run.get("direct_kv_summary", []))
    write_csv(args.out_dir / "live_direct_kv_load_evidence.csv", pref_run.get("direct_kv_evidence", []))
    write_csv(args.out_dir / "live_direct_kv_load_events.csv", pref_run.get("direct_kv_events", []))
    write_csv(args.out_dir / "live_replay_kv_demand_evidence.csv", pref_run.get("replay_kv_evidence", []))

    report = {
        "no_prefetch": {key: value for key, value in no_run.items() if key not in {"requests", "gaps"}},
        "live_prefetch": {key: value for key, value in pref_run.items() if key not in {"requests", "gaps"}},
        "mode_summary": mode_rows,
        "paired_summary": pair_summary_rows,
        "key_deductions": deductions,
        "paired_session_evidence": pair_rows,
        "no_prefetch_gaps": no_run["gaps"],
        "prefetch_gaps": pref_run["gaps"],
        "prefetch_kv_movement_gaps": interesting_pref_gaps,
        "prefetch_kv_movement_diagnostics": movement_diagnostics,
        "prefetch_direct_hook_decisions": hook_decisions,
        "replay_kv_evidence": pref_run.get("replay_kv_evidence", []),
    }
    json_path = args.out_dir / "live_paired_agentbench_report.json"
    md_path = args.out_dir / "live_paired_agentbench_report.md"
    html_path = args.out_dir / "live_paired_agentbench_report.html"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(mode_rows, pair_summary_rows, deductions, pair_rows), encoding="utf-8")
    html_path.write_text(
        render_html(no_run, pref_run, mode_rows, pair_summary_rows, deductions, pair_rows, args.max_timeline_gaps),
        encoding="utf-8",
    )

    if args.latest_root:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        latest_real = args.latest_root / "latest_real"
        latest_real.mkdir(parents=True, exist_ok=True)
        latest_pairs = [
            (html_path, "latest_master_report.html"),
        ]
        for source, name in latest_pairs:
            if source.exists():
                shutil.copyfile(source, args.latest_root / name)
        detail_pairs = [
            (html_path, "master_report.html"),
            (md_path, "master_report.md"),
            (json_path, "master_report.json"),
            (html_path, "m24_live_paired_report.html"),
            (md_path, "m24_live_paired_report.md"),
            (json_path, "m24_live_paired_report.json"),
            (args.out_dir / "live_paired_session_evidence.csv", "session_evidence.csv"),
            (args.out_dir / "live_paired_mode_summary.csv", "mode_summary.csv"),
            (args.out_dir / "live_paired_summary.csv", "paired_summary.csv"),
            (args.out_dir / "live_paired_key_deductions.csv", "key_deductions.csv"),
            (args.out_dir / "live_prefetch_checkpoint_results.csv", "prefetch_checkpoint_results.csv"),
            (args.out_dir / "live_prefetch_timeline_summary.csv", "prefetch_timeline_summary.csv"),
            (args.out_dir / "live_prefetch_session_observations.csv", "prefetch_session_observations.csv"),
            (args.out_dir / "live_prefetch_kv_movement_gaps.csv", "prefetch_kv_movement_gaps.csv"),
            (args.out_dir / "live_prefetch_kv_movement_diagnostics.csv", "prefetch_kv_movement_diagnostics.csv"),
            (args.out_dir / "live_prefetch_direct_hook_decisions.csv", "prefetch_direct_hook_decisions.csv"),
            (args.out_dir / "live_direct_kv_load_summary.csv", "direct_kv_load_summary.csv"),
            (args.out_dir / "live_direct_kv_load_evidence.csv", "direct_kv_load_evidence.csv"),
            (args.out_dir / "live_direct_kv_load_events.csv", "direct_kv_load_events.csv"),
            (args.out_dir / "live_replay_kv_demand_evidence.csv", "replay_kv_demand_evidence.csv"),
        ]
        for source, name in detail_pairs:
            if source.exists():
                shutil.copyfile(source, latest_real / name)

    print(f"Wrote live paired AgentBench report: {html_path}")
    print(f"No-prefetch gaps: {len(no_run['gaps'])}")
    print(f"Live-prefetch gaps: {len(pref_run['gaps'])}")
    print(f"Paired gaps: {pair_summary_rows[0].get('paired_tool_gaps', 0) if pair_summary_rows else 0}")


if __name__ == "__main__":
    main()
