#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any

from correlate_torch_profile_with_agent_trace import KV_EVENTS
from correlate_torch_profile_with_agent_trace import annotate_kv_windows_with_agent
from correlate_torch_profile_with_agent_trace import annotate_kv_windows_with_request_maps
from correlate_torch_profile_with_agent_trace import build_request_maps
from correlate_torch_profile_with_agent_trace import kv_columns
from correlate_torch_profile_with_agent_trace import paired_method_windows
from correlate_torch_profile_with_agent_trace import read_jsonl
from correlate_torch_profile_with_agent_trace import request_windows


def ns_to_ms(ts_ns: int, start_ns: int) -> float:
    return round((ts_ns - start_ns) / 1_000_000, 3)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def first_event(events: list[dict[str, Any]], name: str, session_id: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") == name and event.get("session_id") == session_id:
            return event
    return None


def request_window(events: list[dict[str, Any]], session_id: str, phase: str) -> tuple[int | None, int | None]:
    start = first_event([event for event in events if event.get("phase") == phase], "agent.request.start", session_id)
    end = first_event([event for event in events if event.get("phase") == phase], "agent.request.end", session_id)
    if not start or not end:
        return None, None
    return int(start["ts_ns"]), int(end["ts_ns"])


def build_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows = request_windows(events)
    windows.extend(paired_method_windows(events, KV_EVENTS))
    annotate_kv_windows_with_agent(windows)
    node_map, index_map = build_request_maps(windows)
    annotate_kv_windows_with_request_maps(windows, node_map, index_map)
    return windows


def session_ids(events: list[dict[str, Any]]) -> list[str]:
    ids = {
        str(event["session_id"])
        for event in events
        if isinstance(event.get("session_id"), str) and str(event.get("session_id")).startswith("agent_")
    }
    return sorted(ids)


def selected_kv_windows(
    windows: list[dict[str, Any]],
    session_id: str,
    hint_start_ns: int | None,
    hint_end_ns: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    preferred_events = {"hostpool.load_to_device_per_layer"}
    fallback_events = {"hicache.start_loading", "hicache.load"}

    def inside_hint_request(window: dict[str, Any]) -> bool:
        if hint_start_ns is None or hint_end_ns is None:
            return True
        return int(window["start_ns"]) < hint_end_ns and int(window["end_ns"]) > hint_start_ns

    def matches(cols: dict[str, Any]) -> bool:
        if cols.get("kv_agent_session_id") == session_id and cols.get("kv_agent_phase") == "hint_prefetch":
            return True
        session_ids = {
            item.strip()
            for item in str(cols.get("kv_agent_session_ids", "")).split(",")
            if item.strip()
        }
        phases = {
            item.strip()
            for item in str(cols.get("kv_agent_phases", "")).split(",")
            if item.strip()
        }
        if session_id in session_ids and "hint_prefetch" in phases:
            return True
        return False

    for event_names in (preferred_events, fallback_events):
        for window in windows:
            cols = kv_columns(window)
            if (
                window.get("window_type") == "sglang_kv_method"
                and window.get("event") in event_names
                and cols.get("kv_direction") == "host_to_device"
                and inside_hint_request(window)
                and matches(cols)
            ):
                out.append(window)
        if out:
            break
    return sorted(out, key=lambda item: int(item["start_ns"]))


def read_copy_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_copy_telemetry(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    starts: dict[str, dict[str, Any]] = {}
    windows: list[dict[str, Any]] = []
    for event in read_jsonl(path):
        call_id = str(event.get("call_id", ""))
        if not call_id:
            continue
        name = event.get("event")
        if name == "kv_telemetry.copy.start":
            starts[call_id] = event
        elif name in {"kv_telemetry.copy.end", "kv_telemetry.copy.error"}:
            start = starts.get(call_id, {})
            if not start:
                continue
            start_ns = start.get("ts_ns")
            end_ns = event.get("ts_ns")
            if not isinstance(start_ns, int) or not isinstance(end_ns, int):
                continue
            merged = dict(start)
            merged.update({key: value for key, value in event.items() if value not in (None, "", [], {})})
            merged["start_ns"] = start_ns
            merged["end_ns"] = end_ns
            merged["status"] = "error" if name == "kv_telemetry.copy.error" else "ok"
            windows.append(merged)
    return sorted(windows, key=lambda item: int(item["start_ns"]))


def read_outcome_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row.get("session_id", "")): row for row in csv.DictReader(f) if row.get("session_id")}


def read_profiler_coverage(profile_dir: Path | None) -> list[dict[str, Any]]:
    if profile_dir is None or not profile_dir.exists():
        return []

    windows: list[dict[str, Any]] = []
    for path in sorted(profile_dir.glob("torch_profiler_status_pid*.jsonl")):
        active: dict[str, Any] | None = None
        for event in read_jsonl(path):
            name = event.get("event")
            if name == "torch_profiler.start":
                active = {
                    "pid": event.get("pid", ""),
                    "start_ns": event.get("ts_ns"),
                    "start_label": event.get("label", ""),
                    "status_path": str(path),
                }
            elif name == "torch_profiler.export" and active is not None:
                active.update(
                    {
                        "end_ns": event.get("ts_ns"),
                        "reason": event.get("reason", ""),
                        "event_count": event.get("event_count", ""),
                        "trace_path": event.get("trace_path", ""),
                    }
                )
                windows.append(active)
                active = None
        if active is not None:
            active.update({"end_ns": None, "reason": "started_but_no_export", "event_count": "", "trace_path": ""})
            windows.append(active)
    return windows


def coverage_ms_fields(coverage: list[dict[str, Any]], trace_start_ns: int) -> tuple[str, str, str]:
    starts = [int(item["start_ns"]) for item in coverage if isinstance(item.get("start_ns"), int)]
    ends = [int(item["end_ns"]) for item in coverage if isinstance(item.get("end_ns"), int)]
    reasons = [str(item.get("reason", "")) for item in coverage if item.get("reason")]
    start_ms = ns_to_ms(min(starts), trace_start_ns) if starts else ""
    end_ms = ns_to_ms(max(ends), trace_start_ns) if ends else ""
    return str(start_ms), str(end_ms), ",".join(sorted(set(reasons)))


def profiler_window_status(
    coverage: list[dict[str, Any]],
    start_ms: float | None,
    end_ms: float | None,
    trace_start_ns: int,
) -> str:
    if start_ms is None or end_ms is None:
        return "no_sglang_kv_window"
    if not coverage:
        return "no_profiler_status"

    saw_before = False
    saw_after = False
    saw_overlap = False
    for item in coverage:
        start_ns = item.get("start_ns")
        end_ns = item.get("end_ns")
        if not isinstance(start_ns, int):
            continue
        cov_start_ms = ns_to_ms(start_ns, trace_start_ns)
        cov_end_ms = ns_to_ms(end_ns, trace_start_ns) if isinstance(end_ns, int) else float("inf")
        if cov_start_ms <= start_ms and end_ms <= cov_end_ms:
            return "inside_profiler_window"
        if end_ms < cov_start_ms:
            saw_before = True
        elif start_ms > cov_end_ms:
            saw_after = True
        elif start_ms < cov_end_ms and end_ms > cov_start_ms:
            saw_overlap = True

    if saw_overlap:
        return "partly_outside_profiler_window"
    if saw_after and not saw_before:
        return "after_profiler_stopped"
    if saw_before and not saw_after:
        return "before_profiler_started"
    return "outside_profiler_window"


def missing_h2d_reason(torch_copy_count: int, sglang_event_count: int, profile_status: str) -> str:
    if torch_copy_count > 0:
        return ""
    if sglang_event_count == 0:
        return "no_sglang_kv_load"
    if profile_status == "after_profiler_stopped":
        return "profiler_stopped_before_kv_load"
    if profile_status == "before_profiler_started":
        return "kv_load_before_profiler_started"
    if profile_status == "partly_outside_profiler_window":
        return "kv_load_partly_outside_profiler_window"
    if profile_status == "no_profiler_status":
        return "no_profiler_status"
    if profile_status == "inside_profiler_window":
        return "inside_profiler_window_but_no_h2d_attribution"
    return profile_status


def overlaps_ms(row: dict[str, Any], start_ms: float, end_ms: float) -> bool:
    row_start = to_float(row.get("start_ms_from_trace_start"))
    row_end = to_float(row.get("end_ms_from_trace_start"))
    if row_start is None or row_end is None:
        return False
    return row_start < end_ms and row_end > start_ms


def selected_copy_rows(
    rows: list[dict[str, Any]],
    session_id: str,
    kv_windows: list[dict[str, Any]],
    trace_start_ns: int,
) -> list[dict[str, Any]]:
    kv_windows_ms = [
        (ns_to_ms(int(window["start_ns"]), trace_start_ns), ns_to_ms(int(window["end_ns"]), trace_start_ns))
        for window in kv_windows
    ]

    def matches_session(row: dict[str, Any]) -> bool:
        session_ids = {
            item.strip()
            for item in str(row.get("overlap_kv_agent_session_ids", "")).split(",")
            if item.strip()
        }
        phases = {
            item.strip()
            for item in str(row.get("overlap_kv_agent_phases", "")).split(",")
            if item.strip()
        }
        if session_id in session_ids and "hint_prefetch" in phases:
            return True
        if row.get("overlap_kv_agent_session_id") == session_id and row.get("overlap_kv_agent_phase") == "hint_prefetch":
            return True
        if row.get("enclosing_agent_session_id") == session_id and row.get("enclosing_agent_phase") == "hint_prefetch":
            return True
        if row.get("overlap_kv_agent_session_id"):
            return False
        return any(overlaps_ms(row, start_ms, end_ms) for start_ms, end_ms in kv_windows_ms)

    out = [
        row
        for row in rows
        if row.get("direction") == "h2d"
        and row.get("overlap_event") == "hostpool.load_to_device_per_layer"
        and matches_session(row)
    ]
    return sorted(out, key=lambda row: float(row.get("start_ms_from_trace_start") or 0.0))


def selected_telemetry_windows(windows: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for window in windows:
        if window.get("source_event") != "hostpool.load_to_device_per_layer":
            continue
        if window.get("direction") != "host_to_device":
            continue
        if window.get("agent_session_id") == session_id and window.get("agent_phase") == "hint_prefetch":
            out.append(window)
            continue
        sessions = window.get("agent_sessions")
        if isinstance(sessions, list):
            for item in sessions:
                if (
                    isinstance(item, dict)
                    and item.get("agent_session_id") == session_id
                    and item.get("agent_phase") == "hint_prefetch"
                ):
                    out.append(window)
                    break
    return sorted(out, key=lambda item: int(item["start_ns"]))


def event_ms(events: list[dict[str, Any]], name: str, session_id: str, trace_start_ns: int) -> float | None:
    event = first_event(events, name, session_id)
    if not event or not event.get("ts_ns"):
        return None
    return ns_to_ms(int(event["ts_ns"]), trace_start_ns)


def phase_metric(events: list[dict[str, Any]], session_id: str, phase: str, key: str) -> Any:
    for event in events:
        if event.get("event") == "agent.request.end" and event.get("session_id") == session_id and event.get("phase") == phase:
            return event.get(key)
    return ""


def build_rows(
    events: list[dict[str, Any]],
    copy_rows: list[dict[str, Any]],
    profiler_coverage: list[dict[str, Any]] | None = None,
    outcome_rows: dict[str, dict[str, Any]] | None = None,
    telemetry_windows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_start_ns = min(int(event["ts_ns"]) for event in events if event.get("ts_ns"))
    windows = build_windows(events)
    coverage = profiler_coverage or []
    telemetry = telemetry_windows or []
    profiler_start_ms, profiler_end_ms, profiler_stop_reason = coverage_ms_fields(coverage, trace_start_ns)
    rows: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    outcomes = outcome_rows or {}

    for session_id in session_ids(events):
        outcome = outcomes.get(session_id, {})
        arrival = first_event(events, "agent.session_arrival", session_id) or {}
        initial_start, initial_end = request_window(events, session_id, "initial_turn")
        hint_start, hint_end = request_window(events, session_id, "hint_prefetch")
        replay_start, replay_end = request_window(events, session_id, "replay")
        hint_submitted_ms = event_ms(events, "agent.hint_submitted", session_id, trace_start_ns)
        tool_wait_start_ms = event_ms(events, "agent.tool_wait_start", session_id, trace_start_ns)
        replay_due_ms = event_ms(events, "agent.replay_due", session_id, trace_start_ns)
        resume_start_ms = event_ms(events, "agent.resume_start", session_id, trace_start_ns)
        hint_start_ms = ns_to_ms(hint_start, trace_start_ns) if hint_start else None
        hint_end_ms = ns_to_ms(hint_end, trace_start_ns) if hint_end else None
        replay_start_ms = ns_to_ms(replay_start, trace_start_ns) if replay_start else None
        replay_end_ms = ns_to_ms(replay_end, trace_start_ns) if replay_end else None

        kv = selected_kv_windows(windows, session_id, hint_start, hint_end)
        sglang_start_ms = ns_to_ms(int(kv[0]["start_ns"]), trace_start_ns) if kv else None
        sglang_end_ms = ns_to_ms(max(int(item["end_ns"]) for item in kv), trace_start_ns) if kv else None
        sglang_event_count = len(kv)
        sglang_bytes_like = ""
        if kv:
            counts = [kv_columns(item).get("host_index_count") for item in kv if kv_columns(item).get("host_index_count")]
            if counts:
                sglang_bytes_like = str(counts[0])

        telemetry_kv = selected_telemetry_windows(telemetry, session_id)
        telemetry_start_ms = ns_to_ms(int(telemetry_kv[0]["start_ns"]), trace_start_ns) if telemetry_kv else None
        telemetry_end_ms = ns_to_ms(max(int(item["end_ns"]) for item in telemetry_kv), trace_start_ns) if telemetry_kv else None
        telemetry_event_count = len(telemetry_kv)
        telemetry_host_index_count = ""
        if telemetry_kv:
            counts = [item.get("host_index_count") for item in telemetry_kv if item.get("host_index_count")]
            if counts:
                telemetry_host_index_count = str(counts[0])
        if sglang_start_ms is None and telemetry_start_ms is not None:
            sglang_start_ms = telemetry_start_ms
            sglang_end_ms = telemetry_end_ms
            sglang_event_count = telemetry_event_count
            sglang_bytes_like = telemetry_host_index_count

        copies = selected_copy_rows(copy_rows, session_id, kv, trace_start_ns)
        torch_start_ms = to_float(copies[0].get("start_ms_from_trace_start")) if copies else None
        torch_end_ms = max((to_float(row.get("end_ms_from_trace_start")) or 0.0 for row in copies), default=None)
        torch_copy_count = len(copies)
        torch_bytes = sum(int(float(row.get("bytes") or 0)) for row in copies)
        profile_status = profiler_window_status(coverage, sglang_start_ms, sglang_end_ms, trace_start_ns)
        h2d_missing_reason = missing_h2d_reason(torch_copy_count, sglang_event_count, profile_status)

        visible_copy_start_ms = torch_start_ms if torch_start_ms is not None else telemetry_start_ms
        visible_copy_end_ms = torch_end_ms if torch_end_ms is not None else telemetry_end_ms
        visible_copy_source = (
            "torch_profiler_h2d"
            if torch_end_ms is not None
            else "sglang_lightweight_h2d_telemetry"
            if telemetry_end_ms is not None
            else ""
        )
        prefetch_done_ms = visible_copy_end_ms if visible_copy_end_ms is not None else sglang_end_ms
        prefetch_margin_ms = (
            round(float(replay_due_ms) - float(prefetch_done_ms), 3)
            if replay_due_ms is not None and prefetch_done_ms is not None
            else None
        )
        late_prefetch = prefetch_margin_ms is not None and prefetch_margin_ms < 0
        no_visible_prefetch = prefetch_done_ms is None
        cuda_copy_ready_before_replay = bool(
            torch_end_ms is not None and replay_due_ms is not None and torch_end_ms <= replay_due_ms
        )
        kv_copy_ready_before_replay = bool(
            visible_copy_end_ms is not None and replay_due_ms is not None and visible_copy_end_ms <= replay_due_ms
        )
        hint_replay_overlap_ms = 0.0
        if hint_start_ms is not None and hint_end_ms is not None and replay_start_ms is not None and replay_end_ms is not None:
            hint_replay_overlap_ms = max(0.0, min(hint_end_ms, replay_end_ms) - max(hint_start_ms, replay_start_ms))
        hint_overlaps_replay = hint_replay_overlap_ms > 0

        def inside_hint(start_ms: float | None, end_ms: float | None) -> bool:
            return bool(
                start_ms is not None
                and end_ms is not None
                and hint_start_ms is not None
                and hint_end_ms is not None
                and hint_start_ms <= start_ms
                and end_ms <= hint_end_ms
            )

        telemetry_copy_inside_hint = inside_hint(telemetry_start_ms, telemetry_end_ms)
        torch_copy_inside_hint = inside_hint(torch_start_ms, torch_end_ms)
        visible_copy_inside_hint = inside_hint(visible_copy_start_ms, visible_copy_end_ms)
        full_hint_done_before_replay = str(outcome.get("hint_completed_before_replay", "")).strip() == "1"
        resume_load_count = int(float(outcome.get("resume_load_count") or 0))
        resume_hicache_load_count = int(float(outcome.get("resume_hicache_load_count") or 0))
        replay_reloaded_kv = resume_load_count > 0 or resume_hicache_load_count > 0
        hint_outcome = str(outcome.get("outcome", ""))
        if hint_outcome == "no_prefetch_needed":
            checkpoint_result = "no_prefetch_needed"
        elif cuda_copy_ready_before_replay and full_hint_done_before_replay and not replay_reloaded_kv:
            checkpoint_result = "clean_success"
        elif cuda_copy_ready_before_replay and full_hint_done_before_replay and replay_reloaded_kv:
            checkpoint_result = "copy_ready_but_replay_reloaded"
        elif cuda_copy_ready_before_replay and not full_hint_done_before_replay:
            checkpoint_result = "copy_ready_but_hint_not_done"
        elif not cuda_copy_ready_before_replay and full_hint_done_before_replay:
            checkpoint_result = "hint_done_but_no_cuda_ready"
        elif no_visible_prefetch:
            checkpoint_result = "no_visible_prefetch"
        else:
            checkpoint_result = "not_ready"

        rows.append(
            {
                "session_id": session_id,
                "priority": arrival.get("priority", ""),
                "arrival_ms": arrival.get("arrival_ms", ""),
                "tool_wait_ms": arrival.get("tool_wait_ms", ""),
                "prompt_tokens": arrival.get("prompt_tokens", ""),
                "hint_submitted_ms": hint_submitted_ms if hint_submitted_ms is not None else "",
                "hint_request_start_ms": hint_start_ms if hint_start_ms is not None else "",
                "hint_request_end_ms": hint_end_ms if hint_end_ms is not None else "",
                "sglang_copy_start_ms": sglang_start_ms if sglang_start_ms is not None else "",
                "sglang_copy_end_ms": sglang_end_ms if sglang_end_ms is not None else "",
                "sglang_copy_events": sglang_event_count,
                "sglang_host_index_count": sglang_bytes_like,
                "telemetry_copy_start_ms": telemetry_start_ms if telemetry_start_ms is not None else "",
                "telemetry_copy_end_ms": telemetry_end_ms if telemetry_end_ms is not None else "",
                "telemetry_h2d_copy_events": telemetry_event_count,
                "telemetry_host_index_count": telemetry_host_index_count,
                "torch_copy_start_ms": torch_start_ms if torch_start_ms is not None else "",
                "torch_copy_end_ms": torch_end_ms if torch_end_ms is not None else "",
                "torch_h2d_copy_events": torch_copy_count,
                "torch_h2d_bytes": torch_bytes,
                "visible_copy_start_ms": visible_copy_start_ms if visible_copy_start_ms is not None else "",
                "visible_copy_end_ms": visible_copy_end_ms if visible_copy_end_ms is not None else "",
                "visible_copy_source": visible_copy_source,
                "profiler_start_ms": profiler_start_ms,
                "profiler_end_ms": profiler_end_ms,
                "profiler_stop_reason": profiler_stop_reason,
                "sglang_kv_profiler_status": profile_status,
                "h2d_missing_reason": h2d_missing_reason,
                "replay_due_ms": replay_due_ms if replay_due_ms is not None else "",
                "replay_start_ms": replay_start_ms if replay_start_ms is not None else "",
                "replay_end_ms": replay_end_ms if replay_end_ms is not None else "",
                "resume_start_ms": resume_start_ms if resume_start_ms is not None else "",
                "hint_replay_overlap_ms": round(hint_replay_overlap_ms, 3),
                "hint_overlaps_replay": int(hint_overlaps_replay),
                "telemetry_copy_inside_hint": int(telemetry_copy_inside_hint),
                "torch_copy_inside_hint": int(torch_copy_inside_hint),
                "visible_copy_inside_hint": int(visible_copy_inside_hint),
                "prefetch_margin_ms": prefetch_margin_ms if prefetch_margin_ms is not None else "",
                "late_prefetch": late_prefetch,
                "no_visible_prefetch": no_visible_prefetch,
                "kv_copy_ready_before_replay": int(kv_copy_ready_before_replay),
                "cuda_copy_ready_before_replay": int(cuda_copy_ready_before_replay),
                "full_hint_done_before_replay": int(full_hint_done_before_replay),
                "replay_reloaded_kv": int(replay_reloaded_kv),
                "resume_load_count": resume_load_count,
                "resume_hicache_load_count": resume_hicache_load_count,
                "eviction_pressure_after_prefetch": outcome.get("eviction_pressure_after_prefetch", ""),
                "hint_total_duration_ms": outcome.get("hint_total_duration_ms", ""),
                "hint_outcome": hint_outcome,
                "checkpoint_result": checkpoint_result,
                "replay_ttft_ms": phase_metric(events, session_id, "replay", "ttft_ms"),
            }
        )

        def add_bar(kind: str, start: int | float | None, end: int | float | None, label: str) -> None:
            if start is None or end is None:
                return
            timeline.append({"session_id": session_id, "kind": kind, "start_ms": round(float(start), 3), "end_ms": round(float(end), 3), "label": label})

        def add_marker(kind: str, at: int | float | None, label: str) -> None:
            if at is None:
                return
            timeline.append({"session_id": session_id, "kind": kind, "start_ms": round(float(at), 3), "end_ms": round(float(at), 3), "label": label})

        add_bar("initial", ns_to_ms(initial_start, trace_start_ns) if initial_start else None, ns_to_ms(initial_end, trace_start_ns) if initial_end else None, "initial")
        add_bar("tool_wait", tool_wait_start_ms, replay_due_ms, "tool wait")
        add_marker("hint_submitted", hint_submitted_ms, "hint")
        add_bar("hint_request", hint_start_ms, hint_end_ms, "hint request")
        add_bar("sglang_copy", sglang_start_ms, sglang_end_ms, "SGLang KV load")
        add_bar("telemetry_copy", telemetry_start_ms, telemetry_end_ms, "Lightweight KV HtoD telemetry")
        add_bar("torch_copy", torch_start_ms, torch_end_ms, "CUDA HtoD copies")
        add_marker("replay_due", replay_due_ms, "replay due")
        add_bar("replay", replay_start_ms, replay_end_ms, "replay")

    return rows, timeline


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sessions": rows, "timeline": timeline}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def fmt_ms(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "not attributed"
    return f"{number:.3f} ms"


def yes_no(value: Any) -> str:
    return "yes" if str(value).strip() in {"1", "True", "true", "yes"} else "no"


def checkpoint_class(value: Any, good_when_yes: bool = True) -> str:
    is_yes = yes_no(value) == "yes"
    good = is_yes if good_when_yes else not is_yes
    return "good" if good else "bad"


def session_observation(row: dict[str, Any]) -> tuple[str, str, str]:
    margin = to_float(row.get("prefetch_margin_ms"))
    torch_copy_events = int(row.get("torch_h2d_copy_events") or 0)
    sglang_events = int(row.get("sglang_copy_events") or 0)
    missing_reason = str(row.get("h2d_missing_reason") or "")
    cuda_ready = yes_no(row.get("cuda_copy_ready_before_replay"))
    hint_done = yes_no(row.get("full_hint_done_before_replay"))
    replay_reloaded = yes_no(row.get("replay_reloaded_kv"))

    if row.get("no_visible_prefetch"):
        status = "No visible prefetch"
        observation = "The trace did not show a SGLang KV load or CUDA HtoD copy for this hint."
        deduction = "This session is weak evidence for movement timing; use it mainly to show missing visibility."
    elif row.get("late_prefetch") is True and margin is not None:
        status = "Late prefetch"
        if missing_reason == "profiler_stopped_before_kv_load":
            observation = (
                f"SGLang KV movement completed {abs(margin):.3f} ms after replay was already due. "
                "CUDA HtoD is not shown because torch.profiler had already stopped before this KV window."
            )
            deduction = "This is a real late SGLang prefetch, but the missing green bar is a profiler-coverage issue, not proof that no CUDA copy happened."
        else:
            observation = f"The hinted KV movement completed {abs(margin):.3f} ms after replay was already due."
            deduction = "The software hint existed, but the normal serving/KV path did not act early enough."
    elif torch_copy_events > 0 and margin is not None:
        status = "Clean useful prefetch"
        observation = f"SGLang KV load and CUDA HtoD copies were visible, and movement completed {margin:.3f} ms before replay."
        deduction = "This is the clean success case: the hint moved KV early enough for the agent replay."
    elif sglang_events > 0 and margin is not None:
        status = "SGLang-level useful prefetch"
        if missing_reason:
            observation = (
                f"SGLang KV load was visible and completed {margin:.3f} ms before replay, "
                f"but CUDA HtoD attribution is missing because: {missing_reason}."
            )
        else:
            observation = f"SGLang KV load was visible and completed {margin:.3f} ms before replay, but CUDA HtoD rows were not confidently attributed to this session."
        deduction = "This is useful SGLang-level evidence, but weaker CUDA-level evidence."
    else:
        status = "Incomplete timing"
        observation = "Some required timing fields were missing."
        deduction = "Do not use this row as a strong success or failure example."

    evidence = (
        f"tool wait {fmt_ms(row.get('tool_wait_ms'))}; "
        f"CUDA copy ready before replay {cuda_ready}; "
        f"full hint done before replay {hint_done}; "
        f"replay reloaded KV {replay_reloaded}; "
        f"SGLang load {fmt_ms(row.get('sglang_copy_start_ms'))} -> {fmt_ms(row.get('sglang_copy_end_ms'))}; "
        f"CUDA HtoD {fmt_ms(row.get('torch_copy_start_ms'))} -> {fmt_ms(row.get('torch_copy_end_ms'))}; "
        f"profiler window {fmt_ms(row.get('profiler_start_ms'))} -> {fmt_ms(row.get('profiler_end_ms'))}; "
        f"HtoD missing reason {fmt(row.get('h2d_missing_reason')) or 'none'}; "
        f"replay due {fmt_ms(row.get('replay_due_ms'))}; "
        f"margin {fmt_ms(row.get('prefetch_margin_ms'))}"
    )
    return status, observation, deduction + " " + evidence


def timeline_status(row: dict[str, Any]) -> tuple[str, str]:
    margin = to_float(row.get("prefetch_margin_ms"))
    if row.get("checkpoint_result") == "no_prefetch_needed":
        return "NO LOAD", "#6b7280"
    if row.get("checkpoint_result") == "clean_success":
        return f"CLEAN +{margin:.0f} ms" if margin is not None else "CLEAN", "#166534"
    if row.get("checkpoint_result") == "copy_ready_but_replay_reloaded":
        return f"RELOAD +{margin:.0f} ms" if margin is not None else "RELOAD", "#92400e"
    if row.get("checkpoint_result") == "copy_ready_but_hint_not_done":
        return f"HINT LATE +{margin:.0f} ms" if margin is not None else "HINT LATE", "#b45309"
    if row.get("late_prefetch") is True and margin is not None:
        return f"LATE {margin:.0f} ms", "#b91c1c"
    if int(row.get("torch_h2d_copy_events") or 0) > 0 and margin is not None:
        return f"SUCCESS +{margin:.0f} ms", "#166534"
    if int(row.get("sglang_copy_events") or 0) > 0 and margin is not None:
        return f"SGLang OK +{margin:.0f} ms", "#92400e"
    return "INCOMPLETE", "#6b7280"


def choose_sessions(rows: list[dict[str, Any]], max_sessions: int) -> list[str]:
    late = [row for row in rows if row.get("late_prefetch") is True]
    visible = [row for row in rows if row.get("torch_h2d_copy_events", 0)]
    ordered: list[str] = []
    for group in (late, visible, rows):
        for row in sorted(group, key=lambda item: float(item.get("prefetch_margin_ms") or 1_000_000)):
            sid = str(row["session_id"])
            if sid not in ordered:
                ordered.append(sid)
            if len(ordered) >= max_sessions:
                return ordered
    return ordered


def write_html(path: Path, rows: list[dict[str, Any]], timeline: list[dict[str, Any]], max_sessions: int) -> None:
    selected = choose_sessions(rows, max_sessions)
    row_by_session = {str(row["session_id"]): row for row in rows}
    selected_rows = [row_by_session[sid] for sid in selected if sid in row_by_session]
    selected_timeline = [item for item in timeline if item["session_id"] in selected]
    if selected_timeline:
        start = min(float(item["start_ms"]) for item in selected_timeline)
        end = max(float(item["end_ms"]) for item in selected_timeline)
    else:
        start, end = 0.0, 1.0
    span = max(1.0, end - start)
    width = 1600
    left = 255
    right = 60
    row_h = 138
    top = 90
    height = top + row_h * max(1, len(selected)) + 118
    plot_w = width - left - right
    colors = {
        "initial": "#2563eb",
        "tool_wait": "#d1d5db",
        "hint_submitted": "#7c3aed",
        "hint_request": "#a855f7",
        "sglang_copy": "#f59e0b",
        "telemetry_copy": "#22c55e",
        "torch_copy": "#16a34a",
        "replay_due": "#111827",
        "replay": "#dc2626",
    }

    def x_pos(ms: float) -> float:
        return left + (ms - start) / span * plot_w

    def lane(kind: str) -> tuple[int, int, str]:
        lanes = {
            "initial": (8, 18, "initial"),
            "tool_wait": (30, 70, "tool wait"),
            "hint_submitted": (33, 20, "hint"),
            "hint_request": (34, 18, "hint request"),
            "sglang_copy": (59, 14, "SGLang KV"),
            "telemetry_copy": (57, 18, "KV"),
            "torch_copy": (79, 18, "HtoD"),
            "replay_due": (28, 94, "due"),
            "replay": (104, 18, "replay"),
        }
        return lanes.get(kind, (34, 18, kind))

    def layer_order(item: dict[str, Any]) -> int:
        order = {
            "tool_wait": 0,
            "initial": 1,
            "hint_submitted": 2,
            "hint_request": 3,
            "replay": 4,
            "sglang_copy": 5,
            "telemetry_copy": 6,
            "torch_copy": 7,
        "replay_due": 8,
        }
        return order.get(str(item.get("kind", "")), 10)

    row_index = {sid: idx for idx, sid in enumerate(selected)}
    svg: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Agentic prefetch timeline">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    for tick in range(6):
        ms = start + span * tick / 5
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 30}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{ms:.0f} ms</text>')
    for sid in selected:
        y = top + row_index[sid] * row_h
        row = row_by_session.get(sid, {})
        status_label, status_color = timeline_status(row)
        svg.append(f'<text x="10" y="{y + 17}" font-weight="700">{html.escape(sid)}</text>')
        svg.append(f'<text x="10" y="{y + 38}" font-size="13" fill="{status_color}" font-weight="700">{html.escape(status_label)}</text>')
        overlap_ms = to_float(row.get("hint_replay_overlap_ms")) or 0.0
        if overlap_ms > 0:
            svg.append(f'<text x="10" y="{y + 59}" font-size="12" fill="#b91c1c" font-weight="700">overlap {overlap_ms:.0f} ms</text>')
        for label, offset in (("hint", 47), ("copy", 70), ("replay", 117)):
            svg.append(f'<text x="{left - 8}" y="{y + offset}" text-anchor="end" font-size="10" fill="#64748b">{label}</text>')
        for offset in (20, 54, 77, 101, 126):
            svg.append(f'<line x1="{left}" y1="{y + offset}" x2="{left + plot_w}" y2="{y + offset}" stroke="#f8fafc"/>')
        prefetch_done = to_float(row.get("visible_copy_end_ms")) or to_float(row.get("sglang_copy_end_ms"))
        replay_due = to_float(row.get("replay_due_ms"))
        margin = to_float(row.get("prefetch_margin_ms"))
        hint_start_for_overlap = to_float(row.get("hint_request_start_ms"))
        hint_end_for_overlap = to_float(row.get("hint_request_end_ms"))
        replay_start_for_overlap = to_float(row.get("replay_start_ms"))
        replay_end_for_overlap = to_float(row.get("replay_end_ms"))
        if (
            hint_start_for_overlap is not None
            and hint_end_for_overlap is not None
            and replay_start_for_overlap is not None
            and replay_end_for_overlap is not None
        ):
            overlap_start = max(hint_start_for_overlap, replay_start_for_overlap)
            overlap_end = min(hint_end_for_overlap, replay_end_for_overlap)
            if overlap_end > overlap_start:
                svg.append(
                    f'<rect x="{x_pos(overlap_start):.1f}" y="{y + 28}" width="{max(2, x_pos(overlap_end) - x_pos(overlap_start)):.1f}" '
                    'height="98" fill="#fecaca" opacity="0.45"><title>hint and replay overlap</title></rect>'
                )
        if prefetch_done is not None and replay_due is not None and margin is not None:
            x_done = x_pos(prefetch_done)
            x_due = x_pos(replay_due)
            y_margin = y + 128
            y_label = y + 136
            if margin >= 0:
                svg.append(
                    f'<line x1="{x_done:.1f}" y1="{y_margin}" x2="{x_due:.1f}" y2="{y_margin}" '
                    'stroke="#16a34a" stroke-width="4" stroke-dasharray="8 5"/>'
                )
                svg.append(f'<circle cx="{x_done:.1f}" cy="{y_margin}" r="5" fill="#16a34a"><title>prefetch done</title></circle>')
                svg.append(f'<text x="{(x_done + x_due) / 2:.1f}" y="{y_label}" text-anchor="middle" font-size="12" fill="#166534" font-weight="700">ready +{margin:.0f} ms</text>')
            else:
                x1 = min(x_due, x_done)
                x2 = max(x_due, x_done)
                svg.append(f'<rect x="{x1:.1f}" y="{y + 28}" width="{max(2, x2 - x1):.1f}" height="98" fill="#fee2e2" opacity="0.55"/>')
                svg.append(
                    f'<line x1="{x_due:.1f}" y1="{y_margin}" x2="{x_done:.1f}" y2="{y_margin}" '
                    'stroke="#dc2626" stroke-width="4" stroke-dasharray="8 5"/>'
                )
                svg.append(f'<circle cx="{x_done:.1f}" cy="{y_margin}" r="5" fill="#dc2626"><title>prefetch done after replay due</title></circle>')
                svg.append(f'<text x="{(x_done + x_due) / 2:.1f}" y="{y_label}" text-anchor="middle" font-size="12" fill="#b91c1c" font-weight="700">{abs(margin):.0f} ms late</text>')
    for item in sorted(selected_timeline, key=layer_order):
        sid = item["session_id"]
        y = top + row_index[sid] * row_h
        kind = item["kind"]
        color = colors.get(kind, "#6b7280")
        x1 = x_pos(float(item["start_ms"]))
        x2 = x_pos(float(item["end_ms"]))
        if x1 == x2:
            lane_y, lane_h, _ = lane(kind)
            stroke_width = 6 if kind == "replay_due" else 3
            svg.append(f'<line x1="{x1:.1f}" y1="{y + lane_y}" x2="{x1:.1f}" y2="{y + lane_y + lane_h}" stroke="{color}" stroke-width="{stroke_width}"><title>{html.escape(item["label"])}</title></line>')
            if kind == "replay_due":
                svg.append(f'<text x="{x1:.1f}" y="{y + 27}" text-anchor="middle" font-size="11" fill="#111827" font-weight="700">due</text>')
        else:
            display_x2 = x2
            lane_y, lane_h, lane_label = lane(kind)
            bar_y = y + lane_y
            bar_h = lane_h
            opacity = "0.88"
            stroke = ""
            if kind in {"telemetry_copy", "torch_copy"}:
                display_x2 = max(x2, x1 + 24)
                opacity = "1"
                stroke = ' stroke="#f8fafc" stroke-width="3"'
            svg.append(f'<rect x="{x1:.1f}" y="{bar_y}" width="{max(2, display_x2 - x1):.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="{opacity}"{stroke}><title>{html.escape(item["label"])}</title></rect>')
            if kind in {"telemetry_copy", "torch_copy"}:
                label = "KV" if kind == "telemetry_copy" else "HtoD"
                svg.append(f'<text x="{(x1 + display_x2) / 2:.1f}" y="{bar_y + 13}" text-anchor="middle" font-size="10" fill="white" font-weight="700">{label}</text>')
            elif kind == "sglang_copy" and x2 - x1 > 45:
                label = "KV load" if kind == "sglang_copy" else "HtoD"
                svg.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{bar_y + 11}" text-anchor="middle" font-size="11" fill="white" font-weight="700">{label}</text>')
    legend_x = left
    legend_y = height - 32
    for idx, (kind, color) in enumerate(colors.items()):
        lx = legend_x + idx * 130
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}">{html.escape(kind)}</text>')
    svg.append(f'<line x1="{left}" y1="{height - 8}" x2="{left + 90}" y2="{height - 8}" stroke="#16a34a" stroke-width="4" stroke-dasharray="8 5"/>')
    svg.append(f'<text x="{left + 100}" y="{height - 4}" fill="#166534">green gap = prefetch done before replay</text>')
    svg.append(f'<line x1="{left + 420}" y1="{height - 8}" x2="{left + 510}" y2="{height - 8}" stroke="#dc2626" stroke-width="4" stroke-dasharray="8 5"/>')
    svg.append(f'<text x="{left + 520}" y="{height - 4}" fill="#b91c1c">red gap = prefetch finished after replay was due</text>')
    svg.append("</svg>")

    late_count = sum(1 for row in rows if row.get("late_prefetch") is True)
    visible_count = sum(1 for row in rows if row.get("visible_copy_end_ms") not in ("", None))
    telemetry_count = sum(1 for row in rows if int(row.get("telemetry_h2d_copy_events") or 0) > 0)
    torch_count = sum(1 for row in rows if int(row.get("torch_h2d_copy_events") or 0) > 0)
    missing_reasons = sorted(
        {
            str(row.get("h2d_missing_reason"))
            for row in rows
            if row.get("h2d_missing_reason")
        }
    )
    margins = [float(row["prefetch_margin_ms"]) for row in rows if row.get("prefetch_margin_ms") not in ("", None)]
    summary = {
        "sessions": len(rows),
        "sessions_with_visible_copy_telemetry": visible_count,
        "sessions_with_lightweight_kv_copy_telemetry": telemetry_count,
        "sessions_with_profiler_cuda_h2d_copy": torch_count,
        "late_prefetch_sessions": late_count,
        "kv_copy_ready_before_replay": sum(1 for row in rows if yes_no(row.get("kv_copy_ready_before_replay")) == "yes"),
        "cuda_copy_ready_before_replay": sum(1 for row in rows if yes_no(row.get("cuda_copy_ready_before_replay")) == "yes"),
        "full_hint_done_before_replay": sum(1 for row in rows if yes_no(row.get("full_hint_done_before_replay")) == "yes"),
        "sessions_where_replay_reloaded_kv": sum(1 for row in rows if yes_no(row.get("replay_reloaded_kv")) == "yes"),
        "clean_success_sessions": sum(1 for row in rows if row.get("checkpoint_result") == "clean_success"),
        "h2d_missing_reasons": ", ".join(missing_reasons),
        "avg_prefetch_margin_ms": round(mean(margins), 3) if margins else "",
    }
    columns = [
        "session_id",
        "priority",
        "tool_wait_ms",
        "prompt_tokens",
        "hint_submitted_ms",
        "hint_request_start_ms",
        "hint_request_end_ms",
        "sglang_copy_start_ms",
        "sglang_copy_end_ms",
        "telemetry_copy_start_ms",
        "telemetry_copy_end_ms",
        "telemetry_h2d_copy_events",
        "telemetry_host_index_count",
        "torch_copy_start_ms",
        "torch_copy_end_ms",
        "torch_h2d_copy_events",
        "torch_h2d_bytes",
        "visible_copy_start_ms",
        "visible_copy_end_ms",
        "visible_copy_source",
        "sglang_kv_profiler_status",
        "h2d_missing_reason",
        "profiler_start_ms",
        "profiler_end_ms",
        "profiler_stop_reason",
        "replay_due_ms",
        "replay_start_ms",
        "replay_end_ms",
        "hint_replay_overlap_ms",
        "hint_overlaps_replay",
        "telemetry_copy_inside_hint",
        "torch_copy_inside_hint",
        "visible_copy_inside_hint",
        "prefetch_margin_ms",
        "late_prefetch",
        "kv_copy_ready_before_replay",
        "cuda_copy_ready_before_replay",
        "full_hint_done_before_replay",
        "replay_reloaded_kv",
        "resume_load_count",
        "resume_hicache_load_count",
        "eviction_pressure_after_prefetch",
        "hint_total_duration_ms",
        "hint_outcome",
        "checkpoint_result",
        "replay_ttft_ms",
    ]
    visible_h2d_rows = [
        {
            "session_id": row.get("session_id", ""),
            "copy_start_ms": row.get("visible_copy_start_ms", ""),
            "copy_end_ms": row.get("visible_copy_end_ms", ""),
            "copy_source": row.get("visible_copy_source", ""),
            "telemetry_events": row.get("telemetry_h2d_copy_events", ""),
            "torch_h2d_events": row.get("torch_h2d_copy_events", ""),
            "torch_h2d_bytes": row.get("torch_h2d_bytes", ""),
            "copy_inside_hint": yes_no(row.get("visible_copy_inside_hint")),
            "replay_due_ms": row.get("replay_due_ms", ""),
            "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
        }
        for row in selected_rows
        if row.get("visible_copy_end_ms") not in ("", None)
    ]
    invariant_rows = [
        {
            "session_id": row.get("session_id", ""),
            "green_inside_purple": yes_no(row.get("visible_copy_inside_hint")),
            "light_green_inside_purple": yes_no(row.get("telemetry_copy_inside_hint")),
            "dark_green_inside_purple": yes_no(row.get("torch_copy_inside_hint")),
            "hint_overlaps_replay": yes_no(row.get("hint_overlaps_replay")),
            "hint_replay_overlap_ms": row.get("hint_replay_overlap_ms", ""),
            "replay_reloaded_kv": yes_no(row.get("replay_reloaded_kv")),
        }
        for row in selected_rows
    ]
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Agentic Prefetch Timeline</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}",
        "h1,h2{margin:0 0 12px}",
        ".panel{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin:16px 0}",
        ".caption{margin:0 0 12px;color:#374151;line-height:1.45}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;white-space:nowrap}",
        "th{background:#f3f4f6;font-weight:700}",
        ".wrap{white-space:normal;line-height:1.35;min-width:260px}",
        ".status{font-weight:700}",
        ".bad{color:#b91c1c;font-weight:700}",
        ".good{color:#166534;font-weight:700}",
        ".warn{color:#92400e;font-weight:700}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Agentic Prefetch Timeline</h1>",
        '<div class="panel"><h2>Summary</h2><table><tbody>',
    ]
    for key, value in summary.items():
        lines.append(f"<tr><th>{html.escape(key)}</th><td>{fmt(value)}</td></tr>")
    lines.extend(
        [
            "</tbody></table></div>",
            '<div class="panel"><h2>Timeline</h2>',
            '<p class="caption">How to read this: each agent now has separate lanes. Gray is the tool-wait window. Purple is the software hint request. Bright green is lightweight SGLang KV-copy telemetry. Dark green is profiler-attributed CUDA HtoD copy. The green copy lanes should normally sit within the purple hint time window. Red is the replay request on its own lower lane, so purple/red overlap means the hint was still running when replay started. The black line is replay due.</p>',
            *svg,
            "</div>",
        ]
    )
    if visible_h2d_rows:
        lines.append('<div class="panel"><h2>Visible KV Copy Telemetry</h2>')
        lines.append('<p class="caption">These rows are the exact sessions represented by green copy bars in the timeline. `sglang_lightweight_h2d_telemetry` is the scalable source; `torch_profiler_h2d` is the heavier CUDA validation source.</p>')
        lines.append("<table><thead><tr>")
        for col in visible_h2d_rows[0]:
            lines.append(f"<th>{html.escape(col)}</th>")
        lines.append("</tr></thead><tbody>")
        for row in visible_h2d_rows:
            lines.append("<tr>")
            for col in row:
                lines.append(f"<td>{fmt(row.get(col, ''))}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table></div>")
    lines.append('<div class="panel"><h2>Timeline Sanity Checks</h2>')
    lines.append('<p class="caption">This table makes the visual invariants explicit. For hint-side KV movement, green copy windows should usually be inside the purple hint request. If hint overlaps replay, the software prefetch path was still running when the real agent turn arrived.</p>')
    lines.append("<table><thead><tr>")
    for col in (invariant_rows[0] if invariant_rows else []):
        lines.append(f"<th>{html.escape(col)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in invariant_rows:
        lines.append("<tr>")
        for col in row:
            cls = ""
            if col == "hint_overlaps_replay":
                cls = f' class="{checkpoint_class(row.get(col), good_when_yes=False)}"'
            elif col.endswith("inside_purple"):
                cls = f' class="{checkpoint_class(row.get(col))}"'
            elif col == "replay_reloaded_kv":
                cls = f' class="{checkpoint_class(row.get(col), good_when_yes=False)}"'
            lines.append(f"<td{cls}>{fmt(row.get(col, ''))}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    lines.append('<div class="panel"><h2>Timeline Layers</h2><table><thead><tr>')
    for col in ("Layer", "Meaning", "Why it matters"):
        lines.append(f"<th>{html.escape(col)}</th>")
    lines.append("</tr></thead><tbody>")
    layer_rows = [
        (
            "gray tool_wait",
            "The agent is waiting for a tool result, such as tests, search, or build output.",
            "This is the opportunity window where prefetch can happen before the next model turn.",
        ),
        (
            "purple hint_request",
            "The software request we currently send to SGLang to trigger KV load-back. It has its own lane above the copy lane.",
            "This is not pure DMA. It includes scheduling, prefix matching, KV load-back, model work, and request bookkeeping.",
        ),
        (
            "bright green telemetry_copy",
            "Lightweight SGLang host-to-device KV copy telemetry from the exact KV load path. Short green bars are visually widened so they are easy to see.",
            "This is the scalable evidence path for larger experiments. It avoids huge torch.profiler traces while preserving per-session KV movement timing.",
        ),
        (
            "dark green torch_copy",
            "Profiler-attributed CUDA host-to-device copy activity inside the hint request. On the chart, short dark-green bars are visually widened so they are easy to see.",
            "This is the closest signal we have for actual GPU-side KV movement. It should usually live inside the purple hint request. If it is missing, check the profiler coverage columns before concluding no copy happened. Use the HtoD table for exact start/end times.",
        ),
        (
            "red replay",
            "The real agent replay request. It has its own lower lane so it does not visually merge with the purple hint request.",
            "If the purple hint request overlaps the red replay in time, the prefetch path was still running when the agent needed to resume.",
        ),
        (
            "profiler window",
            "The time range where torch.profiler was actually recording CUDA work.",
            "A session can have SGLang KV movement but no green bar if that movement happened after the profiler stopped.",
        ),
        (
            "green/red dashed gap",
            "The time between KV/copy completion and replay due.",
            "Green means KV was ready before replay. Red means the replay deadline passed before KV movement finished.",
        ),
    ]
    for layer, meaning, why in layer_rows:
        lines.append("<tr>")
        lines.append(f'<td class="status">{fmt(layer)}</td>')
        lines.append(f'<td class="wrap">{fmt(meaning)}</td>')
        lines.append(f'<td class="wrap">{fmt(why)}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    lines.append('<div class="panel"><h2>Prefetch Checkpoints</h2><table><thead><tr>')
    for col in ("Checkpoint", "Simple meaning", "Why it matters"):
        lines.append(f"<th>{html.escape(col)}</th>")
    lines.append("</tr></thead><tbody>")
    checkpoint_help = [
        (
            "KV copy ready before replay",
            "The lightweight green KV-copy telemetry ended before the black replay-due line.",
            "This is the scalable checkpoint for larger runs where torch.profiler is off.",
        ),
        (
            "CUDA copy ready before replay",
            "The green HtoD copy ended before the black replay-due line.",
            "This proves profiler-visible GPU copy work happened early enough, but only for the copied slice we attributed.",
        ),
        (
            "Full hint done before replay",
            "The whole purple hint request finished before replay resumed.",
            "This catches cases where the copy started early but the normal SGLang request path was still busy.",
        ),
        (
            "Replay reloaded KV",
            "The real replay request still triggered SGLang KV load-back events.",
            "This catches cases where prefetched KV was incomplete, evicted, or not enough for the replay.",
        ),
    ]
    for checkpoint, meaning, why in checkpoint_help:
        lines.append("<tr>")
        lines.append(f'<td class="status">{fmt(checkpoint)}</td>')
        lines.append(f'<td class="wrap">{fmt(meaning)}</td>')
        lines.append(f'<td class="wrap">{fmt(why)}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")

    lines.append('<div class="panel"><h2>Checkpoint Results Per Session</h2><table><thead><tr>')
    checkpoint_columns = [
        "session_id",
        "kv_copy_ready_before_replay",
        "cuda_copy_ready_before_replay",
        "full_hint_done_before_replay",
        "replay_reloaded_kv",
        "resume_load_count",
        "hint_outcome",
        "checkpoint_result",
    ]
    for col in checkpoint_columns:
        lines.append(f"<th>{html.escape(col)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in selected_rows:
        lines.append("<tr>")
        lines.append(f"<td>{fmt(row.get('session_id', ''))}</td>")
        lines.append(f'<td class="{checkpoint_class(row.get("kv_copy_ready_before_replay"))}">{yes_no(row.get("kv_copy_ready_before_replay"))}</td>')
        lines.append(f'<td class="{checkpoint_class(row.get("cuda_copy_ready_before_replay"))}">{yes_no(row.get("cuda_copy_ready_before_replay"))}</td>')
        lines.append(f'<td class="{checkpoint_class(row.get("full_hint_done_before_replay"))}">{yes_no(row.get("full_hint_done_before_replay"))}</td>')
        lines.append(f'<td class="{checkpoint_class(row.get("replay_reloaded_kv"), good_when_yes=False)}">{yes_no(row.get("replay_reloaded_kv"))}</td>')
        lines.append(f"<td>{fmt(row.get('resume_load_count', ''))}</td>")
        lines.append(f"<td>{fmt(row.get('hint_outcome', ''))}</td>")
        lines.append(f"<td>{fmt(row.get('checkpoint_result', ''))}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table></div>")

    lines.append('<div class="panel"><h2>Key Observations Per Session</h2><table><thead><tr>')
    for col in ("session_id", "status", "what happened", "deduction and evidence"):
        lines.append(f"<th>{html.escape(col)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in selected_rows:
        status, observation, deduction = session_observation(row)
        if row.get("late_prefetch"):
            status_class = "bad"
        elif int(row.get("torch_h2d_copy_events") or 0) > 0 and not row.get("late_prefetch"):
            status_class = "good"
        else:
            status_class = "warn"
        lines.append("<tr>")
        lines.append(f"<td>{fmt(row.get('session_id', ''))}</td>")
        lines.append(f'<td class="status {status_class}">{fmt(status)}</td>')
        lines.append(f'<td class="wrap">{fmt(observation)}</td>')
        lines.append(f'<td class="wrap">{fmt(deduction)}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    lines.append('<div class="panel"><h2>Session Details</h2><table><thead><tr>')
    for col in columns:
        lines.append(f"<th>{html.escape(col)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in selected_rows:
        cls = ' class="bad"' if row.get("late_prefetch") else ""
        lines.append(f"<tr{cls}>")
        for col in columns:
            lines.append(f"<td>{fmt(row.get(col, ''))}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody></table></div>", "</body>", "</html>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-session agentic KV prefetch timeline.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--copy-csv", required=True)
    parser.add_argument("--telemetry-jsonl")
    parser.add_argument("--profile-dir")
    parser.add_argument("--outcome-csv")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--max-sessions", type=int, default=12)
    args = parser.parse_args()

    events = read_jsonl(Path(args.trace))
    copy_rows = read_copy_rows(Path(args.copy_csv))
    telemetry_windows = read_copy_telemetry(Path(args.telemetry_jsonl)) if args.telemetry_jsonl else []
    profiler_coverage = read_profiler_coverage(Path(args.profile_dir)) if args.profile_dir else []
    outcome_rows = read_outcome_rows(Path(args.outcome_csv)) if args.outcome_csv else {}
    rows, timeline = build_rows(events, copy_rows, profiler_coverage, outcome_rows, telemetry_windows)
    write_csv(Path(args.out_csv), rows)
    write_json(Path(args.out_json), rows, timeline)
    write_html(Path(args.out_html), rows, timeline, args.max_sessions)
    print(f"Wrote timeline CSV to {args.out_csv}")
    print(f"Wrote timeline JSON to {args.out_json}")
    print(f"Wrote timeline HTML to {args.out_html}")
    print(f"Sessions: {len(rows)}")
    print(f"Late prefetch sessions: {sum(1 for row in rows if row.get('late_prefetch') is True)}")


if __name__ == "__main__":
    main()
