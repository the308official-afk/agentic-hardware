#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPLAY_FRICTION_COLUMNS = [
    "case_id",
    "mode",
    "request_id",
    "session_id",
    "request_group",
    "tool_wait_step",
    "tool_wait_class",
    "tool_wait_ms",
    "replay_due_abs_ms",
    "replay_due_ms",
    "request_start_ms",
    "sglang_receive_ms",
    "first_token_ms",
    "request_end_ms",
    "due_to_request_start_ms",
    "due_to_sglang_receive_ms",
    "sglang_receive_to_first_token_ms",
    "ttft_ms",
    "lateness_ms",
    "local_pending_before_acquire",
    "local_inflight_before_acquire",
    "local_queue_wait_ms",
    "replay_requests_already_submitted_at_due",
    "replay_requests_still_active_at_due",
    "later_due_replays_already_started",
    "lower_due_work_ahead_inferred",
    "nearest_scheduler_waiting_queue_len",
    "nearest_scheduler_running_batch_size",
    "nearest_scheduler_kv_usage_pct",
    "nearest_scheduler_snapshot_delta_ms",
    "sglang_priority",
    "controller_replay_rank",
    "controller_priority_preserved",
    "prefill_full_input_tokens",
    "prefill_cached_prefix_tokens",
    "prefill_uncached_token_count",
    "cache_hit_ratio",
    "host_hit_tokens",
    "load_back_events_between_due_and_first_token",
    "h2d_events_between_due_and_first_token",
    "evict_events_in_5s_before_due",
    "scheduler_friction_score",
    "memory_friction_score",
    "dominant_friction",
    "friction_explanation",
    "observation_level",
    "blocker_snapshot_available",
    "blocker_snapshot_events",
    "blocker_pending_ahead_count",
    "blocker_inflight_count",
    "blocker_pending_ahead_ids",
    "blocker_inflight_ids",
    "blocker_reasonable_count",
    "blocker_questionable_count",
    "blocker_avoidable_count",
    "blocker_unknown_count",
    "blocker_reasonable_ids",
    "blocker_questionable_ids",
    "blocker_avoidable_ids",
    "blocker_unknown_ids",
    "blocker_snapshot_build_ms_total",
    "blocker_snapshot_build_ms_max",
    "blocker_snapshot_bytes_total",
]

SUMMARY_COLUMNS = [
    "case_id",
    "mode",
    "requests",
    "avg_ttft_ms",
    "avg_lateness_ms",
    "avg_scheduler_friction_score",
    "avg_memory_friction_score",
    "requests_with_local_queue",
    "requests_with_inflight_at_submit",
    "requests_with_lower_due_work_ahead",
    "requests_with_uncached_prefill",
    "requests_with_load_back_or_h2d",
    "requests_with_recent_eviction",
    "dominant_scheduler",
    "dominant_memory",
    "dominant_mixed",
    "dominant_low_observed_friction",
    "dominant_unknown",
    "requests_with_blocker_snapshots",
    "avg_blocker_snapshot_build_ms",
    "max_blocker_snapshot_build_ms",
    "total_blocker_snapshot_bytes",
    "avg_reasonable_blockers",
    "avg_questionable_blockers",
    "avg_avoidable_blockers",
]

LOAD_BACK_EVENTS = {
    "hiradix.init_load_back.start",
    "hiradix.init_load_back.end",
    "hiradix.load_back.start",
    "hiradix.load_back.end",
    "hicache.load.start",
    "hicache.load.end",
}
H2D_EVENTS = {
    "hostpool.load_to_device_per_layer.start",
    "hostpool.load_to_device_per_layer.end",
}
EVICT_EVENTS = {
    "hiradix.evict.start",
    "hiradix.evict.end",
    "hiradix.evict_device.start",
    "hiradix.evict_device.end",
    "hiradix.evict_host.start",
    "hiradix.evict_host.end",
    "hicache.evict_device.start",
    "hicache.evict_device.end",
    "hicache.evict_host.start",
    "hicache.evict_host.end",
}
SCHEDULER_SNAPSHOT_EVENTS = {
    "kv_telemetry.scheduler.start",
    "kv_telemetry.scheduler.end",
    "kv_telemetry.request_stage",
}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f):
        return None
    return value_f


def as_int(value: Any) -> int | None:
    value_f = as_float(value)
    if value_f is None:
        return None
    return int(round(value_f))


def as_boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def numeric_or_blank(value: Any) -> Any:
    value_i = as_int(value)
    return value_i if value_i is not None else ""


def fmt(value: Any, digits: int = 1) -> str:
    value_f = as_float(value)
    if value_f is None:
        return ""
    return f"{value_f:.{digits}f}"


def fmt_seconds(value_ms: Any) -> str:
    value_f = as_float(value_ms)
    if value_f is None:
        return "unknown"
    return f"{value_f / 1000.0:.1f}s"


def fmt_ms_or_seconds(value_ms: Any) -> str:
    value_f = as_float(value_ms)
    if value_f is None:
        return "unknown"
    if abs(value_f) >= 1000.0:
        return f"{value_f / 1000.0:.1f}s"
    if abs(value_f) < 1.0:
        return f"{value_f:.3f}ms"
    return f"{value_f:.0f}ms"


def fmt_delta_ms(value_ms: Any, *, good_when_positive: bool = True) -> str:
    value_f = as_float(value_ms)
    if value_f is None:
        return "unknown"
    direction = "improved" if (value_f >= 0) == good_when_positive else "got worse"
    return f"{direction} by {fmt_ms_or_seconds(abs(value_f))}"


def event_time_ms(row: dict[str, Any], workload_start_ts_ns: int | None) -> float | None:
    offset = as_float(row.get("offset_ms"))
    if offset is not None:
        return offset
    snapshot_offset = as_float(row.get("snapshot_offset_ms"))
    if snapshot_offset is not None:
        return snapshot_offset
    ts_ns = as_int(row.get("ts_ns"))
    if ts_ns is not None and workload_start_ts_ns is not None:
        return (ts_ns - workload_start_ts_ns) / 1_000_000.0
    return None


def local_case_dir(row: dict[str, Any], run_root: Path | None) -> Path | None:
    case_id = str(row.get("case_id") or "")
    if run_root and case_id:
        candidate = run_root / case_id
        if candidate.exists():
            return candidate
    raw = str(row.get("case_dir") or "")
    if raw:
        candidate = Path(raw)
        if candidate.exists():
            return candidate
    return None


def first_nonempty(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return ""


class CaseTrace:
    def __init__(self, case_dir: Path | None) -> None:
        self.case_dir = case_dir
        self.rows: list[dict[str, Any]] = []
        self.gateway_rows: list[dict[str, Any]] = []
        self.workload_start_ts_ns: int | None = None
        self.request_events: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.harness_inputs: dict[str, dict[str, Any]] = {}
        self.scheduler_snapshots: list[dict[str, Any]] = []
        self.load_back_times: list[float] = []
        self.h2d_times: list[float] = []
        self.evict_times: list[float] = []
        self.cache_events_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.prefill_events_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.blocker_snapshots_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if case_dir is not None:
            self._load(case_dir)

    def _load(self, case_dir: Path) -> None:
        self.rows = read_jsonl(case_dir / "m27_trace.jsonl")
        self.gateway_rows = read_jsonl(case_dir / "harness_gateway_events.jsonl")
        for row in self.rows:
            if row.get("event") == "m27.workload_start":
                self.workload_start_ts_ns = as_int(row.get("ts_ns"))
                break
        for row in self.rows:
            event = str(row.get("event") or "")
            when = event_time_ms(row, self.workload_start_ts_ns)
            request_id = str(first_nonempty(row, ("request_id", "label", "expected_replay_request_id")) or "")
            if event in {"m27.request.submitted", "m27.request.start", "m27.request.end"} and request_id:
                self.request_events[request_id][event] = row
            elif event == "m27.harness.request_input" and request_id:
                self.harness_inputs[request_id] = row
            elif event in SCHEDULER_SNAPSHOT_EVENTS and when is not None:
                self.scheduler_snapshots.append({**row, "_offset_ms": when})
            elif event in LOAD_BACK_EVENTS and when is not None:
                self.load_back_times.append(when)
            elif event in H2D_EVENTS and when is not None:
                self.h2d_times.append(when)
            elif event in EVICT_EVENTS and when is not None:
                self.evict_times.append(when)
            elif event == "kv_telemetry.cache.end" and request_id:
                self.cache_events_by_request[request_id].append(row)
            elif event == "kv_telemetry.prefill.end":
                self._index_prefill_event(row)
            elif event == "m27.replay_blocker_snapshot" and request_id:
                row_with_offset = dict(row)
                if when is not None:
                    row_with_offset["_offset_ms"] = when
                self.blocker_snapshots_by_request[request_id].append(row_with_offset)
        self.scheduler_snapshots.sort(key=lambda item: float(item["_offset_ms"]))

    def _index_prefill_event(self, row: dict[str, Any]) -> None:
        request_id = str(row.get("request_id") or "")
        if request_id:
            self.prefill_events_by_request[request_id].append(row)
        attribution = row.get("request_prefill_attribution")
        if isinstance(attribution, list):
            for item in attribution:
                if not isinstance(item, dict):
                    continue
                rid = str(first_nonempty(item, ("request_id", "rid", "label")) or "")
                if rid:
                    self.prefill_events_by_request[rid].append(row)

    def request_row(self, request_id: str, event: str) -> dict[str, Any]:
        return self.request_events.get(request_id, {}).get(event, {})

    def synthetic_request_row(self, request_id: str) -> dict[str, Any]:
        merged: dict[str, Any] = {"request_id": request_id}
        for source in (
            self.harness_inputs.get(request_id, {}),
            self.request_row(request_id, "m27.request.submitted"),
            self.request_row(request_id, "m27.request.start"),
            self.request_row(request_id, "m27.request.end"),
        ):
            for key, value in source.items():
                if value not in (None, "", [], {}) and key not in merged:
                    merged[key] = value
        start = self.request_row(request_id, "m27.request.start")
        end = self.request_row(request_id, "m27.request.end")
        if start.get("ts_ns") and "request_start_ts_ns" not in merged:
            merged["request_start_ts_ns"] = start.get("ts_ns")
        if end.get("ts_ns") and "request_end_ts_ns" not in merged:
            merged["request_end_ts_ns"] = end.get("ts_ns")
        deadline_offset_ms = as_float(merged.get("deadline_offset_ms"))
        if deadline_offset_ms is not None and self.workload_start_ts_ns is not None and "replay_due_ts_ns" not in merged:
            merged["replay_due_ts_ns"] = self.workload_start_ts_ns + int(deadline_offset_ms * 1_000_000)
        return merged

    def nearest_scheduler_snapshot(self, when_ms: float | None) -> dict[str, Any]:
        if when_ms is None or not self.scheduler_snapshots:
            return {}
        return min(
            self.scheduler_snapshots,
            key=lambda row: abs(float(row.get("_offset_ms") or 0.0) - when_ms),
        )

    @staticmethod
    def count_between(times: list[float], start_ms: float | None, end_ms: float | None) -> int:
        if start_ms is None or end_ms is None:
            return 0
        lo = min(start_ms, end_ms)
        hi = max(start_ms, end_ms)
        return sum(1 for value in times if lo <= value <= hi)


def derived_replay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if not request_id:
            continue
        if str(row.get("has_replay_deadline") or "").lower() not in {"true", "yes", "1"}:
            continue
        if str(row.get("phase") or "") not in {"replay", "pressure_filler"}:
            continue
        out.append(row)
    return out


def count_replay_context(row: dict[str, Any], case_rows: list[dict[str, Any]]) -> dict[str, int]:
    due = as_float(row.get("replay_due_ts_ns"))
    start = as_float(row.get("request_start_ts_ns"))
    first = as_float(row.get("first_token_ts_ns"))
    request_id = str(row.get("request_id") or "")
    already_submitted = 0
    active_at_due = 0
    later_due_started = 0
    for other in case_rows:
        other_id = str(other.get("request_id") or "")
        if not other_id or other_id == request_id:
            continue
        other_start = as_float(other.get("request_start_ts_ns"))
        other_end = as_float(other.get("request_end_ts_ns"))
        other_due = as_float(other.get("replay_due_ts_ns"))
        if due is not None and other_start is not None and other_start <= due:
            already_submitted += 1
            if other_end is None or other_end > due:
                active_at_due += 1
        if (
            due is not None
            and start is not None
            and other_due is not None
            and other_start is not None
            and other_due > due
            and other_start < start
            and (first is None or other_start < first)
        ):
            later_due_started += 1
    return {
        "replay_requests_already_submitted_at_due": already_submitted,
        "replay_requests_still_active_at_due": active_at_due,
        "later_due_replays_already_started": later_due_started,
    }


def list_from_snapshot(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split("|") if item.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
    return []


def join_ids(values: Iterable[str]) -> str:
    return " | ".join(str(value) for value in values if str(value))


def classify_blocker(
    victim_row: dict[str, Any],
    blocker_id: str,
    rows_by_request: dict[str, dict[str, Any]],
) -> str:
    blocker = rows_by_request.get(blocker_id)
    if not blocker:
        return "unknown"
    victim_due = as_float(victim_row.get("replay_due_ts_ns"))
    victim_start = as_float(victim_row.get("request_start_ts_ns"))
    blocker_due = as_float(blocker.get("replay_due_ts_ns"))
    blocker_start = as_float(blocker.get("request_start_ts_ns"))
    blocker_end = as_float(blocker.get("request_end_ts_ns"))
    if victim_due is None:
        return "unknown"
    if blocker_due is not None and blocker_due <= victim_due:
        return "reasonable"
    if blocker_end is not None and blocker_end <= victim_due:
        return "reasonable"
    if blocker_due is None:
        return "avoidable"
    if victim_start is not None and blocker_start is not None and blocker_start < victim_start:
        return "questionable"
    return "avoidable"


def blocker_context(
    row: dict[str, Any],
    case_rows: list[dict[str, Any]],
    trace: CaseTrace,
) -> dict[str, Any]:
    snapshots = trace.blocker_snapshots_by_request.get(str(row.get("request_id") or ""), [])
    if not snapshots:
        return {
            "blocker_snapshot_available": "no",
            "blocker_snapshot_events": "",
            "blocker_pending_ahead_count": "",
            "blocker_inflight_count": "",
            "blocker_pending_ahead_ids": "",
            "blocker_inflight_ids": "",
            "blocker_reasonable_count": "",
            "blocker_questionable_count": "",
            "blocker_avoidable_count": "",
            "blocker_unknown_count": "",
            "blocker_reasonable_ids": "",
            "blocker_questionable_ids": "",
            "blocker_avoidable_ids": "",
            "blocker_unknown_ids": "",
            "blocker_snapshot_build_ms_total": "",
            "blocker_snapshot_build_ms_max": "",
            "blocker_snapshot_bytes_total": "",
        }
    # before_acquire matches the historical local pending/in-flight count. Fall
    # back to the first available snapshot when older traces only have one event.
    primary = next((item for item in snapshots if item.get("snapshot_event") == "before_acquire"), snapshots[0])
    pending_ids = list_from_snapshot(primary.get("pending_ahead_ids"))
    inflight_ids = list_from_snapshot(primary.get("inflight_snapshot_ids"))
    blocker_ids = []
    for blocker_id in [*pending_ids, *inflight_ids]:
        if blocker_id and blocker_id not in blocker_ids and blocker_id != row.get("request_id"):
            blocker_ids.append(blocker_id)
    rows_by_request = {str(item.get("request_id") or ""): item for item in case_rows if item.get("request_id")}
    for blocker_id in blocker_ids:
        if blocker_id not in rows_by_request:
            synthetic = trace.synthetic_request_row(blocker_id)
            if len(synthetic) > 1:
                rows_by_request[blocker_id] = synthetic
    classified: dict[str, list[str]] = {
        "reasonable": [],
        "questionable": [],
        "avoidable": [],
        "unknown": [],
    }
    for blocker_id in blocker_ids:
        classified[classify_blocker(row, blocker_id, rows_by_request)].append(blocker_id)
    build_times = [value for value in (as_float(item.get("snapshot_build_ms")) for item in snapshots) if value is not None]
    snapshot_bytes = sum(len(json.dumps(item, sort_keys=True, default=str)) for item in snapshots)
    events = [str(item.get("snapshot_event") or "") for item in snapshots if item.get("snapshot_event")]
    return {
        "blocker_snapshot_available": "yes",
        "blocker_snapshot_events": join_ids(events),
        "blocker_pending_ahead_count": as_int(primary.get("pending_ahead_count")) or len(pending_ids),
        "blocker_inflight_count": as_int(primary.get("inflight_snapshot_count")) or len(inflight_ids),
        "blocker_pending_ahead_ids": join_ids(pending_ids),
        "blocker_inflight_ids": join_ids(inflight_ids),
        "blocker_reasonable_count": len(classified["reasonable"]),
        "blocker_questionable_count": len(classified["questionable"]),
        "blocker_avoidable_count": len(classified["avoidable"]),
        "blocker_unknown_count": len(classified["unknown"]),
        "blocker_reasonable_ids": join_ids(classified["reasonable"]),
        "blocker_questionable_ids": join_ids(classified["questionable"]),
        "blocker_avoidable_ids": join_ids(classified["avoidable"]),
        "blocker_unknown_ids": join_ids(classified["unknown"]),
        "blocker_snapshot_build_ms_total": fmt(sum(build_times), 3),
        "blocker_snapshot_build_ms_max": fmt(max(build_times) if build_times else None, 3),
        "blocker_snapshot_bytes_total": snapshot_bytes,
    }


def build_friction_row(
    row: dict[str, Any],
    case_rows: list[dict[str, Any]],
    all_case_rows: list[dict[str, Any]],
    trace: CaseTrace,
) -> dict[str, Any]:
    request_id = str(row.get("request_id") or "")
    blockers = blocker_context(row, all_case_rows, trace)
    due_to_start = as_float(row.get("due_to_request_start_ms"))
    due_to_sglang = as_float(row.get("due_to_sglang_receive_ms"))
    receive_to_first = as_float(row.get("sglang_receive_to_first_token_ms"))
    ttft = as_float(row.get("ttft_ms"))
    lateness = as_float(row.get("replay_debt_ms")) or as_float(row.get("first_token_lateness_ms"))
    replay_due_ms = None
    request_start_ms = None
    sglang_receive_ms = None
    first_token_ms = None
    request_end_ms = None
    due_ns = as_float(row.get("replay_due_ts_ns"))
    due_abs_ms = None
    if due_ns is not None:
        if trace.workload_start_ts_ns is not None:
            due_abs_ms = (due_ns - trace.workload_start_ts_ns) / 1_000_000.0
        start_ns = as_float(row.get("request_start_ts_ns"))
        receive_ns = as_float(row.get("sglang_receive_ts_ns"))
        first_ns = as_float(row.get("first_token_ts_ns"))
        end_ns = as_float(row.get("request_end_ts_ns"))
        request_start_ms = ((start_ns - due_ns) / 1_000_000.0) if start_ns is not None else due_to_start
        sglang_receive_ms = ((receive_ns - due_ns) / 1_000_000.0) if receive_ns is not None else due_to_sglang
        first_token_ms = ((first_ns - due_ns) / 1_000_000.0) if first_ns is not None else lateness
        request_end_ms = ((end_ns - due_ns) / 1_000_000.0) if end_ns is not None else None
        replay_due_ms = 0.0
    request_start_trace = trace.request_row(request_id, "m27.request.start")
    if not request_start_trace:
        request_start_trace = trace.request_row(request_id, "m27.request.submitted")
    harness_input = trace.harness_inputs.get(request_id, {})
    context = count_replay_context(row, case_rows)
    local_pending = as_int(first_nonempty(request_start_trace, ("client_pending_before_acquire", "client_pending_at_submit")))
    local_inflight = as_int(first_nonempty(request_start_trace, ("client_inflight_before_acquire", "client_inflight_at_submit")))
    local_queue_wait = as_float(first_nonempty(request_start_trace, ("client_queue_wait_ms",)))
    scheduler_when = as_float(first_nonempty(request_start_trace, ("offset_ms",)))
    if scheduler_when is None and due_to_start is not None:
        scheduler_when = (due_abs_ms + due_to_start) if due_abs_ms is not None else due_to_start
    scheduler = trace.nearest_scheduler_snapshot(scheduler_when)
    scheduler_delta = ""
    if scheduler and scheduler_when is not None:
        scheduler_delta = abs(float(scheduler.get("_offset_ms") or 0.0) - scheduler_when)
    cached_tokens = as_float(row.get("prefill_cached_prefix_tokens"))
    full_tokens = as_float(row.get("prefill_full_input_tokens"))
    uncached_tokens = as_float(row.get("prefill_uncached_token_count"))
    cache_ratio = ""
    if cached_tokens is not None and full_tokens and full_tokens > 0:
        cache_ratio = cached_tokens / full_tokens
    host_hit_tokens = 0.0
    for cache_event in trace.cache_events_by_request.get(request_id, []):
        host_hit_tokens += as_float(cache_event.get("host_hit_tokens")) or 0.0
    first_token_delta = as_float(first_token_ms)
    window_end = (due_abs_ms + first_token_delta) if due_abs_ms is not None and first_token_delta is not None else None
    load_back_count = trace.count_between(trace.load_back_times, due_abs_ms, window_end)
    h2d_count = trace.count_between(trace.h2d_times, due_abs_ms, window_end)
    evict_count = trace.count_between(trace.evict_times, (due_abs_ms - 5000.0) if due_abs_ms is not None else None, due_abs_ms)
    lower_due_ahead = context["later_due_replays_already_started"]
    local_queue_present = bool((local_pending or 0) > 0 or (local_inflight or 0) > 0 or (local_queue_wait or 0.0) > 1)
    scheduler_score = 0
    if (due_to_start or 0.0) > 1:
        scheduler_score += 1
    if local_queue_present:
        scheduler_score += 1
    if lower_due_ahead > 0:
        scheduler_score += 1
    if as_int(scheduler.get("scheduler_waiting_queue_len")) or 0:
        scheduler_score += 1
    if as_int(scheduler.get("scheduler_running_batch_batch_size")) or 0:
        scheduler_score += 1
    memory_score = 0
    if uncached_tokens is not None and uncached_tokens > 0:
        memory_score += 1
    if isinstance(cache_ratio, float) and cache_ratio < 0.8:
        memory_score += 1
    if host_hit_tokens > 0:
        memory_score += 1
    if load_back_count or h2d_count:
        memory_score += 1
    if evict_count:
        memory_score += 1
    if scheduler_score > memory_score and scheduler_score > 0:
        dominant = "scheduler"
    elif memory_score > scheduler_score and memory_score > 0:
        dominant = "memory"
    elif scheduler_score and memory_score:
        dominant = "mixed"
    elif trace.case_dir is None:
        dominant = "unknown"
    else:
        dominant = "low_observed_friction"
    explanations: list[str] = []
    if local_pending:
        explanations.append(f"{local_pending} local pending before submit")
    if local_inflight:
        explanations.append(f"{local_inflight} local in-flight before submit")
    if lower_due_ahead:
        explanations.append(f"{lower_due_ahead} later-due replay(s) started earlier")
    if uncached_tokens is not None and uncached_tokens > 0:
        explanations.append(f"{uncached_tokens:.0f} uncached prefill token(s)")
    if host_hit_tokens:
        explanations.append(f"{host_hit_tokens:.0f} host-hit token(s)")
    if load_back_count or h2d_count:
        explanations.append(f"{load_back_count + h2d_count} load-back/H2D event(s) before first token")
    if evict_count:
        explanations.append(f"{evict_count} eviction event(s) in 5s before due")
    if not explanations:
        explanations.append("no strong friction directly observed" if trace.case_dir is not None else "trace-rich artifacts unavailable")
    priority = first_nonempty(row, ("sglang_priority",))
    if not priority:
        priority = first_nonempty(request_start_trace, ("sglang_priority", "controller_sglang_priority"))
    controller_preserved = "unknown"
    mode = str(row.get("mode") or "")
    if mode.startswith("controller"):
        controller_preserved = "yes" if priority not in (None, "") else "no"
    return {
        "case_id": row.get("case_id", ""),
        "mode": mode,
        "request_id": request_id,
        "session_id": row.get("session_id", ""),
        "request_group": row.get("request_group", ""),
        "tool_wait_step": row.get("tool_wait_step", ""),
        "tool_wait_class": row.get("tool_wait_class", ""),
        "tool_wait_ms": row.get("tool_wait_ms", ""),
        "replay_due_abs_ms": fmt(due_abs_ms),
        "replay_due_ms": fmt(replay_due_ms),
        "request_start_ms": fmt(request_start_ms),
        "sglang_receive_ms": fmt(sglang_receive_ms),
        "first_token_ms": fmt(first_token_ms),
        "request_end_ms": fmt(request_end_ms),
        "due_to_request_start_ms": fmt(due_to_start),
        "due_to_sglang_receive_ms": fmt(due_to_sglang),
        "sglang_receive_to_first_token_ms": fmt(receive_to_first),
        "ttft_ms": fmt(ttft),
        "lateness_ms": fmt(lateness),
        "local_pending_before_acquire": local_pending if local_pending is not None else "",
        "local_inflight_before_acquire": local_inflight if local_inflight is not None else "",
        "local_queue_wait_ms": fmt(local_queue_wait),
        **context,
        "lower_due_work_ahead_inferred": "yes" if lower_due_ahead else "no",
        "nearest_scheduler_waiting_queue_len": numeric_or_blank(
            first_nonempty(scheduler, ("scheduler_waiting_queue_len",))
        ),
        "nearest_scheduler_running_batch_size": numeric_or_blank(
            first_nonempty(scheduler, ("scheduler_running_batch_batch_size", "request_count"))
        ),
        "nearest_scheduler_kv_usage_pct": fmt(first_nonempty(scheduler, ("kv_pool_usage_pct",))),
        "nearest_scheduler_snapshot_delta_ms": fmt(scheduler_delta),
        "sglang_priority": priority,
        "controller_replay_rank": row.get("controller_replay_rank", ""),
        "controller_priority_preserved": controller_preserved,
        "prefill_full_input_tokens": fmt(full_tokens, 0),
        "prefill_cached_prefix_tokens": fmt(cached_tokens, 0),
        "prefill_uncached_token_count": fmt(uncached_tokens, 0),
        "cache_hit_ratio": fmt(cache_ratio, 3),
        "host_hit_tokens": fmt(host_hit_tokens, 0),
        "load_back_events_between_due_and_first_token": load_back_count,
        "h2d_events_between_due_and_first_token": h2d_count,
        "evict_events_in_5s_before_due": evict_count,
        "scheduler_friction_score": scheduler_score,
        "memory_friction_score": memory_score,
        "dominant_friction": dominant,
        "friction_explanation": "; ".join(explanations),
        "observation_level": "trace_rich" if trace.case_dir is not None else "summary_csv_only",
        **blockers,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("case_id") or ""), str(row.get("mode") or ""))].append(row)
    out: list[dict[str, Any]] = []
    for (case_id, mode), group in sorted(grouped.items()):
        count = len(group)
        dom = Counter(str(row.get("dominant_friction") or "unknown") for row in group)

        def avg(key: str) -> str:
            vals = [value for value in (as_float(row.get(key)) for row in group) if value is not None]
            return f"{(sum(vals) / len(vals)):.1f}" if vals else ""

        def avg_precise(key: str) -> str:
            vals = [value for value in (as_float(row.get(key)) for row in group) if value is not None]
            return f"{(sum(vals) / len(vals)):.3f}" if vals else ""

        out.append(
            {
                "case_id": case_id,
                "mode": mode,
                "requests": count,
                "avg_ttft_ms": avg("ttft_ms"),
                "avg_lateness_ms": avg("lateness_ms"),
                "avg_scheduler_friction_score": avg("scheduler_friction_score"),
                "avg_memory_friction_score": avg("memory_friction_score"),
                "requests_with_local_queue": sum(
                    1
                    for row in group
                    if (as_int(row.get("local_pending_before_acquire")) or 0)
                    or (as_int(row.get("local_inflight_before_acquire")) or 0)
                    or (as_float(row.get("local_queue_wait_ms")) or 0.0) > 1
                ),
                "requests_with_inflight_at_submit": sum(
                    1 for row in group if (as_int(row.get("local_inflight_before_acquire")) or 0) > 0
                ),
                "requests_with_lower_due_work_ahead": sum(
                    1 for row in group if str(row.get("lower_due_work_ahead_inferred") or "") == "yes"
                ),
                "requests_with_uncached_prefill": sum(
                    1 for row in group if (as_float(row.get("prefill_uncached_token_count")) or 0.0) > 0
                ),
                "requests_with_load_back_or_h2d": sum(
                    1
                    for row in group
                    if (as_int(row.get("load_back_events_between_due_and_first_token")) or 0)
                    or (as_int(row.get("h2d_events_between_due_and_first_token")) or 0)
                ),
                "requests_with_recent_eviction": sum(
                    1 for row in group if (as_int(row.get("evict_events_in_5s_before_due")) or 0) > 0
                ),
                "dominant_scheduler": dom["scheduler"],
                "dominant_memory": dom["memory"],
                "dominant_mixed": dom["mixed"],
                "dominant_low_observed_friction": dom["low_observed_friction"],
                "dominant_unknown": dom["unknown"],
                "requests_with_blocker_snapshots": sum(
                    1 for row in group if str(row.get("blocker_snapshot_available") or "") == "yes"
                ),
                "avg_blocker_snapshot_build_ms": avg_precise("blocker_snapshot_build_ms_total"),
                "max_blocker_snapshot_build_ms": (
                    f"{max(values):.3f}"
                    if (
                        values := [
                            value
                            for value in (as_float(row.get("blocker_snapshot_build_ms_max")) for row in group)
                            if value is not None
                        ]
                    )
                    else ""
                ),
                "total_blocker_snapshot_bytes": sum(
                    as_int(row.get("blocker_snapshot_bytes_total")) or 0 for row in group
                ),
                "avg_reasonable_blockers": avg("blocker_reasonable_count"),
                "avg_questionable_blockers": avg("blocker_questionable_count"),
                "avg_avoidable_blockers": avg("blocker_avoidable_count"),
            }
        )
    return out


def avg_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [value for value in (as_float(row.get(key)) for row in rows) if value is not None]
    return (sum(vals) / len(vals)) if vals else None


def preferred_modes(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    modes = sorted({str(row.get("mode") or "") for row in rows if row.get("mode")})
    baseline = "no_prefetch" if "no_prefetch" in modes else (modes[0] if modes else None)
    controller = next((mode for mode in modes if mode.startswith("controller")), None)
    if controller is None:
        controller = next((mode for mode in modes if mode != baseline), None)
    return baseline, controller


def paired_rows_by_request(
    rows: list[dict[str, Any]], baseline_mode: str | None, controller_mode: str | None
) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        request_id = str(row.get("request_id") or "")
        mode = str(row.get("mode") or "")
        if not request_id or mode not in {baseline_mode, controller_mode}:
            continue
        pairs[request_id][mode] = row
    return pairs


def select_reader_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_mode, controller_mode = preferred_modes(rows)
    if not controller_mode:
        return []
    pairs = paired_rows_by_request(rows, baseline_mode, controller_mode)
    complete_pairs = {
        request_id: pair
        for request_id, pair in pairs.items()
        if controller_mode in pair and (baseline_mode is None or baseline_mode in pair)
    }

    def improvement_value(request_id: str, key: str) -> float:
        pair = complete_pairs[request_id]
        baseline = pair.get(baseline_mode or "")
        controller = pair[controller_mode]
        if not baseline:
            return float("-inf")
        before = as_float(baseline.get(key))
        after = as_float(controller.get(key))
        if before is None or after is None:
            return float("-inf")
        return before - after

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(label: str, request_id: str | None) -> None:
        if not request_id or request_id in seen:
            return
        pair = complete_pairs.get(request_id)
        if not pair:
            return
        seen.add(request_id)
        selected.append({"label": label, "request_id": request_id, "pair": pair})

    if complete_pairs:
        add("Biggest TTFT improvement", max(complete_pairs, key=lambda rid: improvement_value(rid, "ttft_ms")))
        add("Biggest lateness improvement", max(complete_pairs, key=lambda rid: improvement_value(rid, "lateness_ms")))
        add(
            "Worst remaining lateness under controller",
            max(
                complete_pairs,
                key=lambda rid: as_float(complete_pairs[rid][controller_mode].get("lateness_ms")) or float("-inf"),
            ),
        )
        add(
            "Highest scheduler friction under controller",
            max(
                complete_pairs,
                key=lambda rid: (
                    as_float(complete_pairs[rid][controller_mode].get("scheduler_friction_score")) or float("-inf"),
                    as_float(complete_pairs[rid][controller_mode].get("lateness_ms")) or float("-inf"),
                ),
            ),
        )
        add(
            "Highest memory friction under controller",
            max(
                complete_pairs,
                key=lambda rid: (
                    as_float(complete_pairs[rid][controller_mode].get("memory_friction_score")) or float("-inf"),
                    as_float(complete_pairs[rid][controller_mode].get("lateness_ms")) or float("-inf"),
                ),
            ),
        )
    return selected[:5]


def simple_meaning(row: dict[str, Any]) -> str:
    dominant = str(row.get("dominant_friction") or "")
    if dominant == "scheduler":
        return (
            "The controller moved this replay forward, but the system still had scheduling pressure. "
            "In simple terms, it was still waiting behind other work."
        )
    if dominant == "memory":
        return (
            "The controller helped scheduling, but memory was still the main drag. "
            "In simple terms, some KV/cache work still had to be rebuilt or moved before the first token."
        )
    if dominant == "mixed":
        return (
            "The controller helped, but two things were still happening at once: queue pressure and memory work. "
            "In simple terms, the request was moved forward, but the kitchen was still crowded and ingredients were still being fetched."
        )
    if dominant == "low_observed_friction":
        return "The trace did not show a strong remaining bottleneck for this request."
    return "The available traces were not rich enough to name the main remaining bottleneck."


def sentence_from_counts(row: dict[str, Any]) -> str:
    if str(row.get("blocker_snapshot_available") or "") == "yes":
        pending = as_int(row.get("blocker_pending_ahead_count")) or 0
        inflight = as_int(row.get("blocker_inflight_count")) or 0
        reasonable = as_int(row.get("blocker_reasonable_count")) or 0
        questionable = as_int(row.get("blocker_questionable_count")) or 0
        avoidable = as_int(row.get("blocker_avoidable_count")) or 0
        unknown = as_int(row.get("blocker_unknown_count")) or 0
        return (
            f"When replay became due, the trace identified {pending} pending request(s) ahead "
            f"and {inflight} in-flight request(s). Among the captured blocker IDs, "
            f"{reasonable} looked reasonable, {questionable} looked questionable, "
            f"{avoidable} looked avoidable, and {unknown} could not be classified."
        )
    local_pending = as_int(row.get("local_pending_before_acquire")) or 0
    local_inflight = as_int(row.get("local_inflight_before_acquire")) or 0
    uncached = as_int(row.get("prefill_uncached_token_count")) or 0
    load_back = as_int(row.get("load_back_events_between_due_and_first_token")) or 0
    h2d = as_int(row.get("h2d_events_between_due_and_first_token")) or 0
    evictions = as_int(row.get("evict_events_in_5s_before_due")) or 0
    later_due = as_int(row.get("later_due_replays_already_started")) or 0
    parts: list[str] = []
    if local_pending or local_inflight:
        parts.append(f"{local_pending} local pending request(s) and {local_inflight} in-flight request(s)")
    if later_due:
        parts.append(f"{later_due} later-due replay(s) had already started")
    if uncached:
        parts.append(f"{uncached} uncached prefill token(s)")
    if load_back or h2d:
        parts.append(f"{load_back + h2d} H2D/load-back event(s) before first token")
    if evictions:
        parts.append(f"{evictions} recent eviction event(s)")
    if not parts:
        return "The trace did not show a strong non-ideal condition for this request."
    return "When replay became due, the trace showed " + ", ".join(parts) + "."


def render_example_card(
    example: dict[str, Any], baseline_mode: str | None, controller_mode: str | None
) -> str:
    request_id = str(example["request_id"])
    pair = example["pair"]
    baseline = pair.get(baseline_mode or "")
    controller = pair.get(controller_mode or "")
    before_ttft = baseline.get("ttft_ms") if baseline else None
    before_late = baseline.get("lateness_ms") if baseline else None
    after_ttft = controller.get("ttft_ms") if controller else None
    after_late = controller.get("lateness_ms") if controller else None
    ttft_delta = (as_float(before_ttft) - as_float(after_ttft)) if baseline and controller and as_float(before_ttft) is not None and as_float(after_ttft) is not None else None
    late_delta = (as_float(before_late) - as_float(after_late)) if baseline and controller and as_float(before_late) is not None and as_float(after_late) is not None else None

    baseline_text = (
        f"In baseline, TTFT was <strong>{fmt_seconds(before_ttft)}</strong> and lateness was <strong>{fmt_seconds(before_late)}</strong>."
        if baseline
        else "No paired baseline row was available for this request."
    )
    controller_text = (
        f"With the controller, TTFT was <strong>{fmt_seconds(after_ttft)}</strong> and lateness was <strong>{fmt_seconds(after_late)}</strong>."
        if controller
        else "No paired controller row was available for this request."
    )
    changed_parts = []
    if ttft_delta is not None:
        changed_parts.append(f"TTFT {fmt_delta_ms(ttft_delta)}")
    if late_delta is not None:
        changed_parts.append(f"lateness {fmt_delta_ms(late_delta)}")
    changed_text = "; ".join(changed_parts) + "." if changed_parts else "The paired change could not be computed."
    friction_text = sentence_from_counts(controller or {})
    meaning_text = simple_meaning(controller or {})
    blocker_bits: list[str] = []
    if controller and str(controller.get("blocker_snapshot_available") or "") == "yes":
        for label, key in [
            ("Questionable", "blocker_questionable_ids"),
            ("Avoidable", "blocker_avoidable_ids"),
            ("Reasonable", "blocker_reasonable_ids"),
        ]:
            ids = str(controller.get(key) or "")
            if ids:
                blocker_bits.append(f"<p><strong>{label} blocker IDs:</strong> {html.escape(ids)}</p>")
    blocker_html = "\n        ".join(blocker_bits)
    return f"""
      <article class="example-card">
        <div class="example-label">{html.escape(str(example["label"]))}</div>
        <h3>{html.escape(request_id)}</h3>
        <p>{baseline_text}</p>
        <p>{controller_text}</p>
        <p><strong>What changed:</strong> {html.escape(changed_text)}</p>
        <p><strong>Why it was still not perfect:</strong> {html.escape(friction_text)}</p>
        {blocker_html}
        <p><strong>Simple meaning:</strong> {html.escape(meaning_text)}</p>
      </article>
    """


def render_blocker_quality_charts(controller_rows: list[dict[str, Any]]) -> str:
    rows = [row for row in controller_rows if str(row.get("blocker_snapshot_available") or "") == "yes"]
    if not rows:
        return ""
    rows = sorted(
        rows,
        key=lambda row: (
            as_float(row.get("replay_due_abs_ms")) if as_float(row.get("replay_due_abs_ms")) is not None else float("inf"),
            as_float(row.get("request_start_ms")) if as_float(row.get("request_start_ms")) is not None else float("inf"),
            str(row.get("request_id") or ""),
        ),
    )

    def counts(row: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            as_int(row.get("blocker_reasonable_count")) or 0,
            as_int(row.get("blocker_questionable_count")) or 0,
            as_int(row.get("blocker_avoidable_count")) or 0,
            as_int(row.get("blocker_unknown_count")) or 0,
        )

    max_total = max(max(1, sum(counts(row))) for row in rows)
    bars: list[str] = []
    fractions: list[float] = []
    for idx, row in enumerate(rows, start=1):
        reasonable, questionable, avoidable, unknown = counts(row)
        total = reasonable + questionable + avoidable + unknown
        fraction = (avoidable + questionable) / total if total else 0.0
        fractions.append(fraction)
        title = (
            f"#{idx} {row.get('request_id', '')}: "
            f"{reasonable} reasonable, {questionable} questionable, "
            f"{avoidable} avoidable, {unknown} unknown; "
            f"lateness {fmt_seconds(row.get('lateness_ms'))}"
        )

        def segment(value: int, class_name: str) -> str:
            if value <= 0:
                return ""
            height = max(3.0, value / max_total * 100.0)
            return f'<span class="{class_name}" style="height:{height:.2f}%"></span>'

        bars.append(
            f"""
            <div class="blocker-bar" title="{html.escape(title)}" aria-label="{html.escape(title)}">
              {segment(unknown, "seg unknown")}
              {segment(avoidable, "seg avoidable")}
              {segment(questionable, "seg questionable")}
              {segment(reasonable, "seg reasonable")}
            </div>
            """
        )

    width = 900
    height = 220
    pad_l = 36
    pad_r = 18
    pad_t = 16
    pad_b = 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    point_count = max(1, len(fractions) - 1)
    points: list[str] = []
    dots: list[str] = []
    for idx, frac in enumerate(fractions):
        x = pad_l + (idx / point_count) * plot_w if point_count else pad_l
        y = pad_t + (1.0 - frac) * plot_h
        points.append(f"{x:.1f},{y:.1f}")
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2"><title>Request {idx + 1}: avoidable/questionable fraction {frac:.2f}</title></circle>'
        )
    line_svg = f"""
      <svg class="fraction-line" viewBox="0 0 {width} {height}" role="img" aria-label="Avoidable blocker fraction by replay request">
        <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" class="axis" />
        <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" class="axis" />
        <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l + plot_w}" y2="{pad_t}" class="grid" />
        <line x1="{pad_l}" y1="{pad_t + plot_h / 2}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h / 2}" class="grid" />
        <text x="4" y="{pad_t + 4}" class="axis-label">1.0</text>
        <text x="4" y="{pad_t + plot_h / 2 + 4}" class="axis-label">0.5</text>
        <text x="4" y="{pad_t + plot_h + 4}" class="axis-label">0.0</text>
        <polyline points="{' '.join(points)}" />
        {''.join(dots)}
      </svg>
    """
    return f"""
  <h2>Blocker Quality Across Requests</h2>
  <p class="lede">These charts look across controller-mode replay requests in replay-due order. Green blockers were reasonable. Amber/red blockers are where the controller may still have room to improve.</p>
  <div class="note"><strong>How to read this:</strong> if the avoidable fraction line drops to 0.0, that request was only blocked by work that was already due or reasonably in flight. Higher values mean more room for better admission or scheduling.</div>
  <div class="legend">
    <span><i class="legend-swatch reasonable"></i>Reasonable</span>
    <span><i class="legend-swatch questionable"></i>Questionable</span>
    <span><i class="legend-swatch avoidable"></i>Avoidable</span>
    <span><i class="legend-swatch unknown"></i>Unknown</span>
  </div>
  <div class="chart-panel">
    <h3>Per-Request Blocker Quality</h3>
    <div class="blocker-bars">{''.join(bars)}</div>
    <p class="chart-caption">Each vertical bar is one replay request. Bar height is blocker count; color shows blocker quality.</p>
  </div>
  <div class="chart-panel">
    <h3>Avoidable / Questionable Blocker Fraction</h3>
    {line_svg}
    <p class="chart-caption">0.0 means the captured blockers were all reasonable. Higher values mean a larger share looked avoidable or questionable.</p>
  </div>
"""


def render_reader_html(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    report_label: str,
    full_detail_href: str,
) -> str:
    baseline_mode, controller_mode = preferred_modes(rows)
    baseline_rows = [row for row in rows if str(row.get("mode") or "") == baseline_mode]
    controller_rows = [row for row in rows if str(row.get("mode") or "") == controller_mode]
    baseline_ttft = avg_numeric(baseline_rows, "ttft_ms")
    controller_ttft = avg_numeric(controller_rows, "ttft_ms")
    baseline_late = avg_numeric(baseline_rows, "lateness_ms")
    controller_late = avg_numeric(controller_rows, "lateness_ms")
    examples = select_reader_examples(rows)
    controller_summary = next(
        (row for row in summary_rows if str(row.get("mode") or "") == controller_mode),
        {},
    )
    common_causes = [
        (
            "Local queue pressure",
            f"{controller_summary.get('requests_with_local_queue', 0)} request(s) had local pending, in-flight, or local queue wait.",
        ),
        (
            "In-flight work",
            f"{controller_summary.get('requests_with_inflight_at_submit', 0)} request(s) entered while other work was already in flight.",
        ),
        (
            "Uncached prefill",
            f"{controller_summary.get('requests_with_uncached_prefill', 0)} request(s) still had tokens to prefill.",
        ),
        (
            "H2D/load-back movement",
            f"{controller_summary.get('requests_with_load_back_or_h2d', 0)} request(s) saw memory movement before first token.",
        ),
        (
            "Recent eviction",
            f"{controller_summary.get('requests_with_recent_eviction', 0)} request(s) had eviction activity shortly before replay was due.",
        ),
    ]
    causes_html = "".join(
        f"<li><strong>{html.escape(title)}:</strong> {html.escape(text)}</li>" for title, text in common_causes
    )
    blocker_snapshot_count = controller_summary.get("requests_with_blocker_snapshots", 0)
    overhead_html = ""
    if str(blocker_snapshot_count) not in {"", "0"}:
        overhead_html = f"""
  <h2>Instrumentation Guardrail</h2>
  <p>This blocker analysis was generated from bounded request-ID snapshots. The controller-mode summary recorded {html.escape(str(blocker_snapshot_count))} request(s) with blocker snapshots, {html.escape(str(controller_summary.get("total_blocker_snapshot_bytes", "")))} total snapshot bytes, average snapshot build time {html.escape(fmt_ms_or_seconds(controller_summary.get("avg_blocker_snapshot_build_ms")))} and max snapshot build time {html.escape(fmt_ms_or_seconds(controller_summary.get("max_blocker_snapshot_build_ms")))}.</p>
"""
    example_html = "".join(render_example_card(example, baseline_mode, controller_mode) for example in examples)
    blocker_charts_html = render_blocker_quality_charts(controller_rows)
    style = """
    :root { --text: #172033; --muted: #526174; --line: #dbe3ef; --soft: #f8fafc; --accent: #155eef; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: var(--text); background: #ffffff; }
    main { max-width: 1120px; margin: 0 auto; padding: 34px 24px 48px; }
    h1, h2, h3 { color: #101827; line-height: 1.15; }
    h1 { margin: 0 0 10px; font-size: 34px; }
    h2 { margin-top: 30px; font-size: 22px; }
    h3 { margin: 8px 0 10px; font-size: 16px; overflow-wrap: anywhere; }
    p, li { line-height: 1.5; }
    .lede { color: var(--muted); max-width: 920px; }
    .note { background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin: 18px 0; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 18px 0; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }
    .metric-name { color: var(--muted); font-size: 13px; font-weight: 750; }
    .metric-value { font-size: 24px; font-weight: 850; margin-top: 6px; }
    .examples { display: grid; gap: 14px; }
    .example-card { border: 1px solid var(--line); border-radius: 8px; padding: 15px 16px; background: #fff; }
    .example-label { color: var(--accent); font-weight: 850; font-size: 13px; text-transform: uppercase; letter-spacing: .03em; }
    .example-card p { margin: 8px 0; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; color: var(--muted); font-size: 13px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .legend-swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
    .chart-panel { border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; margin: 14px 0; overflow-x: auto; }
    .blocker-bars { min-width: 760px; height: 220px; display: flex; align-items: flex-end; gap: 5px; padding: 8px 0 4px; border-bottom: 1px solid var(--line); }
    .blocker-bar { width: 18px; height: 100%; display: flex; flex-direction: column-reverse; justify-content: flex-start; background: #f8fafc; border-radius: 4px 4px 0 0; overflow: hidden; outline: 1px solid #e5ebf3; }
    .seg { display: block; width: 100%; }
    .reasonable { background: #2e7d55; }
    .questionable { background: #d48a19; }
    .avoidable { background: #c24132; }
    .unknown { background: #94a3b8; }
    .fraction-line { min-width: 760px; width: 100%; height: 260px; }
    .fraction-line .axis { stroke: #94a3b8; stroke-width: 1.2; }
    .fraction-line .grid { stroke: #e5ebf3; stroke-width: 1; }
    .fraction-line polyline { fill: none; stroke: #c24132; stroke-width: 3; }
    .fraction-line circle { fill: #c24132; }
    .axis-label, .chart-caption { fill: var(--muted); color: var(--muted); font-size: 12px; }
    a { color: var(--accent); font-weight: 750; }
    code { background: #edf2f7; padding: 1px 4px; border-radius: 4px; }
    """
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Replay Friction Deep Dive</title>
  <style>{style}</style>
</head>
<body>
<main>
  <h1>Replay Friction Deep Dive</h1>
  <p class="lede">Report label: <code>{html.escape(report_label)}</code>. This page is generated automatically from the experiment traces. It explains, in simple words, why replay requests were still late or slow even when the controller helped.</p>
  <div class="note"><strong>Guardrail:</strong> this is a reader summary. The full per-request table and raw CSV remain in the run report folder.</div>

  <h2>Headline Result</h2>
  <div class="metric-grid">
    <div class="metric">
      <div class="metric-name">Average TTFT</div>
      <div class="metric-value">{html.escape(fmt_seconds(baseline_ttft))} -> {html.escape(fmt_seconds(controller_ttft))}</div>
      <p>Lower is better. This is how long requests waited for their first token after submission.</p>
    </div>
    <div class="metric">
      <div class="metric-name">Average Lateness</div>
      <div class="metric-value">{html.escape(fmt_seconds(baseline_late))} -> {html.escape(fmt_seconds(controller_late))}</div>
      <p>Lower is better. This is how late requests were after their replay became due.</p>
    </div>
    <div class="metric">
      <div class="metric-name">Requests Explained</div>
      <div class="metric-value">{len(controller_rows)}</div>
      <p>The examples below are selected by fixed rules from the controller and baseline rows.</p>
    </div>
  </div>

  {blocker_charts_html}

  <h2>Concrete Request Examples</h2>
  <div class="examples">
    {example_html}
  </div>

  <h2>Common Causes Seen In Controller Mode</h2>
  <ul>{causes_html}</ul>

  {overhead_html}

  <h2>Full Evidence</h2>
  <p><a href="{html.escape(full_detail_href)}">Open the full detailed table</a>. The CSV and JSON versions sit beside that file in the same run report folder.</p>
</main>
</body>
</html>
"""


def render_html(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], report_label: str) -> str:
    def table(columns: list[str], table_rows: list[dict[str, Any]], limit: int | None = None) -> str:
        visible = table_rows if limit is None else table_rows[:limit]
        head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
        body = []
        for row in visible:
            body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    summary_cols = SUMMARY_COLUMNS
    detail_cols = [
        "mode",
        "request_id",
        "request_group",
        "tool_wait_class",
        "ttft_ms",
        "lateness_ms",
        "local_pending_before_acquire",
        "local_inflight_before_acquire",
        "later_due_replays_already_started",
        "nearest_scheduler_waiting_queue_len",
        "prefill_uncached_token_count",
        "load_back_events_between_due_and_first_token",
        "evict_events_in_5s_before_due",
        "dominant_friction",
        "friction_explanation",
        "blocker_snapshot_available",
        "blocker_pending_ahead_count",
        "blocker_inflight_count",
        "blocker_reasonable_count",
        "blocker_questionable_count",
        "blocker_avoidable_count",
        "blocker_pending_ahead_ids",
        "blocker_inflight_ids",
        "blocker_questionable_ids",
        "blocker_avoidable_ids",
        "observation_level",
    ]
    style = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; }
    h1, h2 { color: #101827; }
    .lede { color: #526174; max-width: 980px; line-height: 1.5; }
    .note { background: #f8fafc; border: 1px solid #dbe3ef; border-radius: 8px; padding: 12px 14px; margin: 16px 0; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 14px 0 28px; }
    th, td { border-bottom: 1px solid #e5ebf3; padding: 7px 8px; text-align: left; vertical-align: top; }
    th { background: #f1f5f9; font-weight: 800; position: sticky; top: 0; }
    code { background: #edf2f7; padding: 1px 4px; border-radius: 4px; }
    """
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Replay Friction Deep Dive</title>
  <style>{style}</style>
</head>
<body>
  <h1>Replay Friction Deep Dive</h1>
  <p class="lede">Report label: <code>{html.escape(report_label)}</code>. This page explains, per replay request, what non-ideal scheduler or memory condition was visible around replay time. Values are marked <code>trace_rich</code> when per-case logs were available and <code>summary_csv_only</code> when only report CSVs were present.</p>
  <div class="note"><strong>Interpretation guardrail:</strong> queue and memory friction are best-effort attribution signals. Scheduler queue counts from local harness fields are directly observed; lower-due work ahead and dominant friction are inferred. KV readiness is direct only when cache/prefill telemetry is present.</div>
  <h2>Case Summary</h2>
  {table(summary_cols, summary_rows)}
  <h2>Per-Request Detail</h2>
  {table(detail_cols, rows)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-request replay friction deep-dive artifacts.")
    parser.add_argument("--report-dir", type=Path, required=True, help="Report directory containing global_kv_readiness_by_mode.csv.")
    parser.add_argument("--run-root", type=Path, help="Optional local run root containing per-case trace directories.")
    parser.add_argument("--out-dir", type=Path, help="Output directory. Defaults to --report-dir.")
    parser.add_argument("--report-label", default="", help="Label shown in HTML.")
    parser.add_argument("--update-latest-root", type=Path, help="Optional artifacts root for latest_* copies.")
    parser.add_argument("--top-level-copy-dir", type=Path, help="Optional project root for a stable replay_friction_deep_dive.html copy.")
    args = parser.parse_args()

    out_dir = args.out_dir or args.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_label = args.report_label or args.report_dir.name
    source_csv = args.report_dir / "global_kv_readiness_by_mode.csv"
    source_rows = read_csv_rows(source_csv)
    global_rows = derived_replay_rows(source_rows)
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in global_rows:
        rows_by_case[str(row.get("case_id") or "")].append(row)
    all_rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        all_rows_by_case[str(row.get("case_id") or "")].append(row)
    traces: dict[str, CaseTrace] = {}
    for case_id, group in rows_by_case.items():
        case_dir = local_case_dir(group[0], args.run_root)
        traces[case_id] = CaseTrace(case_dir)
    friction_rows = [
        build_friction_row(
            row,
            rows_by_case[str(row.get("case_id") or "")],
            all_rows_by_case[str(row.get("case_id") or "")],
            traces[str(row.get("case_id") or "")],
        )
        for row in global_rows
    ]
    summary_rows = summarize(friction_rows)
    write_csv(out_dir / "replay_friction_deep_dive.csv", friction_rows, REPLAY_FRICTION_COLUMNS)
    write_csv(out_dir / "replay_friction_summary.csv", summary_rows, SUMMARY_COLUMNS)
    payload_json = json.dumps({"summary": summary_rows, "requests": friction_rows}, indent=2, sort_keys=True)
    html_body = render_html(friction_rows, summary_rows, report_label)
    atomic_write_text(out_dir / "replay_friction_deep_dive.json", payload_json)
    atomic_write_text(out_dir / "replay_friction_deep_dive.html", html_body)
    if args.update_latest_root:
        args.update_latest_root.mkdir(parents=True, exist_ok=True)
        write_csv(args.update_latest_root / "latest_replay_friction_deep_dive.csv", friction_rows, REPLAY_FRICTION_COLUMNS)
        write_csv(args.update_latest_root / "latest_replay_friction_summary.csv", summary_rows, SUMMARY_COLUMNS)
        atomic_write_text(args.update_latest_root / "latest_replay_friction_deep_dive.html", html_body)
        atomic_write_text(args.update_latest_root / "latest_replay_friction_deep_dive.json", payload_json)
    if args.top_level_copy_dir:
        args.top_level_copy_dir.mkdir(parents=True, exist_ok=True)
        full_detail_href = os.path.relpath(out_dir / "replay_friction_deep_dive.html", args.top_level_copy_dir)
        reader_html = render_reader_html(
            friction_rows,
            summary_rows,
            report_label,
            full_detail_href.replace(os.sep, "/"),
        )
        atomic_write_text(args.top_level_copy_dir / "replay_friction_deep_dive.html", reader_html)
    print(f"replay_friction_rows={len(friction_rows)} summary_rows={len(summary_rows)}")


if __name__ == "__main__":
    main()
