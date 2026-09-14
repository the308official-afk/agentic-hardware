#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from agentic_kv.controller.runtime_calibration import runtime_class_key


FIELDS = [
    "case_id",
    "harness",
    "mode",
    "pressure_level",
    "session_id",
    "request_id",
    "phase",
    "stage",
    "request_group",
    "task_index",
    "tool_wait_step",
    "tool_wait_class",
    "agentic_workload_profile",
    "workload_phase_family",
    "workload_request_kind",
    "prompt_tokens",
    "max_tokens",
    "prompt_chars",
    "body_bytes",
    "concurrency",
    "filler_sessions",
    "filler_backlog_mode",
    "filler_backlog_target",
    "filler_backlog_total",
    "client_submit_seq",
    "client_sem_capacity",
    "client_pending_before_acquire",
    "client_inflight_before_acquire",
    "client_pending_at_submit",
    "client_inflight_at_submit",
    "client_queue_wait_ms",
    "sglang_priority",
    "ttft_ms",
    "total_latency_ms",
    "actual_runtime_ms",
    "runtime_source",
    "raw_estimated_runtime_ms",
    "estimated_runtime_ms",
    "estimation_error_ms",
    "actual_overshoot_ms",
    "verdict",
    "runtime_class",
    "oracle_runtime_key",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def is_filler(row: dict[str, Any]) -> bool:
    phase = str(row.get("phase") or "")
    request_id = str(row.get("request_id") or row.get("label") or "")
    request_group = str(row.get("request_group") or "")
    return request_group == "filler" or phase.startswith("pressure_filler") or "_pressure_" in request_id


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_row(case_dir: Path, row: dict[str, Any], case_meta: dict[str, str], source: str) -> dict[str, Any]:
    ttft = as_float(row.get("ttft_ms"))
    total_latency = as_float(row.get("total_latency_ms"))
    actual_runtime = as_float(row.get("actual_runtime_ms"))
    if actual_runtime is None:
        actual_runtime = ttft if ttft is not None else total_latency
    out = {
        "case_id": case_dir.name,
        "harness": row.get("harness", case_meta.get("harness", "")),
        "mode": row.get("mode", case_meta.get("mode", "")),
        "pressure_level": row.get("pressure_level", case_meta.get("pressure_level", "")),
        "session_id": row.get("session_id", ""),
        "request_id": row.get("request_id") or row.get("label", ""),
        "phase": row.get("phase", ""),
        "stage": row.get("stage", ""),
        "request_group": row.get("request_group") or "filler",
        "task_index": row.get("task_index", ""),
        "tool_wait_step": row.get("tool_wait_step", ""),
        "tool_wait_class": row.get("tool_wait_class", ""),
        "agentic_workload_profile": row.get("agentic_workload_profile", case_meta.get("agentic_workload_profile", "")),
        "workload_phase_family": row.get("workload_phase_family", ""),
        "workload_request_kind": row.get("workload_request_kind", ""),
        "prompt_tokens": row.get("prompt_tokens", ""),
        "max_tokens": row.get("max_tokens", ""),
        "prompt_chars": row.get("prompt_chars", ""),
        "body_bytes": row.get("body_bytes", ""),
        "concurrency": row.get("concurrency", "") or case_meta.get("concurrency", ""),
        "filler_sessions": row.get("filler_sessions", "") or case_meta.get("filler_sessions", ""),
        "filler_backlog_mode": row.get("filler_backlog_mode", "") or case_meta.get("filler_backlog_mode", ""),
        "filler_backlog_target": row.get("filler_backlog_target", "") or case_meta.get("filler_backlog_target", ""),
        "filler_backlog_total": row.get("filler_backlog_total", "") or case_meta.get("filler_backlog_total", ""),
        "client_submit_seq": row.get("client_submit_seq", ""),
        "client_sem_capacity": row.get("client_sem_capacity", ""),
        "client_pending_before_acquire": row.get("client_pending_before_acquire", ""),
        "client_inflight_before_acquire": row.get("client_inflight_before_acquire", ""),
        "client_pending_at_submit": row.get("client_pending_at_submit", ""),
        "client_inflight_at_submit": row.get("client_inflight_at_submit", ""),
        "client_queue_wait_ms": row.get("client_queue_wait_ms", ""),
        "sglang_priority": row.get("sglang_priority", ""),
        "ttft_ms": round(ttft, 3) if ttft is not None else "",
        "total_latency_ms": round(total_latency, 3) if total_latency is not None else "",
        "actual_runtime_ms": round(actual_runtime, 3) if actual_runtime is not None else "",
        "runtime_source": source,
        "raw_estimated_runtime_ms": row.get("raw_estimated_runtime_ms", ""),
        "estimated_runtime_ms": row.get("estimated_runtime_ms", ""),
        "estimation_error_ms": row.get("estimation_error_ms", ""),
        "actual_overshoot_ms": row.get("actual_overshoot_ms", ""),
        "verdict": row.get("verdict", ""),
        "runtime_class": row.get("runtime_class", ""),
        "oracle_runtime_key": row.get("oracle_runtime_key", ""),
    }
    if not str(out["runtime_class"]).strip():
        out["runtime_class"] = runtime_class_key(out)
    return out


def infer_case_meta_from_name(case_dir: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    match = re.search(r"_enccalibc(\d+)(?:_|$)", case_dir.name)
    if match:
        meta["concurrency"] = match.group(1)
    match = re.search(r"_f(\d+)(?:_|$)", case_dir.name)
    if match:
        meta["filler_sessions"] = match.group(1)
    return meta


def case_meta_from_trace(case_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    meta = infer_case_meta_from_name(case_dir)
    for row in rows:
        if str(row.get("event") or "") != "m27.workload_start":
            continue
        for key in (
            "harness",
            "mode",
            "pressure_level",
            "concurrency",
            "filler_sessions",
            "filler_backlog_mode",
            "filler_backlog_target",
            "filler_backlog_total",
            "agentic_workload_profile",
        ):
            if row.get(key) not in (None, ""):
                meta[key] = str(row.get(key))
        break
    return meta


def iter_case_dirs(root: Path) -> list[Path]:
    if (root / "m27_trace.jsonl").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("m27_trace.jsonl"))


def calibration_rows(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    case_dirs = iter_case_dirs(root)
    seen_end_request_ids: set[tuple[str, str]] = set()
    for case_dir in case_dirs:
        rows = read_jsonl(case_dir / "m27_trace.jsonl")
        case_meta = case_meta_from_trace(case_dir, rows)
        for row in rows:
            if str(row.get("event") or "") != "m27.request.end" or not is_filler(row):
                continue
            if as_float(row.get("ttft_ms")) is None and as_float(row.get("total_latency_ms")) is None:
                continue
            request_id = str(row.get("request_id") or row.get("label") or "")
            seen_end_request_ids.add((case_dir.name, request_id))
            out.append(normalize_row(case_dir, row, case_meta, "gateway_request_end"))
        for row in rows:
            if str(row.get("event") or "") != "m27.controller_completion_linkage" or not is_filler(row):
                continue
            request_id = str(row.get("request_id") or row.get("label") or "")
            if (case_dir.name, request_id) in seen_end_request_ids:
                # The gateway row is the cleaner timing source for model fitting;
                # completion linkage is still useful when a request never reached
                # a normal gateway end row.
                continue
            out.append(normalize_row(case_dir, row, case_meta, "controller_completion_linkage"))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract filler timing calibration rows from completed traces.")
    parser.add_argument("--root", type=Path, required=True, help="Run root or single case directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output filler_timing_calibration.csv.")
    args = parser.parse_args()
    rows = calibration_rows(args.root)
    write_csv(args.out, rows)
    print(f"wrote {args.out}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
