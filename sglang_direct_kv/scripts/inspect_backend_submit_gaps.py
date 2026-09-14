#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "harness",
    "harness_label",
    "pressure_level",
    "pressure_level_label",
    "mode",
    "mode_label",
    "case_id",
    "gap_index",
    "gap_duration_ms",
    "gap_start_ms",
    "gap_end_ms",
    "gap_location",
    "request_starts_inside_gap",
    "request_start_labels",
    "request_start_phases",
    "gateway_receives_inside_gap",
    "gateway_receive_labels",
    "gateway_forwards_inside_gap",
    "next_batch_after_gap_ms",
    "likely_cause",
]


def read_csv_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
    tmp_path.replace(path)


def timestamp_ns(row: dict[str, Any]) -> int | None:
    value = row.get("ts_ns")
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact(values: list[str], limit: int = 8) -> str:
    clean = [value for value in values if value]
    if len(clean) <= limit:
        return ";".join(clean)
    return ";".join(clean[:limit] + [f"...+{len(clean) - limit} more"])


def rows_between(rows: list[dict[str, Any]], start_ns: int, end_ns: int) -> list[dict[str, Any]]:
    return [row for row in rows if (ts := timestamp_ns(row)) and start_ns <= ts <= end_ns]


def inspect_gap(gap: dict[str, str]) -> dict[str, Any]:
    case_dir = Path(gap.get("case_dir") or "")
    trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
    gateway_rows = read_jsonl(case_dir / "harness_gateway_events.jsonl")
    workload_start = next((row for row in trace_rows if row.get("event") == "m27.workload_start"), {})
    workload_start_ns = timestamp_ns(workload_start)
    if not workload_start_ns:
        starts = [timestamp_ns(row) for row in trace_rows + gateway_rows if timestamp_ns(row)]
        workload_start_ns = min(starts or [0])
    gap_start_ms = float_value(gap.get("gap_start_ms"))
    gap_end_ms = float_value(gap.get("gap_end_ms"))
    if not workload_start_ns or gap_start_ms is None or gap_end_ms is None:
        return {**gap, "likely_cause": "missing_gap_timestamp"}

    gap_start_ns = workload_start_ns + int(gap_start_ms * 1_000_000)
    gap_end_ns = workload_start_ns + int(gap_end_ms * 1_000_000)
    request_starts = [
        row
        for row in rows_between(trace_rows, gap_start_ns, gap_end_ns)
        if row.get("event") == "m27.request.start"
    ]
    gateway_receives = [
        row
        for row in rows_between(gateway_rows, gap_start_ns, gap_end_ns)
        if row.get("event") == "gateway.request_received"
    ]
    gateway_forwards = [
        row
        for row in rows_between(gateway_rows, gap_start_ns, gap_end_ns)
        if row.get("event") == "gateway.forwarded_request"
    ]

    next_batch_after_gap_ms = ""
    next_batch_delay_ms = None
    for row in trace_rows:
        if row.get("event") != "scheduler.run_batch.end":
            continue
        end_ns = timestamp_ns(row)
        duration_ms = float_value(row.get("duration_ms"))
        if not end_ns or duration_ms is None:
            continue
        start_ns = end_ns - int(duration_ms * 1_000_000)
        if start_ns >= gap_end_ns:
            next_batch_delay_ms = (start_ns - gap_end_ns) / 1_000_000.0
            next_batch_after_gap_ms = round(next_batch_delay_ms, 3)
            break

    if not request_starts and not gateway_receives:
        likely_cause = "no_submission_visible_in_gap"
    elif gateway_receives and not gateway_forwards:
        likely_cause = "gateway_received_but_forward_not_visible"
    elif gateway_forwards and (next_batch_delay_ms is None or next_batch_delay_ms > 25):
        likely_cause = "forwarded_before_sglang_batch_visible"
    elif request_starts and not gateway_receives:
        likely_cause = "driver_started_before_gateway_receive_visible"
    else:
        likely_cause = "short_submit_to_batch_delay_or_trace_resolution"

    return {
        "harness": gap.get("harness", ""),
        "harness_label": gap.get("harness_label", ""),
        "pressure_level": gap.get("pressure_level", ""),
        "pressure_level_label": gap.get("pressure_level_label", ""),
        "mode": gap.get("mode", ""),
        "mode_label": gap.get("mode_label", ""),
        "case_id": gap.get("case_id", ""),
        "gap_index": gap.get("gap_index", ""),
        "gap_duration_ms": gap.get("gap_duration_ms", ""),
        "gap_start_ms": gap.get("gap_start_ms", ""),
        "gap_end_ms": gap.get("gap_end_ms", ""),
        "gap_location": gap.get("gap_location", ""),
        "request_starts_inside_gap": len(request_starts),
        "request_start_labels": compact([str(row.get("label") or row.get("request_id") or "") for row in request_starts]),
        "request_start_phases": compact([str(row.get("phase") or "") for row in request_starts]),
        "gateway_receives_inside_gap": len(gateway_receives),
        "gateway_receive_labels": compact([str(row.get("label") or row.get("request_id") or "") for row in gateway_receives]),
        "gateway_forwards_inside_gap": len(gateway_forwards),
        "next_batch_after_gap_ms": next_batch_after_gap_ms,
        "likely_cause": likely_cause,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect backend-submit idle gaps from a completed idle-gap audit.")
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    candidates = [
        row
        for row in read_csv_table(args.audit_csv)
        if row.get("idle_reason") in {"backend_submit_gap", "backend_submit_or_batching_gap"}
    ]
    candidates.sort(key=lambda row: float_value(row.get("gap_duration_ms")) or 0.0, reverse=True)
    rows = [inspect_gap(row) for row in candidates[: max(0, args.top)]]
    write_csv(args.out, rows)
    if args.latest_root:
        write_csv(args.latest_root / "latest_backend_submit_gap_inspection.csv", rows)
    print(f"backend_submit_gap_inspection_rows={len(rows)}")


if __name__ == "__main__":
    main()
