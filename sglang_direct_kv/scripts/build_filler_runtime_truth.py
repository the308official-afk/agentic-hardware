#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "oracle_runtime_key",
    "case_id",
    "harness",
    "pressure_level",
    "mode",
    "session_id",
    "request_id",
    "phase",
    "tool_wait_step",
    "agentic_workload_profile",
    "workload_phase_family",
    "workload_request_kind",
    "tool_wait_class",
    "prompt_hash",
    "prompt_chars",
    "ttft_ms",
    "total_latency_ms",
    "actual_runtime_ms",
    "runtime_source",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def request_is_filler(row: dict[str, Any]) -> bool:
    phase = str(row.get("phase") or "")
    label = str(row.get("label") or row.get("request_id") or "")
    return phase.startswith("pressure_filler") or "_pressure_" in label


def truth_rows(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    case_dirs = [root] if (root / "m27_trace.jsonl").exists() else sorted(path for path in root.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        for row in read_jsonl(case_dir / "m27_trace.jsonl"):
            if str(row.get("event") or "") != "m27.request.end":
                continue
            if not request_is_filler(row):
                continue
            key = str(row.get("oracle_runtime_key") or "").strip()
            if not key:
                # Old traces cannot serve as exact oracle input because the
                # stage-two lookup must match the stage-one request identity.
                continue
            try:
                ttft_ms = float(row.get("ttft_ms") or "")
            except ValueError:
                continue
            try:
                total_latency_ms = float(row.get("total_latency_ms") or "")
            except ValueError:
                total_latency_ms = ttft_ms
            out.append(
                {
                    "oracle_runtime_key": key,
                    "case_id": case_dir.name,
                    "harness": row.get("harness", ""),
                    "pressure_level": row.get("pressure_level", ""),
                    "mode": row.get("mode", ""),
                    "session_id": row.get("session_id", ""),
                    "request_id": row.get("request_id") or row.get("label", ""),
                    "phase": row.get("phase", ""),
                    "tool_wait_step": row.get("tool_wait_step", ""),
                    "agentic_workload_profile": row.get("agentic_workload_profile", ""),
                    "workload_phase_family": row.get("workload_phase_family", ""),
                    "workload_request_kind": row.get("workload_request_kind", ""),
                    "tool_wait_class": row.get("tool_wait_class", ""),
                    "prompt_hash": row.get("prompt_hash", ""),
                    "prompt_chars": row.get("prompt_chars", ""),
                    "ttft_ms": round(ttft_ms, 3),
                    "total_latency_ms": round(total_latency_ms, 3),
                    "actual_runtime_ms": round(ttft_ms, 3),
                    "runtime_source": "observed_filler_ttft",
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact filler runtime truth table from a completed run.")
    parser.add_argument("--root", type=Path, required=True, help="Run root or single case directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output filler_runtime_truth.csv path.")
    args = parser.parse_args()
    rows = truth_rows(args.root)
    write_csv(args.out, rows)
    print(f"wrote {args.out}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
