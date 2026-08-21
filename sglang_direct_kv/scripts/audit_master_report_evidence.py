#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentic_kv.evidence_audit import audit_markdown, audit_report_data


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def resolve_report_json(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "controlled_replay_report.json",
        path / "report" / "controlled_replay_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find controlled_replay_report.json under {path}")


def supplement_from_csvs(data: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    """Backfill audit inputs when the JSON was created before newer rows existed."""

    mapping = {
        "exact_kv_movement_attribution": "exact_kv_movement_attribution.csv",
        "kv_block_ledger": "kv_block_ledger.csv",
        "replay_delay_stage_trace": "replay_delay_stage_trace.csv",
        "replay_queue_timing": "replay_queue_timing.csv",
        "client_dispatch_kv_movement_summary": "client_dispatch_kv_movement_summary.csv",
        "client_dispatch_kv_movement_events": "client_dispatch_kv_movement_events.csv",
    }
    for key, filename in mapping.items():
        if data.get(key):
            continue
        rows = read_csv(report_dir / filename)
        if rows:
            data[key] = rows
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit evidence strength for a generated master report.")
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to controlled_replay_report.json or a report directory containing it.",
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory. Defaults to the report JSON directory.")
    args = parser.parse_args()

    json_path = resolve_report_json(args.report)
    out_dir = args.out_dir or json_path.parent
    data = supplement_from_csvs(load_json(json_path), json_path.parent)
    audit = audit_report_data(data)

    write_csv(out_dir / "instrumentation_evidence_audit_summary.csv", audit["summary"])
    write_csv(out_dir / "instrumentation_evidence_audit_matrix.csv", audit["matrix"])
    write_csv(out_dir / "instrumentation_chart_inventory.csv", audit["chart_inventory"])
    write_csv(out_dir / "instrumentation_artifact_inventory.csv", audit["artifact_inventory"])
    (out_dir / "instrumentation_evidence_audit.md").write_text(audit_markdown(audit), encoding="utf-8")

    print(f"Wrote audit outputs to {out_dir}")
    for key, rows in audit.items():
        print(f"{key}: {len(rows)} rows")


if __name__ == "__main__":
    main()
