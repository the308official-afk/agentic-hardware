#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_multi_harness_deadline_summary import (
    CONTROLLER_IDLE_GAP_AUDIT_COLUMNS,
    collect_controller_idle_gap_audit,
    write_csv,
)


def read_csv_table(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controller idle-gap audit artifacts for one completed run.")
    parser.add_argument("--root", type=Path, required=True, help="Completed run root containing per-case directories.")
    parser.add_argument("--rows-csv", type=Path, required=True, help="global_kv_readiness_by_mode.csv for the run.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Report/artifact output directory.")
    parser.add_argument("--latest-root", type=Path, help="Optional latest artifacts root.")
    args = parser.parse_args()

    rows = read_csv_table(args.rows_csv)
    audit_rows = collect_controller_idle_gap_audit(args.root, rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "controller_idle_gap_audit.csv", audit_rows, CONTROLLER_IDLE_GAP_AUDIT_COLUMNS)
    write_json(args.out_dir / "controller_idle_gap_audit.json", audit_rows)
    if args.latest_root:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        write_csv(
            args.latest_root / "latest_controller_idle_gap_audit.csv",
            audit_rows,
            CONTROLLER_IDLE_GAP_AUDIT_COLUMNS,
        )
        write_json(args.latest_root / "latest_controller_idle_gap_audit.json", audit_rows)
    print(f"controller_idle_gap_audit_rows={len(audit_rows)}")


if __name__ == "__main__":
    main()
