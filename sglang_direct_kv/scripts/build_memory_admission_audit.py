#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEVICE_EVICT_EVENTS = {
    "hiradix.evict_device.end",
    "hiradix.evict_device_node.end",
    "hicache.evict_device.end",
}
HOST_EVICT_EVENTS = {
    "hiradix.evict_host.end",
    "hiradix.evict_host_node.end",
    "hicache.evict_host.end",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_index_count(row: dict[str, Any]) -> int:
    for key in ("device_index_count", "host_index_count", "index_count"):
        value = row.get(key)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    for key in ("device_indices", "host_indices", "indices"):
        value = row.get(key)
        if isinstance(value, dict):
            try:
                return int(float(value.get("index_count") or value.get("numel") or 0))
            except (TypeError, ValueError):
                continue
    return 0


def fmt_range(values: list[float]) -> str:
    if not values:
        return ""
    return f"{min(values):.0f}..{max(values):.0f}"


def audit_case(case_dir: Path) -> dict[str, Any]:
    trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
    gateway_rows = read_jsonl(case_dir / "harness_gateway_events.jsonl")
    decisions = [row for row in trace_rows if row.get("event") == "m27.controller_memory_admission.decision"]
    releases = [row for row in trace_rows if row.get("event") == "m27.controller_memory_admission.released"]
    registered = [row for row in trace_rows if row.get("event") == "m27.controller_memory_admission.replay_registered"]
    decision_counts = Counter(str(row.get("decision") or "") for row in decisions)
    delayed_ms = [value for value in (as_float(row.get("actual_delay_ms")) for row in releases) if value is not None]
    predicted_counts = [value for value in (as_float(row.get("predicted_replay_count")) for row in decisions) if value is not None]
    predicted_tokens = [value for value in (as_float(row.get("predicted_replay_tokens")) for row in decisions) if value is not None]
    candidate_tokens = [value for value in (as_float(row.get("candidate_tokens")) for row in decisions) if value is not None]
    gateway_delayed = [
        row
        for row in gateway_rows
        if str(row.get("controller_memory_admission_decision") or "") == "delay"
    ]
    evict_counts = Counter()
    evict_indices = Counter()
    for row in trace_rows:
        event = str(row.get("event") or "")
        if event in DEVICE_EVICT_EVENTS:
            evict_counts["device"] += 1
            evict_indices["device"] += collect_index_count(row)
        elif event in HOST_EVICT_EVENTS:
            evict_counts["host"] += 1
            evict_indices["host"] += collect_index_count(row)
    modes = {str(row.get("mode") or "") for row in trace_rows if row.get("mode")}
    mode = next(iter(modes), "")
    if not mode:
        mode = "controller_memory_admission" if "controller_memory_admission" in case_dir.name else "no_prefetch"
    return {
        "case_id": case_dir.name,
        "mode": mode,
        "registered_replay_predictions": len(registered),
        "admission_decisions": len(decisions),
        "admitted_candidates": decision_counts["admit"],
        "delayed_candidates": decision_counts["delay"],
        "released_candidates": len(releases),
        "gateway_delayed_requests": len(gateway_delayed),
        "delay_ms_range": fmt_range(delayed_ms),
        "total_delay_ms": round(sum(delayed_ms), 3) if delayed_ms else "",
        "predicted_replay_count_range": fmt_range(predicted_counts),
        "predicted_replay_tokens_range": fmt_range(predicted_tokens),
        "candidate_tokens_range": fmt_range(candidate_tokens),
        "native_evict_device_events": evict_counts["device"],
        "native_evict_host_events": evict_counts["host"],
        "native_evict_device_indices": evict_indices["device"],
        "native_evict_host_indices": evict_indices["host"],
    }


def iter_case_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(path for path in run_root.iterdir() if path.is_dir())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else ["case_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        columns = list(rows[0])
        header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
        body = "\n".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>"
            for row in rows
        )
    else:
        header = "<th>case_id</th>"
        body = "<tr><td>No Scenario 6 cases found.</td></tr>"
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Memory Admission Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; }}
    h1 {{ font-size: 24px; }}
    p {{ max-width: 920px; line-height: 1.5; color: #475569; }}
    table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; color: #334155; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <h1>Memory Admission Audit</h1>
  <p>This audit checks Scenario 6 controller decisions: predicted replay demand, candidate admission delays, and native SGLang/HiCache eviction evidence.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        audit_case(case_dir)
        for case_dir in iter_case_dirs(args.run_root)
        if "controller_memory_admission" in case_dir.name or "no_prefetch" in case_dir.name
    ]
    write_csv(args.report_dir / "memory_admission_audit.csv", rows)
    write_html(args.report_dir / "memory_admission_audit.html", rows)
    print(f"Wrote {args.report_dir / 'memory_admission_audit.csv'}")
    print(f"Wrote {args.report_dir / 'memory_admission_audit.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
