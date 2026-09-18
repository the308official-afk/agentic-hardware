#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVICT_DEVICE_EVENTS = {"hicache.evict_device.end", "hicache_evict_device"}
EVICT_HOST_EVENTS = {"hicache.evict_host.end", "hicache_evict_host"}


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return ""


def event_session_id(event: dict[str, Any]) -> str:
    value = first_present(
        event,
        (
            "session_id",
            "agent_session_id",
            "ledger_session_id",
            "request_session_id",
        ),
    )
    if value:
        return str(value)
    ctx = event.get("kv_context")
    if isinstance(ctx, dict):
        value = first_present(ctx, ("agent_session_id", "session_id", "ledger_session_id"))
        if value:
            return str(value)
        sessions = ctx.get("agent_sessions")
        if isinstance(sessions, list) and sessions:
            first = sessions[0]
            if isinstance(first, dict):
                value = first_present(first, ("agent_session_id", "session_id", "ledger_session_id"))
                if value:
                    return str(value)
    return ""


def collect_index_count(value: Any) -> int:
    if isinstance(value, dict):
        total = 0
        for key, child in value.items():
            if key == "index_count":
                try:
                    total += int(float(child))
                except (TypeError, ValueError):
                    pass
            elif key.endswith("_indices") or key in {"indices", "device_indices", "host_indices"}:
                if isinstance(child, dict):
                    try:
                        total += int(float(child.get("index_count")))
                    except (TypeError, ValueError):
                        total += collect_index_count(child)
                elif isinstance(child, list):
                    total += len(child)
            elif isinstance(child, (dict, list)):
                total += collect_index_count(child)
        return total
    if isinstance(value, list):
        return sum(collect_index_count(child) for child in value)
    return 0


def iter_case_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(path for path in run_root.iterdir() if path.is_dir())


def audit_case(case_dir: Path) -> dict[str, Any]:
    gateway_events = case_dir / "harness_gateway_events.jsonl"
    trace = case_dir / "m27_trace.jsonl"
    high_scores: list[float] = []
    normal_scores: list[float] = []
    low_scores: list[float] = []
    high_priorities: list[float] = []
    normal_priorities: list[float] = []
    low_priorities: list[float] = []
    request_counts = Counter()
    sessions_by_class: dict[str, set[str]] = defaultdict(set)
    modes = set()
    priority_mismatches = 0

    for row in read_jsonl(gateway_events) or []:
        mode = str(row.get("mode") or "")
        if mode:
            modes.add(mode)
        value_class = str(row.get("controller_eviction_value_class") or "")
        phase = str(row.get("phase") or "")
        priority = as_float(row.get("sglang_priority"))
        score = as_float(row.get("controller_eviction_value_score"))
        if value_class:
            request_counts[f"{value_class}_requests"] += 1
            session = str(row.get("session_id") or "")
            if session:
                sessions_by_class[value_class].add(session)
            expected_priority = {
                "protected_high_value": 100.0,
                "normal_value": 0.0,
                "evictable_low_value": -100.0,
            }.get(value_class)
            if expected_priority is not None and priority is not None and priority != expected_priority:
                priority_mismatches += 1
        if value_class == "protected_high_value":
            if priority is not None:
                high_priorities.append(priority)
            if score is not None:
                high_scores.append(score)
            if phase == "replay":
                request_counts["protected_high_value_replays"] += 1
        elif value_class == "evictable_low_value":
            if priority is not None:
                low_priorities.append(priority)
            if score is not None:
                low_scores.append(score)
            if phase.startswith("pressure_filler"):
                request_counts["evictable_low_value_pressure_requests"] += 1
        elif value_class == "normal_value":
            if priority is not None:
                normal_priorities.append(priority)
            if score is not None:
                normal_scores.append(score)
            if phase == "replay" or phase.startswith("pressure_filler"):
                request_counts["normal_value_replays"] += 1

    evict_counts = Counter()
    evict_tokens = Counter()
    evict_sessions: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(trace) or []:
        event = str(row.get("event") or "")
        if event in EVICT_DEVICE_EVENTS:
            evict_counts["native_evict_device_events"] += 1
            evict_tokens["native_evict_device_indices"] += collect_index_count(row)
            session = event_session_id(row)
            if session:
                evict_sessions["device"].add(session)
        elif event in EVICT_HOST_EVENTS:
            evict_counts["native_evict_host_events"] += 1
            evict_tokens["native_evict_host_indices"] += collect_index_count(row)
            session = event_session_id(row)
            if session:
                evict_sessions["host"].add(session)

    def fmt_range(values: list[float]) -> str:
        if not values:
            return ""
        return f"{min(values):.0f}..{max(values):.0f}"

    mode = next(iter(modes), "")
    if not mode:
        parts = case_dir.name.split("_")
        mode = "controller_value_aware_eviction" if "controller" in parts and "eviction" in parts else ""

    return {
        "case_id": case_dir.name,
        "mode": mode,
        "protected_high_value_requests": request_counts["protected_high_value_requests"],
        "protected_high_value_replays": request_counts["protected_high_value_replays"],
        "evictable_low_value_requests": request_counts["evictable_low_value_requests"],
        "evictable_low_value_pressure_requests": request_counts["evictable_low_value_pressure_requests"],
        "normal_value_requests": request_counts["normal_value_requests"],
        "normal_value_replays": request_counts["normal_value_replays"],
        "protected_high_value_sessions": len(sessions_by_class["protected_high_value"]),
        "normal_value_sessions": len(sessions_by_class["normal_value"]),
        "evictable_low_value_sessions": len(sessions_by_class["evictable_low_value"]),
        "priority_mismatch_count": priority_mismatches,
        "priority_alignment_ok": "yes" if priority_mismatches == 0 else "no",
        "high_priority_range": fmt_range(high_priorities),
        "normal_priority_range": fmt_range(normal_priorities),
        "low_priority_range": fmt_range(low_priorities),
        "high_value_score_range": fmt_range(high_scores),
        "normal_value_score_range": fmt_range(normal_scores),
        "low_value_score_range": fmt_range(low_scores),
        "native_evict_device_events": evict_counts["native_evict_device_events"],
        "native_evict_host_events": evict_counts["native_evict_host_events"],
        "native_evict_device_indices": evict_tokens["native_evict_device_indices"],
        "native_evict_host_indices": evict_tokens["native_evict_host_indices"],
        "evict_device_sessions_with_context": len(evict_sessions["device"]),
        "evict_host_sessions_with_context": len(evict_sessions["host"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else ["case_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        header = "".join(f"<th>{html.escape(col)}</th>" for col in rows[0])
        body = "\n".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in rows[0]) + "</tr>"
            for row in rows
        )
    else:
        header = "<th>case_id</th>"
        body = "<tr><td>No Scenario 3 cases found.</td></tr>"
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Value-Aware Eviction Audit</title>
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
  <h1>Value-Aware Eviction Audit</h1>
  <p>This audit checks that Scenario 3 tagged high-value and low-value KV differently, then looks for native SGLang/HiCache eviction events. The controller should supply value signals and priorities; SGLang should still perform the actual eviction.</p>
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
        if "controller_value_aware_eviction" in case_dir.name or "no_prefetch" in case_dir.name
    ]
    write_csv(args.report_dir / "value_aware_eviction_audit.csv", rows)
    write_html(args.report_dir / "value_aware_eviction_audit.html", rows)
    print(f"Wrote {args.report_dir / 'value_aware_eviction_audit.csv'}")
    print(f"Wrote {args.report_dir / 'value_aware_eviction_audit.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
