#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


LOAD_EVENTS = {
    "hicache.load.end",
    "hiradix.init_load_back.end",
    "hiradix.load_back.end",
}
EVICT_EVENTS = {
    "hicache.evict_device.end",
    "hiradix.evict.end",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def event_ts(event: dict[str, Any]) -> int:
    value = event.get("ts_ns", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def events_between(events: list[dict[str, Any]], start_ts: int | None, end_ts: int | None) -> list[dict[str, Any]]:
    if start_ts is None or end_ts is None:
        return []
    lo = min(start_ts, end_ts)
    hi = max(start_ts, end_ts)
    return [event for event in events if lo <= event_ts(event) <= hi]


def first_event(events: list[dict[str, Any]], name: str, session_id: str) -> dict[str, Any] | None:
    matches = [
        event
        for event in events
        if event.get("event") == name and event.get("session_id") == session_id
    ]
    return min(matches, key=event_ts) if matches else None


def request_windows(events: list[dict[str, Any]], session_id: str, phase: str) -> list[tuple[int, int]]:
    starts = [
        event
        for event in events
        if event.get("event") == "agent.request.start"
        and event.get("session_id") == session_id
        and event.get("phase") == phase
    ]
    ends = [
        event
        for event in events
        if event.get("event") == "agent.request.end"
        and event.get("session_id") == session_id
        and event.get("phase") == phase
    ]
    windows: list[tuple[int, int]] = []
    used_end_ids: set[int] = set()
    for start in sorted(starts, key=event_ts):
        start_ts = event_ts(start)
        candidates = [
            (idx, event)
            for idx, event in enumerate(ends)
            if idx not in used_end_ids and event_ts(event) >= start_ts
        ]
        if not candidates:
            continue
        idx, end = min(candidates, key=lambda item: event_ts(item[1]))
        used_end_ids.add(idx)
        windows.append((start_ts, event_ts(end)))
    return windows


def count_named(events: list[dict[str, Any]], names: set[str]) -> int:
    return sum(1 for event in events if event.get("event") in names)


def avg_metric(metrics: list[dict[str, Any]], session_id: str, phase: str, key: str) -> float:
    values = [
        float(row[key])
        for row in metrics
        if row.get("session_id") == session_id and row.get("phase") == phase and key in row
    ]
    return round(mean(values), 3) if values else 0.0


def classify(
    *,
    has_hint: bool,
    hint_completed_before_replay: bool,
    prefetch_load_count: int,
    eviction_pressure_after_prefetch: int,
    resume_load_count: int,
    replay_ttft_ms: float,
) -> str:
    if not has_hint:
        return "no_hint"
    if not hint_completed_before_replay:
        return "late_prefetch"
    if prefetch_load_count == 0 and resume_load_count == 0:
        return "no_prefetch_needed"
    if prefetch_load_count == 0 and eviction_pressure_after_prefetch > 0 and resume_load_count > 0:
        return "too_early_no_load_then_evicted"
    if prefetch_load_count > 0 and resume_load_count == 0:
        return "useful_prefetch"
    if prefetch_load_count > 0 and eviction_pressure_after_prefetch > 0 and resume_load_count > 0:
        return "too_early_or_unprotected"
    if resume_load_count > 0:
        return "resume_still_loaded_kv"
    if replay_ttft_ms <= 0:
        return "wasted_prefetch"
    return "unknown"


def build_rows(trace: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    session_ids = sorted(
        {
            str(row["session_id"])
            for row in metrics
            if row.get("phase") in {"initial_turn", "replay", "hint_prefetch"} and row.get("session_id")
        }
    )
    rows: list[dict[str, Any]] = []
    for session_id in session_ids:
        session_metrics = [row for row in metrics if row.get("session_id") == session_id]
        first_metric = session_metrics[0] if session_metrics else {}
        hint_start = first_event(trace, "agent.hint_prefetch_start", session_id)
        hint_end = first_event(trace, "agent.hint_prefetch_end", session_id)
        resume_start = first_event(trace, "agent.resume_start", session_id)
        replay_due = first_event(trace, "agent.replay_due", session_id)
        has_hint = hint_start is not None
        hint_start_ts = event_ts(hint_start) if hint_start else None
        hint_end_ts = event_ts(hint_end) if hint_end else None
        resume_start_ts = event_ts(resume_start) if resume_start else None

        hint_events: list[dict[str, Any]] = []
        for start_ts, end_ts in request_windows(trace, session_id, "hint_prefetch"):
            hint_events.extend(events_between(trace, start_ts, end_ts))
        resume_events: list[dict[str, Any]] = []
        for start_ts, end_ts in request_windows(trace, session_id, "replay"):
            resume_events.extend(events_between(trace, start_ts, end_ts))

        prefetch_load_count = count_named(hint_events, LOAD_EVENTS)
        prefetch_hicache_load_count = count_named(hint_events, {"hicache.load.end"})
        resume_load_count = count_named(resume_events, LOAD_EVENTS)
        resume_hicache_load_count = count_named(resume_events, {"hicache.load.end"})
        after_hint_before_resume = events_between(trace, hint_end_ts, resume_start_ts)
        eviction_pressure_after_prefetch = count_named(after_hint_before_resume, EVICT_EVENTS)
        hint_completed_before_replay = bool(
            has_hint
            and hint_end_ts is not None
            and resume_start_ts is not None
            and hint_end_ts <= resume_start_ts
        )

        replay_ttft_ms = avg_metric(metrics, session_id, "replay", "ttft_ms")
        hint_ttft_ms = avg_metric(metrics, session_id, "hint_prefetch", "ttft_ms")
        initial_ttft_ms = avg_metric(metrics, session_id, "initial_turn", "ttft_ms")
        outcome = classify(
            has_hint=has_hint,
            hint_completed_before_replay=hint_completed_before_replay,
            prefetch_load_count=prefetch_load_count,
            eviction_pressure_after_prefetch=eviction_pressure_after_prefetch,
            resume_load_count=resume_load_count,
            replay_ttft_ms=replay_ttft_ms,
        )
        rows.append(
            {
                "session_id": session_id,
                "mode": first_metric.get("mode", "unknown"),
                "priority": first_metric.get("priority", "unknown"),
                "tool_wait_ms": int(first_metric.get("tool_wait_ms", 0)),
                "prompt_tokens": int(first_metric.get("prompt_tokens", 0)),
                "initial_ttft_ms": initial_ttft_ms,
                "hint_ttft_ms": hint_ttft_ms,
                "replay_ttft_ms": replay_ttft_ms,
                "has_hint": int(has_hint),
                "hint_completed_before_replay": int(hint_completed_before_replay),
                "prefetch_load_count": prefetch_load_count,
                "prefetch_hicache_load_count": prefetch_hicache_load_count,
                "resume_load_count": resume_load_count,
                "resume_hicache_load_count": resume_hicache_load_count,
                "eviction_pressure_after_prefetch": eviction_pressure_after_prefetch,
                "direct_load_attempts": count_named(
                    [event for event in trace if event.get("session_id") == session_id],
                    {"agent.direct_kv_load_attempt"},
                ),
                "hint_start_ms_from_replay": (
                    round((hint_start_ts - event_ts(replay_due)) / 1_000_000, 3)
                    if hint_start_ts is not None and replay_due is not None
                    else ""
                ),
                "hint_end_ms_before_replay": (
                    round((resume_start_ts - hint_end_ts) / 1_000_000, 3)
                    if resume_start_ts is not None and hint_end_ts is not None
                    else ""
                ),
                "outcome": outcome,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["outcome"] for row in rows)
    lines = ["# Hint Outcome Report", ""]
    lines.append("## Summary")
    lines.append("")
    for name, count in counts.most_common():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Sessions")
    lines.append("")
    headers = [
        "session_id",
        "mode",
        "tool_wait_ms",
        "prompt_tokens",
        "replay_ttft_ms",
        "prefetch_load_count",
        "resume_load_count",
        "eviction_pressure_after_prefetch",
        "outcome",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["outcome"] for row in rows)
    max_count = max(counts.values()) if counts else 1
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        "  <title>Hint Outcome Report</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 32px; color: #111827; background: #f9fafb; }",
        "    h1 { margin: 0 0 18px; }",
        "    .bar { display: flex; align-items: center; gap: 10px; margin: 8px 0; }",
        "    .label { width: 230px; font-weight: 600; }",
        "    .fill { height: 18px; background: #2563eb; border-radius: 4px; }",
        "    table { margin-top: 28px; width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }",
        "    th, td { border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: right; white-space: nowrap; }",
        "    th { background: #f3f4f6; }",
        "    th:first-child, td:first-child, th:last-child, td:last-child { text-align: left; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Hint Outcome Report</h1>",
        "  <h2>Outcome Counts</h2>",
    ]
    for name, count in counts.most_common():
        width = max(3.0, count * 100.0 / max_count)
        lines.append('  <div class="bar">')
        lines.append(f'    <div class="label">{html.escape(name)}</div>')
        lines.append(f'    <div class="fill" style="width:{width:.1f}%"></div>')
        lines.append(f"    <div>{count}</div>")
        lines.append("  </div>")
    headers = [
        "session_id",
        "mode",
        "tool_wait_ms",
        "prompt_tokens",
        "initial_ttft_ms",
        "hint_ttft_ms",
        "replay_ttft_ms",
        "prefetch_load_count",
        "resume_load_count",
        "eviction_pressure_after_prefetch",
        "outcome",
    ]
    lines.extend(["  <table>", "    <thead>", "      <tr>"])
    for header in headers:
        lines.append(f"        <th>{html.escape(header)}</th>")
    lines.extend(["      </tr>", "    </thead>", "    <tbody>"])
    for row in rows:
        lines.append("      <tr>")
        for header in headers:
            lines.append(f"        <td>{html.escape(str(row.get(header, '')))}</td>")
        lines.append("      </tr>")
    lines.extend(["    </tbody>", "  </table>", "</body>", "</html>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify hint/prefetch outcomes for agentic traffic traces.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    trace = read_jsonl(Path(args.trace))
    metrics = read_jsonl(Path(args.metrics))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(trace, metrics)
    rows.sort(key=lambda row: (row["mode"], row["tool_wait_ms"], row["session_id"]))

    json_path = out_dir / "hint_outcomes.json"
    csv_path = out_dir / "hint_outcomes.csv"
    md_path = out_dir / "hint_outcomes.md"
    html_path = out_dir / "hint_outcomes.html"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_md(md_path, rows)
    write_html(html_path, rows)

    counts = Counter(row["outcome"] for row in rows)
    print(f"Wrote {len(rows)} session outcomes to {out_dir}")
    for name, count in counts.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
