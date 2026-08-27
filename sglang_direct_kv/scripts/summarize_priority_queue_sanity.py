#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def aliases_for(row: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for key in ("request_id", "agent_request_id", "dynamo_hint_request_id", "rid"):
        value = row.get(key)
        if value not in (None, "", [], {}):
            aliases.add(str(value))
            aliases.add(f"{key}:{value}")
            aliases.add(f"id:{value}")
    for key in ("aliases", "matched_aliases", "priority_aliases"):
        value = row.get(key)
        if isinstance(value, list):
            aliases.update(str(item) for item in value)
    return aliases


def priority_int(value: Any) -> int | None:
    try:
        if value not in (None, ""):
            return int(value)
    except (TypeError, ValueError):
        return None
    return None


def flatten_priority_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    receive_rows: list[dict[str, Any]] = []
    admission_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        base = {
            "trace_index": idx,
            "event": event.get("event", ""),
            "method": event.get("method", ""),
            "pid": event.get("pid", ""),
            "ts_ns": event.get("ts_ns", ""),
        }
        for row in event.get("priority_receive_order") or []:
            if isinstance(row, dict):
                receive_rows.append({**base, **row})
        for key in ("priority_admission_sequence", "priority_admission_order"):
            for row in event.get(key) or []:
                if isinstance(row, dict):
                    admission_rows.append({**base, "source": key, **row})
        audit = event.get("priority_queue_audit")
        if isinstance(audit, dict):
            queue_rows.append({**base, "source": "priority_queue_audit", **audit})
        for snap in event.get("priority_queue_snapshots") or []:
            if isinstance(snap, dict):
                shallow = dict(snap)
                shallow["priority_histogram"] = json.dumps(shallow.get("priority_histogram", {}), sort_keys=True)
                shallow["queue_head_sample"] = json.dumps(shallow.get("queue_head_sample", []), sort_keys=True)
                queue_rows.append({**base, "source": "priority_queue_snapshots", **shallow})
    return receive_rows, admission_rows, queue_rows


def event_matches_request(row: dict[str, Any], request_id: str) -> bool:
    if request_id in aliases_for(row):
        return True
    raw = json.dumps(row, sort_keys=True)
    return request_id in raw


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    metrics = read_jsonl(Path(args.metrics))
    trace = read_jsonl(Path(args.trace))
    receive_rows, admission_rows, queue_rows = flatten_priority_events(trace)

    high_metrics = [row for row in metrics if row.get("request_id") == args.high_request_id]
    high_submit_ns = min((int(row["client_submit_ns"]) for row in high_metrics if row.get("client_submit_ns")), default=0)
    low_metrics = [row for row in metrics if int(row.get("priority") or 0) < int(args.high_priority)]
    low_before = [row for row in low_metrics if high_submit_ns and int(row.get("client_submit_ns") or 0) < high_submit_ns]

    high_receive = [row for row in receive_rows if event_matches_request(row, args.high_request_id)]
    high_admissions = [row for row in admission_rows if event_matches_request(row, args.high_request_id)]
    high_queues = [row for row in queue_rows if event_matches_request(row, args.high_request_id)]

    admission_by_request: dict[str, dict[str, Any]] = {}
    for row in admission_rows:
        rid = str(row.get("request_id") or "")
        if not rid:
            continue
        seq = priority_int(row.get("admission_seq"))
        old = admission_by_request.get(rid)
        old_seq = priority_int(old.get("admission_seq")) if old else None
        if old is None or (seq is not None and (old_seq is None or seq < old_seq)):
            admission_by_request[rid] = row

    high_seq = min(
        (seq for seq in (priority_int(row.get("admission_seq")) for row in high_admissions) if seq is not None),
        default=None,
    )
    older_low_admitted_after_high: list[str] = []
    older_low_admitted_before_high: list[str] = []
    if high_seq is not None:
        for row in low_before:
            rid = str(row.get("request_id") or "")
            adm = admission_by_request.get(rid)
            seq = priority_int(adm.get("admission_seq")) if adm else None
            if seq is None:
                continue
            if seq > high_seq:
                older_low_admitted_after_high.append(rid)
            elif seq < high_seq:
                older_low_admitted_before_high.append(rid)

    max_lower_ahead = 0
    queue_positions: list[int] = []
    for row in high_queues:
        lower = priority_int(row.get("lower_priority_ahead"))
        pos = priority_int(row.get("target_position") or row.get("queue_position"))
        if lower is not None:
            max_lower_ahead = max(max_lower_ahead, lower)
        if pos is not None:
            queue_positions.append(pos)

    high_priority_seen = any(priority_int(row.get("priority")) == int(args.high_priority) for row in high_receive + high_admissions) or any(
        priority_int(row.get("request_priority")) == int(args.high_priority) for row in high_queues
    )
    jump_observed = high_seq is not None and bool(older_low_admitted_after_high)
    if jump_observed:
        verdict = "priority_jump_observed"
        meaning = (
            f"The high-priority request was admitted before {len(older_low_admitted_after_high)} older low-priority "
            "requests. This is direct evidence of queue reordering."
        )
    elif high_priority_seen and high_seq is not None:
        verdict = "priority_seen_but_no_jump_observed"
        meaning = (
            "SGLang saw and admitted the high-priority request, but this run did not capture older low-priority "
            "requests being jumped over."
        )
    elif high_priority_seen:
        verdict = "priority_seen_without_admission_proof"
        meaning = "SGLang saw the priority metadata, but admission order was not captured for the target request."
    else:
        verdict = "priority_not_proven"
        meaning = "The trace did not prove that SGLang saw the high-priority request."

    return {
        "high_request_id": args.high_request_id,
        "high_priority": args.high_priority,
        "high_client_submit_ns": high_submit_ns,
        "total_client_requests": len(metrics),
        "low_requests_submitted_before_high": len(low_before),
        "high_receive_events": len(high_receive),
        "high_queue_events": len(high_queues),
        "high_admission_events": len(high_admissions),
        "high_priority_seen": int(high_priority_seen),
        "high_admission_seq": high_seq if high_seq is not None else "",
        "older_low_admitted_before_high": len(older_low_admitted_before_high),
        "older_low_admitted_after_high": len(older_low_admitted_after_high),
        "older_low_jumped_ids": ";".join(older_low_admitted_after_high[:32]),
        "max_lower_priority_ahead_observed_for_high": max_lower_ahead,
        "best_high_queue_position_observed": min(queue_positions) if queue_positions else "",
        "verdict": verdict,
        "meaning": meaning,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a priority queue jump sanity run.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--high-request-id", default="pq_high_0000")
    parser.add_argument("--high-priority", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_jsonl(Path(args.metrics))
    trace = read_jsonl(Path(args.trace))
    receive_rows, admission_rows, queue_rows = flatten_priority_events(trace)
    summary = summarize(args)

    write_csv(out_dir / "priority_queue_sanity_summary.csv", [summary])
    write_csv(out_dir / "priority_queue_sanity_client_metrics.csv", metrics)
    write_csv(out_dir / "priority_queue_sanity_receive_events.csv", receive_rows)
    write_csv(out_dir / "priority_queue_sanity_admission_events.csv", admission_rows)
    write_csv(out_dir / "priority_queue_sanity_queue_events.csv", queue_rows)
    md = [
        "# Priority Queue Jump Sanity",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        summary["meaning"],
        "",
        "Key numbers:",
        "",
        f"- Low-priority requests submitted before high-priority request: {summary['low_requests_submitted_before_high']}",
        f"- High-priority SGLang receive events: {summary['high_receive_events']}",
        f"- High-priority queue snapshot events: {summary['high_queue_events']}",
        f"- High-priority admission events: {summary['high_admission_events']}",
        f"- Older low-priority requests admitted after high priority: {summary['older_low_admitted_after_high']}",
        f"- Older low-priority requests admitted before high priority: {summary['older_low_admitted_before_high']}",
    ]
    (out_dir / "priority_queue_sanity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
