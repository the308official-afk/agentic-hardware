#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def elapsed_ms(start: datetime | None, end: datetime | None) -> int:
    if start is None or end is None:
        return 0
    return max(0, int(round((end - start).total_seconds() * 1000.0)))


def prompt_tokens_estimate(prompt: str) -> int:
    return max(1, int(round(len(prompt.split()) * 1.35)))


def request_context_key(event: dict[str, Any]) -> tuple[str, str]:
    context = event.get("request_context") if isinstance(event.get("request_context"), dict) else {}
    return str(event.get("phase") or "unknown"), str(context.get("request_id") or event.get("sequence_index") or "")


def pair_turns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts = [
        event
        for event in events
        if event.get("event_kind") == "request_dispatch"
        and str(event.get("stage", "")).endswith("_request_dispatched")
        and event.get("prompt")
    ]
    ends = [
        event
        for event in events
        if event.get("event_kind") == "response"
        and str(event.get("stage", "")).endswith("_response_received")
    ]
    turns: list[dict[str, Any]] = []
    used: set[int] = set()
    for start in starts:
        start_ts = parse_ts(start.get("timestamp"))
        phase, request_id = request_context_key(start)
        match_idx = None
        match_end = None
        for idx, end in enumerate(ends):
            if idx in used:
                continue
            end_ts = parse_ts(end.get("timestamp"))
            if end_ts is None or start_ts is None or end_ts < start_ts:
                continue
            if str(end.get("phase") or "unknown") != phase:
                continue
            match_idx = idx
            match_end = end
            break
        if match_idx is None or match_end is None:
            continue
        used.add(match_idx)
        turns.append(
            {
                "phase": phase,
                "request_id": request_id,
                "start_event": start,
                "end_event": match_end,
                "start_ts": start_ts,
                "end_ts": parse_ts(match_end.get("timestamp")),
                "prompt": str(start.get("prompt") or ""),
                "hints": start.get("hints") if isinstance(start.get("hints"), dict) else {},
                "measurement": match_end.get("measurement") if isinstance(match_end.get("measurement"), dict) else {},
                "tool_progress": match_end.get("tool_progress") if isinstance(match_end.get("tool_progress"), dict) else {},
            }
        )
    return turns


def build_sessions(index_rows: list[dict[str, Any]], max_sessions: int, min_gap_ms: int) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    arrival_ms = 0
    for index_row in index_rows:
        run_id = str(index_row.get("run_id") or "")
        run_dir = Path(str(index_row.get("result_dir") or ""))
        lifecycle = read_json(run_dir / "others" / "stage_lifecycle_trace_raw.json")
        result = read_json(run_dir / "others" / "result.json") or {}
        if not isinstance(lifecycle, list):
            continue
        turns = pair_turns(lifecycle)
        for idx in range(len(turns) - 1):
            current = turns[idx]
            nxt = turns[idx + 1]
            wait_ms = elapsed_ms(current.get("end_ts"), nxt.get("start_ts"))
            if wait_ms < min_gap_ms:
                continue
            prompt = str(current.get("prompt") or "")
            replay_prompt = str(nxt.get("prompt") or "")
            if not prompt or not replay_prompt:
                continue
            hints = current.get("hints") if isinstance(current.get("hints"), dict) else {}
            priority = str(hints.get("priority") or "normal")
            sessions.append(
                {
                    "session_id": f"agentbench_{len(sessions):03d}_{run_id}_{current['phase']}_to_{nxt['phase']}",
                    "source": "agentbench_phase_trace",
                    "run_id": run_id,
                    "task_index": index_row.get("task_index", ""),
                    "repo": index_row.get("repo", result.get("task", {}).get("repo", "")),
                    "instance_id": result.get("task", {}).get("instance_id", ""),
                    "from_phase": current["phase"],
                    "to_phase": nxt["phase"],
                    "arrival_ms": arrival_ms,
                    "tool_wait_ms": wait_ms,
                    "priority": priority,
                    "prompt_tokens": prompt_tokens_estimate(prompt),
                    "replay_prompt_tokens": prompt_tokens_estimate(replay_prompt),
                    "prompt": prompt,
                    "replay_prompt": replay_prompt,
                    "current_latency_ms": current.get("measurement", {}).get("latency_ms", ""),
                    "next_latency_ms": nxt.get("measurement", {}).get("latency_ms", ""),
                    "current_tool_call_count": current.get("tool_progress", {}).get("tool_call_count", ""),
                    "next_tool_call_count": nxt.get("tool_progress", {}).get("tool_call_count", ""),
                }
            )
            arrival_ms += int(max(50, min(wait_ms, 500)))
            if len(sessions) >= max_sessions:
                return sessions
    return sessions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    compact_rows = [
        {key: value for key, value in row.items() if key not in {"prompt", "replay_prompt"}}
        for row in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compact_rows[0]))
        writer.writeheader()
        writer.writerows(compact_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a replayable workload from real AgentBench/DeepAgents model-turn traces."
    )
    parser.add_argument("--index-csv", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--max-sessions", type=int, default=24)
    parser.add_argument(
        "--min-gap-ms",
        type=int,
        default=0,
        help="Only include phase pairs with at least this much gap between model turns.",
    )
    args = parser.parse_args()

    index_rows = read_csv(args.index_csv)
    sessions = build_sessions(index_rows, max_sessions=args.max_sessions, min_gap_ms=args.min_gap_ms)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for session in sessions:
            f.write(json.dumps(session, sort_keys=True) + "\n")
    write_csv(args.out_csv, sessions)
    print(f"Wrote {len(sessions)} AgentBench replay sessions to {args.out_jsonl}")


if __name__ == "__main__":
    main()
