#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kv_context(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("kv_context")
    return value if isinstance(value, dict) else {}


def request_context(event: dict[str, Any]) -> dict[str, Any]:
    value = kv_context(event).get("request")
    return value if isinstance(value, dict) else {}


def agent_session(event: dict[str, Any]) -> str:
    ctx = kv_context(event)
    req = request_context(event)
    return str(
        ctx.get("agent_session_id")
        or req.get("agent_session_id")
        or event.get("session_id")
        or ""
    )


def agent_phase(event: dict[str, Any]) -> str:
    ctx = kv_context(event)
    req = request_context(event)
    return str(ctx.get("agent_phase") or req.get("agent_phase") or event.get("phase") or "unknown")


def index_count(value: Any) -> int:
    if isinstance(value, dict):
        for key in ("index_count", "numel", "count"):
            try:
                return int(value[key])
            except Exception:
                pass
    return 0


def index_signature(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    count = index_count(value)
    sha = value.get("sha1_16") or ""
    min_value = value.get("min")
    max_value = value.get("max")
    if min_value is not None and max_value is not None:
        return f"{min_value}..{max_value} n={count} sha={sha}"
    return f"n={count} sha={sha}"


def match_result_summary(event: dict[str, Any]) -> dict[str, Any]:
    result = event.get("result")
    out: dict[str, Any] = {
        "match_tokens": "",
        "host_hit_tokens": "",
        "node_id": "",
        "node_key_len": "",
        "node_evicted": "",
        "node_host_tokens": "",
        "node_gpu_tokens": "",
    }
    if not isinstance(result, list):
        return out
    if result:
        out["match_tokens"] = index_count(result[0])
    if len(result) > 3:
        out["host_hit_tokens"] = result[3]
    if len(result) > 1 and isinstance(result[1], dict):
        node = result[1]
        out.update(
            {
                "node_id": node.get("id", ""),
                "node_key_len": node.get("key_len", ""),
                "node_evicted": node.get("evicted", ""),
                "node_host_tokens": index_count(node.get("host_value")),
                "node_gpu_tokens": index_count(node.get("value")),
            }
        )
    return out


def case_sort_key(path: Path) -> tuple[str, int]:
    match = re.search(r"_f(\d+)$", path.name)
    fillers = int(match.group(1)) if match else -1
    return (path.name.split("_tw")[0], fillers)


def summarize_case(case_dir: Path, row_by_case: dict[str, dict[str, str]]) -> dict[str, Any]:
    events = read_jsonl(case_dir / "m27_trace.jsonl")
    row = row_by_case.get(case_dir.name, {})
    session_id = row.get("session_id") or ""
    t0 = min((int(event.get("ts_ns") or 0) for event in events if event.get("ts_ns")), default=0)
    phase_totals: dict[str, Counter[str]] = defaultdict(Counter)
    match_rows: list[dict[str, Any]] = []
    movement_rows: list[dict[str, Any]] = []

    for event in events:
        if session_id and agent_session(event) != session_id:
            continue
        name = str(event.get("event") or "")
        phase = agent_phase(event)
        ctx = kv_context(event)
        ts_ns = int(event.get("ts_ns") or 0)
        rel_ms = round((ts_ns - t0) / 1_000_000, 3) if t0 and ts_ns else ""

        if name == "hiradix.match_prefix.end":
            match_rows.append(
                {
                    "time_ms": rel_ms,
                    "phase": phase,
                    "duration_ms": event.get("duration_ms", ""),
                    **match_result_summary(event),
                }
            )
        elif name == "hicache.write.end":
            tokens = index_count(ctx.get("device_indices"))
            phase_totals[phase]["d2h_write_tokens"] += tokens
            movement_rows.append(
                {
                    "time_ms": rel_ms,
                    "phase": phase,
                    "event": name,
                    "direction": "device_to_host",
                    "tokens": tokens,
                    "node_id": ctx.get("node_id", ""),
                    "indices": index_signature(ctx.get("device_indices")),
                }
            )
        elif name == "hicache.evict_device.end":
            tokens = index_count(ctx.get("device_indices"))
            phase_totals[phase]["device_evict_tokens"] += tokens
            movement_rows.append(
                {
                    "time_ms": rel_ms,
                    "phase": phase,
                    "event": name,
                    "direction": "device_evict",
                    "tokens": tokens,
                    "node_id": ctx.get("node_id", ""),
                    "indices": index_signature(ctx.get("device_indices")),
                }
            )
        elif name == "hicache.evict_host.end":
            tokens = index_count(ctx.get("host_indices"))
            phase_totals[phase]["host_evict_tokens"] += tokens
            movement_rows.append(
                {
                    "time_ms": rel_ms,
                    "phase": phase,
                    "event": name,
                    "direction": "host_evict",
                    "tokens": tokens,
                    "node_id": ctx.get("node_id", ""),
                    "indices": index_signature(ctx.get("host_indices")),
                }
            )
        elif name == "hicache.load.end":
            tokens = index_count(ctx.get("host_indices")) or index_count(ctx.get("device_indices"))
            phase_totals[phase]["h2d_load_tokens"] += tokens
            movement_rows.append(
                {
                    "time_ms": rel_ms,
                    "phase": phase,
                    "event": name,
                    "direction": "host_to_device",
                    "tokens": tokens,
                    "node_id": ctx.get("node_id", ""),
                    "indices": index_signature(ctx.get("host_indices")),
                }
            )
        elif name == "hiradix.init_load_back.end":
            phase_totals[phase]["init_load_back_events"] += 1
        elif name == "hiradix.load_back.end":
            phase_totals[phase]["load_back_events"] += 1

    first_replay_match = next((item for item in match_rows if item["phase"] == "replay"), {})
    diagnosis = diagnose(row, phase_totals, first_replay_match)
    return {
        "case": case_dir.name,
        "session_id": session_id,
        "mode": row.get("mode", ""),
        "tool_gap_ms": row.get("tool_gap_ms", ""),
        "resume_ttft_ms": row.get("resume_ttft_ms", ""),
        "replay_initial_cached_prefix_tokens": row.get("replay_initial_cached_prefix_tokens", ""),
        "replay_final_cached_prefix_tokens": row.get("replay_final_cached_prefix_tokens", ""),
        "replay_new_prefill_tokens_est": row.get("replay_new_prefill_tokens_est", ""),
        "replay_host_load_tokens": row.get("replay_host_load_tokens", ""),
        "replay_kv_h2d_events": row.get("replay_kv_h2d_events", ""),
        "final_path": row.get("final_path", ""),
        "phase_totals": {phase: dict(counter) for phase, counter in phase_totals.items()},
        "first_replay_match": first_replay_match,
        "matches": match_rows,
        "movement_rows": movement_rows,
        "diagnosis": diagnosis,
    }


def diagnose(row: dict[str, str], phase_totals: dict[str, Counter[str]], first_replay_match: dict[str, Any]) -> str:
    initial = phase_totals.get("initial_turn", Counter())
    replay = phase_totals.get("replay", Counter())
    matched = int(float(row.get("replay_initial_cached_prefix_tokens") or 0))
    replay_h2d = int(float(row.get("replay_kv_h2d_events") or 0))
    host_load = int(float(row.get("replay_host_load_tokens") or 0))
    host_evicted = initial.get("host_evict_tokens", 0)
    device_evicted = initial.get("device_evict_tokens", 0)
    initial_written = initial.get("d2h_write_tokens", 0)
    replay_written = replay.get("d2h_write_tokens", 0)

    if matched > 1000 and not replay_h2d and not host_load:
        return "Replay reused a large GPU/radix prefix. HiCache existed but load-back was not needed."
    if matched <= 64 and host_evicted:
        return (
            "Target KV was written to host, then target host cache entries were evicted before replay. "
            "Replay could not load them back, so it rebuilt/prefilled the missing prefix."
        )
    if matched <= 64 and device_evicted and not host_evicted:
        return (
            "Target KV was evicted from GPU, but replay did not observe host load-back. "
            "This suggests the host-backed radix path was not reachable or load-back was skipped."
        )
    if replay_h2d or host_load:
        return "Replay used the host-to-device KV load-back path."
    if initial_written and replay_written:
        return "KV was written to host, but replay mainly produced new KV and wrote it back; this is recompute/prefill behavior."
    return "No single cause found from trace; inspect match and movement rows."


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    cases = report["cases"]
    lines = [
        "# HiCache Reuse Investigation",
        "",
        "This report answers a narrow question: HiCache was enabled, so why did some replay requests recompute instead of loading KV back from host?",
        "",
        "## Summary",
        "",
        table(
            [
                {
                    "case": c["case"],
                    "mode": c["mode"],
                    "TTFT ms": c["resume_ttft_ms"],
                    "initial match": c["replay_initial_cached_prefix_tokens"],
                    "new prefill": c["replay_new_prefill_tokens_est"],
                    "replay H2D": c["replay_kv_h2d_events"],
                    "path": c["final_path"],
                    "diagnosis": c["diagnosis"],
                }
                for c in cases
            ],
            ["case", "mode", "TTFT ms", "initial match", "new prefill", "replay H2D", "path", "diagnosis"],
        ),
        "",
        "## Per-Case Movement Totals",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"### `{case['case']}`",
                "",
                f"- diagnosis: {case['diagnosis']}",
                f"- first replay match: `{json.dumps(case['first_replay_match'], sort_keys=True)}`",
                "",
                table(
                    [
                        {"phase": phase, **totals}
                        for phase, totals in sorted(case["phase_totals"].items())
                    ],
                    [
                        "phase",
                        "d2h_write_tokens",
                        "device_evict_tokens",
                        "host_evict_tokens",
                        "h2d_load_tokens",
                        "init_load_back_events",
                        "load_back_events",
                    ],
                ),
                "",
            ]
        )
    lines.extend(["## Target Movement Events", ""])
    for case in cases:
        lines.extend(
            [
                f"### `{case['case']}`",
                "",
                table(
                    case["movement_rows"][:80],
                    ["time_ms", "phase", "event", "direction", "tokens", "node_id", "indices"],
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--gaps-csv", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    gaps_csv = args.gaps_csv or args.run_root / "controlled_replay_report" / "controlled_replay_gaps.csv"
    rows = read_csv(gaps_csv)
    row_by_case = {Path(row.get("case_dir", "")).name: row for row in rows}
    case_dirs = sorted(
        [path for path in args.run_root.iterdir() if path.is_dir() and (path / "m27_trace.jsonl").exists()],
        key=case_sort_key,
    )
    report = {"run_root": str(args.run_root), "gaps_csv": str(gaps_csv), "cases": [summarize_case(path, row_by_case) for path in case_dirs]}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
