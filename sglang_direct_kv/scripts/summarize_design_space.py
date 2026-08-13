#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
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


def avg(rows: list[dict[str, Any]], phase: str) -> float:
    values = [float(row["ttft_ms"]) for row in rows if row.get("phase") == phase and "ttft_ms" in row]
    return round(mean(values), 3) if values else 0.0


def count_phase(rows: list[dict[str, Any]], phase: str) -> int:
    return sum(1 for row in rows if row.get("phase") == phase)


def summarize_case(metrics_path: Path) -> dict[str, Any] | None:
    metrics = read_jsonl(metrics_path)
    if not metrics:
        return None

    first = metrics[0]
    trace_path = metrics_path.with_name(metrics_path.name.replace("_metrics.jsonl", "_trace.jsonl"))
    trace = read_jsonl(trace_path)
    event_counts = Counter(str(event.get("event")) for event in trace)

    return {
        "case": metrics_path.name.removesuffix("_metrics.jsonl"),
        "mode": first.get("mode", "unknown"),
        "hint_timing": first.get("hint_prefetch_timing", "unknown"),
        "filler_sessions": int(first.get("filler_sessions", 0)),
        "prompt_tokens": int(first.get("prompt_tokens", 0)),
        "warm_count": count_phase(metrics, "target_warm"),
        "warm_ttft_avg_ms": avg(metrics, "target_warm"),
        "prefetch_count": count_phase(metrics, "hint_prefetch") + count_phase(metrics, "generic_prefetch"),
        "prefetch_ttft_avg_ms": avg(metrics, "hint_prefetch") or avg(metrics, "generic_prefetch"),
        "resume_count": count_phase(metrics, "target_resume"),
        "resume_ttft_avg_ms": avg(metrics, "target_resume"),
        "hicache_write": event_counts.get("hicache.write.end", 0),
        "hicache_load": event_counts.get("hicache.load.end", 0),
        "hicache_evict_device": event_counts.get("hicache.evict_device.end", 0),
        "hiradix_evict": event_counts.get("hiradix.evict.end", 0),
        "agent_hint_prefetch": event_counts.get("agent.hint_prefetch_end", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Milestone 6 design-space sweep.")
    parser.add_argument("--root", default="artifacts/results/milestone6_design_space")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [
        row
        for path in sorted(root.glob("*_metrics.jsonl"))
        if (row := summarize_case(path)) is not None
    ]
    if not rows:
        print(f"No metrics files found under {root}")
        return

    baselines = {
        (row["filler_sessions"], row["prompt_tokens"]): row["resume_ttft_avg_ms"]
        for row in rows
        if row["mode"] == "no_prefetch"
    }
    for row in rows:
        baseline = baselines.get((row["filler_sessions"], row["prompt_tokens"]), 0.0)
        row["benefit_vs_no_prefetch_ms"] = round(baseline - row["resume_ttft_avg_ms"], 3) if baseline else 0.0
        row["benefit_vs_no_prefetch_pct"] = (
            round((baseline - row["resume_ttft_avg_ms"]) * 100.0 / baseline, 2)
            if baseline
            else 0.0
        )

    rows.sort(key=lambda row: (row["prompt_tokens"], row["filler_sessions"], row["mode"], row["hint_timing"]))

    fields = [
        "mode",
        "hint_timing",
        "filler_sessions",
        "prompt_tokens",
        "warm_ttft_avg_ms",
        "prefetch_ttft_avg_ms",
        "resume_ttft_avg_ms",
        "benefit_vs_no_prefetch_ms",
        "benefit_vs_no_prefetch_pct",
        "hicache_load",
        "hicache_evict_device",
    ]
    widths = {field: max(len(field), *(len(str(row[field])) for row in rows)) for field in fields}
    print(" | ".join(field.ljust(widths[field]) for field in fields))
    print("-+-".join("-" * widths[field] for field in fields))
    for row in rows:
        print(" | ".join(str(row[field]).ljust(widths[field]) for field in fields))

    json_path = root / "summary.json"
    csv_path = root / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    print()
    print(f"Wrote summary to {json_path}")
    print(f"Wrote summary to {csv_path}")


if __name__ == "__main__":
    main()
