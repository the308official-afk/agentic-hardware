#!/usr/bin/env python
from __future__ import annotations

import argparse
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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def summarize_mode(root: Path, mode: str) -> dict[str, Any]:
    metrics = read_jsonl(root / f"{mode}_metrics.jsonl")
    trace = read_jsonl(root / f"{mode}_trace.jsonl")
    resume_rows = [row for row in metrics if row.get("phase") == "target_resume"]
    ttfts = [float(row["ttft_ms"]) for row in resume_rows if "ttft_ms" in row]
    event_counts = Counter(str(event.get("event")) for event in trace)

    return {
        "mode": mode,
        "resume_count": len(resume_rows),
        "resume_ttft_avg_ms": round(mean(ttfts), 3) if ttfts else 0.0,
        "resume_ttft_p95_ms": round(percentile(ttfts, 95), 3) if ttfts else 0.0,
        "hicache_write": event_counts.get("hicache.write.end", 0),
        "hicache_load": event_counts.get("hicache.load.end", 0),
        "hicache_evict_device": event_counts.get("hicache.evict_device.end", 0),
        "hiradix_evict": event_counts.get("hiradix.evict.end", 0),
        "agent_hint_submitted": event_counts.get("agent.hint_submitted", 0),
        "agent_generic_prefetch": event_counts.get("agent.generic_prefetch_end", 0),
        "agent_hint_prefetch": event_counts.get("agent.hint_prefetch_end", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Milestone 5 mode comparison results.")
    parser.add_argument("--root", default="artifacts/results/milestone5")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_prefetch", "generic_prefetch", "hint_aware"],
    )
    args = parser.parse_args()

    root = Path(args.root)
    summaries = [summarize_mode(root, mode) for mode in args.modes]

    if not summaries:
        print("No summaries found.")
        return

    fields = list(summaries[0])
    widths = {
        field: max(len(field), *(len(str(row[field])) for row in summaries))
        for field in fields
    }
    print(" | ".join(field.ljust(widths[field]) for field in fields))
    print("-+-".join("-" * widths[field] for field in fields))
    for row in summaries:
        print(" | ".join(str(row[field]).ljust(widths[field]) for field in fields))

    out_path = root / "summary.json"
    out_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print()
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
