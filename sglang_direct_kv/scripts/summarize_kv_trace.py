#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SGLang KV movement trace events.")
    parser.add_argument(
        "--trace",
        default="artifacts/kv_movement_trace.jsonl",
        help="Path to JSONL trace file.",
    )
    args = parser.parse_args()

    path = Path(args.trace)
    events = read_events(path)
    if not events:
        print(f"No events found at {path}")
        return

    counts = Counter(str(event.get("event")) for event in events)
    durations: dict[str, list[float]] = defaultdict(list)
    for event in events:
        duration = event.get("duration_ms")
        if isinstance(duration, (int, float)):
            durations[str(event.get("event"))].append(float(duration))

    print(f"Trace: {path}")
    print(f"Total events: {len(events)}")
    print()
    print("Event counts:")
    for name, count in counts.most_common():
        print(f"  {name}: {count}")

    if durations:
        print()
        print("Durations:")
        for name in sorted(durations):
            values = durations[name]
            print(
                f"  {name}: count={len(values)} "
                f"avg_ms={mean(values):.3f} max_ms={max(values):.3f}"
            )

    print()
    print("First events:")
    for event in events[:5]:
        print(json.dumps(event, sort_keys=True))

    print()
    print("Last events:")
    for event in events[-5:]:
        print(json.dumps(event, sort_keys=True))


if __name__ == "__main__":
    main()
