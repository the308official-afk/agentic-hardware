#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


INTERESTING_EVENTS = {
    "hicache.load.start",
    "hicache.load.end",
    "hicache.prefetch.start",
    "hicache.prefetch.end",
    "hiradix.ready_to_load_host_cache.start",
    "hiradix.ready_to_load_host_cache.end",
    "hiradix.match_prefix.start",
    "hiradix.match_prefix.end",
    "hiradix.load_back.start",
    "hiradix.load_back.end",
    "hiradix.init_load_back.start",
    "hiradix.init_load_back.end",
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


def compact(value: Any, limit: int = 800) -> str:
    text = json.dumps(value, sort_keys=True)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def extract_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in events:
        name = event.get("event")
        if name not in INTERESTING_EVENTS:
            continue
        calls.append(
            {
                "event": name,
                "class": event.get("class"),
                "method": event.get("method"),
                "duration_ms": event.get("duration_ms"),
                "self": event.get("self"),
                "args": event.get("args"),
                "kwargs": event.get("kwargs"),
                "result_metadata": event.get("result_metadata"),
                "result": event.get("result"),
                "ts_ns": event.get("ts_ns"),
            }
        )
    return calls


def write_markdown(calls: list[dict[str, Any]], events: list[dict[str, Any]], path: Path) -> None:
    counts = Counter(str(event.get("event")) for event in events)
    lines: list[str] = []
    lines.append("# HiCache Direct-Load Call Report")
    lines.append("")
    lines.append("## What We Are Looking For")
    lines.append("")
    lines.append("```text")
    lines.append("HiCacheController.load(host_indices, priority=None, node_id=-1)")
    lines.append("```")
    lines.append("")
    lines.append("The direct-load milestone needs the target session's `host_indices`.")
    lines.append("This report shows the argument/result shapes SGLang uses when it naturally loads host KV.")
    lines.append("")
    lines.append("## Event Counts")
    lines.append("")
    for event_name in sorted(INTERESTING_EVENTS):
        lines.append(f"- `{event_name}`: {counts.get(event_name, 0)}")
    lines.append("")
    lines.append("## Calls")
    lines.append("")
    if not calls:
        lines.append("No load/prefetch/match calls were found in this trace.")
    for idx, call in enumerate(calls, start=1):
        lines.append(f"### {idx}. `{call.get('event')}`")
        lines.append("")
        lines.append(f"- class: `{call.get('class')}`")
        lines.append(f"- method: `{call.get('method')}`")
        if call.get("duration_ms") is not None:
            lines.append(f"- duration_ms: `{call.get('duration_ms')}`")
        self_info = call.get("self") or {}
        if isinstance(self_info, dict):
            lines.append(f"- self.object_type: `{self_info.get('object_type')}`")
            lines.append(f"- self.object_id: `{self_info.get('object_id')}`")
            for key in ("page_size", "device", "token_to_kv_pool_host", "device_pool", "host_pool"):
                if key in self_info:
                    lines.append(f"- self.{key}: `{compact(self_info[key], 400)}`")
        for key in ("args", "kwargs", "result_metadata", "result"):
            if call.get(key) is not None:
                lines.append(f"- {key}: `{compact(call.get(key))}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HiCache direct-load call shapes from a trace.")
    parser.add_argument("--trace", default="artifacts/results/milestone7_pressure/direct_hooks_trace.jsonl")
    parser.add_argument("--out-json", default="artifacts/results/milestone7_pressure/hicache_call_report.json")
    parser.add_argument("--out-md", default="artifacts/results/milestone7_pressure/hicache_call_report.md")
    args = parser.parse_args()

    events = read_jsonl(Path(args.trace))
    calls = extract_calls(events)
    report = {
        "summary": {
            "total_events": len(events),
            "interesting_calls": len(calls),
            "hicache_load_start": sum(1 for call in calls if call["event"] == "hicache.load.start"),
            "hicache_load_end": sum(1 for call in calls if call["event"] == "hicache.load.end"),
        },
        "calls": calls,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(calls, events, out_md)

    print(f"Wrote JSON report to {out_json}")
    print(f"Wrote Markdown report to {out_md}")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
