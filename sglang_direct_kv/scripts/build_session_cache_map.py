#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                pass
    return events


def is_cache_event(event: dict[str, Any]) -> bool:
    name = str(event.get("event", ""))
    return name.startswith(("hicache.", "hiradix.", "radix."))


def session_timeline(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    timeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        session_id = event.get("session_id")
        if not session_id:
            continue
        item = {
            "event": event.get("event"),
            "ts_ns": event.get("ts_ns"),
            "label": event.get("label"),
            "phase": event.get("phase"),
            "prompt_hash": event.get("prompt_hash"),
            "timing": event.get("timing") or event.get("prefetch_timing"),
            "prefetch_action": event.get("prefetch_action"),
            "ttft_ms": event.get("ttft_ms"),
        }
        timeline[str(session_id)].append({k: v for k, v in item.items() if v is not None})
    for values in timeline.values():
        values.sort(key=lambda item: int(item.get("ts_ns") or 0))
    return dict(sorted(timeline.items()))


def cache_object_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for event in events:
        if not is_cache_event(event):
            continue
        self_info = event.get("self")
        if not isinstance(self_info, dict):
            continue
        object_id = self_info.get("object_id")
        if not object_id:
            continue
        entry = objects.setdefault(
            str(object_id),
            {
                "object_id": object_id,
                "object_type": self_info.get("object_type"),
                "events": Counter(),
                "metadata": {},
            },
        )
        entry["events"][str(event.get("event"))] += 1
        for key in ("size", "page_size", "dtype", "device", "mem_layout"):
            if key in self_info:
                entry["metadata"][key] = self_info[key]
        for key in ("device_pool", "host_pool", "token_to_kv_pool", "token_to_kv_pool_host"):
            if key in self_info:
                entry["metadata"][key] = self_info[key]

    out: list[dict[str, Any]] = []
    for entry in objects.values():
        entry["events"] = dict(entry["events"].most_common())
        out.append(entry)
    return sorted(out, key=lambda item: str(item.get("object_id")))


def build_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(event.get("event")) for event in events)
    prefix_maps = [event for event in events if event.get("event") == "agent.session_prefix_map"]
    direct_probes = [event for event in events if event.get("event") == "agent.direct_kv_prefetch_probe"]
    cache_counts = Counter(str(event.get("event")) for event in events if is_cache_event(event))
    return {
        "summary": {
            "total_events": len(events),
            "agent_sessions": len({event.get("session_id") for event in prefix_maps}),
            "direct_probe_events": len(direct_probes),
            "cache_event_types": len(cache_counts),
        },
        "event_counts": dict(counts.most_common()),
        "cache_event_counts": dict(cache_counts.most_common()),
        "session_prefix_map": [
            {
                "session_id": event.get("session_id"),
                "prompt_hash": event.get("prompt_hash"),
                "prompt_chars": event.get("prompt_chars"),
                "prompt_tokens_target": event.get("prompt_tokens_target"),
            }
            for event in prefix_maps
        ],
        "direct_probe_events": [
            {
                "session_id": event.get("session_id"),
                "prompt_hash": event.get("prompt_hash"),
                "timing": event.get("timing"),
                "intended_action": event.get("intended_action"),
                "probe_only": event.get("probe_only"),
            }
            for event in direct_probes
        ],
        "session_timeline": session_timeline(events),
        "cache_objects": cache_object_summary(events),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    summary = report["summary"]
    lines.append("# Milestone 7 Session/Cache Map")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Session Prefixes")
    lines.append("")
    lines.append("| session | prompt hash | prompt chars | target prompt tokens |")
    lines.append("| --- | --- | ---: | ---: |")
    for item in report["session_prefix_map"]:
        lines.append(
            f"| `{item.get('session_id')}` | `{item.get('prompt_hash')}` | "
            f"{item.get('prompt_chars')} | {item.get('prompt_tokens_target')} |"
        )
    lines.append("")
    lines.append("## Direct KV Prefetch Probe Events")
    lines.append("")
    lines.append("| session | prompt hash | timing | intended action | probe only |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in report["direct_probe_events"]:
        lines.append(
            f"| `{item.get('session_id')}` | `{item.get('prompt_hash')}` | "
            f"`{item.get('timing')}` | `{item.get('intended_action')}` | {item.get('probe_only')} |"
        )
    lines.append("")
    lines.append("## Cache Event Counts")
    lines.append("")
    for name, count in report["cache_event_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    lines.append("## Cache Objects Observed")
    lines.append("")
    for item in report["cache_objects"]:
        lines.append(f"### `{item.get('object_type')}` `{item.get('object_id')}`")
        lines.append("")
        lines.append(f"- events: `{json.dumps(item.get('events'), sort_keys=True)}`")
        metadata = item.get("metadata") or {}
        for key, value in metadata.items():
            lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True)[:300]}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a session-to-cache evidence report from a KV trace.")
    parser.add_argument("--trace", default="artifacts/results/milestone7/direct_hooks_trace.jsonl")
    parser.add_argument("--out-json", default="artifacts/results/milestone7/session_cache_map.json")
    parser.add_argument("--out-md", default="artifacts/results/milestone7/session_cache_map.md")
    args = parser.parse_args()

    events = read_jsonl(Path(args.trace))
    report = build_report(events)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, out_md)

    print(f"Wrote JSON report to {out_json}")
    print(f"Wrote Markdown report to {out_md}")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
