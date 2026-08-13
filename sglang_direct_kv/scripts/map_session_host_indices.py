#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    return sorted(rows, key=lambda item: int(item.get("ts_ns") or 0))


def tensor_numel(summary: Any) -> int | None:
    if isinstance(summary, dict) and "numel" in summary:
        try:
            return int(summary["numel"])
        except Exception:
            return None
    return None


def tensor_shape(summary: Any) -> Any:
    if isinstance(summary, dict):
        return summary.get("shape")
    return None


def node_summary(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and value.get("type") == "TreeNode":
        return value
    return None


def request_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    windows: list[dict[str, Any]] = []
    for event in events:
        name = event.get("event")
        label = event.get("label")
        if not label:
            continue
        if name == "agent.request.start":
            starts[str(label)] = event
        elif name == "agent.request.end" and str(label) in starts:
            start = starts.pop(str(label))
            windows.append(
                {
                    "label": label,
                    "session_id": event.get("session_id") or start.get("session_id"),
                    "phase": event.get("phase") or start.get("phase"),
                    "prompt_hash": event.get("prompt_hash") or start.get("prompt_hash"),
                    "start_ts_ns": start.get("ts_ns"),
                    "end_ts_ns": event.get("ts_ns"),
                    "ttft_ms": event.get("ttft_ms"),
                    "total_latency_ms": event.get("total_latency_ms"),
                }
            )
    return windows


def summarize_match(event: dict[str, Any]) -> dict[str, Any]:
    result = event.get("result")
    out: dict[str, Any] = {
        "ts_ns": event.get("ts_ns"),
        "duration_ms": event.get("duration_ms"),
    }
    if isinstance(result, list) and len(result) >= 4:
        device_indices = result[0]
        last_device_node = node_summary(result[1])
        last_host_node = node_summary(result[2])
        out["device_indices_numel"] = tensor_numel(device_indices)
        out["device_indices_shape"] = tensor_shape(device_indices)
        out["last_device_node"] = last_device_node
        out["last_host_node"] = last_host_node
        out["host_hit_length"] = result[3]
        if last_host_node:
            out["last_host_node_id"] = last_host_node.get("id")
            out["last_host_node_host_value_numel"] = tensor_numel(last_host_node.get("host_value"))
            out["last_host_node_host_value_shape"] = tensor_shape(last_host_node.get("host_value"))
    return out


def summarize_load(event: dict[str, Any]) -> dict[str, Any]:
    kwargs = event.get("kwargs") or {}
    host_indices = kwargs.get("host_indices") if isinstance(kwargs, dict) else None
    return {
        "ts_ns": event.get("ts_ns"),
        "node_id": kwargs.get("node_id") if isinstance(kwargs, dict) else None,
        "host_indices_numel": tensor_numel(host_indices),
        "host_indices_shape": tensor_shape(host_indices),
        "host_indices": host_indices,
    }


def build_mapping(events: list[dict[str, Any]]) -> dict[str, Any]:
    windows = request_windows(events)
    mapped: list[dict[str, Any]] = []
    for window in windows:
        start = int(window.get("start_ts_ns") or 0)
        end = int(window.get("end_ts_ns") or 0)
        scoped = [event for event in events if start <= int(event.get("ts_ns") or 0) <= end]
        matches = [
            summarize_match(event)
            for event in scoped
            if event.get("event") == "hiradix.match_prefix.end"
        ]
        loads = [
            summarize_load(event)
            for event in scoped
            if event.get("event") == "hicache.load.start"
        ]
        evidence = []
        for match in matches:
            for load in loads:
                same_length = (
                    match.get("host_hit_length") is not None
                    and match.get("host_hit_length") == load.get("host_indices_numel")
                )
                same_node = (
                    match.get("last_host_node_id") is not None
                    and match.get("last_host_node_id") == load.get("node_id")
                )
                if same_length or same_node:
                    evidence.append(
                        {
                            "match_ts_ns": match.get("ts_ns"),
                            "load_ts_ns": load.get("ts_ns"),
                            "same_host_length": same_length,
                            "same_node_id": same_node,
                            "host_hit_length": match.get("host_hit_length"),
                            "host_node_id": match.get("last_host_node_id"),
                            "load_node_id": load.get("node_id"),
                            "load_host_indices_numel": load.get("host_indices_numel"),
                        }
                    )
        mapped.append(
            {
                **window,
                "match_prefix_count": len(matches),
                "hicache_load_count": len(loads),
                "matches": matches,
                "loads": loads,
                "mapping_evidence": evidence,
            }
        )
    return {
        "summary": {
            "total_events": len(events),
            "request_windows": len(windows),
            "windows_with_hicache_load": sum(1 for item in mapped if item["hicache_load_count"]),
            "mapping_evidence_count": sum(len(item["mapping_evidence"]) for item in mapped),
        },
        "requests": mapped,
    }


def compact(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, sort_keys=True)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Session To Host Indices Map")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in report["summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Request Windows")
    lines.append("")
    lines.append("| label | session | phase | TTFT ms | match calls | load calls | mapping evidence |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for item in report["requests"]:
        lines.append(
            f"| `{item.get('label')}` | `{item.get('session_id')}` | `{item.get('phase')}` | "
            f"{item.get('ttft_ms')} | {item.get('match_prefix_count')} | "
            f"{item.get('hicache_load_count')} | {len(item.get('mapping_evidence') or [])} |"
        )
    lines.append("")
    lines.append("## Load Evidence")
    lines.append("")
    for item in report["requests"]:
        evidence = item.get("mapping_evidence") or []
        if not evidence:
            continue
        lines.append(f"### `{item.get('label')}`")
        lines.append("")
        lines.append(f"- session: `{item.get('session_id')}`")
        lines.append(f"- phase: `{item.get('phase')}`")
        lines.append(f"- prompt_hash: `{item.get('prompt_hash')}`")
        for idx, ev in enumerate(evidence, start=1):
            lines.append(f"- evidence {idx}: `{compact(ev, 500)}`")
        lines.append("")
    lines.append("## Host Node Details")
    lines.append("")
    for item in report["requests"]:
        for match in item.get("matches") or []:
            host_hit_length = match.get("host_hit_length")
            host_node = match.get("last_host_node")
            if not host_hit_length or not host_node:
                continue
            lines.append(f"### `{item.get('label')}` host hit `{host_hit_length}`")
            lines.append("")
            lines.append(f"- host_node_id: `{host_node.get('id')}`")
            lines.append(f"- host_node.evicted: `{host_node.get('evicted')}`")
            lines.append(f"- host_node.backuped: `{host_node.get('backuped')}`")
            lines.append(f"- host_node.host_value: `{compact(host_node.get('host_value'), 700)}`")
            lines.append("")
    lines.append("## HiCache Load Details")
    lines.append("")
    for item in report["requests"]:
        loads = item.get("loads") or []
        if not loads:
            continue
        lines.append(f"### `{item.get('label')}`")
        lines.append("")
        for idx, load in enumerate(loads, start=1):
            lines.append(f"- load {idx} node_id: `{load.get('node_id')}`")
            lines.append(f"- load {idx} host_indices_numel: `{load.get('host_indices_numel')}`")
            lines.append(f"- load {idx} host_indices: `{compact(load.get('host_indices'), 700)}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map request/session windows to HiCache host-index loads.")
    parser.add_argument("--trace", default="artifacts/results/milestone7_pressure/direct_hooks_trace.jsonl")
    parser.add_argument("--out-json", default="artifacts/results/milestone7_pressure/session_host_indices_map.json")
    parser.add_argument("--out-md", default="artifacts/results/milestone7_pressure/session_host_indices_map.md")
    args = parser.parse_args()

    events = read_jsonl(Path(args.trace))
    report = build_mapping(events)

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
