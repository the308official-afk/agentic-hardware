#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def read_trace(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    return [event for event in events if isinstance(event, dict)]


def duration_us(event: dict[str, Any]) -> float:
    value = event.get("dur", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def event_name(event: dict[str, Any]) -> str:
    return str(event.get("name", ""))


def event_cat(event: dict[str, Any]) -> str:
    return str(event.get("cat", ""))


def is_cuda_event(event: dict[str, Any]) -> bool:
    text = f"{event_name(event)} {event_cat(event)}".lower()
    return any(token in text for token in ("cuda", "gpu", "kernel", "memcpy", "memset", "triton"))


def is_kernel_event(event: dict[str, Any]) -> bool:
    text = f"{event_name(event)} {event_cat(event)}".lower()
    return "kernel" in text or "triton" in text or event_cat(event).lower() in {"kernel"}


def is_memcpy_event(event: dict[str, Any]) -> bool:
    text = f"{event_name(event)} {event_cat(event)}".lower()
    return any(token in text for token in ("memcpy", "memset", "copy", "dma"))


def copy_direction(event: dict[str, Any]) -> str:
    text = event_name(event).lower()
    if "htod" in text or "host -> device" in text or "host to device" in text:
        return "h2d"
    if "dtoh" in text or "device -> host" in text or "device -> pinned" in text or "device to host" in text:
        return "d2h"
    if "dtod" in text or "device -> device" in text or "device to device" in text:
        return "d2d"
    if "memset" in text:
        return "memset"
    return "unknown"


def summarize_file(path: Path) -> dict[str, Any]:
    events = read_trace(path)
    cuda_events = [event for event in events if is_cuda_event(event)]
    kernel_events = [event for event in cuda_events if is_kernel_event(event)]
    memcpy_events = [event for event in cuda_events if is_memcpy_event(event)]
    runtime_events = [
        event
        for event in cuda_events
        if "cuda" in event_name(event).lower() or "cuda" in event_cat(event).lower()
    ]
    cuda_durations = [duration_us(event) / 1000.0 for event in cuda_events if duration_us(event) > 0]
    kernel_durations = [duration_us(event) / 1000.0 for event in kernel_events if duration_us(event) > 0]
    memcpy_durations = [duration_us(event) / 1000.0 for event in memcpy_events if duration_us(event) > 0]
    top_names = Counter(event_name(event) for event in cuda_events).most_common(20)
    direction_counts = Counter(copy_direction(event) for event in memcpy_events)
    memcpy_names = Counter(event_name(event) for event in memcpy_events).most_common(20)
    return {
        "profile": str(path),
        "total_events": len(events),
        "cuda_like_events": len(cuda_events),
        "kernel_like_events": len(kernel_events),
        "memcpy_like_events": len(memcpy_events),
        "runtime_like_events": len(runtime_events),
        "cuda_like_total_ms": round(sum(cuda_durations), 3),
        "cuda_like_avg_ms": round(mean(cuda_durations), 6) if cuda_durations else 0.0,
        "kernel_like_total_ms": round(sum(kernel_durations), 3),
        "kernel_like_avg_ms": round(mean(kernel_durations), 6) if kernel_durations else 0.0,
        "memcpy_like_total_ms": round(sum(memcpy_durations), 3),
        "memcpy_like_avg_ms": round(mean(memcpy_durations), 6) if memcpy_durations else 0.0,
        "memcpy_h2d_events": direction_counts["h2d"],
        "memcpy_d2h_events": direction_counts["d2h"],
        "memcpy_d2d_events": direction_counts["d2d"],
        "memcpy_memset_events": direction_counts["memset"],
        "memcpy_unknown_events": direction_counts["unknown"],
        "top_cuda_like_names": [{"name": name, "count": count} for name, count in top_names],
        "top_memcpy_like_names": [{"name": name, "count": count} for name, count in memcpy_names],
    }


def md_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No profiler JSON files found.", ""]
    headers = [
        "profile",
        "cuda_like_events",
        "kernel_like_events",
        "memcpy_like_events",
        "runtime_like_events",
        "cuda_like_total_ms",
        "kernel_like_total_ms",
        "memcpy_like_total_ms",
        "memcpy_h2d_events",
        "memcpy_d2h_events",
        "memcpy_d2d_events",
        "memcpy_unknown_events",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.append("")
    return lines


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Torch CUDA Profile Summary",
        "",
        "This summarizes Chrome traces exported by `torch.profiler` inside SGLang worker processes.",
        "",
        "## Profile Files",
        "",
    ]
    lines.extend(md_table(rows))
    lines.extend(["## Top CUDA-Like Event Names", ""])
    for row in rows:
        lines.append(f"### {Path(str(row['profile'])).name}")
        lines.append("")
        for item in row.get("top_cuda_like_names", []):
            lines.append(f"- `{item['name']}`: {item['count']}")
        lines.append("")
    lines.extend(["## Top Memcpy-Like Event Names", ""])
    for row in rows:
        lines.append(f"### {Path(str(row['profile'])).name}")
        lines.append("")
        for item in row.get("top_memcpy_like_names", []):
            lines.append(f"- `{item['name']}`: {item['count']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize torch.profiler CUDA Chrome traces.")
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    profiles = sorted(profile_dir.glob("torch_cuda_profile_*.json"))
    rows = [summarize_file(path) for path in profiles]
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_md, rows)
    print(f"Wrote torch CUDA profile JSON summary to {out_json}")
    print(f"Wrote torch CUDA profile Markdown summary to {out_md}")
    print(f"Profile files: {len(rows)}")
    print(f"Kernel-like events: {sum(int(row['kernel_like_events']) for row in rows)}")
    print(f"Memcpy-like events: {sum(int(row['memcpy_like_events']) for row in rows)}")


if __name__ == "__main__":
    main()
