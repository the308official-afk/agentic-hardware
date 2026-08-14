#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


LOAD_EVENTS = {"hicache.load"}
KV_EVENTS = {
    "hicache.load",
    "hicache.write",
    "hicache.evict_device",
    "hiradix.load_back",
    "hiradix.init_load_back",
    "hicache.start_loading",
    "hicache.start_writing",
    "hostpool.load_to_device_per_layer",
    "hostpool.backup_from_device_all_layer",
    "hiradix.cache_finished_req",
    "hiradix.cache_unfinished_req",
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


def read_profile(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    return [event for event in events if isinstance(event, dict) and event.get("ph") == "X"]


def event_name(event: dict[str, Any]) -> str:
    return str(event.get("name", ""))


def event_cat(event: dict[str, Any]) -> str:
    return str(event.get("cat", ""))


def duration_us(event: dict[str, Any]) -> float:
    value = event.get("dur", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def is_kernel_event(event: dict[str, Any]) -> bool:
    text = f"{event_name(event)} {event_cat(event)}".lower()
    return "kernel" in text or "triton" in text or event_cat(event).lower() in {"kernel"}


def is_memcpy_event(event: dict[str, Any]) -> bool:
    text = f"{event_name(event)} {event_cat(event)}".lower()
    return any(token in text for token in ("memcpy", "memset", "copy", "dma"))


def is_transfer_event(event: dict[str, Any]) -> bool:
    name = event_name(event).lower()
    category = event_cat(event).lower()
    return category in {"gpu_memcpy", "gpu_memset"} or name.startswith("memcpy ") or name.startswith("memset ")


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


def bytes_for(event: dict[str, Any]) -> int:
    args = event.get("args", {})
    if not isinstance(args, dict):
        return 0
    value = args.get("bytes", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def pid_from_profile_name(path: Path) -> int | None:
    parts = path.name.split("_")
    for part in parts:
        if part.startswith("pid"):
            try:
                return int(part.removeprefix("pid"))
            except ValueError:
                return None
    return None


def profiler_start_by_pid(profile_dir: Path) -> dict[int, int]:
    starts: dict[int, int] = {}
    for path in profile_dir.glob("torch_profiler_status_pid*.jsonl"):
        for event in read_jsonl(path):
            if event.get("event") != "torch_profiler.start":
                continue
            pid = event.get("pid")
            ts_ns = event.get("ts_ns")
            if isinstance(pid, int) and isinstance(ts_ns, int):
                starts[pid] = ts_ns
                break
    return starts


def profile_cuda_events(profile_path: Path, start_wall_ns: int) -> list[dict[str, Any]]:
    events = read_profile(profile_path)
    if not events:
        return []
    min_ts_us = min(float(event.get("ts", 0.0)) for event in events if isinstance(event.get("ts"), (int, float)))
    rows: list[dict[str, Any]] = []
    for event in events:
        is_kernel = is_kernel_event(event)
        is_memcpy = is_memcpy_event(event)
        if not is_kernel and not is_memcpy:
            continue
        ts_us = float(event.get("ts", 0.0))
        dur_us = duration_us(event)
        start_ns = int(start_wall_ns + (ts_us - min_ts_us) * 1000)
        end_ns = int(start_ns + dur_us * 1000)
        rows.append(
            {
                "profile": str(profile_path),
                "name": event_name(event),
                "category": event_cat(event),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": dur_us / 1000.0,
                "kind": "memcpy" if is_memcpy else "kernel",
                "direction": copy_direction(event) if is_memcpy else "",
                "bytes": bytes_for(event),
            }
        )
    return rows


def request_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts = [event for event in events if event.get("event") == "agent.request.start"]
    ends = [event for event in events if event.get("event") == "agent.request.end"]
    windows: list[dict[str, Any]] = []
    used: set[int] = set()
    for start in sorted(starts, key=lambda e: int(e.get("ts_ns", 0))):
        start_ts = int(start.get("ts_ns", 0))
        session_id = str(start.get("session_id", ""))
        phase = str(start.get("phase", ""))
        candidates = [
            (idx, end)
            for idx, end in enumerate(ends)
            if idx not in used
            and int(end.get("ts_ns", 0)) >= start_ts
            and str(end.get("session_id", "")) == session_id
            and str(end.get("phase", "")) == phase
        ]
        if not candidates:
            continue
        idx, end = min(candidates, key=lambda item: int(item[1].get("ts_ns", 0)))
        used.add(idx)
        windows.append(
            {
                "window_type": "agent_request",
                "event": "agent.request",
                "session_id": session_id,
                "phase": phase,
                "mode": start.get("mode", ""),
                "start_ns": start_ts,
                "end_ns": int(end.get("ts_ns", 0)),
            }
        )
    return windows


def paired_method_windows(events: list[dict[str, Any]], bases: set[str]) -> list[dict[str, Any]]:
    starts = [
        event
        for event in events
        if isinstance(event.get("event"), str)
        and str(event["event"]).endswith(".start")
        and str(event["event"]).removesuffix(".start") in bases
    ]
    ends = [
        event
        for event in events
        if isinstance(event.get("event"), str)
        and str(event["event"]).endswith(".end")
        and str(event["event"]).removesuffix(".end") in bases
    ]
    windows: list[dict[str, Any]] = []
    used: set[int] = set()
    for start in sorted(starts, key=lambda e: int(e.get("ts_ns", 0))):
        base = str(start["event"]).removesuffix(".start")
        pid = start.get("pid")
        call_id = start.get("call_id")
        start_ts = int(start.get("ts_ns", 0))
        candidates = [
            (idx, end)
            for idx, end in enumerate(ends)
            if idx not in used
            and str(end["event"]).removesuffix(".end") == base
            and end.get("pid") == pid
            and (call_id is None or end.get("call_id") == call_id)
            and int(end.get("ts_ns", 0)) >= start_ts
        ]
        if not candidates:
            continue
        idx, end = min(candidates, key=lambda item: int(item[1].get("ts_ns", 0)))
        used.add(idx)
        windows.append(
            {
                "window_type": "sglang_kv_method",
                "event": base,
                "session_id": "",
                "phase": "",
                "mode": "",
                "pid": pid,
                "call_id": start.get("call_id") or "",
                "start_ns": start_ts,
                "end_ns": int(end.get("ts_ns", 0)),
                "duration_ms": float(end.get("duration_ms", 0.0) or 0.0),
                "kv_context_start": start.get("kv_context") or {},
                "kv_context_end": end.get("kv_context") or {},
            }
        )
    return windows


def overlaps(event: dict[str, Any], window: dict[str, Any]) -> bool:
    return int(event["start_ns"]) < int(window["end_ns"]) and int(event["end_ns"]) > int(window["start_ns"])


def overlap_ns(left: dict[str, Any], right: dict[str, Any]) -> int:
    return max(0, min(int(left["end_ns"]), int(right["end_ns"])) - max(int(left["start_ns"]), int(right["start_ns"])))


def annotate_kv_windows_with_agent(windows: list[dict[str, Any]]) -> None:
    requests = [window for window in windows if window.get("window_type") == "agent_request"]
    for window in windows:
        if window.get("window_type") != "sglang_kv_method":
            continue
        matches = [request for request in requests if overlaps(window, request)]
        if not matches:
            continue
        matches.sort(key=lambda request: overlap_ns(window, request), reverse=True)
        best = matches[0]
        window["session_id"] = best.get("session_id", "")
        window["phase"] = best.get("phase", "")
        window["agent_overlap_count"] = len(matches)


def merged_kv_context(window: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("kv_context_start", "kv_context_end"):
        value = window.get(key)
        if isinstance(value, dict):
            merged.update({k: v for k, v in value.items() if v not in (None, "", [], {})})
    return merged


def compact_json(value: Any, limit: int = 220) -> str:
    if value in (None, "", [], {}):
        return ""
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "..."


def first_queue_op(context: dict[str, Any]) -> dict[str, Any]:
    ops = context.get("queued_ops")
    if isinstance(ops, list) and ops and isinstance(ops[0], dict):
        return ops[0]
    return {}


def context_index(context: dict[str, Any], name: str) -> dict[str, Any]:
    value = context.get(name)
    if isinstance(value, dict):
        return value
    op = first_queue_op(context)
    value = op.get(name)
    return value if isinstance(value, dict) else {}


def context_node_id(context: dict[str, Any]) -> Any:
    if "node_id" in context:
        return context.get("node_id")
    op = first_queue_op(context)
    if "node_id" in op:
        return op.get("node_id")
    node_ids = op.get("node_ids")
    if node_ids:
        return node_ids
    return ""


def request_context(context: dict[str, Any]) -> dict[str, Any]:
    value = context.get("request")
    return value if isinstance(value, dict) else {}


def node_ids_from_value(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def index_signature(index: dict[str, Any]) -> str:
    if not isinstance(index, dict):
        return ""
    sha = index.get("sha1_16")
    count = index.get("index_count") or index.get("numel")
    if sha and count:
        return f"{count}:{sha}"
    values = index.get("values")
    if isinstance(values, list):
        return f"{len(values)}:{','.join(str(item) for item in values)}"
    return ""


def build_request_maps(windows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    node_map: dict[str, dict[str, Any]] = {}
    index_map: dict[str, dict[str, Any]] = {}

    for window in windows:
        context = merged_kv_context(window)
        req = request_context(context)
        session_id = req.get("agent_session_id")
        if not session_id:
            continue
        info = {
            "agent_session_id": session_id,
            "rid": req.get("rid", ""),
            "agent_phase": req.get("agent_phase", ""),
            "agent_label": req.get("agent_label", ""),
            "agent_prompt_hash": req.get("agent_prompt_hash", ""),
            "source_event": window.get("event", ""),
        }
        for node_key in ("last_node_id", "last_host_node_id"):
            for node_id in node_ids_from_value(req.get(node_key)):
                node_map[node_id] = info
        for index_name in ("prefix_indices",):
            sig = index_signature(req.get(index_name) if isinstance(req.get(index_name), dict) else {})
            if sig:
                index_map[sig] = info

    for window in windows:
        context = merged_kv_context(window)
        info = None
        for node_id in node_ids_from_value(context_node_id(context)):
            if node_id in node_map:
                info = node_map[node_id]
                break
        if info is None:
            for index_name in ("host_indices", "device_indices"):
                sig = index_signature(context_index(context, index_name))
                if sig and sig in index_map:
                    info = index_map[sig]
                    break
        if info:
            for index_name in ("host_indices", "device_indices"):
                sig = index_signature(context_index(context, index_name))
                if sig:
                    index_map.setdefault(sig, info)

    return node_map, index_map


def annotate_kv_windows_with_request_maps(
    windows: list[dict[str, Any]],
    node_map: dict[str, dict[str, Any]],
    index_map: dict[str, dict[str, Any]],
) -> None:
    for window in windows:
        if window.get("window_type") != "sglang_kv_method":
            continue
        context = merged_kv_context(window)
        req = request_context(context)
        info: dict[str, Any] | None = None
        if req.get("agent_session_id"):
            info = {
                "agent_session_id": req.get("agent_session_id"),
                "rid": req.get("rid", ""),
                "agent_phase": req.get("agent_phase", ""),
                "agent_label": req.get("agent_label", ""),
                "agent_prompt_hash": req.get("agent_prompt_hash", ""),
                "source_event": window.get("event", ""),
            }
        if info is None:
            for node_id in node_ids_from_value(context_node_id(context)):
                if node_id in node_map:
                    info = node_map[node_id]
                    break
        if info is None:
            for index_name in ("host_indices", "device_indices"):
                sig = index_signature(context_index(context, index_name))
                if sig and sig in index_map:
                    info = index_map[sig]
                    break
        if info:
            window["kv_agent_session_id"] = info.get("agent_session_id", "")
            window["kv_request_rid"] = info.get("rid", "")
            window["kv_agent_phase"] = info.get("agent_phase", "")
            window["kv_agent_label"] = info.get("agent_label", "")
            window["kv_agent_prompt_hash"] = info.get("agent_prompt_hash", "")
            window["kv_request_source_event"] = info.get("source_event", "")


def index_values_text(index: dict[str, Any]) -> str:
    if "values" in index:
        return compact_json(index["values"], 360)
    parts = []
    if "head" in index:
        parts.append(f"head={compact_json(index['head'], 140)}")
    if "tail" in index:
        parts.append(f"tail={compact_json(index['tail'], 140)}")
    if "sha1_16" in index:
        parts.append(f"sha1_16={index['sha1_16']}")
    return "; ".join(parts)


def kv_columns(window: dict[str, Any]) -> dict[str, Any]:
    context = merged_kv_context(window)
    host = context_index(context, "host_indices")
    device = context_index(context, "device_indices")
    return {
        "kv_direction": context.get("direction", ""),
        "kv_agent_session_id": window.get("kv_agent_session_id", ""),
        "kv_agent_phase": window.get("kv_agent_phase", ""),
        "kv_agent_label": window.get("kv_agent_label", ""),
        "kv_request_rid": window.get("kv_request_rid", ""),
        "kv_node_id": context_node_id(context),
        "kv_layer_id": context.get("layer_id", ""),
        "kv_io_backend": context.get("io_backend", ""),
        "host_index_count": host.get("index_count") or host.get("numel") or "",
        "host_indices": index_values_text(host),
        "device_index_count": device.get("index_count") or device.get("numel") or "",
        "device_indices": index_values_text(device),
    }


def summarize_window(window: dict[str, Any], cuda_events: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [event for event in cuda_events if overlaps(event, window)]
    copy_like = [event for event in matched if event["kind"] == "memcpy"]
    transfers = [event for event in copy_like if is_transfer_event(event)]
    kernels = [event for event in matched if event["kind"] == "kernel"]
    directions = Counter(event["direction"] for event in transfers)
    first_copy_offset = ""
    last_copy_end_offset = ""
    if transfers:
        first_copy_offset = round((min(int(event["start_ns"]) for event in transfers) - int(window["start_ns"])) / 1_000_000, 3)
        last_copy_end_offset = round((max(int(event["end_ns"]) for event in transfers) - int(window["start_ns"])) / 1_000_000, 3)
    top_transfers = Counter(event["name"] for event in transfers).most_common(5)
    return {
        "window_type": window.get("window_type", ""),
        "event": window.get("event", ""),
        "call_id": window.get("call_id", ""),
        "session_id": window.get("session_id", ""),
        "phase": window.get("phase", ""),
        **kv_columns(window),
        "duration_ms": round((int(window["end_ns"]) - int(window["start_ns"])) / 1_000_000, 3),
        "kernel_events": len(kernels),
        "kernel_total_ms": round(sum(float(event["duration_ms"]) for event in kernels), 3),
        "copy_like_events": len(copy_like),
        "transfer_events": len(transfers),
        "transfer_total_ms": round(sum(float(event["duration_ms"]) for event in transfers), 3),
        "transfer_total_mb": round(sum(int(event["bytes"]) for event in transfers) / 1_000_000, 3),
        "h2d_events": directions["h2d"],
        "d2h_events": directions["d2h"],
        "d2d_events": directions["d2d"],
        "memset_events": directions["memset"],
        "unknown_transfer_events": directions["unknown"],
        "first_transfer_offset_ms": first_copy_offset,
        "last_transfer_end_offset_ms": last_copy_end_offset,
        "top_transfer_names": "; ".join(f"{name} x{count}" for name, count in top_transfers),
    }


def md_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows.", ""]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.append("")
    return lines


def write_report(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        "# Torch Profile / Agent Trace Correlation",
        "",
        "This report aligns the SGLang JSONL trace with torch-profiler Chrome traces from SGLang worker processes.",
        "Clock alignment is approximate: the profiler status timestamp is used as the wall-clock anchor for the first Chrome-trace event.",
        "In this report, transfer events mean profiler GPU memcpy/memset rows such as Memcpy HtoD, Memcpy DtoH, and Memcpy DtoD.",
        "Copy-like events include higher-level PyTorch copy operations and are shown only as extra context.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Windows With CUDA Activity",
        "",
    ]
    lines.extend(md_table(rows))
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_timeline_rows(cuda_events: list[dict[str, Any]], windows: list[dict[str, Any]], trace_start_ns: int) -> list[dict[str, Any]]:
    copies = [event for event in cuda_events if event["kind"] == "memcpy" and is_transfer_event(event)]
    rows: list[dict[str, Any]] = []
    for event in sorted(copies, key=lambda item: int(item["start_ns"])):
        matched_windows = [window for window in windows if overlaps(event, window)]
        matched_agent_windows = [window for window in matched_windows if window.get("window_type") == "agent_request"]
        best_window = min(
            matched_windows,
            key=lambda window: int(window["end_ns"]) - int(window["start_ns"]),
            default={},
        )
        best_agent_window = max(
            matched_agent_windows,
            key=lambda window: overlap_ns(event, window),
            default={},
        )
        rows.append(
            {
                "start_ms_from_trace_start": round((int(event["start_ns"]) - trace_start_ns) / 1_000_000, 6),
                "end_ms_from_trace_start": round((int(event["end_ns"]) - trace_start_ns) / 1_000_000, 6),
                "duration_ms": round(float(event["duration_ms"]), 6),
                "direction": event["direction"],
                "bytes": event["bytes"],
                "name": event["name"],
                "overlap_window_type": best_window.get("window_type", ""),
                "overlap_event": best_window.get("event", ""),
                "overlap_call_id": best_window.get("call_id", ""),
                "overlap_session_id": best_window.get("session_id", ""),
                "overlap_phase": best_window.get("phase", ""),
                "enclosing_agent_session_id": best_agent_window.get("session_id", ""),
                "enclosing_agent_phase": best_agent_window.get("phase", ""),
                "enclosing_agent_overlap_count": len(matched_agent_windows),
                **{f"overlap_{key}": value for key, value in kv_columns(best_window).items()},
                "offset_ms_from_window_start": (
                    round((int(event["start_ns"]) - int(best_window["start_ns"])) / 1_000_000, 6)
                    if best_window
                    else ""
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate SGLang JSONL windows with torch-profiler CUDA events.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-copy-csv")
    parser.add_argument("--include-zero", action="store_true")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    profile_dir = Path(args.profile_dir)
    trace_events = read_jsonl(trace_path)
    starts = profiler_start_by_pid(profile_dir)
    cuda_events: list[dict[str, Any]] = []
    profile_count = 0
    for profile in sorted(profile_dir.glob("torch_cuda_profile_*.json")):
        pid = pid_from_profile_name(profile)
        if pid is None or pid not in starts:
            continue
        profile_count += 1
        cuda_events.extend(profile_cuda_events(profile, starts[pid]))

    windows = request_windows(trace_events)
    windows.extend(paired_method_windows(trace_events, KV_EVENTS))
    annotate_kv_windows_with_agent(windows)
    node_map, index_map = build_request_maps(windows)
    annotate_kv_windows_with_request_maps(windows, node_map, index_map)
    rows = [summarize_window(window, cuda_events) for window in windows]
    if not args.include_zero:
        rows = [row for row in rows if int(row["kernel_events"]) > 0 or int(row["transfer_events"]) > 0]
    rows.sort(key=lambda row: (row["window_type"], row["event"], row["session_id"], row["phase"]))

    metadata = {
        "trace": str(trace_path),
        "profile_dir": str(profile_dir),
        "profile_count": profile_count,
        "cuda_event_count": len(cuda_events),
        "window_count": len(windows),
        "reported_window_count": len(rows),
        "total_kernel_events": sum(1 for event in cuda_events if event["kind"] == "kernel"),
        "total_copy_like_events": sum(1 for event in cuda_events if event["kind"] == "memcpy"),
        "total_transfer_events": sum(1 for event in cuda_events if event["kind"] == "memcpy" and is_transfer_event(event)),
        "avg_cuda_event_duration_ms": round(mean([float(event["duration_ms"]) for event in cuda_events]), 6)
        if cuda_events
        else 0.0,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_md, rows, metadata)
    if args.out_copy_csv:
        trace_start_ns = min((int(event.get("ts_ns", 0)) for event in trace_events if event.get("ts_ns")), default=0)
        write_csv(Path(args.out_copy_csv), copy_timeline_rows(cuda_events, windows, trace_start_ns))
    print(f"Wrote correlation JSON to {out_json}")
    print(f"Wrote correlation Markdown to {out_md}")
    if args.out_copy_csv:
        print(f"Wrote copy timeline CSV to {args.out_copy_csv}")
    print(f"Profiles: {profile_count}")
    print(f"CUDA events: {len(cuda_events)}")
    print(f"Reported windows: {len(rows)}")


if __name__ == "__main__":
    main()
