#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("select name from sqlite_master where type='table' order by name").fetchall()
    return [str(row[0]) for row in rows]


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'pragma table_info("{table}")').fetchall()]


def first_existing(cols: list[str], names: list[str]) -> str | None:
    lowered = {col.lower(): col for col in cols}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def enum_map(conn: sqlite3.Connection, tables: list[str], pattern: str) -> dict[int, str]:
    enum_table = next((table for table in tables if pattern.lower() in table.lower()), None)
    if enum_table is None:
        return {}
    cols = columns(conn, enum_table)
    id_col = first_existing(cols, ["id", "value", "number"])
    label_col = first_existing(cols, ["label", "name", "value", "description"])
    if id_col is None or label_col is None or id_col == label_col:
        return {}
    out: dict[int, str] = {}
    for row in conn.execute(f'select "{id_col}", "{label_col}" from "{enum_table}"'):
        try:
            out[int(row[0])] = str(row[1])
        except Exception:
            pass
    return out


def summarize_memcpy(conn: sqlite3.Connection, tables: list[str]) -> list[dict[str, Any]]:
    memcpy_tables = [table for table in tables if "MEMCPY" in table.upper()]
    kind_names = enum_map(conn, tables, "MEMCPY")
    rows_out: list[dict[str, Any]] = []
    for table in memcpy_tables:
        if table.upper().startswith("ENUM_"):
            continue
        cols = columns(conn, table)
        start_col = first_existing(cols, ["start", "startNs"])
        end_col = first_existing(cols, ["end", "endNs"])
        bytes_col = first_existing(cols, ["bytes", "size"])
        kind_col = first_existing(cols, ["copyKind", "copykind", "kind"])
        stream_col = first_existing(cols, ["streamId", "stream"])
        if start_col is None or end_col is None:
            continue
        select_cols = [start_col, end_col]
        if bytes_col:
            select_cols.append(bytes_col)
        if kind_col:
            select_cols.append(kind_col)
        if stream_col:
            select_cols.append(stream_col)
        quoted_cols = ", ".join([f'"{col}"' for col in select_cols])
        sql = f'select {quoted_cols} from "{table}"'
        groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "total_duration_ms": 0.0,
                "total_bytes": 0,
                "durations_ms": [],
                "streams": Counter(),
            }
        )
        for row in conn.execute(sql):
            item = dict(zip(select_cols, row, strict=False))
            duration_ms = max(0, int(item[end_col]) - int(item[start_col])) / 1_000_000
            raw_kind = item.get(kind_col) if kind_col else None
            kind = kind_names.get(int(raw_kind), str(raw_kind)) if raw_kind is not None else "unknown"
            group = groups[kind]
            group["count"] += 1
            group["total_duration_ms"] += duration_ms
            group["durations_ms"].append(duration_ms)
            if bytes_col and item.get(bytes_col) is not None:
                try:
                    group["total_bytes"] += int(item[bytes_col])
                except Exception:
                    pass
            if stream_col and item.get(stream_col) is not None:
                group["streams"][str(item[stream_col])] += 1
        for kind, group in groups.items():
            durations = group["durations_ms"]
            rows_out.append(
                {
                    "table": table,
                    "copy_kind": kind,
                    "count": group["count"],
                    "total_duration_ms": round(group["total_duration_ms"], 3),
                    "avg_duration_ms": round(mean(durations), 3) if durations else 0.0,
                    "max_duration_ms": round(max(durations), 3) if durations else 0.0,
                    "total_mb": round(group["total_bytes"] / 1_000_000, 3),
                    "streams": len(group["streams"]),
                }
            )
    return sorted(rows_out, key=lambda row: (row["table"], row["copy_kind"]))


def summarize_kernels(conn: sqlite3.Connection, tables: list[str]) -> list[dict[str, Any]]:
    kernel_tables = [
        table
        for table in tables
        if "KERNEL" in table.upper() and not table.upper().startswith("ENUM_")
    ]
    rows_out: list[dict[str, Any]] = []
    for table in kernel_tables:
        cols = columns(conn, table)
        start_col = first_existing(cols, ["start", "startNs"])
        end_col = first_existing(cols, ["end", "endNs"])
        if start_col is None or end_col is None:
            continue
        durations: list[float] = []
        for start, end in conn.execute(f'select "{start_col}", "{end_col}" from "{table}"'):
            durations.append(max(0, int(end) - int(start)) / 1_000_000)
        if durations:
            rows_out.append(
                {
                    "table": table,
                    "count": len(durations),
                    "total_duration_ms": round(sum(durations), 3),
                    "avg_duration_ms": round(mean(durations), 3),
                    "max_duration_ms": round(max(durations), 3),
                }
            )
    return sorted(rows_out, key=lambda row: row["table"])


def summarize_nvtx(conn: sqlite3.Connection, tables: list[str]) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for table in tables:
        if "NVTX" not in table.upper() or table.upper().startswith("ENUM_"):
            continue
        cols = columns(conn, table)
        count = conn.execute(f'select count(*) from "{table}"').fetchone()[0]
        rows_out.append({"table": table, "count": int(count), "columns": ", ".join(cols[:12])})
    return rows_out


def string_map(conn: sqlite3.Connection, tables: list[str]) -> dict[int, str]:
    if "StringIds" not in tables:
        return {}
    cols = columns(conn, "StringIds")
    id_col = first_existing(cols, ["id"])
    value_col = first_existing(cols, ["value", "string"])
    if id_col is None or value_col is None:
        return {}
    out: dict[int, str] = {}
    for row in conn.execute(f'select "{id_col}", "{value_col}" from "StringIds"'):
        try:
            out[int(row[0])] = str(row[1])
        except Exception:
            pass
    return out


def summarize_cuda_runtime(conn: sqlite3.Connection, tables: list[str]) -> list[dict[str, Any]]:
    table = next((name for name in tables if name.upper() == "CUPTI_ACTIVITY_KIND_RUNTIME"), None)
    if table is None:
        return []
    names = string_map(conn, tables)
    cols = columns(conn, table)
    start_col = first_existing(cols, ["start"])
    end_col = first_existing(cols, ["end"])
    name_col = first_existing(cols, ["nameId"])
    if start_col is None or end_col is None or name_col is None:
        return []
    groups: dict[str, list[float]] = defaultdict(list)
    for start, end, name_id in conn.execute(f'select "{start_col}", "{end_col}", "{name_col}" from "{table}"'):
        name = names.get(int(name_id), str(name_id))
        groups[name].append(max(0, int(end) - int(start)) / 1_000_000)
    rows = [
        {
            "runtime_api": name,
            "count": len(durations),
            "total_duration_ms": round(sum(durations), 3),
            "avg_duration_ms": round(mean(durations), 6) if durations else 0.0,
            "max_duration_ms": round(max(durations), 6) if durations else 0.0,
        }
        for name, durations in groups.items()
    ]
    return sorted(rows, key=lambda row: (-int(row["count"]), str(row["runtime_api"])))[:40]


def summarize_diagnostics(conn: sqlite3.Connection, tables: list[str]) -> list[dict[str, Any]]:
    if "DIAGNOSTIC_EVENT" not in tables:
        return []
    cols = columns(conn, "DIAGNOSTIC_EVENT")
    text_col = first_existing(cols, ["text"])
    severity_col = first_existing(cols, ["severity"])
    if text_col is None:
        return []
    counter: Counter[str] = Counter()
    severity_by_text: dict[str, Any] = {}
    select_cols = [text_col] + ([severity_col] if severity_col else [])
    quoted_cols = ", ".join([f'"{col}"' for col in select_cols])
    for row in conn.execute(f'select {quoted_cols} from "DIAGNOSTIC_EVENT"'):
        text = str(row[0]).strip()
        if not text:
            continue
        if not any(word in text.lower() for word in ("cuda", "cupti", "gpu", "profil")):
            continue
        counter[text] += 1
        if severity_col:
            severity_by_text[text] = row[1]
    return [
        {
            "count": count,
            "severity": severity_by_text.get(text, ""),
            "diagnostic": text.replace("\n", " "),
        }
        for text, count in counter.most_common(20)
    ]


def summarize_agent_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(event.get("event", "unknown")) for event in events)
    hint_durations = [
        float(event.get("duration_ms", 0.0))
        for event in events
        if event.get("event") == "hicache.load.end" and isinstance(event.get("duration_ms"), (int, float))
    ]
    hint_total_events = [
        event for event in events if event.get("event") in {"agent.hint_prefetch_start", "agent.hint_prefetch_end"}
    ]
    return {
        "total_events": len(events),
        "event_counts": dict(counts.most_common(20)),
        "hicache_load_count": len(hint_durations),
        "hicache_load_total_ms": round(sum(hint_durations), 3),
        "hicache_load_avg_ms": round(mean(hint_durations), 3) if hint_durations else 0.0,
        "hint_boundary_events": len(hint_total_events),
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


def write_report(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Milestone 10 DMA Timeline Summary",
        "",
        "This report summarizes observable CUDA memory-copy activity from an Nsight Systems SQLite export.",
        "It does not expose the GPU DMA engine's private internal queue directly.",
        "Instead, it shows when CUDA copy work and GPU kernels actually appeared in the profiled timeline.",
        "",
        "## Agent Trace Summary",
        "",
        "```json",
        json.dumps(data["agent_trace_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## CUDA Memcpy Summary",
        "",
    ]
    lines.extend(md_table(data["memcpy_summary"]))
    lines.extend(["## CUDA Kernel Summary", ""])
    lines.extend(md_table(data["kernel_summary"]))
    lines.extend(["## CUDA Runtime API Summary", ""])
    lines.extend(md_table(data["runtime_summary"]))
    lines.extend(["## Nsight CUDA Diagnostics", ""])
    lines.extend(md_table(data["diagnostics_summary"]))
    lines.extend(["## NVTX Tables", ""])
    lines.extend(md_table(data["nvtx_summary"]))
    lines.extend(
        [
            "## How To Read This",
            "",
            "Useful evidence looks like this:",
            "",
            "```text",
            "agent hint starts much earlier than replay",
            "hicache.load duration is small",
            "CUDA copy activity appears late or overlaps heavy kernel activity",
            "replay arrives before the hint path finishes",
            "```",
            "",
            "That pattern supports the hardware argument:",
            "",
            "```text",
            "The bytes are not necessarily expensive by themselves.",
            "The problem is getting the right KV movement scheduled, prioritized, and protected before the agent resumes.",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Nsight Systems CUDA memcpy activity for Milestone 10.")
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite)
    trace_path = Path(args.trace)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(trace_path)
    with sqlite3.connect(sqlite_path) as conn:
        tables = table_names(conn)
        data = {
            "sqlite": str(sqlite_path),
            "trace": str(trace_path),
            "tables": tables,
            "agent_trace_summary": summarize_agent_trace(events),
            "memcpy_summary": summarize_memcpy(conn, tables),
            "kernel_summary": summarize_kernels(conn, tables),
            "runtime_summary": summarize_cuda_runtime(conn, tables),
            "diagnostics_summary": summarize_diagnostics(conn, tables),
            "nvtx_summary": summarize_nvtx(conn, tables),
        }

    out_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_md, data)
    print(f"Wrote Nsight/DMA summary JSON to {out_json}")
    print(f"Wrote Nsight/DMA summary Markdown to {out_md}")
    print(f"Memcpy groups: {len(data['memcpy_summary'])}")
    print(f"Kernel groups: {len(data['kernel_summary'])}")


if __name__ == "__main__":
    main()
