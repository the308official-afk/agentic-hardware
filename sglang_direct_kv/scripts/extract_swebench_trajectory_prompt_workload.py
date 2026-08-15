#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


PROMPT_PATH_COLUMNS = (
    "prompt_text_path",
    "prompt_path",
    "request_text_path",
    "request_path",
    "text_path",
)
PROMPT_TEXT_COLUMNS = ("prompt", "request_prompt", "prompt_text", "messages_text", "input")
GROUP_COLUMNS = (
    "run_id",
    "trajectory_id",
    "instance_id",
    "task_instance_id",
    "task_id",
    "repo",
)
SORT_COLUMNS = (
    "sequence_index",
    "step_index",
    "turn_index",
    "prompt_index",
    "request_index",
    "phase_index",
    "timestamp",
)
PHASE_ORDER = {
    "planning": 0,
    "execution": 1,
    "execution_loop_inspect": 2,
    "execution_loop_edit": 3,
    "execution_loop_validate": 4,
    "patch_generation": 5,
    "review": 6,
    "baseline_execution": 7,
}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.replace(",", " ").split() if item.strip()]
    if not values:
        raise ValueError("expected at least one tool wait value")
    return values


def as_int(value: Any, default: int) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(str(value)))
    except Exception:
        return default


def estimate_tokens(text: str) -> int:
    # Rough tokenizer-independent estimate. This is only metadata; SGLang tokenizes the real prompt.
    return max(1, int(len(text) / 4))


def first_present(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if value not in ("", None):
            return str(value)
    return ""


def resolve_prompt_path(raw_path: str, catalog_csv: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        path,
        catalog_csv.parent / path,
        catalog_csv.parent.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def prompt_for_row(row: dict[str, Any], catalog_csv: Path) -> tuple[str, str]:
    raw_path = first_present(row, PROMPT_PATH_COLUMNS)
    if raw_path:
        path = resolve_prompt_path(raw_path, catalog_csv)
        if not path.exists():
            raise FileNotFoundError(f"prompt_text_path does not exist: {raw_path} (resolved as {path})")
        return path.read_text(encoding="utf-8", errors="replace"), str(path)
    text = first_present(row, PROMPT_TEXT_COLUMNS)
    if text:
        return text, ""
    raise ValueError("catalog row has no prompt_text_path-like column and no inline prompt text column")


def group_key(row: dict[str, Any]) -> tuple[str, ...]:
    values = [str(row.get(column, "")) for column in GROUP_COLUMNS if row.get(column) not in ("", None)]
    return tuple(values) if values else ("__all__",)


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for column in SORT_COLUMNS:
        value = row.get(column)
        if value in ("", None):
            values.append((2, 10**12))
        else:
            try:
                values.append((0, float(str(value))))
            except Exception:
                values.append((1, str(value)))
    phase = str(row.get("phase") or row.get("stage") or "").lower()
    values.append((0, PHASE_ORDER.get(phase, 999)))
    values.append((0, as_int(row.get("_catalog_row_index"), 10**12)))
    return tuple(values)


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def make_workload(
    *,
    catalog_csv: Path,
    max_sessions: int,
    tool_wait_values: list[int],
    seed: int,
    arrival_gap_ms: int,
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    rows = read_csv_rows(catalog_csv)
    for idx, row in enumerate(rows):
        row["_catalog_row_index"] = idx
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    rng = random.Random(seed)
    workload: list[dict[str, Any]] = []
    for _, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered = sorted(group_rows, key=sort_key)
        for current, replay in zip(ordered, ordered[1:]):
            prompt, prompt_path = prompt_for_row(current, catalog_csv)
            replay_prompt, replay_path = prompt_for_row(replay, catalog_csv)
            prompt = truncate_text(prompt, max_prompt_chars)
            replay_prompt = truncate_text(replay_prompt, max_prompt_chars)
            idx = len(workload)
            session_id = (
                current.get("session_id")
                or current.get("request_id")
                or f"swepro_{idx:03d}_{current.get('run_id') or current.get('instance_id') or 'trace'}"
            )
            from_phase = current.get("phase") or current.get("stage") or current.get("request_phase") or "unknown"
            to_phase = replay.get("phase") or replay.get("stage") or replay.get("request_phase") or "unknown"
            workload.append(
                {
                    "session_id": str(session_id),
                    "arrival_ms": idx * arrival_gap_ms,
                    "tool_wait_ms": rng.choice(tool_wait_values),
                    "priority": str(current.get("priority") or replay.get("priority") or "5"),
                    "prompt": prompt,
                    "replay_prompt": replay_prompt,
                    "prompt_tokens": estimate_tokens(prompt),
                    "replay_prompt_tokens": estimate_tokens(replay_prompt),
                    "repo": current.get("repo") or replay.get("repo") or "",
                    "run_id": current.get("run_id") or replay.get("run_id") or "",
                    "instance_id": current.get("instance_id") or replay.get("instance_id") or "",
                    "from_phase": str(from_phase),
                    "to_phase": str(to_phase),
                    "source": "swebench_trajectory_prompt_catalog",
                    "catalog_csv": str(catalog_csv),
                    "prompt_text_path": prompt_path,
                    "replay_prompt_text_path": replay_path,
                    "catalog_row_index": current.get("_catalog_row_index", ""),
                    "replay_catalog_row_index": replay.get("_catalog_row_index", ""),
                }
            )
            if max_sessions and len(workload) >= max_sessions:
                return workload
    return workload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "session_id",
        "arrival_ms",
        "tool_wait_ms",
        "priority",
        "repo",
        "run_id",
        "instance_id",
        "from_phase",
        "to_phase",
        "prompt_tokens",
        "replay_prompt_tokens",
        "prompt_text_path",
        "replay_prompt_text_path",
        "catalog_row_index",
        "replay_catalog_row_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SWE-bench trajectory prompt catalog rows into replay workload JSONL.")
    parser.add_argument("--catalog-csv", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--max-sessions", type=int, default=24)
    parser.add_argument("--tool-wait-list-ms", default="250 500 900 1600 3000")
    parser.add_argument("--arrival-gap-ms", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    args = parser.parse_args()

    catalog_csv = args.catalog_csv.expanduser().resolve()
    if not catalog_csv.exists():
        raise SystemExit(f"catalog CSV not found: {catalog_csv}")
    rows = make_workload(
        catalog_csv=catalog_csv,
        max_sessions=args.max_sessions,
        tool_wait_values=parse_int_list(args.tool_wait_list_ms),
        seed=args.seed,
        arrival_gap_ms=args.arrival_gap_ms,
        max_prompt_chars=args.max_prompt_chars,
    )
    if not rows:
        raise SystemExit(f"no replay sessions could be extracted from {catalog_csv}")
    write_jsonl(args.out_jsonl, rows)
    write_csv(args.out_csv, rows)
    waits = sorted({row["tool_wait_ms"] for row in rows})
    print(f"Wrote {len(rows)} SWE-bench trajectory replay sessions to {args.out_jsonl}")
    print(f"CSV summary: {args.out_csv}")
    print(f"Tool waits used: {waits}")


if __name__ == "__main__":
    main()
