#!/usr/bin/env python3
"""Scan real agent trajectories for request-local shorthand opportunities."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_prompt_codec import CodecConfig, PromptEncoder  # noqa: E402
from agentic_prompt_codec.config import load_encoder  # noqa: E402


class ApproxTokenCounter:
    """Portable fallback for scans where the model tokenizer is unavailable."""

    identity = "approx-char4-counter"

    def count_text(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def count_request(self, payload: Any, api_kind: str) -> int:
        return self.count_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


TEXT_KEYS = (
    "prompt",
    "content",
    "input",
    "output",
    "response",
    "response_text",
    "observation",
    "thought",
    "action",
    "system_prompt",
)

TEXT_SUFFIXES = {".csv", ".log", ".md", ".txt"}


def iter_json_values(value: Any, path: tuple[str | int, ...] = ()) -> Iterable[tuple[tuple[str | int, ...], str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (key,)
            if key in TEXT_KEYS and isinstance(child, str):
                yield child_path, child
            else:
                yield from iter_json_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_json_values(child, path + (index,))


def iter_records(path: Path) -> Iterable[tuple[int, Any]]:
    if path.suffix == ".jsonl":
        for index, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                yield index, json.loads(line)
            except json.JSONDecodeError:
                continue
        return
    try:
        value = json.loads(path.read_text(errors="ignore"))
    except json.JSONDecodeError:
        return
    if isinstance(value, list):
        for index, item in enumerate(value, start=1):
            yield index, item
    else:
        yield 1, value


def discover_files(root: Path) -> list[Path]:
    names = {
        "stage_lifecycle_trace.json",
        "stage_lifecycle_trace.jsonl",
        "trajectory.json",
        "trajectory.jsonl",
        "trace.json",
        "trace.jsonl",
    }
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and (path.name in names or path.suffix.lower() in TEXT_SUFFIXES)
    )


def iter_text_records(path: Path, max_chars: int) -> Iterable[tuple[int, str]]:
    text = path.read_text(errors="ignore")
    if len(text) > max_chars:
        text = text[:max_chars]
    yield 1, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Trajectory root to scan.")
    parser.add_argument("--config", default=str(ROOT / "configs/prompt_codecs/agent_trace_relations_v1.json"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--out-csv", default=str(ROOT / "artifacts/results/prompt_shorthand_trajectory_scan.csv"))
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--max-text-chars", type=int, default=200_000)
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--approx-token-counter", action="store_true",
                        help="Use a portable char/4 counter instead of a Hugging Face tokenizer.")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    config = Path(args.config)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.approx_token_counter:
        raw = json.loads(config.read_text())
        codec_raw = dict(raw.get("codec", {}))
        if isinstance(codec_raw.get("enabled_rules"), list):
            codec_raw["enabled_rules"] = tuple(codec_raw["enabled_rules"])
        encoder = PromptEncoder(CodecConfig(**codec_raw), ApproxTokenCounter())
    else:
        encoder = load_encoder(config, args.model, allow_tokenizer_download=args.allow_tokenizer_download)

    files = discover_files(root)
    rows: list[dict[str, Any]] = []
    scanned = 0
    for file_path in files:
        if file_path.suffix.lower() in TEXT_SUFFIXES:
            text_records = ((index, (("text",), text)) for index, text in iter_text_records(file_path, args.max_text_chars))
        else:
            text_records = (
                (record_index, (text_path, text))
                for record_index, record in iter_records(file_path)
                for text_path, text in iter_json_values(record)
            )
        for record_index, (text_path, text) in text_records:
            if not text.strip():
                continue
            scanned += 1
            result = encoder.encode(text, "text")
            rows.append({
                "file": str(file_path.relative_to(root)),
                "record_index": record_index,
                "text_path": ".".join(str(part) for part in text_path),
                "status": result.status,
                "reason": result.reason,
                "original_tokens": result.original_tokens,
                "encoded_tokens": result.encoded_tokens,
                "candidate_tokens": result.candidate_tokens,
                "candidate_saved_tokens": (
                    result.original_tokens - result.candidate_tokens
                    if result.original_tokens is not None and result.candidate_tokens is not None
                    else None
                ),
                "saved_tokens": (
                    result.original_tokens - result.encoded_tokens
                    if result.original_tokens is not None and result.encoded_tokens is not None
                    else None
                ),
                "changed_segments": result.changed_segments,
                "rule_counts": json.dumps(result.rule_counts, sort_keys=True),
            })
            if scanned >= args.max_records:
                break
        if scanned >= args.max_records:
            break

    fields = [
        "file",
        "record_index",
        "text_path",
        "status",
        "reason",
        "original_tokens",
        "encoded_tokens",
        "candidate_tokens",
        "candidate_saved_tokens",
        "saved_tokens",
        "changed_segments",
        "rule_counts",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    applied = sum(1 for row in rows if row["status"] == "applied")
    print(f"scanned_files={len(files)} scanned_text_fields={scanned} applied={applied} out={out_csv}")


if __name__ == "__main__":
    main()
