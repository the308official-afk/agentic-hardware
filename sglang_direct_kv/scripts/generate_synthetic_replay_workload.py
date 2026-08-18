#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    return max(1, int(round(len(text.split()) * 1.35)))


def load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return None


def token_len(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        return estimate_tokens(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_shared_prefix(session_id: str, target_tokens: int, tokenizer: Any) -> str:
    lines = [
        f"SYNTHETIC_SHARED_PREFIX_BEGIN session={session_id}",
        "This prefix is intentionally identical in the first request and the replay request.",
        "The experiment uses it to test whether SGLang keeps KV in GPU, reloads KV from host, or recomputes it.",
    ]
    idx = 0
    while token_len(tokenizer, "\n".join(lines)) < target_tokens:
        digest = hashlib.sha256(f"{session_id}:{idx}".encode("utf-8")).hexdigest()[:12]
        lines.append(
            " ".join(
                [
                    "shared_repo_context",
                    digest,
                    f"file_{idx % 257}.py",
                    f"symbol_{idx % 509}",
                    f"test_case_{idx % 313}",
                    "failure_trace",
                    "patch_history",
                    "tool_output",
                    "dependency_note",
                    "review_comment",
                ]
            )
        )
        idx += 1
    lines.append("SYNTHETIC_SHARED_PREFIX_END")
    return "\n".join(lines)


def make_suffix(session_id: str, target_tokens: int, tokenizer: Any) -> str:
    lines = [
        f"SYNTHETIC_REPLAY_SUFFIX_BEGIN session={session_id}",
        "Tool result: the test command returned a new failure after the first model turn.",
        "Continue the debugging task using the already shared repository context.",
    ]
    idx = 0
    while token_len(tokenizer, "\n".join(lines)) < target_tokens:
        lines.append(
            " ".join(
                [
                    "new_tool_result",
                    f"line_{idx}",
                    f"assertion_{idx % 89}",
                    "resume_reasoning",
                    "next_patch_step",
                ]
            )
        )
        idx += 1
    lines.append("SYNTHETIC_REPLAY_SUFFIX_END")
    return "\n".join(lines)


def write_rows(out_jsonl: Path, out_csv: Path, rows: list[dict[str, Any]]) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if not rows:
        return
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic first/replay prompt pairs with a large shared prefix.")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--replay-suffix-tokens", type=int, default=256)
    parser.add_argument("--disable-tokenizer", action="store_true")
    args = parser.parse_args()

    tokenizer = None if args.disable_tokenizer else load_tokenizer(args.model)
    rows: list[dict[str, Any]] = []
    for idx in range(args.pairs):
        session_id = f"synthetic_large_prefix_{idx:03d}"
        shared_prefix = make_shared_prefix(session_id, args.prompt_tokens, tokenizer)
        first_prompt = "\n\n".join(
            [
                "You are a coding agent. Read the repository context and decide the next tool call.",
                shared_prefix,
                "First turn instruction: identify the most likely source file and prepare to run tests.",
            ]
        )
        replay_suffix = make_suffix(session_id, args.replay_suffix_tokens, tokenizer)
        replay_prompt = "\n\n".join([first_prompt, replay_suffix])
        rows.append(
            {
                "session_id": session_id,
                "source": "synthetic_large_shared_prefix",
                "task_index": idx,
                "repo": "synthetic_repo",
                "instance_id": f"synthetic_instance_{idx:03d}",
                "from_phase": "first_turn",
                "to_phase": "replay_turn",
                "tool_names": "synthetic_tool_wait",
                "priority": "high",
                "prompt_tokens": token_len(tokenizer, first_prompt),
                "replay_prompt_tokens": token_len(tokenizer, replay_prompt),
                "shared_prefix_target_tokens": args.prompt_tokens,
                "replay_suffix_target_tokens": args.replay_suffix_tokens,
                "prompt": first_prompt,
                "replay_prompt": replay_prompt,
            }
        )

    write_rows(args.out_jsonl, args.out_csv, rows)
    print(f"Wrote {len(rows)} synthetic replay pairs to {args.out_jsonl}")
    for row in rows[:5]:
        print(
            json.dumps(
                {
                    "session_id": row["session_id"],
                    "prompt_tokens": row["prompt_tokens"],
                    "replay_prompt_tokens": row["replay_prompt_tokens"],
                    "shared_prefix_target_tokens": row["shared_prefix_target_tokens"],
                    "replay_suffix_target_tokens": row["replay_suffix_target_tokens"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
