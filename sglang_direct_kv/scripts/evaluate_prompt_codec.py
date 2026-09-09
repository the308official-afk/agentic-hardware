#!/usr/bin/env python3
"""Paired request-local token and optional model-quality evaluation.

Without --base-url this performs token accounting only, never model inference.
With --base-url it submits both arms (counterbalanced) and scores exact answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

from agentic_prompt_codec.config import load_encoder
from agentic_prompt_codec.proxy import text_delta
from agentic_prompt_codec.reporting import percentile, write_csv
from agentic_prompt_codec.tokenizers import HuggingFaceTokenCounter


def complete(base_url, payload):
    import httpx
    started = time.perf_counter()
    first = None
    output = []
    usage = {}
    with httpx.Client(timeout=120) as client:
        with client.stream("POST", base_url.rstrip("/") + "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                item = json.loads(line[5:])
                text = text_delta(item)
                if text:
                    if first is None:
                        first = time.perf_counter()
                    output.append(text)
                if item.get("usage"):
                    usage = item["usage"]
    return "".join(output), ((first - started) * 1000 if first else None), (time.perf_counter() - started) * 1000, usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--base-url")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-condition", choices=("natural", "cold", "warm"), default="natural")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    encoder = load_encoder(args.config, args.model, allow_tokenizer_download=args.allow_tokenizer_download)
    counter = encoder.counter or HuggingFaceTokenCounter(args.model, local_files_only=not args.allow_tokenizer_download)
    workload_bytes = args.workload.read_bytes()
    tasks = [json.loads(line) for line in workload_bytes.decode().splitlines() if line.strip()]
    if not tasks or len({task["id"] for task in tasks}) != len(tasks):
        parser.error("workload requires nonempty tasks with unique ids")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "config.json").write_text(args.config.read_text())
    manifest = {"model": args.model, "tokenizer_id": counter.identity, "seed": args.seed, "repeats": args.repeats,
                "config_hash": encoder.fingerprint, "codec": encoder.config.codec,
                "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
                "inference_enabled": bool(args.base_url), "quality_method": "exact_casefold_answer",
                "cache_policy": args.cache_condition,
                "cache_policy_evidence": "cold/warm request isolated SGLang cache_salt namespaces; confirm backend support in telemetry"}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rng = random.Random(args.seed)
    rows = []
    for repeat in range(args.repeats):
        for task in tasks:
            original = {"model": args.model, "messages": [{"role": "user", "content": task["replay_prompt"]}],
                        "temperature": 0, "max_tokens": task.get("max_tokens", 128), "stream": True,
                        "stream_options": {"include_usage": True}}
            arms = ["identity", "candidate"]
            rng.shuffle(arms)
            for arm in arms:
                request = dict(original)
                if args.cache_condition != "natural":
                    request["cache_salt"] = "prompt-codec-eval-" + uuid.uuid4().hex
                started = time.perf_counter()
                result = encoder.encode(request) if arm == "candidate" else None
                payload = result.payload if result else request
                preparation_ms = (time.perf_counter() - started) * 1000
                row = {"task_id": task["id"], "repeat": repeat, "arm": arm, "codec": encoder.config.codec if result else "identity",
                       "input_tokens": counter.count_request(payload, "openai_chat"),
                       "preparation_ms": preparation_ms, "quality": "not_measured", "error": ""}
                if result:
                    row.update(result.evidence())
                if args.base_url:
                    try:
                        if args.cache_condition == "warm":
                            _, _, warmup_ms, _ = complete(args.base_url, {**payload, "max_tokens": 1})
                            row.update(warmup_requests=1, warmup_elapsed_ms=warmup_ms)
                        output, ttft, latency, usage = complete(args.base_url, payload)
                        row.update(output=output, ttft_ms=ttft, total_latency_ms=latency + preparation_ms,
                                   end_to_end_ttft_ms=ttft + preparation_ms if ttft is not None else None,
                                   quality=int(output.strip().casefold() == str(task["expected_answer"]).strip().casefold()),
                                   observed_prompt_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
                        row["observed_cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                        if not output:
                            row["error"] = "empty_backend_content"
                    except Exception as exc:
                        row.update(error=type(exc).__name__, quality=0)
                rows.append(row)
                write_csv(args.out_dir / "evaluation.csv", rows)
    summary = {}
    for arm in ("identity", "candidate"):
        selected = [r for r in rows if r["arm"] == arm]
        scored = [r for r in selected if isinstance(r["quality"], int)]
        summary[arm] = {"requests": len(selected), "input_tokens": sum(r["input_tokens"] for r in selected),
                        "errors": sum(bool(r["error"]) for r in selected),
                        "accuracy": sum(r["quality"] for r in scored) / len(scored) if scored else None}
        times = [r["end_to_end_ttft_ms"] for r in selected if isinstance(r.get("end_to_end_ttft_ms"), (int, float))]
        for q in (0.5, 0.95, 0.99):
            summary[arm][f"p{int(q * 100)}_end_to_end_ttft_ms"] = percentile(times, q)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
