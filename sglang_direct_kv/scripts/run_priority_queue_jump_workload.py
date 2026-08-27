#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx


def now_ns() -> int:
    return time.perf_counter_ns()


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def prompt_text(label: str, approx_tokens: int) -> str:
    # Deliberately deterministic and prefix-divergent so low-priority requests
    # are real independent work, not trivial prefix-cache clones.
    words = max(64, approx_tokens)
    seed = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
    chunks = [
        f"You are request {label}. Unique seed {seed}.",
        "Write a careful, concrete technical explanation.",
        "Do not call tools. Do not stop early.",
    ]
    for idx in range(words // 12):
        chunks.append(f"{label} unique_segment_{idx}_{seed} analyze scheduler pressure and cache behavior.")
    return "\n".join(chunks)


def priority_payload(
    *,
    model: str,
    request_id: str,
    role: str,
    priority: int,
    max_tokens: int,
    prompt_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    label = "high" if priority > 0 else "low" if priority < 0 else "normal"
    nvext = {
        "agent_hints": {
            "priority": priority,
            "priority_label": label,
            "osl": max_tokens,
            "expected_output_tokens": max_tokens,
            "request_id": request_id,
            "phase": role,
            "expected_action": "priority_queue_sanity",
        },
        "request_context": {
            "experiment": "priority_queue_jump_sanity",
            "request_role": role,
            "request_id": request_id,
        },
    }
    agentic_kv = {
        "session_id": request_id,
        "phase": role,
        "label": request_id,
        "mode": "priority_queue_sanity",
        "prompt_hash": hashlib.sha1(request_id.encode("utf-8")).hexdigest()[:16],
        "priority": label,
        "request_id": request_id,
        "case_id": "priority_queue_jump",
        "gap_id": "0",
    }
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text(request_id, prompt_tokens)}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        # This is what SGLang's OpenAI path can consume directly.
        "priority": priority,
        # This is the Dynamo-faithful shape we want to prove survives the path.
        "nvext": nvext,
        "custom_params": {
            "agentic_kv": agentic_kv,
            "request_context": nvext["request_context"],
            "nvext": nvext,
            "dynamo_priority_bridge": {
                "dynamo_agent_priority": label,
                "dynamo_hint_priority": priority,
                "dynamo_hint_priority_label": label,
                "sglang_priority": priority,
                "priority_translation": "nvext.agent_hints.priority -> sglang priority",
            },
        },
    }


async def post_chat(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, Any],
    request_id: str,
    priority: int,
    metrics_path: Path,
    trace_path: Path,
) -> dict[str, Any]:
    submit_ns = now_ns()
    base = {
        "request_id": request_id,
        "priority": priority,
        "role": "high" if priority > 0 else "low" if priority < 0 else "normal",
    }
    write_jsonl(trace_path, {"event": "priority_sanity.client_submit", "ts_ns": submit_ns, **base})
    first_token_ns: int | None = None
    start_ns = now_ns()
    write_jsonl(trace_path, {"event": "priority_sanity.http_start", "ts_ns": start_ns, **base})
    status_code = 0
    error = ""
    output_chars = 0
    try:
        async with client.stream("POST", url, json=payload) as response:
            status_code = response.status_code
            response.raise_for_status()
            async for chunk in response.aiter_text():
                if chunk and first_token_ns is None:
                    first_token_ns = now_ns()
                    write_jsonl(trace_path, {"event": "priority_sanity.first_chunk", "ts_ns": first_token_ns, **base})
                output_chars += len(chunk or "")
    except Exception as exc:  # pragma: no cover - exercised on EC2 only
        error = repr(exc)
    end_ns = now_ns()
    row = {
        **base,
        "client_submit_ns": submit_ns,
        "http_start_ns": start_ns,
        "first_chunk_ns": first_token_ns,
        "end_ns": end_ns,
        "status_code": status_code,
        "error": error,
        "output_chars": output_chars,
        "latency_ms": (end_ns - start_ns) / 1e6,
        "ttfc_ms": ((first_token_ns - start_ns) / 1e6) if first_token_ns else "",
    }
    write_jsonl(metrics_path, row)
    write_jsonl(trace_path, {"event": "priority_sanity.http_end", "ts_ns": end_ns, **base, "status_code": status_code, "error": error})
    return row


async def main_async(args: argparse.Namespace) -> None:
    metrics_path = Path(args.out)
    trace_path = Path(args.trace_out)
    url = args.base_url.rstrip("/") + "/chat/completions"
    if metrics_path.exists():
        metrics_path.unlink()
    write_jsonl(trace_path, {"event": "priority_sanity.workload_start", "ts_ns": now_ns(), "args": vars(args)})

    limits = httpx.Limits(max_connections=args.request_concurrency + 8, max_keepalive_connections=args.request_concurrency + 8)
    timeout = httpx.Timeout(args.timeout_s)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks: list[asyncio.Task[dict[str, Any]]] = []

        for idx in range(args.low_before_count):
            rid = f"pq_low_before_{idx:04d}"
            payload = priority_payload(
                model=args.model,
                request_id=rid,
                role="low_before",
                priority=args.low_priority,
                max_tokens=args.low_max_tokens,
                prompt_tokens=args.low_prompt_tokens,
                stream=not args.no_stream,
            )
            tasks.append(asyncio.create_task(post_chat(client, url=url, payload=payload, request_id=rid, priority=args.low_priority, metrics_path=metrics_path, trace_path=trace_path)))
            if args.low_submit_stagger_ms > 0:
                await asyncio.sleep(args.low_submit_stagger_ms / 1000)

        await asyncio.sleep(args.high_submit_delay_ms / 1000)
        high_payload = priority_payload(
            model=args.model,
            request_id=args.high_request_id,
            role="high_target",
            priority=args.high_priority,
            max_tokens=args.high_max_tokens,
            prompt_tokens=args.high_prompt_tokens,
            stream=not args.no_stream,
        )
        tasks.append(asyncio.create_task(post_chat(client, url=url, payload=high_payload, request_id=args.high_request_id, priority=args.high_priority, metrics_path=metrics_path, trace_path=trace_path)))

        for idx in range(args.low_after_count):
            rid = f"pq_low_after_{idx:04d}"
            payload = priority_payload(
                model=args.model,
                request_id=rid,
                role="low_after",
                priority=args.low_priority,
                max_tokens=args.low_max_tokens,
                prompt_tokens=args.low_prompt_tokens,
                stream=not args.no_stream,
            )
            tasks.append(asyncio.create_task(post_chat(client, url=url, payload=payload, request_id=rid, priority=args.low_priority, metrics_path=metrics_path, trace_path=trace_path)))
            if args.low_submit_stagger_ms > 0:
                await asyncio.sleep(args.low_submit_stagger_ms / 1000)

        rows = await asyncio.gather(*tasks)

    high = next((row for row in rows if row["request_id"] == args.high_request_id), None)
    write_jsonl(
        trace_path,
        {
            "event": "priority_sanity.workload_end",
            "ts_ns": now_ns(),
            "request_count": len(rows),
            "high_request": high or {},
        },
    )
    print(f"Wrote metrics to {metrics_path}")
    if high:
        print(f"High request status={high['status_code']} latency_ms={high['latency_ms']:.1f} ttfc_ms={high.get('ttfc_ms')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused SGLang priority queue jump sanity workload.")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace-out", required=True)
    parser.add_argument("--high-request-id", default="pq_high_0000")
    parser.add_argument("--low-before-count", type=int, default=24)
    parser.add_argument("--low-after-count", type=int, default=8)
    parser.add_argument("--request-concurrency", type=int, default=64)
    parser.add_argument("--high-submit-delay-ms", type=int, default=100)
    parser.add_argument("--low-submit-stagger-ms", type=int, default=5)
    parser.add_argument("--low-prompt-tokens", type=int, default=2048)
    parser.add_argument("--high-prompt-tokens", type=int, default=512)
    parser.add_argument("--low-max-tokens", type=int, default=96)
    parser.add_argument("--high-max-tokens", type=int, default=24)
    parser.add_argument("--high-priority", type=int, default=100)
    parser.add_argument("--low-priority", type=int, default=-100)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--no-stream", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
