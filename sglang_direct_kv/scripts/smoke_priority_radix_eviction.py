#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def run_help_check(enable_shim: bool, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    if enable_shim:
        env["AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE"] = "1"
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{env.get('PYTHONPATH', '')}"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "sglang.launch_server", "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
        )
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    help_text = proc.stdout or ""
    radix_window = ""
    marker = "--radix-eviction-policy"
    if marker in help_text:
        index = help_text.find(marker)
        radix_window = help_text[index : index + 180]
    return {
        "ok": proc.returncode == 0 and "{lru,lfu,slru,priority}" in help_text,
        "exit_code": proc.returncode,
        "radix_help_window": radix_window,
    }


def run_simulated_cache_check() -> dict[str, Any]:
    import torch

    from agentic_kv.sglang_compat import enable_priority_radix_eviction_choice

    shim_enabled = enable_priority_radix_eviction_choice()

    from sglang.srt import server_args
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.evict_policy import PriorityStrategy
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

    class MockAllocator:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.freed: list[list[int]] = []

        def free(self, value: Any) -> None:
            if hasattr(value, "detach"):
                self.freed.append([int(item) for item in value.detach().cpu().tolist()])
            else:
                self.freed.append([int(item) for item in value])

    allocator = MockAllocator()
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy="priority",
        )
    )
    cache.insert(
        InsertParams(
            key=RadixKey(token_ids=[1, 2, 3], extra_key=None),
            value=torch.tensor([101, 102, 103]),
            priority=100,
        )
    )
    cache.insert(
        InsertParams(
            key=RadixKey(token_ids=[9, 8, 7], extra_key=None),
            value=torch.tensor([201, 202, 203]),
            priority=-100,
        )
    )
    before = sorted(
        (
            {
                "tokens": [int(item) for item in node.key.token_ids],
                "priority": int(node.priority),
                "value": [int(item) for item in node.value.tolist()],
            }
            for node in cache.evictable_leaves
        ),
        key=lambda row: row["priority"],
    )
    result = cache.evict(EvictParams(num_tokens=3))
    after = sorted(
        (
            {
                "tokens": [int(item) for item in node.key.token_ids],
                "priority": int(node.priority),
                "value": [int(item) for item in node.value.tolist()],
            }
            for node in cache.evictable_leaves
        ),
        key=lambda row: row["priority"],
    )
    low_priority_evicted = allocator.freed == [[201, 202, 203]]
    high_priority_remained = after == [{"tokens": [1, 2, 3], "priority": 100, "value": [101, 102, 103]}]
    return {
        "ok": shim_enabled
        and "priority" in getattr(server_args, "RADIX_EVICTION_POLICY_CHOICES", [])
        and isinstance(cache.eviction_strategy, PriorityStrategy)
        and low_priority_evicted
        and high_priority_remained,
        "shim_enabled": shim_enabled,
        "radix_eviction_policy_choices": list(getattr(server_args, "RADIX_EVICTION_POLICY_CHOICES", [])),
        "selected_strategy": type(cache.eviction_strategy).__name__,
        "before": before,
        "evicted_tokens": int(result.num_tokens_evicted),
        "freed_values": allocator.freed,
        "after": after,
        "low_priority_evicted": low_priority_evicted,
        "high_priority_remained": high_priority_remained,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test SGLang priority radix eviction without launching a model server."
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--skip-help-check", action="store_true")
    parser.add_argument("--help-timeout", type=int, default=45)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "ok": False,
        "checks": {},
    }
    try:
        payload["checks"]["simulated_cache"] = run_simulated_cache_check()
        if not args.skip_help_check:
            payload["checks"]["launch_help"] = run_help_check(enable_shim=True, timeout=args.help_timeout)
        payload["ok"] = all(bool(row.get("ok")) for row in payload["checks"].values())
    except Exception as exc:
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["ok"] = False

    if args.out_json is not None:
        atomic_write_json(args.out_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
