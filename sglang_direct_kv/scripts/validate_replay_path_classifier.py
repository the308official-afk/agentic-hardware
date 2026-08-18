#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from replay_path_classifier import classify_replay_path, hardware_counterfactual


def assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def main() -> int:
    cases = [
        {
            "name": "replay host-to-device load",
            "row": {
                "replay_input_tokens": 1536,
                "replay_cached_prefix_tokens": 1536,
                "replay_host_load_tokens": 1536,
                "replay_kv_h2d_events": 2,
                "replay_kv_h2d_duration_ms": 12,
                "resume_ttft_ms": 180,
            },
            "expected": {
                "final_path": "host_to_device_kv_load",
                "confidence": "high",
            },
        },
        {
            "name": "partial recompute",
            "row": {
                "replay_input_tokens": 2400,
                "replay_cached_prefix_tokens": 900,
                "replay_new_prefill_tokens_est": 1500,
                "resume_ttft_ms": 1400,
            },
            "expected": {
                "final_path": "partial_prefix_miss_recompute",
                "bottleneck_label": "recompute dominated",
            },
        },
        {
            "name": "scheduler wait after cache hit",
            "row": {
                "replay_input_tokens": 2400,
                "replay_cached_prefix_tokens": 2400,
                "replay_first_cache_event_delay_ms": 250,
                "resume_ttft_ms": 1200,
            },
            "expected": {
                "final_path": "gpu_resident_or_logical_cache_hit_waited",
                "bottleneck_label": "scheduler dominated",
            },
        },
        {
            "name": "fast cache hit",
            "row": {
                "replay_input_tokens": 2000,
                "replay_cached_prefix_tokens": 1980,
                "resume_ttft_ms": 90,
            },
            "expected": {
                "final_path": "gpu_resident_or_logical_cache_hit",
                "confidence": "medium",
            },
        },
        {
            "name": "late software prefetch with short copy opportunity",
            "row": {
                "tool_gap_ms": 100,
                "prefetch_margin_ms": -800,
                "prefetch_duration_ms": 900,
                "replay_kv_h2d_duration_ms": 12,
            },
            "counterfactual": {
                "counterfactual_verdict": "hardware opportunity",
            },
        },
    ]

    results: list[dict[str, Any]] = []
    for case in cases:
        path = classify_replay_path(case["row"])
        counterfactual = hardware_counterfactual(case["row"])
        for key, expected in case.get("expected", {}).items():
            assert_equal(f"{case['name']} {key}", path.get(key), expected)
        for key, expected in case.get("counterfactual", {}).items():
            assert_equal(f"{case['name']} {key}", counterfactual.get(key), expected)
        results.append({"case": case["name"], "path": path, "counterfactual": counterfactual})

    out = Path("artifacts/results/replay_path_classifier_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated {len(cases)} replay-path classifier cases.")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
