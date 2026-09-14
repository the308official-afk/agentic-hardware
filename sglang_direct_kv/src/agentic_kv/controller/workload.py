from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolWaitSpec:
    step_index: int
    wait_ms: int
    wait_class: str


@dataclass(frozen=True)
class WorkloadShape:
    kind: str
    phase: str
    prompt_tokens: int
    max_tokens: int
    description: str


TOOL_WAIT_PROFILE_DISTRIBUTIONS = {
    "agentic_mixed": (
        ("quick", 70.0, 200),
        ("moderate", 25.0, 2_000),
        ("slow", 5.0, 20_000),
    ),
    "realistic_agentic_mix": (
        ("quick_file", 45.0, 250),
        ("repo_search", 25.0, 800),
        ("small_command", 15.0, 2_000),
        ("test_or_build", 10.0, 7_000),
        ("slow_external", 5.0, 18_000),
    ),
}

REALISTIC_AGENTIC_PROFILE = "realistic_agentic_mix"
SYNTHETIC_PRESSURE_PROFILE = "synthetic_pressure"
SUPPORTED_AGENTIC_WORKLOAD_PROFILES = (SYNTHETIC_PRESSURE_PROFILE, REALISTIC_AGENTIC_PROFILE)
REALISTIC_INITIAL_SHAPES = (
    ("planning_routing", 15.0, 768, 32, "planning", "plan the repository task and choose the next tool"),
    ("file_search_inspect", 25.0, 1536, 32, "file_inspect", "inspect search results and choose likely files"),
    ("patch_reasoning", 20.0, 4096, 128, "patch_reasoning", "reason over code context before editing"),
    ("test_build_reasoning", 15.0, 4096, 128, "test_build", "interpret test or build output"),
    ("review_final", 15.0, 1536, 256, "review", "review changes and produce a final response"),
    ("slow_external_resume", 10.0, 8192, 128, "slow_external", "resume after a slower external or long-running command"),
)
REALISTIC_REPLAY_SHAPES_BY_WAIT = {
    "quick_file": ("file_search_inspect", 1536, 32, "file_inspect", "resume after a quick file read"),
    "repo_search": ("file_search_inspect", 2048, 32, "file_inspect", "resume after repository search"),
    "small_command": ("patch_reasoning", 4096, 96, "patch_reasoning", "resume after a short command"),
    "test_or_build": ("test_build_reasoning", 4096, 128, "test_build", "resume after a test or build command"),
    "slow_external": ("slow_external_resume", 8192, 128, "slow_external", "resume after a slow external or long command"),
    "quick": ("file_search_inspect", 1536, 32, "file_inspect", "resume after a quick tool"),
    "moderate": ("patch_reasoning", 4096, 96, "patch_reasoning", "resume after a moderate tool"),
    "slow": ("slow_external_resume", 8192, 128, "slow_external", "resume after a slow tool"),
    "pressure_fixed": ("patch_reasoning", 4096, 64, "patch_reasoning", "resume after a fixed synthetic wait"),
}


def parse_tool_wait_profile_spec(spec: str) -> tuple[tuple[str, float, int], ...]:
    out: list[tuple[str, float, int]] = []
    for item in spec.split(","):
        raw = item.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) == 1:
            wait_class = "fixed"
            weight_raw = "1"
            wait_raw = parts[0]
        elif len(parts) == 2:
            wait_class = f"bucket_{len(out) + 1}"
            weight_raw, wait_raw = parts
        elif len(parts) == 3:
            wait_class, weight_raw, wait_raw = parts
        else:
            raise ValueError(
                "TOOL_WAIT_PROFILE_SPEC entries must be wait_ms, class:weight:wait_ms, or weight:wait_ms"
            )
        weight = float(weight_raw)
        wait_ms = int(float(wait_raw))
        if weight <= 0 or wait_ms < 0:
            raise ValueError("tool wait profile weights must be positive and waits must be non-negative")
        out.append((wait_class.strip() or f"bucket_{len(out) + 1}", weight, wait_ms))
    if not out:
        raise ValueError("TOOL_WAIT_PROFILE_SPEC did not contain any buckets")
    return tuple(out)


def tool_wait_distribution(profile: str, base_wait_ms: int, custom_spec: str = "") -> tuple[tuple[str, float, int], ...]:
    normalized = profile.strip().lower() if profile else "fixed"
    if custom_spec.strip():
        return parse_tool_wait_profile_spec(custom_spec)
    if normalized in {"fixed", "pressure_fixed"}:
        return (("pressure_fixed", 1.0, int(base_wait_ms)),)
    try:
        return TOOL_WAIT_PROFILE_DISTRIBUTIONS[normalized]
    except KeyError as exc:
        supported = ", ".join(["fixed", *sorted(TOOL_WAIT_PROFILE_DISTRIBUTIONS)])
        raise ValueError(f"unknown tool wait profile {profile!r}; supported profiles: {supported}") from exc


def sample_tool_wait_specs(
    *,
    profile: str,
    base_wait_ms: int,
    custom_spec: str,
    steps: int,
    seed: int,
    stream_key: str,
) -> list[ToolWaitSpec]:
    if steps < 1:
        raise ValueError("TASK_REPLAY_STEPS must be at least 1")
    distribution = tool_wait_distribution(profile, base_wait_ms, custom_spec)
    if len(distribution) == 1:
        wait_class, _, wait_ms = distribution[0]
        return [ToolWaitSpec(step_index=idx, wait_ms=wait_ms, wait_class=wait_class) for idx in range(1, steps + 1)]

    rng = random.Random(f"{seed}:{stream_key}:{profile}:{custom_spec}")
    total_weight = sum(weight for _, weight, _ in distribution)
    specs: list[ToolWaitSpec] = []
    for idx in range(1, steps + 1):
        pick = rng.random() * total_weight
        cumulative = 0.0
        selected = distribution[-1]
        for bucket in distribution:
            cumulative += bucket[1]
            if pick <= cumulative:
                selected = bucket
                break
        specs.append(ToolWaitSpec(step_index=idx, wait_ms=int(selected[2]), wait_class=str(selected[0])))
    return specs


def normalize_agentic_workload_profile(profile: str) -> str:
    normalized = (profile or SYNTHETIC_PRESSURE_PROFILE).strip().lower()
    aliases = {
        "default": SYNTHETIC_PRESSURE_PROFILE,
        "synthetic": SYNTHETIC_PRESSURE_PROFILE,
        "pressure": SYNTHETIC_PRESSURE_PROFILE,
        "realistic": REALISTIC_AGENTIC_PROFILE,
        "agentic_realistic": REALISTIC_AGENTIC_PROFILE,
        "coding_agent": REALISTIC_AGENTIC_PROFILE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_AGENTIC_WORKLOAD_PROFILES:
        supported = ", ".join(SUPPORTED_AGENTIC_WORKLOAD_PROFILES)
        raise ValueError(f"unknown agentic workload profile {profile!r}; supported profiles: {supported}")
    return normalized


def stable_rng(seed: int, *parts: object) -> random.Random:
    return random.Random(":".join([str(seed), *(str(part) for part in parts)]))


def weighted_choice(
    choices: tuple[tuple[str, float, int, int, str, str], ...],
    *,
    seed: int,
    stream_key: str,
) -> tuple[str, float, int, int, str, str]:
    rng = stable_rng(seed, stream_key)
    total = sum(choice[1] for choice in choices)
    pick = rng.random() * total
    cumulative = 0.0
    for choice in choices:
        cumulative += choice[1]
        if pick <= cumulative:
            return choice
    return choices[-1]


def realistic_prompt_token_ceiling() -> int:
    return max(512, int(os.environ.get("REALISTIC_AGENTIC_MAX_PROMPT_TOKENS", "4096") or "4096"))


def jitter_tokens(base_tokens: int, *, seed: int, stream_key: str, floor: int = 256, ceiling: int | None = None) -> int:
    rng = stable_rng(seed, stream_key, "tokens")
    multiplier = rng.choice((0.75, 1.0, 1.25, 1.5))
    ceiling = realistic_prompt_token_ceiling() if ceiling is None else ceiling
    return max(floor, min(ceiling, int(round(base_tokens * multiplier))))


def realistic_initial_shape(*, seed: int, stream_key: str) -> WorkloadShape:
    kind, _, base_tokens, max_tokens, phase, description = weighted_choice(
        REALISTIC_INITIAL_SHAPES,
        seed=seed,
        stream_key=stream_key,
    )
    return WorkloadShape(
        kind=kind,
        phase=phase,
        prompt_tokens=jitter_tokens(base_tokens, seed=seed, stream_key=stream_key),
        max_tokens=max_tokens,
        description=description,
    )


def realistic_replay_shape(*, wait_class: str, seed: int, stream_key: str) -> WorkloadShape:
    kind, base_tokens, max_tokens, phase, description = REALISTIC_REPLAY_SHAPES_BY_WAIT.get(
        wait_class,
        REALISTIC_REPLAY_SHAPES_BY_WAIT["pressure_fixed"],
    )
    return WorkloadShape(
        kind=kind,
        phase=phase,
        prompt_tokens=jitter_tokens(base_tokens, seed=seed, stream_key=stream_key),
        max_tokens=max_tokens,
        description=description,
    )


def workload_meta(shape: WorkloadShape, profile: str) -> dict[str, object]:
    return {
        "agentic_workload_profile": profile,
        "workload_request_kind": shape.kind,
        "workload_phase_family": shape.phase,
        "workload_prompt_tokens_target": shape.prompt_tokens,
        "workload_max_tokens": shape.max_tokens,
        "workload_description": shape.description,
    }


def estimate_request_runtime_ms(prompt_tokens: int, max_tokens: int, *, floor_ms: int = 50) -> int:
    """Portable oracle estimate for one request's backend occupancy."""

    prefill_tokens_per_ms = float(os.environ.get("CONTROLLER_ORACLE_PREFILL_TOKENS_PER_MS", "2.5") or "2.5")
    decode_tokens_per_ms = float(os.environ.get("CONTROLLER_ORACLE_DECODE_TOKENS_PER_MS", "0.25") or "0.25")
    fixed_overhead_ms = float(os.environ.get("CONTROLLER_ORACLE_FIXED_OVERHEAD_MS", "25") or "25")
    prefill_ms = float(prompt_tokens) / max(prefill_tokens_per_ms, 0.001)
    decode_ms = float(max_tokens) / max(decode_tokens_per_ms, 0.001)
    return max(floor_ms, int(round(fixed_overhead_ms + prefill_ms + decode_ms)))
