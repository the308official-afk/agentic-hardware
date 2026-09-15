from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


@dataclass(frozen=True)
class RealScenarioSpec:
    """Concrete mapping from a scenario claim to the existing instrumented runner."""

    scenario_id: str
    claim: str
    report_label_prefix: str
    harnesses: str
    pressure_levels: str
    modes: str
    trace_profile: str
    report_builder_mode: str = "lightweight"
    pressure_knobs: dict[str, str] = field(default_factory=dict)
    update_latest: bool = False


REAL_SCENARIOS: dict[str, RealScenarioSpec] = {
    "claim1_deadline_scheduling": RealScenarioSpec(
        scenario_id="claim1_deadline_scheduling",
        claim="Better deadline scheduling",
        report_label_prefix="claim1_deadline_scheduling",
        harnesses="hatcher",
        pressure_levels="p1_mild p3_high",
        modes="no_prefetch pre_harness_priority_hints controller_scheduler_priority",
        trace_profile="controller_decision",
        pressure_knobs={
            "P3_HIGH_KNOBS": (
                "tool_wait_ms=1000 target_prompt_tokens=2048 "
                "filler_sessions=32 filler_prompt_tokens=2048 session_count=1 concurrency=8"
            ),
        },
    ),
    "claim2_proactive_kv_hostmem": RealScenarioSpec(
        scenario_id="claim2_proactive_kv_hostmem",
        claim="Proactive KV-cache management",
        report_label_prefix="claim2_proactive_kv_hostmem",
        harnesses="hatcher",
        pressure_levels="p3_high",
        modes="no_prefetch controller_speculative_preload controller_targeted_kv_prefetch",
        trace_profile="cache_debug",
        pressure_knobs={
            "P3_HIGH_KNOBS": (
                "tool_wait_ms=1000 target_prompt_tokens=4096 "
                "filler_sessions=32 filler_prompt_tokens=2048 session_count=1 concurrency=8"
            ),
        },
    ),
}


def get_real_scenario_spec(scenario_id: str) -> RealScenarioSpec:
    try:
        return REAL_SCENARIOS[scenario_id]
    except KeyError as exc:
        known = ", ".join(sorted(REAL_SCENARIOS))
        raise ValueError(f"Unknown real scenario {scenario_id!r}. Known scenarios: {known}") from exc


def build_real_scenario_command(
    spec: RealScenarioSpec,
    *,
    repo_root: str | Path,
    model: str = DEFAULT_MODEL,
    report_label: str | None = None,
) -> list[str]:
    """Return a shell command that uses the existing SGLang-instrumented runner.

    The scenario layer deliberately delegates to `scripts/run_harness_deadline_pressure.sh`
    so it shares the same SGLang install, trace hooks, controller metadata, and report
    builders as the rest of `sglang_direct_kv`.
    """

    root = Path(repo_root)
    label = report_label or f"{spec.report_label_prefix}_$(date +%Y%m%d_%H%M%S)"
    env_parts = {
        "HARNESSES": spec.harnesses,
        "PRESSURE_LEVELS": spec.pressure_levels,
        "MODES": spec.modes,
        "REPORT_BUILDER_MODE": spec.report_builder_mode,
        "TRACE_PROFILE": spec.trace_profile,
        "REPORT_LABEL": label,
        "UPDATE_LATEST": "1" if spec.update_latest else "0",
        **spec.pressure_knobs,
    }
    exports = " ".join(f"{key}={_shell_quote(value)}" for key, value in env_parts.items())
    command = (
        f"cd {_shell_quote(str(root))} && "
        "source .venv/bin/activate && "
        f"{exports} bash scripts/run_harness_deadline_pressure.sh {_shell_quote(model)}"
    )
    return ["bash", "-lc", command]


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
