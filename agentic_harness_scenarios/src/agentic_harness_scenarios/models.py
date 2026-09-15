from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MetricValue = int | float | str | bool | None


@dataclass(frozen=True)
class Scenario:
    """Declarative minimal scenario for one harness-aware benefit."""

    scenario_id: str
    brief_claim: str
    scenario_type: str
    description: str
    harness_signals: dict[str, Any]
    backend_state: dict[str, Any]
    baseline_policy: str
    harness_aware_policy: str
    expected_benefit: str
    chart: str = "benefit_bar"


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    brief_claim: str
    mode: str
    policy: str
    decision: str
    metrics: dict[str, MetricValue]
    benefit_summary: str
    harness_signals_used: tuple[str, ...] = field(default_factory=tuple)
    backend_state_used: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScenarioRun:
    scenario: Scenario
    baseline: ScenarioResult
    harness_aware: ScenarioResult

    @property
    def primary_metric(self) -> str:
        for key in (
            "deadline_lateness_ms",
            "tool_complete_to_first_token_ms",
            "recompute_tokens",
            "scheduler_wait_ms",
            "workflow_completion_ms",
            "wasted_work_tokens",
            "urgent_p95_ttft_ms",
            "cost_proxy",
        ):
            if key in self.baseline.metrics and key in self.harness_aware.metrics:
                return key
        return next(iter(self.baseline.metrics), "")

    @property
    def improvement(self) -> float | None:
        key = self.primary_metric
        if not key:
            return None
        baseline_value = self.baseline.metrics.get(key)
        aware_value = self.harness_aware.metrics.get(key)
        if not isinstance(baseline_value, (int, float)) or not isinstance(aware_value, (int, float)):
            return None
        return float(baseline_value) - float(aware_value)
