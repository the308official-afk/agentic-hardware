from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import Scenario, ScenarioResult, ScenarioRun
from .policies.baseline import baseline_decision
from .policies.harness_aware import harness_aware_decision


def run_scenarios(scenarios: Iterable[Scenario]) -> list[ScenarioRun]:
    return [run_scenario(scenario) for scenario in scenarios]


def run_scenario(scenario: Scenario) -> ScenarioRun:
    baseline = baseline_decision(scenario)
    aware = harness_aware_decision(scenario)
    return ScenarioRun(scenario=scenario, baseline=baseline, harness_aware=aware)


def result(
    scenario: Scenario,
    *,
    mode: str,
    policy: str,
    decision: str,
    metrics: dict[str, Any],
    benefit_summary: str,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        brief_claim=scenario.brief_claim,
        mode=mode,
        policy=policy,
        decision=decision,
        metrics=metrics,
        benefit_summary=benefit_summary,
        harness_signals_used=tuple(sorted(scenario.harness_signals)),
        backend_state_used=tuple(sorted(scenario.backend_state)),
    )
