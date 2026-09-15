from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Scenario


def load_scenario_manifest(path: str | Path) -> list[Scenario]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("Scenario manifest must contain a list at 'scenarios'")
    return [_scenario_from_dict(row) for row in scenarios]


def _scenario_from_dict(row: dict[str, Any]) -> Scenario:
    required = [
        "scenario_id",
        "brief_claim",
        "scenario_type",
        "description",
        "harness_signals",
        "backend_state",
        "baseline_policy",
        "harness_aware_policy",
        "expected_benefit",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Scenario {row.get('scenario_id', '<unknown>')} missing fields: {missing}")
    return Scenario(
        scenario_id=str(row["scenario_id"]),
        brief_claim=str(row["brief_claim"]),
        scenario_type=str(row["scenario_type"]),
        description=str(row["description"]),
        harness_signals=dict(row["harness_signals"]),
        backend_state=dict(row["backend_state"]),
        baseline_policy=str(row["baseline_policy"]),
        harness_aware_policy=str(row["harness_aware_policy"]),
        expected_benefit=str(row["expected_benefit"]),
        chart=str(row.get("chart", "benefit_bar")),
    )
