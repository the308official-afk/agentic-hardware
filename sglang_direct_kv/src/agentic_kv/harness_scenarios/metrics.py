from __future__ import annotations

from .models import ScenarioRun


def improvement_label(run: ScenarioRun) -> str:
    key = run.primary_metric
    improvement = run.improvement
    if not key or improvement is None:
        return "n/a"
    unit = "ms" if key.endswith("_ms") else "tokens" if key.endswith("_tokens") else "units"
    return f"{_fmt(improvement)} {unit} lower"


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"
