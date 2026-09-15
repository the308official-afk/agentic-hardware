"""Harness-aware scenario definitions and launch helpers for this testbed."""

from .manifest import load_scenario_manifest
from .models import Scenario, ScenarioResult, ScenarioRun
from .real_runner import RealScenarioSpec, build_real_scenario_command, get_real_scenario_spec
from .simulator import run_scenarios

__all__ = [
    "RealScenarioSpec",
    "Scenario",
    "ScenarioResult",
    "ScenarioRun",
    "build_real_scenario_command",
    "get_real_scenario_spec",
    "load_scenario_manifest",
    "run_scenarios",
]
