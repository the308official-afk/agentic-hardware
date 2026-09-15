from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_harness_scenarios import load_scenario_manifest, run_scenarios
from agentic_harness_scenarios.adapters.controller_signal import to_harness_controller_meta
from agentic_harness_scenarios.report import write_outputs


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "harness_aware_scenarios" / "minimal.json"


class HarnessAwareScenarioTests(unittest.TestCase):
    def test_manifest_loads_minimal_scenarios(self) -> None:
        scenarios = load_scenario_manifest(MANIFEST)

        self.assertEqual(len(scenarios), 11)
        self.assertEqual(scenarios[0].scenario_id, "deadline_scheduling_minimal")
        self.assertIn("deadline_after_ready_ms", scenarios[0].harness_signals)
        self.assertIn("background_prefill_ms", scenarios[0].backend_state)

    def test_every_minimal_scenario_improves_primary_metric(self) -> None:
        runs = run_scenarios(load_scenario_manifest(MANIFEST))

        for run in runs:
            with self.subTest(run.scenario.scenario_id):
                self.assertIsNotNone(run.improvement)
                self.assertGreaterEqual(run.improvement or 0, 0)

    def test_deadline_scenario_separates_harness_signal_from_backend_state(self) -> None:
        run = run_scenarios(load_scenario_manifest(MANIFEST))[0]

        self.assertEqual(run.primary_metric, "deadline_lateness_ms")
        self.assertGreater(run.baseline.metrics["deadline_lateness_ms"], 0)
        self.assertLess(run.harness_aware.metrics["deadline_lateness_ms"], 0)
        self.assertIn("priority", run.harness_aware.harness_signals_used)
        self.assertIn("background_prefill_ms", run.harness_aware.backend_state_used)

    def test_report_outputs_are_written(self) -> None:
        runs = run_scenarios(load_scenario_manifest(MANIFEST))
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(runs, tmp)
            out = Path(tmp)
            self.assertTrue((out / "scenario_results.csv").exists())
            self.assertTrue((out / "scenario_results.json").exists())
            html = (out / "scenario_report.html").read_text(encoding="utf-8")
            self.assertIn("Executive Scorecard", html)
            self.assertIn("Better deadline scheduling", html)

    def test_controller_adapter_maps_mvp_signal_names(self) -> None:
        meta = to_harness_controller_meta(
            {
                "session_id": "s1",
                "prefix_id": "p1",
                "phase": "tool_wait",
                "expected_tool_return_ms": 120000,
                "deadline_after_ready_ms": 100,
                "priority": 100,
                "user_waiting": True,
            }
        )

        self.assertEqual(meta["session_id"], "s1")
        self.assertEqual(meta["prefix_id"], "p1")
        self.assertEqual(meta["tool_wait_ms"], 120000)
        self.assertEqual(meta["deadline_after_tool_ms"], 100)
        self.assertEqual(meta["controller_sglang_priority"], 100)


if __name__ == "__main__":
    unittest.main()
