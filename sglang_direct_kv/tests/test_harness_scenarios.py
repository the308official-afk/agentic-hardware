from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_kv.harness_scenarios import (
    build_real_scenario_command,
    get_real_scenario_spec,
    load_scenario_manifest,
    run_scenarios,
)
from agentic_kv.harness_scenarios.adapters.controller_signal import to_harness_controller_meta
from agentic_kv.harness_scenarios.report import write_outputs


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "harness_scenarios" / "minimal.json"


class HarnessScenarioTests(unittest.TestCase):
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

    def test_real_scenario_runner_delegates_to_existing_instrumented_script(self) -> None:
        spec = get_real_scenario_spec("claim2_proactive_kv_hostmem")
        cmd = build_real_scenario_command(
            spec,
            repo_root=ROOT,
            report_label="claim2_test",
        )

        shell = cmd[-1]
        self.assertIn("scripts/run_harness_deadline_pressure.sh", shell)
        self.assertIn("controller_targeted_kv_prefetch", shell)
        self.assertIn("TRACE_PROFILE='cache_debug'", shell)
        self.assertIn("REPORT_LABEL='claim2_test'", shell)


if __name__ == "__main__":
    unittest.main()
