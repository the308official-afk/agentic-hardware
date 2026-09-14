from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agentic_kv.hint_benchmark import (
    build_nat_payload_observations,
    build_dry_run,
    build_fixture_observations,
    load_knob_profiles,
    load_benchmark_inputs,
    select_knob_profile,
    select_scenarios,
    validate_hint_evidence,
    write_dry_run_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "hint_benchmark" / "nat_hints.json"
SCENARIOS = ROOT / "configs" / "hint_benchmark" / "nat_scenarios.json"
KNOBS = ROOT / "configs" / "hint_benchmark" / "nat_knobs.json"


class HintBenchmarkRunnerTests(unittest.TestCase):
    def test_loads_nat_manifest_and_scenarios(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        self.assertEqual(manifest["harness"]["id"], "nemo_agent_toolkit")
        self.assertEqual(len(manifest["hints"]), 13)
        self.assertEqual(len(scenarios["scenarios"]), 16)

    def test_selects_smoke_scenarios(self):
        _, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "smoke")
        self.assertEqual(
            [scenario["id"] for scenario in selected],
            [
                "nat_no_hints_baseline",
                "nat_priority_high",
                "nat_priority_low",
                "nat_prefix_reuse_id",
                "nat_cache_control_ttl",
            ],
        )

    def test_selects_knob_profile_scenarios(self):
        _, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        knobs = load_knob_profiles(KNOBS)
        profile = select_knob_profile(knobs, "passthrough_only")
        selected = select_scenarios(scenarios, profile["scenario_selectors"])
        self.assertEqual([scenario["id"] for scenario in selected], ["nat_cache_namespace", "nat_provider_qos"])
        self.assertIn("nvext.cache_salt", profile["expected_signal_visibility"])

    def test_dry_run_records_are_not_real_observations(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "nat_priority_high")
        result = build_dry_run(manifest, selected, run_id="unit_test", created_at=1.0)
        self.assertEqual(result["run"]["execution_mode"], "dry_run")
        self.assertEqual(result["run"]["scenario_count"], 1)
        self.assertEqual(result["run"]["expectation_count"], 1)
        self.assertEqual(result["scenario_records"][0]["result"], "not_evaluated")
        self.assertEqual(result["expectation_rows"][0]["observed"], "not_executed")
        self.assertEqual(result["expectation_rows"][0]["injection_level"], "workflow_level")

    def test_validation_passes_expected_observed_hint(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "nat_priority_high")
        result = build_dry_run(manifest, selected, run_id="unit_test", created_at=1.0)
        validation = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            [{"scenario_id": "nat_priority_high", "raw_field": "priority", "value": 100}],
            execution_mode="observed_file",
        )
        self.assertEqual(validation["validation_rows"][0]["result"], "pass")
        self.assertEqual(validation["scenario_summaries"][0]["result"], "pass")

    def test_validation_fails_missing_expected_hint(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "nat_cache_control_ttl")
        result = build_dry_run(manifest, selected, run_id="unit_test", created_at=1.0)
        validation = validate_hint_evidence(manifest, result["scenario_records"], [], execution_mode="observed_file")
        self.assertEqual(validation["validation_rows"][0]["result"], "fail")
        self.assertEqual(validation["scenario_summaries"][0]["result"], "fail")

    def test_validation_marks_missing_optional_hint_without_failing(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "nat_provider_qos")
        result = build_dry_run(manifest, selected, run_id="unit_test", created_at=1.0)
        validation = validate_hint_evidence(manifest, result["scenario_records"], [], execution_mode="observed_file")
        self.assertEqual(validation["validation_rows"][0]["result"], "optional_missing")
        self.assertEqual(validation["scenario_summaries"][0]["result"], "optional_not_observed")

    def test_validation_marks_unknown_hint_for_classification(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "nat_priority_high")
        result = build_dry_run(manifest, selected, run_id="unit_test", created_at=1.0)
        validation = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            [
                {"scenario_id": "nat_priority_high", "raw_field": "priority", "value": 100},
                {"scenario_id": "nat_priority_high", "raw_field": "future_hint", "value": 1},
            ],
            execution_mode="observed_file",
        )
        self.assertEqual(len(validation["unknown_hint_rows"]), 1)
        self.assertEqual(validation["unknown_hint_rows"][0]["status"], "needs_classification")
        self.assertEqual(validation["unknown_hint_rows"][0]["injection_level"], "unknown")

    def test_fixture_observations_pass_smoke_expected_emissions(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "smoke")
        result = build_dry_run(
            manifest,
            selected,
            run_id="unit_test",
            created_at=1.0,
            execution_mode="fixture_smoke",
        )
        observations = build_fixture_observations(result["scenario_records"])
        validation = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            observations,
            execution_mode="fixture_smoke",
        )
        by_scenario = {row["scenario_id"]: row for row in validation["scenario_summaries"]}
        self.assertEqual(by_scenario["nat_priority_high"]["result"], "pass")
        self.assertEqual(by_scenario["nat_priority_low"]["result"], "pass")
        self.assertEqual(by_scenario["nat_prefix_reuse_id"]["result"], "pass")
        self.assertEqual(by_scenario["nat_cache_control_ttl"]["result"], "pass")
        self.assertEqual(by_scenario["nat_no_hints_baseline"]["result"], "pass")
        self.assertEqual(len(observations), 4)

    def test_nat_payload_observations_extract_nested_agent_hints(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "nat_priority_high,nat_cache_control_ttl")
        result = build_dry_run(
            manifest,
            selected,
            run_id="unit_test",
            created_at=1.0,
            execution_mode="nat_dynamo_transport_capture",
        )
        observations = build_nat_payload_observations(
            manifest,
            result["scenario_records"],
            {
                "nat_priority_high": [{"nvext": {"agent_hints": {"priority": 100}}}],
                "nat_cache_control_ttl": [{"nvext": {"cache_control": {"ttl": "1s"}}}],
            },
        )
        validation = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            observations,
            execution_mode="nat_dynamo_transport_capture",
        )
        by_scenario = {row["scenario_id"]: row for row in validation["scenario_summaries"]}
        self.assertEqual(by_scenario["nat_priority_high"]["result"], "pass")
        self.assertEqual(by_scenario["nat_cache_control_ttl"]["result"], "pass")

    def test_payload_index_expectations_can_check_first_only_cache_control(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "nat_cache_control_first_only")
        result = build_dry_run(
            manifest,
            selected,
            run_id="unit_test",
            created_at=1.0,
            execution_mode="nat_dynamo_transport_capture",
        )
        observations = build_nat_payload_observations(
            manifest,
            result["scenario_records"],
            {
                "nat_cache_control_first_only": [
                    {"nvext": {"cache_control": {"type": "ephemeral", "ttl": "1s"}}},
                    {"nvext": {"agent_hints": {"prefix_id": "nat_bench_first_only_prefix_001"}}},
                ]
            },
        )
        validation = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            observations,
            execution_mode="nat_dynamo_transport_capture",
        )
        self.assertTrue(all(row["result"] == "pass" for row in validation["validation_rows"]))
        self.assertEqual(validation["scenario_summaries"][0]["result"], "pass")

    def test_write_dry_run_outputs(self):
        manifest, scenarios = load_benchmark_inputs(MANIFEST, SCENARIOS)
        selected = select_scenarios(scenarios, "smoke")
        result = build_dry_run(manifest, selected, run_id="unit_test", created_at=1.0)
        result["validation"] = validate_hint_evidence(manifest, result["scenario_records"])
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            result["knob_profile"] = {"id": "unit_profile", "scenario_selectors": ["smoke"]}
            write_dry_run_outputs(result, out_dir)
            self.assertTrue((out_dir / "run.json").exists())
            self.assertTrue((out_dir / "knob_profile.json").exists())
            self.assertTrue((out_dir / "scenario_records.jsonl").exists())
            self.assertTrue((out_dir / "scenario_records.json").exists())
            self.assertTrue((out_dir / "expected_hint_evidence.csv").exists())
            self.assertTrue((out_dir / "hint_validation.csv").exists())
            self.assertTrue((out_dir / "hint_validation.json").exists())
            self.assertTrue((out_dir / "scenario_validation_summary.csv").exists())
            self.assertTrue((out_dir / "hint_support_matrix.csv").exists())
            self.assertTrue((out_dir / "unknown_hints.csv").exists())
            run = json.loads((out_dir / "run.json").read_text())
            self.assertEqual(run["scenario_count"], 5)


if __name__ == "__main__":
    unittest.main()
