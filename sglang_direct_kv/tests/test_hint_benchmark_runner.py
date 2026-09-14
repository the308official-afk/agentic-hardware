from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agentic_kv.hint_benchmark import (
    build_direct_api_payloads,
    build_nat_payload_observations,
    build_payload_observations,
    build_dry_run,
    build_fixture_observations,
    build_hint_support_matrix,
    evidence_tier_for_mode,
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
CLAUDE_MANIFEST = ROOT / "configs" / "hint_benchmark" / "claude_hints.json"
CLAUDE_SCENARIOS = ROOT / "configs" / "hint_benchmark" / "claude_scenarios.json"
CLAUDE_KNOBS = ROOT / "configs" / "hint_benchmark" / "claude_knobs.json"


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
        self.assertEqual(result["run"]["evidence_tier"], "fixture_plumbing_only")
        self.assertTrue(
            all(row["evidence_tier"] == "fixture_plumbing_only" for row in validation["validation_rows"])
        )
        support_matrix = [
            row for row in build_hint_support_matrix(validation["validation_rows"])
            if row["scenario_id"] != "nat_no_hints_baseline"
        ]
        self.assertTrue(all(row["support"] == "fixture_plumbing_only" for row in support_matrix))
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
        self.assertTrue(
            all(row["evidence_tier"] == "native_client_or_transport_capture" for row in observations)
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

    def test_evidence_tier_classifies_native_and_fixture_modes(self):
        self.assertEqual(evidence_tier_for_mode("fixture_smoke"), "fixture_plumbing_only")
        self.assertEqual(evidence_tier_for_mode("dry_run"), "recipe_only")
        self.assertEqual(
            evidence_tier_for_mode("claude_native_capture"),
            "native_client_or_transport_capture",
        )
        self.assertEqual(
            evidence_tier_for_mode("nat_dynamo_transport_capture"),
            "native_client_or_transport_capture",
        )
        self.assertEqual(
            evidence_tier_for_mode("anthropic_api_payload_capture"),
            "documented_direct_api_payload",
        )
        self.assertEqual(
            evidence_tier_for_mode("claude_real_provider_capture"),
            "native_client_real_provider_response",
        )

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

    def test_loads_claude_manifest_and_scenarios(self):
        manifest, scenarios = load_benchmark_inputs(CLAUDE_MANIFEST, CLAUDE_SCENARIOS)
        self.assertEqual(manifest["harness"]["id"], "claude_code")
        self.assertEqual(len(manifest["hints"]), 8)
        self.assertEqual(len(scenarios["scenarios"]), 19)

    def test_selects_claude_knob_profile_scenarios(self):
        _, scenarios = load_benchmark_inputs(CLAUDE_MANIFEST, CLAUDE_SCENARIOS)
        knobs = load_knob_profiles(CLAUDE_KNOBS)
        profile = select_knob_profile(knobs, "qos_only")
        selected = select_scenarios(scenarios, profile["scenario_selectors"])
        self.assertEqual(
            [scenario["id"] for scenario in selected],
            ["claude_service_tier_auto", "claude_service_tier_standard_only", "claude_fast_mode_setting"],
        )

    def test_generic_payload_observations_extract_claude_array_paths(self):
        manifest, scenarios = load_benchmark_inputs(CLAUDE_MANIFEST, CLAUDE_SCENARIOS)
        selected = select_scenarios(scenarios, "claude_tool_heavy_request")
        result = build_dry_run(
            manifest,
            selected,
            run_id="unit_test",
            created_at=1.0,
            execution_mode="claude_native_capture",
        )
        observations = build_payload_observations(
            manifest,
            result["scenario_records"],
            {
                "claude_tool_heavy_request": [
                    {"tools": [{"cache_control": {"type": "ephemeral"}}]},
                ]
            },
            evidence_source="claude_native_capture",
        )
        validation = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            observations,
            execution_mode="claude_native_capture",
        )
        self.assertEqual(validation["scenario_summaries"][0]["result"], "pass")

    def test_wildcard_paths_find_claude_cache_control_at_nonzero_indexes(self):
        manifest, scenarios = load_benchmark_inputs(CLAUDE_MANIFEST, CLAUDE_SCENARIOS)
        selected = select_scenarios(scenarios, "claude_long_running_cache_session")
        result = build_dry_run(
            manifest,
            selected,
            run_id="unit_test",
            created_at=1.0,
            execution_mode="claude_native_capture",
        )
        observations = build_payload_observations(
            manifest,
            result["scenario_records"],
            {
                "claude_long_running_cache_session": [
                    {
                        "system": [
                            {"type": "text", "text": "not cached"},
                            {"type": "text", "cache_control": {"type": "ephemeral"}, "text": "cached"},
                        ],
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "not cached"}]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "cache_control": {"type": "ephemeral"},
                                        "text": "cached context",
                                    }
                                ],
                            },
                        ],
                    }
                ]
            },
            evidence_source="claude_native_capture",
        )
        validation = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            observations,
            execution_mode="claude_native_capture",
        )
        self.assertEqual(validation["scenario_summaries"][0]["result"], "pass")

    def test_claude_fixture_smoke_passes_expected_emissions(self):
        manifest, scenarios = load_benchmark_inputs(CLAUDE_MANIFEST, CLAUDE_SCENARIOS)
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
        self.assertEqual(by_scenario["claude_no_hints_baseline"]["result"], "pass")
        self.assertEqual(by_scenario["claude_service_tier_auto"]["result"], "pass")
        self.assertEqual(by_scenario["claude_repeated_session_prefix"]["result"], "pass")
        self.assertEqual(by_scenario["claude_long_running_cache_session"]["result"], "pass")

    def test_direct_anthropic_api_payloads_cover_missing_claude_signals(self):
        manifest, scenarios = load_benchmark_inputs(CLAUDE_MANIFEST, CLAUDE_SCENARIOS)
        selected = select_scenarios(scenarios, "direct_anthropic_api_coverage")
        result = build_dry_run(
            manifest,
            selected,
            run_id="unit_test",
            created_at=1.0,
            execution_mode="anthropic_api_payload_capture",
        )
        captured_payloads = build_direct_api_payloads(selected)
        observations = build_payload_observations(
            manifest,
            result["scenario_records"],
            captured_payloads,
            evidence_source="anthropic_api_payload_capture",
        )
        validation = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            observations,
            execution_mode="anthropic_api_payload_capture",
        )
        self.assertTrue(all(row["result"] == "pass" for row in validation["scenario_summaries"]))


if __name__ == "__main__":
    unittest.main()
