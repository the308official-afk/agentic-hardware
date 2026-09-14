"""Runner skeleton for harness hint benchmarks.

The first implementation is intentionally conservative: dry-run mode records
the benchmark scenarios and expected hint evidence, but does not claim that a
real harness emitted those hints.
"""
from __future__ import annotations

import csv
import json
import time
import uuid
from pathlib import Path
from typing import Any


class HintBenchmarkConfigError(ValueError):
    """Raised when a hint benchmark manifest or scenario file is invalid."""


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise HintBenchmarkConfigError(f"{path} must contain a JSON object")
    return data


def load_benchmark_inputs(manifest_path: Path, scenarios_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(manifest_path)
    scenarios = load_json(scenarios_path)
    validate_benchmark_inputs(manifest, scenarios)
    return manifest, scenarios


def load_knob_profiles(path: Path) -> dict[str, Any]:
    knobs = load_json(path)
    validate_knob_profiles(knobs)
    return knobs


def validate_knob_profiles(knobs: dict[str, Any]) -> None:
    profiles = knobs.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise HintBenchmarkConfigError("knob file requires a nonempty profiles object")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, dict):
            raise HintBenchmarkConfigError(f"knob profile {profile_id!r} must be an object")
        selectors = profile.get("scenario_selectors")
        if not isinstance(selectors, list) or not selectors:
            raise HintBenchmarkConfigError(f"knob profile {profile_id!r} requires scenario_selectors")
        for selector in selectors:
            if not isinstance(selector, str) or not selector:
                raise HintBenchmarkConfigError(f"knob profile {profile_id!r} has invalid selector {selector!r}")


def select_knob_profile(knobs: dict[str, Any], profile_id: str) -> dict[str, Any]:
    profile = knobs.get("profiles", {}).get(profile_id)
    if not isinstance(profile, dict):
        valid = ", ".join(sorted(knobs.get("profiles", {})))
        raise HintBenchmarkConfigError(f"unknown knob profile {profile_id!r}; available profiles: {valid}")
    return {"id": profile_id, **profile}


def validate_benchmark_inputs(manifest: dict[str, Any], scenarios: dict[str, Any]) -> None:
    harness = manifest.get("harness", {}).get("id")
    if not harness:
        raise HintBenchmarkConfigError("manifest.harness.id is required")
    if scenarios.get("harness") != harness:
        raise HintBenchmarkConfigError(
            f"scenario harness {scenarios.get('harness')!r} does not match manifest harness {harness!r}"
        )

    valid_levels = set(manifest.get("injection_levels", {})) | set(scenarios.get("injection_levels", {}))
    if not valid_levels:
        raise HintBenchmarkConfigError("injection_levels are required")

    hints = manifest.get("hints")
    scenario_items = scenarios.get("scenarios")
    if not isinstance(hints, list) or not hints:
        raise HintBenchmarkConfigError("manifest.hints must be a nonempty list")
    if not isinstance(scenario_items, list) or not scenario_items:
        raise HintBenchmarkConfigError("scenarios.scenarios must be a nonempty list")

    hint_ids: set[str] = set()
    scenario_ids: set[str] = set()
    for hint in hints:
        hint_id = hint.get("id")
        if not hint_id:
            raise HintBenchmarkConfigError("every hint requires an id")
        if hint_id in hint_ids:
            raise HintBenchmarkConfigError(f"duplicate hint id: {hint_id}")
        hint_ids.add(hint_id)
        level = hint.get("default_injection_level")
        if level not in valid_levels:
            raise HintBenchmarkConfigError(f"hint {hint_id} has invalid default_injection_level {level!r}")

    for scenario in scenario_items:
        scenario_id = scenario.get("id")
        if not scenario_id:
            raise HintBenchmarkConfigError("every scenario requires an id")
        if scenario_id in scenario_ids:
            raise HintBenchmarkConfigError(f"duplicate scenario id: {scenario_id}")
        scenario_ids.add(scenario_id)
        level = scenario.get("injection_level")
        if level not in valid_levels:
            raise HintBenchmarkConfigError(f"scenario {scenario_id} has invalid injection_level {level!r}")
        for hint_id in scenario.get("target_hint_ids", []):
            if hint_id not in hint_ids:
                raise HintBenchmarkConfigError(f"scenario {scenario_id} references unknown hint {hint_id}")

    manifest_refs = {scenario_id for hint in hints for scenario_id in hint.get("scenario_ids", [])}
    baseline_scenario_id = manifest.get("baseline_scenario_id")
    if baseline_scenario_id:
        manifest_refs.add(baseline_scenario_id)
    missing = sorted(manifest_refs - scenario_ids)
    if missing:
        raise HintBenchmarkConfigError(f"manifest references missing scenarios: {missing}")

    smoke = scenarios.get("scenario_groups", {}).get("smoke", [])
    missing_smoke = sorted(set(manifest.get("required_smoke_scenarios", [])) - set(smoke))
    if missing_smoke:
        raise HintBenchmarkConfigError(f"smoke group missing required scenarios: {missing_smoke}")


def select_scenarios(scenarios: dict[str, Any], selector: str | list[str]) -> list[dict[str, Any]]:
    all_scenarios = scenarios.get("scenarios", [])
    by_id = {scenario["id"]: scenario for scenario in all_scenarios}
    groups = scenarios.get("scenario_groups", {})
    if isinstance(selector, str):
        selectors = [part.strip() for part in selector.split(",") if part.strip()]
    else:
        selectors = selector
    if not selectors:
        raise HintBenchmarkConfigError("at least one scenario selector is required")

    selected_ids: list[str] = []
    for item in selectors:
        if item == "all":
            selected_ids.extend(scenario["id"] for scenario in all_scenarios)
        elif item in groups:
            selected_ids.extend(groups[item])
        elif item in by_id:
            selected_ids.append(item)
        else:
            raise HintBenchmarkConfigError(f"unknown scenario selector: {item}")

    deduped_ids = list(dict.fromkeys(selected_ids))
    missing = [scenario_id for scenario_id in deduped_ids if scenario_id not in by_id]
    if missing:
        raise HintBenchmarkConfigError(f"scenario group references missing scenarios: {missing}")
    return [by_id[scenario_id] for scenario_id in deduped_ids]


def build_dry_run(
    manifest: dict[str, Any],
    scenarios: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    created_at: float | None = None,
    execution_mode: str = "dry_run",
) -> dict[str, Any]:
    run_id = run_id or f"hint_bench_{uuid.uuid4().hex[:12]}"
    created_at = time.time() if created_at is None else created_at
    harness = manifest["harness"]["id"]
    tier = evidence_tier_for_mode(execution_mode)
    scenario_records = []
    expectation_rows = []

    for scenario in scenarios:
        base = {
            "run_id": run_id,
            "created_at_unix": created_at,
            "harness": harness,
            "execution_mode": execution_mode,
            "evidence_tier": tier,
            "scenario_id": scenario["id"],
            "scenario_name": scenario.get("display_name", scenario["id"]),
            "scenario_status": scenario.get("status", ""),
            "source_class": scenario.get("source_class", ""),
            "injection_level": scenario.get("injection_level", ""),
            "scope": scenario.get("scope", ""),
            "trigger_type": scenario.get("trigger_type", ""),
        }
        scenario_records.append(
            {
                **base,
                "target_hint_ids": scenario.get("target_hint_ids", []),
                "workload_shape": scenario.get("workload_shape", {}),
                "client_setup": scenario.get("client_setup", {}),
                "synthetic_setup": scenario.get("synthetic_setup", {}),
                "expected_emissions": scenario.get("expected_emissions", []),
                "negative_expectations": scenario.get("negative_expectations", []),
                "observed_emissions": [],
                "result": "not_evaluated",
                "result_reason": "dry_run_records_recipe_only",
                "benchmark_question": scenario.get("benchmark_question", ""),
                "notes": scenario.get("notes", ""),
            }
        )
        for expectation_type, entries in (
            ("expected_emission", scenario.get("expected_emissions", [])),
            ("negative_expectation", scenario.get("negative_expectations", [])),
        ):
            for entry in entries:
                expectation_rows.append(
                    {
                        **base,
                        "expectation_type": expectation_type,
                        "hint_id": entry.get("hint_id", ""),
                        "raw_field": entry.get("raw_field", ""),
                        "payload_index": entry.get("payload_index", ""),
                        "expected_value_json": json.dumps(entry.get("expected_value"), sort_keys=True),
                        "required": bool(entry.get("required", expectation_type == "expected_emission")),
                        "observed": "not_executed",
                        "observed_value_json": "",
                        "result": "not_evaluated",
                        "result_reason": "dry_run_records_recipe_only",
                    }
                )

    return {
        "run": {
            "run_id": run_id,
            "created_at_unix": created_at,
            "harness": harness,
            "execution_mode": execution_mode,
            "evidence_tier": tier,
            "scenario_count": len(scenario_records),
            "expectation_count": len(expectation_rows),
            "manifest_schema_version": manifest.get("schema_version"),
        },
        "scenario_records": scenario_records,
        "expectation_rows": expectation_rows,
    }


def evidence_tier_for_mode(execution_mode: str) -> str:
    if execution_mode in {"nat_dynamo_transport_capture", "claude_native_capture"}:
        return "native_client_or_transport_capture"
    if execution_mode == "claude_real_provider_capture":
        return "native_client_real_provider_response"
    if execution_mode == "anthropic_api_payload_capture":
        return "documented_direct_api_payload"
    if execution_mode == "observed_file":
        return "external_observed_file"
    if execution_mode == "fixture_smoke":
        return "fixture_plumbing_only"
    if execution_mode == "dry_run":
        return "recipe_only"
    return "unknown"


def build_fixture_observations(scenario_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for scenario in scenario_records:
        for entry in scenario.get("expected_emissions", []):
            observations.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_name": scenario.get("scenario_name", ""),
                    "harness": scenario.get("harness", ""),
                    "hint_id": entry.get("hint_id", ""),
                    "raw_field": entry.get("raw_field", ""),
                    "value": fixture_value_for_expected(entry.get("expected_value")),
                    "source_class": scenario.get("source_class", ""),
                    "injection_level": scenario.get("injection_level", ""),
                    "scope": scenario.get("scope", ""),
                    "evidence_source": "fixture_from_scenario_expected_emissions",
                    "evidence_tier": "fixture_plumbing_only",
                }
            )
    return observations


def normalize_raw_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def dotted_values(payload: Any, field: str) -> list[Any]:
    """Return all values matching a dotted path.

    The path language is intentionally small: `*` matches every item in a list
    or every value in a dict. Numeric components still address list indexes.
    """
    parts = [part for part in field.split(".") if part]

    def visit(current: Any, remaining: list[str]) -> list[Any]:
        if not remaining:
            return [current]
        part = remaining[0]
        rest = remaining[1:]
        if part == "*":
            if isinstance(current, list):
                values: list[Any] = []
                for item in current:
                    values.extend(visit(item, rest))
                return values
            if isinstance(current, dict):
                values = []
                for item in current.values():
                    values.extend(visit(item, rest))
                return values
            return []
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return []
            if index < 0 or index >= len(current):
                return []
            return visit(current[index], rest)
        if isinstance(current, dict):
            if part not in current:
                return []
            return visit(current[part], rest)
        return []

    return visit(payload, parts)


def dotted_get(payload: dict[str, Any], field: str) -> tuple[bool, Any]:
    values = dotted_values(payload, field)
    if not values:
        return False, None
    return True, values[0]


def hint_raw_fields(hint: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for key in ("raw_fields", "raw_nat_fields", "raw_claude_fields"):
        for raw_field in hint.get(key, []):
            if raw_field and raw_field not in fields:
                fields.append(raw_field)
    return fields


def observed_field_value(observation: dict[str, Any], raw_field: str) -> tuple[bool, Any]:
    if observation.get("raw_field") == raw_field:
        return True, observation.get("value")
    if raw_field in observation:
        return True, observation[raw_field]
    for key in ("raw_emitted_value", "raw_emitted_values", "metadata", "payload"):
        nested = observation.get(key)
        if isinstance(nested, dict):
            found, value = dotted_get(nested, raw_field)
            if found:
                return True, value
    return False, None


def value_matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "one_of" in expected:
            return any(value_matches(observed, item) for item in expected["one_of"])
        if "contains" in expected:
            return str(expected["contains"]) in str(observed)
        if "numeric_range" in expected:
            lo, hi = expected["numeric_range"]
            try:
                observed_number = float(observed)
            except (TypeError, ValueError):
                return False
            return float(lo) <= observed_number <= float(hi)
    if expected is None or expected == "present":
        return observed is not None
    return observed == expected


def fixture_value_for_expected(expected: Any) -> Any:
    if isinstance(expected, dict):
        if "one_of" in expected and expected["one_of"]:
            return expected["one_of"][0]
        if "contains" in expected:
            return str(expected["contains"])
        if "numeric_range" in expected:
            lo, hi = expected["numeric_range"]
            return (float(lo) + float(hi)) / 2
    if expected == "present":
        return "present"
    return expected


def nat_payload_field_value(payload: dict[str, Any], raw_field: str) -> tuple[bool, Any]:
    """Find a NAT hint value in either its raw location or nvext.agent_hints."""
    found, value = dotted_get(payload, raw_field)
    if found:
        return True, value
    agent_hints = payload.get("nvext", {}).get("agent_hints", {})
    if isinstance(agent_hints, dict):
        found, value = dotted_get(agent_hints, raw_field)
        if found:
            return True, value
    return False, None


def build_nat_payload_observations(
    manifest: dict[str, Any],
    scenario_records: list[dict[str, Any]],
    captured_payloads: dict[str, list[dict[str, Any]]],
    *,
    evidence_source: str = "nat_dynamo_transport_capture",
) -> list[dict[str, Any]]:
    """Convert captured NAT-boundary request bodies into observed hint rows."""
    hint_by_field: dict[str, str] = {}
    for hint in manifest.get("hints", []):
        for raw_field in hint_raw_fields(hint):
            hint_by_field.setdefault(raw_field, hint["id"])

    observations: list[dict[str, Any]] = []
    for scenario in scenario_records:
        scenario_id = scenario["scenario_id"]
        for payload_index, payload in enumerate(captured_payloads.get(scenario_id, []), 1):
            fields = {
                entry.get("raw_field", "")
                for entry in scenario.get("expected_emissions", []) + scenario.get("negative_expectations", [])
                if entry.get("raw_field")
            }
            fields.update(hint_by_field)
            for raw_field in sorted(fields):
                found, value = nat_payload_field_value(payload, raw_field)
                if not found:
                    continue
                observations.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_name": scenario.get("scenario_name", ""),
                        "harness": scenario.get("harness", ""),
                        "hint_id": hint_by_field.get(raw_field, "unknown"),
                        "raw_field": raw_field,
                        "value": value,
                        "source_class": scenario.get("source_class", ""),
                        "injection_level": scenario.get("injection_level", ""),
                        "scope": scenario.get("scope", ""),
                        "payload_index": payload_index,
                        "evidence_source": evidence_source,
                        "evidence_tier": evidence_tier_for_mode(evidence_source),
                        "raw_emitted_value": payload,
                    }
                )
    return observations


def build_payload_observations(
    manifest: dict[str, Any],
    scenario_records: list[dict[str, Any]],
    captured_payloads: dict[str, list[dict[str, Any]]],
    *,
    evidence_source: str,
) -> list[dict[str, Any]]:
    """Convert generic captured request/response bodies into observed hint rows."""
    hint_by_field: dict[str, str] = {}
    for hint in manifest.get("hints", []):
        for raw_field in hint_raw_fields(hint):
            hint_by_field.setdefault(raw_field, hint["id"])

    observations: list[dict[str, Any]] = []
    for scenario in scenario_records:
        scenario_id = scenario["scenario_id"]
        for payload_index, payload in enumerate(captured_payloads.get(scenario_id, []), 1):
            fields = {
                entry.get("raw_field", "")
                for entry in scenario.get("expected_emissions", []) + scenario.get("negative_expectations", [])
                if entry.get("raw_field")
            }
            fields.update(hint_by_field)
            for raw_field in sorted(fields):
                found, value = dotted_get(payload, raw_field)
                if not found:
                    continue
                observations.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_name": scenario.get("scenario_name", ""),
                        "harness": scenario.get("harness", ""),
                        "hint_id": hint_by_field.get(raw_field, "unknown"),
                        "raw_field": raw_field,
                        "value": value,
                        "source_class": scenario.get("source_class", ""),
                        "injection_level": scenario.get("injection_level", ""),
                        "scope": scenario.get("scope", ""),
                        "payload_index": payload_index,
                        "evidence_source": evidence_source,
                        "evidence_tier": evidence_tier_for_mode(evidence_source),
                        "raw_emitted_value": payload,
                    }
                )
    return observations


def replace_placeholders(value: Any, *, scenario_id: str, invocation_index: int) -> Any:
    if isinstance(value, str):
        return value.replace("{scenario_id}", scenario_id).replace("{invocation_index}", str(invocation_index))
    if isinstance(value, list):
        return [replace_placeholders(item, scenario_id=scenario_id, invocation_index=invocation_index) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_placeholders(item, scenario_id=scenario_id, invocation_index=invocation_index)
            for key, item in value.items()
        }
    return value


def build_direct_api_payloads(scenarios: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build documented direct Anthropic API request/response payload examples.

    These rows are deliberately separate from Claude Code native capture. They
    prove the benchmark can represent lower-level API fields, not that the
    Claude Code CLI emitted them organically.
    """
    captured: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        setup = scenario.get("client_setup") or scenario.get("synthetic_setup", {})
        request_count = max(1, int(scenario.get("workload_shape", {}).get("request_count") or 1))
        api_payloads = setup.get("api_payloads")
        api_payload = setup.get("api_payload")
        api_response = setup.get("api_response_payload")
        api_headers = setup.get("api_headers", {})
        if not isinstance(api_headers, dict):
            api_headers = {}

        payloads: list[dict[str, Any]] = []
        for invocation_index in range(request_count):
            if isinstance(api_payloads, list) and api_payloads:
                template = api_payloads[min(invocation_index, len(api_payloads) - 1)]
            elif isinstance(api_payload, dict):
                template = api_payload
            else:
                template = {
                    "model": "claude-opus-5",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": f"Direct API benchmark {scenario['id']}"}],
                }
            payload = replace_placeholders(template, scenario_id=scenario["id"], invocation_index=invocation_index)
            if not isinstance(payload, dict):
                payload = {"body": payload}
            payload = dict(payload)
            if api_headers:
                payload["_capture"] = {
                    "kind": "request",
                    "method": "POST",
                    "path": "/v1/messages",
                    "headers": {str(key).lower(): value for key, value in api_headers.items()},
                }
            payloads.append(payload)

        if isinstance(api_response, dict):
            response_payload = replace_placeholders(api_response, scenario_id=scenario["id"], invocation_index=0)
            response_payload = dict(response_payload)
            response_payload.setdefault("_capture", {"kind": "response", "path": "/v1/messages"})
            payloads.append(response_payload)

        captured[scenario["id"]] = payloads
    return captured


def load_observations_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HintBenchmarkConfigError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise HintBenchmarkConfigError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def validate_hint_evidence(
    manifest: dict[str, Any],
    scenario_records: list[dict[str, Any]],
    observations: list[dict[str, Any]] | None = None,
    *,
    execution_mode: str = "dry_run",
) -> dict[str, Any]:
    observations = observations or []
    tier = evidence_tier_for_mode(execution_mode)
    known_fields = {raw_field for hint in manifest.get("hints", []) for raw_field in hint_raw_fields(hint)}
    known_hint_ids = {hint["id"] for hint in manifest.get("hints", [])}

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        by_scenario.setdefault(observation.get("scenario_id", ""), []).append(observation)

    validation_rows: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []

    for scenario in scenario_records:
        scenario_id = scenario["scenario_id"]
        scenario_observations = by_scenario.get(scenario_id, [])
        scenario_result = "pass"
        pass_count = fail_count = not_evaluated_count = optional_missing_count = 0

        for expectation_type, entries in (
            ("expected_emission", scenario.get("expected_emissions", [])),
            ("negative_expectation", scenario.get("negative_expectations", [])),
        ):
            for entry in entries:
                raw_field = entry.get("raw_field", "")
                expected_value = entry.get("expected_value")
                required = bool(entry.get("required", expectation_type == "expected_emission"))
                expected_payload_index = entry.get("payload_index")
                matches = []
                for observation in scenario_observations:
                    if expected_payload_index and observation.get("payload_index") != expected_payload_index:
                        continue
                    found, value = observed_field_value(observation, raw_field)
                    if found:
                        matches.append((observation, value))
                if execution_mode == "dry_run" and not scenario_observations:
                    observed = "not_executed"
                    observed_value = ""
                    result = "not_evaluated"
                    reason = "dry_run_has_no_observed_harness_evidence"
                    not_evaluated_count += 1
                elif expectation_type == "expected_emission":
                    good = [value for _, value in matches if value_matches(value, expected_value)]
                    if good:
                        observed = "yes"
                        observed_value = good[0]
                        result = "pass"
                        reason = "expected_hint_observed"
                        pass_count += 1
                    elif not required:
                        observed = "no"
                        observed_value = matches[0][1] if matches else ""
                        result = "optional_missing"
                        reason = "optional_hint_not_observed_in_this_capture"
                        optional_missing_count += 1
                    else:
                        observed = "no"
                        observed_value = matches[0][1] if matches else ""
                        result = "fail"
                        reason = "expected_hint_missing_or_value_mismatch"
                        fail_count += 1
                else:
                    if matches:
                        observed = "yes"
                        observed_value = matches[0][1]
                        result = "fail"
                        reason = "negative_expectation_was_observed"
                        fail_count += 1
                    else:
                        observed = "no"
                        observed_value = ""
                        result = "pass"
                        reason = "negative_expectation_absent"
                        pass_count += 1

                validation_rows.append(
                    {
                        "run_id": scenario.get("run_id", ""),
                        "harness": scenario.get("harness", ""),
                        "execution_mode": execution_mode,
                        "evidence_tier": tier,
                        "scenario_id": scenario_id,
                        "scenario_name": scenario.get("scenario_name", ""),
                        "scenario_status": scenario.get("scenario_status", ""),
                        "expectation_type": expectation_type,
                        "hint_id": entry.get("hint_id", ""),
                        "raw_field": raw_field,
                        "payload_index": expected_payload_index or "",
                        "source_class": scenario.get("source_class", ""),
                        "injection_level": scenario.get("injection_level", ""),
                        "scope": scenario.get("scope", ""),
                        "expected_value_json": normalize_raw_value(expected_value),
                        "required": required,
                        "observed": observed,
                        "observed_value_json": normalize_raw_value(observed_value) if observed_value != "" else "",
                        "result": result,
                        "result_reason": reason,
                    }
                )

        for observation in scenario_observations:
            observed_hint_id = observation.get("hint_id")
            observed_raw_field = observation.get("raw_field")
            unknown_hint = observed_hint_id and observed_hint_id not in known_hint_ids
            unknown_field = observed_raw_field and observed_raw_field not in known_fields
            if unknown_hint or unknown_field:
                unknown_rows.append(
                    {
                        "run_id": scenario.get("run_id", ""),
                        "harness": scenario.get("harness", ""),
                        "execution_mode": execution_mode,
                        "evidence_tier": tier,
                        "scenario_id": scenario_id,
                        "hint_id": observed_hint_id or "unknown",
                        "raw_field": observed_raw_field or "unknown",
                        "source_class": "unknown",
                        "injection_level": "unknown",
                        "status": "needs_classification",
                        "raw_observation_json": json.dumps(observation, sort_keys=True),
                    }
                )

        if fail_count:
            scenario_result = "fail"
        elif not_evaluated_count and not pass_count:
            scenario_result = "not_evaluated"
        elif not_evaluated_count:
            scenario_result = "partial"
        elif optional_missing_count and not pass_count:
            scenario_result = "optional_not_observed"
        elif optional_missing_count:
            scenario_result = "partial_optional"
        scenario_summaries.append(
            {
                "run_id": scenario.get("run_id", ""),
                "harness": scenario.get("harness", ""),
                "execution_mode": execution_mode,
                "evidence_tier": tier,
                "scenario_id": scenario_id,
                "scenario_name": scenario.get("scenario_name", ""),
                "result": scenario_result,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "optional_missing_count": optional_missing_count,
                "not_evaluated_count": not_evaluated_count,
                "observation_count": len(scenario_observations),
                "unknown_hint_count": sum(row["scenario_id"] == scenario_id for row in unknown_rows),
            }
        )

    return {
        "validation_rows": validation_rows,
        "scenario_summaries": scenario_summaries,
        "unknown_hint_rows": unknown_rows,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_dry_run_outputs(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "run.json").write_text(json.dumps(result["run"], indent=2, sort_keys=True) + "\n")
    if "captured_payload_counts" in result:
        (out_dir / "captured_payload_counts.json").write_text(
            json.dumps(result["captured_payload_counts"], indent=2, sort_keys=True) + "\n"
        )
    if "client_runs" in result:
        (out_dir / "client_runs.json").write_text(
            json.dumps(result["client_runs"], indent=2, sort_keys=True) + "\n"
        )
    if "knob_profile" in result:
        (out_dir / "knob_profile.json").write_text(
            json.dumps(result["knob_profile"], indent=2, sort_keys=True) + "\n"
        )
    write_jsonl(out_dir / "scenario_records.jsonl", result["scenario_records"])
    if "observations" in result:
        write_jsonl(out_dir / "observed_hint_evidence.jsonl", result["observations"])
    write_csv(out_dir / "expected_hint_evidence.csv", result["expectation_rows"])
    (out_dir / "scenario_records.json").write_text(
        json.dumps(result["scenario_records"], indent=2, sort_keys=True) + "\n"
    )
    validation = result.get("validation")
    if validation:
        write_csv(out_dir / "hint_validation.csv", validation["validation_rows"])
        write_csv(out_dir / "scenario_validation_summary.csv", validation["scenario_summaries"])
        write_csv(out_dir / "hint_support_matrix.csv", build_hint_support_matrix(validation["validation_rows"]))
        write_csv(out_dir / "unknown_hints.csv", validation["unknown_hint_rows"])
        (out_dir / "hint_validation.json").write_text(
            json.dumps(validation["validation_rows"], indent=2, sort_keys=True) + "\n"
        )


def build_hint_support_matrix(validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in validation_rows:
        if row.get("expectation_type") != "expected_emission":
            continue
        result = row.get("result", "")
        evidence_tier = row.get("evidence_tier", "")
        if result == "pass" and evidence_tier == "fixture_plumbing_only":
            support = "fixture_plumbing_only"
        elif result == "pass":
            support = "observed"
        elif result == "optional_missing":
            support = "optional_not_observed"
        elif result == "fail":
            support = "missing_required_or_mismatch"
        elif result == "not_evaluated":
            support = "not_evaluated"
        else:
            support = result or "unknown"
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "scenario_name": row.get("scenario_name", ""),
                "hint_id": row.get("hint_id", ""),
                "raw_field": row.get("raw_field", ""),
                "source_class": row.get("source_class", ""),
                "execution_mode": row.get("execution_mode", ""),
                "evidence_tier": evidence_tier,
                "injection_level": row.get("injection_level", ""),
                "scope": row.get("scope", ""),
                "required": row.get("required", ""),
                "support": support,
                "observed_value_json": row.get("observed_value_json", ""),
                "result_reason": row.get("result_reason", ""),
            }
        )
    return rows
