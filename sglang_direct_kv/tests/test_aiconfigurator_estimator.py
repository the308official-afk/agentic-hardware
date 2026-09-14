from __future__ import annotations

import os

from agentic_kv.controller.aiconfigurator_estimator import (
    AIConfiguratorRuntimeCalibrator,
    _parse_latency_ms,
)
from agentic_kv.controller.runtime_calibration import RuntimeCalibrator


def test_parse_aiconfigurator_json_latency() -> None:
    assert _parse_latency_ms('{"ttft_ms": 123.4}') == 123.4
    assert _parse_latency_ms('{"nested": {"prefill_latency_ms": 55}}') == 55


def test_parse_aiconfigurator_table_latency() -> None:
    assert _parse_latency_ms("TTFT: 295.71ms") == 295.71
    assert _parse_latency_ms("Request Latency: 5.27s") == 5270.0


def test_aiconfigurator_falls_back_when_cli_missing(monkeypatch) -> None:
    monkeypatch.setenv("CONTROLLER_AICONFIGURATOR_BIN", "__missing_aiconfigurator__")
    monkeypatch.setenv("CONTROLLER_AICONFIGURATOR_MODEL_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
    monkeypatch.setenv("CONTROLLER_AICONFIGURATOR_SYSTEM", "h200_sxm")
    monkeypatch.setenv("CONTROLLER_CALIBRATED_UNKNOWN_FLOOR_MS", "4321")
    estimator = AIConfiguratorRuntimeCalibrator.from_env(fallback=RuntimeCalibrator())

    estimate = estimator.estimate(
        {
            "harness": "hatcher",
            "agentic_workload_profile": "realistic_agentic_mix",
            "workload_phase_family": "pressure_filler",
            "workload_request_kind": "file_search_inspect",
            "prompt_tokens": "2048",
            "max_tokens": "32",
        },
        100,
    )

    assert estimate.calibrated_estimate_ms == 4321
    assert estimate.calibration_source.startswith("aiconfigurator_cli_not_found")


def test_aiconfigurator_uses_fake_cli_output(tmp_path, monkeypatch) -> None:
    cli = tmp_path / "fake_aiconfigurator"
    cli.write_text("#!/usr/bin/env bash\necho 'TTFT: 777ms'\n", encoding="utf-8")
    os.chmod(cli, 0o755)
    monkeypatch.setenv("CONTROLLER_AICONFIGURATOR_BIN", str(cli))
    monkeypatch.setenv("CONTROLLER_AICONFIGURATOR_MODEL_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
    monkeypatch.setenv("CONTROLLER_AICONFIGURATOR_SYSTEM", "h200_sxm")
    estimator = AIConfiguratorRuntimeCalibrator.from_env(fallback=RuntimeCalibrator())

    estimate = estimator.estimate(
        {
            "harness": "hatcher",
            "agentic_workload_profile": "realistic_agentic_mix",
            "workload_phase_family": "pressure_filler",
            "workload_request_kind": "file_search_inspect",
            "prompt_tokens": "2048",
            "max_tokens": "32",
            "client_inflight_at_submit": "4",
        },
        100,
    )

    assert estimate.calibrated_estimate_ms == 777
    assert estimate.calibration_source == "aiconfigurator_cli_estimate"
