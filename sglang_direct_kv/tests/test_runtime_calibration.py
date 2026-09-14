from __future__ import annotations

import csv
from pathlib import Path

from agentic_kv.controller.runtime_calibration import RuntimeCalibrator, runtime_class_key
from scripts.build_filler_timing_calibration import normalize_row


def test_calibrator_recomputes_blank_runtime_class(tmp_path: Path, monkeypatch) -> None:
    history = tmp_path / "history.csv"
    row = {
        "runtime_class": "",
        "actual_runtime_ms": "1234",
        "harness": "hatcher",
        "agentic_workload_profile": "realistic_agentic_mix",
        "workload_phase_family": "file_inspect",
        "workload_request_kind": "file_search_inspect",
        "tool_wait_class": "quick_file",
        "prompt_tokens": "1536",
        "max_tokens": "32",
    }
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    monkeypatch.setenv("CONTROLLER_CALIBRATED_MIN_SAMPLES", "1")
    calibrator = RuntimeCalibrator(history)
    estimate = calibrator.estimate(dict(row), raw_estimate_ms=100)

    assert estimate.runtime_class == runtime_class_key(row)
    assert estimate.calibration_source == "history_pquantile_floor"
    assert estimate.calibration_sample_count == 1


def test_filler_timing_builder_writes_runtime_class() -> None:
    row = normalize_row(
        Path("case"),
        {
            "event": "m27.request.end",
            "harness": "hatcher",
            "agentic_workload_profile": "realistic_agentic_mix",
            "workload_phase_family": "file_inspect",
            "workload_request_kind": "file_search_inspect",
            "tool_wait_class": "quick_file",
            "prompt_tokens": "1536",
            "max_tokens": "32",
            "ttft_ms": "1000",
            "request_id": "case_pressure_000_initial",
            "phase": "pressure_filler_initial",
        },
        {},
        "gateway_request_end",
    )

    assert row["runtime_class"] == runtime_class_key(row)
