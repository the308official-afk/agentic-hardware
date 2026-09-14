#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

from agentic_kv.controller.aiconfigurator_estimator import AIConfiguratorRuntimeCalibrator
from agentic_kv.controller.runtime_calibration import RuntimeCalibrator


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "request_id",
        "phase",
        "stage",
        "runtime_class",
        "prompt_tokens",
        "max_tokens",
        "client_inflight_at_submit",
        "client_pending_at_submit",
        "client_queue_wait_ms",
        "actual_runtime_ms",
        "raw_estimated_runtime_ms",
        "predicted_runtime_ms",
        "prediction_error_ms",
        "underestimated",
        "calibration_source",
        "calibration_sample_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AIConfigurator runtime predictions against filler calibration rows."
    )
    parser.add_argument("--input", type=Path, required=True, help="filler_timing_calibration.csv")
    parser.add_argument("--out", type=Path, required=True, help="Prediction comparison CSV")
    parser.add_argument("--summary-out", type=Path, help="Optional summary JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for a quick feasibility pass")
    parser.add_argument("--model-path", default="", help="Override CONTROLLER_AICONFIGURATOR_MODEL_PATH")
    parser.add_argument("--system", default="", help="Override CONTROLLER_AICONFIGURATOR_SYSTEM")
    parser.add_argument("--backend", default="", help="Override CONTROLLER_AICONFIGURATOR_BACKEND")
    parser.add_argument("--estimate-mode", default="", help="Override CONTROLLER_AICONFIGURATOR_ESTIMATE_MODE")
    parser.add_argument("--tp-size", type=int, default=0, help="Override CONTROLLER_AICONFIGURATOR_TP_SIZE")
    args = parser.parse_args()

    if args.model_path:
        os.environ["CONTROLLER_AICONFIGURATOR_MODEL_PATH"] = args.model_path
    if args.system:
        os.environ["CONTROLLER_AICONFIGURATOR_SYSTEM"] = args.system
    if args.backend:
        os.environ["CONTROLLER_AICONFIGURATOR_BACKEND"] = args.backend
    if args.estimate_mode:
        os.environ["CONTROLLER_AICONFIGURATOR_ESTIMATE_MODE"] = args.estimate_mode
    if args.tp_size:
        os.environ["CONTROLLER_AICONFIGURATOR_TP_SIZE"] = str(args.tp_size)

    estimator = AIConfiguratorRuntimeCalibrator.from_env(fallback=RuntimeCalibrator.from_env())
    output_rows: list[dict[str, Any]] = []
    for row in rows_from_csv(args.input):
        actual = as_float(row.get("actual_runtime_ms"))
        if actual is None or actual <= 0:
            continue
        raw = as_float(row.get("raw_estimated_runtime_ms") or row.get("estimated_runtime_ms"))
        if raw is None:
            raw = as_float(row.get("ttft_ms")) or actual
        estimate = estimator.estimate(row, max(1, int(round(raw))))
        predicted = float(estimate.calibrated_estimate_ms)
        error = predicted - actual
        output_rows.append(
            {
                **row,
                "runtime_class": estimate.runtime_class,
                "raw_estimated_runtime_ms": estimate.raw_estimate_ms,
                "predicted_runtime_ms": round(predicted, 3),
                "prediction_error_ms": round(error, 3),
                "underestimated": "yes" if error < 0 else "no",
                "calibration_source": estimate.calibration_source,
                "calibration_sample_count": estimate.calibration_sample_count,
            }
        )
        if args.limit > 0 and len(output_rows) >= args.limit:
            break

    write_csv(args.out, output_rows)
    errors = [float(row["prediction_error_ms"]) for row in output_rows]
    abs_errors = [abs(value) for value in errors]
    under = [value for value in errors if value < 0]
    sources: dict[str, int] = {}
    for row in output_rows:
        source = str(row.get("calibration_source") or "")
        sources[source] = sources.get(source, 0) + 1
    summary = {
        "row_count": len(output_rows),
        "mean_abs_error_ms": round(mean(abs_errors), 3) if abs_errors else 0,
        "median_abs_error_ms": round(median(abs_errors), 3) if abs_errors else 0,
        "p90_abs_error_ms": round(percentile(abs_errors, 0.90), 3) if abs_errors else 0,
        "p95_abs_error_ms": round(percentile(abs_errors, 0.95), 3) if abs_errors else 0,
        "underestimate_count": len(under),
        "underestimate_rate": round(len(under) / len(output_rows), 4) if output_rows else 0,
        "worst_underestimate_ms": round(abs(min(under)), 3) if under else 0,
        "calibration_sources": sources,
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
