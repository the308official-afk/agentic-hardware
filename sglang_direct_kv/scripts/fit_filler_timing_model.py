#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


FEATURES = [
    "bias",
    "prompt_tokens",
    "max_tokens",
    "concurrency",
    "client_inflight_at_submit",
    "client_pending_at_submit",
    "client_queue_wait_ms",
    "is_replay",
    "is_test_build",
    "is_slow_external",
]
SUMMARY_FIELDS = [
    "row_count",
    "train_rows",
    "validation_rows",
    "mae_ms",
    "median_abs_error_ms",
    "p90_abs_error_ms",
    "p95_abs_error_ms",
    "underestimate_count",
    "underestimate_rate",
    "worst_underestimate_ms",
    "recommended_p90_safety_margin_ms",
    "recommended_p95_safety_margin_ms",
    "recommended_max_safety_margin_ms",
]
SAFE_SUMMARY_FIELDS = [
    "model",
    "row_count",
    "train_rows",
    "validation_rows",
    "quantile",
    "min_group_samples",
    "underestimate_count",
    "underestimate_rate",
    "worst_underestimate_ms",
    "overestimate_median_ms",
    "overestimate_p90_ms",
    "mae_ms",
]


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            runtime = as_float(row.get("actual_runtime_ms"))
            prompt_tokens = as_float(row.get("prompt_tokens"))
            if prompt_tokens <= 0:
                approx_chars = as_float(row.get("prompt_chars")) or as_float(row.get("body_bytes"))
                if approx_chars > 0:
                    prompt_tokens = max(1.0, approx_chars / 4.0)
                    row["prompt_tokens"] = round(prompt_tokens, 3)
                    row["prompt_tokens_inferred_from_chars"] = "1"
            if runtime <= 0 or prompt_tokens <= 0:
                continue
            if as_float(row.get("max_tokens")) <= 0:
                row["max_tokens"] = "2"
            rows.append(row)
    return rows


def feature_vector(row: dict[str, Any]) -> list[float]:
    phase = str(row.get("phase") or "")
    workload_kind = str(row.get("workload_request_kind") or "")
    workload_family = str(row.get("workload_phase_family") or "")
    return [
        1.0,
        as_float(row.get("prompt_tokens")),
        as_float(row.get("max_tokens")),
        as_float(row.get("concurrency")),
        as_float(row.get("client_inflight_at_submit")),
        as_float(row.get("client_pending_at_submit")),
        as_float(row.get("client_queue_wait_ms")),
        1.0 if "replay" in phase else 0.0,
        1.0 if "test_build" in workload_kind or "test_build" in workload_family else 0.0,
        1.0 if "slow_external" in workload_kind or "slow_external" in workload_family else 0.0,
    ]


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-9:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        aug[col] = [value / denom for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            aug[row] = [value - factor * aug[col][idx] for idx, value in enumerate(aug[row])]
    return [aug[i][-1] for i in range(n)]


def fit_ridge(rows: list[dict[str, Any]], ridge: float = 1e-6) -> list[float]:
    n = len(FEATURES)
    xtx = [[0.0 for _ in range(n)] for _ in range(n)]
    xty = [0.0 for _ in range(n)]
    for row in rows:
        x = feature_vector(row)
        y = as_float(row.get("actual_runtime_ms"))
        for i in range(n):
            xty[i] += x[i] * y
            for j in range(n):
                xtx[i][j] += x[i] * x[j]
    for i in range(n):
        xtx[i][i] += ridge
    return solve_linear_system(xtx, xty)


def predict(row: dict[str, Any], coefficients: list[float]) -> float:
    return max(0.0, sum(value * coefficients[idx] for idx, value in enumerate(feature_vector(row))))


def deterministic_split(rows: list[dict[str, Any]], validation_mod: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        request_id = str(row.get("request_id") or idx)
        bucket = sum(ord(ch) for ch in request_id) % validation_mod
        if bucket == 0:
            validation.append(row)
        else:
            train.append(row)
    if not train or not validation:
        midpoint = max(1, int(len(rows) * 0.8))
        return rows[:midpoint], rows[midpoint:] or rows[:]
    return train, validation


def bucket(value: Any, bounds: tuple[int, ...]) -> str:
    numeric = int(as_float(value))
    previous = 0
    for bound in bounds:
        if numeric <= bound:
            return f"{previous + 1}-{bound}"
        previous = bound
    return f">{bounds[-1]}"


def phase_family(row: dict[str, Any]) -> str:
    phase = str(row.get("phase") or "")
    if "initial" in phase:
        return "initial"
    if "replay" in phase or phase == "pressure_filler":
        return "replay"
    return phase or "unknown"


def group_keys(row: dict[str, Any]) -> list[tuple[str, ...]]:
    kind = str(row.get("workload_request_kind") or "unknown_kind")
    family = str(row.get("workload_phase_family") or "unknown_family")
    concurrency = str(row.get("concurrency") or "unknown_concurrency")
    prompt_bucket = bucket(row.get("prompt_tokens"), (1024, 2048, 4096, 8192, 16384))
    max_bucket = bucket(row.get("max_tokens"), (16, 32, 64, 128, 256, 512))
    phase = phase_family(row)
    return [
        ("phase_kind_conc_prompt_max", phase, kind, concurrency, prompt_bucket, max_bucket),
        ("phase_kind_conc_prompt", phase, kind, concurrency, prompt_bucket),
        ("phase_kind_conc", phase, kind, concurrency),
        ("phase_family_conc", phase, family, concurrency),
        ("phase_conc", phase, concurrency),
        ("phase_kind", phase, kind),
        ("phase", phase),
        ("global",),
    ]


def fit_upper_bound_tables(
    rows: list[dict[str, Any]],
    *,
    quantile: float,
) -> dict[tuple[str, ...], float]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        runtime = as_float(row.get("actual_runtime_ms"))
        if runtime <= 0:
            continue
        for key in group_keys(row):
            groups[key].append(runtime)
    return {key: percentile(values, quantile) for key, values in groups.items()}


def safe_predict(
    row: dict[str, Any],
    tables: dict[tuple[str, ...], float],
    *,
    train_rows: list[dict[str, Any]],
    min_group_samples: int,
) -> tuple[float, tuple[str, ...]]:
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for train_row in train_rows:
        for key in group_keys(train_row):
            counts[key] += 1
    for key in group_keys(row):
        if key in tables and counts[key] >= min_group_samples:
            return tables[key], key
    key = ("global",)
    return tables.get(key, percentile([as_float(r.get("actual_runtime_ms")) for r in train_rows], 0.95)), key


def summarize_safe_model(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    tables: dict[tuple[str, ...], float],
    *,
    quantile: float,
    min_group_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    abs_errors: list[float] = []
    under_errors: list[float] = []
    over_errors: list[float] = []
    for row in rows:
        actual = as_float(row.get("actual_runtime_ms"))
        predicted, key = safe_predict(row, tables, train_rows=train_rows, min_group_samples=min_group_samples)
        signed = actual - predicted
        abs_errors.append(abs(signed))
        if signed > 0:
            under_errors.append(signed)
        else:
            over_errors.append(-signed)
        detail.append(
            {
                **row,
                "safe_predicted_runtime_ms": round(predicted, 3),
                "safe_prediction_error_ms": round(signed, 3),
                "safe_prediction_group": "|".join(key),
                "safe_underestimated": signed > 0,
            }
        )
    summary = {
        "model": "grouped_upper_bound",
        "row_count": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(rows),
        "quantile": f"p{int(round(quantile * 100))}",
        "min_group_samples": min_group_samples,
        "underestimate_count": len(under_errors),
        "underestimate_rate": round(len(under_errors) / len(rows), 4) if rows else 0,
        "worst_underestimate_ms": round(max(under_errors), 3) if under_errors else 0,
        "overestimate_median_ms": round(median(over_errors), 3) if over_errors else 0,
        "overestimate_p90_ms": round(percentile(over_errors, 0.90), 3),
        "mae_ms": round(sum(abs_errors) / len(abs_errors), 3) if abs_errors else 0,
    }
    return summary, detail


def summarize_errors(rows: list[dict[str, Any]], coefficients: list[float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    abs_errors: list[float] = []
    under_errors: list[float] = []
    signed_errors: list[float] = []
    for row in rows:
        actual = as_float(row.get("actual_runtime_ms"))
        predicted = predict(row, coefficients)
        signed = actual - predicted
        abs_error = abs(signed)
        signed_errors.append(signed)
        abs_errors.append(abs_error)
        if signed > 0:
            under_errors.append(signed)
        detail.append(
            {
                **row,
                "predicted_runtime_ms": round(predicted, 3),
                "prediction_error_ms": round(signed, 3),
                "absolute_error_ms": round(abs_error, 3),
                "underestimated": signed > 0,
            }
        )
    summary = {
        "mae_ms": round(sum(abs_errors) / len(abs_errors), 3) if abs_errors else 0,
        "median_abs_error_ms": round(median(abs_errors), 3) if abs_errors else 0,
        "p90_abs_error_ms": round(percentile(abs_errors, 0.90), 3),
        "p95_abs_error_ms": round(percentile(abs_errors, 0.95), 3),
        "underestimate_count": len(under_errors),
        "underestimate_rate": round(len(under_errors) / len(rows), 4) if rows else 0,
        "worst_underestimate_ms": round(max(under_errors), 3) if under_errors else 0,
        "recommended_p90_safety_margin_ms": round(percentile(under_errors, 0.90), 3),
        "recommended_p95_safety_margin_ms": round(percentile(under_errors, 0.95), 3),
        "recommended_max_safety_margin_ms": round(max(under_errors), 3) if under_errors else 0,
    }
    return summary, detail


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a portable filler TTFT timing model.")
    parser.add_argument("--input", type=Path, required=True, help="filler_timing_calibration.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for model artifacts.")
    args = parser.parse_args()
    rows = read_rows(args.input)
    if len(rows) < 3:
        raise SystemExit(f"Need at least 3 calibration rows, got {len(rows)}")
    train, validation = deterministic_split(rows)
    coefficients = fit_ridge(train)
    validation_summary, validation_detail = summarize_errors(validation, coefficients)
    all_summary, all_detail = summarize_errors(rows, coefficients)
    safe_quantile = 0.95
    min_group_samples = 3
    safe_tables = fit_upper_bound_tables(train, quantile=safe_quantile)
    safe_summary, safe_validation_detail = summarize_safe_model(
        validation,
        train,
        safe_tables,
        quantile=safe_quantile,
        min_group_samples=min_group_samples,
    )
    safe_all_summary, _ = summarize_safe_model(
        rows,
        train,
        safe_tables,
        quantile=safe_quantile,
        min_group_samples=min_group_samples,
    )
    summary = {
        "row_count": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        **validation_summary,
    }
    by_concurrency: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_detail:
        grouped[str(row.get("concurrency") or "")].append(row)
    for concurrency, group in sorted(grouped.items(), key=lambda item: as_float(item[0])):
        group_summary, _ = summarize_errors(group, coefficients)
        by_concurrency.append({"concurrency": concurrency, "rows": len(group), **group_summary})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "features": FEATURES,
        "coefficients": {name: round(coefficients[idx], 6) for idx, name in enumerate(FEATURES)},
        "summary": summary,
        "all_rows_summary": all_summary,
        "safe_upper_bound_model": {
            "quantile": safe_summary["quantile"],
            "min_group_samples": min_group_samples,
            "summary": safe_summary,
            "all_rows_summary": safe_all_summary,
        },
    }
    (args.out_dir / "filler_timing_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "filler_timing_model_summary.csv", [summary], SUMMARY_FIELDS)
    write_csv(args.out_dir / "filler_timing_safe_model_summary.csv", [safe_summary], SAFE_SUMMARY_FIELDS)
    write_csv(
        args.out_dir / "filler_timing_model_by_concurrency.csv",
        by_concurrency,
        ["concurrency", "rows", *SUMMARY_FIELDS[3:]],
    )
    detail_fields = list(rows[0].keys()) + ["predicted_runtime_ms", "prediction_error_ms", "absolute_error_ms", "underestimated"]
    write_csv(args.out_dir / "filler_timing_model_predictions.csv", validation_detail, detail_fields)
    safe_detail_fields = list(rows[0].keys()) + [
        "safe_predicted_runtime_ms",
        "safe_prediction_error_ms",
        "safe_prediction_group",
        "safe_underestimated",
    ]
    write_csv(args.out_dir / "filler_timing_safe_model_predictions.csv", safe_validation_detail, safe_detail_fields)
    print(f"wrote {args.out_dir / 'filler_timing_model.json'}")
    print(json.dumps(summary, sort_keys=True))
    print(json.dumps(safe_summary, sort_keys=True))


if __name__ == "__main__":
    main()
