from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles
from typing import Any


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or str(default))
    except ValueError:
        return default


def _bucket(value: Any, bounds: tuple[int, ...]) -> str:
    try:
        numeric = int(float(value or 0))
    except (TypeError, ValueError):
        numeric = 0
    previous = 0
    for bound in bounds:
        if numeric <= bound:
            return f"{previous + 1}-{bound}"
        previous = bound
    return f">{bounds[-1]}"


def runtime_class_key(meta: dict[str, Any]) -> str:
    """Stable, backend-neutral key for comparable filler runtime observations."""

    parts = [
        str(meta.get("harness") or "unknown_harness"),
        str(meta.get("agentic_workload_profile") or "unknown_profile"),
        str(meta.get("workload_phase_family") or meta.get("phase") or "unknown_phase"),
        str(meta.get("workload_request_kind") or "unknown_kind"),
        str(meta.get("tool_wait_class") or "unknown_wait"),
        f"prompt:{_bucket(meta.get('prompt_tokens'), (1024, 2048, 4096, 8192, 16384))}",
        f"max:{_bucket(meta.get('max_tokens'), (16, 32, 64, 128, 256, 512))}",
    ]
    return "|".join(parts)


def relaxed_runtime_class_key(runtime_class: str) -> str:
    """Drop the tool-wait subtype while preserving the main request shape."""

    parts = runtime_class.split("|")
    if len(parts) != 7:
        return runtime_class
    parts[4] = "*"
    return "|".join(parts)


def oracle_runtime_key(meta: dict[str, Any]) -> str:
    """Exact-repeat key for two-stage oracle admission experiments.

    This intentionally omits mode so a profiling run and oracle run can share
    the same workload identity. For exact oracle use, set
    WORKLOAD_SHAPE_MODE_INDEPENDENT=1 so synthetic prompt shapes also repeat
    across modes.
    """

    explicit = str(meta.get("oracle_runtime_key") or "").strip()
    if explicit:
        return explicit
    label = str(meta.get("label") or meta.get("request_id") or "")
    session_id = str(meta.get("session_id") or "")
    if label and session_id and label.startswith(session_id):
        label_suffix = label[len(session_id):]
    else:
        label_suffix = label
    parts = [
        str(meta.get("harness") or "unknown_harness"),
        str(meta.get("pressure_level") or "unknown_pressure"),
        str(meta.get("agentic_workload_profile") or "unknown_profile"),
        str(meta.get("task_index") or "0"),
        str(meta.get("phase") or "unknown_phase"),
        label_suffix or str(meta.get("workload_request_kind") or "unknown_request"),
        str(meta.get("tool_wait_step") or "0"),
        str(meta.get("tool_wait_class") or "unknown_wait"),
        str(meta.get("workload_phase_family") or "unknown_phase_family"),
        str(meta.get("workload_request_kind") or "unknown_kind"),
        str(meta.get("prompt_tokens") or "unknown_prompt_tokens"),
        str(meta.get("max_tokens") or "unknown_max_tokens"),
    ]
    return "|".join(part.replace("\n", " ").strip() for part in parts)


@dataclass(frozen=True)
class RuntimeEstimate:
    raw_estimate_ms: int
    calibrated_estimate_ms: int
    runtime_class: str
    calibration_source: str
    calibration_sample_count: int = 0
    calibration_quantile: str = ""
    calibration_floor_ms: int = 0

    def as_trace_fields(self) -> dict[str, Any]:
        return {
            "raw_estimated_runtime_ms": self.raw_estimate_ms,
            "estimated_runtime_ms": self.calibrated_estimate_ms,
            "calibrated_runtime_ms": self.calibrated_estimate_ms,
            "runtime_class": self.runtime_class,
            "calibration_source": self.calibration_source,
            "calibration_sample_count": self.calibration_sample_count,
            "calibration_quantile": self.calibration_quantile,
            "calibration_floor_ms": self.calibration_floor_ms,
        }


class RuntimeCalibrator:
    """Conservative request-runtime estimator for admission control.

    The estimator is deliberately backend-neutral. It can consume a CSV of prior
    completion-linkage rows, but it also works with conservative fallback floors
    when no history exists.
    """

    def __init__(self, history_csv: Path | None = None) -> None:
        self.default_floor_ms = _int_env("CONTROLLER_CALIBRATED_DEFAULT_FLOOR_MS", 0)
        self.unknown_floor_ms = _int_env("CONTROLLER_CALIBRATED_UNKNOWN_FLOOR_MS", 5_500)
        self.min_samples = _int_env("CONTROLLER_CALIBRATED_MIN_SAMPLES", 3)
        self.quantile_name = os.environ.get("CONTROLLER_CALIBRATED_QUANTILE", "p95").strip().lower() or "p95"
        self.history: dict[str, list[float]] = {}
        if history_csv is not None and history_csv.exists():
            self._load_history(history_csv)

    @classmethod
    def from_env(cls) -> RuntimeCalibrator:
        raw = os.environ.get("CONTROLLER_CALIBRATION_HISTORY_CSV", "").strip()
        return cls(Path(raw) if raw else None)

    def _load_history(self, path: Path) -> None:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                runtime_class = str(row.get("runtime_class") or "").strip()
                if not runtime_class:
                    runtime_class = runtime_class_key(row)
                try:
                    runtime_ms = float(row.get("actual_runtime_ms") or "")
                except ValueError:
                    continue
                if runtime_ms > 0:
                    self.history.setdefault(runtime_class, []).append(runtime_ms)
                    relaxed_class = relaxed_runtime_class_key(runtime_class)
                    if relaxed_class != runtime_class:
                        self.history.setdefault(relaxed_class, []).append(runtime_ms)

    def estimate(self, meta: dict[str, Any], raw_estimate_ms: int) -> RuntimeEstimate:
        runtime_class = str(meta.get("runtime_class") or "").strip() or runtime_class_key(meta)
        samples = self.history.get(runtime_class, [])
        calibration_source = "history_pquantile_floor"
        if len(samples) < self.min_samples:
            relaxed_class = relaxed_runtime_class_key(runtime_class)
            relaxed_samples = self.history.get(relaxed_class, [])
            if len(relaxed_samples) >= self.min_samples:
                samples = relaxed_samples
                calibration_source = "history_relaxed_tool_wait_pquantile_floor"
        if len(samples) >= self.min_samples:
            historical = self._quantile(samples)
            calibrated = max(raw_estimate_ms, int(round(historical)), self.default_floor_ms)
            return RuntimeEstimate(
                raw_estimate_ms=raw_estimate_ms,
                calibrated_estimate_ms=calibrated,
                runtime_class=runtime_class,
                calibration_source=calibration_source,
                calibration_sample_count=len(samples),
                calibration_quantile=self.quantile_name,
                calibration_floor_ms=self.default_floor_ms,
            )
        calibrated = max(raw_estimate_ms, self.unknown_floor_ms)
        return RuntimeEstimate(
            raw_estimate_ms=raw_estimate_ms,
            calibrated_estimate_ms=calibrated,
            runtime_class=runtime_class,
            calibration_source="unknown_class_fallback_floor",
            calibration_sample_count=len(samples),
            calibration_quantile="",
            calibration_floor_ms=self.unknown_floor_ms,
        )

    def _quantile(self, samples: list[float]) -> float:
        if not samples:
            return float(self.default_floor_ms)
        ordered = sorted(samples)
        if self.quantile_name in {"max", "p100"}:
            return ordered[-1]
        if len(ordered) < 2:
            return ordered[-1]
        q = 0.95
        if self.quantile_name == "p90":
            q = 0.90
        elif self.quantile_name == "p99":
            q = 0.99
        # statistics.quantiles is exclusive by default; inclusive behaves better
        # for small calibration samples.
        cut = int(round(q * 100))
        return quantiles(ordered, n=100, method="inclusive")[min(99, max(1, cut)) - 1]


class OracleExactRuntimeTable:
    """Runtime lookup table for best-case admission experiments."""

    def __init__(self, truth_csv: Path | None = None) -> None:
        self.fallback_ms = _int_env("CONTROLLER_ORACLE_EXACT_FALLBACK_MS", 5_500)
        self.by_key: dict[str, float] = {}
        if truth_csv is not None and truth_csv.exists():
            self._load_truth(truth_csv)

    @classmethod
    def from_env(cls) -> OracleExactRuntimeTable:
        raw = os.environ.get("CONTROLLER_ORACLE_RUNTIME_TRUTH_CSV", "").strip()
        return cls(Path(raw) if raw else None)

    def _load_truth(self, path: Path) -> None:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = str(row.get("oracle_runtime_key") or "").strip()
                if not key:
                    continue
                try:
                    runtime_ms = float(
                        row.get("actual_runtime_ms")
                        or row.get("ttft_ms")
                        or row.get("total_latency_ms")
                        or ""
                    )
                except ValueError:
                    continue
                if runtime_ms > 0:
                    self.by_key[key] = runtime_ms

    def estimate(self, meta: dict[str, Any], raw_estimate_ms: int) -> RuntimeEstimate:
        key = oracle_runtime_key(meta)
        runtime_ms = self.by_key.get(key)
        if runtime_ms is not None:
            return RuntimeEstimate(
                raw_estimate_ms=raw_estimate_ms,
                calibrated_estimate_ms=max(1, int(round(runtime_ms))),
                runtime_class=key,
                calibration_source="oracle_exact_runtime_truth",
                calibration_sample_count=1,
                calibration_quantile="exact",
                calibration_floor_ms=0,
            )
        return RuntimeEstimate(
            raw_estimate_ms=raw_estimate_ms,
            calibrated_estimate_ms=max(raw_estimate_ms, self.fallback_ms),
            runtime_class=key,
            calibration_source="oracle_exact_runtime_miss_fallback",
            calibration_sample_count=0,
            calibration_quantile="miss",
            calibration_floor_ms=self.fallback_ms,
        )
