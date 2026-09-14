from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from .runtime_calibration import RuntimeCalibrator, RuntimeEstimate, runtime_class_key


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or str(default))
    except ValueError:
        return default


def _float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_meta(meta: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        value = _float_value(meta.get(key))
        if value is not None:
            return max(0, int(round(value)))
    return default


def _recursive_number_for_key(value: Any, keys: set[str]) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in keys:
                numeric = _float_value(item)
                if numeric is not None:
                    return numeric
        for item in value.values():
            found = _recursive_number_for_key(item, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _recursive_number_for_key(item, keys)
            if found is not None:
                return found
    return None


def _extract_json_objects(text: str) -> list[Any]:
    out: list[Any] = []
    for raw in (text.strip(),):
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            out.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return out


def _parse_latency_ms(stdout: str) -> float | None:
    keys = {
        "ttft",
        "ttft_ms",
        "estimated_ttft",
        "estimated_ttft_ms",
        "prefill",
        "prefill_ms",
        "prefill_latency",
        "prefill_latency_ms",
        "latency",
        "latency_ms",
        "request_latency",
        "request_latency_ms",
    }
    for obj in _extract_json_objects(stdout):
        found = _recursive_number_for_key(obj, keys)
        if found is not None:
            return found

    patterns = (
        r"\bTTFT\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)?",
        r"\bPrefill(?:\s+Latency)?\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)?",
        r"\bRequest\s+Latency\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)?",
        r"\blatency_ms\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bttft_ms\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower() if len(match.groups()) > 1 and match.group(2) else "ms"
        return value * 1000.0 if unit == "s" else value
    return None


@dataclass(frozen=True)
class AIConfiguratorRequest:
    model_path: str
    system: str
    backend: str
    estimate_mode: str
    input_tokens: int
    output_tokens: int
    prefix_tokens: int
    batch_size: int
    tp_size: int

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.model_path,
            self.system,
            self.backend,
            self.estimate_mode,
            self.input_tokens,
            self.output_tokens,
            self.prefix_tokens,
            self.batch_size,
            self.tp_size,
        )


class AIConfiguratorRuntimeCalibrator:
    """Optional AIConfigurator-backed runtime estimator for SJF admission.

    This adapter is intentionally a soft dependency. If the CLI is not present,
    if the model/system is unsupported, or if the estimate output cannot be
    parsed, it falls back to the provided conservative calibrator.
    """

    def __init__(
        self,
        *,
        fallback: RuntimeCalibrator | None = None,
        command_bin: str | None = None,
        model_path: str | None = None,
        system: str | None = None,
        backend: str = "sglang",
        estimate_mode: str = "static_ctx",
        tp_size: int = 1,
        timeout_s: int = 10,
        safety_multiplier: float = 1.0,
        safety_margin_ms: int = 0,
    ) -> None:
        self.fallback = fallback or RuntimeCalibrator.from_env()
        self.command_bin = command_bin or os.environ.get("CONTROLLER_AICONFIGURATOR_BIN", "aiconfigurator")
        self.model_path = model_path or os.environ.get("CONTROLLER_AICONFIGURATOR_MODEL_PATH", "").strip()
        self.system = system or os.environ.get("CONTROLLER_AICONFIGURATOR_SYSTEM", "").strip()
        self.backend = backend
        self.estimate_mode = estimate_mode
        self.tp_size = max(1, int(tp_size))
        self.timeout_s = max(1, int(timeout_s))
        self.safety_multiplier = max(0.01, float(safety_multiplier))
        self.safety_margin_ms = max(0, int(safety_margin_ms))
        self._cache: dict[tuple[Any, ...], int] = {}

    @classmethod
    def from_env(cls, *, fallback: RuntimeCalibrator | None = None) -> AIConfiguratorRuntimeCalibrator:
        return cls(
            fallback=fallback,
            command_bin=os.environ.get("CONTROLLER_AICONFIGURATOR_BIN", "aiconfigurator"),
            model_path=os.environ.get("CONTROLLER_AICONFIGURATOR_MODEL_PATH", "").strip(),
            system=os.environ.get("CONTROLLER_AICONFIGURATOR_SYSTEM", "").strip(),
            backend=os.environ.get("CONTROLLER_AICONFIGURATOR_BACKEND", "sglang").strip() or "sglang",
            estimate_mode=os.environ.get("CONTROLLER_AICONFIGURATOR_ESTIMATE_MODE", "static_ctx").strip()
            or "static_ctx",
            tp_size=_int_env("CONTROLLER_AICONFIGURATOR_TP_SIZE", 1),
            timeout_s=_int_env("CONTROLLER_AICONFIGURATOR_TIMEOUT_S", 10),
            safety_multiplier=float(os.environ.get("CONTROLLER_AICONFIGURATOR_SAFETY_MULTIPLIER", "1.0") or "1.0"),
            safety_margin_ms=_int_env("CONTROLLER_AICONFIGURATOR_SAFETY_MARGIN_MS", 0),
        )

    def estimate(self, meta: dict[str, Any], raw_estimate_ms: int) -> RuntimeEstimate:
        runtime_class = runtime_class_key(meta)
        request = self._request_from_meta(meta)
        if not request.model_path:
            return self._fallback(meta, raw_estimate_ms, "aiconfigurator_missing_model_path")
        if not request.system:
            return self._fallback(meta, raw_estimate_ms, "aiconfigurator_missing_system")
        if shutil.which(self.command_bin) is None and not os.path.exists(self.command_bin):
            return self._fallback(meta, raw_estimate_ms, "aiconfigurator_cli_not_found")

        cache_key = request.cache_key()
        if cache_key not in self._cache:
            predicted_ms, source = self._run_estimate(request)
            if predicted_ms is None:
                return self._fallback(meta, raw_estimate_ms, source)
            safe_ms = int(round(predicted_ms * self.safety_multiplier + self.safety_margin_ms))
            self._cache[cache_key] = max(1, safe_ms)

        return RuntimeEstimate(
            raw_estimate_ms=raw_estimate_ms,
            calibrated_estimate_ms=max(1, self._cache[cache_key]),
            runtime_class=runtime_class,
            calibration_source="aiconfigurator_cli_estimate",
            calibration_sample_count=1,
            calibration_quantile="model",
            calibration_floor_ms=self.safety_margin_ms,
        )

    def _request_from_meta(self, meta: dict[str, Any]) -> AIConfiguratorRequest:
        prompt_tokens = _int_meta(
            meta,
            ("prompt_tokens", "workload_prompt_tokens_target", "input_tokens"),
            0,
        )
        output_tokens = _int_meta(meta, ("max_tokens", "workload_max_tokens", "output_tokens"), 1)
        prefix_tokens = _int_meta(
            meta,
            (
                "cached_prefix_tokens",
                "prefix_tokens",
                "cached_tokens",
                "sglang_cached_tokens",
                "prefix_cache_tokens",
            ),
            _int_env("CONTROLLER_AICONFIGURATOR_PREFIX_TOKENS", 0),
        )
        batch_size = _int_meta(
            meta,
            ("client_inflight_at_submit", "batch_size", "concurrency"),
            _int_env("CONTROLLER_AICONFIGURATOR_BATCH_SIZE", 1),
        )
        return AIConfiguratorRequest(
            model_path=str(meta.get("model") or self.model_path),
            system=self.system,
            backend=self.backend,
            estimate_mode=self.estimate_mode,
            input_tokens=max(1, prompt_tokens),
            output_tokens=max(1, output_tokens),
            prefix_tokens=max(0, min(prefix_tokens, prompt_tokens)),
            batch_size=max(1, batch_size),
            tp_size=self.tp_size,
        )

    def _run_estimate(self, request: AIConfiguratorRequest) -> tuple[float | None, str]:
        template = os.environ.get("CONTROLLER_AICONFIGURATOR_COMMAND_TEMPLATE", "").strip()
        values = {
            "bin": self.command_bin,
            "model_path": request.model_path,
            "system": request.system,
            "backend": request.backend,
            "estimate_mode": request.estimate_mode,
            "isl": request.input_tokens,
            "osl": request.output_tokens,
            "prefix": request.prefix_tokens,
            "batch_size": request.batch_size,
            "tp_size": request.tp_size,
        }
        if template:
            command = shlex.split(template.format(**values))
        else:
            command = [
                self.command_bin,
                "cli",
                "estimate",
                "--model-path",
                request.model_path,
                "--system",
                request.system,
                "--backend",
                request.backend,
                "--estimate-mode",
                request.estimate_mode,
                "--isl",
                str(request.input_tokens),
                "--osl",
                str(request.output_tokens),
                "--prefix",
                str(request.prefix_tokens),
                "--batch-size",
                str(request.batch_size),
                "--tp-size",
                str(request.tp_size),
            ]
            extra = os.environ.get("CONTROLLER_AICONFIGURATOR_EXTRA_ARGS", "").strip()
            if extra:
                command.extend(shlex.split(extra))
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, "aiconfigurator_timeout_fallback"
        except OSError:
            return None, "aiconfigurator_exec_error_fallback"
        if completed.returncode != 0:
            return None, "aiconfigurator_nonzero_exit_fallback"
        parsed = _parse_latency_ms(completed.stdout)
        if parsed is None:
            return None, "aiconfigurator_unparsed_output_fallback"
        return parsed, "aiconfigurator_cli_estimate"

    def _fallback(self, meta: dict[str, Any], raw_estimate_ms: int, source: str) -> RuntimeEstimate:
        fallback = self.fallback.estimate(meta, raw_estimate_ms)
        return RuntimeEstimate(
            raw_estimate_ms=fallback.raw_estimate_ms,
            calibrated_estimate_ms=fallback.calibrated_estimate_ms,
            runtime_class=fallback.runtime_class,
            calibration_source=f"{source}+{fallback.calibration_source}",
            calibration_sample_count=fallback.calibration_sample_count,
            calibration_quantile=fallback.calibration_quantile,
            calibration_floor_ms=fallback.calibration_floor_ms,
        )
