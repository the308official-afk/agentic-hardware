from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_calibration import RuntimeEstimate


@dataclass
class SafeFillerAdmissionRequest:
    seq: int
    decision_id: str
    request_id: str
    stage: str
    estimated_runtime_ms: int
    raw_estimated_runtime_ms: int
    runtime_class: str
    calibration_source: str
    calibration_sample_count: int
    calibration_quantile: str
    calibration_floor_ms: int
    created_offset_ms: float
    meta: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]


class SafeFillerAdmissionScheduler:
    """Admit short filler work while protecting the next replay window."""

    def __init__(
        self,
        *,
        trace: Path | None,
        now_ms: Callable[[], float],
        gate_state: dict[str, Any],
        safety_margin_ms: int,
        max_in_flight: int,
        require_fit_before_replay: bool,
        idle_override: bool,
        estimate_runtime_ms: Callable[[int, int], int],
        calibrate_runtime: Callable[[dict[str, Any], int], RuntimeEstimate] | None = None,
    ) -> None:
        self.trace = trace
        self.now_ms = now_ms
        self.gate_state = gate_state
        self.safety_margin_ms = safety_margin_ms
        self.max_in_flight = max(1, max_in_flight)
        self.require_fit_before_replay = require_fit_before_replay
        self.idle_override = idle_override
        self.estimate_runtime_ms = estimate_runtime_ms
        self.calibrate_runtime = calibrate_runtime
        self._lock = asyncio.Lock()
        self._seq = 0
        self._pending: list[SafeFillerAdmissionRequest] = []
        self._in_flight: dict[str, SafeFillerAdmissionRequest] = {}

    def _decision_id(self, seq: int, meta: dict[str, Any], request_id: str) -> str:
        gate_meta = self._gate_meta()
        parts = [
            str(meta.get("harness") or "harness"),
            str(meta.get("pressure_level") or "pressure"),
            str(meta.get("mode") or "mode"),
            str(gate_meta.get("session_id") or meta.get("session_id") or "session"),
            str(gate_meta.get("tool_wait_step") or meta.get("tool_wait_step") or "step"),
            str(request_id),
            str(seq),
        ]
        return "sjf-" + "-".join(part.replace(" ", "_").replace("/", "_") for part in parts)

    def _gate_release_event(self) -> asyncio.Event | None:
        release_event = self.gate_state.get("release_event")
        return release_event if isinstance(release_event, asyncio.Event) else None

    def _gate_meta(self) -> dict[str, Any]:
        meta = self.gate_state.get("meta")
        return meta if isinstance(meta, dict) else {}

    def _write(self, row: dict[str, Any]) -> None:
        if self.trace is None:
            return
        if os.environ.get("TRACE_CONTROLLER_DECISIONS", "1").strip().lower() in {"0", "false", "no", "off"}:
            return
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        row.setdefault("ts_ns", time.time_ns())
        row.setdefault("pid", os.getpid())
        with self.trace.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _base_row(self, request: SafeFillerAdmissionRequest, *, event: str) -> dict[str, Any]:
        meta = request.meta
        gate_meta = self._gate_meta()
        replay_due_offset_ms = float(gate_meta.get("replay_due_offset_ms") or 0)
        now = float(self.now_ms())
        time_until_replay_ms = replay_due_offset_ms - now
        available_window_ms = time_until_replay_ms - self.safety_margin_ms
        expected_finish_offset_ms = now + float(request.estimated_runtime_ms)
        expected_overshoot_ms = expected_finish_offset_ms - replay_due_offset_ms
        return {
            "event": event,
            "decision_id": request.decision_id,
            "session_id": meta.get("session_id", ""),
            "mode": meta.get("mode", ""),
            "harness": meta.get("harness", ""),
            "pressure_level": meta.get("pressure_level", ""),
            "phase": meta.get("phase", ""),
            "request_id": request.request_id,
            "label": request.request_id,
            "task_index": meta.get("task_index", ""),
            "tool_wait_step": meta.get("tool_wait_step", ""),
            "task_replay_steps": meta.get("task_replay_steps", ""),
            "stage": request.stage,
            "admission_seq": request.seq,
            "raw_estimated_runtime_ms": request.raw_estimated_runtime_ms,
            "estimated_runtime_ms": request.estimated_runtime_ms,
            "calibrated_runtime_ms": request.estimated_runtime_ms,
            "runtime_class": request.runtime_class,
            "calibration_source": request.calibration_source,
            "calibration_sample_count": request.calibration_sample_count,
            "calibration_quantile": request.calibration_quantile,
            "calibration_floor_ms": request.calibration_floor_ms,
            "safety_margin_ms": self.safety_margin_ms,
            "max_in_flight": self.max_in_flight,
            "require_fit_before_replay": self.require_fit_before_replay,
            "idle_override": self.idle_override,
            "time_until_next_replay_ms": round(replay_due_offset_ms - now, 3),
            "next_replay_due_offset_ms": round(replay_due_offset_ms, 3),
            "target_replay_due_offset_ms": round(replay_due_offset_ms, 3),
            "target_session_id": gate_meta.get("session_id", ""),
            "target_tool_wait_step": gate_meta.get("tool_wait_step", ""),
            "available_window_ms": round(available_window_ms, 3),
            "expected_finish_offset_ms": round(expected_finish_offset_ms, 3),
            "expected_overshoot_ms": round(expected_overshoot_ms, 3),
            "expected_fit_before_replay": request.estimated_runtime_ms + self.safety_margin_ms < time_until_replay_ms,
            "gate_owner_session_id": gate_meta.get("session_id", ""),
            "gate_tool_wait_step": gate_meta.get("tool_wait_step", ""),
            "gate_open_offset_ms": gate_meta.get("open_offset_ms", ""),
            "offset_ms": round(now, 3),
        }

    async def request(self, meta: dict[str, Any], *, request_id: str, stage: str) -> dict[str, Any]:
        if not self.gate_state.get("active"):
            return {"decision": "admit", "reason": "gate_inactive", "admitted_by_safe_sjf": False}
        release_event = self._gate_release_event()
        if release_event is None or release_event.is_set():
            return {"decision": "admit", "reason": "gate_released", "admitted_by_safe_sjf": False}

        loop = asyncio.get_running_loop()
        raw_estimated_runtime_ms = int(
            meta.get("estimated_runtime_ms")
            or self.estimate_runtime_ms(int(meta.get("prompt_tokens") or 0), int(meta.get("max_tokens") or 2))
        )
        if self.calibrate_runtime is not None:
            runtime_estimate = self.calibrate_runtime(meta, raw_estimated_runtime_ms)
        else:
            runtime_estimate = RuntimeEstimate(
                raw_estimate_ms=raw_estimated_runtime_ms,
                calibrated_estimate_ms=raw_estimated_runtime_ms,
                runtime_class="uncalibrated",
                calibration_source="raw_estimate",
            )
        async with self._lock:
            self._seq += 1
            decision_id = self._decision_id(self._seq, meta, request_id)
            req = SafeFillerAdmissionRequest(
                seq=self._seq,
                decision_id=decision_id,
                request_id=request_id,
                stage=stage,
                estimated_runtime_ms=runtime_estimate.calibrated_estimate_ms,
                raw_estimated_runtime_ms=runtime_estimate.raw_estimate_ms,
                runtime_class=runtime_estimate.runtime_class,
                calibration_source=runtime_estimate.calibration_source,
                calibration_sample_count=runtime_estimate.calibration_sample_count,
                calibration_quantile=runtime_estimate.calibration_quantile,
                calibration_floor_ms=runtime_estimate.calibration_floor_ms,
                created_offset_ms=float(self.now_ms()),
                meta=dict(meta),
                future=loop.create_future(),
            )
            self._pending.append(req)
            self._write(
                {
                    **self._base_row(req, event="m27.controller_oracle_safe_sjf.candidate_queued"),
                    "pending_count": len(self._pending),
                    "in_flight_count": len(self._in_flight),
                    "in_flight_request_ids": " ".join(sorted(self._in_flight)),
                }
            )

        # Let near-simultaneous filler requests collect so shortest-first has a real choice.
        coalesce_ms = float(os.environ.get("CONTROLLER_ORACLE_SAFE_SJF_COALESCE_MS", "5") or "5")
        if coalesce_ms > 0:
            await asyncio.sleep(coalesce_ms / 1000.0)
        await self.evaluate()

        release_task = asyncio.create_task(release_event.wait())
        try:
            done, _ = await asyncio.wait({req.future, release_task}, return_when=asyncio.FIRST_COMPLETED)
            if req.future in done:
                return req.future.result()
            async with self._lock:
                if req in self._pending:
                    self._pending.remove(req)
                if not req.future.done():
                    result = {
                        "decision": "admit",
                        "reason": "gate_released_after_target_replay",
                        "admitted_by_safe_sjf": False,
                    }
                    req.future.set_result(result)
                    self._write({**self._base_row(req, event="m27.controller_oracle_safe_sjf.released_after_target")})
                    return result
                return req.future.result()
        finally:
            release_task.cancel()

    async def evaluate(self) -> None:
        async with self._lock:
            if not self._pending or not self.gate_state.get("active"):
                return
            if len(self._in_flight) >= self.max_in_flight:
                return
            gate_meta = self._gate_meta()
            replay_due_offset_ms = float(gate_meta.get("replay_due_offset_ms") or 0)
            time_until_replay_ms = replay_due_offset_ms - float(self.now_ms())
            target_session_id = str(gate_meta.get("session_id") or "")
            target_tool_wait_step = str(gate_meta.get("tool_wait_step") or "")
            available_slots = max(0, self.max_in_flight - len(self._in_flight))
            if self.require_fit_before_replay:
                safe = [
                    req
                    for req in self._pending
                    if req.estimated_runtime_ms + self.safety_margin_ms < time_until_replay_ms
                ]
            else:
                safe = list(self._pending)
            if not safe:
                if self.idle_override and not self._in_flight and available_slots > 0:
                    chosen = min(self._pending, key=lambda req: (req.estimated_runtime_ms, req.seq))
                    self._pending.remove(chosen)
                    self._in_flight[chosen.request_id] = chosen
                    decision_offset_ms = float(self.now_ms())
                    expected_finish_offset_ms = decision_offset_ms + chosen.estimated_runtime_ms
                    reason = "idle_override_shortest_filler_admitted_to_avoid_empty_backend"
                    result = {
                        "decision": "admit",
                        "reason": reason,
                        "admitted_by_safe_sjf": True,
                        "decision_id": chosen.decision_id,
                        "admission_seq": chosen.seq,
                        "idle_override": True,
                        "raw_estimated_runtime_ms": chosen.raw_estimated_runtime_ms,
                        "estimated_runtime_ms": chosen.estimated_runtime_ms,
                        "calibrated_runtime_ms": chosen.estimated_runtime_ms,
                        "runtime_class": chosen.runtime_class,
                        "calibration_source": chosen.calibration_source,
                        "calibration_sample_count": chosen.calibration_sample_count,
                        "calibration_quantile": chosen.calibration_quantile,
                        "calibration_floor_ms": chosen.calibration_floor_ms,
                        "decision_offset_ms": decision_offset_ms,
                        "target_session_id": target_session_id,
                        "target_tool_wait_step": target_tool_wait_step,
                        "target_replay_due_offset_ms": replay_due_offset_ms,
                        "available_window_ms": time_until_replay_ms - self.safety_margin_ms,
                        "time_until_next_replay_ms": time_until_replay_ms,
                        "safety_margin_ms": self.safety_margin_ms,
                        "expected_finish_offset_ms": expected_finish_offset_ms,
                        "expected_overshoot_ms": expected_finish_offset_ms - replay_due_offset_ms,
                    }
                    if not chosen.future.done():
                        chosen.future.set_result(result)
                    self._write(
                        {
                            **self._base_row(chosen, event="m27.controller_oracle_safe_sjf.decision"),
                            "decision": "admit",
                            "reason": reason,
                            "pending_count": len(self._pending),
                            "in_flight_count": len(self._in_flight),
                            "selected_rank": 1,
                            "idle_override_applied": "yes",
                        }
                    )
                    return
                for req in self._pending:
                    self._write(
                        {
                            **self._base_row(req, event="m27.controller_oracle_safe_sjf.decision"),
                            "decision": "hold",
                            "reason": "no_pending_filler_fits_before_next_replay",
                            "pending_count": len(self._pending),
                            "in_flight_count": len(self._in_flight),
                            "idle_override_applied": "no",
                        }
                    )
                return
            for rank, chosen in enumerate(
                sorted(safe, key=lambda req: (req.estimated_runtime_ms, req.seq))[:available_slots],
                start=1,
            ):
                self._pending.remove(chosen)
                self._in_flight[chosen.request_id] = chosen
                decision_offset_ms = float(self.now_ms())
                expected_finish_offset_ms = decision_offset_ms + chosen.estimated_runtime_ms
                reason = (
                    "shortest_safe_filler_fits_before_next_replay"
                    if self.require_fit_before_replay
                    else "maxfill_shortest_filler_admitted_without_fit_requirement"
                )
                result = {
                    "decision": "admit",
                    "reason": reason,
                    "admitted_by_safe_sjf": True,
                    "decision_id": chosen.decision_id,
                    "admission_seq": chosen.seq,
                    "idle_override": False,
                    "raw_estimated_runtime_ms": chosen.raw_estimated_runtime_ms,
                    "estimated_runtime_ms": chosen.estimated_runtime_ms,
                    "calibrated_runtime_ms": chosen.estimated_runtime_ms,
                    "runtime_class": chosen.runtime_class,
                    "calibration_source": chosen.calibration_source,
                    "calibration_sample_count": chosen.calibration_sample_count,
                    "calibration_quantile": chosen.calibration_quantile,
                    "calibration_floor_ms": chosen.calibration_floor_ms,
                    "decision_offset_ms": decision_offset_ms,
                    "target_session_id": target_session_id,
                    "target_tool_wait_step": target_tool_wait_step,
                    "target_replay_due_offset_ms": replay_due_offset_ms,
                    "available_window_ms": time_until_replay_ms - self.safety_margin_ms,
                    "time_until_next_replay_ms": time_until_replay_ms,
                    "safety_margin_ms": self.safety_margin_ms,
                    "expected_finish_offset_ms": expected_finish_offset_ms,
                    "expected_overshoot_ms": expected_finish_offset_ms - replay_due_offset_ms,
                }
                if not chosen.future.done():
                    chosen.future.set_result(result)
                self._write(
                    {
                        **self._base_row(chosen, event="m27.controller_oracle_safe_sjf.decision"),
                        "decision": "admit",
                        "reason": reason,
                        "pending_count": len(self._pending),
                        "in_flight_count": len(self._in_flight),
                        "selected_rank": rank,
                        "idle_override_applied": "no",
                    }
                )

    async def complete(self, request_id: str) -> None:
        async with self._lock:
            completed = self._in_flight.pop(request_id, None)
            if completed is not None:
                self._write(
                    {
                        **self._base_row(completed, event="m27.controller_oracle_safe_sjf.completed"),
                        "pending_count": len(self._pending),
                        "in_flight_count": len(self._in_flight),
                    }
                )
        await self.evaluate()
