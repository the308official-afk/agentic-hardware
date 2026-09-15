from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentic_kv.controller.modes import (
    CONTROLLER_DEADLINE_FAIR_MODE,
    controller_mode,
    controller_safe_sjf_degree,
    controller_scheduler_priority_mode,
)
from agentic_kv.controller.sjf import SafeFillerAdmissionScheduler


async def _run_scheduler_case(
    tmp_path: Path,
    *,
    idle_override: bool,
    require_fit: bool,
    estimated_runtime_ms: int = 80,
    replay_due_ms: float = 100.0,
) -> tuple[dict[str, object], dict[str, object]]:
    trace = tmp_path / "trace.jsonl"
    gate_state = {
        "active": True,
        "release_event": asyncio.Event(),
        "meta": {"replay_due_offset_ms": replay_due_ms},
    }
    scheduler = SafeFillerAdmissionScheduler(
        trace=trace,
        now_ms=lambda: 0.0,
        gate_state=gate_state,
        safety_margin_ms=150,
        max_in_flight=1,
        require_fit_before_replay=require_fit,
        idle_override=idle_override,
        estimate_runtime_ms=lambda prompt_tokens, max_tokens: estimated_runtime_ms,
    )

    task = asyncio.create_task(scheduler.request({"mode": "test"}, request_id="filler", stage="replay"))
    await asyncio.sleep(0.02)
    if not task.done():
        gate_state["release_event"].set()
    result = await task

    events = [json.loads(line) for line in trace.read_text().splitlines()]
    decisions = [row for row in events if row.get("event") == "m27.controller_oracle_safe_sjf.decision"]
    return result, decisions[-1] if decisions else {}


def test_strict_safe_sjf_holds_risky_filler(tmp_path: Path) -> None:
    result, event = asyncio.run(_run_scheduler_case(tmp_path, idle_override=False, require_fit=True))

    assert result["reason"] == "gate_released_after_target_replay"
    assert event["decision"] == "hold"
    assert event["idle_override_applied"] == "no"


def test_idle_aware_sjf_admits_shortest_filler_when_backend_empty(tmp_path: Path) -> None:
    result, event = asyncio.run(_run_scheduler_case(tmp_path, idle_override=True, require_fit=True))

    assert result["reason"] == "idle_override_shortest_filler_admitted_to_avoid_empty_backend"
    assert event["decision"] == "admit"
    assert event["idle_override_applied"] == "yes"


def test_maxfill_sjf_does_not_require_fit_before_replay(tmp_path: Path) -> None:
    result, event = asyncio.run(_run_scheduler_case(tmp_path, idle_override=True, require_fit=False))

    assert result["reason"] == "maxfill_shortest_filler_admitted_without_fit_requirement"
    assert event["decision"] == "admit"
    assert event["idle_override_applied"] == "no"


def test_sjf_mode_degrees_are_stable() -> None:
    assert controller_safe_sjf_degree("controller_oracle_safe_sjf") == (150, 1, True, False)
    assert controller_safe_sjf_degree("controller_oracle_safe_sjf_balanced") == (75, 3, True, True)
    assert controller_safe_sjf_degree("controller_oracle_safe_sjf_aggressive") == (25, 6, True, True)
    assert controller_safe_sjf_degree("controller_oracle_safe_sjf_maxfill") == (0, 12, False, True)


def test_deadline_fair_is_scheduler_controller_mode() -> None:
    assert controller_mode(CONTROLLER_DEADLINE_FAIR_MODE)
    assert controller_scheduler_priority_mode(CONTROLLER_DEADLINE_FAIR_MODE)
