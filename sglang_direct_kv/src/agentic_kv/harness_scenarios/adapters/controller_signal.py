from __future__ import annotations

from typing import Any


def to_harness_controller_meta(harness_signals: dict[str, Any]) -> dict[str, Any]:
    """Map the scenario MVP signal names onto existing controller metadata.

    The adapter is intentionally shallow: the scenario package remains backend
    neutral, while callers that use the current controller can opt into this
    conversion at the boundary.
    """

    phase = str(harness_signals.get("phase") or "request")
    meta = {
        "session_id": harness_signals.get("session_id", "scenario_session"),
        "prefix_id": harness_signals.get("prefix_id", harness_signals.get("session_id", "scenario_session")),
        "phase": phase,
        "user_waiting": harness_signals.get("user_waiting", False),
        "priority_label": "high" if float(harness_signals.get("priority", 0) or 0) >= 50 else "normal",
        "replay_likely": phase in {"tool_wait", "prepare", "replay_ready"},
        "deadline_after_tool_ms": harness_signals.get("deadline_after_ready_ms"),
        "tool_wait_ms": harness_signals.get("expected_tool_return_ms") or harness_signals.get("next_ready_eta_ms"),
    }
    if "priority" in harness_signals:
        meta["controller_sglang_priority"] = harness_signals["priority"]
    return meta
