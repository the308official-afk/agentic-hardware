#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from agentic_kv.controller import (
    BackendCapabilities,
    ControllerEvent,
    ControllerPolicy,
    ControllerStateStore,
    EventType,
    ObserveOnlyBackendAdapter,
    PolicyConfig,
)


def emit(row: dict[str, object]) -> None:
    print(json.dumps(row, sort_keys=True))


def main() -> None:
    store = ControllerStateStore()
    policy = ControllerPolicy(PolicyConfig(observe_only=True))
    backend = ObserveOnlyBackendAdapter(
        BackendCapabilities(
            priority_queue=True,
            background_prefill_budget=True,
            kv_demote=True,
            kv_prefetch=True,
            kv_release=True,
            live_metrics=True,
            observe_only=True,
            backend_name="portable-smoke",
            backend_version="0",
        )
    )

    events = [
        ControllerEvent(
            event_id="event-tool-started",
            event=EventType.TOOL_STARTED,
            session_id="demo-session",
            prefix_id="demo-session-turn-1",
            session_generation=1,
            monotonic_ms=0,
            expected_completion_ms=500,
            eta_uncertainty_ms=25,
            deadline_after_completion_ms=50,
            execution_priority=100,
            hbm_residency_priority=8,
            host_retention_priority=10,
        ),
        ControllerEvent(
            event_id="event-tool-started",
            event=EventType.TOOL_STARTED,
            session_id="demo-session",
            prefix_id="demo-session-turn-1",
            session_generation=1,
            monotonic_ms=1,
        ),
        ControllerEvent(
            event_id="event-tool-completed",
            event=EventType.TOOL_COMPLETED,
            session_id="demo-session",
            prefix_id="demo-session-turn-1",
            session_generation=1,
            monotonic_ms=502,
        ),
        ControllerEvent(
            event_id="event-session-finished",
            event=EventType.SESSION_FINISHED,
            session_id="demo-session",
            prefix_id="demo-session-turn-1",
            session_generation=1,
            monotonic_ms=650,
        ),
    ]

    for event in events:
        state, accepted = store.apply_event(event)
        decision = policy.plan(state, backend.capabilities(), now_ms=event.monotonic_ms)
        results = [backend.apply(command).to_dict() for command in decision.commands]
        emit(
            {
                "event_id": event.event_id,
                "accepted": accepted,
                "phase": state.phase.value,
                "decision": decision.to_dict(),
                "backend_results": results,
            }
        )

    assert store.get("demo-session") is not None
    assert store.get("demo-session").phase.value == "finished"
    assert len(backend.commands) == 4


if __name__ == "__main__":
    main()
