from __future__ import annotations

from typing import Any

from ..models import Scenario, ScenarioResult


def harness_aware_decision(scenario: Scenario) -> ScenarioResult:
    from ..simulator import result

    state = scenario.backend_state
    signals = scenario.harness_signals
    kind = scenario.scenario_type
    if kind == "deadline_scheduling":
        first_token_ms = _num(state, "urgent_service_ms")
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Deadline, priority, and user-waiting signals move the urgent replay ahead of background prefill.",
            metrics={
                "scheduler_wait_ms": 0,
                "tool_complete_to_first_token_ms": first_token_ms,
                "deadline_lateness_ms": first_token_ms - _num(signals, "deadline_after_ready_ms"),
            },
            benefit_summary="The backend protects the replay deadline instead of treating the queue as anonymous.",
        )
    if kind == "proactive_kv":
        first_token_ms = _num(state, "prefetch_validation_ms") + _num(state, "urgent_service_ms")
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Tool-wait ETA, reuse probability, and recompute cost trigger KV preparation before replay.",
            metrics={
                "recompute_tokens": 0,
                "tool_complete_to_first_token_ms": first_token_ms,
                "deadline_lateness_ms": first_token_ms - _num(signals, "deadline_after_ready_ms"),
            },
            benefit_summary="Useful KV is ready before the tool returns, avoiding replay-side recompute.",
        )
    if kind == "eviction_value":
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Reuse probability and recompute cost protect the high-value session and evict the low-value branch.",
            metrics={
                "recompute_tokens": _num(state, "low_value_recompute_tokens"),
                "evicted_value_score": _num(state, "low_value_score"),
                "deadline_lateness_ms": _num(state, "low_value_recompute_ms"),
            },
            benefit_summary="Memory pressure still forces eviction, but the cheaper context is lost.",
        )
    if kind == "jit_headroom":
        deadline = _num(signals, "deadline_after_ready_ms")
        first_token_ms = _num(state, "urgent_service_ms")
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="ETA and deadline open a narrow prepare window; background admission is reduced only near replay.",
            metrics={
                "reserved_headroom_ms": _num(state, "jit_headroom_ms"),
                "background_work_ms": _num(state, "background_work_available_ms") - _num(state, "jit_headroom_ms"),
                "deadline_lateness_ms": first_token_ms - deadline,
            },
            benefit_summary="The backend keeps throughput early and creates headroom only when the replay is near.",
        )
    if kind == "prefill_chunking":
        wait = _num(state, "small_chunk_ms")
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Upcoming urgent replay signal shrinks background chunks near the ETA.",
            metrics={
                "scheduler_wait_ms": wait,
                "chunk_ms": _num(state, "small_chunk_ms"),
                "deadline_lateness_ms": wait + _num(state, "urgent_service_ms") - _num(signals, "deadline_after_ready_ms"),
            },
            benefit_summary="Smaller chunks create safe interruption points for replay.",
        )
    if kind == "admission":
        safe_admits = _num(state, "safe_background_admits")
        lateness = _num(state, "background_candidate_ms") * safe_admits + _num(state, "urgent_service_ms") - _num(signals, "deadline_after_ready_ms")
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Future replay deadline holds risky background admission during the critical window.",
            metrics={
                "admitted_background_count": safe_admits,
                "bad_admits": 0,
                "deadline_lateness_ms": lateness,
            },
            benefit_summary="Admission uses future readiness, not only current free capacity.",
        )
    if kind == "critical_path":
        critical_ms = _num(state, "critical_request_ms")
        independent_ms = _num(state, "independent_request_ms")
        downstream_ms = _num(state, "downstream_total_ms")
        workflow_completion_ms = critical_ms + max(downstream_ms, independent_ms)
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Critical-path signal runs the branch-unlocking request before independent work.",
            metrics={
                "workflow_completion_ms": workflow_completion_ms,
                "downstream_unblock_ms": critical_ms,
            },
            benefit_summary="Downstream tasks start as soon as the critical request finishes.",
        )
    if kind == "cancellation":
        released = _num(state, "speculative_branch_tokens") * _num(state, "losing_branches")
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Cancelable speculative branches are dropped once their result has no value.",
            metrics={
                "wasted_work_tokens": 0,
                "released_kv_tokens": released,
            },
            benefit_summary="Optional work stops consuming backend capacity after it becomes useless.",
        )
    if kind == "batching":
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Interactive urgent work is isolated from flexible background batches.",
            metrics={
                "urgent_p95_ttft_ms": _num(state, "semantic_batch_urgent_p95_ms"),
                "background_throughput_units": _num(state, "semantic_batch_throughput"),
            },
            benefit_summary="Urgent tail latency improves while background throughput remains visible.",
        )
    if kind == "path_selection":
        critical = _num(state, "critical_request_count")
        low_value = _num(state, "low_value_request_count")
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Critical work uses the strong path; low-value work uses the cheaper path.",
            metrics={
                "cost_proxy": critical * _num(state, "strong_path_cost") + low_value * _num(state, "cheap_path_cost"),
                "critical_quality_pass": True,
            },
            benefit_summary="Execution cost drops without weakening critical user-facing work.",
        )
    if kind == "predictive":
        delay = _num(state, "prepared_first_token_ms")
        return result(
            scenario,
            mode="harness_aware",
            policy=scenario.harness_aware_policy,
            decision="Tool-wait, ETA, deadline, and KV value signals prepare capacity before replay arrives.",
            metrics={
                "tool_complete_to_first_token_ms": delay,
                "prepared_before_replay": True,
                "deadline_lateness_ms": delay - _num(signals, "deadline_after_ready_ms"),
            },
            benefit_summary="The backend shifts from post-arrival reaction to pre-arrival preparation.",
        )
    raise ValueError(f"Unknown scenario_type: {kind}")


def _num(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    return float(value)
