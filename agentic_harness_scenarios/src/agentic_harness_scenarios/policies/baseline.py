from __future__ import annotations

from typing import Any

from ..models import Scenario, ScenarioResult


def baseline_decision(scenario: Scenario) -> ScenarioResult:
    from ..simulator import result

    state = scenario.backend_state
    kind = scenario.scenario_type
    if kind == "deadline_scheduling":
        background_ms = _num(state, "background_prefill_ms")
        deadline_ms = _num(scenario.harness_signals, "deadline_after_ready_ms")
        first_token_ms = background_ms + _num(state, "urgent_service_ms")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="FIFO runs the queued background prefill before the urgent replay.",
            metrics={
                "scheduler_wait_ms": background_ms,
                "tool_complete_to_first_token_ms": first_token_ms,
                "deadline_lateness_ms": first_token_ms - deadline_ms,
            },
            benefit_summary="Urgent replay inherits the background prefill delay.",
        )
    if kind == "proactive_kv":
        recompute = _num(state, "recompute_tokens_if_lost")
        first_token_ms = _num(state, "recompute_ms") + _num(state, "urgent_service_ms")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="Inactive KV is not prepared before replay; replay recomputes the missing context.",
            metrics={
                "recompute_tokens": recompute,
                "tool_complete_to_first_token_ms": first_token_ms,
                "deadline_lateness_ms": first_token_ms - _num(scenario.harness_signals, "deadline_after_ready_ms"),
            },
            benefit_summary="Replay pays the full recompute cost after the tool returns.",
        )
    if kind == "eviction_value":
        high_value_tokens = _num(state, "high_value_recompute_tokens")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="LRU evicts the older high-value session because it lacks future reuse context.",
            metrics={
                "recompute_tokens": high_value_tokens,
                "evicted_value_score": _num(state, "high_value_score"),
                "deadline_lateness_ms": _num(state, "high_value_recompute_ms"),
            },
            benefit_summary="The soon-returning expensive context is lost.",
        )
    if kind == "jit_headroom":
        lateness = _num(state, "background_fill_ms") + _num(state, "urgent_service_ms") - _num(scenario.harness_signals, "deadline_after_ready_ms")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="Background admission continues until replay arrives, leaving no headroom.",
            metrics={
                "reserved_headroom_ms": 0,
                "background_work_ms": _num(state, "background_work_available_ms"),
                "deadline_lateness_ms": lateness,
            },
            benefit_summary="Capacity is fully occupied at the replay boundary.",
        )
    if kind == "prefill_chunking":
        wait = _num(state, "large_chunk_ms")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="Static large prefill chunks leave the urgent replay waiting for the next boundary.",
            metrics={
                "scheduler_wait_ms": wait,
                "chunk_ms": _num(state, "large_chunk_ms"),
                "deadline_lateness_ms": wait + _num(state, "urgent_service_ms") - _num(scenario.harness_signals, "deadline_after_ready_ms"),
            },
            benefit_summary="The urgent request cannot interrupt a large background chunk.",
        )
    if kind == "admission":
        bad_admits = _num(state, "background_candidates")
        lateness = _num(state, "background_candidate_ms") * bad_admits + _num(state, "urgent_service_ms") - _num(scenario.harness_signals, "deadline_after_ready_ms")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="Static admission accepts background requests even though a replay is approaching.",
            metrics={
                "admitted_background_count": bad_admits,
                "bad_admits": bad_admits,
                "deadline_lateness_ms": lateness,
            },
            benefit_summary="Extra background work occupies the replay-critical window.",
        )
    if kind == "critical_path":
        independent_ms = _num(state, "independent_request_ms")
        critical_ms = _num(state, "critical_request_ms")
        downstream_ms = _num(state, "downstream_total_ms")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="Equal priority requests run in arrival order, so independent work runs before the branch-unlocking request.",
            metrics={
                "workflow_completion_ms": independent_ms + critical_ms + downstream_ms,
                "downstream_unblock_ms": independent_ms + critical_ms,
            },
            benefit_summary="Downstream work waits behind unrelated independent work.",
        )
    if kind == "cancellation":
        wasted = _num(state, "speculative_branch_tokens") * _num(state, "losing_branches")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="Speculative branches continue after another branch has already found the answer.",
            metrics={
                "wasted_work_tokens": wasted,
                "released_kv_tokens": 0,
            },
            benefit_summary="The backend spends tokens and KV on results that will not be used.",
        )
    if kind == "batching":
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="Urgent interactive work is batched with long flexible background work.",
            metrics={
                "urgent_p95_ttft_ms": _num(state, "mixed_batch_urgent_p95_ms"),
                "background_throughput_units": _num(state, "mixed_batch_throughput"),
            },
            benefit_summary="Batch compatibility is optimized for throughput, not urgent tail latency.",
        )
    if kind == "path_selection":
        requests = _num(state, "request_count")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="Every request uses the same expensive execution path.",
            metrics={
                "cost_proxy": requests * _num(state, "strong_path_cost"),
                "critical_quality_pass": True,
            },
            benefit_summary="Low-value work consumes the expensive path unnecessarily.",
        )
    if kind == "predictive":
        delay = _num(state, "reactive_discovery_ms") + _num(state, "reactive_kv_prepare_ms") + _num(state, "urgent_service_ms")
        return result(
            scenario,
            mode="baseline",
            policy=scenario.baseline_policy,
            decision="The backend waits until replay arrives before discovering missing KV and headroom.",
            metrics={
                "tool_complete_to_first_token_ms": delay,
                "prepared_before_replay": False,
                "deadline_lateness_ms": delay - _num(scenario.harness_signals, "deadline_after_ready_ms"),
            },
            benefit_summary="Preparation starts after the user-visible deadline clock has already begun.",
        )
    raise ValueError(f"Unknown scenario_type: {kind}")


def _num(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    return float(value)
