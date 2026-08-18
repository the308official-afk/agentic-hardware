#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def has_events(value: Any) -> bool:
    parsed = as_int(value)
    return parsed is not None and parsed > 0


def _sum_ms(*values: Any) -> float:
    total = 0.0
    for value in values:
        parsed = as_float(value)
        if parsed is not None:
            total += parsed
    return round(total, 3)


def _ratio(numerator: int | None, denominator: int | None) -> float | str:
    if numerator is None or denominator in (None, 0):
        return ""
    return round(numerator * 100.0 / denominator, 2)


def classify_replay_path(row: dict[str, Any]) -> dict[str, Any]:
    """Classify one replay gap using the strongest evidence currently available."""

    input_tokens = as_int(row.get("replay_input_tokens"))
    active_input_tokens = as_int(row.get("replay_active_input_tokens"))
    scheduler_trimmed_tokens = as_int(row.get("replay_scheduler_trimmed_tokens"))
    cached_prefix = as_int(row.get("replay_cached_prefix_tokens"))
    host_hit = as_int(row.get("replay_host_hit_tokens")) or 0
    host_load = as_int(row.get("replay_host_load_tokens")) or 0
    replay_h2d_events = as_int(row.get("replay_kv_h2d_events")) or 0
    direct_h2d_events = as_int(row.get("direct_kv_h2d_events")) or 0
    new_prefill = as_int(row.get("replay_new_prefill_tokens_est"))
    ttft_ms = as_float(row.get("resume_ttft_ms"))
    first_cache_delay = as_float(row.get("replay_first_cache_event_delay_ms"))
    scheduler_total = as_float(row.get("replay_scheduler_total_ms"))
    scheduler_events = as_int(row.get("replay_scheduler_event_count")) or 0
    model_forward_total = as_float(row.get("replay_model_forward_total_ms"))
    model_forward_events = as_int(row.get("replay_model_forward_event_count")) or 0
    cache_to_first = as_float(row.get("replay_cache_work_end_to_first_token_ms"))
    prefetch_margin = as_float(row.get("prefetch_margin_ms"))
    hint_host_load = as_int(row.get("hint_host_load_tokens")) or 0
    hint_host_hit = as_int(row.get("hint_host_hit_tokens")) or 0

    if input_tokens is not None and cached_prefix is not None:
        unmatched_tokens = max(0, input_tokens - cached_prefix)
    elif new_prefill is not None:
        unmatched_tokens = new_prefill
    else:
        unmatched_tokens = None

    if new_prefill is None:
        new_prefill = unmatched_tokens

    gpu_resident_hit_tokens = ""
    if cached_prefix is not None:
        gpu_resident_hit_tokens = max(0, cached_prefix - max(host_hit, host_load))

    host_load_ms = _sum_ms(
        row.get("replay_kv_h2d_duration_ms"),
        row.get("replay_hicache_load_total_ms"),
        row.get("replay_init_load_back_total_ms"),
    )
    kv_prepare_ms = _sum_ms(
        row.get("replay_match_prefix_total_ms"),
        row.get("replay_init_load_back_total_ms"),
        row.get("replay_hicache_load_total_ms"),
        row.get("replay_kv_h2d_duration_ms"),
    )

    scheduler_wait_ms: float | str = ""
    if scheduler_total is not None and scheduler_events > 0:
        scheduler_wait_ms = round(scheduler_total, 3)
    elif first_cache_delay is not None and first_cache_delay > 0:
        scheduler_wait_ms = round(first_cache_delay, 3)

    prefill_compute_ms: float | str = ""
    if ttft_ms is not None:
        known = kv_prepare_ms if isinstance(kv_prepare_ms, (int, float)) else 0.0
        if model_forward_total is not None and model_forward_events > 0:
            prefill_compute_ms = round(model_forward_total, 3)
        elif cache_to_first is not None:
            prefill_compute_ms = round(max(0.0, ttft_ms - known - max(0.0, first_cache_delay or 0.0)), 3)
        elif new_prefill is not None and new_prefill > 0:
            prefill_compute_ms = round(max(0.0, ttft_ms - known - max(0.0, first_cache_delay or 0.0)), 3)

    direct = bool(direct_h2d_events or hint_host_load or hint_host_hit)
    replay_loaded = bool(replay_h2d_events or host_load > 0)
    recompute = bool(new_prefill is not None and new_prefill >= 128)
    full_or_near_hit = bool(input_tokens and cached_prefix is not None and cached_prefix >= int(0.9 * input_tokens))
    long_ttft = bool(ttft_ms is not None and ttft_ms >= 1000)
    scheduler_delay = bool((scheduler_total is not None and scheduler_total >= 50) or (first_cache_delay is not None and first_cache_delay >= 50))

    if replay_loaded and replay_h2d_events:
        final_path = "host_to_device_kv_load"
        bottleneck = "host-load dominated" if host_load_ms >= 10 or not long_ttft else "mixed bottleneck"
        confidence = "high"
    elif replay_loaded:
        final_path = "host_cache_load_back"
        bottleneck = "host-load dominated"
        confidence = "medium"
    elif recompute and not full_or_near_hit:
        final_path = "partial_prefix_miss_recompute"
        bottleneck = "recompute dominated"
        confidence = "medium"
    elif full_or_near_hit and scheduler_delay:
        final_path = "gpu_resident_or_logical_cache_hit_waited"
        bottleneck = "scheduler dominated"
        confidence = "medium" if scheduler_events else "low"
    elif full_or_near_hit:
        final_path = "gpu_resident_or_logical_cache_hit"
        bottleneck = "GPU-resident cache hit"
        confidence = "medium"
    elif long_ttft:
        final_path = "slow_replay_without_direct_load_evidence"
        bottleneck = "scheduler/cache wait suspected"
        confidence = "low"
    else:
        final_path = "fast_or_uninstrumented_replay"
        bottleneck = "unknown / needs deeper trace"
        confidence = "low"

    if prefetch_margin is None:
        prefetch_outcome = "no prefetch"
    elif prefetch_margin < 0:
        prefetch_outcome = "prefetch late"
    elif direct and replay_loaded:
        prefetch_outcome = "prefetch ready but replay still loaded KV"
    elif direct:
        prefetch_outcome = "prefetch useful"
    elif full_or_near_hit:
        prefetch_outcome = "prefetch finished; cache already reusable"
    else:
        prefetch_outcome = "prefetch finished; no visible KV movement"

    evidence_bits = []
    if input_tokens is not None:
        evidence_bits.append(f"input={input_tokens}")
    if cached_prefix is not None:
        evidence_bits.append(f"matched={cached_prefix}")
    if active_input_tokens is not None and scheduler_trimmed_tokens is not None:
        evidence_bits.append(f"active={active_input_tokens}")
        evidence_bits.append(f"trimmed_before_later_hook={scheduler_trimmed_tokens}")
    if new_prefill is not None:
        evidence_bits.append(f"new_prefill={new_prefill}")
    evidence_bits.append(f"host_load_tokens={host_load}")
    evidence_bits.append(f"replay_h2d_events={replay_h2d_events}")
    if scheduler_events:
        evidence_bits.append(f"scheduler_events={scheduler_events}")
    if model_forward_events:
        evidence_bits.append(f"model_forward_events={model_forward_events}")
    if ttft_ms is not None:
        evidence_bits.append(f"TTFT={ttft_ms:.1f} ms")
    if first_cache_delay is not None:
        evidence_bits.append(f"first_cache_event_delay={first_cache_delay:.1f} ms")

    return {
        "input_tokens": input_tokens if input_tokens is not None else "",
        "active_input_tokens": active_input_tokens if active_input_tokens is not None else "",
        "scheduler_trimmed_tokens": scheduler_trimmed_tokens if scheduler_trimmed_tokens is not None else "",
        "matched_prefix_tokens": cached_prefix if cached_prefix is not None else "",
        "unmatched_tokens": unmatched_tokens if unmatched_tokens is not None else "",
        "cache_hit_ratio_pct": _ratio(cached_prefix, input_tokens),
        "gpu_resident_hit_tokens": gpu_resident_hit_tokens,
        "host_hit_tokens": host_hit,
        "host_load_tokens": host_load,
        "recomputed_tokens_est": new_prefill if new_prefill is not None else "",
        "scheduler_wait_ms": scheduler_wait_ms,
        "kv_prepare_ms": kv_prepare_ms,
        "host_load_ms": host_load_ms,
        "prefill_compute_ms_est": prefill_compute_ms,
        "model_forward_ms": model_forward_total if model_forward_total is not None else "",
        "final_path": final_path,
        "bottleneck_label": bottleneck,
        "confidence": confidence,
        "prefetch_outcome": prefetch_outcome,
        "evidence_summary": "; ".join(evidence_bits),
    }


def hardware_counterfactual(row: dict[str, Any]) -> dict[str, Any]:
    margin = as_float(row.get("prefetch_margin_ms"))
    tool_wait = as_float(row.get("tool_gap_ms"))
    software_duration = as_float(row.get("prefetch_duration_ms"))
    direct_copy = as_float(row.get("direct_kv_h2d_duration_ms"))
    replay_copy = as_float(row.get("replay_kv_h2d_duration_ms"))
    copy_ms = direct_copy or replay_copy

    if margin is None:
        verdict = "not applicable"
        reason = "no prefetch attempt for this row"
    elif margin >= 0:
        verdict = "already met deadline"
        reason = "software path finished before replay due"
    elif copy_ms is not None and tool_wait is not None and copy_ms <= max(tool_wait * 0.8, tool_wait - 5):
        verdict = "hardware opportunity"
        reason = "measured KV copy time appears short enough for the tool gap, but software path missed the deadline"
    elif software_duration is not None and tool_wait is not None and software_duration > tool_wait:
        verdict = "needs earlier or faster enforcement"
        reason = "full software hint path is longer than the available tool gap"
    else:
        verdict = "unclear"
        reason = "not enough direct copy evidence to estimate hardware outcome"

    return {
        "observed_software_prefetch_duration_ms": software_duration if software_duration is not None else "",
        "observed_copy_ms": copy_ms if copy_ms is not None else "",
        "available_tool_gap_ms": tool_wait if tool_wait is not None else "",
        "deadline_miss_ms": abs(margin) if margin is not None and margin < 0 else "",
        "counterfactual_verdict": verdict,
        "counterfactual_reason": reason,
    }


def build_replay_path_ledger(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for idx, row in enumerate(gaps):
        path = classify_replay_path(row)
        counterfactual = hardware_counterfactual(row)
        label = row.get("timeline_label") or f"G{idx:02d}"
        ledger.append(
            {
                "row": label,
                "session_id": row.get("session_id", ""),
                "mode": row.get("mode", ""),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", ""),
                "tool_gap_ms": row.get("tool_gap_ms", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "resume_ttft_ms": row.get("resume_ttft_ms", ""),
                "final_path": path["final_path"],
                "bottleneck_label": path["bottleneck_label"],
                "confidence": path["confidence"],
                "prefetch_outcome": path["prefetch_outcome"],
                "input_tokens": path["input_tokens"],
                "active_input_tokens": path["active_input_tokens"],
                "scheduler_trimmed_tokens": path["scheduler_trimmed_tokens"],
                "matched_prefix_tokens": path["matched_prefix_tokens"],
                "unmatched_tokens": path["unmatched_tokens"],
                "cache_hit_ratio_pct": path["cache_hit_ratio_pct"],
                "gpu_resident_hit_tokens": path["gpu_resident_hit_tokens"],
                "host_hit_tokens": path["host_hit_tokens"],
                "host_load_tokens": path["host_load_tokens"],
                "recomputed_tokens_est": path["recomputed_tokens_est"],
                "scheduler_wait_ms": path["scheduler_wait_ms"],
                "kv_prepare_ms": path["kv_prepare_ms"],
                "host_load_ms": path["host_load_ms"],
                "prefill_compute_ms_est": path["prefill_compute_ms_est"],
                "model_forward_ms": path["model_forward_ms"],
                "direct_h2d_events": row.get("direct_kv_h2d_events", ""),
                "replay_h2d_events": row.get("replay_kv_h2d_events", ""),
                "pre_replay_expected_reuse": row.get("pre_replay_expected_reuse", ""),
                "pre_replay_gpu_resident_tokens": row.get("pre_replay_gpu_resident_tokens", ""),
                "pre_replay_host_resident_tokens": row.get("pre_replay_host_resident_tokens", ""),
                "pre_replay_missing_tokens": row.get("pre_replay_missing_tokens", ""),
                "pre_replay_protected_tokens": row.get("pre_replay_protected_tokens", ""),
                "evidence_summary": path["evidence_summary"],
                **counterfactual,
            }
        )
    return ledger


def attach_replay_path_fields(row: dict[str, Any]) -> None:
    path = classify_replay_path(row)
    counterfactual = hardware_counterfactual(row)
    row.update(
        {
            "final_path": path["final_path"],
            "bottleneck_label": path["bottleneck_label"],
            "path_confidence": path["confidence"],
            "prefetch_outcome": path["prefetch_outcome"],
            "replay_active_input_tokens": path["active_input_tokens"],
            "replay_scheduler_trimmed_tokens": path["scheduler_trimmed_tokens"],
            "replay_unmatched_tokens": path["unmatched_tokens"],
            "gpu_resident_hit_tokens": path["gpu_resident_hit_tokens"],
            "recomputed_tokens_est": path["recomputed_tokens_est"],
            "scheduler_wait_ms": path["scheduler_wait_ms"],
            "kv_prepare_ms": path["kv_prepare_ms"],
            "host_load_ms": path["host_load_ms"],
            "prefill_compute_ms_est": path["prefill_compute_ms_est"],
            "model_forward_ms": path["model_forward_ms"],
            "path_evidence_summary": path["evidence_summary"],
            **counterfactual,
        }
    )


def bottleneck_summary_rows(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = Counter(str(row.get("bottleneck_label") or "unknown") for row in ledger)
    by_confidence: dict[str, Counter[str]] = {}
    for row in ledger:
        label = str(row.get("bottleneck_label") or "unknown")
        confidence = str(row.get("confidence") or "unknown")
        by_confidence.setdefault(label, Counter())[confidence] += 1
    total = len(ledger)
    rows: list[dict[str, Any]] = []
    for label, count in totals.most_common():
        ttfts = [as_float(row.get("resume_ttft_ms")) for row in ledger if row.get("bottleneck_label") == label]
        ttft_values = [value for value in ttfts if value is not None]
        rows.append(
            {
                "bottleneck_label": label,
                "gaps": count,
                "pct": round(count * 100.0 / total, 2) if total else 0.0,
                "avg_ttft_ms": round(mean(ttft_values), 3) if ttft_values else "",
                "confidence_mix": ", ".join(f"{key}:{value}" for key, value in sorted(by_confidence[label].items())),
            }
        )
    return rows


def confidence_summary_rows(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(ledger)
    counts = Counter(str(row.get("confidence") or "unknown") for row in ledger)
    return [
        {
            "confidence": label,
            "gaps": count,
            "pct": round(count * 100.0 / total, 2) if total else 0.0,
            "meaning": {
                "high": "direct SGLang counters plus replay-side CUDA/HtoD movement evidence",
                "medium": "direct SGLang cache/prefix counters, but no matching low-level HtoD event",
                "low": "mostly inferred from TTFT and timeline shape",
            }.get(label, "not classified"),
        }
        for label, count in sorted(counts.items())
    ]


def counterfactual_summary_rows(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(ledger)
    counts = Counter(str(row.get("counterfactual_verdict") or "unknown") for row in ledger)
    return [
        {
            "counterfactual_verdict": label,
            "gaps": count,
            "pct": round(count * 100.0 / total, 2) if total else 0.0,
        }
        for label, count in counts.most_common()
    ]


def instrumentation_coverage_rows(gaps: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(gaps)
    def pct(count: int) -> float:
        return round(count * 100.0 / total, 2) if total else 0.0

    checks = [
        ("request window", sum(1 for row in gaps if row.get("resume_start_ms") not in ("", None))),
        ("SGLang cache telemetry", sum(1 for row in gaps if as_int(row.get("replay_cache_event_count")) not in (None, 0))),
        ("prefix token counters", sum(1 for row in gaps if row.get("replay_cached_prefix_tokens") not in ("", None))),
        ("host load counters", sum(1 for row in gaps if row.get("replay_host_load_tokens") not in ("", None))),
        ("hint-side H2D evidence", sum(1 for row in gaps if has_events(row.get("direct_kv_h2d_events")))),
        ("replay-side H2D evidence", sum(1 for row in gaps if has_events(row.get("replay_kv_h2d_events")))),
        ("medium/high replay-path confidence", sum(1 for row in ledger if row.get("confidence") in {"medium", "high"})),
    ]
    return [
        {
            "coverage_item": label,
            "covered_gaps": count,
            "total_gaps": total,
            "coverage_pct": pct(count),
        }
        for label, count in checks
    ]
