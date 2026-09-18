#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


AUDIT_COLUMNS = [
    "session_id",
    "prefetch_request_id",
    "expected_replay_request_id",
    "tool_wait_class",
    "tool_wait_ms",
    "direct_load_started",
    "direct_load_request_submitted",
    "direct_load_completed",
    "direct_load_timed_out",
    "direct_load_contract",
    "direct_load_allowed_wait_classes",
    "direct_load_admission_reason",
    "direct_load_mechanism",
    "prepare_plan_status",
    "prepare_plan_admission",
    "prepare_plan_reason",
    "prepare_plan_host_tokens",
    "selective_admission_reason",
    "selective_admission_status",
    "selective_admission_host_tokens",
    "selective_admission_min_saved_tokens",
    "prepared_control_status",
    "prepared_control_path",
    "prepared_loaded_tokens",
    "direct_load_queue_wait_ms",
    "direct_kv_lane_policy",
    "direct_kv_lane_priority",
    "direct_kv_lane_due_sort_key",
    "direct_kv_lane_order_valid",
    "movement_before_replay_compute",
    "load_back_events_before_replay_compute",
    "h2d_copy_events_before_replay_compute",
    "target_replay_h2d_events",
    "target_replay_load_back_events",
    "controller_replay_uncached_tokens",
    "controller_replay_full_tokens",
    "controller_replay_cached_prefix_tokens",
    "baseline_replay_uncached_tokens",
    "uncached_delta_vs_baseline",
    "replay_reuse_category",
    "sglang_awareness_answer",
    "verdict",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    tmp.replace(path)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1"}


def find_controller_trace(run_dir: Path) -> Path | None:
    candidates = sorted(
        path
        for path in run_dir.glob("*/m27_trace.jsonl")
        if "controller_proactive_kv_management" in path.parent.name
    )
    return candidates[0] if candidates else None


def collect_direct_load_events(trace_path: Path | None) -> dict[str, dict[str, Any]]:
    by_source_request: dict[str, dict[str, Any]] = {}
    by_prefetch_request: dict[str, dict[str, Any]] = {}
    if trace_path is None or not trace_path.exists():
        return by_source_request

    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if (
                "targeted_kv_prefetch" not in line
                and "harness.request_" not in line
                and "prepare_prefix" not in line
            ):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_name = str(event.get("event") or "")
            command = event.get("command") if isinstance(event.get("command"), dict) else {}
            source_id = str(event.get("request_id") or command.get("request_id") or "")
            prefetch_id = str(event.get("prefetch_request_id") or "")
            if not source_id and not prefetch_id:
                continue
            if event_name.startswith("m27.targeted_kv_prefetch"):
                slot = by_source_request.setdefault(source_id, {})
                if prefetch_id:
                    slot["prefetch_request_id"] = prefetch_id
                slot["direct_load_contract"] = event.get("direct_load_contract", slot.get("direct_load_contract", ""))
                slot["direct_load_allowed_wait_classes"] = event.get(
                    "direct_load_allowed_wait_classes",
                    slot.get("direct_load_allowed_wait_classes", ""),
                )
                slot["direct_load_admission_reason"] = event.get(
                    "direct_load_admission_reason",
                    slot.get("direct_load_admission_reason", ""),
                )
                slot["direct_load_mechanism"] = event.get("direct_load_mechanism", slot.get("direct_load_mechanism", ""))
                for key in (
                    "selective_admission_reason",
                    "selective_admission_status",
                    "selective_admission_host_tokens",
                    "selective_admission_min_saved_tokens",
                ):
                    if event.get(key, "") not in ("", None):
                        slot[key] = event.get(key, "")
            else:
                slot = by_prefetch_request.setdefault(source_id, {})

            if event_name == "m27.targeted_kv_prefetch.direct_load_start":
                slot["direct_load_started"] = "yes"
                slot["direct_load_start_offset_ms"] = event.get("offset_ms", "")
            elif event_name == "m27.targeted_kv_prefetch.direct_load_end":
                slot["direct_load_completed"] = "yes"
                slot["direct_load_end_offset_ms"] = event.get("offset_ms", "")
            elif event_name == "m27.targeted_kv_prefetch.prepare_prefix_control_request":
                slot["direct_load_request_submitted"] = "yes"
                slot["prepare_control_url"] = event.get("prepare_control_url", "")
            elif event_name == "m27.targeted_kv_prefetch.prepare_prefix_control_result":
                result = event.get("result") if isinstance(event.get("result"), dict) else {}
                slot["prepared_control_status"] = result.get("status", "")
                slot["prepared_control_path"] = result.get("control_path", "")
                slot["prepared_loaded_tokens"] = result.get("loaded_tokens", "")
            elif event_name == "m27.targeted_kv_prefetch.prepare_prefix_control_plan_result":
                result = event.get("result") if isinstance(event.get("result"), dict) else {}
                slot["prepare_plan_status"] = result.get("status", "")
                slot["prepare_plan_admission"] = result.get("admission", "")
                slot["prepare_plan_reason"] = result.get("admission_reason", "")
                slot["prepare_plan_host_tokens"] = result.get("host_tokens", "")
            elif event_name == "agentic_kv.prepare_prefix.result" and not is_yes(command.get("plan_only")):
                slot["prepared_control_status"] = event.get("status", "")
                slot["prepared_control_path"] = event.get("control_path", "")
                slot["prepared_loaded_tokens"] = event.get("loaded_tokens", "")
            elif event_name == "m27.targeted_kv_prefetch.direct_load_timeout_before_replay":
                slot["direct_load_timed_out"] = "yes"
                slot["direct_load_timeout_offset_ms"] = event.get("offset_ms", "")
            elif event_name == "m27.harness.request_input" and event.get("phase") == "hint_prefetch":
                slot["direct_load_request_submitted"] = "yes"
                slot["direct_load_queue_wait_ms"] = event.get("client_queue_wait_ms", "")
                slot["direct_kv_lane_policy"] = event.get("client_submission_policy", "")
                slot["direct_kv_lane_priority"] = event.get("client_submission_priority", "")
                slot["direct_kv_lane_due_sort_key"] = event.get("direct_kv_h2d_priority_sort_key", "")
            elif event_name == "m27.harness.request_done" and event.get("phase") == "hint_prefetch":
                slot["direct_load_request_done"] = "yes"

    for source_slot in by_source_request.values():
        prefetch_id = str(source_slot.get("prefetch_request_id") or "")
        if prefetch_id and prefetch_id in by_prefetch_request:
            source_slot.update(by_prefetch_request[prefetch_id])
    return by_source_request


def collect_replay_load_events(trace_path: Path | None) -> dict[str, dict[str, int]]:
    by_request: dict[str, dict[str, int]] = {}
    if trace_path is None or not trace_path.exists():
        return by_request
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if "load_back" not in line and "host_to_device" not in line and "load_to_device" not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            request_id = str(event.get("request_id") or event.get("agent_request_id") or event.get("agent_label") or "")
            if not request_id:
                continue
            slot = by_request.setdefault(request_id, {"h2d_events": 0, "load_back_events": 0})
            event_name = str(event.get("event") or "")
            method = str(event.get("method") or "")
            category = str(event.get("category") or "")
            if event.get("direction") == "host_to_device" or category == "host_to_device_copy" or method == "load_to_device_per_layer":
                slot["h2d_events"] += 1
            if "load_back" in event_name or method in {"init_load_back", "load_back"} or category in {"init_load_back", "load_back"}:
                slot["load_back_events"] += 1
    return by_request


def classify_reuse(
    *,
    full_tokens: float | None,
    controller_uncached: float | None,
    baseline_uncached: float | None,
    movement: bool,
    completed: bool,
    timed_out: bool,
) -> tuple[str, str, str]:
    if controller_uncached is None or full_tokens in (None, 0):
        return (
            "unknown",
            "unknown: missing replay cache accounting",
            "incomplete proof: replay cache accounting was unavailable",
        )

    uncached_ratio = controller_uncached / max(1.0, float(full_tokens))
    low_uncached = controller_uncached <= 512 or uncached_ratio <= 0.10
    baseline_delta = None if baseline_uncached is None else controller_uncached - baseline_uncached
    materially_better = baseline_delta is not None and baseline_delta <= -512
    materially_worse = baseline_delta is not None and baseline_delta >= 512

    if completed and low_uncached and not materially_worse:
        return (
            "clean_replay_reuse",
            "yes: replay saw the prefix as mostly ready after a completed prefetch",
            "clean proof: completed prefetch and replay avoided most rebuild/reload work",
        )
    if movement and low_uncached and not materially_worse:
        return (
            "useful_cache_reuse",
            "partly: replay saw useful cache state, but prefetch did not cleanly complete",
            "partial proof: KV movement happened and replay avoided most uncached work",
        )
    if movement and not low_uncached:
        return (
            "movement_but_replay_rebuilt",
            "no: movement occurred, but replay still had large uncached/rebuild work",
            "failed proof: prefetch movement did not translate into replay-ready KV",
        )
    if timed_out and low_uncached:
        return (
            "timeout_but_replay_cache_hit",
            "unclear: prefetch timed out, yet replay saw useful cache state",
            "ambiguous proof: replay benefited from cache, but not from a clean completed prefetch",
        )
    if baseline_delta is not None and materially_better:
        return (
            "better_than_baseline_without_clean_proof",
            "unclear: replay improved versus baseline, but direct movement proof is weak",
            "partial proof: replay uncached tokens improved, but causality is not clean",
        )
    return (
        "no_replay_reuse",
        "no: replay did not show useful reuse from the prefetch path",
        "failed proof: no clean movement-to-replay reuse evidence",
    )


def build_audit(report_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    proof_rows = read_csv(report_dir / "targeted_kv_prefetch_proof.csv")
    readiness_rows = read_csv(report_dir / "global_kv_readiness_by_mode.csv")
    controller_trace = find_controller_trace(run_dir)
    direct_events = collect_direct_load_events(controller_trace)
    replay_load_events = collect_replay_load_events(controller_trace)

    controller_replay_by_id = {
        row.get("request_id", ""): row
        for row in readiness_rows
        if row.get("mode") == "controller_proactive_kv_management"
    }
    baseline_replay_by_key = {
        (row.get("session_id", ""), row.get("tool_wait_step", "")): row
        for row in readiness_rows
        if row.get("mode") == "no_prefetch"
    }

    due_sort_values: list[float] = []
    for event in direct_events.values():
        value = as_float(event.get("direct_kv_lane_due_sort_key"))
        if value is not None:
            due_sort_values.append(value)
    lane_order_valid = all(a <= b for a, b in zip(due_sort_values, due_sort_values[1:]))

    rows: list[dict[str, Any]] = []
    for proof in proof_rows:
        prefetch_id = proof.get("request_id", "")
        replay_id = proof.get("expected_replay_request_id", "")
        replay = controller_replay_by_id.get(replay_id, {})
        baseline = baseline_replay_by_key.get((replay.get("session_id", ""), replay.get("tool_wait_step", "")), {})
        events = direct_events.get(prefetch_id, {})

        full_tokens = as_float(replay.get("prefill_full_input_tokens"))
        controller_uncached = as_float(replay.get("prefill_uncached_token_count"))
        baseline_uncached = as_float(baseline.get("prefill_uncached_token_count"))
        uncached_delta = (
            round(controller_uncached - baseline_uncached, 3)
            if controller_uncached is not None and baseline_uncached is not None
            else ""
        )
        movement = is_yes(proof.get("first_movement_before_replay_compute"))
        completed = events.get("direct_load_completed") == "yes"
        timed_out = events.get("direct_load_timed_out") == "yes"
        prepared_loaded_tokens = as_float(events.get("prepared_loaded_tokens"))
        if prepared_loaded_tokens and prepared_loaded_tokens > 0:
            movement = True
        skipped_by_contract = str(events.get("direct_load_admission_reason") or proof.get("direct_load_admission_reason") or "").startswith(
            "skip_prefetch_"
        )
        if skipped_by_contract and not completed and not (prepared_loaded_tokens and prepared_loaded_tokens > 0):
            category, awareness, verdict = (
                "skipped_by_safety_contract",
                "not applicable: controller intentionally skipped direct KV movement",
                "clean skip: no direct KV proof is expected because the safe prefetch window was not admitted",
            )
            movement = False
        else:
            category, awareness, verdict = classify_reuse(
                full_tokens=full_tokens,
                controller_uncached=controller_uncached,
                baseline_uncached=baseline_uncached,
                movement=movement,
                completed=completed,
                timed_out=timed_out,
            )

        rows.append(
            {
                "session_id": proof.get("session_id", ""),
                "prefetch_request_id": prefetch_id,
                "expected_replay_request_id": replay_id,
                "tool_wait_class": replay.get("tool_wait_class", ""),
                "tool_wait_ms": replay.get("tool_wait_ms", ""),
                "direct_load_started": events.get("direct_load_started", "no"),
                "direct_load_request_submitted": events.get("direct_load_request_submitted", "no"),
                "direct_load_completed": events.get("direct_load_completed", "no"),
                "direct_load_timed_out": events.get("direct_load_timed_out", "no"),
                "direct_load_contract": events.get("direct_load_contract", ""),
                "direct_load_allowed_wait_classes": events.get("direct_load_allowed_wait_classes", ""),
                "direct_load_admission_reason": events.get("direct_load_admission_reason", ""),
                "direct_load_mechanism": events.get("direct_load_mechanism", proof.get("direct_load_mechanism", "")),
                "prepare_plan_status": events.get("prepare_plan_status", proof.get("prepare_plan_status", "")),
                "prepare_plan_admission": events.get("prepare_plan_admission", proof.get("prepare_plan_admission", "")),
                "prepare_plan_reason": events.get("prepare_plan_reason", proof.get("prepare_plan_reason", "")),
                "prepare_plan_host_tokens": events.get("prepare_plan_host_tokens", proof.get("prepare_plan_host_tokens", "")),
                "selective_admission_reason": events.get(
                    "selective_admission_reason",
                    proof.get("selective_admission_reason", ""),
                ),
                "selective_admission_status": events.get(
                    "selective_admission_status",
                    proof.get("selective_admission_status", ""),
                ),
                "selective_admission_host_tokens": events.get(
                    "selective_admission_host_tokens",
                    proof.get("selective_admission_host_tokens", ""),
                ),
                "selective_admission_min_saved_tokens": events.get(
                    "selective_admission_min_saved_tokens",
                    proof.get("selective_admission_min_saved_tokens", ""),
                ),
                "prepared_control_status": events.get("prepared_control_status", proof.get("prepared_control_status", "")),
                "prepared_control_path": events.get("prepared_control_path", proof.get("prepared_control_path", "")),
                "prepared_loaded_tokens": events.get("prepared_loaded_tokens", proof.get("prepared_loaded_tokens", "")),
                "direct_load_queue_wait_ms": events.get("direct_load_queue_wait_ms", ""),
                "direct_kv_lane_policy": events.get("direct_kv_lane_policy", ""),
                "direct_kv_lane_priority": events.get("direct_kv_lane_priority", ""),
                "direct_kv_lane_due_sort_key": events.get("direct_kv_lane_due_sort_key", ""),
                "direct_kv_lane_order_valid": "yes" if lane_order_valid else "no",
                "movement_before_replay_compute": proof.get("first_movement_before_replay_compute", ""),
                "load_back_events_before_replay_compute": proof.get("load_back_events_before_replay_compute", ""),
                "h2d_copy_events_before_replay_compute": proof.get("h2d_copy_events_before_replay_compute", ""),
                "target_replay_h2d_events": replay_load_events.get(replay_id, {}).get("h2d_events", 0),
                "target_replay_load_back_events": replay_load_events.get(replay_id, {}).get("load_back_events", 0),
                "controller_replay_uncached_tokens": replay.get("prefill_uncached_token_count", ""),
                "controller_replay_full_tokens": replay.get("prefill_full_input_tokens", ""),
                "controller_replay_cached_prefix_tokens": replay.get("prefill_cached_prefix_tokens", ""),
                "baseline_replay_uncached_tokens": baseline.get("prefill_uncached_token_count", ""),
                "uncached_delta_vs_baseline": uncached_delta,
                "replay_reuse_category": category,
                "sglang_awareness_answer": awareness,
                "verdict": verdict,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether proactive KV prefetch became replay-visible cache state.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = build_audit(args.report_dir, args.run_dir)
    write_csv(args.out, rows, AUDIT_COLUMNS)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("replay_reuse_category") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {args.out}")
    print(json.dumps({"rows": len(rows), "replay_reuse_category_counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
