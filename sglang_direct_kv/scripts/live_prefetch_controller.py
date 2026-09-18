#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


STOP = False


def handle_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global STOP
    STOP = True


def read_new_lines(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows, handle.tell()


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts", time.time())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def append_marker_to_payload(payload: dict[str, Any], hint: dict[str, Any], max_tokens: int, action: str) -> dict[str, Any]:
    raise RuntimeError(
        "Legacy request-triggered live KV prefetch has been removed. "
        "Use prepared_prefix_control so SGLang/HiCache performs the host-to-device load directly."
    )


def post_prefetch(target_base: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, str, str]:
    url = f"{target_base.rstrip('/')}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            text = response.read().decode("utf-8", errors="replace")
            return response.status, text[:500], ""
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, text[:500], f"HTTPError: {exc}"
    except Exception as exc:  # noqa: BLE001
        return 0, "", f"{type(exc).__name__}: {exc}"


def handle_hint(args: argparse.Namespace, hint: dict[str, Any], seen: set[str]) -> None:
    hint_id = str(hint.get("hint_id") or "")
    if not hint_id or hint_id in seen:
        return
    seen.add(hint_id)
    payload_path = Path(str(hint.get("payload_path") or ""))
    start_ts = time.time()
    write_jsonl(
        args.controller_log,
        {
            "event": "live_prefetch.start",
            "prefetch_action": args.action,
            "hint_id": hint_id,
            "source_proxy_ordinal": hint.get("source_proxy_ordinal"),
            "source_request_id": hint.get("source_request_id") or "",
            "source_parent_run_id": hint.get("source_parent_run_id") or "",
            "source_task_instance_id": hint.get("source_task_instance_id") or "",
            "source_phase": hint.get("source_phase") or "",
            "source_request_end_ts": hint.get("source_request_end_ts"),
            "tool_names": hint.get("tool_names") or [],
            "payload_path": str(payload_path),
        },
    )
    if not payload_path.exists():
        write_jsonl(
            args.controller_log,
            {
                "event": "live_prefetch.error",
                "hint_id": hint_id,
                "source_proxy_ordinal": hint.get("source_proxy_ordinal"),
                "error": f"missing payload_path: {payload_path}",
            },
        )
        return
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    prefetch_payload = append_marker_to_payload(payload, hint, args.max_tokens, args.action)
    status, response_preview, error = post_prefetch(args.target_base, prefetch_payload, args.request_timeout_s)
    end_ts = time.time()
    write_jsonl(
        args.controller_log,
        {
            "event": "live_prefetch.end" if not error and status < 400 else "live_prefetch.error",
            "prefetch_action": args.action,
            "direct_load_trigger_injected": args.action == "direct_load",
            "hint_id": hint_id,
            "source_proxy_ordinal": hint.get("source_proxy_ordinal"),
            "source_request_id": hint.get("source_request_id") or "",
            "source_parent_run_id": hint.get("source_parent_run_id") or "",
            "source_task_instance_id": hint.get("source_task_instance_id") or "",
            "source_phase": hint.get("source_phase") or "",
            "source_request_end_ts": hint.get("source_request_end_ts"),
            "prefetch_request_id": prefetch_payload["custom_params"]["request_context"]["request_id"],
            "status": status,
            "duration_ms": round((end_ts - start_ts) * 1000.0, 3),
            "request_start_ts": start_ts,
            "request_end_ts": end_ts,
            "error": error,
            "response_preview": response_preview,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail live AgentBench tool-call hints and issue prefetch/direct-load requests.")
    parser.add_argument("--hint-log", type=Path, required=True)
    parser.add_argument("--controller-log", type=Path, required=True)
    parser.add_argument("--target-base", default="http://127.0.0.1:30000")
    parser.add_argument("--poll-ms", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--action",
        choices=("direct_load",),
        default="direct_load",
        help="Use the direct SGLang load-back trigger. Prompt-based request warming is intentionally disabled.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--idle-exit-ms", type=int, default=0, help="Exit after this much idle time. 0 means run until signaled.")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    write_jsonl(
        args.controller_log,
        {
            "event": "live_prefetch_controller.start",
            "hint_log": str(args.hint_log),
            "target_base": args.target_base,
            "max_tokens": args.max_tokens,
            "prefetch_action": args.action,
        },
    )
    offset = 0
    seen: set[str] = set()
    last_activity = time.monotonic()
    while not STOP:
        rows, offset = read_new_lines(args.hint_log, offset)
        if rows:
            last_activity = time.monotonic()
        for row in rows:
            if row.get("event") == "live_hint.submitted":
                handle_hint(args, row, seen)
        if args.idle_exit_ms > 0 and (time.monotonic() - last_activity) * 1000.0 >= args.idle_exit_ms:
            break
        time.sleep(max(1, args.poll_ms) / 1000.0)
    write_jsonl(args.controller_log, {"event": "live_prefetch_controller.stop", "processed_hints": len(seen)})


if __name__ == "__main__":
    main()
