#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

MARKER = "HARNESS_REPLAY_EXPERIMENT_JSON:"
PRIORITY_ENABLED_MODES = {
    "e2e_priority_hints",
    "pre_harness_priority_hints",
    "e2e_priority_hints_speculative_prefill",
}
PRE_HARNESS_PRIORITY_MODE = "pre_harness_priority_hints"


def write_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts_ns", time.time_ns())
    row.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def parse_b64_json(value: str) -> dict[str, Any]:
    padded = value + "=" * (-len(value) % 4)
    return as_dict(json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")))


def iter_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(iter_text(item))
        return out
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return [value["text"]]
        if isinstance(value.get("content"), str):
            return [value["content"]]
        out: list[str] = []
        for key in ("content", "input", "messages", "system"):
            if key in value:
                out.extend(iter_text(value[key]))
        return out
    return []


def strip_marker(text: str) -> str:
    return re.sub(rf"\n?{re.escape(MARKER)}[A-Za-z0-9_=-]+", "", text).strip()


def marker_from_payload(payload: Any) -> dict[str, Any]:
    joined = "\n".join(iter_text(payload))
    match = re.search(rf"{re.escape(MARKER)}([A-Za-z0-9_=-]+)", joined)
    if not match:
        return {}
    try:
        return parse_b64_json(match.group(1))
    except Exception as exc:  # noqa: BLE001
        return {"marker_parse_error": str(exc)}


def payload_text_chars(payload: Any) -> int:
    return sum(len(part) for part in iter_text(payload))


def payload_stream_requested(payload: Any) -> bool:
    return bool(as_dict(payload).get("stream"))


def payload_model(payload: Any) -> str:
    model = as_dict(payload).get("model")
    return str(model) if model is not None else ""


def request_shape(body: bytes, payload: Any, api_kind: str) -> dict[str, Any]:
    text_parts = iter_text(payload)
    joined = "\n".join(text_parts)
    return {
        "api_kind": api_kind,
        "body_size_bytes": len(body),
        "text_part_count": len(text_parts),
        "prompt_chars": payload_text_chars(payload),
        "prompt_hash": hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()[:32] if joined else "",
        "stream_requested": payload_stream_requested(payload),
        "model_requested": payload_model(payload),
    }


def is_bookkeeping_payload(payload: Any) -> bool:
    joined = "\n".join(iter_text(payload))
    bookkeeping_needles = (
        "Write the title in the predominant language of the session",
        "You are naming a coding session",
    )
    return any(needle in joined for needle in bookkeeping_needles)


def text_from_openai_chat(payload: dict[str, Any]) -> tuple[str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return "", ""
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = "\n".join(iter_text(message.get("content")))
        if not content:
            continue
        if role == "system":
            system_parts.append(strip_marker(content))
        else:
            user_parts.append(strip_marker(content))
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def text_from_anthropic_messages(payload: dict[str, Any]) -> tuple[str, str]:
    system = "\n\n".join(strip_marker(part) for part in iter_text(payload.get("system")) if strip_marker(part))
    user_parts: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = strip_marker("\n".join(iter_text(message.get("content"))))
            if content:
                user_parts.append(f"{role}: {content}")
    return system, "\n\n".join(user_parts)


def text_from_responses(payload: dict[str, Any]) -> tuple[str, str]:
    system = strip_marker(str(payload.get("instructions") or ""))
    input_value = payload.get("input")
    if isinstance(input_value, str):
        user = strip_marker(input_value)
    else:
        user = "\n\n".join(strip_marker(part) for part in iter_text(input_value) if strip_marker(part))
    return system, user


def metadata_context(meta: dict[str, Any], prompt_hash: str = "") -> dict[str, Any]:
    session_id = str(meta.get("session_id") or "harness_session")
    phase = str(meta.get("phase") or "request")
    label = str(meta.get("label") or f"{session_id}_{phase}")
    harness = str(meta.get("harness") or "unknown")
    priority_label = str(meta.get("priority_label") or ("low" if phase == "pressure_filler" else "high"))
    return {
        "experiment": "multi_harness_replay_deadline_pressure",
        "harness": harness,
        "request_role": priority_label,
        "request_id": label,
        "parent_run_id": session_id,
        "phase": phase,
        "case_id": session_id,
        "gap_id": str(meta.get("task_index") or "0"),
        "task_index": str(meta.get("task_index") or "0"),
        "correlation_id": f"{session_id}:{phase}:{label}",
        "prompt_hash": prompt_hash,
    }


def priority_enabled(mode: str) -> bool:
    return mode in PRIORITY_ENABLED_MODES


def sglang_priority(meta: dict[str, Any]) -> int | None:
    if not priority_enabled(str(meta.get("mode") or "")):
        return None
    phase = str(meta.get("phase") or "")
    if phase == "pressure_filler":
        return int(meta.get("low_priority") or -100)
    if phase == "speculative_prefill":
        return int(meta.get("speculative_prefill_priority") or 50)
    if str(meta.get("mode") or "") == PRE_HARNESS_PRIORITY_MODE:
        intent = as_dict(meta.get("priority_intent"))
        priority_class = str(intent.get("class") or "")
        if priority_class not in {"urgent", "high"}:
            return None
    return int(meta.get("high_priority") or 100)


def emitted_priority_signal(payload: dict[str, Any]) -> dict[str, str]:
    signals: list[str] = []
    sources: list[str] = []
    service_tier = payload.get("service_tier")
    if service_tier not in (None, "", [], {}):
        signals.append(f"service_tier={service_tier}")
        sources.append("service_tier")
    speed = payload.get("speed")
    if speed not in (None, "", [], {}):
        signals.append(f"speed={speed}")
        sources.append("speed")
    metadata = as_dict(payload.get("metadata"))
    metadata_priority = metadata.get("priority_class") or metadata.get("urgency")
    if metadata_priority not in (None, "", [], {}):
        signals.append(f"metadata.priority_class={metadata_priority}")
        sources.append("metadata")
    extra_body = as_dict(payload.get("extra_body"))
    agentic_hints = as_dict(extra_body.get("agentic_hints"))
    extra_priority = agentic_hints.get("priority_class") or agentic_hints.get("urgency")
    if extra_priority not in (None, "", [], {}):
        signals.append(f"extra_body.agentic_hints.priority_class={extra_priority}")
        sources.append("extra_body.agentic_hints")
    return {
        "harness_emit_priority_signal": "; ".join(signals),
        "harness_emit_priority_signal_source": ", ".join(dict.fromkeys(sources)),
    }


def priority_translation_context(meta: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    emitted = emitted_priority_signal(payload)
    priority = sglang_priority(meta)
    mode = str(meta.get("mode") or "")
    if priority is None:
        source = "none"
    elif mode == PRE_HARNESS_PRIORITY_MODE:
        if emitted["harness_emit_priority_signal"]:
            source = "harness_emitted_signal"
        elif meta.get("priority_intent"):
            source = "experiment_marker_priority_intent"
        else:
            source = "none"
    elif str(meta.get("phase") or "") == "speculative_prefill":
        source = "speculative_prefill_background_priority"
    else:
        source = "gateway_boundary_priority_mode"
    return {
        "experiment_priority_intent": compact_json(meta.get("priority_intent")),
        "harness_input_priority_signal": compact_json(meta.get("harness_input_priority_signal")),
        "harness_input_priority_signal_source": compact_json(meta.get("harness_input_priority_signal_source")),
        **emitted,
        "gateway_priority_translation": priority if priority is not None else "",
        "gateway_priority_translation_source": source,
    }


def build_sglang_payload(payload: dict[str, Any], meta: dict[str, Any], api_kind: str, model: str) -> dict[str, Any]:
    if api_kind == "anthropic":
        system, user = text_from_anthropic_messages(payload)
    elif api_kind == "responses":
        system, user = text_from_responses(payload)
    else:
        system, user = text_from_openai_chat(payload)
    prompt_text = "\n\n".join(part for part in [system, user] if part).strip() or "Continue."
    context = metadata_context(meta, prompt_hash=str(meta.get("prompt_hash") or "")[:32])
    priority = sglang_priority(meta)
    priority_chain = priority_translation_context(meta, payload)
    agent_hints = {
        "schema": "nvext.agent_hints",
        "session_id": context["parent_run_id"],
        "request_id": context["request_id"],
        "phase": context["phase"],
        "task_index": context["task_index"],
        "prompt_hash": context["prompt_hash"],
        "priority": priority if priority is not None else 0,
        "priority_label": context["request_role"],
        "expected_action": "consume_kv" if context["phase"] == "replay" else "mark_session_priority",
        "deadline_offset_ms": meta.get("deadline_offset_ms", ""),
        "kv_cache_relevant": context["phase"] in {"initial_turn", "replay"},
        "speculative_prefill": bool(meta.get("speculative_prefill")),
        "speculative_prefill_role": meta.get("speculative_prefill_role", ""),
        "speculative_prefill_strategy": meta.get("speculative_prefill_strategy", ""),
        "parent_request_id": meta.get("parent_request_id", ""),
        "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
        "experiment_priority_intent": priority_chain["experiment_priority_intent"],
        "harness_input_priority_signal": priority_chain["harness_input_priority_signal"],
        "harness_emit_priority_signal": priority_chain["harness_emit_priority_signal"],
        "gateway_priority_translation": priority_chain["gateway_priority_translation"],
        "gateway_priority_translation_source": priority_chain["gateway_priority_translation_source"],
    }
    if context["phase"] == "speculative_prefill":
        agent_hints["expected_action"] = "warm_next_turn_prefix"
        agent_hints["kv_cache_relevant"] = True
    custom_params = {
        "agentic_kv": {
            "session_id": context["parent_run_id"],
            "phase": context["phase"],
            "label": context["request_id"],
            "mode": str(meta.get("mode") or ""),
            "harness": context["harness"],
            "prompt_hash": context["prompt_hash"],
            "priority": context["request_role"],
            "task_index": context["task_index"],
            "request_id": context["request_id"],
            "parent_run_id": context["parent_run_id"],
            "correlation_id": context["correlation_id"],
            "case_id": context["case_id"],
            "gap_id": context["gap_id"],
            "dynamo_agent_priority": context["request_role"],
            "sglang_priority": priority if priority is not None else "",
            "dynamo_hint_priority": priority if priority is not None else "",
            "deadline_offset_ms": meta.get("deadline_offset_ms", ""),
            "speculative_prefill": bool(meta.get("speculative_prefill")),
            "speculative_prefill_role": meta.get("speculative_prefill_role", ""),
            "speculative_prefill_strategy": meta.get("speculative_prefill_strategy", ""),
            "parent_request_id": meta.get("parent_request_id", ""),
            "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
            "warmup_prompt_tokens": meta.get("warmup_prompt_tokens", ""),
            **priority_chain,
        },
        "request_context": context,
        "nvext": {"agent_hints": agent_hints, "request_context": context},
        "dynamo_speculative_prefill_bridge": {
            "speculative_prefill": bool(meta.get("speculative_prefill")),
            "role": meta.get("speculative_prefill_role", ""),
            "strategy": meta.get("speculative_prefill_strategy", ""),
            "parent_request_id": meta.get("parent_request_id", ""),
            "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
            "warmup_prompt_tokens": meta.get("warmup_prompt_tokens", ""),
        },
    }
    out: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": int(meta.get("max_tokens") or payload.get("max_tokens") or payload.get("max_output_tokens") or 8),
        "temperature": 0,
        "stream": True,
        "custom_params": custom_params,
    }
    if priority is not None:
        out["priority"] = priority
        out["nvext"] = {"agent_hints": agent_hints, "request_context": context}
    return out


def extract_chat_delta(line: str) -> str:
    if not line.startswith("data: "):
        return ""
    data = line.removeprefix("data: ").strip()
    if not data or data == "[DONE]":
        return ""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = as_dict(as_dict(choices[0]).get("delta"))
    value = delta.get("content")
    return value if isinstance(value, str) else ""


def call_sglang(target_base: str, payload: dict[str, Any]) -> tuple[str, float, float, int, int]:
    start = time.perf_counter()
    first = None
    text_parts: list[str] = []
    chunks = 0
    status = 502
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{target_base.rstrip('/')}/v1/chat/completions", json=payload) as response:
            status = response.status_code
            response.raise_for_status()
            for line in response.iter_lines():
                chunk = extract_chat_delta(line)
                if chunk and first is None:
                    first = time.perf_counter()
                if chunk:
                    text_parts.append(chunk)
                if line.startswith("data: ") and line.removeprefix("data: ").strip() != "[DONE]":
                    chunks += 1
    end = time.perf_counter()
    first = first if first is not None else end
    return "".join(text_parts) or "ok", (first - start) * 1000.0, (end - start) * 1000.0, chunks, status


def fake_chat_response(text: str, model: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl_harness_gateway",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode("utf-8")


def chat_sse(text: str, model: str) -> bytes:
    chunk_id = f"chatcmpl_harness_gateway_{int(time.time() * 1_000_000)}"
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    done = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (
        f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
        f"data: {json.dumps(done, separators=(',', ':'))}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def anthropic_response(text: str, model: str) -> bytes:
    return json.dumps(
        {
            "id": "msg_harness_gateway",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": max(1, len(text.split()))},
        }
    ).encode("utf-8")


def response_object(text: str, model: str) -> dict[str, Any]:
    message_id = f"msg_harness_{int(time.time() * 1_000_000)}"
    return {
        "id": f"resp_harness_{int(time.time() * 1_000_000)}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "metadata": {},
        "error": None,
        "incomplete_details": None,
        "reasoning": {"effort": None, "summary": None},
        "usage": {
            "input_tokens": 1,
            "output_tokens": max(1, len(text.split())),
            "total_tokens": 1 + max(1, len(text.split())),
            "output_tokens_details": {"reasoning_tokens": 0},
            "input_tokens_details": {"cached_tokens": 0},
        },
    }


def responses_sse(text: str, model: str) -> bytes:
    response = response_object(text, model)
    item = response["output"][0]
    part = item["content"][0]
    in_progress = dict(response)
    in_progress["status"] = "in_progress"
    in_progress["output"] = []
    events = [
        ("response.created", {"type": "response.created", "sequence_number": 0, "response": in_progress}),
        ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": {**item, "content": []}}),
        ("response.content_part.added", {"type": "response.content_part.added", "sequence_number": 2, "item_id": item["id"], "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 3, "item_id": item["id"], "output_index": 0, "content_index": 0, "delta": text}),
        ("response.output_text.done", {"type": "response.output_text.done", "sequence_number": 4, "item_id": item["id"], "output_index": 0, "content_index": 0, "text": text}),
        ("response.content_part.done", {"type": "response.content_part.done", "sequence_number": 5, "item_id": item["id"], "output_index": 0, "content_index": 0, "part": part}),
        ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 6, "output_index": 0, "item": item}),
        ("response.completed", {"type": "response.completed", "sequence_number": 7, "response": response}),
    ]
    return "".join(
        f"event: {event_name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        for event_name, data in events
    ).encode("utf-8")


def api_kind_from_path(path: str) -> str:
    if "/messages" in path:
        return "anthropic"
    if "/responses" in path:
        return "responses"
    return "chat"


def make_handler(target_base: str, trace_path: Path | None, log_path: Path | None, model: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HarnessSGLangGateway/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(fmt % args, file=sys.stderr)

        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/v1/models"):
                models = [model]
                if "sglang-qwen-coder" not in models:
                    models.append("sglang-qwen-coder")
                body = json.dumps(
                    {
                        "object": "list",
                        "data": [{"id": model_id, "object": "model"} for model_id in models],
                        "models": models,
                    }
                ).encode()
            else:
                body = json.dumps({"ok": True, "gateway": "harness_sglang_gateway"}).encode()
            self._send_bytes(200, body, "application/json")

        def do_HEAD(self) -> None:
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:
            started = time.perf_counter()
            body = self.rfile.read(int(self.headers.get("content-length", "0") or 0))
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                payload = {}
            api_kind = api_kind_from_path(self.path)
            payload_dict = as_dict(payload)
            shape = request_shape(body, payload, api_kind)
            write_jsonl(log_path, {"event": "gateway.request_received", "path": self.path, **shape})
            meta = marker_from_payload(payload)
            if is_bookkeeping_payload(payload) or not meta or meta.get("marker_parse_error"):
                write_jsonl(log_path, {"event": "gateway.unmarked_request", "path": self.path, **shape})
                if api_kind == "anthropic":
                    return self._send_bytes(200, anthropic_response("probe-ok", str(payload_dict.get("model") or model)), "application/json")
                if api_kind == "responses":
                    return self._send_bytes(200, responses_sse("probe-ok", str(payload_dict.get("model") or model)), "text/event-stream")
                if payload_stream_requested(payload):
                    return self._send_bytes(200, chat_sse("probe-ok", str(payload_dict.get("model") or model)), "text/event-stream")
                return self._send_bytes(200, fake_chat_response("probe-ok", str(payload_dict.get("model") or model)), "application/json")

            session_id = str(meta.get("session_id") or "")
            phase = str(meta.get("phase") or "")
            mode = str(meta.get("mode") or "")
            harness = str(meta.get("harness") or "")
            label = str(meta.get("label") or f"{session_id}_{phase}")
            seen_labels = getattr(self.server, "seen_request_labels", set())
            if label in seen_labels:
                write_jsonl(
                    log_path,
                    {
                        "event": "gateway.duplicate_marked_request",
                        "path": self.path,
                        "api_kind": api_kind,
                        "harness": harness,
                        "phase": phase,
                        "label": label,
                    },
                )
                if api_kind == "anthropic":
                    return self._send_bytes(200, anthropic_response("probe-ok", str(payload_dict.get("model") or model)), "application/json")
                if api_kind == "responses":
                    return self._send_bytes(200, responses_sse("probe-ok", str(payload_dict.get("model") or model)), "text/event-stream")
                if payload_stream_requested(payload):
                    return self._send_bytes(200, chat_sse("probe-ok", str(payload_dict.get("model") or model)), "text/event-stream")
                return self._send_bytes(200, fake_chat_response("probe-ok", str(payload_dict.get("model") or model)), "application/json")
            seen_labels.add(label)
            setattr(self.server, "seen_request_labels", seen_labels)
            priority = sglang_priority(meta)
            priority_chain = priority_translation_context(meta, payload_dict)
            common = {
                "session_id": session_id,
                "phase": phase,
                "mode": mode,
                "harness": harness,
                "label": label,
                "request_id": label,
                "prompt_hash": meta.get("prompt_hash", ""),
                "hint_source": "harness_gateway_intercept",
                "dynamo_agent_priority": "high" if priority is not None and priority > 0 else "low" if priority is not None else "",
                "sglang_priority": priority if priority is not None else "",
                "dynamo_hint_priority": priority if priority is not None else "",
                "deadline_offset_ms": meta.get("deadline_offset_ms", ""),
                "priority_policy": "harness_gateway_intercepted_sglang_priority" if priority is not None else "none",
                "speculative_prefill": bool(meta.get("speculative_prefill")),
                "speculative_prefill_role": meta.get("speculative_prefill_role", ""),
                "speculative_prefill_strategy": meta.get("speculative_prefill_strategy", ""),
                "parent_request_id": meta.get("parent_request_id", ""),
                "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
                "warmup_prompt_tokens": meta.get("warmup_prompt_tokens", ""),
                **priority_chain,
                **shape,
            }
            write_jsonl(trace_path, {"event": "m27.request.submitted", **common})
            write_jsonl(trace_path, {"event": "m27.request.start", **common})
            status = 502
            text = ""
            ttft_ms = 0.0
            latency_ms = 0.0
            chunks = 0
            error = ""
            try:
                sglang_payload = build_sglang_payload(as_dict(payload), meta, api_kind, model)
                write_jsonl(log_path, {"event": "gateway.forward_start", "path": self.path, **common})
                text, ttft_ms, latency_ms, chunks, status = call_sglang(target_base, sglang_payload)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                text = "gateway-error"
                latency_ms = (time.perf_counter() - started) * 1000.0
                ttft_ms = latency_ms
            write_jsonl(trace_path, {"event": "m27.request.end", **common, "ttft_ms": round(ttft_ms, 3), "total_latency_ms": round(latency_ms, 3), "stream_chunks": chunks, "status": status, "error": error})
            write_jsonl(log_path, {"event": "gateway.forwarded_request", "path": self.path, "api_kind": api_kind, **common, "ttft_ms": round(ttft_ms, 3), "total_latency_ms": round(latency_ms, 3), "stream_chunks": chunks, "status": status, "error": error})
            if error:
                return self._send_bytes(502, json.dumps({"error": error}).encode("utf-8"), "application/json")
            if api_kind == "anthropic":
                return self._send_bytes(200, anthropic_response(text, str(payload_dict.get("model") or model)), "application/json")
            if api_kind == "responses":
                return self._send_bytes(200, responses_sse(text, str(payload_dict.get("model") or model)), "text/event-stream")
            if payload_stream_requested(payload):
                return self._send_bytes(200, chat_sse(text, str(payload_dict.get("model") or model)), "text/event-stream")
            return self._send_bytes(200, fake_chat_response(text, str(payload_dict.get("model") or model)), "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Codex/Claude/Hatcher harness traffic into SGLang requests.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=31080)
    parser.add_argument("--target-base", default="http://127.0.0.1:30000")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), make_handler(args.target_base, args.trace, args.log, args.model))
    setattr(server, "seen_request_labels", set())
    print(f"harness gateway listening on http://{args.listen_host}:{args.listen_port} -> {args.target_base}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
