#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def request_context_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    custom_params = as_dict(payload.get("custom_params"))
    nvext = as_dict(payload.get("nvext"))
    extra_args = as_dict(payload.get("extra_args"))
    runtime_observability = as_dict(extra_args.get("runtime_observability"))
    return first_dict(
        custom_params.get("request_context"),
        payload.get("request_context"),
        nvext.get("request_context"),
        extra_args.get("request_context"),
        runtime_observability.get("request_context"),
    )


def agent_hints_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    custom_params = as_dict(payload.get("custom_params"))
    nvext = as_dict(payload.get("nvext"))
    extra_args = as_dict(payload.get("extra_args"))
    runtime_observability = as_dict(extra_args.get("runtime_observability"))
    return first_dict(
        custom_params.get("agent_hints"),
        payload.get("agent_hints"),
        nvext.get("agent_hints"),
        extra_args.get("agent_hints"),
        runtime_observability.get("agent_hints"),
    )


def agentic_kv_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    custom_params = as_dict(payload.get("custom_params"))
    nvext = as_dict(payload.get("nvext"))
    extra_args = as_dict(payload.get("extra_args"))
    return first_dict(
        custom_params.get("agentic_kv"),
        payload.get("agentic_kv"),
        extra_args.get("agentic_kv"),
        nvext.get("agent_context"),
    )


def response_tool_names(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    names: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.append(function["name"])
    return names


def summarize_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"json_payload": False}
    messages = payload.get("messages")
    tools = payload.get("tools")
    extra_body_keys = []
    for key in ("custom_params", "nvext", "extra_args"):
        if key in payload:
            extra_body_keys.append(key)
    request_context = request_context_from_payload(payload)
    agentic_kv = agentic_kv_from_payload(payload)
    agent_hints = agent_hints_from_payload(payload)
    return {
        "json_payload": True,
        "model": payload.get("model"),
        "message_count": len(messages) if isinstance(messages, list) else None,
        "tools_count": len(tools) if isinstance(tools, list) else 0,
        "tool_choice": payload.get("tool_choice"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "extra_body_keys": extra_body_keys,
        "request_context": request_context,
        "agentic_kv": agentic_kv,
        "agent_hints": agent_hints,
        "request_id": request_context.get("request_id") or agentic_kv.get("request_id") or "",
        "parent_run_id": request_context.get("parent_run_id") or agentic_kv.get("parent_run_id") or "",
        "task_instance_id": request_context.get("task_instance_id") or agentic_kv.get("task_id") or "",
        "phase": request_context.get("phase") or agentic_kv.get("phase") or "",
        "step_title": request_context.get("step_title") or "",
        "sequence_index": request_context.get("sequence_index", request_context.get("step_index", "")),
        "agent_session_id": agentic_kv.get("session_id") or "",
        "agent_priority": agentic_kv.get("priority") or agent_hints.get("priority") or "",
        "agent_reuse_likelihood": agentic_kv.get("reuse_likelihood") or agent_hints.get("reuse_likelihood") or "",
    }


def request_tool_names(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def summarize_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"json_response": False}
    choices = payload.get("choices")
    message = {}
    finish_reason = None
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            finish_reason = first.get("finish_reason")
            maybe_message = first.get("message")
            if isinstance(maybe_message, dict):
                message = maybe_message
    tool_calls = message.get("tool_calls")
    tool_names = response_tool_names(payload)
    return {
        "json_response": True,
        "finish_reason": finish_reason,
        "response_tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "response_tool_call_names": tool_names,
        "content_preview": str(message.get("content") or "")[:240],
    }


def strip_markdown_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def extract_tagged_tool_payload(content: str) -> str | None:
    tag_patterns = [
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        r"<tool_calls>\s*(.*?)\s*</tool_calls>",
        r"<tools>\s*(.*?)\s*</tools>",
    ]
    for pattern in tag_patterns:
        match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def iter_embedded_json_objects(content: str) -> list[str]:
    objects: list[str] = []
    for start, char in enumerate(content):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(content)):
            current = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    objects.append(content[start : index + 1])
                    break
    return objects


def parse_jsonish_tool_payload(content: str) -> Any:
    candidates = []
    tagged = extract_tagged_tool_payload(content)
    if tagged:
        candidates.append(tagged)
    candidates.append(strip_markdown_json_fence(content))
    candidates.extend(iter_embedded_json_objects(content))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def normalize_tool_call_response(response_json: Any, request_json: Any) -> tuple[Any, list[str]]:
    """Convert common Qwen/Hermes text tool calls into OpenAI tool_calls.

    This is a lightweight stand-in for the parser/normalizer behavior Dynamo
    usually provides in the old AgentBench stack.
    """

    if not isinstance(response_json, dict):
        return response_json, []

    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return response_json, []

    valid_tool_names = request_tool_names(request_json)
    normalized_names: list[str] = []

    for choice_index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
            continue

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        parsed = parse_jsonish_tool_payload(content)
        if parsed is None:
            continue

        items = parsed if isinstance(parsed, list) else [parsed]
        tool_calls = []
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool_name")
            arguments = item.get("arguments") or item.get("args") or {}
            if not isinstance(name, str):
                continue
            if valid_tool_names and name not in valid_tool_names:
                continue
            if not isinstance(arguments, (dict, list)):
                arguments = {"value": arguments}
            tool_calls.append(
                {
                    "id": f"call_proxy_{int(time.time() * 1_000_000)}_{choice_index}_{item_index}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                    },
                }
            )
            normalized_names.append(name)

        if tool_calls:
            message["content"] = ""
            message["tool_calls"] = tool_calls
            choice["finish_reason"] = "tool_calls"

    return response_json, normalized_names


def make_handler(target_base: str, log_path: Path, normalize_tool_calls: bool):
    class ProxyHandler(BaseHTTPRequestHandler):
        server_version = "OpenAIProxyLogger/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(fmt % args, file=sys.stderr)

        def _write_log(self, row: dict[str, Any]) -> None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        def _proxy(self) -> None:
            started = time.perf_counter()
            started_wall = time.time()
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            request_json = None
            if body:
                try:
                    request_json = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    request_json = None

            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
            }
            target_url = f"{target_base.rstrip('/')}{self.path}"
            request = Request(target_url, data=body or None, headers=headers, method=self.command)

            status = 502
            response_body = b""
            response_headers: dict[str, str] = {"Content-Type": "application/json"}
            error = ""
            response_json = None
            normalized_tool_call_names: list[str] = []
            try:
                with urlopen(request, timeout=600) as response:
                    status = response.status
                    response_body = response.read()
                    response_headers = dict(response.headers.items())
                    try:
                        response_json = json.loads(response_body.decode("utf-8"))
                    except json.JSONDecodeError:
                        response_json = None
                    if normalize_tool_calls and response_json is not None:
                        response_json, normalized_tool_call_names = normalize_tool_call_response(
                            response_json, request_json
                        )
                        response_body = json.dumps(response_json).encode("utf-8")
            except HTTPError as exc:
                status = exc.code
                response_body = exc.read()
                response_headers = dict(exc.headers.items())
                error = f"HTTPError: {exc}"
            except Exception as exc:  # noqa: BLE001
                response_body = json.dumps({"error": str(exc)}).encode("utf-8")
                error = type(exc).__name__ + ": " + str(exc)

            elapsed_ms = (time.perf_counter() - started) * 1000
            ended_wall = time.time()
            row = {
                "ts": ended_wall,
                "request_start_ts": started_wall,
                "request_end_ts": ended_wall,
                "method": self.command,
                "path": self.path,
                "status": status,
                "elapsed_ms": round(elapsed_ms, 3),
                "error": error,
                "normalized_tool_call_count": len(normalized_tool_call_names),
                "normalized_tool_call_names": normalized_tool_call_names,
                **summarize_request(request_json),
                **summarize_response(response_json),
            }
            self._write_log(row)

            self.send_response(status)
            for key, value in response_headers.items():
                if key.lower() in {"content-length", "connection", "transfer-encoding", "content-encoding"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def do_GET(self) -> None:
            self._proxy()

        def do_POST(self) -> None:
            self._proxy()

    return ProxyHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Log and forward OpenAI-compatible HTTP requests.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=31000)
    parser.add_argument("--target-base", default="http://127.0.0.1:30000")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--normalize-tool-calls",
        action="store_true",
        help="Convert common Qwen/Hermes textual tool-call payloads into OpenAI tool_calls.",
    )
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        (args.listen_host, args.listen_port),
        make_handler(args.target_base, args.log, args.normalize_tool_calls),
    )
    print(f"proxy listening on http://{args.listen_host}:{args.listen_port} -> {args.target_base}")
    print(f"log: {args.log}")
    print(f"normalize tool calls: {args.normalize_tool_calls}")
    server.serve_forever()


if __name__ == "__main__":
    main()
