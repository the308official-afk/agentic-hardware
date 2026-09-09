"""Protocol-preserving adapters: replace text in place, never flatten messages."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from ..models import Document, Segment


class PayloadAdapter:
    def __init__(self, api_kind: str = "openai_chat") -> None:
        if api_kind not in {"text", "openai_chat", "anthropic", "responses"}:
            raise ValueError(f"unsupported API: {api_kind}")
        self.api_kind = api_kind

    def extract(self, payload: Any) -> Document:
        if self.api_kind == "text":
            if not isinstance(payload, str):
                raise ValueError("text input must be a string")
            return Document(payload, (Segment((), payload, "user"),), "text")
        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        segments: list[Segment] = []

        def content(value: Any, path: tuple, role: str) -> None:
            if isinstance(value, str):
                segments.append(Segment(path, value, role, role in {"user", "tool"}))
            elif isinstance(value, list):
                for i, block in enumerate(value):
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if kind in {"text", "input_text", "output_text"} and isinstance(block.get("text"), str):
                        content(block["text"], path + (i, "text"), role)
                    elif kind == "tool_result":
                        content(block.get("content"), path + (i, "content"), "tool")

        if self.api_kind in {"openai_chat", "anthropic"}:
            messages = payload.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            for i, message in enumerate(messages):
                if isinstance(message, dict):
                    content(message.get("content"), ("messages", i, "content"), str(message.get("role", "unknown")))
        else:
            inputs = payload.get("input")
            if isinstance(inputs, str):
                content(inputs, ("input",), "user")
            elif isinstance(inputs, list):
                for i, item in enumerate(inputs):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "function_call_output":
                        content(item.get("output"), ("input", i, "output"), "tool")
                    elif item.get("type", "message") == "message":
                        content(item.get("content"), ("input", i, "content"), str(item.get("role", "unknown")))
            else:
                raise ValueError("input must be a string or a list")
        return Document(payload, tuple(segments), self.api_kind)

    def replace(self, document: Document, replacements: dict[tuple, str]) -> Any:
        result = deepcopy(document.payload)
        allowed = {s.path for s in document.segments if s.eligible}
        if not replacements.keys() <= allowed:
            raise ValueError("attempt to change a protected segment")
        for path, text in replacements.items():
            if not path:
                result = text
                continue
            cursor = result
            for part in path[:-1]:
                cursor = cursor[part]
            cursor[path[-1]] = text
        return result


class TextAdapter(PayloadAdapter):
    def __init__(self) -> None:
        super().__init__("text")


class OpenAIChatAdapter(PayloadAdapter):
    def __init__(self) -> None:
        super().__init__("openai_chat")


class AnthropicAdapter(PayloadAdapter):
    def __init__(self) -> None:
        super().__init__("anthropic")


class ResponsesAdapter(PayloadAdapter):
    def __init__(self) -> None:
        super().__init__("responses")
