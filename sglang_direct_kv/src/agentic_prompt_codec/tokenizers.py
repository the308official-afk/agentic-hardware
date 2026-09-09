"""Optional model tokenizer. Native protocols may inject their own renderer."""
from __future__ import annotations

from typing import Any, Callable
from .models import digest


class HuggingFaceTokenCounter:
    def __init__(self, model: str, *, revision: str = "main", local_files_only: bool = True,
                 renderer: Callable[[Any, str], str] | None = None) -> None:
        from transformers import AutoTokenizer  # optional; never loaded by identity
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
                                                       local_files_only=local_files_only,
                                                       trust_remote_code=False)
        self.renderer = renderer
        self.identity = digest({"model": model, "revision": revision,
                                "vocabulary": self.tokenizer.get_vocab(),
                                "chat_template": self.tokenizer.chat_template,
                                "special_tokens": self.tokenizer.special_tokens_map})

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_request(self, payload: Any, api_kind: str) -> int:
        if self.renderer is not None:
            return self.count_text(self.renderer(payload, api_kind))
        if api_kind == "text":
            return self.count_text(payload)
        if api_kind != "openai_chat":
            raise ValueError("native API token accounting requires a matching renderer")
        # Refuse to pretend we can count images or other unsupported modalities.
        messages = payload["messages"]
        if any(not isinstance(m.get("content"), (str, type(None))) for m in messages):
            raise ValueError("multimodal token accounting requires a matching renderer")
        kwargs = {"tools": payload["tools"]} if payload.get("tools") else {}
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True,
                   add_generation_prompt=True, return_dict=False, **kwargs))
