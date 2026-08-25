from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass(frozen=True)
class CompletionTiming:
    ttft_ms: float
    total_latency_ms: float
    text: str


class SGLangClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete_streaming(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionTiming:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if extra_body:
            payload.update(extra_body)
        url = f"{self.base_url}/chat/completions"
        start = time.perf_counter()
        first_token_time: Optional[float] = None
        chunks: list[str] = []

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks.append(data)

        end = time.perf_counter()
        if first_token_time is None:
            first_token_time = end
        return CompletionTiming(
            ttft_ms=(first_token_time - start) * 1000.0,
            total_latency_ms=(end - start) * 1000.0,
            text="\n".join(chunks),
        )
