from __future__ import annotations

import time
from typing import Any
from .adapters import PayloadAdapter
from .codecs import BUILTINS
from .interfaces import Codec, RequestAdapter, TokenCounter
from .models import CodecConfig, EncodingResult, digest
from .validation import validate


class PromptEncoder:
    """Stateless across requests. Dependencies are supplied once by the host.

    Budget checks are cooperative: plugins/tokenizers must bound their own work.
    No abandoned timeout threads or hidden model calls are created here.
    """

    def __init__(self, config: CodecConfig | None = None, counter: TokenCounter | None = None,
                 codec: Codec | None = None) -> None:
        self.config = config or CodecConfig()
        self.counter = counter
        if codec is None and self.config.codec not in BUILTINS:
            raise ValueError(f"unknown codec: {self.config.codec}")
        self.codec = codec or BUILTINS[self.config.codec]()
        if self.codec.name != self.config.codec:
            raise ValueError("codec/config mismatch")

    @property
    def fingerprint(self) -> str:
        return digest({"config": self.config.fingerprint,
                       "tokenizer": getattr(self.counter, "identity", ""),
                       "codec_version": getattr(self.codec, "version", self.codec.name)})[:16]

    def encode(self, payload: Any, api_kind: str = "openai_chat", *,
               budget_ms: float | None = None, adapter: RequestAdapter | None = None) -> EncodingResult:
        started_ns, started = time.time_ns(), time.perf_counter()
        original_hash = digest(payload)
        tokens = candidate_tokens = legend_tokens = None
        changed = 0
        validation = "not_run"
        limit = min(self.config.max_encode_ms, budget_ms) if budget_ms is not None else self.config.max_encode_ms

        def check() -> None:
            if (time.perf_counter() - started) * 1000 >= limit:
                raise TimeoutError("encoding budget exhausted")

        def finish(status, reason, chosen=payload):
            now = time.time_ns()
            return EncodingResult(chosen, status, reason, self.codec.name, self.fingerprint,
                original_hash, digest(chosen), started_ns, now,
                (time.perf_counter() - started) * 1000,
                getattr(self.counter, "identity", ""), tokens,
                candidate_tokens if status == "applied" else tokens,
                candidate_tokens, legend_tokens, changed if status == "applied" else 0, validation)

        if self.codec.name == "identity":
            return finish("unchanged", "identity")
        if self.counter is None:
            return finish("skipped", "tokenizer_unavailable")
        try:
            check()
            adapter = adapter or PayloadAdapter(api_kind)
            document = adapter.extract(payload)
            if sum(len(s.text) for s in document.segments) > self.config.max_input_chars:
                return finish("skipped", "input_limit")
            tokens = self.counter.count_request(payload, api_kind)
            check()
            replacements = {}
            legend_tokens = 0
            all_reversible = True
            for segment in document.segments:
                check()
                if not segment.eligible:
                    continue
                candidate = self.codec.encode(segment.text, self.config, self.counter, check)
                validate(segment.text, candidate)
                all_reversible = all_reversible and candidate.reversible
                if candidate.text != segment.text:
                    replacements[segment.path] = candidate.text
                    legend_tokens += self.counter.count_text(candidate.legend)
            validation = "round_trip" if all_reversible else "protected_content_only"
            if not replacements:
                return finish("unchanged", "no_candidate")
            result = adapter.replace(document, replacements)
            candidate_tokens = self.counter.count_request(result, api_kind)
            changed = len(replacements)
            check()
            saved = tokens - candidate_tokens
            if saved < max(1, self.config.min_saved_tokens) or saved / max(1, tokens) < self.config.min_saved_ratio:
                return finish("skipped", "insufficient_net_savings")
            return finish("applied", "validated_net_savings", result)
        except TimeoutError:
            return finish("skipped", "budget_exhausted")
        except Exception as exc:
            # Do not leak source text through exception messages in routine traces.
            return finish("failed", type(exc).__name__)
