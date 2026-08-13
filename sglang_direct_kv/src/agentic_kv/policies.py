from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from .agent_trace import AgentSession, generate_sessions
from .hints import PrefetchHint
from .instrumentation import ProbeOnlySGLangKVAdapter
from .metrics import MetricsWriter
from .sglang_client import SGLangClient


class BasePolicy(ABC):
    def __init__(self, mode: str, config: dict[str, Any]) -> None:
        self.mode = mode
        self.config = config
        self.sessions = generate_sessions(config)
        self.semaphore = asyncio.Semaphore(int(config["workload"]["max_concurrency"]))

    async def run(self, client: SGLangClient, metrics: MetricsWriter) -> None:
        await asyncio.gather(
            *(self._run_one(client, metrics, session) for session in self.sessions)
        )

    async def _run_one(
        self,
        client: SGLangClient,
        metrics: MetricsWriter,
        session: AgentSession,
    ) -> None:
        async with self.semaphore:
            prefetch_task = asyncio.create_task(self.before_tool_return(session))
            await asyncio.sleep(session.tool_wait_ms / 1000.0)
            prefetch_attempted, prefetch_success = await prefetch_task
            timing = await client.complete_streaming(
                prompt=self.resume_prompt(session),
                max_tokens=int(self.config["generation"]["max_tokens"]),
                temperature=float(self.config["generation"]["temperature"]),
            )
            metrics.write(
                {
                    "session_id": session.session_id,
                    "mode": self.mode,
                    "priority": session.priority,
                    "tool_wait_ms": session.tool_wait_ms,
                    "ttft_ms": round(timing.ttft_ms, 3),
                    "total_latency_ms": round(timing.total_latency_ms, 3),
                    "prefetch_attempted": prefetch_attempted,
                    "prefetch_success": prefetch_success,
                }
            )

    def resume_prompt(self, session: AgentSession) -> str:
        return (
            session.prompt
            + "\nTool result: pytest returned a failing assertion in test_agent.py.\n"
            + "Decide the next concrete debugging step."
        )

    @abstractmethod
    async def before_tool_return(self, session: AgentSession) -> tuple[bool, bool]:
        ...


class NoPrefetchPolicy(BasePolicy):
    async def before_tool_return(self, session: AgentSession) -> tuple[bool, bool]:
        return False, False


class GenericPrefetchPolicy(BasePolicy):
    async def before_tool_return(self, session: AgentSession) -> tuple[bool, bool]:
        delay_ms = int(self.config["policy"]["generic_prefetch_delay_ms"])
        await asyncio.sleep(min(delay_ms, session.tool_wait_ms) / 1000.0)
        # Placeholder until direct KV prefetch path is found.
        return True, False


class HintAwarePolicy(BasePolicy):
    def __init__(self, mode: str, config: dict[str, Any]) -> None:
        super().__init__(mode, config)
        self.adapter = ProbeOnlySGLangKVAdapter()

    async def before_tool_return(self, session: AgentSession) -> tuple[bool, bool]:
        lead_time_ms = int(self.config["policy"]["hint_prefetch_lead_time_ms"])
        sleep_ms = max(0, session.tool_wait_ms - lead_time_ms)
        await asyncio.sleep(sleep_ms / 1000.0)
        hint = PrefetchHint(
            session_id=session.session_id,
            priority=session.priority,
            expected_resume_ms=session.tool_wait_ms,
            reuse_confidence=self.reuse_confidence(session),
            protect_ms=int(self.config["policy"]["protection_window_ms"]),
        )
        self.adapter.tag_session_kv(hint)
        success = await self.adapter.prefetch_session_kv(hint)
        if success:
            self.adapter.protect_session_kv(session.session_id, hint.protect_ms)
        return True, success

    def reuse_confidence(self, session: AgentSession) -> float:
        if session.priority == "high":
            return 0.9
        if session.priority == "medium":
            return 0.7
        return 0.4


def make_policy(mode: str, config: dict[str, Any]) -> BasePolicy:
    if mode == "no_prefetch":
        return NoPrefetchPolicy(mode, config)
    if mode == "generic_prefetch":
        return GenericPrefetchPolicy(mode, config)
    if mode == "hint_aware":
        return HintAwarePolicy(mode, config)
    raise ValueError(f"unknown mode: {mode}")
