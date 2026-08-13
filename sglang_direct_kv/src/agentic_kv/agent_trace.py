from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AgentSession:
    session_id: str
    priority: str
    tool_wait_ms: int
    prompt: str


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_prompt(session_id: str, turn: int, target_tokens: int) -> str:
    base = (
        "You are a coding agent working on a SWE-bench style bug. "
        f"Session {session_id}, turn {turn}. "
        "You have inspected files, run tests, and need to decide the next step. "
    )
    filler = (
        "Repository context: failing unit test, stack trace, candidate files, "
        "recent edits, and tool observations. "
    )
    # Rough word-to-token approximation. Good enough for trace shaping.
    repeat_count = max(1, target_tokens // 20)
    return base + filler * repeat_count


def sample_priority(rng: random.Random, weights: dict[str, float]) -> str:
    labels = list(weights)
    values = [weights[label] for label in labels]
    return rng.choices(labels, weights=values, k=1)[0]


def generate_sessions(config: dict[str, Any]) -> list[AgentSession]:
    rng = random.Random(config["workload"].get("seed", 0))
    sessions: list[AgentSession] = []
    num_agents = int(config["workload"]["num_agents"])
    turns = int(config["workload"]["turns_per_agent"])
    agent_cfg = config["agent"]

    for agent_idx in range(num_agents):
        for turn in range(turns):
            session_id = f"agent_{agent_idx:04d}_turn_{turn:02d}"
            tool_wait_ms = rng.randint(
                int(agent_cfg["tool_wait_ms_min"]),
                int(agent_cfg["tool_wait_ms_max"]),
            )
            sessions.append(
                AgentSession(
                    session_id=session_id,
                    priority=sample_priority(rng, agent_cfg["priorities"]),
                    tool_wait_ms=tool_wait_ms,
                    prompt=make_prompt(
                        session_id,
                        turn,
                        int(agent_cfg["prompt_tokens_target"]),
                    ),
                )
            )
    rng.shuffle(sessions)
    return sessions
