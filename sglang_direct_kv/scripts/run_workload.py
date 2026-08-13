#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio

from agentic_kv.agent_trace import load_config
from agentic_kv.metrics import MetricsWriter
from agentic_kv.policies import make_policy
from agentic_kv.sglang_client import SGLangClient


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=("no_prefetch", "generic_prefetch", "hint_aware"),
        required=True,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    client = SGLangClient(config["server"]["base_url"], config["server"]["model"])
    policy = make_policy(args.mode, config)
    metrics = MetricsWriter(config["output"]["results_dir"], args.mode)

    await policy.run(client, metrics)
    metrics.close()


if __name__ == "__main__":
    asyncio.run(main_async())
