#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code
