#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-14}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-layer_first}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
CUDA_GRAPH_FLAG="${CUDA_GRAPH_FLAG:---disable-cuda-graph}"
OVERLAP_FLAG="${OVERLAP_FLAG:---disable-overlap-schedule}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-triton}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-triton}"
AGENTIC_KV_TRACE_ENABLE="${AGENTIC_KV_TRACE_ENABLE:-1}"
AGENTIC_KV_TRACE_PATH="${AGENTIC_KV_TRACE_PATH:-artifacts/kv_movement_trace.jsonl}"

mkdir -p "$(dirname "${AGENTIC_KV_TRACE_PATH}")"
export AGENTIC_KV_TRACE_ENABLE
export AGENTIC_KV_TRACE_PATH
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN_DIR="$(dirname "$(command -v nvcc)")"
  export CUDA_HOME="${CUDA_HOME:-$(cd "${CUDA_BIN_DIR}/.." && pwd)}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-size "${HICACHE_SIZE_GB}" \
  --hicache-io-backend "${HICACHE_IO_BACKEND}" \
  --hicache-mem-layout "${HICACHE_MEM_LAYOUT}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --prefill-attention-backend "${PREFILL_ATTENTION_BACKEND}" \
  --decode-attention-backend "${DECODE_ATTENTION_BACKEND}" \
  ${CUDA_GRAPH_FLAG} \
  ${OVERLAP_FLAG}
