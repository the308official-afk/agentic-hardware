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
AGENTIC_KV_COPY_TELEMETRY_ENABLE="${AGENTIC_KV_COPY_TELEMETRY_ENABLE:-0}"
AGENTIC_KV_COPY_TELEMETRY_PATH="${AGENTIC_KV_COPY_TELEMETRY_PATH:-}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-}"
SGLANG_DOCKER_PULL="${SGLANG_DOCKER_PULL:-0}"
SGLANG_DOCKER_GPU_ARGS="${SGLANG_DOCKER_GPU_ARGS:---gpus all}"
SGLANG_DOCKER_EXTRA_ARGS="${SGLANG_DOCKER_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
AGENTIC_RUNTIME_TELEMETRY="${AGENTIC_RUNTIME_TELEMETRY:-0}"
AGENTIC_RUNTIME_TELEMETRY_PATH="${AGENTIC_RUNTIME_TELEMETRY_PATH:-}"
AGENTIC_RUNTIME_TELEMETRY_BACKEND="${AGENTIC_RUNTIME_TELEMETRY_BACKEND:-sglang}"

mkdir -p "$(dirname "${AGENTIC_KV_TRACE_PATH}")"
if [[ -n "${AGENTIC_KV_COPY_TELEMETRY_PATH}" ]]; then
  mkdir -p "$(dirname "${AGENTIC_KV_COPY_TELEMETRY_PATH}")"
fi
if [[ -n "${AGENTIC_RUNTIME_TELEMETRY_PATH}" ]]; then
  mkdir -p "$(dirname "${AGENTIC_RUNTIME_TELEMETRY_PATH}")"
fi
export AGENTIC_KV_TRACE_ENABLE
export AGENTIC_KV_TRACE_PATH
export AGENTIC_KV_COPY_TELEMETRY_ENABLE
export AGENTIC_KV_COPY_TELEMETRY_PATH
export AGENTIC_RUNTIME_TELEMETRY
export AGENTIC_RUNTIME_TELEMETRY_PATH
export AGENTIC_RUNTIME_TELEMETRY_BACKEND
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN_DIR="$(dirname "$(command -v nvcc)")"
  export CUDA_HOME="${CUDA_HOME:-$(cd "${CUDA_BIN_DIR}/.." && pwd)}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

launch_args=(
  "${PYTHON_BIN}" -m sglang.launch_server
  --model-path "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --enable-hierarchical-cache
  --hicache-size "${HICACHE_SIZE_GB}"
  --hicache-io-backend "${HICACHE_IO_BACKEND}"
  --hicache-mem-layout "${HICACHE_MEM_LAYOUT}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --attention-backend "${ATTENTION_BACKEND}"
  --prefill-attention-backend "${PREFILL_ATTENTION_BACKEND}"
  --decode-attention-backend "${DECODE_ATTENTION_BACKEND}"
)

if [[ -n "${CUDA_GRAPH_FLAG}" ]]; then
  # shellcheck disable=SC2206
  launch_args+=( ${CUDA_GRAPH_FLAG} )
fi
if [[ -n "${OVERLAP_FLAG}" ]]; then
  # shellcheck disable=SC2206
  launch_args+=( ${OVERLAP_FLAG} )
fi
if [[ -n "${EXTRA_SERVER_ARGS}" ]]; then
  # shellcheck disable=SC2206
  launch_args+=( ${EXTRA_SERVER_ARGS} )
fi

if [[ -n "${SGLANG_DOCKER_IMAGE}" ]]; then
  if [[ "${SGLANG_DOCKER_PULL}" == "1" ]]; then
    docker pull "${SGLANG_DOCKER_IMAGE}"
  fi
  echo "Launching SGLang in Docker image: ${SGLANG_DOCKER_IMAGE}"
  exec docker run --rm \
    ${SGLANG_DOCKER_GPU_ARGS} \
    --network host \
    --ipc host \
    --shm-size 16g \
    -v "$(pwd):$(pwd)" \
    -w "$(pwd)" \
    -e AGENTIC_KV_TRACE_ENABLE \
    -e AGENTIC_KV_TRACE_PATH \
    -e AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-0}" \
    -e AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL:-0}" \
    -e AGENTIC_KV_COPY_TELEMETRY_ENABLE \
    -e AGENTIC_KV_COPY_TELEMETRY_PATH \
    -e AGENTIC_RUNTIME_TELEMETRY \
    -e AGENTIC_RUNTIME_TELEMETRY_PATH \
    -e AGENTIC_RUNTIME_TELEMETRY_BACKEND \
    -e PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    ${SGLANG_DOCKER_EXTRA_ARGS} \
    "${SGLANG_DOCKER_IMAGE}" \
    "${launch_args[@]}"
fi

exec "${launch_args[@]}"
