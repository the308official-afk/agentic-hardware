#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
LOG="${LOG:-artifacts/milestone4_pressure_server.log}"
TRACE="${AGENTIC_KV_TRACE_PATH:-artifacts/milestone4_kv_movement_trace.jsonl}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
TARGET_SESSIONS="${TARGET_SESSIONS:-2}"
FILLER_SESSIONS="${FILLER_SESSIONS:-18}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1024}"
PRESSURE_CONCURRENCY="${PRESSURE_CONCURRENCY:-1}"

mkdir -p artifacts artifacts/results
rm -f "${LOG}" "${TRACE}"

server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "-${server_pid}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "-${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running this pressure trace." >&2
  exit 1
fi

export AGENTIC_KV_TRACE_ENABLE=1
export AGENTIC_KV_TRACE_PATH="${TRACE}"
export EXTRA_SERVER_ARGS="--max-total-tokens ${MAX_TOTAL_TOKENS}"

setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${LOG}" 2>&1 &
server_pid="$!"

ready=0
for _ in $(seq 1 300); do
  if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
    echo "SGLang exited before becoming ready. Log tail:"
    tail -120 "${LOG}" || true
    exit 1
  fi
  sleep 1
done

if [[ "${ready}" != "1" ]]; then
  echo "SGLang did not become ready. Log tail:"
  tail -120 "${LOG}" || true
  exit 1
fi

echo "SGLang pressure server is ready at ${HOST_URL}"
echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"

python scripts/run_pressure_resume_workload.py \
  --base-url "${HOST_URL}/v1" \
  --model "${MODEL}" \
  --target-sessions "${TARGET_SESSIONS}" \
  --filler-sessions "${FILLER_SESSIONS}" \
  --prompt-tokens "${PROMPT_TOKENS}" \
  --concurrency "${PRESSURE_CONCURRENCY}" \
  --out artifacts/results/milestone4_pressure_resume_metrics.jsonl

echo
python scripts/summarize_kv_trace.py --trace "${TRACE}"

echo
echo "Server log tail:"
tail -80 "${LOG}" || true
