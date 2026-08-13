#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone7}"
TRACE="${TRACE:-${RESULT_ROOT}/direct_hooks_trace.jsonl}"
METRICS="${METRICS:-${RESULT_ROOT}/direct_hooks_metrics.jsonl}"
LOG="${LOG:-${RESULT_ROOT}/direct_hooks_server.log}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
TARGET_SESSIONS="${TARGET_SESSIONS:-2}"
FILLER_SESSIONS="${FILLER_SESSIONS:-24}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1024}"
PRESSURE_CONCURRENCY="${PRESSURE_CONCURRENCY:-1}"
HINT_TIMING="${HINT_TIMING:-near_resume}"

mkdir -p "${RESULT_ROOT}"
rm -f "${TRACE}" "${METRICS}" "${LOG}"

server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "-${server_pid}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "-${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
    server_pid=""
  fi
}
trap cleanup EXIT

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 7." >&2
  exit 1
fi

export AGENTIC_KV_TRACE_ENABLE=1
export AGENTIC_KV_TRACE_PATH="${TRACE}"
export EXTRA_SERVER_ARGS="--max-total-tokens ${MAX_TOTAL_TOKENS}"

echo "Starting traced SGLang server for Milestone 7 direct-hook probe..."
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

echo "Running direct-probe workload..."
python scripts/run_pressure_resume_workload.py \
  --base-url "${HOST_URL}/v1" \
  --model "${MODEL}" \
  --mode hint_aware \
  --hint-prefetch-timing "${HINT_TIMING}" \
  --prefetch-action direct_probe \
  --target-sessions "${TARGET_SESSIONS}" \
  --filler-sessions "${FILLER_SESSIONS}" \
  --prompt-tokens "${PROMPT_TOKENS}" \
  --concurrency "${PRESSURE_CONCURRENCY}" \
  --out "${METRICS}"

echo
python scripts/summarize_kv_trace.py --trace "${TRACE}" | head -80

echo
python scripts/build_session_cache_map.py \
  --trace "${TRACE}" \
  --out-json "${RESULT_ROOT}/session_cache_map.json" \
  --out-md "${RESULT_ROOT}/session_cache_map.md"

echo
echo "Milestone 7 artifacts:"
echo "  ${TRACE}"
echo "  ${METRICS}"
echo "  ${RESULT_ROOT}/session_cache_map.json"
echo "  ${RESULT_ROOT}/session_cache_map.md"
