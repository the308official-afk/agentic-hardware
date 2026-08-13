#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
TARGET_SESSIONS="${TARGET_SESSIONS:-2}"
FILLER_SESSIONS="${FILLER_SESSIONS:-36}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1536}"
PRESSURE_CONCURRENCY="${PRESSURE_CONCURRENCY:-1}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone5}"
MODES="${MODES:-no_prefetch generic_prefetch hint_aware}"

mkdir -p artifacts "${RESULT_ROOT}"

server_pid=""

cleanup_server() {
  if [[ -n "${server_pid}" ]]; then
    kill "-${server_pid}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "-${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

wait_for_server() {
  local log="$1"
  local ready=0
  for _ in $(seq 1 300); do
    if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "SGLang exited before becoming ready. Log tail:"
      tail -120 "${log}" || true
      exit 1
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "SGLang did not become ready. Log tail:"
    tail -120 "${log}" || true
    exit 1
  fi
}

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 5." >&2
  exit 1
fi

for mode in ${MODES}; do
  echo
  echo "==== Milestone 5 mode: ${mode} ===="
  echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
  echo "TARGET_SESSIONS=${TARGET_SESSIONS}"
  echo "FILLER_SESSIONS=${FILLER_SESSIONS}"
  echo "PROMPT_TOKENS=${PROMPT_TOKENS}"
  echo "PRESSURE_CONCURRENCY=${PRESSURE_CONCURRENCY}"
  trace="${RESULT_ROOT}/${mode}_trace.jsonl"
  metrics="${RESULT_ROOT}/${mode}_metrics.jsonl"
  log="${RESULT_ROOT}/${mode}_server.log"
  rm -f "${trace}" "${metrics}" "${log}"

  export AGENTIC_KV_TRACE_ENABLE=1
  export AGENTIC_KV_TRACE_PATH="${trace}"
  export EXTRA_SERVER_ARGS="--max-total-tokens ${MAX_TOTAL_TOKENS}"

  setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${log}" 2>&1 &
  server_pid="$!"
  wait_for_server "${log}"

  python scripts/run_pressure_resume_workload.py \
    --base-url "${HOST_URL}/v1" \
    --model "${MODEL}" \
    --mode "${mode}" \
    --target-sessions "${TARGET_SESSIONS}" \
    --filler-sessions "${FILLER_SESSIONS}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --concurrency "${PRESSURE_CONCURRENCY}" \
    --out "${metrics}"

  python scripts/summarize_kv_trace.py --trace "${trace}" | head -45
  cleanup_server
done

echo
python scripts/summarize_mode_comparison.py --root "${RESULT_ROOT}"
