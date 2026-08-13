#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
TARGET_SESSIONS="${TARGET_SESSIONS:-2}"
FILLER_LIST="${FILLER_LIST:-12 24 96 192}"
PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST:-1024 2048}"
TIMINGS="${TIMINGS:-very_early_before_pressure early_before_pressure middle_during_pressure late_after_pressure}"
PRESSURE_CONCURRENCY="${PRESSURE_CONCURRENCY:-1}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone6_design_space}"

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

run_case() {
  local mode="$1"
  local timing="$2"
  local fillers="$3"
  local prompt_tokens="$4"
  local case_id="${mode}_${timing}_f${fillers}_p${prompt_tokens}"
  local trace="${RESULT_ROOT}/${case_id}_trace.jsonl"
  local metrics="${RESULT_ROOT}/${case_id}_metrics.jsonl"
  local log="${RESULT_ROOT}/${case_id}_server.log"

  echo
  echo "==== Milestone 6 case: ${case_id} ===="
  echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
  echo "TARGET_SESSIONS=${TARGET_SESSIONS}"
  echo "FILLER_SESSIONS=${fillers}"
  echo "PROMPT_TOKENS=${prompt_tokens}"
  echo "HINT_PREFETCH_TIMING=${timing}"

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
    --hint-prefetch-timing "${timing}" \
    --target-sessions "${TARGET_SESSIONS}" \
    --filler-sessions "${fillers}" \
    --prompt-tokens "${prompt_tokens}" \
    --concurrency "${PRESSURE_CONCURRENCY}" \
    --out "${metrics}"

  python scripts/summarize_kv_trace.py --trace "${trace}" | head -35
  cleanup_server
}

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 6." >&2
  exit 1
fi

for prompt_tokens in ${PROMPT_TOKEN_LIST}; do
  for fillers in ${FILLER_LIST}; do
    run_case "no_prefetch" "late_after_pressure" "${fillers}" "${prompt_tokens}"
    for timing in ${TIMINGS}; do
      run_case "hint_aware" "${timing}" "${fillers}" "${prompt_tokens}"
    done
  done
done

echo
python scripts/summarize_design_space.py --root "${RESULT_ROOT}"
python scripts/plot_design_space.py --root "${RESULT_ROOT}"
