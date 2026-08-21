#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
TARGET_SESSIONS="${TARGET_SESSIONS:-2}"
FILLER_LIST="${FILLER_LIST:-12 24 96 192}"
PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST:-1024 1536}"
TIMINGS="${TIMINGS:-pre_pressure near_resume}"
PREFETCH_ACTIONS="${PREFETCH_ACTIONS:-direct_load}"
PRESSURE_CONCURRENCY="${PRESSURE_CONCURRENCY:-1}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone8_direct_load_design_space}"

mkdir -p artifacts "${RESULT_ROOT}"

server_pid=""
case_idx=0

count_words() {
  local count=0
  local item
  for item in $1; do
    count=$((count + 1))
  done
  echo "${count}"
}

prompt_token_count="$(count_words "${PROMPT_TOKEN_LIST}")"
filler_count="$(count_words "${FILLER_LIST}")"
timing_count="$(count_words "${TIMINGS}")"
action_count="$(count_words "${PREFETCH_ACTIONS}")"
total_cases=$((prompt_token_count * filler_count * (1 + timing_count * action_count)))

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
  local action="$3"
  local fillers="$4"
  local prompt_tokens="$5"
  case_idx=$((case_idx + 1))
  local case_id="${mode}_${action}_${timing}_f${fillers}_p${prompt_tokens}"
  local trace="${RESULT_ROOT}/${case_id}_trace.jsonl"
  local metrics="${RESULT_ROOT}/${case_id}_metrics.jsonl"
  local log="${RESULT_ROOT}/${case_id}_server.log"

  echo
  echo "==== Milestone 8 case [${case_idx}/${total_cases}]: ${case_id} ===="
  echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
  echo "TARGET_SESSIONS=${TARGET_SESSIONS}"
  echo "FILLER_SESSIONS=${fillers}"
  echo "PROMPT_TOKENS=${prompt_tokens}"
  echo "HINT_PREFETCH_TIMING=${timing}"
  echo "PREFETCH_ACTION=${action}"

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
    --prefetch-action "${action}" \
    --target-sessions "${TARGET_SESSIONS}" \
    --filler-sessions "${fillers}" \
    --prompt-tokens "${prompt_tokens}" \
    --concurrency "${PRESSURE_CONCURRENCY}" \
    --out "${metrics}"

  python scripts/summarize_kv_trace.py --trace "${trace}" | head -45
  cleanup_server
  echo "==== Completed Milestone 8 case [${case_idx}/${total_cases}]: ${case_id} ===="
}

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 8." >&2
  exit 1
fi

echo "Milestone 8 direct-load design-space sweep"
echo "Total cases: ${total_cases}"
echo "Prompt token values: ${PROMPT_TOKEN_LIST}"
echo "Filler values: ${FILLER_LIST}"
echo "Hint timings: ${TIMINGS}"
echo "Prefetch actions: ${PREFETCH_ACTIONS}"
echo "Each case starts a fresh SGLang server."

for prompt_tokens in ${PROMPT_TOKEN_LIST}; do
  for fillers in ${FILLER_LIST}; do
    run_case "no_prefetch" "near_resume" "direct_load" "${fillers}" "${prompt_tokens}"
    for timing in ${TIMINGS}; do
      for action in ${PREFETCH_ACTIONS}; do
        run_case "hint_aware" "${timing}" "${action}" "${fillers}" "${prompt_tokens}"
      done
    done
  done
done

echo
python scripts/summarize_design_space.py --root "${RESULT_ROOT}"
python scripts/plot_design_space.py --root "${RESULT_ROOT}"
