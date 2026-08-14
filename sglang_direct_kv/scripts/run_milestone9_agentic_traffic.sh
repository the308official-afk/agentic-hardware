#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone9_agentic_traffic}"
MODES="${MODES:-no_prefetch request_warm direct_load oracle_direct_load}"
SESSION_COUNT="${SESSION_COUNT:-12}"
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS:-120}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-250 500 900 1600}"
PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST:-768 1024 1536}"
HINT_DELAY_MS="${HINT_DELAY_MS:-120}"
ORACLE_LEAD_MS="${ORACLE_LEAD_MS:-120}"
TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY:-8}"
RANDOMIZE_TRAFFIC="${RANDOMIZE_TRAFFIC:-0}"
RANDOM_SEED="${RANDOM_SEED:-7}"
ARRIVAL_GAP_RANGE_MS="${ARRIVAL_GAP_RANGE_MS:-60 220}"
TOOL_WAIT_RANGE_MS="${TOOL_WAIT_RANGE_MS:-250 2200}"

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

mode_count="$(count_words "${MODES}")"

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

run_mode() {
  local mode="$1"
  case_idx=$((case_idx + 1))
  local trace="${RESULT_ROOT}/${mode}_traffic_trace.jsonl"
  local metrics="${RESULT_ROOT}/${mode}_traffic_metrics.jsonl"
  local log="${RESULT_ROOT}/${mode}_server.log"
  local out_dir="${RESULT_ROOT}/${mode}_outcomes"

  echo
  echo "==== Milestone 9 traffic case [${case_idx}/${mode_count}]: ${mode} ===="
  echo "SESSION_COUNT=${SESSION_COUNT}"
  echo "ARRIVAL_GAP_MS=${ARRIVAL_GAP_MS}"
  echo "RANDOMIZE_TRAFFIC=${RANDOMIZE_TRAFFIC}"
  echo "RANDOM_SEED=${RANDOM_SEED}"
  echo "ARRIVAL_GAP_RANGE_MS=${ARRIVAL_GAP_RANGE_MS}"
  echo "TOOL_WAIT_LIST_MS=${TOOL_WAIT_LIST_MS}"
  echo "TOOL_WAIT_RANGE_MS=${TOOL_WAIT_RANGE_MS}"
  echo "PROMPT_TOKEN_LIST=${PROMPT_TOKEN_LIST}"
  echo "HINT_DELAY_MS=${HINT_DELAY_MS}"
  echo "ORACLE_LEAD_MS=${ORACLE_LEAD_MS}"

  rm -f "${trace}" "${metrics}" "${log}"
  rm -rf "${out_dir}"

  export AGENTIC_KV_TRACE_ENABLE=1
  export AGENTIC_KV_TRACE_PATH="${trace}"
  export EXTRA_SERVER_ARGS="--max-total-tokens ${MAX_TOTAL_TOKENS}"

  setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${log}" 2>&1 &
  server_pid="$!"
  wait_for_server "${log}"

  traffic_args=(
    --base-url "${HOST_URL}/v1" \
    --model "${MODEL}" \
    --mode "${mode}" \
    --session-count "${SESSION_COUNT}" \
    --arrival-gap-ms "${ARRIVAL_GAP_MS}" \
    --tool-wait-list-ms "${TOOL_WAIT_LIST_MS}" \
    --prompt-token-list "${PROMPT_TOKEN_LIST}" \
    --hint-delay-ms "${HINT_DELAY_MS}" \
    --oracle-lead-ms "${ORACLE_LEAD_MS}" \
    --concurrency "${TRAFFIC_CONCURRENCY}" \
    --out "${metrics}"
  )

  if [[ "${RANDOMIZE_TRAFFIC}" == "1" ]]; then
    traffic_args+=(
      --randomize-traffic
      --seed "${RANDOM_SEED}"
      --arrival-gap-range-ms "${ARRIVAL_GAP_RANGE_MS}"
      --tool-wait-range-ms "${TOOL_WAIT_RANGE_MS}"
    )
  fi

  python scripts/run_agentic_traffic_workload.py "${traffic_args[@]}"

  python scripts/summarize_kv_trace.py --trace "${trace}" | head -45
  python scripts/analyze_hint_outcomes.py \
    --trace "${trace}" \
    --metrics "${metrics}" \
    --out-dir "${out_dir}"

  cleanup_server
  echo "==== Completed Milestone 9 traffic case [${case_idx}/${mode_count}]: ${mode} ===="
}

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 9." >&2
  exit 1
fi

echo "Milestone 9 multi-session agentic traffic"
echo "Total cases: ${mode_count}"
echo "Modes: ${MODES}"
echo "Each mode starts a fresh SGLang server."
echo "RANDOMIZE_TRAFFIC=${RANDOMIZE_TRAFFIC}"
echo "RANDOM_SEED=${RANDOM_SEED}"

for mode in ${MODES}; do
  run_mode "${mode}"
done

echo
python scripts/summarize_agentic_traffic_results.py \
  --root "${RESULT_ROOT}" \
  --modes "${MODES}"

echo
echo "Milestone 9 outputs written under ${RESULT_ROOT}"
