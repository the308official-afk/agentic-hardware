#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone27_real_prompt_controlled_replay_$(date +%Y%m%d_%H%M%S)}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
WORKLOAD_JSONL="${WORKLOAD_JSONL:-}"
TRACE_INDEX_CSV="${TRACE_INDEX_CSV:-}"
MAX_PAIRS="${MAX_PAIRS:-12}"
MODES="${MODES:-no_prefetch direct_prefetch oracle_prefetch}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-100 250 500 1000}"
FILLER_LIST="${FILLER_LIST:-16 64}"
FILLER_PROMPT_TOKENS="${FILLER_PROMPT_TOKENS:-1024}"
PREFETCH_TIMING="${PREFETCH_TIMING:-near_resume}"
HINT_DELAY_MS="${HINT_DELAY_MS:-20}"
ORACLE_LEAD_MS="${ORACLE_LEAD_MS:-250}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-8}"
PREFETCH_MAX_TOKENS="${PREFETCH_MAX_TOKENS:-1}"
SERVER_READY_TIMEOUT_SECS="${SERVER_READY_TIMEOUT_SECS:-900}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-18}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
LATEST_REPORT_ROOT="$(mkdir -p "${LATEST_REPORT_ROOT}" && cd "${LATEST_REPORT_ROOT}" && pwd)"
REPORT_ROOT="${RESULT_ROOT}/controlled_replay_report"

count_words() {
  local count=0
  local item
  for item in $1; do
    count=$((count + 1))
  done
  echo "${count}"
}

mode_count="$(count_words "${MODES}")"
filler_count="$(count_words "${FILLER_LIST}")"
total_cases=$((mode_count * filler_count))
case_idx=0
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
  for _ in $(seq 1 "${SERVER_READY_TIMEOUT_SECS}"); do
    if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "SGLang exited before becoming ready. Log tail:"
      tail -160 "${log}" || true
      exit 1
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "SGLang did not become ready. Log tail:"
    tail -160 "${log}" || true
    exit 1
  fi
}

if [[ -z "${WORKLOAD_JSONL}" && -n "${TRACE_INDEX_CSV}" && -s "${TRACE_INDEX_CSV}" ]]; then
  WORKLOAD_JSONL="${RESULT_ROOT}/real_prompt_pairs.jsonl"
  "${PYTHON_BIN}" scripts/extract_agentbench_trace_replay_workload.py \
    --index-csv "${TRACE_INDEX_CSV}" \
    --out-jsonl "${WORKLOAD_JSONL}" \
    --out-csv "${RESULT_ROOT}/real_prompt_pairs.csv" \
    --max-sessions "${MAX_PAIRS}" \
    --min-gap-ms 0
fi

if [[ -z "${WORKLOAD_JSONL}" || ! -s "${WORKLOAD_JSONL}" ]]; then
  echo "WARNING: WORKLOAD_JSONL was not provided or is empty."
  echo "The driver will use fallback realistic prompts. For a manager-grade run, pass WORKLOAD_JSONL=/path/to/real_prompt_pairs.jsonl."
fi

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 27." >&2
  exit 1
fi

echo "Milestone 27: Real-Prompt Controlled Replay"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "WORKLOAD_JSONL=${WORKLOAD_JSONL:-fallback}"
echo "MODES=${MODES}"
echo "FILLER_LIST=${FILLER_LIST}"
echo "TOOL_WAIT_LIST_MS=${TOOL_WAIT_LIST_MS}"
echo "MAX_PAIRS=${MAX_PAIRS}"
echo "Total cases: ${total_cases}"

run_case() {
  local mode="$1"
  local fillers="$2"
  case_idx=$((case_idx + 1))
  local case_id="${mode}_tw${TOOL_WAIT_LIST_MS// /-}_f${fillers}"
  case_id="${case_id//[^A-Za-z0-9_.-]/_}"
  local case_root="${RESULT_ROOT}/${case_id}"
  local trace="${case_root}/m27_trace.jsonl"
  local telemetry="${case_root}/m27_copy_telemetry.jsonl"
  local metrics="${case_root}/m27_metrics.jsonl"
  local server_log="${case_root}/sglang_server.log"

  mkdir -p "${case_root}"
  rm -f "${trace}" "${telemetry}" "${metrics}" "${server_log}"

  echo
  echo "==== Milestone 27 case [${case_idx}/${total_cases}]: ${case_id} ===="
  echo "mode=${mode}"
  echo "fillers=${fillers}"

  export AGENTIC_KV_TRACE_ENABLE=1
  export AGENTIC_KV_TRACE_PATH="${trace}"
  export AGENTIC_KV_COPY_TELEMETRY_ENABLE=1
  export AGENTIC_KV_COPY_TELEMETRY_PATH="${telemetry}"
  export HICACHE_SIZE_GB
  export MEM_FRACTION_STATIC
  export EXTRA_SERVER_ARGS="${BASE_EXTRA_SERVER_ARGS} --max-total-tokens ${MAX_TOTAL_TOKENS}"

  setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${server_log}" 2>&1 &
  server_pid="$!"
  wait_for_server "${server_log}"

  driver_args=(
    scripts/run_real_prompt_controlled_replay.py
    --base-url "${HOST_URL}/v1"
    --model "${MODEL}"
    --mode "${mode}"
    --max-pairs "${MAX_PAIRS}"
    --tool-wait-list-ms "${TOOL_WAIT_LIST_MS}"
    --filler-sessions "${fillers}"
    --filler-prompt-tokens "${FILLER_PROMPT_TOKENS}"
    --prefetch-timing "${PREFETCH_TIMING}"
    --hint-delay-ms "${HINT_DELAY_MS}"
    --oracle-lead-ms "${ORACLE_LEAD_MS}"
    --max-tokens "${MAX_TOKENS}"
    --prefetch-max-tokens "${PREFETCH_MAX_TOKENS}"
    --concurrency "${REQUEST_CONCURRENCY}"
    --out "${metrics}"
  )
  if [[ -n "${WORKLOAD_JSONL}" && -s "${WORKLOAD_JSONL}" ]]; then
    driver_args+=(--workload-jsonl "${WORKLOAD_JSONL}")
  fi

  "${PYTHON_BIN}" "${driver_args[@]}" | tee "${case_root}/driver.log"
  cleanup_server
  echo "==== Completed Milestone 27 case [${case_idx}/${total_cases}]: ${case_id} ===="
}

for mode in ${MODES}; do
  for fillers in ${FILLER_LIST}; do
    run_case "${mode}" "${fillers}"
  done
done

echo
echo "Building Milestone 27 report."
"${PYTHON_BIN}" scripts/build_milestone27_controlled_replay_report.py \
  --root "${RESULT_ROOT}" \
  --out-dir "${REPORT_ROOT}" \
  --latest-root "${LATEST_REPORT_ROOT}" \
  --max-timeline-gaps "${MAX_TIMELINE_GAPS}"

echo
echo "Milestone 27 finished."
echo "Report: ${REPORT_ROOT}/controlled_replay_report.html"
echo "Latest controlled replay report: ${LATEST_REPORT_ROOT}/latest_controlled_replay_report.html"
