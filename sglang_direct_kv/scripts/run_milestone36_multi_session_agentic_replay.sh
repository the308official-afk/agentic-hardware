#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone36_multi_session_agentic_replay_$(date +%Y%m%d_%H%M%S)}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
WORKLOAD_JSONL="${WORKLOAD_JSONL:-}"
WORKLOAD_SOURCE="${WORKLOAD_SOURCE:-real}"
TRACE_INDEX_CSV="${TRACE_INDEX_CSV:-}"
SESSION_COUNT="${SESSION_COUNT:-16}"
MODES="${MODES:-no_prefetch direct_prefetch}"
ARRIVAL_SHAPE="${ARRIVAL_SHAPE:-staggered}"
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS:-120}"
ARRIVAL_GAP_RANGE_MS="${ARRIVAL_GAP_RANGE_MS:-60 240}"
BURST_SIZE="${BURST_SIZE:-4}"
BURST_GAP_MS="${BURST_GAP_MS:-800}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-100 250 500 1000}"
TOOL_WAIT_JITTER_MS="${TOOL_WAIT_JITTER_MS:-0}"
PREFETCH_TIMING="${PREFETCH_TIMING:-early}"
HINT_DELAY_MS="${HINT_DELAY_MS:-20}"
PREFETCH_LEAD_MS="${PREFETCH_LEAD_MS:-120}"
PRIORITY_PREFETCH_WINDOW_MS="${PRIORITY_PREFETCH_WINDOW_MS:-500}"
PRIORITY_POST_PREFETCH_QUIET_MS="${PRIORITY_POST_PREFETCH_QUIET_MS:-0}"
DEADLINE_RESERVE_WINDOW_MS="${DEADLINE_RESERVE_WINDOW_MS:-300}"
BACKGROUND_FILLERS_PER_SESSION="${BACKGROUND_FILLERS_PER_SESSION:-0}"
FILLER_PROMPT_TOKENS="${FILLER_PROMPT_TOKENS:-1024}"
TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-0}"
SYNTHETIC_PROMPT_TOKENS="${SYNTHETIC_PROMPT_TOKENS:-4096}"
SYNTHETIC_REPLAY_SUFFIX_TOKENS="${SYNTHETIC_REPLAY_SUFFIX_TOKENS:-256}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-8}"
PREFETCH_MAX_TOKENS="${PREFETCH_MAX_TOKENS:-1}"
FILLER_MAX_TOKENS="${FILLER_MAX_TOKENS:-2}"
SEED="${SEED:-0}"
SERVER_READY_TIMEOUT_SECS="${SERVER_READY_TIMEOUT_SECS:-900}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-32}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule}"
AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-1}"
AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL:-1}"

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
total_cases="${mode_count}"
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

case "${WORKLOAD_SOURCE}" in
  real|synthetic|fallback) ;;
  *)
    echo "ERROR: WORKLOAD_SOURCE must be one of: real, synthetic, fallback" >&2
    exit 2
    ;;
esac

if [[ "${WORKLOAD_SOURCE}" == "synthetic" && -z "${WORKLOAD_JSONL}" ]]; then
  WORKLOAD_JSONL="${RESULT_ROOT}/synthetic_multi_session_prompt_pairs.jsonl"
  "${PYTHON_BIN}" scripts/generate_synthetic_replay_workload.py \
    --model "${MODEL}" \
    --out-jsonl "${WORKLOAD_JSONL}" \
    --out-csv "${RESULT_ROOT}/synthetic_multi_session_prompt_pairs.csv" \
    --pairs "${SESSION_COUNT}" \
    --prompt-tokens "${SYNTHETIC_PROMPT_TOKENS}" \
    --replay-suffix-tokens "${SYNTHETIC_REPLAY_SUFFIX_TOKENS}"
elif [[ -z "${WORKLOAD_JSONL}" && "${WORKLOAD_SOURCE}" == "real" && -n "${TRACE_INDEX_CSV}" && -s "${TRACE_INDEX_CSV}" ]]; then
  WORKLOAD_JSONL="${RESULT_ROOT}/real_multi_session_prompt_pairs.jsonl"
  "${PYTHON_BIN}" scripts/extract_agentbench_trace_replay_workload.py \
    --index-csv "${TRACE_INDEX_CSV}" \
    --out-jsonl "${WORKLOAD_JSONL}" \
    --out-csv "${RESULT_ROOT}/real_multi_session_prompt_pairs.csv" \
    --max-sessions "${SESSION_COUNT}" \
    --min-gap-ms 0
fi

if [[ -z "${WORKLOAD_JSONL}" || ! -s "${WORKLOAD_JSONL}" ]]; then
  echo "WARNING: WORKLOAD_JSONL was not provided or is empty."
  echo "The multi-session driver will use fallback SWE-bench-style prompts."
fi

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 36." >&2
  exit 1
fi

echo "Milestone 36: Multi-Session Agentic Replay"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "WORKLOAD_SOURCE=${WORKLOAD_SOURCE}"
echo "WORKLOAD_JSONL=${WORKLOAD_JSONL:-fallback}"
echo "MODES=${MODES}"
echo "SESSION_COUNT=${SESSION_COUNT}"
echo "ARRIVAL_SHAPE=${ARRIVAL_SHAPE}"
echo "ARRIVAL_GAP_MS=${ARRIVAL_GAP_MS}"
echo "TOOL_WAIT_LIST_MS=${TOOL_WAIT_LIST_MS}"
echo "PREFETCH_TIMING=${PREFETCH_TIMING}"
echo "HINT_DELAY_MS=${HINT_DELAY_MS}"
echo "PREFETCH_LEAD_MS=${PREFETCH_LEAD_MS}"
echo "PRIORITY_PREFETCH_WINDOW_MS=${PRIORITY_PREFETCH_WINDOW_MS}"
echo "PRIORITY_POST_PREFETCH_QUIET_MS=${PRIORITY_POST_PREFETCH_QUIET_MS}"
echo "DEADLINE_RESERVE_WINDOW_MS=${DEADLINE_RESERVE_WINDOW_MS}"
echo "BACKGROUND_FILLERS_PER_SESSION=${BACKGROUND_FILLERS_PER_SESSION}"
echo "REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY}"
echo "HICACHE_SIZE_GB=${HICACHE_SIZE_GB}"
echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"
echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
echo "AGENTIC_KV_TRACE_SCHEDULER=${AGENTIC_KV_TRACE_SCHEDULER}"
echo "AGENTIC_KV_TRACE_KV_POOL=${AGENTIC_KV_TRACE_KV_POOL}"

run_case() {
  local mode="$1"
  case_idx=$((case_idx + 1))
  local case_id="${mode}_sessions${SESSION_COUNT}_${ARRIVAL_SHAPE}_tw${TOOL_WAIT_LIST_MS// /-}"
  case_id="${case_id//[^A-Za-z0-9_.-]/_}"
  local case_root="${RESULT_ROOT}/${case_id}"
  local trace="${case_root}/m27_trace.jsonl"
  local telemetry="${case_root}/m27_copy_telemetry.jsonl"
  local metrics="${case_root}/m35_metrics.jsonl"
  local server_log="${case_root}/sglang_server.log"

  mkdir -p "${case_root}"
  rm -f "${trace}" "${telemetry}" "${metrics}" "${server_log}"

  echo
  echo "==== Milestone 36 case [${case_idx}/${total_cases}]: ${case_id} ===="

  export AGENTIC_KV_TRACE_ENABLE=1
  export AGENTIC_KV_TRACE_PATH="${trace}"
  export AGENTIC_KV_TRACE_SCHEDULER
  export AGENTIC_KV_TRACE_KV_POOL
  export AGENTIC_KV_COPY_TELEMETRY_ENABLE=1
  export AGENTIC_KV_COPY_TELEMETRY_PATH="${telemetry}"
  export HICACHE_SIZE_GB
  export MEM_FRACTION_STATIC
  export EXTRA_SERVER_ARGS="${BASE_EXTRA_SERVER_ARGS} --max-total-tokens ${MAX_TOTAL_TOKENS}"

  setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${server_log}" 2>&1 &
  server_pid="$!"
  wait_for_server "${server_log}"

  driver_args=(
    scripts/run_multi_session_agentic_replay.py
    --base-url "${HOST_URL}/v1"
    --model "${MODEL}"
    --mode "${mode}"
    --session-count "${SESSION_COUNT}"
    --arrival-shape "${ARRIVAL_SHAPE}"
    --arrival-gap-ms "${ARRIVAL_GAP_MS}"
    --arrival-gap-range-ms "${ARRIVAL_GAP_RANGE_MS}"
    --burst-size "${BURST_SIZE}"
    --burst-gap-ms "${BURST_GAP_MS}"
    --tool-wait-list-ms "${TOOL_WAIT_LIST_MS}"
    --tool-wait-jitter-ms "${TOOL_WAIT_JITTER_MS}"
    --prefetch-timing "${PREFETCH_TIMING}"
    --hint-delay-ms "${HINT_DELAY_MS}"
    --prefetch-lead-ms "${PREFETCH_LEAD_MS}"
    --priority-prefetch-window-ms "${PRIORITY_PREFETCH_WINDOW_MS}"
    --priority-post-prefetch-quiet-ms "${PRIORITY_POST_PREFETCH_QUIET_MS}"
    --deadline-reserve-window-ms "${DEADLINE_RESERVE_WINDOW_MS}"
    --background-fillers-per-session "${BACKGROUND_FILLERS_PER_SESSION}"
    --filler-prompt-tokens "${FILLER_PROMPT_TOKENS}"
    --target-prompt-tokens "${TARGET_PROMPT_TOKENS}"
    --max-tokens "${MAX_TOKENS}"
    --prefetch-max-tokens "${PREFETCH_MAX_TOKENS}"
    --filler-max-tokens "${FILLER_MAX_TOKENS}"
    --concurrency "${REQUEST_CONCURRENCY}"
    --seed "${SEED}"
    --out "${metrics}"
  )
  if [[ -n "${WORKLOAD_JSONL}" && -s "${WORKLOAD_JSONL}" ]]; then
    driver_args+=(--workload-jsonl "${WORKLOAD_JSONL}")
  fi

  "${PYTHON_BIN}" "${driver_args[@]}" | tee "${case_root}/driver.log"
  cleanup_server
  echo "==== Completed Milestone 36 case [${case_idx}/${total_cases}]: ${case_id} ===="
}

for mode in ${MODES}; do
  case "${mode}" in
    no_prefetch|direct_prefetch|priority_direct_prefetch|deadline_priority_prefetch) ;;
    *)
      echo "ERROR: Milestone 36 only supports no_prefetch, direct_prefetch, priority_direct_prefetch, and deadline_priority_prefetch. Got: ${mode}" >&2
      exit 2
      ;;
  esac
  run_case "${mode}"
done

echo
echo "Building Milestone 36 report with the standard master report builder."
"${PYTHON_BIN}" scripts/build_milestone27_controlled_replay_report.py \
  --root "${RESULT_ROOT}" \
  --out-dir "${REPORT_ROOT}" \
  --latest-root "${LATEST_REPORT_ROOT}" \
  --max-timeline-gaps "${MAX_TIMELINE_GAPS}"

echo
echo "Milestone 36 finished."
echo "Report: ${REPORT_ROOT}/controlled_replay_report.html"
