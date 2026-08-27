#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone27_real_prompt_controlled_replay_$(date +%Y%m%d_%H%M%S)}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
WORKLOAD_JSONL="${WORKLOAD_JSONL:-}"
WORKLOAD_SOURCE="${WORKLOAD_SOURCE:-real}"
TRACE_INDEX_CSV="${TRACE_INDEX_CSV:-}"
MAX_PAIRS="${MAX_PAIRS:-12}"
MODES="${MODES:-no_prefetch direct_prefetch dynamo_priority_hints}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-100 250 500 1000}"
FILLER_LIST="${FILLER_LIST:-16 64}"
FILLER_PROMPT_TOKENS="${FILLER_PROMPT_TOKENS:-1024}"
TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-0}"
SYNTHETIC_PROMPT_TOKENS="${SYNTHETIC_PROMPT_TOKENS:-4096}"
SYNTHETIC_REPLAY_SUFFIX_TOKENS="${SYNTHETIC_REPLAY_SUFFIX_TOKENS:-256}"
FILLER_DIVERGE_EARLY="${FILLER_DIVERGE_EARLY:-1}"
PREFETCH_TIMING="${PREFETCH_TIMING:-near_resume}"
HINT_DELAY_MS="${HINT_DELAY_MS:-20}"
ORACLE_LEAD_MS="${ORACLE_LEAD_MS:-250}"
PRIORITY_DIRECT_PREFETCH="${PRIORITY_DIRECT_PREFETCH:-0}"
PRIORITY_PREFETCH_HEAD_START_MS="${PRIORITY_PREFETCH_HEAD_START_MS:-50}"
PRIORITY_REPLAY_GUARD_MS="${PRIORITY_REPLAY_GUARD_MS:-120}"
PRIORITY_REPLAY_RELEASE_MS="${PRIORITY_REPLAY_RELEASE_MS:-80}"
PRIORITY_FILLER_STAGGER_MS="${PRIORITY_FILLER_STAGGER_MS:-2}"
DYNAMO_HIGH_PRIORITY="${DYNAMO_HIGH_PRIORITY:-100}"
DYNAMO_NORMAL_PRIORITY="${DYNAMO_NORMAL_PRIORITY:-0}"
DYNAMO_LOW_PRIORITY="${DYNAMO_LOW_PRIORITY:--100}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-8}"
PREFETCH_MAX_TOKENS="${PREFETCH_MAX_TOKENS:-1}"
SERVER_READY_TIMEOUT_SECS="${SERVER_READY_TIMEOUT_SECS:-900}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
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

case "${WORKLOAD_SOURCE}" in
  real|synthetic|fallback) ;;
  *)
    echo "ERROR: WORKLOAD_SOURCE must be one of: real, synthetic, fallback" >&2
    exit 2
    ;;
esac

if [[ "${WORKLOAD_SOURCE}" == "synthetic" && -z "${WORKLOAD_JSONL}" ]]; then
  WORKLOAD_JSONL="${RESULT_ROOT}/synthetic_prompt_pairs.jsonl"
  "${PYTHON_BIN}" scripts/generate_synthetic_replay_workload.py \
    --model "${MODEL}" \
    --out-jsonl "${WORKLOAD_JSONL}" \
    --out-csv "${RESULT_ROOT}/synthetic_prompt_pairs.csv" \
    --pairs "${MAX_PAIRS}" \
    --prompt-tokens "${SYNTHETIC_PROMPT_TOKENS}" \
    --replay-suffix-tokens "${SYNTHETIC_REPLAY_SUFFIX_TOKENS}"
elif [[ -z "${WORKLOAD_JSONL}" && "${WORKLOAD_SOURCE}" == "real" && -n "${TRACE_INDEX_CSV}" && -s "${TRACE_INDEX_CSV}" ]]; then
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
  echo "The driver will use fallback realistic prompts. For real prompts, pass WORKLOAD_JSONL=/path/to/real_prompt_pairs.jsonl. For synthetic prompts, set WORKLOAD_SOURCE=synthetic."
fi

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 27." >&2
  exit 1
fi

echo "Milestone 27: Real-Prompt Controlled Replay"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "WORKLOAD_SOURCE=${WORKLOAD_SOURCE}"
echo "WORKLOAD_JSONL=${WORKLOAD_JSONL:-fallback}"
echo "MODES=${MODES}"
echo "FILLER_LIST=${FILLER_LIST}"
echo "FILLER_PROMPT_TOKENS=${FILLER_PROMPT_TOKENS}"
echo "TARGET_PROMPT_TOKENS=${TARGET_PROMPT_TOKENS}"
echo "SYNTHETIC_PROMPT_TOKENS=${SYNTHETIC_PROMPT_TOKENS}"
echo "SYNTHETIC_REPLAY_SUFFIX_TOKENS=${SYNTHETIC_REPLAY_SUFFIX_TOKENS}"
echo "FILLER_DIVERGE_EARLY=${FILLER_DIVERGE_EARLY}"
echo "TOOL_WAIT_LIST_MS=${TOOL_WAIT_LIST_MS}"
echo "MAX_PAIRS=${MAX_PAIRS}"
echo "PRIORITY_DIRECT_PREFETCH=${PRIORITY_DIRECT_PREFETCH}"
echo "PRIORITY_PREFETCH_HEAD_START_MS=${PRIORITY_PREFETCH_HEAD_START_MS}"
echo "PRIORITY_REPLAY_GUARD_MS=${PRIORITY_REPLAY_GUARD_MS}"
echo "PRIORITY_REPLAY_RELEASE_MS=${PRIORITY_REPLAY_RELEASE_MS}"
echo "DYNAMO_HIGH_PRIORITY=${DYNAMO_HIGH_PRIORITY}"
echo "DYNAMO_NORMAL_PRIORITY=${DYNAMO_NORMAL_PRIORITY}"
echo "DYNAMO_LOW_PRIORITY=${DYNAMO_LOW_PRIORITY}"
echo "AGENTIC_KV_TRACE_SCHEDULER=${AGENTIC_KV_TRACE_SCHEDULER}"
echo "AGENTIC_KV_TRACE_KV_POOL=${AGENTIC_KV_TRACE_KV_POOL}"
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
  export AGENTIC_KV_TRACE_SCHEDULER
  export AGENTIC_KV_TRACE_KV_POOL
  export AGENTIC_KV_COPY_TELEMETRY_ENABLE=1
  export AGENTIC_KV_COPY_TELEMETRY_PATH="${telemetry}"
  export HICACHE_SIZE_GB
  export MEM_FRACTION_STATIC
  export EXTRA_SERVER_ARGS="${BASE_EXTRA_SERVER_ARGS} --max-total-tokens ${MAX_TOTAL_TOKENS}"
  if [[ "${mode}" == "dynamo_priority_hints" ]]; then
    export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS} --enable-cache-report --enable-priority-scheduling --default-priority-value ${DYNAMO_NORMAL_PRIORITY} --radix-eviction-policy priority"
  fi

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
    --target-prompt-tokens "${TARGET_PROMPT_TOKENS}"
    --prefetch-timing "${PREFETCH_TIMING}"
    --hint-delay-ms "${HINT_DELAY_MS}"
    --oracle-lead-ms "${ORACLE_LEAD_MS}"
    --priority-prefetch-head-start-ms "${PRIORITY_PREFETCH_HEAD_START_MS}"
    --priority-replay-guard-ms "${PRIORITY_REPLAY_GUARD_MS}"
    --priority-replay-release-ms "${PRIORITY_REPLAY_RELEASE_MS}"
    --priority-filler-stagger-ms "${PRIORITY_FILLER_STAGGER_MS}"
    --dynamo-high-priority "${DYNAMO_HIGH_PRIORITY}"
    --dynamo-normal-priority "${DYNAMO_NORMAL_PRIORITY}"
    --dynamo-low-priority "${DYNAMO_LOW_PRIORITY}"
    --max-tokens "${MAX_TOKENS}"
    --prefetch-max-tokens "${PREFETCH_MAX_TOKENS}"
    --concurrency "${REQUEST_CONCURRENCY}"
    --out "${metrics}"
  )
  if [[ -n "${WORKLOAD_JSONL}" && -s "${WORKLOAD_JSONL}" ]]; then
    driver_args+=(--workload-jsonl "${WORKLOAD_JSONL}")
  fi
  if [[ "${FILLER_DIVERGE_EARLY}" == "1" ]]; then
    driver_args+=(--filler-diverge-early)
  else
    driver_args+=(--no-filler-diverge-early)
  fi
  if [[ "${PRIORITY_DIRECT_PREFETCH}" == "1" ]]; then
    driver_args+=(--priority-direct-prefetch)
  else
    driver_args+=(--no-priority-direct-prefetch)
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
