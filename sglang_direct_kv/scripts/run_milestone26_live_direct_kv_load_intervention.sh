#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone26_live_direct_kv_load_$(date +%Y%m%d_%H%M%S)}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
LIVE_PREFETCH_ACTION="${LIVE_PREFETCH_ACTION:-direct_load}"
LIVE_PREFETCH_MAX_TOKENS="${LIVE_PREFETCH_MAX_TOKENS:-1}"
LIVE_PREFETCH_POLL_MS="${LIVE_PREFETCH_POLL_MS:-25}"
LIVE_PREFETCH_DRAIN_SECS="${LIVE_PREFETCH_DRAIN_SECS:-3}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-24}"
AGENTIC_KV_TRACE_MAX_EXACT_INDICES="${AGENTIC_KV_TRACE_MAX_EXACT_INDICES:-512}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PROMPT_EVOLUTION_BATCH_ID="${PROMPT_EVOLUTION_BATCH_ID:-milestone26_live_direct_kv_load_$(date +%Y%m%d_%H%M%S)}"
export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:-15}"
export REUSE_SERVER="${REUSE_SERVER:-0}"
export SERVER_MODE="${SERVER_MODE:-hicache}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
export HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
export SERVER_READY_TIMEOUT_SECS="${SERVER_READY_TIMEOUT_SECS:-1800}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-auto}"
export SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
export SAMPLING_DEFAULTS="${SAMPLING_DEFAULTS:-openai}"
export ENABLE_TOOL_NORMALIZER_PROXY="${ENABLE_TOOL_NORMALIZER_PROXY:-1}"
export PROMPT_EVOLUTION_TOOL_LOOP_CASE="${PROMPT_EVOLUTION_TOOL_LOOP_CASE:-ls-read-execute}"
export PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP="${PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP:-1}"
export RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
export AGENTBENCH_INSTALL_DEPS="${AGENTBENCH_INSTALL_DEPS:-0}"
export AGENTBENCH_DEEPAGENTS_SOURCE="${AGENTBENCH_DEEPAGENTS_SOURCE:-upstream}"
export AGENTBENCH_EXECUTION_LOOP="${AGENTBENCH_EXECUTION_LOOP:-1}"
export AGENTBENCH_EXECUTION_LOOP_MAX_STEPS="${AGENTBENCH_EXECUTION_LOOP_MAX_STEPS:-10}"
export AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST="${AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST:-0}"
export AGENTBENCH_EXECUTION_GUARD="${AGENTBENCH_EXECUTION_GUARD:-0}"
export AGENTBENCH_FORCE_TOOL_CHOICE="${AGENTBENCH_FORCE_TOOL_CHOICE:-auto}"
export AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT="${AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT:-1}"
export AGENTBENCH_SOFT_STOP_RECURSION="${AGENTBENCH_SOFT_STOP_RECURSION:-1}"
export AGENTBENCH_AGENT_RECURSION_LIMIT="${AGENTBENCH_AGENT_RECURSION_LIMIT:-1000}"
export AGENTBENCH_TRACE_AGENT_STREAM="${AGENTBENCH_TRACE_AGENT_STREAM:-0}"
export AGENTBENCH_DIRECT_SGLANG_TOOL_RICH="${AGENTBENCH_DIRECT_SGLANG_TOOL_RICH:-1}"
export AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT="${AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT:-1}"
export AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS="${AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS:-1}"
export AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD="${AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD:-1}"
export AGENTBENCH_BATCH_CONTINUE_ON_ERROR="${AGENTBENCH_BATCH_CONTINUE_ON_ERROR:-1}"
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
LATEST_REPORT_ROOT="$(mkdir -p "${LATEST_REPORT_ROOT}" && cd "${LATEST_REPORT_ROOT}" && pwd)"

export LIVE_HINT_LOG="${RESULT_ROOT}/live_hint_events.jsonl"
export LIVE_HINT_PAYLOAD_DIR="${RESULT_ROOT}/live_hint_payloads"
export AGENTIC_KV_TRACE_ENABLE=1
export AGENTIC_KV_TRACE_PATH="${RESULT_ROOT}/live_direct_kv_trace.jsonl"
export AGENTIC_KV_TRACE_MAX_EXACT_INDICES
export AGENTIC_KV_COPY_TELEMETRY_ENABLE=1
export AGENTIC_KV_COPY_TELEMETRY_PATH="${RESULT_ROOT}/live_direct_kv_copy_telemetry.jsonl"

CONTROLLER_LOG="${RESULT_ROOT}/live_prefetch_controller.jsonl"
PROXY_JSONL="${RESULT_ROOT}/tool_normalizer_proxy.jsonl"
TASK_INDEX_CSV="${RESULT_ROOT}/exp6_direct_sglang_task_index.csv"
OUT_DIR="${RESULT_ROOT}/live_agentbench_prefetch_report"
DIRECT_KV_OUT_DIR="${RESULT_ROOT}/live_direct_kv_load_report"

mkdir -p "${LIVE_HINT_PAYLOAD_DIR}"
: > "${LIVE_HINT_LOG}"
: > "${CONTROLLER_LOG}"
: > "${AGENTIC_KV_TRACE_PATH}"
: > "${AGENTIC_KV_COPY_TELEMETRY_PATH}"

controller_pid=""
cleanup_controller() {
  if [[ -n "${controller_pid}" ]]; then
    kill "${controller_pid}" >/dev/null 2>&1 || true
    wait "${controller_pid}" >/dev/null 2>&1 || true
    controller_pid=""
  fi
}
trap cleanup_controller EXIT

echo "Milestone 26: Live Direct KV Load Intervention"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "TASK_RANGE=${START_INDEX}-${END_INDEX}"
echo "HOST_URL=${HOST_URL}"
echo "SERVER_MODE=${SERVER_MODE}"
echo "LIVE_PREFETCH_ACTION=${LIVE_PREFETCH_ACTION}"
echo "AGENTIC_KV_TRACE_PATH=${AGENTIC_KV_TRACE_PATH}"
echo "AGENTIC_KV_COPY_TELEMETRY_PATH=${AGENTIC_KV_COPY_TELEMETRY_PATH}"

"${PYTHON_BIN}" scripts/live_prefetch_controller.py \
  --hint-log "${LIVE_HINT_LOG}" \
  --controller-log "${CONTROLLER_LOG}" \
  --target-base "${HOST_URL}" \
  --poll-ms "${LIVE_PREFETCH_POLL_MS}" \
  --max-tokens "${LIVE_PREFETCH_MAX_TOKENS}" \
  --action "${LIVE_PREFETCH_ACTION}" \
  >"${RESULT_ROOT}/live_prefetch_controller.log" 2>&1 &
controller_pid="$!"
echo "${controller_pid}" > "${RESULT_ROOT}/live_prefetch_controller.pid"

RESULT_ROOT="${RESULT_ROOT}" \
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT}" \
HOST_URL="${HOST_URL}" \
LIVE_HINT_LOG="${LIVE_HINT_LOG}" \
LIVE_HINT_PAYLOAD_DIR="${LIVE_HINT_PAYLOAD_DIR}" \
bash scripts/run_milestone21_exp6_direct_sglang.sh "${MODEL}"

sleep "${LIVE_PREFETCH_DRAIN_SECS}"
cleanup_controller

"${PYTHON_BIN}" scripts/build_live_agentbench_tool_gap_report.py \
  --proxy-jsonl "${PROXY_JSONL}" \
  --task-index-csv "${TASK_INDEX_CSV}" \
  --hint-log "${LIVE_HINT_LOG}" \
  --controller-log "${CONTROLLER_LOG}" \
  --out-dir "${OUT_DIR}" \
  --latest-root "${LATEST_REPORT_ROOT}" \
  --max-timeline-gaps "${MAX_TIMELINE_GAPS}"

"${PYTHON_BIN}" scripts/build_live_direct_kv_load_report.py \
  --proxy-jsonl "${PROXY_JSONL}" \
  --task-index-csv "${TASK_INDEX_CSV}" \
  --hint-log "${LIVE_HINT_LOG}" \
  --controller-log "${CONTROLLER_LOG}" \
  --trace "${AGENTIC_KV_TRACE_PATH}" \
  --copy-telemetry "${AGENTIC_KV_COPY_TELEMETRY_PATH}" \
  --out-dir "${DIRECT_KV_OUT_DIR}" \
  --latest-root "${LATEST_REPORT_ROOT}"

echo
echo "Milestone 26 finished."
echo "Live tool-gap report: ${OUT_DIR}/live_agentbench_tool_gap_report.html"
echo "Direct KV evidence report: ${DIRECT_KV_OUT_DIR}/live_direct_kv_load_report.html"
echo "Latest Milestone 26 report: ${LATEST_REPORT_ROOT}/latest_real/m26_live_direct_kv_load_report.html"
echo "Trace: ${AGENTIC_KV_TRACE_PATH}"
echo "Copy telemetry: ${AGENTIC_KV_COPY_TELEMETRY_PATH}"
