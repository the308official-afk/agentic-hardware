#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone26_live_paired_direct_kv_$(date +%Y%m%d_%H%M%S)}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-16}"
RUN_NO_PREFETCH="${RUN_NO_PREFETCH:-1}"
RUN_DIRECT_KV="${RUN_DIRECT_KV:-1}"
NO_PREFETCH_ROOT="${NO_PREFETCH_ROOT:-}"
DIRECT_KV_ROOT="${DIRECT_KV_ROOT:-}"
TOOL_NORMALIZER_PORT_NO_PREFETCH="${TOOL_NORMALIZER_PORT_NO_PREFETCH:-31061}"
TOOL_NORMALIZER_PORT_DIRECT_KV="${TOOL_NORMALIZER_PORT_DIRECT_KV:-31062}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:-15}"
export AGENTBENCH_EXECUTION_LOOP_MAX_STEPS="${AGENTBENCH_EXECUTION_LOOP_MAX_STEPS:-10}"
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

if [[ -z "${NO_PREFETCH_ROOT}" ]]; then
  NO_PREFETCH_ROOT="${RESULT_ROOT}/no_prefetch_live"
fi
if [[ -z "${DIRECT_KV_ROOT}" ]]; then
  DIRECT_KV_ROOT="${RESULT_ROOT}/live_direct_kv_load"
fi
REPORT_ROOT="${RESULT_ROOT}/live_paired_direct_kv_report"

echo "Milestone 26: Live Paired Direct KV Report"
echo "MODEL=${MODEL}"
echo "TASK_RANGE=${START_INDEX}-${END_INDEX}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "NO_PREFETCH_ROOT=${NO_PREFETCH_ROOT}"
echo "DIRECT_KV_ROOT=${DIRECT_KV_ROOT}"
echo "SERVER_MODE=${SERVER_MODE}"

if [[ "${RUN_NO_PREFETCH}" = "1" ]]; then
  echo
  echo "Milestone 26A: running live no-prefetch baseline with the same HiCache server mode."
  RESULT_ROOT="${NO_PREFETCH_ROOT}" \
  LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT}" \
  PROMPT_EVOLUTION_BATCH_ID="${PROMPT_EVOLUTION_BATCH_ID_NO_PREFETCH:-milestone26_no_prefetch_$(date +%Y%m%d_%H%M%S)}" \
  TOOL_NORMALIZER_PORT="${TOOL_NORMALIZER_PORT_NO_PREFETCH}" \
  LIVE_HINT_LOG="" \
  LIVE_HINT_PAYLOAD_DIR="" \
  bash scripts/run_milestone21_exp6_direct_sglang.sh "${MODEL}"
else
  echo
  echo "Milestone 26A: reusing existing no-prefetch root: ${NO_PREFETCH_ROOT}"
fi

if [[ "${RUN_DIRECT_KV}" = "1" ]]; then
  echo
  echo "Milestone 26B: running live direct KV load intervention."
  RESULT_ROOT="${DIRECT_KV_ROOT}" \
  LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT}" \
  PROMPT_EVOLUTION_BATCH_ID="${PROMPT_EVOLUTION_BATCH_ID_DIRECT_KV:-milestone26_direct_kv_$(date +%Y%m%d_%H%M%S)}" \
  TOOL_NORMALIZER_PORT="${TOOL_NORMALIZER_PORT_DIRECT_KV}" \
  LIVE_PREFETCH_ACTION=direct_load \
  bash scripts/run_milestone26_live_direct_kv_load_intervention.sh "${MODEL}"
else
  echo
  echo "Milestone 26B: reusing existing direct-KV root: ${DIRECT_KV_ROOT}"
fi

echo
echo "Milestone 26C: building paired live report."
"${PYTHON_BIN}" scripts/build_live_paired_agentbench_report.py \
  --no-prefetch-root "${NO_PREFETCH_ROOT}" \
  --prefetch-root "${DIRECT_KV_ROOT}" \
  --out-dir "${REPORT_ROOT}" \
  --latest-root "${LATEST_REPORT_ROOT}" \
  --max-timeline-gaps "${MAX_TIMELINE_GAPS}"

echo
echo "Milestone 26 finished."
echo "Paired direct-KV report: ${REPORT_ROOT}/live_paired_agentbench_report.html"
echo "Direct KV evidence report: ${DIRECT_KV_ROOT}/live_direct_kv_load_report/live_direct_kv_load_report.html"
echo "Latest master report: ${LATEST_REPORT_ROOT}/latest_master_report.html"
