#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
RUN_AGENTBENCH="${RUN_AGENTBENCH:-1}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone22_live_agentbench_bridge_$(date +%Y%m%d_%H%M%S)}"
EXISTING_RESULT_ROOT="${EXISTING_RESULT_ROOT:-}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-24}"
INCLUDE_PREFLIGHT_IN_REPORT="${INCLUDE_PREFLIGHT_IN_REPORT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PROMPT_EVOLUTION_BATCH_ID="${PROMPT_EVOLUTION_BATCH_ID:-milestone22_live_agentbench_$(date +%Y%m%d_%H%M%S)}"
export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:-15}"
export REUSE_SERVER="${REUSE_SERVER:-0}"
export SERVER_MODE="${SERVER_MODE:-simple}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
export SERVER_READY_TIMEOUT_SECS="${SERVER_READY_TIMEOUT_SECS:-1800}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-auto}"
export SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
export SAMPLING_DEFAULTS="${SAMPLING_DEFAULTS:-openai}"
export ENABLE_TOOL_NORMALIZER_PROXY="${ENABLE_TOOL_NORMALIZER_PROXY:-1}"
export PROMPT_EVOLUTION_TOOL_LOOP_CASE="${PROMPT_EVOLUTION_TOOL_LOOP_CASE:-ls-read-execute}"
export PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP="${PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP:-1}"
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
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ "${RUN_AGENTBENCH}" = "1" ]]; then
  RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
  LATEST_REPORT_ROOT="$(mkdir -p "${LATEST_REPORT_ROOT}" && cd "${LATEST_REPORT_ROOT}" && pwd)"
  echo "Milestone 22A: running live AgentBench traffic through direct SGLang."
  echo "MODEL=${MODEL}"
  echo "RESULT_ROOT=${RESULT_ROOT}"
  echo "TASK_RANGE=${START_INDEX}-${END_INDEX}"
  RESULT_ROOT="${RESULT_ROOT}" \
  LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT}" \
  bash scripts/run_milestone21_exp6_direct_sglang.sh "${MODEL}"
else
  if [[ -n "${EXISTING_RESULT_ROOT}" ]]; then
    RESULT_ROOT="${EXISTING_RESULT_ROOT}"
  fi
  RESULT_ROOT="$(cd "${RESULT_ROOT}" && pwd)"
  LATEST_REPORT_ROOT="$(mkdir -p "${LATEST_REPORT_ROOT}" && cd "${LATEST_REPORT_ROOT}" && pwd)"
  echo "Milestone 22A: building live report from existing result root."
  echo "RESULT_ROOT=${RESULT_ROOT}"
fi

PROXY_JSONL="${RESULT_ROOT}/tool_normalizer_proxy.jsonl"
TASK_INDEX_CSV="${RESULT_ROOT}/exp6_direct_sglang_task_index.csv"
OUT_DIR="${RESULT_ROOT}/live_agentbench_tool_gap_report"

if [[ ! -f "${PROXY_JSONL}" ]]; then
  echo "Missing proxy log: ${PROXY_JSONL}" >&2
  echo "Run Milestone 21/22 with ENABLE_TOOL_NORMALIZER_PROXY=1, or set EXISTING_RESULT_ROOT to a result directory that has tool_normalizer_proxy.jsonl." >&2
  exit 1
fi

if [[ "${INCLUDE_PREFLIGHT_IN_REPORT}" = "1" ]]; then
  "${PYTHON_BIN}" scripts/build_live_agentbench_tool_gap_report.py \
    --proxy-jsonl "${PROXY_JSONL}" \
    --task-index-csv "${TASK_INDEX_CSV}" \
    --out-dir "${OUT_DIR}" \
    --latest-root "${LATEST_REPORT_ROOT}" \
    --max-timeline-gaps "${MAX_TIMELINE_GAPS}" \
    --include-preflight
else
  "${PYTHON_BIN}" scripts/build_live_agentbench_tool_gap_report.py \
    --proxy-jsonl "${PROXY_JSONL}" \
    --task-index-csv "${TASK_INDEX_CSV}" \
    --out-dir "${OUT_DIR}" \
    --latest-root "${LATEST_REPORT_ROOT}" \
    --max-timeline-gaps "${MAX_TIMELINE_GAPS}"
fi

echo
echo "Milestone 22 finished."
echo "Report: ${OUT_DIR}/live_agentbench_tool_gap_report.html"
echo "Latest report: ${LATEST_REPORT_ROOT}/latest_live_agentbench_tool_gap_report.html"
echo "Latest CSV: ${LATEST_REPORT_ROOT}/latest_live_agentbench_tool_gaps.csv"
