#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [[ -x ".venv/bin/python" ]]; then
  export PATH="${PWD}/.venv/bin:${PATH}"
  export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
fi

export REPORT_LABEL="${REPORT_LABEL:-memory_admission_all_requests_$(date +%Y%m%d_%H%M%S)}"
export HARNESSES="${HARNESSES:-hatcher}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p3_high}"
export MODES="${MODES:-no_prefetch controller_memory_admission}"

# Scenario 6:
# - every replay-capable session is treated as part of the measured workload
# - the controller delays new candidate work before SGLang when near-future
#   replay demand would make immediate admission risky
# - SGLang remains the source of truth for KV state; the controller gates
#   admission before requests enter SGLang
export P3_HIGH_KNOBS="${P3_HIGH_KNOBS:-tool_wait_ms=1000 target_prompt_tokens=4096 filler_sessions=1 filler_prompt_tokens=4096 session_count=8 concurrency=8}"
export TASK_REPLAY_STEPS="${TASK_REPLAY_STEPS:-2}"
export FILLER_REPLAY_DEADLINES="${FILLER_REPLAY_DEADLINES:-1}"
export TOOL_WAIT_PROFILE="${TOOL_WAIT_PROFILE:-realistic_deadline_fair}"
export TOOL_WAIT_PROFILE_SPEC="${TOOL_WAIT_PROFILE_SPEC:-very_short:20:100-500,short:35:1000-5000,medium:30:10000-30000,long:15:60000-120000}"
export TOOL_WAIT_SEED="${TOOL_WAIT_SEED:-20260917}"
export WORKLOAD_SHAPE_MODE_INDEPENDENT="${WORKLOAD_SHAPE_MODE_INDEPENDENT:-1}"
export AGENTIC_WORKLOAD_PROFILE="${AGENTIC_WORKLOAD_PROFILE:-synthetic_pressure}"

export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-12288}"
export TRACE_PROFILE="${TRACE_PROFILE:-controller_decision}"
export TRACE_CONTROLLER_DECISIONS="${TRACE_CONTROLLER_DECISIONS:-1}"
export TRACE_CONTROLLER_COMPLETION_LINKAGE="${TRACE_CONTROLLER_COMPLETION_LINKAGE:-1}"
export AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL:-1}"
export AGENTIC_KV_COPY_TELEMETRY_ENABLE="${AGENTIC_KV_COPY_TELEMETRY_ENABLE:-1}"

export CONTROLLER_MEMORY_ADMISSION_LOOKAHEAD_MS="${CONTROLLER_MEMORY_ADMISSION_LOOKAHEAD_MS:-15000}"
export CONTROLLER_MEMORY_ADMISSION_RELEASE_GRACE_MS="${CONTROLLER_MEMORY_ADMISSION_RELEASE_GRACE_MS:-50}"
export CONTROLLER_MEMORY_ADMISSION_MAX_DELAY_MS="${CONTROLLER_MEMORY_ADMISSION_MAX_DELAY_MS:-30000}"
export CONTROLLER_MEMORY_ADMISSION_TOKEN_BUDGET="${CONTROLLER_MEMORY_ADMISSION_TOKEN_BUDGET:-${MAX_TOTAL_TOKENS}}"
export CONTROLLER_MEMORY_ADMISSION_RESERVE_FRACTION="${CONTROLLER_MEMORY_ADMISSION_RESERVE_FRACTION:-0.65}"
export CONTROLLER_MEMORY_ADMISSION_MIN_CANDIDATE_TOKENS="${CONTROLLER_MEMORY_ADMISSION_MIN_CANDIDATE_TOKENS:-1024}"

export REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
export UPDATE_LATEST="${UPDATE_LATEST:-1}"

echo "Memory admission realistic run"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "MODES=${MODES}"
echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"
echo "P3_HIGH_KNOBS=${P3_HIGH_KNOBS}"
echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
echo "TOOL_WAIT_PROFILE_SPEC=${TOOL_WAIT_PROFILE_SPEC}"
echo "TOOL_WAIT_SEED=${TOOL_WAIT_SEED}"
echo "WORKLOAD_SHAPE_MODE_INDEPENDENT=${WORKLOAD_SHAPE_MODE_INDEPENDENT}"
echo "CONTROLLER_MEMORY_ADMISSION_LOOKAHEAD_MS=${CONTROLLER_MEMORY_ADMISSION_LOOKAHEAD_MS}"
echo "CONTROLLER_MEMORY_ADMISSION_RESERVE_FRACTION=${CONTROLLER_MEMORY_ADMISSION_RESERVE_FRACTION}"
echo "CONTROLLER_MEMORY_ADMISSION_TOKEN_BUDGET=${CONTROLLER_MEMORY_ADMISSION_TOKEN_BUDGET}"

bash scripts/run_harness_deadline_pressure.sh "${MODEL}"

"${PYTHON_BIN:-python}" scripts/build_memory_admission_audit.py \
  --run-root "${RUN_ROOT:-artifacts/results/runs/controlled/${REPORT_LABEL}}" \
  --report-dir "${REPORT_DIR:-artifacts/results/reports/${REPORT_LABEL}}"
