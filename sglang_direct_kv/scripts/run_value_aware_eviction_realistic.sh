#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [[ -x ".venv/bin/python" ]]; then
  export PATH="${PWD}/.venv/bin:${PATH}"
  export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
fi

export REPORT_LABEL="${REPORT_LABEL:-value_aware_eviction_$(date +%Y%m%d_%H%M%S)}"
export HARNESSES="${HARNESSES:-hatcher}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p3_high}"
export MODES="${MODES:-no_prefetch controller_value_aware_eviction}"

# Scenario 3:
# - every replay-capable session is treated as part of the workload
# - there is no special target request for the win calculation
# - the controller scores prefixes by exposed harness signals
# - SGLang itself performs priority radix eviction with those priorities
export P3_HIGH_KNOBS="${P3_HIGH_KNOBS:-tool_wait_ms=1000 target_prompt_tokens=4096 filler_sessions=15 filler_prompt_tokens=4096 session_count=1 concurrency=6}"
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
export CONTROLLER_VALUE_AWARE_RADIX_EVICTION_POLICY="${CONTROLLER_VALUE_AWARE_RADIX_EVICTION_POLICY:-priority}"
export CONTROLLER_EVICTION_LOW_REUSE_PROBABILITY="${CONTROLLER_EVICTION_LOW_REUSE_PROBABILITY:-0.15}"
export CONTROLLER_EVICTION_LOW_RECOMPUTE_TOKENS="${CONTROLLER_EVICTION_LOW_RECOMPUTE_TOKENS:-256}"
export REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
export UPDATE_LATEST="${UPDATE_LATEST:-1}"

echo "Value-aware eviction realistic run"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "MODES=${MODES}"
echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"
echo "P3_HIGH_KNOBS=${P3_HIGH_KNOBS}"
echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
echo "TOOL_WAIT_PROFILE_SPEC=${TOOL_WAIT_PROFILE_SPEC}"
echo "TOOL_WAIT_SEED=${TOOL_WAIT_SEED}"
echo "WORKLOAD_SHAPE_MODE_INDEPENDENT=${WORKLOAD_SHAPE_MODE_INDEPENDENT}"
echo "CONTROLLER_VALUE_AWARE_RADIX_EVICTION_POLICY=${CONTROLLER_VALUE_AWARE_RADIX_EVICTION_POLICY}"
echo "CONTROLLER_EVICTION_LOW_REUSE_PROBABILITY=${CONTROLLER_EVICTION_LOW_REUSE_PROBABILITY}"
echo "CONTROLLER_EVICTION_LOW_RECOMPUTE_TOKENS=${CONTROLLER_EVICTION_LOW_RECOMPUTE_TOKENS}"
echo "EVICTION_PATH=sglang_priority_radix_eviction"

if [[ "${SKIP_PRIORITY_EVICTION_SMOKE:-0}" != "1" ]]; then
  report_dir="${REPORT_DIR:-artifacts/results/reports/${REPORT_LABEL}}"
  mkdir -p "${report_dir}"
  "${PYTHON_BIN:-python}" scripts/smoke_priority_radix_eviction.py \
    --out-json "${report_dir}/priority_radix_eviction_smoke.json"
fi

bash scripts/run_harness_deadline_pressure.sh "${MODEL}"

"${PYTHON_BIN:-python}" scripts/build_value_aware_eviction_audit.py \
  --run-root "${RUN_ROOT:-artifacts/results/runs/controlled/${REPORT_LABEL}}" \
  --report-dir "${REPORT_DIR:-artifacts/results/reports/${REPORT_LABEL}}"
