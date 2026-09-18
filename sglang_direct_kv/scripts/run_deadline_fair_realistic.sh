#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export REPORT_LABEL="${REPORT_LABEL:-deadline_predictive_queue_realistic_$(date +%Y%m%d_%H%M%S)}"
export HARNESSES="${HARNESSES:-hatcher}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p3_high}"
export MODES="${MODES:-no_prefetch controller_predictive_deadline_queue}"

# First realistic predictive deadline queue pass:
# - 1 main session + 15 peer replay sessions = 16 sessions
# - 2 replay steps per session = 32 replay requests per mode
# - no target/filler interpretation in the manager-facing analysis
export P3_HIGH_KNOBS="${P3_HIGH_KNOBS:-tool_wait_ms=1000 target_prompt_tokens=2048 filler_sessions=15 filler_prompt_tokens=2048 session_count=1 concurrency=6}"
export TASK_REPLAY_STEPS="${TASK_REPLAY_STEPS:-2}"
export FILLER_REPLAY_DEADLINES="${FILLER_REPLAY_DEADLINES:-1}"
export TOOL_WAIT_PROFILE="${TOOL_WAIT_PROFILE:-realistic_deadline_fair}"
export TOOL_WAIT_PROFILE_SPEC="${TOOL_WAIT_PROFILE_SPEC:-very_short:20:100-500,short:35:1000-5000,medium:30:10000-30000,long:15:60000-120000}"
export TOOL_WAIT_SEED="${TOOL_WAIT_SEED:-20260916}"
export WORKLOAD_SHAPE_MODE_INDEPENDENT="${WORKLOAD_SHAPE_MODE_INDEPENDENT:-1}"
# Keep prompt and output sizes fixed in the first run so the new variable is
# timing realism, not mixed replay size. Mixed replay sizes are a later step.
export AGENTIC_WORKLOAD_PROFILE="${AGENTIC_WORKLOAD_PROFILE:-synthetic_pressure}"

export TRACE_PROFILE="${TRACE_PROFILE:-controller_decision}"
export TRACE_CONTROLLER_DECISIONS="${TRACE_CONTROLLER_DECISIONS:-1}"
export TRACE_CONTROLLER_COMPLETION_LINKAGE="${TRACE_CONTROLLER_COMPLETION_LINKAGE:-1}"
export REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
export UPDATE_LATEST="${UPDATE_LATEST:-1}"

echo "Predictive deadline queue realistic run"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "MODES=${MODES}"
echo "TOOL_WAIT_PROFILE_SPEC=${TOOL_WAIT_PROFILE_SPEC}"
echo "TOOL_WAIT_SEED=${TOOL_WAIT_SEED}"
echo "WORKLOAD_SHAPE_MODE_INDEPENDENT=${WORKLOAD_SHAPE_MODE_INDEPENDENT}"

bash scripts/run_harness_deadline_pressure.sh "${MODEL}"
