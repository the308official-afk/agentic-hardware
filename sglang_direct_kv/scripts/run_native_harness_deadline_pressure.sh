#!/usr/bin/env bash
set -euo pipefail

# Runs only harness paths that are currently backed by native clients or the
# in-repo Hatcher control. Adapter-only harnesses stay out of this run by
# default so the Replay Deadline Pressure Chart is not mixed.

export HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw}"
export REPORT_LABEL="${REPORT_LABEL:-native_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)}"
export REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_harness_deadline_pressure.sh" "$@"
