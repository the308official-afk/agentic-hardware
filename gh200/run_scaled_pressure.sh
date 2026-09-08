#!/usr/bin/env bash
set -euo pipefail

# GH200-scale pressure ladder. Run after the sentinel and apples-to-apples
# checks have passed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-gh200}"
export SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline harness_emitted frontend_supplied gateway_injected}"
export HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue}"
export REPORT_LABEL="${REPORT_LABEL:-gh200_scaled_deadline_pressure_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_signal_design_space_docker.sh"
