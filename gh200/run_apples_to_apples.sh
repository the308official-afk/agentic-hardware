#!/usr/bin/env bash
set -euo pipefail

# EC2-scale pressure on GH200. Use this for apples-to-apples comparison with
# the EC2/A10G-class run before scaling pressure upward.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-ec2_a10g}"
export SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline harness_emitted frontend_supplied gateway_injected}"
export HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p3_high p5_boss_queue}"
export REPORT_LABEL="${REPORT_LABEL:-gh200_apples_to_apples_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_signal_design_space_docker.sh"
