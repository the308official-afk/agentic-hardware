#!/usr/bin/env bash
set -euo pipefail

# Fast first GH200 GPU check: Hatcher only, baseline vs gateway priority,
# three pressure levels.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-ec2_a10g}"
export SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline gateway_injected}"
export HARNESSES="${HARNESSES:-hatcher}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p3_high p5_boss_queue}"
export REPORT_LABEL="${REPORT_LABEL:-gh200_sentinel_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_host_signal_design_space.sh"
