#!/usr/bin/env bash
set -euo pipefail

# GH200 controller scale-up run. This uses the same portable controller modes
# validated on EC2, but swaps in the GH200 pressure profile and host-harness /
# Docker-SGLang split.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-gh200}"
export SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline gateway_injected controller_scheduler controller_preload controller_targeted_prefetch controller_demote_restore controller_admission controller_full controller_full_chunked}"
export HARNESSES="${HARNESSES:-hatcher}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue}"
export REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
export REPORT_LABEL="${REPORT_LABEL:-gh200_controller_scaleup_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_host_signal_design_space.sh"
