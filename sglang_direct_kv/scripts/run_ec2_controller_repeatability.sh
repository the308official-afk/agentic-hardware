#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-ec2_a10g}"
export SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline gateway_injected controller_scheduler controller_demote_restore controller_admission controller_full}"
export HARNESSES="${HARNESSES:-hatcher}"
export PRESSURE_LEVELS="${PRESSURE_LEVELS:-p1_mild p3_high p4_cliff p5_boss_queue}"
export REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
export UPDATE_LATEST="${UPDATE_LATEST:-1}"
export SKIP_INTERIM_REPORTS="${SKIP_INTERIM_REPORTS:-1}"
export REPORT_LABEL="${REPORT_LABEL:-ec2_controller_repeatability_$(date +%Y%m%d_%H%M%S)}"

echo "EC2 Controller Repeatability"
echo "MODEL=${MODEL}"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "HARDWARE_PROFILE=${HARDWARE_PROFILE}"
echo "SIGNAL_FAMILIES=${SIGNAL_FAMILIES}"
echo "HARNESSES=${HARNESSES}"
echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"
echo "REPORT_BUILDER_MODE=${REPORT_BUILDER_MODE}"
echo "SKIP_INTERIM_REPORTS=${SKIP_INTERIM_REPORTS}"
echo

bash scripts/run_harness_signal_design_space.sh "${MODEL}"
