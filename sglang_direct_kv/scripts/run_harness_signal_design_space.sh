#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-harness_emitted frontend_supplied gateway_injected}"
if [[ "${SIGNAL_FAMILIES}" == "all" ]]; then
  SIGNAL_FAMILIES="harness_emitted frontend_supplied gateway_injected"
fi

REPORT_LABEL="${REPORT_LABEL:-signal_design_space_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-artifacts/results}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/runs/controlled/${REPORT_LABEL}}"
REPORT_DIR="${REPORT_DIR:-${RESULTS_ROOT}/reports/${REPORT_LABEL}}"
REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
UPDATE_LATEST="${UPDATE_LATEST:-1}"
SKIP_EXISTING_CASES="${SKIP_EXISTING_CASES:-1}"
HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code nemo_agent_toolkit pi_agent_harness openclaw hermes_agent}"
PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p3_high p5_boss_queue}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-ec2_a10g}"
HARDWARE_PROFILE_PATH="${HARDWARE_PROFILE_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

HARNESS_EMITTED_CACHE_MODES="${HARNESS_EMITTED_CACHE_MODES:-no_cache_signal harness_native_cache_lowered}"
HARNESS_EMITTED_PRIORITY_MODES="${HARNESS_EMITTED_PRIORITY_MODES:-no_prefetch nat_inferred_priority_hints}"
HARNESS_EMITTED_PRIORITY_HARNESSES="${HARNESS_EMITTED_PRIORITY_HARNESSES:-nemo_agent_toolkit}"
FRONTEND_SUPPLIED_MODES="${FRONTEND_SUPPLIED_MODES:-no_prefetch pre_harness_priority_hints}"
GATEWAY_INJECTED_MODES="${GATEWAY_INJECTED_MODES:-no_prefetch e2e_priority_hints}"
INCLUDE_HARNESS_EMITTED_PRIORITY="${INCLUDE_HARNESS_EMITTED_PRIORITY:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${REPORT_BUILDER_MODE}" != "lightweight" ]]; then
  echo "run_harness_signal_design_space.sh currently supports REPORT_BUILDER_MODE=lightweight only." >&2
  exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

word_in_list() {
  local needle="$1"
  local item
  for item in ${2}; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

append_unique_word() {
  local current="$1"
  local word="$2"
  if word_in_list "${word}" "${current}"; then
    echo "${current}"
  elif [[ -z "${current}" ]]; then
    echo "${word}"
  else
    echo "${current} ${word}"
  fi
}

intersect_words() {
  local left="$1"
  local right="$2"
  local out=""
  local item
  for item in ${left}; do
    if word_in_list "${item}" "${right}"; then
      out="$(append_unique_word "${out}" "${item}")"
    fi
  done
  echo "${out}"
}

validate_families() {
  local family
  for family in ${SIGNAL_FAMILIES}; do
    case "${family}" in
      harness_emitted|frontend_supplied|gateway_injected) ;;
      *)
        echo "Unknown SIGNAL_FAMILIES entry: ${family}" >&2
        echo "Supported: harness_emitted frontend_supplied gateway_injected all" >&2
        exit 2
        ;;
    esac
  done
}

load_hardware_profile() {
  if [[ -z "${HARDWARE_PROFILE}" || "${HARDWARE_PROFILE}" == "none" ]]; then
    return
  fi
  local profile_path="${HARDWARE_PROFILE_PATH}"
  if [[ -z "${profile_path}" ]]; then
    profile_path="configs/hardware/${HARDWARE_PROFILE}.env"
  fi
  if [[ ! -f "${profile_path}" ]]; then
    echo "Hardware profile not found: ${profile_path}" >&2
    echo "Set HARDWARE_PROFILE=none or HARDWARE_PROFILE_PATH=<file> to override." >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${profile_path}"
  HARDWARE_PROFILE_PATH="${profile_path}"
}

level_knobs_for_config() {
  local profile_value=""
  case "$1" in
    p0_control)
      profile_value="${P0_CONTROL_KNOBS:-}"
      if [[ -n "${profile_value}" ]]; then echo "${profile_value}"; return; fi
      echo "tool_wait_ms=500 target_prompt_tokens=1024 filler_sessions=0 filler_prompt_tokens=768 session_count=1 concurrency=1"
      ;;
    p1_mild)
      profile_value="${P1_MILD_KNOBS:-}"
      if [[ -n "${profile_value}" ]]; then echo "${profile_value}"; return; fi
      echo "tool_wait_ms=250 target_prompt_tokens=2048 filler_sessions=8 filler_prompt_tokens=1024 session_count=1 concurrency=4"
      ;;
    p2_medium)
      profile_value="${P2_MEDIUM_KNOBS:-}"
      if [[ -n "${profile_value}" ]]; then echo "${profile_value}"; return; fi
      echo "tool_wait_ms=100 target_prompt_tokens=3072 filler_sessions=16 filler_prompt_tokens=1536 session_count=1 concurrency=6"
      ;;
    p3_high)
      profile_value="${P3_HIGH_KNOBS:-}"
      if [[ -n "${profile_value}" ]]; then echo "${profile_value}"; return; fi
      echo "tool_wait_ms=50 target_prompt_tokens=4096 filler_sessions=32 filler_prompt_tokens=1536 session_count=1 concurrency=8"
      ;;
    p4_cliff)
      profile_value="${P4_CLIFF_KNOBS:-}"
      if [[ -n "${profile_value}" ]]; then echo "${profile_value}"; return; fi
      echo "tool_wait_ms=25 target_prompt_tokens=4096 filler_sessions=48 filler_prompt_tokens=2048 session_count=1 concurrency=10"
      ;;
    p5_boss_queue)
      profile_value="${P5_BOSS_QUEUE_KNOBS:-}"
      if [[ -n "${profile_value}" ]]; then echo "${profile_value}"; return; fi
      echo "tool_wait_ms=50 target_prompt_tokens=4096 filler_sessions=4 filler_prompt_tokens=2048 session_count=4 concurrency=12"
      ;;
    *) echo "" ;;
  esac
}

write_combined_run_config() {
  {
    echo "REPORT_LABEL=${REPORT_LABEL}"
    echo "MODEL=${MODEL}"
    echo "EXPERIMENT_KIND=harness_signal_design_space"
    echo "SIGNAL_FAMILIES=${SIGNAL_FAMILIES}"
    echo "SIGNAL_FAMILY_EXPANSION=${FAMILY_EXPANSION}"
    echo "RESULTS_ROOT=${RESULTS_ROOT}"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "REPORT_DIR=${REPORT_DIR}"
    echo "HARDWARE_PROFILE=${HARDWARE_PROFILE}"
    echo "HARDWARE_PROFILE_PATH=${HARDWARE_PROFILE_PATH}"
    echo "HARNESSES=${HARNESSES}"
    echo "MODES=${EXPANDED_MODES}"
    echo "HARNESS_EMITTED_CACHE_MODES=${HARNESS_EMITTED_CACHE_MODES}"
    echo "HARNESS_EMITTED_PRIORITY_MODES=${HARNESS_EMITTED_PRIORITY_MODES}"
    echo "FRONTEND_SUPPLIED_MODES=${FRONTEND_SUPPLIED_MODES}"
    echo "GATEWAY_INJECTED_MODES=${GATEWAY_INJECTED_MODES}"
    echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"
    echo "SKIP_EXISTING_CASES=${SKIP_EXISTING_CASES}"
    echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-}"
    echo "HICACHE_SIZE_GB=${HICACHE_SIZE_GB:-}"
    echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}"
    echo "P0_CONTROL=$(level_knobs_for_config p0_control | tr ' ' ',')"
    echo "P1_MILD=$(level_knobs_for_config p1_mild | tr ' ' ',')"
    echo "P2_MEDIUM=$(level_knobs_for_config p2_medium | tr ' ' ',')"
    echo "P3_QUEUE_PRESSURE=$(level_knobs_for_config p3_high | tr ' ' ',')"
    echo "P4_CLIFF=$(level_knobs_for_config p4_cliff | tr ' ' ',')"
    echo "P5_BOSS_QUEUE=$(level_knobs_for_config p5_boss_queue | tr ' ' ',')"
  } >"${REPORT_DIR}/run_config.env"
}

run_family_piece() {
  local family="$1"
  local detail="$2"
  local modes="$3"
  local harnesses="$4"
  if [[ -z "${harnesses}" ]]; then
    return
  fi
  echo
  echo "---- Signal family piece: ${family} / ${detail} ----"
  echo "harnesses: ${harnesses}"
  echo "expanded modes: ${modes}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  SIGNAL_FAMILY_ACTIVE="${family}" \
  SIGNAL_FAMILY_DETAIL="${detail}" \
  REPORT_LABEL="${REPORT_LABEL}" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  REPORT_DIR="${REPORT_DIR}" \
  UPDATE_LATEST=0 \
  SKIP_EXISTING_CASES="${SKIP_EXISTING_CASES}" \
  HARNESSES="${harnesses}" \
  MODES="${modes}" \
  PRESSURE_LEVELS="${PRESSURE_LEVELS}" \
  REPORT_BUILDER_MODE=lightweight \
  HARDWARE_PROFILE="${HARDWARE_PROFILE}" \
  HARDWARE_PROFILE_PATH="${HARDWARE_PROFILE_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/run_harness_deadline_pressure.sh "${MODEL}"
}

validate_families
load_hardware_profile

RESULTS_ROOT="$(mkdir -p "${RESULTS_ROOT}" && cd "${RESULTS_ROOT}" && pwd)"
RUN_ROOT="$(mkdir -p "${RUN_ROOT}" && cd "${RUN_ROOT}" && pwd)"
REPORT_DIR="$(mkdir -p "${REPORT_DIR}" && cd "${REPORT_DIR}" && pwd)"

EXPANDED_MODES=""
FAMILY_EXPANSION=""
for family in ${SIGNAL_FAMILIES}; do
  if [[ "${family}" == "harness_emitted" ]]; then
    for mode in ${HARNESS_EMITTED_CACHE_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "harness_emitted:cache")"
    if [[ "${INCLUDE_HARNESS_EMITTED_PRIORITY}" == "1" ]]; then
      native_priority_harnesses="$(intersect_words "${HARNESSES}" "${HARNESS_EMITTED_PRIORITY_HARNESSES}")"
      if [[ -n "${native_priority_harnesses}" ]]; then
        for mode in ${HARNESS_EMITTED_PRIORITY_MODES}; do
          EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
        done
        FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "harness_emitted:priority")"
      fi
    fi
  elif [[ "${family}" == "frontend_supplied" ]]; then
    for mode in ${FRONTEND_SUPPLIED_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "frontend_supplied:priority")"
  elif [[ "${family}" == "gateway_injected" ]]; then
    for mode in ${GATEWAY_INJECTED_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "gateway_injected:priority")"
  fi
done

echo "Harness Signal Design Space"
echo "MODEL=${MODEL}"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "HARDWARE_PROFILE=${HARDWARE_PROFILE}"
echo "HARDWARE_PROFILE_PATH=${HARDWARE_PROFILE_PATH}"
echo "SIGNAL_FAMILIES=${SIGNAL_FAMILIES}"
echo "HARNESSES=${HARNESSES}"
echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"
echo
echo "Expanded family pieces:"
if word_in_list "harness_emitted" "${SIGNAL_FAMILIES}"; then
  echo "- harness_emitted/cache -> ${HARNESS_EMITTED_CACHE_MODES}"
  if [[ "${INCLUDE_HARNESS_EMITTED_PRIORITY}" == "1" ]]; then
    native_priority_harnesses="$(intersect_words "${HARNESSES}" "${HARNESS_EMITTED_PRIORITY_HARNESSES}")"
    if [[ -n "${native_priority_harnesses}" ]]; then
      echo "- harness_emitted/priority -> ${HARNESS_EMITTED_PRIORITY_MODES} for ${native_priority_harnesses}"
    fi
  fi
fi
if word_in_list "frontend_supplied" "${SIGNAL_FAMILIES}"; then
  echo "- frontend_supplied -> ${FRONTEND_SUPPLIED_MODES}"
fi
if word_in_list "gateway_injected" "${SIGNAL_FAMILIES}"; then
  echo "- gateway_injected -> ${GATEWAY_INJECTED_MODES}"
fi
echo "Combined mode set for final report: ${EXPANDED_MODES}"

if word_in_list "harness_emitted" "${SIGNAL_FAMILIES}"; then
  run_family_piece "harness_emitted" "cache" "${HARNESS_EMITTED_CACHE_MODES}" "${HARNESSES}"
  if [[ "${INCLUDE_HARNESS_EMITTED_PRIORITY}" == "1" ]]; then
    native_priority_harnesses="$(intersect_words "${HARNESSES}" "${HARNESS_EMITTED_PRIORITY_HARNESSES}")"
    run_family_piece "harness_emitted" "priority" "${HARNESS_EMITTED_PRIORITY_MODES}" "${native_priority_harnesses}"
  fi
fi

if word_in_list "frontend_supplied" "${SIGNAL_FAMILIES}"; then
  run_family_piece "frontend_supplied" "priority" "${FRONTEND_SUPPLIED_MODES}" "${HARNESSES}"
fi

if word_in_list "gateway_injected" "${SIGNAL_FAMILIES}"; then
  run_family_piece "gateway_injected" "priority" "${GATEWAY_INJECTED_MODES}" "${HARNESSES}"
fi

write_combined_run_config

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry run complete. No experiments or reports were executed."
  exit 0
fi

latest_args=()
if [[ "${UPDATE_LATEST}" == "1" ]]; then
  latest_args=(--update-latest)
fi

"${PYTHON_BIN}" scripts/build_multi_harness_deadline_summary.py \
  --root "${RUN_ROOT}" \
  --out-dir "${REPORT_DIR}" \
  --latest-root "${RESULTS_ROOT}" \
  --report-label "${REPORT_LABEL}" \
  --run-config "${REPORT_DIR}/run_config.env" \
  "${latest_args[@]}"

echo
echo "Done."
echo "Latest report: ${RESULTS_ROOT}/latest_master_report.html"
echo "Archived labeled report: ${REPORT_DIR}/master_report.html"
