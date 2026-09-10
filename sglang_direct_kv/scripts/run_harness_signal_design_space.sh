#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline harness_emitted frontend_supplied gateway_injected}"
if [[ "${SIGNAL_FAMILIES}" == "all" ]]; then
  SIGNAL_FAMILIES="baseline harness_emitted frontend_supplied gateway_injected controller_observe controller_scheduler controller_preload controller_targeted_prefetch controller_demote_restore controller_priority_demote controller_admission controller_full controller_full_chunked"
fi

REPORT_LABEL="${REPORT_LABEL:-signal_design_space_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-artifacts/results}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/runs/controlled/${REPORT_LABEL}}"
REPORT_DIR="${REPORT_DIR:-${RESULTS_ROOT}/reports/${REPORT_LABEL}}"
REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
UPDATE_LATEST="${UPDATE_LATEST:-1}"
SKIP_EXISTING_CASES="${SKIP_EXISTING_CASES:-1}"
SKIP_INTERIM_REPORTS="${SKIP_INTERIM_REPORTS:-0}"
HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code nemo_agent_toolkit pi_agent_harness openclaw hermes_agent}"
PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p3_high p5_boss_queue}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-ec2_a10g}"
HARDWARE_PROFILE_PATH="${HARDWARE_PROFILE_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FILLER_REPLAY_DEADLINES="${FILLER_REPLAY_DEADLINES:-0}"
FILLER_REPLAY_DEADLINE_MS="${FILLER_REPLAY_DEADLINE_MS:-}"
TOOL_WAIT_PROFILE="${TOOL_WAIT_PROFILE:-fixed}"
TOOL_WAIT_PROFILE_SPEC="${TOOL_WAIT_PROFILE_SPEC:-}"
TOOL_WAIT_SEED="${TOOL_WAIT_SEED:-42}"
TASK_REPLAY_STEPS="${TASK_REPLAY_STEPS:-1}"
CONTROLLER_CHUNKED_PREFILL_SIZE="${CONTROLLER_CHUNKED_PREFILL_SIZE:-512}"
CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS="${CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS:-4096}"
CONTROLLER_CHUNKED_PREFILL_MAX_REQUESTS="${CONTROLLER_CHUNKED_PREFILL_MAX_REQUESTS:-}"

BASELINE_MODES="${BASELINE_MODES:-no_prefetch}"
HARNESS_EMITTED_MODES="${HARNESS_EMITTED_MODES:-harness_emitted_signals}"
FRONTEND_SUPPLIED_MODES="${FRONTEND_SUPPLIED_MODES:-pre_harness_priority_hints}"
GATEWAY_INJECTED_MODES="${GATEWAY_INJECTED_MODES:-e2e_priority_hints}"
CONTROLLER_OBSERVE_MODES="${CONTROLLER_OBSERVE_MODES:-controller_observe_only}"
CONTROLLER_SCHEDULER_MODES="${CONTROLLER_SCHEDULER_MODES:-controller_scheduler_priority}"
CONTROLLER_PRELOAD_MODES="${CONTROLLER_PRELOAD_MODES:-controller_speculative_preload}"
CONTROLLER_TARGETED_PREFETCH_MODES="${CONTROLLER_TARGETED_PREFETCH_MODES:-controller_targeted_kv_prefetch}"
CONTROLLER_DEMOTE_RESTORE_MODES="${CONTROLLER_DEMOTE_RESTORE_MODES:-controller_demote_restore}"
CONTROLLER_PRIORITY_DEMOTE_MODES="${CONTROLLER_PRIORITY_DEMOTE_MODES:-controller_priority_demote}"
CONTROLLER_ADMISSION_MODES="${CONTROLLER_ADMISSION_MODES:-controller_admission_control}"
CONTROLLER_FULL_MODES="${CONTROLLER_FULL_MODES:-controller_full}"
CONTROLLER_FULL_CHUNKED_MODES="${CONTROLLER_FULL_CHUNKED_MODES:-controller_full_chunked_prefill}"
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
      baseline|harness_emitted|frontend_supplied|gateway_injected|controller_observe|controller_scheduler|controller_preload|controller_targeted_prefetch|controller_demote_restore|controller_priority_demote|controller_admission|controller_full|controller_full_chunked) ;;
      *)
        echo "Unknown SIGNAL_FAMILIES entry: ${family}" >&2
        echo "Supported: baseline harness_emitted frontend_supplied gateway_injected controller_observe controller_scheduler controller_preload controller_targeted_prefetch controller_demote_restore controller_priority_demote controller_admission controller_full controller_full_chunked all" >&2
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
    echo "BASELINE_MODES=${BASELINE_MODES}"
    echo "HARNESS_EMITTED_MODES=${HARNESS_EMITTED_MODES}"
    echo "FRONTEND_SUPPLIED_MODES=${FRONTEND_SUPPLIED_MODES}"
    echo "GATEWAY_INJECTED_MODES=${GATEWAY_INJECTED_MODES}"
    echo "CONTROLLER_OBSERVE_MODES=${CONTROLLER_OBSERVE_MODES}"
    echo "CONTROLLER_SCHEDULER_MODES=${CONTROLLER_SCHEDULER_MODES}"
    echo "CONTROLLER_PRELOAD_MODES=${CONTROLLER_PRELOAD_MODES}"
    echo "CONTROLLER_TARGETED_PREFETCH_MODES=${CONTROLLER_TARGETED_PREFETCH_MODES}"
    echo "CONTROLLER_DEMOTE_RESTORE_MODES=${CONTROLLER_DEMOTE_RESTORE_MODES}"
    echo "CONTROLLER_PRIORITY_DEMOTE_MODES=${CONTROLLER_PRIORITY_DEMOTE_MODES}"
    echo "CONTROLLER_ADMISSION_MODES=${CONTROLLER_ADMISSION_MODES}"
    echo "CONTROLLER_FULL_MODES=${CONTROLLER_FULL_MODES}"
    echo "CONTROLLER_FULL_CHUNKED_MODES=${CONTROLLER_FULL_CHUNKED_MODES}"
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
    echo "FILLER_REPLAY_DEADLINES=${FILLER_REPLAY_DEADLINES}"
    echo "FILLER_REPLAY_DEADLINE_MS=${FILLER_REPLAY_DEADLINE_MS}"
    echo "TOOL_WAIT_PROFILE=${TOOL_WAIT_PROFILE}"
    echo "TOOL_WAIT_PROFILE_SPEC=${TOOL_WAIT_PROFILE_SPEC}"
    echo "TOOL_WAIT_SEED=${TOOL_WAIT_SEED}"
    echo "TASK_REPLAY_STEPS=${TASK_REPLAY_STEPS}"
    echo "CONTROLLER_CHUNKED_PREFILL_SIZE=${CONTROLLER_CHUNKED_PREFILL_SIZE}"
    echo "CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS=${CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS}"
    echo "CONTROLLER_CHUNKED_PREFILL_MAX_REQUESTS=${CONTROLLER_CHUNKED_PREFILL_MAX_REQUESTS}"
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
  SKIP_REPORT_BUILD="${SKIP_INTERIM_REPORTS}" \
  SKIP_EXISTING_CASES="${SKIP_EXISTING_CASES}" \
  HARNESSES="${harnesses}" \
  MODES="${modes}" \
  PRESSURE_LEVELS="${PRESSURE_LEVELS}" \
  REPORT_BUILDER_MODE=lightweight \
  HARDWARE_PROFILE="${HARDWARE_PROFILE}" \
  HARDWARE_PROFILE_PATH="${HARDWARE_PROFILE_PATH}" \
  FILLER_REPLAY_DEADLINES="${FILLER_REPLAY_DEADLINES}" \
  FILLER_REPLAY_DEADLINE_MS="${FILLER_REPLAY_DEADLINE_MS}" \
  TOOL_WAIT_PROFILE="${TOOL_WAIT_PROFILE}" \
  TOOL_WAIT_PROFILE_SPEC="${TOOL_WAIT_PROFILE_SPEC}" \
  TOOL_WAIT_SEED="${TOOL_WAIT_SEED}" \
  TASK_REPLAY_STEPS="${TASK_REPLAY_STEPS}" \
  CONTROLLER_CHUNKED_PREFILL_SIZE="${CONTROLLER_CHUNKED_PREFILL_SIZE}" \
  CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS="${CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS}" \
  CONTROLLER_CHUNKED_PREFILL_MAX_REQUESTS="${CONTROLLER_CHUNKED_PREFILL_MAX_REQUESTS}" \
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
  if [[ "${family}" == "baseline" ]]; then
    for mode in ${BASELINE_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "baseline")"
  elif [[ "${family}" == "harness_emitted" ]]; then
    for mode in ${HARNESS_EMITTED_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "harness_emitted")"
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
  elif [[ "${family}" == "controller_observe" ]]; then
    for mode in ${CONTROLLER_OBSERVE_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_observe:lifecycle")"
  elif [[ "${family}" == "controller_scheduler" ]]; then
    for mode in ${CONTROLLER_SCHEDULER_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_scheduler:priority")"
  elif [[ "${family}" == "controller_preload" ]]; then
    for mode in ${CONTROLLER_PRELOAD_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_preload:speculative_kv")"
  elif [[ "${family}" == "controller_targeted_prefetch" ]]; then
    for mode in ${CONTROLLER_TARGETED_PREFETCH_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_targeted_prefetch:direct_kv")"
  elif [[ "${family}" == "controller_demote_restore" ]]; then
    for mode in ${CONTROLLER_DEMOTE_RESTORE_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_demote_restore:background_priority_window")"
  elif [[ "${family}" == "controller_priority_demote" ]]; then
    for mode in ${CONTROLLER_PRIORITY_DEMOTE_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_priority_demote:priority_plus_aggressive_demotion")"
  elif [[ "${family}" == "controller_admission" ]]; then
    for mode in ${CONTROLLER_ADMISSION_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_admission:bounded_speculative_work")"
  elif [[ "${family}" == "controller_full" ]]; then
    for mode in ${CONTROLLER_FULL_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_full:demote_priority_admission_ladder")"
  elif [[ "${family}" == "controller_full_chunked" ]]; then
    for mode in ${CONTROLLER_FULL_CHUNKED_MODES}; do
      EXPANDED_MODES="$(append_unique_word "${EXPANDED_MODES}" "${mode}")"
    done
    FAMILY_EXPANSION="$(append_unique_word "${FAMILY_EXPANSION}" "controller_full_chunked:demote_priority_admission_ladder_chunked_prefill")"
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
echo "TOOL_WAIT_PROFILE=${TOOL_WAIT_PROFILE}"
echo "TASK_REPLAY_STEPS=${TASK_REPLAY_STEPS}"
echo "TOOL_WAIT_SEED=${TOOL_WAIT_SEED}"
if [[ -n "${TOOL_WAIT_PROFILE_SPEC}" ]]; then
  echo "TOOL_WAIT_PROFILE_SPEC=${TOOL_WAIT_PROFILE_SPEC}"
fi
echo
echo "Expanded family pieces:"
if word_in_list "baseline" "${SIGNAL_FAMILIES}"; then
  echo "- baseline -> ${BASELINE_MODES}"
fi
if word_in_list "harness_emitted" "${SIGNAL_FAMILIES}"; then
  echo "- harness_emitted -> ${HARNESS_EMITTED_MODES}"
fi
if word_in_list "frontend_supplied" "${SIGNAL_FAMILIES}"; then
  echo "- frontend_supplied -> ${FRONTEND_SUPPLIED_MODES}"
fi
if word_in_list "gateway_injected" "${SIGNAL_FAMILIES}"; then
  echo "- gateway_injected -> ${GATEWAY_INJECTED_MODES}"
fi
if word_in_list "controller_observe" "${SIGNAL_FAMILIES}"; then
  echo "- controller_observe -> ${CONTROLLER_OBSERVE_MODES}"
fi
if word_in_list "controller_scheduler" "${SIGNAL_FAMILIES}"; then
  echo "- controller_scheduler -> ${CONTROLLER_SCHEDULER_MODES}"
fi
if word_in_list "controller_preload" "${SIGNAL_FAMILIES}"; then
  echo "- controller_preload -> ${CONTROLLER_PRELOAD_MODES}"
fi
if word_in_list "controller_targeted_prefetch" "${SIGNAL_FAMILIES}"; then
  echo "- controller_targeted_prefetch -> ${CONTROLLER_TARGETED_PREFETCH_MODES}"
fi
if word_in_list "controller_demote_restore" "${SIGNAL_FAMILIES}"; then
  echo "- controller_demote_restore -> ${CONTROLLER_DEMOTE_RESTORE_MODES}"
fi
if word_in_list "controller_priority_demote" "${SIGNAL_FAMILIES}"; then
  echo "- controller_priority_demote -> ${CONTROLLER_PRIORITY_DEMOTE_MODES}"
fi
if word_in_list "controller_admission" "${SIGNAL_FAMILIES}"; then
  echo "- controller_admission -> ${CONTROLLER_ADMISSION_MODES}"
fi
if word_in_list "controller_full" "${SIGNAL_FAMILIES}"; then
  echo "- controller_full -> ${CONTROLLER_FULL_MODES}"
fi
if word_in_list "controller_full_chunked" "${SIGNAL_FAMILIES}"; then
  echo "- controller_full_chunked -> ${CONTROLLER_FULL_CHUNKED_MODES}"
fi
echo "Combined mode set for final report: ${EXPANDED_MODES}"

if word_in_list "baseline" "${SIGNAL_FAMILIES}"; then
  run_family_piece "baseline" "none" "${BASELINE_MODES}" "${HARNESSES}"
fi

if word_in_list "harness_emitted" "${SIGNAL_FAMILIES}"; then
  run_family_piece "harness_emitted" "signals" "${HARNESS_EMITTED_MODES}" "${HARNESSES}"
fi

if word_in_list "frontend_supplied" "${SIGNAL_FAMILIES}"; then
  run_family_piece "frontend_supplied" "priority" "${FRONTEND_SUPPLIED_MODES}" "${HARNESSES}"
fi

if word_in_list "gateway_injected" "${SIGNAL_FAMILIES}"; then
  run_family_piece "gateway_injected" "priority" "${GATEWAY_INJECTED_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_observe" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_observe" "lifecycle" "${CONTROLLER_OBSERVE_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_scheduler" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_scheduler" "priority" "${CONTROLLER_SCHEDULER_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_preload" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_preload" "speculative_kv" "${CONTROLLER_PRELOAD_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_targeted_prefetch" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_targeted_prefetch" "direct_kv" "${CONTROLLER_TARGETED_PREFETCH_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_demote_restore" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_demote_restore" "background_priority_window" "${CONTROLLER_DEMOTE_RESTORE_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_priority_demote" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_priority_demote" "priority_plus_aggressive_demotion" "${CONTROLLER_PRIORITY_DEMOTE_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_admission" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_admission" "bounded_speculative_work" "${CONTROLLER_ADMISSION_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_full" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_full" "demote_priority_admission_ladder" "${CONTROLLER_FULL_MODES}" "${HARNESSES}"
fi

if word_in_list "controller_full_chunked" "${SIGNAL_FAMILIES}"; then
  run_family_piece "controller_full_chunked" "demote_priority_admission_ladder_chunked_prefill" "${CONTROLLER_FULL_CHUNKED_MODES}" "${HARNESSES}"
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
