from __future__ import annotations

import os


NAT_INFERRED_PRIORITY_MODE = "nat_inferred_priority_hints"
HARNESS_NATIVE_CACHE_MODE = "harness_native_cache_lowered"
HARNESS_EMITTED_SIGNAL_MODE = "harness_emitted_signals"
CONTROLLER_OBSERVE_ONLY_MODE = "controller_observe_only"
CONTROLLER_SCHEDULER_PRIORITY_MODE = "controller_scheduler_priority"
CONTROLLER_SPECULATIVE_PRELOAD_MODE = "controller_speculative_preload"
CONTROLLER_TARGETED_KV_PREFETCH_MODE = "controller_targeted_kv_prefetch"
CONTROLLER_DEMOTE_RESTORE_MODE = "controller_demote_restore"
CONTROLLER_PRIORITY_DEMOTE_MODE = "controller_priority_demote"
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODE = "controller_priority_demotion_admission"
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SOFT_MODE = "controller_priority_demotion_admission_soft"
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MEDIUM_MODE = "controller_priority_demotion_admission_medium"
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_HARD_MODE = "controller_priority_demotion_admission_hard"
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_EARLYPREPARE_MODE = "controller_priority_demotion_admission_earlyprepare"
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SHORTHAND_MODE = "controller_priority_demotion_admission_shorthand"
CONTROLLER_ORACLE_TIMELINE_MODE = "controller_oracle_timeline"
CONTROLLER_ORACLE_SAFE_SJF_MODE = "controller_oracle_safe_sjf"
CONTROLLER_ORACLE_SAFE_SJF_BALANCED_MODE = "controller_oracle_safe_sjf_balanced"
CONTROLLER_ORACLE_SAFE_SJF_AGGRESSIVE_MODE = "controller_oracle_safe_sjf_aggressive"
CONTROLLER_ORACLE_SAFE_SJF_MAXFILL_MODE = "controller_oracle_safe_sjf_maxfill"
CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE = "controller_priority_demotion_calibrated_admission"
CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE = "controller_oracle_exact_runtime_admission"
CONTROLLER_ADMISSION_CONTROL_MODE = "controller_admission_control"
CONTROLLER_FULL_MODE = "controller_full"
CONTROLLER_FULL_CHUNKED_PREFILL_MODE = "controller_full_chunked_prefill"
STORAGE_HICACHE_BASELINE_MODE = "storage_hicache_baseline"
STORAGE_HICACHE_CONTROLLER_PREFETCH_MODE = "storage_hicache_controller_prefetch"

CONTROLLER_ORACLE_SAFE_SJF_MODES = {
    CONTROLLER_ORACLE_SAFE_SJF_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_BALANCED_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_AGGRESSIVE_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_MAXFILL_MODE,
    CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE,
    CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE,
}

CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES = {
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SOFT_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MEDIUM_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_HARD_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_EARLYPREPARE_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SHORTHAND_MODE,
    CONTROLLER_ORACLE_TIMELINE_MODE,
    *CONTROLLER_ORACLE_SAFE_SJF_MODES,
}

SUPPORTED_MODES = (
    "no_prefetch",
    "e2e_priority_hints",
    "pre_harness_priority_hints",
    NAT_INFERRED_PRIORITY_MODE,
    "e2e_priority_hints_speculative_prefill",
    "no_cache_signal",
    HARNESS_NATIVE_CACHE_MODE,
    HARNESS_EMITTED_SIGNAL_MODE,
    CONTROLLER_OBSERVE_ONLY_MODE,
    CONTROLLER_SCHEDULER_PRIORITY_MODE,
    CONTROLLER_SPECULATIVE_PRELOAD_MODE,
    CONTROLLER_TARGETED_KV_PREFETCH_MODE,
    CONTROLLER_DEMOTE_RESTORE_MODE,
    CONTROLLER_PRIORITY_DEMOTE_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SOFT_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MEDIUM_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_HARD_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_EARLYPREPARE_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SHORTHAND_MODE,
    CONTROLLER_ORACLE_TIMELINE_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_BALANCED_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_AGGRESSIVE_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_MAXFILL_MODE,
    CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE,
    CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE,
    CONTROLLER_ADMISSION_CONTROL_MODE,
    CONTROLLER_FULL_MODE,
    CONTROLLER_FULL_CHUNKED_PREFILL_MODE,
    STORAGE_HICACHE_BASELINE_MODE,
    STORAGE_HICACHE_CONTROLLER_PREFETCH_MODE,
)


def controller_observe_only_mode(mode: str) -> bool:
    return mode == CONTROLLER_OBSERVE_ONLY_MODE


def controller_scheduler_priority_mode(mode: str) -> bool:
    return mode == CONTROLLER_SCHEDULER_PRIORITY_MODE


def controller_speculative_preload_mode(mode: str) -> bool:
    return mode in {CONTROLLER_SPECULATIVE_PRELOAD_MODE, STORAGE_HICACHE_CONTROLLER_PREFETCH_MODE}


def controller_targeted_kv_prefetch_mode(mode: str) -> bool:
    return mode == CONTROLLER_TARGETED_KV_PREFETCH_MODE


def controller_demote_restore_mode(mode: str) -> bool:
    return mode in {
        CONTROLLER_DEMOTE_RESTORE_MODE,
        CONTROLLER_PRIORITY_DEMOTE_MODE,
        *CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES,
    }


def controller_priority_demotion_admission_mode(mode: str) -> bool:
    return mode in CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES


def controller_admission_aggressiveness(mode: str) -> str:
    if mode == CONTROLLER_ORACLE_TIMELINE_MODE or mode in CONTROLLER_ORACLE_SAFE_SJF_MODES:
        return "oracle_timeline"
    if mode == CONTROLLER_PRIORITY_DEMOTION_ADMISSION_EARLYPREPARE_MODE:
        return "earlyprepare"
    if mode == CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SOFT_MODE:
        return "soft"
    if mode == CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MEDIUM_MODE:
        return "medium"
    if mode == CONTROLLER_PRIORITY_DEMOTION_ADMISSION_HARD_MODE:
        return "hard"
    configured = os.environ.get("CONTROLLER_ADMISSION_AGGRESSIVENESS", "hard").strip().lower()
    return configured if configured in {"soft", "medium", "hard"} else "hard"


def controller_admission_lead_ms(mode: str, wait_ms: int) -> float:
    aggressiveness = controller_admission_aggressiveness(mode)
    if aggressiveness == "oracle_timeline":
        return float(wait_ms)
    if aggressiveness == "earlyprepare":
        configured = int(os.environ.get("CONTROLLER_EARLYPREPARE_LEAD_MS", "500") or "500")
        return float(min(wait_ms, max(0, configured)))
    if aggressiveness == "hard":
        return float(wait_ms)
    if aggressiveness == "medium":
        return max(25.0, float(wait_ms) * 0.5)
    return max(25.0, float(wait_ms) * 0.25)


def controller_admission_control_mode(mode: str) -> bool:
    return mode == CONTROLLER_ADMISSION_CONTROL_MODE


def controller_safe_sjf_degree(mode: str) -> tuple[int, int, bool, bool]:
    configured_margin = int(os.environ.get("CONTROLLER_ORACLE_SAFETY_MARGIN_MS", "150") or "150")
    configured_idle_override = os.environ.get("CONTROLLER_ORACLE_IDLE_OVERRIDE", "").strip().lower()
    idle_override_enabled = configured_idle_override in {"1", "true", "yes", "on"}
    if mode == CONTROLLER_ORACLE_SAFE_SJF_MODE:
        return (
            configured_margin,
            int(os.environ.get("CONTROLLER_ORACLE_SAFE_SJF_MAX_IN_FLIGHT", "1") or "1"),
            True,
            idle_override_enabled,
        )
    if mode == CONTROLLER_ORACLE_SAFE_SJF_BALANCED_MODE:
        return 75, 3, True, True
    if mode == CONTROLLER_ORACLE_SAFE_SJF_AGGRESSIVE_MODE:
        return 25, 6, True, True
    if mode == CONTROLLER_ORACLE_SAFE_SJF_MAXFILL_MODE:
        return 0, 12, False, True
    if mode == CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE:
        return (
            int(os.environ.get("CONTROLLER_CALIBRATED_SAFETY_MARGIN_MS", "500") or "500"),
            int(os.environ.get("CONTROLLER_CALIBRATED_MAX_IN_FLIGHT", "1") or "1"),
            True,
            os.environ.get("CONTROLLER_CALIBRATED_IDLE_OVERRIDE", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )
    if mode == CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE:
        return (
            int(os.environ.get("CONTROLLER_ORACLE_EXACT_SAFETY_MARGIN_MS", "50") or "50"),
            int(os.environ.get("CONTROLLER_ORACLE_EXACT_MAX_IN_FLIGHT", "6") or "6"),
            True,
            os.environ.get("CONTROLLER_ORACLE_EXACT_IDLE_OVERRIDE", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )
    return configured_margin, 1, True, idle_override_enabled


def controller_full_mode(mode: str) -> bool:
    return mode in {CONTROLLER_FULL_MODE, CONTROLLER_FULL_CHUNKED_PREFILL_MODE}


def controller_mode(mode: str) -> bool:
    return mode in {
        CONTROLLER_OBSERVE_ONLY_MODE,
        CONTROLLER_SCHEDULER_PRIORITY_MODE,
        CONTROLLER_SPECULATIVE_PRELOAD_MODE,
        CONTROLLER_TARGETED_KV_PREFETCH_MODE,
        CONTROLLER_DEMOTE_RESTORE_MODE,
        CONTROLLER_PRIORITY_DEMOTE_MODE,
        *CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES,
        CONTROLLER_ADMISSION_CONTROL_MODE,
        CONTROLLER_FULL_MODE,
        CONTROLLER_FULL_CHUNKED_PREFILL_MODE,
        STORAGE_HICACHE_CONTROLLER_PREFETCH_MODE,
    }


def storage_hicache_mode(mode: str) -> bool:
    return mode in {STORAGE_HICACHE_BASELINE_MODE, STORAGE_HICACHE_CONTROLLER_PREFETCH_MODE}
