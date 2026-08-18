#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}}"
EXPERIMENT_KIND="${EXPERIMENT_KIND:-both}"
REPORT_LABEL="${REPORT_LABEL:-master_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-artifacts/results}"
UPDATE_LATEST="${UPDATE_LATEST:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_TOPLEVEL="${CLEAN_TOPLEVEL:-1}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-18}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

case "${EXPERIMENT_KIND}" in
  controlled|live|both) ;;
  *)
    echo "ERROR: EXPERIMENT_KIND must be one of: controlled, live, both" >&2
    exit 2
    ;;
esac

RESULTS_ROOT="$(mkdir -p "${RESULTS_ROOT}" && cd "${RESULTS_ROOT}" && pwd)"
REPORTS_ROOT="${RESULTS_ROOT}/reports"
RUNS_ROOT="${RESULTS_ROOT}/runs"
REPORT_DIR="${REPORTS_ROOT}/${REPORT_LABEL}"
CONTROLLED_RUN_ROOT="${CONTROLLED_ROOT:-${RUNS_ROOT}/controlled/${REPORT_LABEL}}"
LIVE_DIRECT_RUN_ROOT="${LIVE_DIRECT_ROOT:-${RUNS_ROOT}/live/${REPORT_LABEL}}"
SCRATCH_LATEST_ROOT="${REPORT_DIR}/_latest_scratch"

discover_controlled_root() {
  if [[ -n "${CONTROLLED_ROOT:-}" ]]; then
    printf '%s\n' "${CONTROLLED_ROOT}"
    return
  fi
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

roots = []
patterns = [
    "artifacts/results/runs/controlled/*",
    "artifacts/results/milestone27_two_mode_*",
    "artifacts/results/milestone27_real_prompt_controlled_replay_*",
]
for pattern in patterns:
    roots.extend(path for path in Path(".").glob(pattern) if path.is_dir())

def evidence_score(root: Path) -> int:
    score = 0
    if (root / "controlled_replay_report" / "controlled_replay_report.html").exists():
        score += 1000
    if (root / "controlled_replay_report" / "controlled_replay_gaps.csv").exists():
        score += 500
    score += len(list(root.glob("*/m27_trace.jsonl")))
    return score

valid = [root for root in roots if evidence_score(root) > 0]
if valid:
    best = max(valid, key=lambda path: (evidence_score(path), path.stat().st_mtime, str(path)))
    print(best)
PY
}

discover_live_root() {
  if [[ -n "${LIVE_DIRECT_ROOT:-}" ]]; then
    printf '%s\n' "${LIVE_DIRECT_ROOT}"
    return
  fi
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import csv

roots = []
patterns = [
    "artifacts/results/runs/live/*",
    "artifacts/results/milestone26_live_direct_only_*",
    "artifacts/results/milestone26_live_direct_kv_load_*",
    "artifacts/results/milestone26_direct_kv_*",
    "artifacts/results/milestone26_paired_direct_kv_*/live_direct_kv_load",
]
for pattern in patterns:
    roots.extend(path for path in Path(".").glob(pattern) if path.is_dir())

def count_gaps(root: Path) -> int:
    for rel in ("live_agentbench_prefetch_report/live_tool_gaps.csv", "live_tool_gaps.csv"):
        path = root / rel
        if path.exists():
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    return sum(1 for _ in csv.DictReader(handle))
            except Exception:
                return 0
    return 0

if roots:
    best = max(roots, key=lambda path: (count_gaps(path), path.stat().st_mtime, str(path)))
    print(best)
PY
}

clean_toplevel() {
  if [[ "${CLEAN_TOPLEVEL}" != "1" ]]; then
    return
  fi
  local archive_dir="${RESULTS_ROOT}/archive/toplevel_cleanup_$(date +%Y%m%d_%H%M%S)"
  local moved=0
  local path
  while IFS= read -r path; do
    local base
    base="$(basename "${path}")"
    case "${base}" in
      latest_master_report.html|latest_synthetic_master_report.html|latest_manifest.json)
        continue
        ;;
    esac
    mkdir -p "${archive_dir}"
    mv "${path}" "${archive_dir}/${base}"
    moved=1
  done < <(find "${RESULTS_ROOT}" -maxdepth 1 -type f -print)

  if [[ "${moved}" == "1" ]]; then
    echo "Archived loose top-level result files under: ${archive_dir}"
  fi
}

write_manifest() {
  local latest_report="${1:-}"
  local archived_report="${2:-}"
  local controlled_root="${3:-}"
  local live_root="${4:-}"
  local manifest_path="${REPORT_DIR}/manifest.json"
  local latest_manifest_path="${RESULTS_ROOT}/latest_manifest.json"

  REPORT_LABEL="${REPORT_LABEL}" \
  EXPERIMENT_KIND="${EXPERIMENT_KIND}" \
  MODEL="${MODEL}" \
  CONTROLLED_ROOT_VALUE="${controlled_root}" \
  LIVE_ROOT_VALUE="${live_root}" \
  LATEST_REPORT_VALUE="${latest_report}" \
  ARCHIVED_REPORT_VALUE="${archived_report}" \
  SCRIPT_VALUE="scripts/run_master_report.sh" \
  UPDATE_LATEST="${UPDATE_LATEST}" \
  MANIFEST_PATH="${manifest_path}" \
  LATEST_MANIFEST_PATH="${latest_manifest_path}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "report_label": os.environ["REPORT_LABEL"],
    "experiment_kind": os.environ["EXPERIMENT_KIND"],
    "model": os.environ["MODEL"],
    "controlled_root": os.environ.get("CONTROLLED_ROOT_VALUE", ""),
    "live_root": os.environ.get("LIVE_ROOT_VALUE", ""),
    "latest_report": os.environ.get("LATEST_REPORT_VALUE", ""),
    "archived_report": os.environ.get("ARCHIVED_REPORT_VALUE", ""),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "script": os.environ["SCRIPT_VALUE"],
}

manifest_path = Path(os.environ["MANIFEST_PATH"])
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if os.environ.get("UPDATE_LATEST") == "1":
    latest_manifest_path = Path(os.environ["LATEST_MANIFEST_PATH"])
    latest_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

print_config() {
  cat <<EOF
Master report run
MODEL=${MODEL}
EXPERIMENT_KIND=${EXPERIMENT_KIND}
REPORT_LABEL=${REPORT_LABEL}
RESULTS_ROOT=${RESULTS_ROOT}
REPORT_DIR=${REPORT_DIR}
CONTROLLED_RUN_ROOT=${CONTROLLED_RUN_ROOT}
LIVE_DIRECT_RUN_ROOT=${LIVE_DIRECT_RUN_ROOT}
UPDATE_LATEST=${UPDATE_LATEST}
BUILD_ONLY=${BUILD_ONLY}
DRY_RUN=${DRY_RUN}
CLEAN_TOPLEVEL=${CLEAN_TOPLEVEL}
MAX_TIMELINE_GAPS=${MAX_TIMELINE_GAPS}
EOF
}

run_controlled() {
  if [[ "${BUILD_ONLY}" == "1" ]]; then
    CONTROLLED_RUN_ROOT="$(discover_controlled_root)"
    echo "Build-only: using controlled root: ${CONTROLLED_RUN_ROOT}"
    return
  fi

  echo
  echo "Running controlled replay experiment."
  RESULT_ROOT="${CONTROLLED_RUN_ROOT}" \
  LATEST_REPORT_ROOT="${CONTROLLED_RUN_ROOT}/_latest_scratch" \
  MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS}" \
  bash scripts/run_milestone27_real_prompt_controlled_replay.sh "${MODEL}"
}

run_live() {
  if [[ "${BUILD_ONLY}" == "1" ]]; then
    LIVE_DIRECT_RUN_ROOT="$(discover_live_root)"
    echo "Build-only: using live root: ${LIVE_DIRECT_RUN_ROOT}"
    return
  fi

  echo
  echo "Running live AgentBench direct-prefetch experiment."
  RESULT_ROOT="${LIVE_DIRECT_RUN_ROOT}" \
  LATEST_REPORT_ROOT="${LIVE_DIRECT_RUN_ROOT}/_latest_scratch" \
  LIVE_PREFETCH_ACTION="${LIVE_PREFETCH_ACTION:-direct_load}" \
  MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS}" \
  bash scripts/run_milestone26_live_direct_kv_load_intervention.sh "${MODEL}"
}

build_report() {
  local build_latest_root="${SCRATCH_LATEST_ROOT}"
  if [[ "${UPDATE_LATEST}" == "1" ]]; then
    build_latest_root="${RESULTS_ROOT}"
  fi

  case "${EXPERIMENT_KIND}" in
    controlled)
      if [[ -z "${CONTROLLED_RUN_ROOT}" || ! -d "${CONTROLLED_RUN_ROOT}" ]]; then
        echo "ERROR: controlled root does not exist: ${CONTROLLED_RUN_ROOT}" >&2
        exit 1
      fi
      "${PYTHON_BIN}" scripts/build_milestone27_controlled_replay_report.py \
        --root "${CONTROLLED_RUN_ROOT}" \
        --out-dir "${REPORT_DIR}/report" \
        --latest-root "${build_latest_root}" \
        --max-timeline-gaps "${MAX_TIMELINE_GAPS}"
      ;;
    live)
      CONTROLLED_RUN_ROOT="$(discover_controlled_root)"
      if [[ -z "${CONTROLLED_RUN_ROOT}" || ! -d "${CONTROLLED_RUN_ROOT}" ]]; then
        echo "ERROR: live-only master report needs an existing controlled root for the shared master layout." >&2
        echo "Set CONTROLLED_ROOT=... or run EXPERIMENT_KIND=both first." >&2
        exit 1
      fi
      if [[ -z "${LIVE_DIRECT_RUN_ROOT}" || ! -d "${LIVE_DIRECT_RUN_ROOT}" ]]; then
        echo "ERROR: live root does not exist: ${LIVE_DIRECT_RUN_ROOT}" >&2
        exit 1
      fi
      CONTROLLED_ROOT="${CONTROLLED_RUN_ROOT}" \
      LIVE_DIRECT_ROOT="${LIVE_DIRECT_RUN_ROOT}" \
      LATEST_REPORT_ROOT="${build_latest_root}" \
      MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS}" \
      bash scripts/build_latest_master_with_live_direct.sh
      mkdir -p "${REPORT_DIR}/report"
      cp -f "${build_latest_root}/latest_master_report.html" "${REPORT_DIR}/report/controlled_replay_report.html"
      ;;
    both)
      if [[ -z "${CONTROLLED_RUN_ROOT}" || ! -d "${CONTROLLED_RUN_ROOT}" ]]; then
        echo "ERROR: controlled root does not exist: ${CONTROLLED_RUN_ROOT}" >&2
        exit 1
      fi
      if [[ -z "${LIVE_DIRECT_RUN_ROOT}" || ! -d "${LIVE_DIRECT_RUN_ROOT}" ]]; then
        echo "ERROR: live root does not exist: ${LIVE_DIRECT_RUN_ROOT}" >&2
        exit 1
      fi
      CONTROLLED_ROOT="${CONTROLLED_RUN_ROOT}" \
      LIVE_DIRECT_ROOT="${LIVE_DIRECT_RUN_ROOT}" \
      LATEST_REPORT_ROOT="${build_latest_root}" \
      MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS}" \
      bash scripts/build_latest_master_with_live_direct.sh
      mkdir -p "${REPORT_DIR}/report"
      cp -f "${build_latest_root}/latest_master_report.html" "${REPORT_DIR}/report/controlled_replay_report.html"
      ;;
  esac

  local built_report="${REPORT_DIR}/report/controlled_replay_report.html"
  if [[ ! -f "${built_report}" && -f "${build_latest_root}/latest_master_report.html" ]]; then
    built_report="${build_latest_root}/latest_master_report.html"
  fi
  if [[ ! -f "${built_report}" ]]; then
    echo "ERROR: expected report was not generated." >&2
    exit 1
  fi

  cp -f "${built_report}" "${REPORT_DIR}/master_report.html"
  for artifact in controlled_replay_report.json controlled_replay_gaps.csv; do
    if [[ -f "${REPORT_DIR}/report/${artifact}" ]]; then
      cp -f "${REPORT_DIR}/report/${artifact}" "${REPORT_DIR}/${artifact}"
    fi
  done

  {
    echo "REPORT_LABEL=${REPORT_LABEL}"
    echo "MODEL=${MODEL}"
    echo "EXPERIMENT_KIND=${EXPERIMENT_KIND}"
    echo "RESULTS_ROOT=${RESULTS_ROOT}"
    echo "REPORT_DIR=${REPORT_DIR}"
    echo "CONTROLLED_ROOT=${CONTROLLED_RUN_ROOT}"
    echo "LIVE_DIRECT_ROOT=${LIVE_DIRECT_RUN_ROOT}"
    echo "UPDATE_LATEST=${UPDATE_LATEST}"
    echo "BUILD_ONLY=${BUILD_ONLY}"
    echo "MAX_TIMELINE_GAPS=${MAX_TIMELINE_GAPS}"
  } > "${REPORT_DIR}/run_config.env"

  write_manifest "${RESULTS_ROOT}/latest_master_report.html" "${REPORT_DIR}/master_report.html" "${CONTROLLED_RUN_ROOT}" "${LIVE_DIRECT_RUN_ROOT}"
}

print_config

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN=1: no experiments or report builds were run."
  exit 0
fi

mkdir -p "${REPORT_DIR}" "${RUNS_ROOT}/controlled" "${RUNS_ROOT}/live" "${SCRATCH_LATEST_ROOT}"

case "${EXPERIMENT_KIND}" in
  controlled)
    run_controlled
    ;;
  live)
    run_live
    ;;
  both)
    run_controlled
    run_live
    ;;
esac

build_report
clean_toplevel

echo
echo "Done."
if [[ "${UPDATE_LATEST}" == "1" ]]; then
  echo "Latest report:"
  echo "  ${RESULTS_ROOT}/latest_master_report.html"
  echo "Latest manifest:"
  echo "  ${RESULTS_ROOT}/latest_manifest.json"
else
  echo "Latest report was not overwritten. Set UPDATE_LATEST=1 to refresh it."
fi
echo "Archived labeled report:"
echo "  ${REPORT_DIR}/master_report.html"
echo "Manifest:"
echo "  ${REPORT_DIR}/manifest.json"
