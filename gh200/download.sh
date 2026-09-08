#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.sh
source "${SCRIPT_DIR}/config.sh"

DOWNLOAD_ALL=0
REPORT_LABEL="${REPORT_LABEL:-}"

usage() {
  cat <<'EOF'
Usage:
  ./gh200/download.sh
  ./gh200/download.sh --label <REPORT_LABEL>
  ./gh200/download.sh --all

Default downloads only the latest compact report files. --label also downloads
one archived report directory. --all downloads the full remote artifacts tree
and may be very large.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      DOWNLOAD_ALL=1
      shift
      ;;
    --label)
      REPORT_LABEL="${2:-}"
      if [[ -z "${REPORT_LABEL}" ]]; then
        echo "--label requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

remote_host="$(gh200_remote)"
ssh_cmd="ssh $(gh200_ssh_args download)"
remote_results="${GH200_REMOTE_DIR}/sglang_direct_kv/artifacts/results"
local_results="${REPO_ROOT}/sglang_direct_kv/artifacts/results"
mkdir -p "${local_results}"

rsync_common_opts=(
  -az
  --human-readable
  --itemize-changes
  --stats
  --omit-dir-times
  --no-perms
  --no-owner
  --no-group
)

echo "==== Downloading GH200 results ===="
echo "Remote source: ${remote_results}/"
echo "Local dest:    ${local_results}/"

if [[ "${DOWNLOAD_ALL}" == "1" ]]; then
  rsync "${rsync_common_opts[@]}" -e "${ssh_cmd}" \
    "${remote_host}:${remote_results}/" \
    "${local_results}/"
else
  for file in latest_master_report.html latest_evidence_tables.html evidence_tables.html latest_manifest.json; do
    if ssh $(gh200_ssh_args download) "${remote_host}" "test -f '${remote_results}/${file}'"; then
      rsync "${rsync_common_opts[@]}" -e "${ssh_cmd}" \
        "${remote_host}:${remote_results}/${file}" \
        "${local_results}/${file}"
    fi
  done

  if [[ -n "${REPORT_LABEL}" ]]; then
    mkdir -p "${local_results}/reports"
    rsync "${rsync_common_opts[@]}" -e "${ssh_cmd}" \
      "${remote_host}:${remote_results}/reports/${REPORT_LABEL}/" \
      "${local_results}/reports/${REPORT_LABEL}/"
  fi
fi

if [[ -f "${local_results}/latest_master_report.html" ]]; then
  cp "${local_results}/latest_master_report.html" "${REPO_ROOT}/latest_master_report.html"
fi

echo "GH200 download complete."
