#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.sh
source "${SCRIPT_DIR}/config.sh"

usage() {
  cat <<'EOF'
Usage:
  ./aws/upload.sh        Upload to all configured servers
  ./aws/upload.sh <idx>  Upload only to server index <idx>

Configure servers by editing aws/config.sh or exporting:
  AGENTIC_HW_SERVERS="1.2.3.4"
EOF
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 1
fi

chmod 400 "${PEM}"

SSH_CMD="ssh $(ssh_opts upload)"

RSYNC_COMMON_OPTS=(
  -az
  --human-readable
  --itemize-changes
  --stats
  --omit-dir-times
  --no-perms
  --no-owner
  --no-group
)

RSYNC_PROGRESS_OPTS=()
if rsync --version 2>/dev/null | head -1 | grep -q 'version 3'; then
  RSYNC_PROGRESS_OPTS=(--info=progress2)
else
  RSYNC_PROGRESS_OPTS=(--progress)
fi

RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '.DS_Store'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
  --exclude '.coverage'
  --exclude 'htmlcov/'
  --exclude '.venv/'
  --exclude 'venv/'
  --exclude 'artifacts/'
  --exclude 'sglang_direct_kv/artifacts/'
)

TARGET_INDICES=()
if [[ $# -eq 1 ]]; then
  validate_server_index "$1"
  TARGET_INDICES=("$1")
else
  TARGET_INDICES=("${!SERVERS[@]}")
fi

for i in "${TARGET_INDICES[@]}"; do
  validate_server_index "${i}"
  ip="${SERVERS[$i]}"
  label="${LABELS[$i]}"
  remote_host="${EC2_USER}@${ip}"
  remote_base="${remote_host}:${REMOTE_PROJECT_DIR}/"

  echo "==== Uploading ${REPO_NAME} to ${label} (${ip}) ===="
  echo "Local source: ${REPO_ROOT}/"
  echo "Remote dest:  ${REMOTE_PROJECT_DIR}/"

  ssh $(ssh_opts upload) "${remote_host}" "mkdir -p '${REMOTE_PROJECT_DIR}'"

  rsync \
    "${RSYNC_COMMON_OPTS[@]}" \
    "${RSYNC_PROGRESS_OPTS[@]}" \
    "${RSYNC_EXCLUDES[@]}" \
    -e "${SSH_CMD}" \
    "${REPO_ROOT}/" \
    "${remote_base}"

  echo "Upload to ${label} complete."
done
