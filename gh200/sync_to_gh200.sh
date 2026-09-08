#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.sh
source "${SCRIPT_DIR}/config.sh"

DRY_RUN=0
usage() {
  cat <<'EOF'
Usage:
  ./gh200/sync_to_gh200.sh
  ./gh200/sync_to_gh200.sh --dry

Syncs source code to GH200 while excluding local environments, git metadata,
node_modules, caches, and artifacts. Remote artifacts are protected.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry|--dry-run)
      DRY_RUN=1
      shift
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
ssh_cmd="ssh $(gh200_ssh_args sync)"

rsync_common_opts=(
  -az
  --human-readable
  --itemize-changes
  --stats
  --omit-dir-times
  --no-perms
  --no-owner
  --no-group
  --delete
  --filter='protect sglang_direct_kv/artifacts/'
)

if [[ "${DRY_RUN}" == "1" ]]; then
  rsync_common_opts+=(--dry-run)
fi

rsync_excludes=(
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
  --exclude '.venvs/'
  --exclude 'node_modules/'
  --exclude 'artifacts/'
  --exclude 'sglang_direct_kv/artifacts/'
  --exclude '*.log'
  --exclude 'tmp/'
)

echo "==== Syncing ${REPO_NAME} to GH200 ===="
echo "Local source: ${REPO_ROOT}/"
echo "Remote dest:  ${GH200_REMOTE_DIR}/"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Mode: dry run; no files will be transferred."
fi

ssh $(gh200_ssh_args sync) "${remote_host}" "mkdir -p '${GH200_REMOTE_DIR}'"

rsync \
  "${rsync_common_opts[@]}" \
  "${rsync_excludes[@]}" \
  -e "${ssh_cmd}" \
  "${REPO_ROOT}/" \
  "${remote_host}:${GH200_REMOTE_DIR}/"

if [[ "${DRY_RUN}" != "1" ]]; then
  ssh $(gh200_ssh_args sync) "${remote_host}" \
    "cd '${GH200_REMOTE_DIR}' && chmod -R ugo=rwX . && find . -name '*.pyc' -delete"
fi

echo "GH200 sync complete."
