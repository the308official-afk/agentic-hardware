#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.sh
source "${SCRIPT_DIR}/config.sh"

usage() {
  cat <<'EOF'
Usage:
  ./gh200/ssh_to_gh200.sh
  ./gh200/ssh_to_gh200.sh '<remote command>'

Environment overrides:
  AGENTIC_GH200_USER, AGENTIC_GH200_HOST, AGENTIC_GH200_JUMP_USER,
  AGENTIC_GH200_JUMP_HOST, AGENTIC_GH200_JUMP_PORT, AGENTIC_GH200_REMOTE_DIR
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

remote_host="$(gh200_remote)"
echo "Connecting to ${remote_host} via ${GH200_JUMP_USER}@${GH200_JUMP_HOST}:${GH200_JUMP_PORT} ..."

if [[ $# -gt 0 ]]; then
  ssh $(gh200_ssh_args shell) "${remote_host}" "$@"
else
  ssh $(gh200_ssh_args shell) "${remote_host}"
fi
