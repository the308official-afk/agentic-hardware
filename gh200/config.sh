#!/usr/bin/env bash

# Shared GH200 connection configuration.
#
# Override these before running the helper scripts if your login details differ:
#   export AGENTIC_GH200_USER="ojaiyeob"
#   export AGENTIC_GH200_HOST="gracehopper"
#   export AGENTIC_GH200_JUMP_HOST="falcon.7elements.com"
#   export AGENTIC_GH200_JUMP_PORT="1337"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_NAME="$(basename "${REPO_ROOT}")"

GH200_USER="${AGENTIC_GH200_USER:-ojaiyeob}"
GH200_HOST="${AGENTIC_GH200_HOST:-gracehopper}"
GH200_JUMP_USER="${AGENTIC_GH200_JUMP_USER:-${GH200_USER}}"
GH200_JUMP_HOST="${AGENTIC_GH200_JUMP_HOST:-falcon.7elements.com}"
GH200_JUMP_PORT="${AGENTIC_GH200_JUMP_PORT:-1337}"
GH200_REMOTE_DIR="${AGENTIC_GH200_REMOTE_DIR:-/home/central/${GH200_USER}/${REPO_NAME}}"

gh200_remote() {
  printf '%s@%s' "${GH200_USER}" "${GH200_HOST}"
}

gh200_ssh_args() {
  local control_name="$1"
  printf '%q ' \
    -J "${GH200_JUMP_USER}@${GH200_JUMP_HOST}:${GH200_JUMP_PORT}" \
    -o ControlMaster=auto \
    -o ControlPersist=10m \
    -o "ControlPath=/tmp/agentic-gh200-${control_name}-%r@%h:%p" \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=20 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3
}
