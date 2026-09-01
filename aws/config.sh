#!/usr/bin/env bash

# Shared EC2 sync configuration for agentic_hardware.
#
# Edit AGENTIC_HW_SERVERS after launching EC2, or export it before running:
#   export AGENTIC_HW_SERVERS="1.2.3.4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_NAME="$(basename "${REPO_ROOT}")"

PEM="${AGENTIC_HW_PEM:-/Users/oluwolejaiyeoba/Documents/GitHub/secrets/projectonekeypair.pem}"
EC2_USER="${AGENTIC_HW_EC2_USER:-ec2-user}"
REMOTE_PROJECT_DIR="${AGENTIC_HW_REMOTE_DIR:-/home/${EC2_USER}/${REPO_NAME}}"

if [[ -n "${AGENTIC_HW_SERVERS:-}" ]]; then
  read -r -a SERVERS <<< "${AGENTIC_HW_SERVERS}"
else
  SERVERS=(
    "54.86.49.107"
  )
fi

LABELS=()
for i in "${!SERVERS[@]}"; do
  LABELS+=("S${i}")
done

ssh_opts() {
  local control_name="$1"
  printf '%q ' \
    -i "${PEM}" \
    -o ControlMaster=auto \
    -o ControlPersist=10m \
    -o "ControlPath=/tmp/agentic-hardware-${control_name}-%r@%h:%p" \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3
}

validate_server_index() {
  local index="$1"
  if ! [[ "${index}" =~ ^[0-9]+$ ]]; then
    echo "Server index must be numeric: ${index}" >&2
    return 1
  fi
  if (( index < 0 || index >= ${#SERVERS[@]} )); then
    echo "Server index out of range: ${index}" >&2
    echo "Valid indices: 0..$(( ${#SERVERS[@]} - 1 ))" >&2
    return 1
  fi
  if [[ -z "${SERVERS[$index]}" ]]; then
    echo "Server index ${index} has no IP configured." >&2
    echo "Edit aws/config.sh or export AGENTIC_HW_SERVERS=\"<ip>\"." >&2
    return 1
  fi
}
