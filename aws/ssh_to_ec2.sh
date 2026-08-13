#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.sh
source "${SCRIPT_DIR}/config.sh"

usage() {
  cat <<'EOF'
Usage:
  ./aws/ssh_to_ec2.sh <idx>
  ./aws/ssh_to_ec2.sh <idx> '<remote command>'

Configure servers by editing aws/config.sh or exporting:
  AGENTIC_HW_SERVERS="1.2.3.4"
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

INDEX="$1"
validate_server_index "${INDEX}"
chmod 400 "${PEM}"

ip="${SERVERS[$INDEX]}"
remote_host="${EC2_USER}@${ip}"

echo "Connecting to ${remote_host} ..."
if [[ $# -gt 1 ]]; then
  shift
  ssh $(ssh_opts shell) "${remote_host}" "$@"
else
  ssh $(ssh_opts shell) "${remote_host}"
fi
