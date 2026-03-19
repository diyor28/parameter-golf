#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
POD_ID="${1:-}"

if [[ -z "${POD_ID}" ]]; then
  if [[ -f "${SCRIPT_DIR}/state/current_pod_id" ]]; then
    POD_ID="$(<"${SCRIPT_DIR}/state/current_pod_id")"
  else
    echo "Usage: $0 <pod-id>" >&2
    echo "No current pod recorded in ${SCRIPT_DIR}/state/current_pod_id" >&2
    exit 1
  fi
fi

ssh_json="$(runpodctl ssh info "${POD_ID}" -o json)"
read -r ssh_host ssh_port ssh_key <<<"$(
  SSH_INFO_JSON="${ssh_json}" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["SSH_INFO_JSON"])
host = data.get("ip")
port = data.get("port")
key_path = (data.get("ssh_key") or {}).get("path")
if not (host and port and key_path):
    raise SystemExit("SSH info incomplete; pod may not be ready")
print(host, port, key_path)
PY
)"

remote_base="/workspace/parameter-golf/records/track_10min_16mb"
ssh -o StrictHostKeyChecking=no -i "${ssh_key}" "root@${ssh_host}" -p "${ssh_port}" "mkdir -p '${remote_base}'"
scp -o StrictHostKeyChecking=no -i "${ssh_key}" -P "${ssh_port}" -r "${RECORD_DIR}" "root@${ssh_host}:${remote_base}/"
