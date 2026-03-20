#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT="${SCRIPT_DIR}/state"
CURRENT_POD_ID_PATH="${STATE_ROOT}/current_pod_id"
CURRENT_POD_JSON_PATH="${STATE_ROOT}/current_pod.json"
REPO_ROOT_REMOTE="/workspace/parameter-golf"
RECORD_ROOT_REMOTE="records/track_10min_16mb/2026-03-19_WorkingBaseline"

resolve_pod_id() {
  local requested_pod_id="${1:-}"
  if [[ -n "${requested_pod_id}" ]]; then
    printf '%s\n' "${requested_pod_id}"
    return 0
  fi
  if [[ -f "${CURRENT_POD_ID_PATH}" ]]; then
    cat "${CURRENT_POD_ID_PATH}"
    return 0
  fi
  echo "No pod id provided and no current pod recorded in ${CURRENT_POD_ID_PATH}" >&2
  return 1
}

save_current_pod_state() {
  local pod_id="$1"
  local ssh_info_json="${2:-}"
  mkdir -p "${STATE_ROOT}"
  printf '%s\n' "${pod_id}" > "${CURRENT_POD_ID_PATH}"
  if [[ -n "${ssh_info_json}" ]]; then
    SSH_INFO_JSON="${ssh_info_json}" POD_ID="${pod_id}" python3 - <<'PY' > "${CURRENT_POD_JSON_PATH}"
import json
import os

payload = {
    "pod_id": os.environ["POD_ID"],
    "ssh_info": json.loads(os.environ["SSH_INFO_JSON"]),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  fi
}

pod_ssh_info_json() {
  local pod_id="$1"
  runpodctl ssh info "${pod_id}" -o json
}

pod_get_json() {
  local pod_id="$1"
  runpodctl pod get "${pod_id}" -o json
}

find_reusable_pod_id() {
  local desired_name="$1"
  local pod_list_json matched_running
  pod_list_json="$(runpodctl pod list -o json 2>/dev/null || printf '[]')"
  matched_running="$(
    POD_LIST_JSON="${pod_list_json}" DESIRED_NAME="${desired_name}" python3 - <<'PY'
import json
import os

pods = json.loads(os.environ["POD_LIST_JSON"])
desired_name = os.environ["DESIRED_NAME"]
for pod in pods:
    if pod.get("name") == desired_name and pod.get("desiredStatus") == "RUNNING":
        print(pod["id"])
        break
PY
  )"
  if [[ -n "${matched_running}" ]]; then
    printf '%s\n' "${matched_running}"
    return 0
  fi
}

ensure_pod_ready() {
  local pod_config_path="$1"
  if [[ ! -f "${pod_config_path}" ]]; then
    echo "Pod config file not found: ${pod_config_path}" >&2
    return 1
  fi

  local requested_name requested_wait requested_attempts requested_interval
  requested_name="$(
    POD_CONFIG_PATH="${pod_config_path}" python3 - <<'PY'
import os
from pathlib import Path

values = {}
for raw_line in Path(os.environ["POD_CONFIG_PATH"]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip()] = value.strip()

def parse_default(raw: str) -> str:
    if ":-" in raw and raw.startswith("${") and raw.endswith("}"):
        raw = raw[2:-1].split(":-", 1)[1]
    return raw.strip().strip("'").strip('"')

for key in ("POD_NAME", "WAIT_FOR_SSH", "WAIT_FOR_SSH_ATTEMPTS", "WAIT_FOR_SSH_INTERVAL_SECONDS"):
    print(parse_default(values.get(key, "")))
PY
  )"
  requested_name="$(printf '%s\n' "${requested_name}" | sed -n '1p')"
  requested_wait="$(POD_CONFIG_PATH="${pod_config_path}" python3 - <<'PY'
import os
from pathlib import Path
values={}
for raw_line in Path(os.environ["POD_CONFIG_PATH"]).read_text(encoding="utf-8").splitlines():
    line=raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key,value=line.split("=",1)
    values[key.strip()]=value.strip()
def parse(key, default):
    raw=values.get(key, default)
    if ":-" in raw and raw.startswith("${") and raw.endswith("}"):
        raw=raw[2:-1].split(":-",1)[1]
    print(raw.strip().strip("'").strip('"'))
parse("WAIT_FOR_SSH","1")
parse("WAIT_FOR_SSH_ATTEMPTS","60")
parse("WAIT_FOR_SSH_INTERVAL_SECONDS","10")
PY
)"
  local wait_for_ssh wait_attempts wait_interval
  wait_for_ssh="$(printf '%s\n' "${requested_wait}" | sed -n '1p')"
  wait_attempts="$(printf '%s\n' "${requested_wait}" | sed -n '2p')"
  wait_interval="$(printf '%s\n' "${requested_wait}" | sed -n '3p')"

  local reusable_pod_id
  reusable_pod_id="$(find_reusable_pod_id "${requested_name}" || true)"
  if [[ -n "${reusable_pod_id}" ]]; then
    save_current_pod_state "${reusable_pod_id}"
    if [[ "${wait_for_ssh}" == "1" ]]; then
      wait_for_ssh_ready "${reusable_pod_id}" "${wait_attempts}" "${wait_interval}" >/dev/null
    fi
    printf '%s\n' "${reusable_pod_id}"
    return 0
  fi

  "${SCRIPT_DIR}/pod_create.sh" "${pod_config_path}" >/dev/null
  resolve_pod_id ""
}

pod_ssh_exports() {
  local pod_id="$1"
  local ssh_json
  ssh_json="$(pod_ssh_info_json "${pod_id}")"
  SSH_INFO_JSON="${ssh_json}" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["SSH_INFO_JSON"])
host = data.get("ip")
port = data.get("port")
key_path = (data.get("ssh_key") or {}).get("path")
ssh_command = data.get("ssh_command")
if not (host and port and key_path and ssh_command):
    raise SystemExit("SSH info incomplete; pod may not be ready")
for key, value in (
    ("SSH_HOST", host),
    ("SSH_PORT", str(port)),
    ("SSH_KEY", key_path),
    ("SSH_COMMAND", ssh_command),
):
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    print(f'{key}="{escaped}"')
PY
}

wait_for_ssh_ready() {
  local pod_id="$1"
  local attempts="${2:-60}"
  local sleep_seconds="${3:-10}"
  local ssh_json=""
  local ssh_exports=""
  local SSH_HOST=""
  local SSH_PORT=""
  local SSH_KEY=""

  for _ in $(seq 1 "${attempts}"); do
    ssh_json="$(pod_ssh_info_json "${pod_id}" 2>/dev/null || true)"
    if [[ -z "${ssh_json}" ]]; then
      sleep "${sleep_seconds}"
      continue
    fi
    if ssh_exports="$(
      SSH_INFO_JSON="${ssh_json}" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["SSH_INFO_JSON"])
host = data.get("ip")
port = data.get("port")
key_path = (data.get("ssh_key") or {}).get("path")
ssh_command = data.get("ssh_command")
if not (host and port and key_path and ssh_command):
    raise SystemExit(1)
for key, value in (
    ("SSH_HOST", host),
    ("SSH_PORT", str(port)),
    ("SSH_KEY", key_path),
    ("SSH_COMMAND", ssh_command),
):
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    print(f'{key}="{escaped}"')
PY
    )"; then
      eval "${ssh_exports}"
      if ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "${SSH_KEY}" "root@${SSH_HOST}" -p "${SSH_PORT}" 'echo ready' >/dev/null 2>&1; then
        save_current_pod_state "${pod_id}" "${ssh_json}"
        printf '%s\n' "${ssh_json}"
        return 0
      fi
    fi
    sleep "${sleep_seconds}"
  done

  echo "Timed out waiting for SSH readiness on pod ${pod_id}" >&2
  return 1
}
