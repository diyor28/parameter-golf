#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/pod_experiment_1x5090.env}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

set -a
source "${CONFIG_PATH}"
set +a

: "${POD_NAME:=pgolf-working}"
: "${RUNPOD_TEMPLATE_ID:=runpod-torch-v280}"
: "${RUNPOD_GPU_ID:=NVIDIA GeForce RTX 5090}"
: "${RUNPOD_GPU_COUNT:=1}"
: "${RUNPOD_CLOUD_TYPE:=COMMUNITY}"
: "${RUNPOD_CONTAINER_DISK_GB:=30}"
: "${RUNPOD_VOLUME_GB:=40}"
: "${RUNPOD_VOLUME_MOUNT_PATH:=/workspace}"
: "${RUNPOD_ENABLE_PUBLIC_IP:=0}"
: "${WAIT_FOR_SSH:=1}"
: "${RUNPOD_SSH:=1}"

STATE_DIR="${SCRIPT_DIR}/state/${POD_NAME}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${STATE_DIR}"

cmd=(
  runpodctl pod create
  --name "${POD_NAME}"
  --template-id "${RUNPOD_TEMPLATE_ID}"
  --gpu-id "${RUNPOD_GPU_ID}"
  --gpu-count "${RUNPOD_GPU_COUNT}"
  --cloud-type "${RUNPOD_CLOUD_TYPE}"
  --container-disk-in-gb "${RUNPOD_CONTAINER_DISK_GB}"
  --volume-in-gb "${RUNPOD_VOLUME_GB}"
  --volume-mount-path "${RUNPOD_VOLUME_MOUNT_PATH}"
  -o json
)

if [[ "${RUNPOD_SSH}" == "1" ]]; then
  cmd+=(--ssh)
fi
if [[ "${RUNPOD_ENABLE_PUBLIC_IP}" == "1" ]]; then
  cmd+=(--public-ip)
fi
if [[ -n "${RUNPOD_PORTS:-}" ]]; then
  cmd+=(--ports "${RUNPOD_PORTS}")
fi
if [[ -n "${RUNPOD_DATA_CENTER_IDS:-}" ]]; then
  cmd+=(--data-center-ids "${RUNPOD_DATA_CENTER_IDS}")
fi
if [[ -n "${RUNPOD_ENV_JSON:-}" ]]; then
  cmd+=(--env "${RUNPOD_ENV_JSON}")
fi

echo "==> Creating pod ${POD_NAME}"
response="$("${cmd[@]}")"
printf '%s\n' "${response}" > "${STATE_DIR}/create_response.json"

pod_id="$(
  RESPONSE_JSON="${response}" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["RESPONSE_JSON"])
if isinstance(data, list):
    data = data[0] if data else {}
if not isinstance(data, dict):
    raise SystemExit("Unexpected runpodctl create response")
for key in ("id", "podId"):
    value = data.get(key)
    if value:
        print(value)
        raise SystemExit(0)
raise SystemExit("Could not find pod id in create response")
PY
)"

printf '%s\n' "${pod_id}" > "${STATE_DIR}/pod_id.txt"
printf '%s\n' "${pod_id}" > "${SCRIPT_DIR}/state/current_pod_id"
echo "==> Pod created: ${pod_id}"
echo "==> State saved in ${STATE_DIR}"

if [[ "${WAIT_FOR_SSH}" == "1" ]]; then
  echo "==> Waiting for SSH info to become available"
  for _ in $(seq 1 60); do
    ssh_info_json="$(runpodctl ssh info "${pod_id}" -o json 2>/dev/null || true)"
    if [[ -n "${ssh_info_json}" ]] && SSH_INFO_JSON="${ssh_info_json}" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["SSH_INFO_JSON"])
raise SystemExit(0 if data.get("ssh_command") else 1)
PY
    then
      printf '%s\n' "${ssh_info_json}" > "${STATE_DIR}/ssh_info.txt"
      echo "==> Pod is ready for SSH"
      cat "${STATE_DIR}/ssh_info.txt"
      exit 0
    fi
    sleep 10
  done
  echo "Timed out waiting for SSH readiness. Try: runpodctl ssh info ${pod_id}" >&2
  exit 1
fi

echo "Next step: runpodctl ssh info ${pod_id}"
