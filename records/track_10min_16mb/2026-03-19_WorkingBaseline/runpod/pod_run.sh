#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pod_helpers.sh"

CONFIG_PATH="${1:-${SCRIPT_DIR}/train_experiment_2x5090.env}"
POD_CONFIG_PATH="${2:-${SCRIPT_DIR}/pod_experiment_2x5090.env}"
RUN_ID_OVERRIDE="${3:-}"
EXTRA_ENV_RAW="${4:-}"
SECRETS_PATH="${SCRIPT_DIR}/secrets.env"
: "${AUTO_STOP_POD:=1}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${POD_CONFIG_PATH}" ]]; then
  echo "Pod config file not found: ${POD_CONFIG_PATH}" >&2
  exit 1
fi
if [[ -f "${SECRETS_PATH}" ]]; then
  set -a
  source "${SECRETS_PATH}"
  set +a
fi

POD_ID="$(ensure_pod_ready "${POD_CONFIG_PATH}")"
eval "$(pod_ssh_exports "${POD_ID}")"

if ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "root@${SSH_HOST}" -p "${SSH_PORT}" \
  "pgrep -f 'train_gpt.py' >/dev/null"; then
  echo "A training process is already running on pod ${POD_ID}. Use 'just status' to inspect it." >&2
  exit 1
fi

echo "==> Syncing local repo snapshot to pod ${POD_ID}"
"${SCRIPT_DIR}/pod_sync.sh" "${POD_ID}"

echo "==> Running remote bootstrap"
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "root@${SSH_HOST}" -p "${SSH_PORT}" \
  "cd ${REPO_ROOT_REMOTE} && bash ${RECORD_ROOT_REMOTE}/runpod/remote_bootstrap.sh ${RECORD_ROOT_REMOTE}/runpod/$(basename "${CONFIG_PATH}")"

cleanup() {
  local exit_code=$?
  if [[ "${AUTO_STOP_POD}" == "1" ]]; then
    echo "==> Stopping pod ${POD_ID}"
    if ! runpodctl pod stop "${POD_ID}" >/dev/null; then
      echo "Warning: failed to stop pod ${POD_ID}" >&2
    fi
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

remote_exports=()
append_export() {
  local key="$1"
  local value="$2"
  local quoted
  printf -v quoted '%q' "${value}"
  remote_exports+=("export ${key}=${quoted};")
}
if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
  append_export "RUN_ID" "${RUN_ID_OVERRIDE}"
  append_export "WANDB_RUN_NAME" "${RUN_ID_OVERRIDE}"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  append_export "WANDB_API_KEY" "${WANDB_API_KEY}"
fi
if [[ -n "${EXTRA_ENV_RAW}" ]]; then
  remote_exports+=("export ${EXTRA_ENV_RAW};")
fi

remote_cmd="cd ${REPO_ROOT_REMOTE} && "
if ((${#remote_exports[@]} > 0)); then
  remote_cmd+="${remote_exports[*]} "
fi
remote_cmd+="bash ${RECORD_ROOT_REMOTE}/runpod/remote_train.sh ${RECORD_ROOT_REMOTE}/runpod/$(basename "${CONFIG_PATH}")"

ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "root@${SSH_HOST}" -p "${SSH_PORT}" \
  "${remote_cmd}"
