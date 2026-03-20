#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pod_helpers.sh"

POD_CONFIG_PATH="${1:-${SCRIPT_DIR}/pod_experiment_2x5090.env}"
TRAIN_CONFIG_PATH="${2:-${SCRIPT_DIR}/train_experiment_2x5090.env}"
BRANCH_NAME="${3:-$(git -C "$(cd "${SCRIPT_DIR}/../../.." && pwd)" branch --show-current)}"
REMOTE_URL="${REMOTE_URL:-$(git -C "$(cd "${SCRIPT_DIR}/../../.." && pwd)" remote get-url origin)}"

"${SCRIPT_DIR}/pod_create.sh" "${POD_CONFIG_PATH}" >/dev/null
POD_ID="$(resolve_pod_id "")"
eval "$(pod_ssh_exports "${POD_ID}")"

echo "==> Ensuring repo exists on pod ${POD_ID}"
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "root@${SSH_HOST}" -p "${SSH_PORT}" \
  "if [[ ! -d ${REPO_ROOT_REMOTE}/.git ]]; then git clone --branch ${BRANCH_NAME} ${REMOTE_URL} ${REPO_ROOT_REMOTE}; else cd ${REPO_ROOT_REMOTE} && git fetch origin && git checkout ${BRANCH_NAME} && git pull --ff-only; fi"

echo "==> Syncing working record"
"${SCRIPT_DIR}/pod_sync.sh" "${POD_ID}"

echo "==> Running remote bootstrap"
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "root@${SSH_HOST}" -p "${SSH_PORT}" \
  "cd ${REPO_ROOT_REMOTE} && bash ${RECORD_ROOT_REMOTE}/runpod/remote_bootstrap.sh ${RECORD_ROOT_REMOTE}/runpod/$(basename "${TRAIN_CONFIG_PATH}")"

echo "==> Pod ${POD_ID} is prepared"
