#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pod_helpers.sh"

REPO_ROOT_LOCAL="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
POD_ID="$(resolve_pod_id "${1:-}")"
eval "$(pod_ssh_exports "${POD_ID}")"
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "root@${SSH_HOST}" -p "${SSH_PORT}" "mkdir -p '${REPO_ROOT_REMOTE}'"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for this workflow" >&2
  exit 1
fi

rsync -az --delete --no-owner --no-group \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude '.venv-runpod' \
  --exclude 'wandb' \
  --exclude 'logs' \
  --exclude 'data/datasets' \
  --exclude 'records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/state' \
  --exclude 'records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/secrets.env' \
  -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY} -p ${SSH_PORT}" \
  "${REPO_ROOT_LOCAL}/" "root@${SSH_HOST}:${REPO_ROOT_REMOTE}/"
