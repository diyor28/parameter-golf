#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pod_helpers.sh"

POD_ID="$(resolve_pod_id "${1:-}")"
pod_json="$(pod_get_json "${POD_ID}")"
desired_status="$(
  POD_JSON="${pod_json}" python3 - <<'PY'
import json
import os
print(json.loads(os.environ["POD_JSON"]).get("desiredStatus", ""))
PY
)"

echo "==> Pod"
POD_JSON="${pod_json}" python3 - <<'PY'
import json
import os

pod = json.loads(os.environ["POD_JSON"])
for key in ("id", "name", "desiredStatus", "gpuCount", "costPerHr", "imageName"):
    value = pod.get(key)
    if value is not None:
        print(f"{key}: {value}")
ssh = pod.get("ssh") or {}
if ssh.get("ip") and ssh.get("port"):
    print(f"ssh: {ssh['ip']}:{ssh['port']}")
PY

if [[ "${desired_status}" != "RUNNING" ]]; then
  echo
  echo "==> SSH"
  echo "Pod is not running, so SSH is unavailable."
  exit 0
fi

echo
echo "==> SSH"
ssh_json="$(wait_for_ssh_ready "${POD_ID}" 6 5)"
printf '%s\n' "${ssh_json}"

eval "$(pod_ssh_exports "${POD_ID}")"

echo
echo "==> Remote"
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "root@${SSH_HOST}" -p "${SSH_PORT}" "
  set -euo pipefail
  cd ${REPO_ROOT_REMOTE}
  printf 'branch: '
  git branch --show-current 2>/dev/null || echo missing
  printf 'latest_run_dir: '
  ls -1dt logs/record_runs/* 2>/dev/null | head -n 1 || echo none
  printf 'training_processes:\n'
  ps -ef | grep -E 'train_gpt|torchrun|python -m torch.distributed.run' | grep -v grep || true
  latest_run=\"\$(ls -1dt logs/record_runs/* 2>/dev/null | head -n 1 || true)\"
  if [[ -n \"\${latest_run}\" ]]; then
    echo 'latest_log_tail:'
    tail -n 20 \"\${latest_run}/train.log\" || true
    latest_wandb_url=\"\$(grep -Eo 'https://wandb.ai/[^ ]+' \"\${latest_run}/console.log\" 2>/dev/null | tail -n 1 || true)\"
    if [[ -n \"\${latest_wandb_url}\" ]]; then
      printf 'wandb_url: %s\n' \"\${latest_wandb_url}\"
    fi
  fi
"
