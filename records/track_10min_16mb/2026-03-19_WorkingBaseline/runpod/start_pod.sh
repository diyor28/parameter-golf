#!/usr/bin/env bash
set -euo pipefail

POD_ID="${1:-}"
if [[ -z "${POD_ID}" ]]; then
  echo "Usage: $0 <pod-id>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\n' "${POD_ID}" > "${SCRIPT_DIR}/state/current_pod_id"
runpodctl pod start "${POD_ID}"
