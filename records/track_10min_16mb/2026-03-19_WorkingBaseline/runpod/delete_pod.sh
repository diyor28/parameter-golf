#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pod_helpers.sh"

POD_ID="$(resolve_pod_id "${1:-}")"
runpodctl pod delete "${POD_ID}"
if [[ -f "${CURRENT_POD_ID_PATH}" ]] && [[ "$(<"${CURRENT_POD_ID_PATH}")" == "${POD_ID}" ]]; then
  rm -f "${CURRENT_POD_ID_PATH}" "${CURRENT_POD_JSON_PATH}"
fi
