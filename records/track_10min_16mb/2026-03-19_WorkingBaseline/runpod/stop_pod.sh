#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pod_helpers.sh"

POD_ID="$(resolve_pod_id "${1:-}")"
save_current_pod_state "${POD_ID}"
runpodctl pod stop "${POD_ID}"
