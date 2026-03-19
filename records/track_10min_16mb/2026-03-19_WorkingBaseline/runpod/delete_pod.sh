#!/usr/bin/env bash
set -euo pipefail

POD_ID="${1:-}"
if [[ -z "${POD_ID}" ]]; then
  echo "Usage: $0 <pod-id>" >&2
  exit 1
fi

runpodctl pod delete "${POD_ID}"
