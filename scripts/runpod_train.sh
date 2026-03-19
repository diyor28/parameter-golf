#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-}"

if [[ -n "${CONFIG_PATH}" ]]; then
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  set -a
  source "${CONFIG_PATH}"
  set +a
fi

: "${PYTHON_BIN:=python3}"
: "${TORCHRUN_BIN:=torchrun}"
: "${TRAIN_ENTRYPOINT:=train_gpt.py}"
: "${GPU_COUNT:=1}"
: "${RUN_ID:=runpod_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${LOG_ROOT:=logs/runpod}"
: "${DOWNLOAD_VARIANT:=sp1024}"
: "${DOWNLOAD_TRAIN_SHARDS:=1}"
: "${WITH_DOCS:=0}"
: "${SKIP_DATA_DOWNLOAD:=0}"
: "${DATA_PATH:=${ROOT_DIR}/data/datasets/fineweb10B_${DOWNLOAD_VARIANT}}"
: "${TOKENIZER_PATH:=${ROOT_DIR}/data/tokenizers/fineweb_1024_bpe.model}"

RUN_DIR="${ROOT_DIR}/${LOG_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

if [[ "${SKIP_DATA_DOWNLOAD}" != "1" ]]; then
  download_cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/data/cached_challenge_fineweb.py"
    --variant "${DOWNLOAD_VARIANT}"
    --train-shards "${DOWNLOAD_TRAIN_SHARDS}"
  )
  if [[ "${WITH_DOCS}" == "1" ]]; then
    download_cmd+=(--with-docs)
  fi
  echo "==> Downloading challenge data (${DOWNLOAD_VARIANT}, train_shards=${DOWNLOAD_TRAIN_SHARDS})"
  "${download_cmd[@]}"
fi

{
  echo "RUN_ID=${RUN_ID}"
  echo "GPU_COUNT=${GPU_COUNT}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
  echo "DOWNLOAD_VARIANT=${DOWNLOAD_VARIANT}"
  echo "DOWNLOAD_TRAIN_SHARDS=${DOWNLOAD_TRAIN_SHARDS}"
  echo "TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}"
  env | LC_ALL=C sort
} > "${RUN_DIR}/run.env"

git -C "${ROOT_DIR}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt"

TRAIN_LOG="${RUN_DIR}/train.log"
echo "==> Run directory: ${RUN_DIR}"
echo "==> Log file: ${TRAIN_LOG}"

cd "${ROOT_DIR}"

export RUN_ID
export DATA_PATH
export TOKENIZER_PATH

set -o pipefail
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${GPU_COUNT}" "${TRAIN_ENTRYPOINT}" 2>&1 | tee "${TRAIN_LOG}"
