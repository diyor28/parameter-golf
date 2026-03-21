#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RECORD_DIR}/../../.." && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/train_experiment_2x5090.env}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

set -a
source "${CONFIG_PATH}"
set +a

: "${PYTHON_BIN:=python3}"
: "${VENV_DIR:=${REPO_ROOT}/.venv-runpod}"
: "${EXTRA_PYTHON_PACKAGES:=numpy tqdm huggingface-hub kernels setuptools typing-extensions==4.15.0 datasets tiktoken sentencepiece}"
: "${DOWNLOAD_VARIANT:=sp1024}"
: "${DOWNLOAD_TRAIN_SHARDS:=1}"
: "${WITH_DOCS:=0}"
: "${SKIP_DATA_DOWNLOAD:=0}"
: "${SKIP_PIP_INSTALL:=0}"

BOOTSTRAP_STAMP="${VENV_DIR}/.bootstrap_${DOWNLOAD_VARIANT}_train${DOWNLOAD_TRAIN_SHARDS}_docs${WITH_DOCS}"
if [[ "${FORCE_BOOTSTRAP:-0}" != "1" && -f "${BOOTSTRAP_STAMP}" ]]; then
  echo "==> Bootstrap already complete (${BOOTSTRAP_STAMP})"
  exit 0
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "==> Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ "${SKIP_PIP_INSTALL}" != "1" ]]; then
  echo "==> Installing Python deps into ${VENV_DIR} (reusing system PyTorch)"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install ${EXTRA_PYTHON_PACKAGES} wandb
else
  echo "==> Skipping pip install (SKIP_PIP_INSTALL=1); assuming deps are baked into the image"
fi

if [[ "${SKIP_DATA_DOWNLOAD}" != "1" ]]; then
  download_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/data/cached_challenge_fineweb.py"
    --variant "${DOWNLOAD_VARIANT}"
    --train-shards "${DOWNLOAD_TRAIN_SHARDS}"
  )
  if [[ "${WITH_DOCS}" == "1" ]]; then
    download_cmd+=(--with-docs)
  fi
  echo "==> Downloading challenge data (${DOWNLOAD_VARIANT}, train_shards=${DOWNLOAD_TRAIN_SHARDS})"
  "${download_cmd[@]}"
fi

touch "${BOOTSTRAP_STAMP}"
echo "==> Bootstrap complete"
