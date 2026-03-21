#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RECORD_DIR}/../../.." && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/train_experiment_2x5090.env}"
SECRETS_PATH="${SCRIPT_DIR}/secrets.env"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

set -a
source "${CONFIG_PATH}"
if [[ -f "${SECRETS_PATH}" ]]; then
  source "${SECRETS_PATH}"
fi
set +a
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

sanitize_name_part() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-'
}

build_default_run_name() {
  local purpose hardware variant arch batch timestamp
  purpose="$(sanitize_name_part "${RUN_PURPOSE:-experiment}")"
  hardware="$(sanitize_name_part "${RUN_HARDWARE:-${GPU_COUNT:-1}gpu}")"
  variant="$(sanitize_name_part "${DOWNLOAD_VARIANT:-sp1024}")"
  arch="l${NUM_LAYERS:-9}d${MODEL_DIM:-512}h${NUM_HEADS:-8}kv${NUM_KV_HEADS:-4}-relu2"
  batch="tbt${TRAIN_BATCH_TOKENS:-0}"
  timestamp="$(date -u +%m%dT%H%M%SZ)"
  printf '%s_%s_%s_%s_%s_%s\n' "${purpose}" "${hardware}" "${variant}" "${arch}" "${batch}" "${timestamp}"
}

: "${PYTHON_BIN:=python3}"
: "${TORCHRUN_BIN:=torchrun}"
: "${GPU_COUNT:=1}"
: "${DOWNLOAD_VARIANT:=sp1024}"
: "${DOWNLOAD_TRAIN_SHARDS:=1}"
: "${WITH_DOCS:=0}"
: "${WANDB_ENABLE:=0}"
: "${VENV_DIR:=${REPO_ROOT}/.venv-runpod}"
: "${EXTRA_PYTHON_PACKAGES:=numpy tqdm huggingface-hub kernels setuptools typing-extensions==4.15.0 datasets tiktoken sentencepiece}"
: "${DATA_PATH:=${REPO_ROOT}/data/datasets/fineweb10B_${DOWNLOAD_VARIANT}}"
: "${TOKENIZER_PATH:=${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="$(build_default_run_name)"
fi
if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
  WANDB_RUN_NAME="${RUN_ID}"
fi

if [[ -z "${INSTALL_REQUIREMENTS+x}" ]]; then
  INSTALL_REQUIREMENTS=1
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    INSTALL_REQUIREMENTS=0
  fi
fi

if [[ -z "${SKIP_DATA_DOWNLOAD+x}" ]]; then
  SKIP_DATA_DOWNLOAD=0
  if [[ -f "${TOKENIZER_PATH}" ]] && compgen -G "${DATA_PATH}/fineweb_train_*.bin" >/dev/null; then
    SKIP_DATA_DOWNLOAD=1
  fi
fi

RUN_DIR="${REPO_ROOT}/logs/record_runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
RUNTIME_DIR="/workspace/pgolf-runtime"
mkdir -p "${RUNTIME_DIR}"
TRAIN_PID_FILE="${RUNTIME_DIR}/train.pid"

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "==> Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  fi
  PYTHON_BIN="${VENV_DIR}/bin/python"
  TORCHRUN_BIN="${VENV_DIR}/bin/torchrun"
  echo "==> Installing Python deps into ${VENV_DIR} (reusing system PyTorch)"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install ${EXTRA_PYTHON_PACKAGES}
else
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
  fi
fi

if [[ -x "${VENV_DIR}/bin/torchrun" ]]; then
  TORCHRUN_BIN="${VENV_DIR}/bin/torchrun"
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

if [[ "${WANDB_ENABLE}" == "1" ]]; then
  echo "==> Ensuring wandb is installed"
  if ! "${PYTHON_BIN}" -c 'import wandb' >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip install wandb
  fi
fi

{
  echo "RUN_ID=${RUN_ID}"
  echo "GPU_COUNT=${GPU_COUNT}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
  echo "TRAIN_ENTRYPOINT=${RECORD_DIR}/train_gpt.py"
  env | LC_ALL=C sort
} > "${RUN_DIR}/run.env"

if [[ -n "${SOURCE_GIT_COMMIT:-}" ]]; then
  printf '%s\n' "${SOURCE_GIT_COMMIT}" > "${RUN_DIR}/git_commit.txt"
elif git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null; then
  :
else
  printf 'snapshot-no-git\n' > "${RUN_DIR}/git_commit.txt"
fi

TRAIN_LOG="${RUN_DIR}/train.log"
CONSOLE_LOG="${RUN_DIR}/console.log"
echo "==> Run directory: ${RUN_DIR}"
echo "==> Structured train log: ${TRAIN_LOG}"
echo "==> Console capture: ${CONSOLE_LOG}"

cd "${REPO_ROOT}"

export RUN_ID
export DATA_PATH
export TOKENIZER_PATH
export EXPERIMENT_DIR="${RUN_DIR}"

schedule_remote_pod_stop() {
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    return 0
  fi
  if ! command -v runpodctl >/dev/null 2>&1; then
    return 0
  fi
  local delay_seconds="${RUNPOD_AUTO_STOP_DELAY_SECONDS:-8}"
  local stop_log="${RUN_DIR}/remote_stop.log"
  nohup bash -lc "sleep ${delay_seconds}; runpodctl pod stop ${RUNPOD_POD_ID}" > "${stop_log}" 2>&1 </dev/null &
}

echo $$ > "${TRAIN_PID_FILE}"
cleanup() {
  rm -f "${TRAIN_PID_FILE}"
  schedule_remote_pod_stop
}
trap cleanup EXIT

set -o pipefail
if [[ -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_CMD=("${TORCHRUN_BIN}" --standalone --nproc_per_node="${GPU_COUNT}")
else
  TORCHRUN_CMD=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPU_COUNT}")
fi
"${TORCHRUN_CMD[@]}" "${RECORD_DIR}/train_gpt.py" 2>&1 | tee "${CONSOLE_LOG}"
