#!/bin/bash
set -euo pipefail

# Psi-0 ACT 训练入口（G1 Sonic 真机）
# 环境/缓存/wandb 在此配置；ACT 超参与 torchrun 见 baselines/act/train_act_g1_real.sh

PROJECT_ROOT="/sh/zzy/Psi0"
VENV_PATH="${PROJECT_ROOT}/.venv-act/bin/activate"
CACHE_ROOT="/sh/zzy/.cache"

# 数据集：root_dir + train_repo_ids -> .../lerobot_v2.1
DATASET_ROOT="/sh/datasets/g1/sonic/walk_to_table_and_place_apple_on_pink_plate_100"
DATA_REPO_ID="lerobot_v2.1"

CHECKPOINT_BASE_DIR="/sh/zzy/checkpoints"
EXP_NAME="ACT_200k_g1_33d_walk_to_table_and_place_apple_on_pink_plate_100"

# ACT 超参数（在此调节，会通过 env 传给 baselines/act/train_act_g1_real.sh）
TRAIN_BATCH_SIZE="128"
GRADIENT_ACCUMULATION_STEPS="1"
MAX_TRAINING_STEPS="200000"
CHECKPOINTING_STEPS="10000"
CHUNK_SIZE="100"
N_ACTION_STEPS="1"

CUDA_DEVICES="0,1"
NUM_GPUS=$(echo "${CUDA_DEVICES}" | tr ',' '\n' | wc -l)

# wandb（设 WANDB_ENABLED=false 可跳过）
WANDB_ENABLED="true"
WANDB_PROJECT="walk_to_table_and_place_apple_on_pink_plate_100"
WANDB_RUN_ID=""

cd "${PROJECT_ROOT}"
source "${VENV_PATH}"

# scripts/train.py 需要项目根目录存在 .env
if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  cat > "${PROJECT_ROOT}/.env" <<EOF
PSI_HOME=${PROJECT_ROOT}
DATA_HOME=${CACHE_ROOT}/data
HF_HOME=${CACHE_ROOT}/huggingface
TORCH_HOME=${CACHE_ROOT}/torch
UV_CACHE_DIR=${CACHE_ROOT}/uv
HF_LEROBOT_HOME=${CACHE_ROOT}/lerobot
OMP_NUM_THREADS=32
TOKENIZERS_PARALLELISM=false
EOF
  echo "Created ${PROJECT_ROOT}/.env"
fi

export PYTHONPATH="${PROJECT_ROOT}"

mkdir -p "${CACHE_ROOT}/huggingface/hub"
mkdir -p "${CACHE_ROOT}/huggingface/datasets"
mkdir -p "${CACHE_ROOT}/lerobot"
mkdir -p "${CACHE_ROOT}/data"
mkdir -p "${CACHE_ROOT}/xdg"
mkdir -p "${CACHE_ROOT}/uv"
mkdir -p "${CACHE_ROOT}/torch"
mkdir -p "${CACHE_ROOT}/torch_extensions"
mkdir -p "${CACHE_ROOT}/triton"
mkdir -p "${CACHE_ROOT}/cuda"
mkdir -p "${CACHE_ROOT}/tmp"
mkdir -p "${CACHE_ROOT}/wandb"
mkdir -p "${CHECKPOINT_BASE_DIR}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME="${CACHE_ROOT}/lerobot"
export DATA_HOME="${CACHE_ROOT}/data"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${CACHE_ROOT}/tmp"
export TMP="${CACHE_ROOT}/tmp"

export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb/cache"
export WANDB_CONFIG_DIR="${CACHE_ROOT}/wandb/config"
export WANDB_DATA_DIR="${CACHE_ROOT}/wandb/data"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"

export OMP_NUM_THREADS="32"
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONUNBUFFERED=1

# 远程机错误代理会导致 wandb ProxyError
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 传给 baselines/act/train_act_g1_real.sh
export DATA_ROOT="${DATASET_ROOT}"
export DATA_REPO_ID
export OUTPUT_DIR="${CHECKPOINT_BASE_DIR}"
export EXP_NAME
export WANDB_PROJECT
export TRAIN_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export MAX_TRAINING_STEPS
export CHECKPOINTING_STEPS
export CHUNK_SIZE
export N_ACTION_STEPS

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  export REPORT_TO=wandb
  unset WANDB_DISABLED
  unset WANDB_MODE
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: 请先 export WANDB_API_KEY='你的key'  （https://wandb.ai/authorize）"
    echo "或设置 WANDB_ENABLED=false 跳过 wandb"
    exit 1
  fi
  export WANDB_API_KEY
  if [[ -n "${WANDB_RUN_ID}" ]]; then
    export WANDB_RUN_ID
    export WANDB_RESUME=allow
  else
    unset WANDB_RUN_ID WANDB_ID WANDB_RESUME
  fi
else
  export REPORT_TO=None
  export WANDB_MODE=disabled
  unset WANDB_RUN_ID WANDB_ID WANDB_RESUME
fi

# HF token 同步（gated 模型）
if [[ ! -f "${HF_HOME}/token" && -f /root/.cache/huggingface/token ]]; then
  cp /root/.cache/huggingface/token "${HF_HOME}/token"
  echo "Synced HF token -> ${HF_HOME}/token"
fi

echo "===== ACT ENV CHECK ====="
echo "PWD=$(pwd)"
echo "VENV=${VENV_PATH}"
echo "PYTHON=$(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "DATA_REPO_ID=${DATA_REPO_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EXP_NAME=${EXP_NAME}"
echo "WANDB_ENABLED=${WANDB_ENABLED}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "HF_HOME=${HF_HOME}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "TMPDIR=${TMPDIR}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} (x grad_accum ${GRADIENT_ACCUMULATION_STEPS} x ${NUM_GPUS} GPU)"
echo "MAX_TRAINING_STEPS=${MAX_TRAINING_STEPS}  CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}  N_ACTION_STEPS=${N_ACTION_STEPS}"

exec bash "${PROJECT_ROOT}/baselines/act/train_act_g1_real.sh" "${DATA_REPO_ID}" "${EXP_NAME}"
