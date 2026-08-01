#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/sh/zzy/Psi0"
VENV_PATH="${PROJECT_ROOT}/.venv-openpi/bin/activate"
CACHE_ROOT="/sh/zzy/.cache"
OPENPI_CACHE="/sh/ycb/.cache/openpi"

DATASET_ROOT="/sh/datasets/g1/sonic/walk_to_table_and_place_apple_on_pink_plate_100/lerobot_v2.1"
CONFIG_NAME="walk_to_table_and_place_apple_on_pink_plate_100"
ASSET_ID="walk_to_table_and_place_apple_on_pink_plate_100"
EXP_NAME="PI05_40k_vlmfreeze_g1_68d_walk_to_table_and_place_apple_on_pink_plate_100"

CHECKPOINT_BASE_DIR="/sh/zzy/checkpoints/openpi-05"
ASSETS_BASE_DIR="${CACHE_ROOT}/openpi/assets"
JAX_CKPT_DIR="${OPENPI_CACHE}/openpi-assets/checkpoints/pi05_base"
PYTORCH_WEIGHT_PATH="${OPENPI_CACHE}/openpi-assets/checkpoints/pi05_base_pytorch"
CONVERT_SCRIPT="/sh/ycb/model/openpi/examples/convert_jax_model_to_pytorch.py"

# 默认超参（与 README / config.py 一致；batch_size 为 global）
BATCH_SIZE="64"
NUM_TRAIN_STEPS="40000"
SAVE_INTERVAL="10000"
CUDA_DEVICES="0,1"
NUM_GPUS=$(echo "${CUDA_DEVICES}" | tr ',' '\n' | wc -l)

WANDB_ENABLED="true"
WANDB_PROJECT="psi"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}"

# OpenPI 要求的 transformers patch（见 baselines/pi05/README.md）
TRANSFORMERS_DIR="${PROJECT_ROOT}/.venv-openpi/lib/python3.10/site-packages/transformers"
PATCH_SRC="${PROJECT_ROOT}/src/openpi/models_pytorch/transformers_replace"
if [[ -d "${PATCH_SRC}" && -d "${TRANSFORMERS_DIR}" ]]; then
  cp -r "${PATCH_SRC}/"* "${TRANSFORMERS_DIR}/"
fi

# scripts require .env
if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  cat > "${PROJECT_ROOT}/.env" <<EOF
PSI_HOME=${PROJECT_ROOT}
DATA_HOME=${CACHE_ROOT}/data
HF_HOME=${CACHE_ROOT}/huggingface
TORCH_HOME=${CACHE_ROOT}/torch
UV_CACHE_DIR=${CACHE_ROOT}/uv
HF_LEROBOT_HOME=${CACHE_ROOT}/lerobot
OPENPI_DATA_HOME=${OPENPI_CACHE}
OMP_NUM_THREADS=32
TOKENIZERS_PARALLELISM=false
EOF
  echo "Created ${PROJECT_ROOT}/.env"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

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
mkdir -p "${ASSETS_BASE_DIR}/${CONFIG_NAME}/${ASSET_ID}"
mkdir -p "${CHECKPOINT_BASE_DIR}"

# 预训练权重 / VLM tokenizer：只读 /sh/ycb/.cache/openpi，不重新下载
export OPENPI_DATA_HOME="${OPENPI_CACHE}"
# config.py 模块加载时会读这些环境变量（转换脚本也会 import）
export PSI_HOME="${PROJECT_ROOT}"
export DATA_HOME="${CACHE_ROOT}/data"

# 训练过程可变缓存放到 /sh/zzy/.cache
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME="${CACHE_ROOT}/lerobot"
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

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  unset WANDB_DISABLED
  unset WANDB_MODE
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: 请先 export WANDB_API_KEY='你的key'  （https://wandb.ai/authorize）"
    echo "或设置 WANDB_ENABLED=false 跳过 wandb"
    exit 1
  fi
  export WANDB_API_KEY
  unset WANDB_RUN_ID WANDB_ID WANDB_RESUME
else
  export WANDB_MODE=disabled
  unset WANDB_RUN_ID WANDB_ID WANDB_RESUME
fi

if [[ ! -f "${HF_HOME}/token" && -f /root/.cache/huggingface/token ]]; then
  cp /root/.cache/huggingface/token "${HF_HOME}/token"
  echo "Synced HF token -> ${HF_HOME}/token"
fi

# ---- norm_stats（openpi 格式）----
NORM_STATS_DIR="${ASSETS_BASE_DIR}/${CONFIG_NAME}/${ASSET_ID}"
if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "Writing openpi norm_stats -> ${NORM_STATS_DIR}/norm_stats.json"
  python - <<PY
import json
from pathlib import Path
src = Path("${DATASET_ROOT}/meta/stats.json")
dst = Path("${NORM_STATS_DIR}/norm_stats.json")
stats = json.loads(src.read_text())
out = {"norm_stats": {"state": stats["states"], "actions": stats["action"]}}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=2))
print(f"Wrote {dst}")
PY
fi

# 用 action_dim=32 的 base config 做转换（与 JAX pi05_base 一致）；
# 训练时 train_pytorch.py 会把 32 维权重 pad 到 config.action_dim（68）
CONVERT_CONFIG_NAME="pi05_libero"

# ---- JAX pi05_base -> PyTorch safetensors（本地转换，不下载）----
if [[ ! -f "${PYTORCH_WEIGHT_PATH}/model.safetensors" ]]; then
  echo "Converting local JAX pi05_base -> PyTorch: ${PYTORCH_WEIGHT_PATH}"
  if [[ ! -d "${JAX_CKPT_DIR}/params" ]]; then
    echo "ERROR: missing JAX ckpt at ${JAX_CKPT_DIR}/params"
    exit 1
  fi
  if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "ERROR: missing convert script ${CONVERT_SCRIPT}"
    exit 1
  fi
  # 清掉上次失败留下的半成品
  rm -rf "${PYTORCH_WEIGHT_PATH}"
  python "${CONVERT_SCRIPT}" \
    --checkpoint_dir="${JAX_CKPT_DIR}" \
    --config_name="${CONVERT_CONFIG_NAME}" \
    --output_path="${PYTORCH_WEIGHT_PATH}"
fi

find_free_port() {
    start_port=${1:-29500}
    port=${start_port}
    while true; do
        CHECK_PORT=${port} python - <<'PY'
import os,sys,socket
port = int(os.environ.get('CHECK_PORT','0'))
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('0.0.0.0', port))
    sock.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
        if [ $? -eq 0 ]; then
            echo ${port}
            return 0
        fi
        port=$((port+1))
        if [ ${port} -gt $((start_port+1000)) ]; then
            echo "Failed to find free port after 1000 attempts" >&2
            return 1
        fi
    done
}

MAIN_PORT=$(find_free_port 29500)
if [ -z "${MAIN_PORT}" ]; then
    echo "Could not find free main process port, aborting." >&2
    exit 1
fi

ulimit -n 65535

echo "===== PI05 ENV CHECK ====="
echo "PWD=$(pwd)"
echo "VENV=${VENV_PATH}"
echo "PYTHON=$(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "PYTORCH_WEIGHT_PATH=${PYTORCH_WEIGHT_PATH}"
echo "ASSETS_BASE_DIR=${ASSETS_BASE_DIR}"
echo "CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}"
echo "BATCH_SIZE(global)=${BATCH_SIZE}  NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}"
echo "WANDB_ENABLED=${WANDB_ENABLED}"
echo "NOTE: pytorch trainer freezes PaliGemma VLM (action-expert finetune); true LoRA is JAX-only"

WANDB_FLAG=""
if [[ "${WANDB_ENABLED}" != "true" ]]; then
  WANDB_FLAG="--no-wandb_enabled"
fi

exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${MAIN_PORT}" \
  src/openpi/train_pytorch.py \
  "${CONFIG_NAME}" \
  --exp_name="${EXP_NAME}" \
  --checkpoint_base_dir="${CHECKPOINT_BASE_DIR}" \
  --assets_base_dir="${ASSETS_BASE_DIR}" \
  --pytorch_weight_path="${PYTORCH_WEIGHT_PATH}" \
  --batch_size="${BATCH_SIZE}" \
  --num_train_steps="${NUM_TRAIN_STEPS}" \
  --save_interval="${SAVE_INTERVAL}" \
  ${WANDB_FLAG}
