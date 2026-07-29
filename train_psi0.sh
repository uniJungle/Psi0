#!/bin/bash
set -euo pipefail

# Psi-0 微调训练入口（G1 Sonic 真机）
# 模仿 train_act.sh 的风格，但针对 PSI0 模型

PROJECT_ROOT="/sh/zzy/Psi0"
VENV_PATH="${PROJECT_ROOT}/.venv-psi/bin/activate"
CACHE_ROOT="/sh/zzy/.cache"

# 数据集：root_dir + train_repo_ids -> .../lerobot
DATASET_ROOT="/hfm/data/sonic/lerobot"
DATA_REPO_ID="${1:-}"

# 默认任务（如果没有通过参数传入）
if [[ -z "${DATA_REPO_ID}" ]]; then
    echo "Usage: $0 <task> [exp_name]"
    echo "Example: $0 Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw pick-toys"
    exit 1
fi

EXP_NAME="${2:-}"
task_words=$(echo "$DATA_REPO_ID" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
EXP_NAME="${EXP_NAME:-$default_exp}"

# PSI0 超参数（在此调节）
TRAIN_BATCH_SIZE="16"
GRADIENT_ACCUMULATION_STEPS="1"
MAX_TRAINING_STEPS="40000"
CHECKPOINTING_STEPS="5000"
VALIDATION_STEPS="1000"
LEARNING_RATE="1e-4"
WARMUP_STEPS="1000"
MAX_GRAD_NORM="1.0"

# 模型配置
ACTION_CHUNK_SIZE="30"
ACTION_DIM="78"
ACTION_EXEC_HORIZON="30"
OBSERVATION_HORIZON="1"
ODIM="43"
VIEW_FEATURE_DIM="2048"
MAX_DELAY="8"
TRAIN_DIFFUSION_STEPS="1000"

# 预训练模型路径
MODEL_CKPT_PATH="/hfm/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
PRETRAINED_ACTION_HEADER_PATH="/hfm/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "${CUDA_DEVICES}" | tr ',' '\n' | wc -l)

# wandb（设 WANDB_ENABLED=false 可跳过）
WANDB_ENABLED="true"
WANDB_PROJECT="psi0-sonic-${DATA_REPO_ID}"
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

export task="${DATA_REPO_ID}"
export exp="${EXP_NAME}"

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

# Find an available TCP port starting at 29500
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

echo "===== PSI0 ENV CHECK ====="
echo "PWD=$(pwd)"
echo "VENV=${VENV_PATH}"
echo "PYTHON=$(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "DATA_REPO_ID=${DATA_REPO_ID}"
echo "EXP_NAME=${EXP_NAME}"
echo "WANDB_ENABLED=${WANDB_ENABLED}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "HF_HOME=${HF_HOME}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "TMPDIR=${TMPDIR}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} (x grad_accum ${GRADIENT_ACCUMULATION_STEPS} x ${NUM_GPUS} GPU)"
echo "MAX_TRAINING_STEPS=${MAX_TRAINING_STEPS}  CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS}"
echo "MODEL_CKPT_PATH=${MODEL_CKPT_PATH}"

# 构建训练参数
args="
finetune_real_psi0_config \
--seed=292285 \
--exp=${EXP_NAME} \
--train.name=sonic \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=${TRAIN_BATCH_SIZE} \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
--train.learning_rate=${LEARNING_RATE} \
--train.max_training_steps=${MAX_TRAINING_STEPS} \
--train.warmup_ratio=None \
--train.warmup_steps=${WARMUP_STEPS} \
--train.checkpointing_steps=${CHECKPOINTING_STEPS} \
--train.validation_steps=${VALIDATION_STEPS} \
--train.val_num_batches=20 \
--train.max_grad_norm=${MAX_GRAD_NORM} \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=${DATASET_ROOT} \
--data.train_repo_ids=${DATA_REPO_ID} \
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.model.img-aug \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--model.model_name_or_path=${MODEL_CKPT_PATH} \
--model.pretrained-action-header-path=${PRETRAINED_ACTION_HEADER_PATH} \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=${TRAIN_DIFFUSION_STEPS} \
--model.n_conditions=0 \
--model.action-chunk-size=${ACTION_CHUNK_SIZE} \
--model.action-dim=${ACTION_DIM} \
--model.action-exec-horizon=${ACTION_EXEC_HORIZON} \
--model.observation-horizon=${OBSERVATION_HORIZON} \
--model.odim=${ODIM} \
--model.view_feature_dim=${VIEW_FEATURE_DIM} \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.rtc \
--model.max-delay=${MAX_DELAY}
"

exec torchrun --nproc_per_node=$NUM_GPUS --master_port=${MAIN_PORT} scripts/train.py ${args}
