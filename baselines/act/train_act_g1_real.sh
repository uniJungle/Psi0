#!/bin/bash
set -euo pipefail

# G1 真机 ACT 训练（由 train_act.sh 调用，也可单独运行）
# Usage: bash baselines/act/train_act_g1_real.sh <repo_id> [exp]
# Example: bash baselines/act/train_act_g1_real.sh lerobot_v2.1 walk-to-table

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CACHE_ROOT="${CACHE_ROOT:-/sh/ycb/.cache}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${CACHE_ROOT}/lerobot}"
export DATA_HOME="${DATA_HOME:-${CACHE_ROOT}/data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "${PROJECT_ROOT}/.venv-act/bin/activate"
fi

nprocs=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
ulimit -n 65535

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <repo_id> [exp]"
  echo "Example: $0 lerobot_v2.1 walk-to-table"
  exit 1
fi

if [[ -n "${DATA_REPO_ID:-}" ]]; then
  task="${DATA_REPO_ID}"
else
  task="${1}"
  task="${task// /_}"
  task="${task//,/}"
fi

if [[ "$#" -ge 2 ]]; then
  exp="$2"
else
  task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
  exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
fi

DATA_ROOT="${DATA_ROOT:-/sh/datasets/g1/sonic/walk_to_table_and_place_apple_on_pink_plate_100}"
OUTPUT_DIR="${OUTPUT_DIR:-/sh/zzy/checkpoints}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-act_g1_sonic_walk_to_table_and_place_apple}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-100000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"

echo "Task (repo_id): ${task}"
echo "Experiment name: ${exp}"
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Report to: ${REPORT_TO}"
echo "Batch size (per GPU): ${TRAIN_BATCH_SIZE}, grad_accum: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Max steps: ${MAX_TRAINING_STEPS}, checkpoint every: ${CHECKPOINTING_STEPS}, chunk_size: ${CHUNK_SIZE}, n_action_steps: ${N_ACTION_STEPS}"

wandb_args=""
if [[ "${REPORT_TO}" == "wandb" ]]; then
  wandb_args="--wandb.project=${WANDB_PROJECT}"
  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    wandb_args="${wandb_args} --wandb.id=${WANDB_RUN_ID} --wandb.resume=allow"
  fi
fi

args="
real_act_config \
--seed=2026 \
--exp=${exp} \
--train.name=act-g1 \
--train.output_dir=${OUTPUT_DIR} \
--log.report_to=${REPORT_TO} \
${wandb_args} \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=${TRAIN_BATCH_SIZE} \
--train.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
--train.validation_steps=1000 \
--train.val_num_batches=10 \
--train.max_training_steps=${MAX_TRAINING_STEPS} \
--train.learning_rate=1e-4 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--train.lr_scheduler_type=cosine \
--train.warmup_steps=1000 \
--train.warmup_ratio=None \
--train.checkpointing_steps=${CHECKPOINTING_STEPS} \
--data.root_dir=${DATA_ROOT} \
--data.train_repo_ids=${task} \
--data.transform.repack.pad_action_dim=68 \
--data.transform.repack.pad_state_dim=33 \
--data.transform.field.stat_path=meta/stats.json \
--data.transform.field.stat_action_key=action \
--data.transform.field.stat_state_key=states \
--data.transform.field.normalize_state \
--data.transform.field.action_norm_type=bounds \
--data.transform.model.img_aug \
--data.transform.repack.action_chunk_size=${CHUNK_SIZE} \
--data.transform.repack.image_keys=observation.images.egocentric_right \
--model.chunk_size=${CHUNK_SIZE} \
--model.n_action_steps=${N_ACTION_STEPS} \
--model.action_dim=68 \
--model.state_dim=33 \
--model.use_vae \
--model.kl_weight=10.0
"

torchrun --standalone --nnodes=1 --nproc-per-node="${nprocs}" scripts/train.py \
  ${args}
