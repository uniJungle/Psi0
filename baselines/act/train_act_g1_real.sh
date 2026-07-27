#!/bin/bash

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_HOME=${HOME}/.cache/torch
export HF_HOME=${HOME}/.cache/huggingface
export HF_LEROBOT_HOME=${HOME}/.cache/lerobot
export DATA_HOME=${HOME}/.cache/data
export UV_CACHE_DIR=${HOME}/.cache/uv

source .venv-act/bin/activate

nprocs=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

ulimit -n 65535

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [exp]"
    echo "Example: $0 pick_apple"
    echo "  (spaces, commas, periods will be removed automatically)"
    exit 1
fi

export task="${1// /_}"
# Remove special characters (keep only alphanumeric and underscore)
task="${task//,/}"
task="${task//./}"
task="${task//:/}"
task="${task//\"/}"
task="${task//\'/}"
task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
export exp=${2:-$default_exp}

echo "Task: $task"
echo "Experiment name: $exp"

args="
real_act_config \
--seed=2026 \
--exp=$exp \
--train.name=act-g1 \
--log.report_to=wandb \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.gradient_accumulation_steps=2 \
--train.validation_steps=1000 \
--train.val_num_batches=10 \
--train.max_training_steps=60000 \
--train.learning_rate=1e-4 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--train.lr_scheduler_type=cosine \
--train.warmup_steps=1000 \
--train.warmup_ratio=None \
--train.checkpointing_steps=10000 \
--data.root_dir=/home/zzz/unitree_sh_disk/tools/ycb/datasets/SONIC_converted/ \
--data.train_repo_ids=$task \
--data.transform.repack.pad_action_dim=68 \
--data.transform.repack.pad_state_dim=33 \
--data.transform.field.stat_path=meta/stats.json \
--data.transform.field.stat_action_key=action \
--data.transform.field.stat_state_key=states \
--data.transform.field.normalize_state \
--data.transform.field.action_norm_type=bounds \
--data.transform.model.img_aug \
--data.transform.repack.action_chunk_size=50 \
--data.transform.repack.image_keys=observation.images.egocentric_right \
--model.chunk_size=50 \
--model.n_action_steps=50 \
--model.action_dim=68 \
--model.state_dim=33 \
--model.use_vae \
--model.kl_weight=10.0
"

torchrun --standalone --nnodes=1 --nproc-per-node=$nprocs scripts/train.py \
    $args
