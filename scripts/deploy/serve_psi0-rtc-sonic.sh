#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Override before running, e.g.:
#   export CHECKPOINT_DIR=/home/karthus_chen/ycb_ws/checkpoints/PSI0_40k_g1_33d_...
#   export CHECKPOINT_STEP=40000
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/karthus_chen/ycb_ws/checkpoints/PSI0_40k_g1_33d_walk_to_table_and_place_apple_on_pink_plate_100.real.flow1000.cosine.lr1.0e-04.b64.gpus2.2607291158}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-40000}"

if [[ ! -d "$CHECKPOINT_DIR/checkpoints/ckpt_${CHECKPOINT_STEP}" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT_DIR/checkpoints/ckpt_${CHECKPOINT_STEP}"
    echo "Set CHECKPOINT_DIR and CHECKPOINT_STEP to your finetuned run."
    exit 1
fi

echo "Serving Psi0 RTC from $CHECKPOINT_DIR (ckpt_${CHECKPOINT_STEP}) on GPU $CUDA_VISIBLE_DEVICES"

python src/psi/deploy/psi_serve_rtc_token-sonic.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir="${CHECKPOINT_DIR}" \
    --ckpt-step="${CHECKPOINT_STEP}"
