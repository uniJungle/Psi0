#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

source .venv-act/bin/activate

RUN_DIR=""
CKPT_STEP=""
N_ACTION_STEPS=""

usage() {
    echo "Usage: $0 --run-dir <RUN_DIR> --ckpt-step <STEP> [--n-action-steps <N>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR="$2"; shift 2 ;;
        --ckpt-step)
            CKPT_STEP="$2"; shift 2 ;;
        --n-action-steps)
            N_ACTION_STEPS="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown argument: $1"
            usage ;;
    esac
done

if [[ -z "${RUN_DIR}" || -z "${CKPT_STEP}" ]]; then
    usage
fi

EXTRA_ARGS=()
if [[ -n "${N_ACTION_STEPS}" ]]; then
    EXTRA_ARGS+=(--action-exec-horizon="${N_ACTION_STEPS}")
fi

python src/act/deploy/act_g1_serve_real.py \
    --host=0.0.0.0 \
    --port=22085 \
    --policy=act \
    --run-dir="${RUN_DIR}" \
    --ckpt-step="${CKPT_STEP}" \
    "${EXTRA_ARGS[@]}"
