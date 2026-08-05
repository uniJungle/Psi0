#!/bin/bash
set -euo pipefail

# Serve π0.5 SONIC policy (OpenPI WebSocket PolicyServer).
# Expects checkpoint layout: <run_dir>/<step>/{model.safetensors,assets/,metadata.pt}

PSI0_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR_DEFAULT="/home/karthus_chen/ycb_ws/checkpoints/PI05_40k_g1_sonic_walk_to_table_place_apple_on_pink_plate_100"
CONFIG_NAME_DEFAULT="walk_to_table_and_place_apple_on_pink_plate_100"

MODEL_PATH="${MODEL_PATH:-$RUN_DIR_DEFAULT}"
CKPT_STEP="${CKPT_STEP:-}"
CONFIG_NAME="${CONFIG_NAME:-$CONFIG_NAME_DEFAULT}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Go to the table, pick up the apple, place the apple on the pink plate.}"

usage() {
    echo "Usage: $0 [--model-path PATH] [--ckpt-step STEP] [--config NAME] [--port PORT]"
    echo ""
    echo "Env overrides: MODEL_PATH, CKPT_STEP, CONFIG_NAME, PORT, CUDA_VISIBLE_DEVICES, DEFAULT_PROMPT"
    echo "Default run dir: $RUN_DIR_DEFAULT"
    echo "Default config:  $CONFIG_NAME_DEFAULT"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"; shift 2 ;;
        --ckpt-step)
            CKPT_STEP="$2"; shift 2 ;;
        --config)
            CONFIG_NAME="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --prompt)
            DEFAULT_PROMPT="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown arg: $1"; usage ;;
    esac
done

resolve_ckpt_dir() {
    local path="$1"
    local step="${2:-}"
    if [[ -n "$step" ]]; then
        if [[ -d "$path/$step" ]]; then
            echo "$path/$step"
            return 0
        fi
        if [[ -d "$path/checkpoint-$step" ]]; then
            echo "$path/checkpoint-$step"
            return 0
        fi
        echo "ERROR: checkpoint step $step not found under $path" >&2
        return 1
    fi
    if [[ -f "$path/model.safetensors" ]]; then
        echo "$path"
        return 0
    fi
    # Prefer highest numeric step dir that has model.safetensors
    local d best=""
    for d in $(ls -1d "$path"/[0-9]* 2>/dev/null | sort -V -r); do
        if [[ -f "$d/model.safetensors" ]]; then
            best="$d"
            break
        fi
    done
    if [[ -n "$best" ]]; then
        echo "$best"
        return 0
    fi
    echo "ERROR: no model.safetensors under $path (pass --ckpt-step)" >&2
    return 1
}

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: model path not found: $MODEL_PATH"
    exit 1
fi

CKPT_DIR="$(resolve_ckpt_dir "$MODEL_PATH" "$CKPT_STEP")"

cd "$PSI0_ROOT"

if [[ -f "$PSI0_ROOT/.venv-openpi/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$PSI0_ROOT/.venv-openpi/bin/activate"
elif [[ -f "/home/karthus_chen/ycb_ws/openpi/.venv/bin/activate" ]]; then
    echo "[WARN] .venv-openpi missing; falling back to /home/karthus_chen/ycb_ws/openpi/.venv"
    # shellcheck disable=SC1091
    source "/home/karthus_chen/ycb_ws/openpi/.venv/bin/activate"
else
    echo "ERROR: need .venv-openpi (see baselines/pi05/README.md)"
    exit 1
fi

if [[ ! -f "$PSI0_ROOT/.env" ]]; then
    cat > "$PSI0_ROOT/.env" <<EOF
PSI_HOME=${PSI0_ROOT}
DATA_HOME=${PSI0_ROOT}/data
HF_HOME=${HOME}/.cache/huggingface
OPENPI_DATA_HOME=${HOME}/.cache/openpi
OMP_NUM_THREADS=8
TOKENIZERS_PARALLELISM=false
EOF
    echo "[INFO] Created $PSI0_ROOT/.env"
fi

# Export cache env so tokenizer does not hit GCS (load_dotenv won't override a bad preset).
set -a
# shellcheck disable=SC1091
source "$PSI0_ROOT/.env"
set +a
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
if [[ -f "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" ]]; then
    export OPENPI_PALIGEMMA_TOKENIZER="${OPENPI_PALIGEMMA_TOKENIZER:-${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model}"
elif [[ -f /home/karthus_chen/.cache/openpi/big_vision/paligemma_tokenizer.model ]]; then
    export OPENPI_DATA_HOME=/home/karthus_chen/.cache/openpi
    export OPENPI_PALIGEMMA_TOKENIZER=/home/karthus_chen/.cache/openpi/big_vision/paligemma_tokenizer.model
elif [[ -f /home/karthus_chen/ycb_ws/model/pi05_base/paligemma_tokenizer.model ]]; then
    export OPENPI_PALIGEMMA_TOKENIZER=/home/karthus_chen/ycb_ws/model/pi05_base/paligemma_tokenizer.model
fi

# OpenPI requires patched transformers (see baselines/pi05/README.md)
PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
TRANSFORMERS_DIR="${VIRTUAL_ENV:-$PSI0_ROOT/.venv-openpi}/lib/python${PY_VER}/site-packages/transformers"
PATCH_SRC="${PSI0_ROOT}/src/openpi/models_pytorch/transformers_replace"
if [[ -d "${PATCH_SRC}" && -d "${TRANSFORMERS_DIR}" ]]; then
    cp -r "${PATCH_SRC}/"* "${TRANSFORMERS_DIR}/"
fi

export PYTHONPATH="${PSI0_ROOT}/src:${PSI0_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "===== π0.5 SONIC PolicyServer (WebSocket) ====="
echo "CONFIG_NAME=$CONFIG_NAME"
echo "CKPT_DIR=$CKPT_DIR"
echo "HOST:PORT=$HOST:$PORT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "OPENPI_PALIGEMMA_TOKENIZER=${OPENPI_PALIGEMMA_TOKENIZER:-<unset>}"

python src/openpi/deploy/serve_policy.py \
    --port="$PORT" \
    --default-prompt="$DEFAULT_PROMPT" \
    policy:checkpoint \
    --policy.config="$CONFIG_NAME" \
    --policy.dir="$CKPT_DIR"
