#!/bin/bash
set -euo pipefail

# Serve GR00T-N1.7 SONIC policy (ZMQ PolicyServer).
# Run from anywhere; uses the sibling GR00T repo + its .venv.

PSI0_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GROOT_ROOT="${GROOT_ROOT:-$(cd "$PSI0_ROOT/../GR00T" && pwd)}"

# Final HF-loadable weights live at the *run root* (config.json + safetensors + processor/).
# Intermediate ``checkpoint-NNNN`` dirs may only contain DeepSpeed shards / empty folders
# (on this machine only checkpoint-10000 is complete; 20k/30k/40k are incomplete).
RUN_DIR_DEFAULT="/home/karthus_chen/ycb_ws/checkpoints/GR00T_N1d7_40k_g1_sonic_walk_to_table_place_apple_on_pink_plate_100"
MODEL_PATH="${MODEL_PATH:-$RUN_DIR_DEFAULT}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-UNITREE_G1_SONIC}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
    echo "Usage: $0 [--model-path PATH] [--ckpt-step STEP] [--port PORT] [--device DEV]"
    echo ""
    echo "Env overrides: MODEL_PATH, GROOT_ROOT, PORT, DEVICE, CUDA_VISIBLE_DEVICES"
    echo "Default model (run root / final weights): $RUN_DIR_DEFAULT"
    echo ""
    echo "Note: prefer the run root (final HF weights)."
    echo "      Incomplete checkpoint-NNNN / empty processor/ are auto-healed via staging."
    exit 1
}

# True if dir has the three processor files GR00T AutoProcessor expects.
has_processor_files() {
    local d="$1"
    [[ -f "$d/processor_config.json" && -f "$d/statistics.json" && -f "$d/embodiment_id.json" ]]
}

# Find a donor dir that has processor files (this run often left processor/ empty
# at the final root; only some intermediate checkpoint-* dirs are complete).
find_processor_donor() {
    local model_dir="$1"
    local d parent
    if has_processor_files "$model_dir"; then
        echo "$model_dir"
        return 0
    fi
    if has_processor_files "$model_dir/processor"; then
        echo "$model_dir/processor"
        return 0
    fi
    parent="$(dirname "$model_dir")"
    # Prefer highest complete checkpoint-* under model_dir or its parent run root.
    for d in $(ls -1d "$model_dir"/checkpoint-* "$parent"/checkpoint-* 2>/dev/null | sort -V -r); do
        if has_processor_files "$d"; then
            echo "$d"
            return 0
        fi
        if has_processor_files "$d/processor"; then
            echo "$d/processor"
            return 0
        fi
    done
    return 1
}

# Build a symlink staging dir: final HF weights + processor files from a donor.
# Needed because Gr00tPolicy prefers model_dir/processor/ when that folder exists,
# and this run's processor/ is empty (no processor_config.json).
stage_loadable_model_dir() {
    local model_dir="$1"
    local donor="$2"
    local stage
    stage="$(mktemp -d /tmp/gr00t_n1d7_serve.XXXXXX)"
    # Weights / model config (no recursive processor/ — empty dir would break load)
    local f
    for f in config.json model.safetensors.index.json \
             model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
             model.safetensors; do
        if [[ -e "$model_dir/$f" ]]; then
            ln -s "$model_dir/$f" "$stage/$f"
        fi
    done
    # Also link any other shard names matching the index
    for f in "$model_dir"/model-*.safetensors; do
        [[ -e "$f" ]] || continue
        local base
        base="$(basename "$f")"
        [[ -e "$stage/$base" ]] || ln -s "$f" "$stage/$base"
    done
    for f in processor_config.json statistics.json embodiment_id.json; do
        ln -s "$donor/$f" "$stage/$f"
    done
    echo "[INFO] Staged loadable model at $stage" >&2
    echo "[INFO]   weights from: $model_dir" >&2
    echo "[INFO]   processor from: $donor" >&2
    echo "$stage"
}

# Resolve a path that AutoModel + AutoProcessor can load.
resolve_hf_model_dir() {
    local path="$1"
    local weights_dir=""
    if [[ -f "$path/config.json" && -f "$path/model.safetensors.index.json" ]]; then
        weights_dir="$path"
    else
        # checkpoint-NNNN without HF files → try parent run root
        local parent
        parent="$(dirname "$path")"
        if [[ "$(basename "$path")" == checkpoint-* ]] \
            && [[ -f "$parent/config.json" ]] \
            && [[ -f "$parent/model.safetensors.index.json" ]]; then
            echo "[WARN] $path is not a full HF checkpoint (missing config.json / weights)." >&2
            echo "[WARN] Falling back to run root weights: $parent" >&2
            weights_dir="$parent"
        fi
    fi
    if [[ -z "$weights_dir" ]]; then
        echo "ERROR: not a loadable GR00T model dir: $path" >&2
        echo "Need config.json + model.safetensors.index.json (+ processor files)." >&2
        if [[ -d "$(dirname "$path")" ]]; then
            echo "Available under $(dirname "$path"):" >&2
            ls -1 "$(dirname "$path")" | sed 's/^/  /' >&2
        fi
        return 1
    fi

    # Processor usable in-place? Prefer root processor_config.json; empty processor/
    # directory alone is NOT enough (Gr00tPolicy will load from it and crash).
    if has_processor_files "$weights_dir"; then
        # If an empty/broken processor/ exists, stage without it so AutoProcessor
        # uses the root processor_config.json.
        if [[ -d "$weights_dir/processor" ]] && ! has_processor_files "$weights_dir/processor"; then
            local donor="$weights_dir"
            stage_loadable_model_dir "$weights_dir" "$donor"
            return 0
        fi
        echo "$weights_dir"
        return 0
    fi
    if has_processor_files "$weights_dir/processor"; then
        echo "$weights_dir"
        return 0
    fi

    local donor
    if ! donor="$(find_processor_donor "$weights_dir")"; then
        echo "ERROR: no processor_config.json / statistics.json / embodiment_id.json under:" >&2
        echo "  $weights_dir (or sibling checkpoint-*)" >&2
        echo "Training left processor/ empty; need at least one complete checkpoint with processor files." >&2
        return 1
    fi
    stage_loadable_model_dir "$weights_dir" "$donor"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"; shift 2 ;;
        --ckpt-step)
            # If MODEL_PATH already points at run root, append checkpoint-N;
            # resolve_hf_model_dir will fall back to root if that ckpt is incomplete.
            if [[ "$(basename "$MODEL_PATH")" == checkpoint-* ]]; then
                MODEL_PATH="$(dirname "$MODEL_PATH")/checkpoint-${2}"
            else
                MODEL_PATH="${MODEL_PATH}/checkpoint-${2}"
            fi
            shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown arg: $1"; usage ;;
    esac
done

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: model path not found: $MODEL_PATH"
    exit 1
fi

MODEL_PATH="$(resolve_hf_model_dir "$MODEL_PATH")"

if [[ ! -d "$GROOT_ROOT" ]]; then
    echo "ERROR: GR00T repo not found: $GROOT_ROOT"
    echo "Set GROOT_ROOT to your Isaac-GR00T N1.7 checkout."
    exit 1
fi

cd "$GROOT_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

# Local Cosmos / HF offline env (do not use /sh/ycb training paths)
export GR00T_ROOT="$GROOT_ROOT"
# shellcheck disable=SC1091
source scripts/local_inference_env.sh

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "===== GR00T-N1.7 SONIC PolicyServer ====="
echo "GROOT_ROOT=$GROOT_ROOT"
echo "MODEL_PATH=$MODEL_PATH"
echo "EMBODIMENT_TAG=$EMBODIMENT_TAG"
echo "HOST:PORT=$HOST:$PORT"
echo "DEVICE=$DEVICE"
echo "COSMOS_REASON2_PATH=${COSMOS_REASON2_PATH:-<unset>}"

python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --strict
