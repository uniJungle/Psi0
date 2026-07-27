#!/bin/bash

# No-tmux version of collect_psi0-sonic-data.sh: run each component in its own
# terminal. Use this if tmux is not available.
#
# Real robot (start the camera server on the robot first):
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy \
#       --low-latency                                                   # 1) C++ controller
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico       # 2) PICO streamer
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
#       --use-stereo-camera                                               # 3) data exporter (records)
#
# PICO options (defaults: Brainco on enx6c1ff7c12485):
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico \
#       --eef brainco --dds-interface enx6c1ff7c12485
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico --eef none
#
# Exporter camera (required; mutually exclusive):
#   --use-stereo-camera   record ego_view_left / ego_view_right
#   --use-mono-camera     record ego_view (Psi0 original)
#
# Simulation teleop test (no robot/camera, no recording):
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh sim         # 1) MuJoCo sim
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy sim  # 2) C++ controller (sim)
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico        # 3) PICO streamer

PSI0_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

ROBOT_IP=192.168.123.164
TASK="Pick bottle and turn and pour into cup."
TASK_NAME="pick_bottle"
FPS=30
OUTPUT_DIR="$PSI0_ROOT/outputs/SONIC"
EEF="brainco"
DDS_INTERFACE="enx6c1ff7c12485"
CAMERA_MODE=""  # stereo | mono
LOW_LATENCY=false

SONIC_DIR="$(cd "$PSI0_ROOT/third_party/GR00T-WholeBodyControl" && pwd)"
cd "$SONIC_DIR"

USAGE="Usage: $0 {sim|deploy [sim|real|IFACE|IP]|pico|exporter} [options]
Options:
  --low-latency                (deploy; use policy/low_latency 4-frame SONIC model)
  --task-prompt TEXT
  --task-name NAME
  --root-output-dir DIR
  --eef {none|brainco|dex3}     (pico: none|brainco; exporter: dex3|brainco; default: brainco)
  --dds-interface IFACE         (pico/exporter; default: enx6c1ff7c12485)
  --use-stereo-camera           (exporter; stereo ego_view_left/right)
  --use-mono-camera             (exporter; mono ego_view)"

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "$USAGE"
    exit 1
fi
shift

DEPLOY_TARGET="real"
if [ "$MODE" = "deploy" ]; then
    if [ -n "${1:-}" ] && [[ "$1" != --* ]]; then
        DEPLOY_TARGET="$1"
        shift
    fi
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --low-latency)
            LOW_LATENCY=true
            shift
            ;;
        --task-prompt)
            TASK="$2"
            shift 2
            ;;
        --task-name)
            TASK_NAME="$2"
            shift 2
            ;;
        --root-output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eef)
            EEF="$2"
            shift 2
            ;;
        --dds-interface)
            DDS_INTERFACE="$2"
            shift 2
            ;;
        --use-stereo-camera)
            if [ -n "$CAMERA_MODE" ]; then
                echo "Choose only one of --use-stereo-camera / --use-mono-camera"
                echo "$USAGE"
                exit 1
            fi
            CAMERA_MODE="stereo"
            shift
            ;;
        --use-mono-camera)
            if [ -n "$CAMERA_MODE" ]; then
                echo "Choose only one of --use-stereo-camera / --use-mono-camera"
                echo "$USAGE"
                exit 1
            fi
            CAMERA_MODE="mono"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "$USAGE"
            exit 1
            ;;
    esac
done

if [ "$LOW_LATENCY" = true ] && [ "$MODE" != "deploy" ]; then
    echo "--low-latency is only valid with deploy"
    echo "$USAGE"
    exit 1
fi

case "$MODE" in
    sim)
        source .venv_teleop/bin/activate
        python gear_sonic/scripts/run_sim_loop.py
        ;;
    deploy)
        cd gear_sonic_deploy
        source scripts/setup_env.sh
        DEPLOY_ARGS=(--input-type zmq_manager)
        if [ "$LOW_LATENCY" = true ]; then
            LOW_LATENCY_MODEL="policy/low_latency/model"
            LOW_LATENCY_CONFIG="policy/low_latency/observation_config.yaml"
            for REQUIRED_FILE in \
                "${LOW_LATENCY_MODEL}_decoder.onnx" \
                "${LOW_LATENCY_MODEL}_encoder.onnx" \
                "$LOW_LATENCY_CONFIG"; do
                if [ ! -f "$REQUIRED_FILE" ]; then
                    echo "Missing low-latency file: $PWD/$REQUIRED_FILE"
                    exit 1
                fi
            done
            DEPLOY_ARGS+=(
                --cp "$LOW_LATENCY_MODEL"
                --obs-config "$LOW_LATENCY_CONFIG"
            )
            echo "[deploy] SONIC model=low-latency (SMPL 4-frame lookahead)"
        else
            echo "[deploy] SONIC model=release (default)"
        fi
        ./deploy.sh "${DEPLOY_ARGS[@]}" "$DEPLOY_TARGET"
        ;;
    pico)
        source .venv_teleop/bin/activate
        echo "[pico] eef=$EEF dds-interface=$DDS_INTERFACE"
        python gear_sonic/scripts/pico_manager_thread_server.py \
            --manager --eef "$EEF" --dds-interface "$DDS_INTERFACE"
        ;;
    exporter)
        if [ -z "$CAMERA_MODE" ]; then
            echo "exporter requires --use-stereo-camera or --use-mono-camera"
            echo "$USAGE"
            exit 1
        fi
        mkdir -p "$OUTPUT_DIR"
        source .venv_data_collection/bin/activate
        EXPORTER_ARGS=(
            --camera-host "$ROBOT_IP"
            --task-prompt "$TASK"
            --task-name "$TASK_NAME"
            --data-collection-frequency "$FPS"
            --root-output-dir "$OUTPUT_DIR"
            --eef "$EEF"
            --dds-interface "$DDS_INTERFACE"
        )
        if [ "$CAMERA_MODE" = "stereo" ]; then
            EXPORTER_ARGS+=(--record-stereo-ego)
        fi
        echo "[exporter] camera=$CAMERA_MODE"
        python gear_sonic/scripts/run_data_exporter.py "${EXPORTER_ARGS[@]}"
        ;;
    *)
        echo "$USAGE"
        exit 1
        ;;
esac
