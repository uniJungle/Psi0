#!/bin/bash

# No-tmux version of collect_psi0-sonic-data.sh: run each component in its own
# terminal. Use this if tmux is not available.
#
# Real robot (start the camera server on the robot first):
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy     # 1) C++ controller
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico       # 2) PICO streamer
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter   # 3) data exporter (records)
#
# PICO options (defaults: Brainco on enp4s0):
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico \
#       --eef brainco --dds-interface enp4s0
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico --eef none
#
# Simulation teleop test (no robot/camera, no recording):
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh sim         # 1) MuJoCo sim
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy sim  # 2) C++ controller (sim)
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico        # 3) PICO streamer

ROBOT_IP=192.168.123.164
TASK="Pick bottle and turn and pour into cup."
TASK_NAME="pick_bottle"
FPS=30
OUTPUT_DIR="/home/karthus_chen/ycb_ws/datasets/SONIC"
EEF="brainco"
DDS_INTERFACE="enp4s0"

SONIC_DIR="$(cd "$(dirname "$0")/../../../third_party/GR00T-WholeBodyControl" && pwd)"
cd "$SONIC_DIR"

USAGE="Usage: $0 {sim|deploy [sim]|pico|exporter} [options]
Options:
  --task-prompt TEXT
  --task-name NAME
  --root-output-dir DIR
  --eef {none|brainco}          (pico; default: brainco)
  --dds-interface IFACE         (pico; default: enp4s0)"

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "$USAGE"
    exit 1
fi
shift

DEPLOY_TARGET="real"
if [ "$MODE" = "deploy" ]; then
    if [ "${1:-}" = "sim" ] || [ "${1:-}" = "real" ]; then
        DEPLOY_TARGET="$1"
        shift
    fi
fi

while [ $# -gt 0 ]; do
    case "$1" in
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
        *)
            echo "Unknown argument: $1"
            echo "$USAGE"
            exit 1
            ;;
    esac
done

case "$MODE" in
    sim)
        source .venv_teleop/bin/activate
        python gear_sonic/scripts/run_sim_loop.py
        ;;
    deploy)
        cd gear_sonic_deploy
        source scripts/setup_env.sh
        ./deploy.sh --input-type zmq_manager "$DEPLOY_TARGET"
        ;;
    pico)
        source .venv_teleop/bin/activate
        echo "[pico] eef=$EEF dds-interface=$DDS_INTERFACE"
        python gear_sonic/scripts/pico_manager_thread_server.py \
            --manager --eef "$EEF" --dds-interface "$DDS_INTERFACE"
        ;;
    exporter)
        mkdir -p "$OUTPUT_DIR"
        source .venv_data_collection/bin/activate
        python gear_sonic/scripts/run_data_exporter.py \
            --camera-host "$ROBOT_IP" \
            --task-prompt "$TASK" \
            --task-name "$TASK_NAME" \
            --data-collection-frequency "$FPS" \
            --root-output-dir "$OUTPUT_DIR" \
            --record-stereo-ego
        ;;
    *)
        echo "$USAGE"
        exit 1
        ;;
esac
