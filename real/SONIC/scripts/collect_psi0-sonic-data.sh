#!/bin/bash

# SONIC teleop with Psi0.
#
#   sim mode  : lightweight teleop test in MuJoCo — no camera render, no recording.
#   real mode : full data collection — starts the C++ controller, PICO streamer,
#               and data exporter, recording LeRobot datasets under OUTPUT_DIR/.
#
# Usage:
#   ./collect_psi0-sonic-data.sh              # real robot (camera server at $ROBOT_IP)
#   ./collect_psi0-sonic-data.sh sim          # simulation teleop test (no recording)
#   ./collect_psi0-sonic-data.sh 192.168.123.164
#   ./collect_psi0-sonic-data.sh --task-prompt "Pick bottle and pour."
#   ./collect_psi0-sonic-data.sh --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC
#   ./collect_psi0-sonic-data.sh 192.168.123.164 \
#       --task-prompt "Pick bottle and pour." \
#       --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC

ROBOT_IP=192.168.123.164
TASK="Pick bottle and turn and pour into cup."
TASK_NAME="pick_bottle"
FPS=30
OUTPUT_DIR="/home/karthus_chen/ycb_ws/datasets/SONIC"
EEF="brainco"
DDS_INTERFACE="enp4s0"

SONIC_DIR="$(cd "$(dirname "$0")/../../../third_party/GR00T-WholeBodyControl" && pwd)"
cd "$SONIC_DIR"

MODE="$ROBOT_IP"
while [ $# -gt 0 ]; do
    case "$1" in
        sim)
            MODE="sim"
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
        --*)
            echo "Unknown argument: $1"
            echo "Usage: $0 [sim|<ROBOT_IP>] [--task-prompt TEXT] [--task-name NAME] [--root-output-dir DIR] [--eef none|brainco] [--dds-interface IFACE]"
            exit 1
            ;;
        *)
            MODE="$1"
            shift
            ;;
    esac
done

if [ "$MODE" = "sim" ]; then
    # Sim: teleop only. Plain run_sim_loop (no --enable-image-publish/--enable-offscreen),
    # so MuJoCo does NOT render/publish camera images — much smoother, but nothing is recorded.
    SESSION=sonic_sim_teleop
    tmux kill-session -t "$SESSION" 2>/dev/null

    tmux new-session -d -s "$SESSION" -c "$SONIC_DIR"
    tmux set-option -t "$SESSION" -g mouse on   # click a pane to focus it (e.g. to press Y in the deploy pane)
    tmux send-keys -t "${SESSION}:0.0" \
        "source .venv_teleop/bin/activate && python gear_sonic/scripts/run_sim_loop.py" C-m

    tmux split-window -h -t "${SESSION}:0.0" -c "$SONIC_DIR/gear_sonic_deploy"
    tmux send-keys -t "${SESSION}:0.1" \
        "source scripts/setup_env.sh && ./deploy.sh --input-type zmq_manager sim" C-m

    tmux split-window -v -t "${SESSION}:0.1" -c "$SONIC_DIR"
    tmux send-keys -t "${SESSION}:0.2" \
        "source .venv_teleop/bin/activate && python gear_sonic/scripts/pico_manager_thread_server.py --manager --eef ${EEF} --dds-interface ${DDS_INTERFACE}" C-m

    tmux select-layout -t "$SESSION" tiled
    tmux attach -t "$SESSION"
else
    # Real robot: full data collection.
    # The launcher's built-in preview runs in the data-collection venv (lerobot's headless
    # OpenCV, no window), so disable it and run a working preview from .venv_teleop alongside.
    mkdir -p "$OUTPUT_DIR"

    ( source .venv_teleop/bin/activate && \
      python gear_sonic/scripts/run_camera_viewer.py --camera-host "$MODE" --camera-port 5555 ) &
    PREVIEW_PID=$!

    python gear_sonic/scripts/launch_data_collection.py \
        --camera-host "$MODE" \
        --task-prompt "$TASK" \
        --task-name "$TASK_NAME" \
        --data-exporter-frequency "$FPS" \
        --root-output-dir "$OUTPUT_DIR" \
        --record-stereo-ego \
        --pico-eef "$EEF" \
        --pico-dds-interface "$DDS_INTERFACE" \
        --no-camera-viewer

    kill "$PREVIEW_PID" 2>/dev/null
fi
