#!/bin/bash
set -euo pipefail

# Psi0 RTC client for SONIC + Brainco (33D state / 68D action).
# Run AFTER: robot camera, C++ deploy, enable_control.py, and serve_psi0-rtc-sonic.sh

PSI0_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PSI0_ROOT"

PORT="${PORT:-8014}"
HOST="${HOST:-localhost}"
INSTRUCTION="${INSTRUCTION:-walk to table and place apple on pink plate}"
CAMERA_ADDRESS="${CAMERA_ADDRESS:-tcp://192.168.123.164:5555}"
DDS_INTERFACE="${DDS_INTERFACE:-enp5s0}"
EEF="${EEF:-brainco}"

source third_party/GR00T-WholeBodyControl/.venv_teleop/bin/activate

python real/deploy/psi_inference.py \
    --host "$HOST" \
    --port "$PORT" \
    --instruction "$INSTRUCTION" \
    --camera-address "$CAMERA_ADDRESS" \
    --eef "$EEF" \
    --dds-interface "$DDS_INTERFACE"
