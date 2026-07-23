#!/bin/bash
#
# Replay simulation launcher: starts MuJoCo + C++ WBC (sim) + replay script.
#
# Usage:
#   bash scripts/replay/replay_sim_launch.sh
#   bash scripts/replay/replay_sim_launch.sh ./data/pick_n_squat 0
#   bash scripts/replay/replay_sim_launch.sh ./data/my_dataset 3
#
# Prerequisites:
#   1. MuJoCo + C++ WBC must already be built:
#        cd third_party/GR00T-WholeBodyControl/gear_sonic_deploy && just build
#
# What this script does (in order):
#   1. Launch MuJoCo sim loop (run_sim_loop.py) in background
#   2. Launch C++ WBC in sim mode (deploy.sh) in background
#   3. Run the Python replay script (replay_sim.py) in foreground
#   4. On exit (Ctrl+C or replay end), kill MuJoCo and C++ WBC processes
#

set -e

# -------------------- Argument parsing --------------------
DATA_DIR="${1:-./data/pick_n_squat}"
EPISODE_IDX="${2:-0}"
FPS="${3:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SONIC_DIR="$REPO_ROOT/third_party/GR00T-WholeBodyControl"

# -------------------- Colors --------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# -------------------- Check prerequisites --------------------
echo -e "${CYAN}=== Replay Simulation Launcher ===${NC}"
echo -e "Data dir   : ${GREEN}$DATA_DIR${NC}"
echo -e "Episode    : ${GREEN}$EPISODE_IDX${NC}"
echo -e "FPS        : ${GREEN}$FPS${NC}"
echo ""

# Check MuJoCo sim loop
if [ ! -f "$SONIC_DIR/gear_sonic/scripts/run_sim_loop.py" ]; then
    echo -e "${RED}ERROR: run_sim_loop.py not found at $SONIC_DIR/gear_sonic/scripts/run_sim_loop.py${NC}"
    exit 1
fi

# Check C++ WBC deploy script
if [ ! -f "$SONIC_DIR/gear_sonic_deploy/deploy.sh" ]; then
    echo -e "${RED}ERROR: deploy.sh not found at $SONIC_DIR/gear_sonic_deploy/deploy.sh${NC}"
    exit 1
fi

# Check replay script
if [ ! -f "$SCRIPT_DIR/replay_sim.py" ]; then
    echo -e "${RED}ERROR: replay_sim.py not found at $SCRIPT_DIR/replay_sim.py${NC}"
    exit 1
fi

# Check dataset
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}ERROR: Dataset not found at $DATA_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] All prerequisites passed${NC}"
echo ""

# -------------------- Step 1: Launch MuJoCo --------------------
echo -e "${CYAN}[Step 1/3] Launching MuJoCo sim loop...${NC}"
cd "$SONIC_DIR"
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py &
PID_MUJOCO=$!
echo -e "${GREEN}[OK] MuJoCo started (PID=$PID_MUJOCO)${NC}"

echo -e "${YELLOW}Waiting 5s for MuJoCo to initialize...${NC}"
sleep 5

# -------------------- Step 2: Launch C++ WBC sim --------------------
echo -e "${CYAN}[Step 2/3] Launching C++ WBC (sim mode)...${NC}"
cd "$SONIC_DIR/gear_sonic_deploy"
source scripts/setup_env.sh
# Deploy in sim mode with --input-type zmq_manager
# The deploy.sh script asks for confirmation; we use yes or --yes flag
# Since deploy.sh uses `read -p` for confirmation, we pipe 'y' to it
echo "y" | ./deploy.sh --input-type zmq_manager sim &
PID_WBC=$!
echo -e "${GREEN}[OK] C++ WBC started (PID=$PID_WBC)${NC}"

echo -e "${YELLOW}Waiting 3s for C++ WBC to initialize...${NC}"
sleep 3

# -------------------- Step 3: Run replay script --------------------
echo -e "${CYAN}[Step 3/3] Running replay script...${NC}"
cd "$REPO_ROOT"
REPLAY_CMD="python scripts/replay/replay_sim.py \
    --data_dir '$DATA_DIR' \
    --episode_idx $EPISODE_IDX \
    --fps $FPS \
    --zmq_port 5556"
echo -e "${GREEN}Executing: $REPLAY_CMD${NC}"
echo ""

eval $REPLAY_CMD
REPLAY_EXIT=$?

# -------------------- Cleanup --------------------
echo ""
echo -e "${CYAN}=== Cleaning up processes ===${NC}"

cleanup() {
    echo -e "${YELLOW}Killing MuJoCo (PID=$PID_MUJOCO)...${NC}"
    kill -TERM $PID_MUJOCO 2>/dev/null || true

    echo -e "${YELLOW}Killing C++ WBC (PID=$PID_WBC)...${NC}"
    kill -TERM $PID_WBC 2>/dev/null || true

    # Give them time to exit
    sleep 2

    # Force kill if still alive
    kill -9 $PID_MUJOCO 2>/dev/null || true
    kill -9 $PID_WBC 2>/dev/null || true

    echo -e "${GREEN}Cleanup done.${NC}"
}

trap cleanup EXIT

echo -e "${CYAN}Replay exited with code $REPLAY_EXIT${NC}"
exit $REPLAY_EXIT
