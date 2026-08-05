# 基于 SONIC 的模型训练/推理

## ACT

### 训练
```bash
# Launch the training script via tmux
tmux new -s train_act
cd /sh/zzy/Psi0
export WANDB_API_KEY='your-wandb-api-key'
wandb login
bash train_act.sh

# Detach / reattach tmux
Detach: Ctrl-b then d
Reattach: tmux attach -t train_act
List sessions: tmux ls
```

### 开环推理
```bash
# 终端 1：启动 ACT policy server
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-act/bin/activate

bash baselines/act/serve_act_g1_real.sh \
  --run-dir /home/karthus_chen/ycb_ws/checkpoints/ACT_200k_g1_33d_walk_to_table_and_place_apple_on_pink_plate_100 \
  --ckpt-step 40000 \
  --n-action-steps 1

# 终端 2：启动开环推理端
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-act/bin/activate

python baselines/act/openloop_act_g1_real.py \
  --data-root /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate \
  --ckpt-step 40000 \
  --episode-idx 1 \
  --n-action-steps 1 \
  --host localhost \
  --port 22085
```

### 闭环推理
```bash
# 终端 1：G1 机载 Brainco hand + SONIC composed_camera（PUB :5555）
ssh unitree@192.168.123.164
bash ./sonic_start_teleop.sh

# 终端 2：SONIC C++ deploy（--input-type zmq_manager，订阅 :5556）
cd /home/karthus_chen/ycb_ws/Psi0
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

# 终端 3：enable_control 进入 PLANNER 并站稳，然后释放 :5556
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-psi/bin/activate

python scripts/replay/enable_control.py

# 终端 4：启动 ACT policy server
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-act/bin/activate

bash baselines/act/serve_act_g1_real.sh \
  --run-dir /home/karthus_chen/ycb_ws/checkpoints/ACT_200k_g1_33d_walk_to_table_and_place_apple_on_pink_plate_100 \
  --ckpt-step 40000 \
  --n-action-steps 1

# 终端 5：ACT client（接管 :5556 → PLANNER → STREAMED_MOTION → 发 token）
cd /home/karthus_chen/ycb_ws/Psi0
source third_party/GR00T-WholeBodyControl/.venv_teleop/bin/activate

python real/deploy/act_inference.py \
  --host localhost \
  --port 22085 \
  --camera-address tcp://192.168.123.164:5555 \
  --eef brainco \
  --dds-interface enp5s0 \
  --visualization \
  --ckpt-step 40000 \
  --save-pred-action /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate
```

## Psi0
### 训练
```bash
# Launch the training script via tmux
tmux new -s train_psi0
cd /sh/zzy/Psi0
export WANDB_API_KEY='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'
wandb login
bash train_psi0.sh

# Detach / reattach tmux
Detach: Ctrl-b then d
Reattach: tmux attach -t train_psi0
List sessions: tmux ls
```

### 开环推理
```bash
# 终端 1：启动 Psi0 policy server（HTTP :22085）
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-psi/bin/activate

bash scripts/deploy/serve_psi0_simple.sh \
  /home/karthus_chen/ycb_ws/checkpoints/PSI0_40k_g1_sonic_walk_to_table_and_place_apple_on_pink_plate_100 \
  40000

# 终端 2：启动开环推理端
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-psi/bin/activate

python baselines/psi0/openloop_psi0_g1_real.py \
  --data-root /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate \
  --ckpt-step 40000 \
  --episode-idx 1 \
  --n-action-steps 1 \
  --host localhost \
  --port 22085
```

### 闭环推理
```bash
# 终端 1：G1 机载 Brainco hand + SONIC composed_camera（PUB :5555）
ssh unitree@192.168.123.164
bash ./sonic_start_teleop.sh

# 终端 2：SONIC C++ deploy（--input-type zmq_manager，订阅 :5556）
cd /home/karthus_chen/ycb_ws/Psi0
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

# 终端 3：enable_control 进入 PLANNER 并站稳，然后释放 :5556
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-psi/bin/activate

python scripts/replay/enable_control.py

# 终端 4：启动 Psi0 policy server（RTC WebSocket :8014）
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-psi/bin/activate

export CHECKPOINT_DIR=/home/karthus_chen/ycb_ws/checkpoints/PSI0_40k_g1_sonic_walk_to_table_and_place_apple_on_pink_plate_100
export CHECKPOINT_STEP=40000
bash ./scripts/deploy/serve_psi0-rtc-sonic.sh

# 终端 5：Psi0 RTC client（WebSocket → 68D action → token + Brainco DDS）
cd /home/karthus_chen/ycb_ws/Psi0
source third_party/GR00T-WholeBodyControl/.venv_teleop/bin/activate

python real/deploy/psi_inference.py \
  --host localhost \
  --port 8014 \
  --camera-address tcp://192.168.123.164:5555 \
  --eef brainco \
  --dds-interface enp5s0 \
  --instruction "Go to the table, pick up the apple, place the apple on the pink plate." \
  --ckpt-step 40000 \
  --save-pred-action /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate
```

## GR00T-N1.7

### 训练
```bash
# Launch the training script via tmux
tmux new -s train_gr00t
cd /sh/zzy/Psi0
export WANDB_API_KEY='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'
wandb login
bash train_gr00t.sh

# Detach / reattach tmux
Detach: Ctrl-b then d
Reattach: tmux attach -t train_gr00t
List sessions: tmux ls
```

### 开环推理
```bash
# 终端 1：启动 GR00T-N1.7 PolicyServer（ZMQ :5555）
cd /home/karthus_chen/ycb_ws/Psi0
bash baselines/gr00t-n1.7/serve_gr00t_n1d7_sonic.sh \
  --model-path /home/karthus_chen/ycb_ws/checkpoints/GR00T_N1d7_40k_g1_sonic_walk_to_table_place_apple_on_pink_plate_100

# 终端 2：启动开环推理端
cd /home/karthus_chen/ycb_ws/Psi0
source /home/karthus_chen/ycb_ws/GR00T/.venv/bin/activate

python baselines/gr00t-n1.7/openloop_gr00t_n1d7_g1_real.py \
  --data-root /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate \
  --ckpt-step 40000 \
  --episode-idx 1 \
  --n-action-steps 1 \
  --host localhost \
  --port 5555
```

### 闭环推理
```bash
# 终端 1：G1 机载 Brainco hand + SONIC composed_camera（PUB :5555，stereo）
ssh unitree@192.168.123.164
bash ./sonic_start_teleop.sh

# 终端 2：SONIC C++ deploy（--input-type zmq_manager，订阅 :5556）
cd /home/karthus_chen/ycb_ws/Psi0
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

# 终端 3：enable_control 进入 PLANNER 并站稳，然后释放 :5556
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-psi/bin/activate
python scripts/replay/enable_control.py

# 终端 4：启动 GR00T-N1.7 PolicyServer（ZMQ :5555）
cd /home/karthus_chen/ycb_ws/Psi0
bash baselines/gr00t-n1.7/serve_gr00t_n1d7_sonic.sh \
  --model-path /home/karthus_chen/ycb_ws/checkpoints/GR00T_N1d7_40k_g1_sonic_walk_to_table_place_apple_on_pink_plate_100

# 终端 5：GR00T client（stereo 相机 → 68D → token ZMQ + Brainco DDS）
cd /home/karthus_chen/ycb_ws/Psi0
source third_party/GR00T-WholeBodyControl/.venv_teleop/bin/activate

python real/deploy/gr00t_n1d7_inference.py \
  --host localhost \
  --port 5555 \
  --camera-address tcp://192.168.123.164:5555 \
  --eef brainco \
  --dds-interface enp5s0 \
  --instruction "Go to the table, pick up the apple, place the apple on the pink plate." \
  --ckpt-step 40000 \
  --execute-horizon 20 \
  --save-pred-action /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate
```

## Pi-0.5

### 训练
```bash
# Launch the training script via tmux
tmux new -s train_pi05
cd /sh/zzy/Psi0
export WANDB_API_KEY='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'
wandb login
bash train_pi05.sh

# Detach / reattach tmux
Detach: Ctrl-b then d
Reattach: tmux attach -t train_gr00t
List sessions: tmux ls

```

### 开环推理
```bash
# 终端 1：启动 π0.5 PolicyServer（WebSocket :9000）
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-openpi/bin/activate
bash baselines/pi05/serve_pi05_sonic.sh \
  --model-path /home/karthus_chen/ycb_ws/checkpoints/PI05_40k_g1_sonic_walk_to_table_place_apple_on_pink_plate_100 \
  --ckpt-step 40000 \
  --port 9000

# 终端 2：启动开环推理端
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-openpi/bin/activate
python baselines/pi05/openloop_pi05_g1_real.py \
  --data-root /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate \
  --ckpt-step 40000 \
  --episode-idx 1 \
  --host localhost \
  --port 9000
```

### 闭环推理
```bash
# 终端 1：G1 机载 Brainco hand + SONIC composed_camera（PUB :5555）
ssh unitree@192.168.123.164
bash ./sonic_start_teleop.sh

# 终端 2：SONIC C++ deploy（--input-type zmq_manager，订阅 :5556）
cd /home/karthus_chen/ycb_ws/Psi0

bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

# 终端 3：enable_control 进入 PLANNER 并站稳，然后释放 :5556
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-psi/bin/activate

python scripts/replay/enable_control.py

# 终端 4：启动 π0.5 PolicyServer（WebSocket :9000）
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-openpi/bin/activate

bash baselines/pi05/serve_pi05_sonic.sh \
  --model-path /home/karthus_chen/ycb_ws/checkpoints/PI05_40k_g1_sonic_walk_to_table_place_apple_on_pink_plate_100 \
  --ckpt-step 40000 \
  --port 9000

# 终端 5：π0.5 client（右目相机 → 68D → token ZMQ + Brainco DDS）
cd /home/karthus_chen/ycb_ws/Psi0
source .venv-openpi/bin/activate

python real/deploy/pi05_inference.py \
  --host localhost \
  --port 9000 \
  --camera-address tcp://192.168.123.164:5555 \
  --eef brainco \
  --dds-interface enp5s0 \
  --instruction "Go to the table, pick up the apple, place the apple on the pink plate." \
  --ckpt-step 40000 \
  --save-pred-action /home/karthus_chen/ycb_ws/Psi0/eval/walk_to_table_and_place_apple_on_pink_plate \
  --execute-horizon 24
```