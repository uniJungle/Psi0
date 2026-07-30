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

### 数据回放
```bash
# 终端 1：启动 C++ deploy 的 sim 模式
cd ~/ycb_ws/Psi0/
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

# 终端 2：进入规划模式保持 IDLE 模式
cd ~/ycb_ws/Psi0/
source .venv-psi/bin/activate

python scripts/replay/enable_control.py

# 终端 3：发送回放
cd ~/ycb_ws/Psi0/
source .venv-psi/bin/activate

python scripts/replay/replay_real.py \
  --input_type zmq_manager \
  --dds-interface enp5s0 \
  --zmq_port 5556 \
  --eef brainco \
  --mode token \
  --episode_idx 0 \
  --data_dir /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate/openloop_act_40000/episode_000001

# 闭环回放路径
--data_dir /home/karthus_chen/ycb_ws/datasets/SONIC/walk_to_table_and_place_apple_on_pink_plate/closeloop_act_40000
```

## Psi0
### Training
```bash
# Launch the training script via tmux
tmux new -s train_psi0
cd /sh/zzy/Psi0
export WANDB_API_KEY='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'
wandb login
bash train_psi0.sh
```
