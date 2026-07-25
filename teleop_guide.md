# SONIC 全身遥操作指南

## SONIC 子模块

本指南依赖的 GR00T / SONIC 改动在 [uniJungle/GR00T-WholeBodyControl](https://github.com/uniJungle/GR00T-WholeBodyControl) 的 `g1_setup` 分支：

```bash
cd third_party/GR00T-WholeBodyControl
git remote add origin git@github.com:uniJungle/GR00T-WholeBodyControl.git   # 若尚未添加
git fetch origin
git checkout g1_setup
```

Brainco 灵巧手在 SONIC 数采中的安装、通信机制与依赖说明见：[real/SONIC/BRAINCO_HAND.md](real/SONIC/BRAINCO_HAND.md)。

## 预处理

1. 在机器人上创建相机所需的 conda 环境（G1 默认 IP `192.168.123.164`）：

    ```bash
    ssh unitree@192.168.123.164
    conda activate vision
    pip install msgpack msgpack-numpy tyro
    ```

2. 从工作站拷贝 SONIC 相机模块到机器人（在 submodule 根目录执行）：

    ```bash
    ssh unitree@192.168.123.164 mkdir -p ~/SONIC_psi0_release/gear_sonic
    scp gear_sonic/__init__.py gear_sonic/version.py unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
    scp -r gear_sonic/camera unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
    scp real/SONIC/realsense_server.py unitree@192.168.123.164:~/SONIC_psi0_release/
    ```

## 按顺序启动所需脚本

### 1. 启动 G1 机载图像服务与 Brainco 手部服务

1. 通过 tmux 一并启动两个服务：

    ```bash
    ssh unitree@192.168.123.164
    bash ./sonic_start_teleop.sh
    ```

2. 仅启动图像服务：

    ```bash
    ssh unitree@192.168.123.164
    conda activate vision
    cd ~/SONIC_psi0_release

    # 发布双目流：`ego_view_left` / `ego_view_right`，各 640×480
    python -m gear_sonic.camera.composed_camera \
        --ego-view-camera usb_stereo --ego-view-device-id 0 \
        --port 5555

    # 发布单目流：`ego_view`，640×480
    python -m gear_sonic.camera.composed_camera \
        --ego-view-camera usb --ego-view-device-id 0 \
        --port 5555

    # 在工作站检查相机画面
    cd ~/ycb_ws/Psi0/third_party/GR00T-WholeBodyControl
    source .venv_teleop/bin/activate

    python gear_sonic/scripts/run_camera_viewer.py \
        --camera-host 192.168.123.164 \
        --camera-port 5555
    ```

3. 仅启动 Brainco 手部服务：

    ```bash
    ssh unitree@192.168.123.164

    sudo systemctl start brainco_hand && systemctl is-active brainco_hand
    ```

### 2. 启动 SONIC C++ 控制器

```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy
```

### 3. 启动 PICO 遥操作系统

```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico \
    --eef brainco \
    --dds-interface enp4s0
```

安装、DDS 通信与依赖见 [BRAINCO_HAND.md](real/SONIC/BRAINCO_HAND.md)。

### 4. 启动数据录制 / 导出脚本

```bash
# 双目：ego_view_left / ego_view_right
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
    --task-prompt "Pick bottle and pour into cup." \
    --task-name "pico_kill_record_test" \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC/test/tts_test \
    --use-stereo-camera \
    --eef brainco

# 单目：ego_view 选择 --use-mono-camera
```

## 数据采集

1. **进入遥操作**（deploy 终端 / PICO）：校准姿态 → **A+B+X+Y** → **A+X**
2. **模式切换**（PICO，在已启动策略后）：
    - **A+X**：遥操（POSE）↔ 规划（PLANNER）
    - **Y**：遥操暂停 ↔ 恢复遥操
    - **A+B+X+Y**：急停并退出策略
3. **PICO 断连安全**（头显串流中断，或 **Ctrl+C 退出 pico 进程**）：
    - pico 退出前会向 deploy 发 **PLANNER + IDLE** handoff，并播报「PICO断开，进入待机模式」
    - deploy 侧：若遥操 pose 流 **>1s** 无数据，也会自动切 **PLANNER + IDLE**（不冻结最后一帧）
    - **deploy 改动需重新编译**：改完 `zmq_manager.hpp` 后重启 `collect_psi0-sonic-data-manual.sh deploy`
    - 串流/进程恢复后可正常 **A+X** 切换模式
4. **规划模式移动**（进入 PLANNER 后）：
    - 默认即锁定 **SLOW_WALK（慢走）**：左摇杆可直接平移，右摇杆左右控制朝向
    - 松杆时下发 IDLE，机器人站定
5. **双手开合**（Brainco，PICO；manager 启动后即可用）：
    - **左 trigger**：左手开合
    - **右 trigger**：右手开合
    - 终端应周期性打印 `[Brainco] trigger L=.. R=..`；若有打印但手不动，查机器人 `brainco_hand` / DDS 网卡
6. **录制 episode**（PICO）：
    - **left grip + A**：开始 / 停止录制
    - **left grip + B**：丢弃当前 episode（仍会落盘，并在 `meta/info.json` 的 `discarded_episode_indices` 中标记）
7. **数据格式**（Lerobot v2.1）：

    ```text
    /<root-output-dir>/<task_name>/<YYYY-MM-DD>/
    ├── data/chunk-000/episode_XXXXXX.parquet
    ├── videos/.../observation.images.ego_view_left/
    ├── videos/.../observation.images.ego_view_right/
    └── meta/{info.json, modality.json, episodes.jsonl, tasks.jsonl}
    ```

## 真机回放数据

使用采数同款控制器（`--input-type zmq_manager`）。回放脚本会 **PUB bind** `tcp://*:5556`，C++ 默认 `--zmq-host localhost` SUB connect；两端都在工作站跑。

### 1. 启动 SONIC C++ 控制器
```bash
# 等到 Init Done / [ZMQManager] Host: localhost:5556
cd ~/ycb_ws/Psi0/
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy
```

### 进入规划模式保持站立

```bash
# 站稳后自动退出并释放 5556
# 默认：start+planner → idle 约 3s → 释放 PUB（不发 stop）→ 进程退出
cd ~/ycb_ws/Psi0/
source .venv-psi/bin/activate

python scripts/replay/enable_control.py
```

### 真机回放（必须 bind 5556）
```bash
cd ~/ycb_ws/Psi0/
source .venv-psi/bin/activate

python scripts/replay/replay_real.py \
  --input_type zmq_manager \
  --dds-interface enp4s0 \
  --zmq_port 5556 \
  --eef brainco \
  --mode token \
  --data_dir /home/karthus_chen/ycb_ws/datasets/SONIC/test/tts_test \
  --episode_idx 1
```

说明：

- `enable_control.py` 与 `replay_real.py` **不能同时** bind `tcp://*:5556`。
- `enable_control` 默认 **handoff**：站稳后退出并释放端口（不发 `stop`）；不要用旧版一直发 idle planner 的方式（会把回放顶回规划）。
- 若需要长时间保持站立再手动开回放：`python scripts/replay/enable_control.py --hold`，**先 Ctrl+C 释放端口**，再跑 `replay_real.py`。
- 回放默认端口为 `5556`；若指定 `--zmq_port 5559`，deploy（听 5556）收不到指令。
- 回放结束后脚本会**切回 idle PLANNER 并释放 5556**（不发 `stop`），**deploy 继续运行**。
- 不要边开 pico 边回放（pico 也会 bind 5556，且会抢 Brainco DDS）。
- `replay_real.py` 会先 `planner=True` 进 CONTROL，再切 STREAMED_MOTION 发 token。
- **Brainco 手不经过 deploy**：数据集有 `teleop.*_hand_joints`（2D）时须加 `--eef brainco`，并保证机器人 `brainco_hand` 服务已起；仅 token 回放时手不会动。
- 默认会弹出 OpenCV 窗口 `Replay Video`，按帧同步播放该 episode 的左右眼视频；不需要时可加 `--no-video`。

成功时 deploy 端应依次出现：

- `[ZMQManager] Planner enabled` / `motion name is planner_motion`
- `[Control] ... transitioning to CONTROL state`
- `[ZMQManager] Switched to: STREAMED MOTION` / `ZMQ STREAMING MODE: ENABLED`
- `[ZMQEndpointInterface] Protocol v4: Received 64D token ...`

## 数据后处理

采数完成后，按以下顺序做后处理（均在 **Psi0 仓库根目录** 执行）：

```text
原始 SONIC LeRobot 数据集
  → ① 清洗（剔除 discarded episode；可选剔除无效 SMPL 帧）
  → ② 转换为 Ψ₀ LeRobot 格式
  → ③ 重新计算 stats
  → ④ 准备 stats_psi0.json → 可开始微调
```

### 1. 清洗原始数据集

`process_dataset.py` **不会删掉** `teleop.smpl_*` 等字段，也不会改 `modality.json` / feature schema。它只做：

1. 按 `meta/info.json` 的 `discarded_episode_indices` **整段删除** discarded episode（parquet + 视频），并去掉该字段
2. （默认开启）删除 **无效帧**：`teleop.smpl_pose` 全零的 stale 帧，及其前的冻结前导帧；有效帧上的 SMPL 原样保留

务必传 `--output-path`，否则会 **覆盖** 原始目录。

```bash
cd /home/karthus_chen/ycb_ws/Psi0/third_party/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate

python gear_sonic/scripts/process_dataset.py \
    --dataset-path /home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-22/origin \
    --output-path /home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-22/clean \
    --remove-discarded \
    --remove-stale-smpl
```

详细处理逻辑与全部参数说明见 [`process_dataset.py`](third_party/GR00T-WholeBodyControl/gear_sonic/scripts/process_dataset.py)。


### 2. 转换为 Ψ₀ LeRobot 格式

```bash
cd /home/karthus_chen/ycb_ws/Psi0/
source .venv-psi/bin/activate

# 单目：ego_view → egocentric
python scripts/data/raw_sonic_to_psi_lerobot.py \
    --data-root=/home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-22/clean \
    --work-dir=/home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-22/ \
    --repo-id=lerobot_v2.1 \
    --robot-type=g1 \
    --use-mono-camera \
    --eef brainco

# 双目：ego_view_left/right → egocentric_left/right
python scripts/data/raw_sonic_to_psi_lerobot.py \
    --data-root=/home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-22/clean \
    --work-dir=/home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-22/ \
    --repo-id=lerobot_v2.1 \
    --robot-type=g1 \
    --use-stereo-camera \
    --eef brainco
```

### 3. 重新计算 stats

```bash
python scripts/data/calc_modality_stats.py \
    --work-dir=/home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-22/ \
    --task=lerobot_v2.1
```

### 4. 生成 Ψ₀ 训练用 stats 副本

```bash
cp $PSI_HOME/data/sonic/lerobot/$TASK_NAME/meta/stats.json \
   $PSI_HOME/data/sonic/lerobot/$TASK_NAME/meta/stats_psi0.json
```

### 5. 开始微调

```bash
bash ./scripts/train/psi0/finetune-real-sonic-psi0.sh $TASK_NAME
```

更多训练/部署说明见 [Psi0 README — Ψ₀ with SONIC](README.md#psi0-sonic)。


## 键盘：Normal / Planner 模式（deploy 终端）

```bash
cd third_party/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type keyboard real   # 仿真：./deploy.sh --input-type keyboard sim
```

| 按键 | 作用 |
| --- | --- |
| `]` | 启动控制 |
| `ENTER` | 切换 Normal ↔ Planner |
| `O` | 急停并退出 |

1. **Normal 模式**（默认，参考动作回放）
    - `T`：播放当前动作；`R`：重置到第 0 帧
    - `N` / `P`：下一个 / 上一个动作序列
    - `Q` / `E`：航向微调
2. **Planner 模式**（`ENTER` 进入）
    - `W`/`S`：前进 / 后退；`A`/`D`：转向并前进；`,`/`.`：侧移
    - `Q`/`E`：原地转向；`9`/`0`：减速 / 加速；`-`/`=`：蹲姿高度
    - `N`/`P`：切换 motion set；`1`–`8`：选择 style
    - `R` / backtick / `~`：立即清零动量急停
