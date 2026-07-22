# SONIC Whole-body Teleoperation Guideline

## SONIC submodule

本指南依赖的 GR00T / SONIC 改动在 [uniJungle/GR00T-WholeBodyControl](https://github.com/uniJungle/GR00T-WholeBodyControl) 的 `g1_setup` 分支：

```bash
cd third_party/GR00T-WholeBodyControl
git remote add origin git@github.com:uniJungle/GR00T-WholeBodyControl.git   # 若尚未添加
git fetch origin
git checkout g1_setup
```

## Pre-processing
```bash
conda activate vision
pip install msgpack msgpack-numpy tyro
```

Copy the SONIC camera module from the workstation (run from the submodule root; G1 default IP `192.168.123.164`):

```bash
ssh unitree@192.168.123.164 mkdir -p ~/SONIC_psi0_release/gear_sonic
scp gear_sonic/__init__.py gear_sonic/version.py unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
scp -r gear_sonic/camera unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
scp real/SONIC/realsense_server.py unitree@192.168.123.164:~/SONIC_psi0_release/
```

## Launch all required scripts in order
### 1. Launch the G1 onboard image server
Start the server on the robot (keep it running). For a **USB head stereo** camera (1280×480 side-by-side):

```bash
ssh unitree@192.168.123.164
conda activate vision
cd ~/SONIC_psi0_release

# This publishes two streams: `ego_view_left` and `ego_view_right`, each 640×480.
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera usb_stereo --ego-view-device-id 0 \
    --port 5555

# Check the image stream
cd ~/ycb_ws/Psi0/third_party/GR00T-WholeBodyControl
source .venv_teleop/bin/activate

python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host 192.168.123.164 \
    --camera-port 5555
```

### 2. Launch the SONIC C++ cotroller
```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy
```

### 3. Launch the PICO teleoperation system
```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico
```

### 4. Launch the data recording/export script
```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
    --task-prompt "Pick bottle and pour into cup." \
    --task-name "test" \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC
```

### For step 2~4, run mutiple panes by tmux in a single terminal
```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data.sh \
    --task-prompt "Pick bottle and pour into cup." \
    --task-name "test" \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC
```

## Data Collection

1. **进入遥操作**（deploy 终端 / PICO）：校准姿态 → **A+B+X+Y** → **A+X**
2. **模式切换**（PICO，在已启动策略后）：
   - **A+X**：遥操（POSE）↔ 规划（PLANNER）
   - **右手 B**：遥操暂停 ↔ 恢复遥操（不要同时按 A/X/Y 或左 grip）
   - **B+Y**：进入 / 退出「上半身冻结规划」
   - **A+B+X+Y**：急停并退出策略
3. **录制 episode**（PICO）：
   - **left grip + A**：开始 / 停止录制
   - **left grip + B**：丢弃当前 episode（仍会落盘，并在 `meta/info.json` 的 `discarded_episode_indices` 中标记）
4. **数据保存路径**（默认）：
   ```
   /<root-output-dir>/<task_name>/<YYYY-MM-DD>/
   ├── data/chunk-000/episode_XXXXXX.parquet
   ├── videos/.../observation.images.ego_view_left/
   ├── videos/.../observation.images.ego_view_right/
   └── meta/{info.json, modality.json, episodes.jsonl, tasks.jsonl}
   ```


## Data Post-processing

采数完成后，按以下顺序做后处理（均在 **Psi0 仓库根目录** 执行）：

```text
原始 SONIC LeRobot 数据集
  → ① 清洗（剔除 discarded / stale SMPL 帧）
  → ② 转换为 Ψ₀ LeRobot 格式
  → ③ 重新计算 stats
  → ④ 准备 stats_psi0.json → 可开始微调
```

### 0. 设置路径变量

根据实际采数路径修改：

```bash
export PSI_HOME=/home/karthus_chen/ycb_ws/Psi0   # 或你的 Psi0 根目录
export TASK_NAME=test                             # 与 --task-name 一致
export DATASET_DATE=2026-07-22                    # 采数当天日期 YYYY-MM-DD

export RAW_DATASET=/home/karthus_chen/ycb_ws/datasets/SONIC/$TASK_NAME/$DATASET_DATE
export CLEAN_DATASET=/home/karthus_chen/ycb_ws/datasets/SONIC/$TASK_NAME/${DATASET_DATE}_cleaned
```

### 1. 清洗原始数据集（推荐）

`process_dataset.py` 会：
- 删除采集时按 **left grip + B** 标记的 discarded episode（`discarded_episode_indices`）
- 删除 stale SMPL 帧（全零 `teleop.smpl_pose` 及之前的冻结前导帧）

```bash
cd $PSI_HOME/third_party/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate

python gear_sonic/scripts/process_dataset.py \
    --dataset-path $RAW_DATASET \
    --output-path $CLEAN_DATASET
```

可选参数：
| 参数 | 默认 | 说明 |
|------|------|------|
| `--remove-discarded` | `True` | 剔除 discarded episode |
| `--no-remove-discarded` | — | 保留 discarded episode |
| `--remove-stale-smpl` | `True` | 剔除 stale SMPL 帧 |
| `--no-remove-stale-smpl` | — | 仅合并/整理，不做 SMPL 清洗 |

> **注意**：此步骤会更新 `parquet` / `video` / `info.json` / `episodes.jsonl`，但**不会**重算 LeRobot 的 `meta/stats.json`。训练用 stats 在步骤 ③ 中重新计算。

若跳过此步直接转换，`raw_sonic_to_psi_lerobot.py` **不会**读取 `discarded_episode_indices`，被 discard 的 episode 可能进入训练集。

### 2. 转换为 Ψ₀ LeRobot 格式

```bash
cd $PSI_HOME

python scripts/data/raw_sonic_to_psi_lerobot.py \
    --data-root=$CLEAN_DATASET \
    --work-dir=$PSI_HOME/data/sonic/lerobot \
    --repo-id=$TASK_NAME \
    --robot-type=g1
```

输出目录：`$PSI_HOME/data/sonic/lerobot/$TASK_NAME/`

> **注意**：当前转换脚本期望单目视频 key `observation.images.ego_view`。若采数时使用了 `--record-stereo-ego`（`ego_view_left` / `ego_view_right`），需先改转换脚本或调整采数配置，否则会报找不到视频文件。

### 3. 重新计算 stats

```bash
python scripts/data/calc_modality_stats.py \
    --work-dir=$PSI_HOME/data/sonic/lerobot \
    --task=$TASK_NAME
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

## Keyboard：Normal / Planner mode（deploy 终端）

```bash
cd third_party/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type keyboard real   # sim: ./deploy.sh --input-type keyboard sim
```

| Key | Action |
| --- | --- |
| `]` | 启动控制 |
| `ENTER` | 切换 Normal ↔ Planner |
| `O` | 急停并退出 |

**Normal mode**（默认，参考动作回放）
- `T`：播放当前动作；`R`：重置到第 0 帧
- `N` / `P`：下一个 / 上一个动作序列
- `Q` / `E`：航向微调

**Planner mode**（`ENTER` 进入）
- `W`/`S`：前进 / 后退；`A`/`D`：转向并前进；`,`/`.`：侧移
- `Q`/`E`：原地转向；`9`/`0`：减速 / 加速；`-`/`=`：蹲姿高度
- `N`/`P`：切换 motion set；`1`–`8`：选择 style
- `R` / backtick / `~`：立即清零动量急停