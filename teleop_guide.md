# SONIC Whole-body Teleoperation Guideline

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
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter

# 可选：覆盖任务提示词 / 保存路径
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
    --task-prompt "Pick bottle and pour into cup." \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC
```

### For step 2~4, run mutiple panes by tmux in a single terminal
```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data.sh
```

## Data collection

1. **进入遥操作**（deploy 终端 / PICO）：校准姿态 → **A+B+X+Y** → **A+X**
2. **录制 episode**（PICO）：
   - **left grip + A**：开始 / 停止录制
   - **left grip + B**：丢弃当前 episode（不保存）
3. **数据保存路径**（默认）：
   ```
   /home/karthus_chen/ycb_ws/datasets/SONIC/<task_name>/<YYYY-MM-DD>/
   ├── data/chunk-000/episode_XXXXXX.parquet
   ├── videos/.../observation.images.ego_view_left/
   ├── videos/.../observation.images.ego_view_right/
   └── meta/{info.json, modality.json, episodes.jsonl, tasks.jsonl}
   ```
   *注：如果在同一天内多次启动录制，系统会自动读取该路径下已有的数据集，并接着最新的 `episode_id` 继续追加录制，不会覆盖之前的数据。*

4. **任务提示词 / 保存路径**：启动 exporter（或一键脚本）时可覆盖：

```bash
# 分终端：只改 exporter
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
    --task-prompt "Pick bottle and pour into cup." \
    --task-name "pick_bottle" \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC

# 一键 tmux：同样支持
bash ./real/SONIC/scripts/collect_psi0-sonic-data.sh \
    --task-prompt "Pick bottle and pour into cup." \
    --task-name "test" \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC
```

可选参数：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task-prompt` | `Pick bottle and turn and pour into cup.` | 写入 `meta/tasks.jsonl` 的语言任务描述 |
| `--task-name` | `pick_bottle` | 任务的简写名称，用于创建具体任务的子目录 |
| `--root-output-dir` | `/home/karthus_chen/ycb_ws/datasets/SONIC` | LeRobot 数据集根目录（其下按 `task_name/YYYY-MM-DD` 建子目录） |
