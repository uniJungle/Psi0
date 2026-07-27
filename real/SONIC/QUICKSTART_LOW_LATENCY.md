# Psi0 + SONIC Low-Latency 真机数采 Quick Start

本文档适用于：

- Unitree G1 真机；
- PICO 全身遥操作；
- Brainco 双手；
- 头部双目相机；
- SONIC low-latency 模型（SMPL 4-frame lookahead）；
- LeRobot episode 数据采集。

工作站上的命令均从当前自己的 Psi0 工程根目录执行：

```bash
cd /home/ubuntu24/work/Psi0
```

所有命令都只使用当前 Psi0。启动脚本会通过
`third_party/GR00T-WholeBodyControl` 自动进入本工程使用的 SONIC。

## 1. 启动前检查

### 1.1 确认当前 SONIC

```bash
cd /home/ubuntu24/work/Psi0
readlink -f third_party/GR00T-WholeBodyControl
```

确认输出指向本次准备使用的 SONIC 工作树。

### 1.2 确认 low-latency 文件

```bash
cd /home/ubuntu24/work/Psi0

ls -lh \
    third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_decoder.onnx \
    third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_encoder.onnx \
    third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/observation_config.yaml
```

三个文件必须全部存在。`policy/low_latency/` 目前可能是未跟踪目录，不要运行
`git clean -fd`，否则模型文件可能被删除。

### 1.3 确认 4-frame observation

```bash
cd /home/ubuntu24/work/Psi0

rg -n \
    "smpl_joints_4frame|smpl_anchor_orientation_4frame|motion_joint_positions_wrists_4frame" \
    third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
```

必须找到：

```text
smpl_joints_4frame_step1
smpl_anchor_orientation_4frame_step1
motion_joint_positions_wrists_4frame_step1
```

### 1.4 检查机器人网络

```bash
ip -br -4 addr show enx6c1ff7c12485
ping -c 3 192.168.123.164
```

预期看到工作站地址 `192.168.123.100/24`。如果网卡名称发生变化，请在后续
deploy、PICO 和 exporter 命令中统一替换。

## 2. 四终端启动

严格按终端 1 → 2 → 3 → 4 的顺序启动。

### 终端 1：机器人图像与 Brainco 服务

```bash
ssh unitree@192.168.123.164

bash ./sonic_start_teleop.sh
```

保持终端运行，确认 Brainco 和双目相机服务没有持续报错。

### 终端 2：low-latency SONIC deploy

```bash
cd /home/ubuntu24/work/Psi0

bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy \
    enx6c1ff7c12485 \
    --low-latency
```

脚本会自动选择：

```text
policy/low_latency/model_decoder.onnx
policy/low_latency/model_encoder.onnx
policy/low_latency/observation_config.yaml
```

出现以下提示时，输入终端键盘的 `Y`，不是按 PICO 手柄 Y：

```text
Proceed with deployment? [Y/n]:
```

第一次启动时，TensorRT 会校验 ONNX、GPU 型号和精度的 hash。如果现有 `.trt`
缓存不匹配，会自动重新生成，期间不要退出终端。

启动日志必须包含：

```text
[deploy] SONIC model=low-latency (SMPL 4-frame lookahead)
Decoder Model: policy/low_latency/model_decoder.onnx
Encoder Model: policy/low_latency/model_encoder.onnx
Obs Config: policy/low_latency/observation_config.yaml
Input Type: zmq_manager
```

模型初始化时重点确认：

```text
Policy input dimension: 994
Encoder input dimension: 1247
Token dimension: 64
```

不能出现：

```text
Unknown observation
dimension mismatch
Failed to initialize encoder
Failed to initialize policy
```

### 终端 3：PICO + Brainco

```bash
cd /home/ubuntu24/work/Psi0

bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico \
    --dds-interface enx6c1ff7c12485 \
    --eef brainco
```

确认日志包含：

```text
[pico] eef=brainco dds-interface=enx6c1ff7c12485
Manager controls: A+X=toggle mode
Brainco hands: left/right trigger = open/close
```

等待 PICO 身体追踪稳定后再操作机器人。

### 终端 4：双目数据 exporter

每个任务建议使用独立的 `--task-name`。相同 task name 会续写同一个数据集。

```bash
cd /home/ubuntu24/work/Psi0

bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
    --root-output-dir /home/ubuntu24/work/Psi0/outputs/SONIC \
    --task-prompt "Pick bottle and pour into cup." \
    --task-name "pick_bottle_pour_low_latency" \
    --use-stereo-camera \
    --eef brainco \
    --dds-interface enx6c1ff7c12485
```

确认日志包含：

```text
[exporter] camera=stereo
[Sonic] Subscribed to: pose, planner (data) + manager_state (control)
Recording to .../pick_bottle_pour_low_latency
```

## 3. PICO 手柄操作

### 3.1 进入遥操作

1. 操作者保持标定零位姿态。
2. 按一次 `A+B+X+Y`，启动策略并完成姿态校准。
3. 等机器人在 PLANNER 模式稳定站立。
4. 按一次 `A+X`，切换到 POSE 遥操作。
5. 先进行小幅手臂动作，确认方向、延迟和稳定性。

### 3.2 模式与安全

- `A+X`：POSE 遥操作与 PLANNER 规划模式切换；
- 单按 `Y`：暂停遥操作；
- 再按一次 `Y`：恢复遥操作；
- `A+B+X+Y`：急停并退出当前 PICO manager；
- PLANNER 下左摇杆控制平移，右摇杆控制朝向；
- 左 trigger 控制左 Brainco 手，右 trigger 控制右 Brainco 手。

正常暂停时，终端 3 应打印：

```text
[Manager] StreamMode switch: POSE -> POSE_PAUSE
```

恢复时应打印：

```text
[Manager] StreamMode switch: POSE_PAUSE -> POSE
```

暂停期间机器人保持最后遥操作目标，Brainco 手保持最后目标，exporter 不写入
episode 帧。

如果终端 2 打印：

```text
[ZMQManager] Stream timeout — publisher disconnected, switching to PLANNER + IDLE
```

这不是正常暂停，表示 PICO manager 或 heartbeat 已断开。立即停止动作并检查终端 3。

### 3.3 录制 episode

- `left gripper + A`：按一次开始录制；
- 再按一次 `left gripper + A`：停止并保存；
- `left gripper + B`：丢弃当前 episode，只在已经开始录制后使用。

推荐单条 episode 流程：

1. 进入 POSE；
2. 摆好初始姿态；
3. `left gripper + A` 开始；
4. 完成任务；
5. 必要时单按 `Y` 暂停，再按一次恢复；
6. `left gripper + A` 停止；
7. 等待 exporter 明确打印保存完成；
8. 机器人复位后再开始下一条。

不要在录制过程中反复点击组合键。

## 4. 正常停止

1. 如果正在录制，先用 `left gripper + A` 停止。
2. 等待终端 4 打印 episode 保存完成。
3. 按 `A+B+X+Y` 退出策略。
4. 终端 3 的 PICO manager 退出后，下次使用需要重新启动终端 3。
5. 在终端 4 按 `Ctrl+C` 停止 exporter。
6. 最后停止终端 2 的 deploy。

紧急情况下直接按 `A+B+X+Y`，但未结束的 episode 可能不会正常保存。

## 5. 常见问题

### 启动后显示 release 模型

如果终端 2 显示：

```text
[deploy] SONIC model=release (default)
```

说明漏写了 `--low-latency`。停止 deploy 后按本文命令重新启动。

### 找不到 4-frame observation

如果出现：

```text
smpl_joints_4frame_step1 not found
```

说明运行的不是当前源码编译出的 deploy。确认 `readlink -f` 的结果，并重新运行终端 2；
`deploy.sh` 会自动重新构建 C++。

### TensorRT 缓存重新生成

不同 GPU、ONNX 或精度对应不同 hash。自动重新生成属于正常行为，完成后后续启动会复用缓存。

### Y 暂停时仍在写数据

确认终端 4 收到了：

```text
[Mode] stream_mode 1 -> 4
```

如果没有，检查 PICO manager 与 exporter 是否连接到同一个 SONIC ZMQ `5556`。

### 数据保存位置

本文示例保存到：

```text
/home/ubuntu24/work/Psi0/outputs/SONIC/pick_bottle_pour_low_latency/
```

修改 `--task-name` 可为不同任务建立独立数据集。
