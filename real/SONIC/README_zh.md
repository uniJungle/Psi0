# Ψ₀ 与 SONIC — 遥操作与数据采集

我们遵循官方 [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 的设置方式。所有命令请从子模块根目录 `third_party/GR00T-WholeBodyControl` 执行。

## 设置（在工作站上）

拉取 LFS 资源，然后使用 SONIC 的安装脚本为每个使用场景安装独立环境（每个脚本会创建一个隔离的 `uv` 虚拟环境）：

```bash
git lfs pull

bash install_scripts/install_pico.sh              # .venv_teleop          — VR 遥操作
bash install_scripts/install_data_collection.sh   # .venv_data_collection — LeRobot 录制器
bash install_scripts/install_mujoco_sim.sh        # .venv_sim             — MuJoCo 仿真

python download_from_hf.py                         # SONIC 策略 + 规划器 ONNX 模型
```

关于 C++ 全身控制器和 PICO VR 硬件，请参阅 SONIC 官方文档：
- [部署构建（TensorRT + `just build`）](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)
- [VR 遥操作设置（XRoboToolkit）](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)

## 相机服务器（在机器人上）

复用机器人的 `vision` conda 环境（已在机器人 [图像服务器设置](../README.md#image-server-robot-only) 中创建，该环境已包含 `pyrealsense2`、`opencv`、`pyzmq`）；只需额外安装三个包：

```bash
conda activate vision
pip install msgpack msgpack-numpy tyro
```

将 SONIC 相机模块从工作站复制到机器人（从子模块根目录执行；G1 默认 IP `192.168.123.164`）：

```bash
ssh unitree@192.168.123.164 mkdir -p ~/SONIC_psi0_release/gear_sonic
scp gear_sonic/__init__.py gear_sonic/version.py unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
scp -r gear_sonic/camera unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
scp real/SONIC/realsense_server.py unitree@192.168.123.164:~/SONIC_psi0_release/
```

在机器人上启动服务器（保持运行）。对于 **USB 头部立体相机**（1280×480 并排模式）：

```bash
conda activate vision
cd ~/SONIC_psi0_release
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera usb_stereo --ego-view-device-id 0 \
    --port 5555
```

这会发布两个流：`ego_view_left` 和 `ego_view_right`，分辨率各为 640×480。

## 运行

编辑脚本顶部的 `ROBOT_IP` / `TASK`（录制时以 30 fps 运行以匹配相机帧率），然后执行：

```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data.sh sim   # MuJoCo 测试（无机器人/相机，无录制）
bash ./real/SONIC/scripts/collect_psi0-sonic-data.sh       # 真机 — 录制到 outputs/（LeRobot 格式）
```

按照 SONIC 的 [数据采集教程](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html) 启动遥操作并录制：校准姿态 → **A+B+X+Y** → **A+X**，然后 **左手握持 + A** 开始/停止一个 episode（**左手握持 + B** 丢弃数据）。数据将保存到 `third_party/GR00T-WholeBodyControl/outputs/`，格式为 LeRobot 格式。

不使用 tmux？可以在各自的终端中分别运行各组件：

```bash
# 仿真遥操作测试
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh sim          # MuJoCo
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy sim   # C++ 控制器（仿真）
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico         # PICO 流媒体
```

```bash
# 真机
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy       # C++ 控制器
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico         # PICO 流媒体
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter     # 数据导出器（录制）
```
