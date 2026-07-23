# Ψ₀ 与 SONIC — 真机部署指南

请按顺序运行以下命令。

## 1. 在机器人上启动相机服务器（保持运行）：

```bash
conda activate vision
cd ~/SONIC_psi0_release
python realsense_server.py
```

## 2. 在工作站上启动策略服务器（保持运行）：

```bash
bash ./scripts/deploy/serve_psi0-rtc-sonic.sh
```

## 3. 在机器人上启动全身控制器（保持运行）：

```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-robot.sh
```

当看到 "Init done." 时，可以按 **]** 按钮让机器人站立。之后按 **ENTER** 键开始部署策略。部署完成后，再次按 **ENTER** 键停止策略并让机器人恢复默认姿态。

## 4. 在工作站上启动策略客户端：

```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-client.sh
```