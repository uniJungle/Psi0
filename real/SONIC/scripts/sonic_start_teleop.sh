#!/bin/bash

SESSION="tv_lab"
# 设置你要选择的 ROS 版本，1 为 foxy, 2 为 noetic
# 根据你的日志，这里默认设为 1
ROS_SELECTION="1"

# 检查会话是否存在
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
  # 1. 创建新会话 (Pane 0: 上半部分)
  tmux new-session -d -s $SESSION

  # 2. 上下切分
  tmux split-window -v
  
  # 3. 下半部分水平切分 (Pane 1: 下左, Pane 2: 下右)
  #tmux split-window -h -t 1

  # ==========================================
  # Pane 0: Inspire Hand (上半部分)
  # ==========================================
  # 1. 先发送 ROS 选择
  tmux send-keys -t 0 "$ROS_SELECTION" C-m
  # 2. 等待一下 shell 加载
  sleep 0.5
  # 3. 激活环境
  #tmux send-keys -t 0 "conda activate tv" C-m
  # 4. 进入目录
  #tmux send-keys -t 0 "cd ~/inspire_hand_ws/inspire_hand_sdk/example" C-m
  # 5. 临时添加上一级目录到 python 路径 (解决 ModuleNotFoundError: inspire_sdkpy)
  #tmux send-keys -t 0 "export PYTHONPATH=\$PYTHONPATH:../" C-m
  # 6. 运行
  tmux send-keys -t 0 "sudo systemctl restart brainco_hand" C-m

  # ==========================================
  # Pane 1: G1 Ctrl (下左)
  # ==========================================
  # 1. ROS 选择
  # tmux send-keys -t 1 "$ROS_SELECTION" C-m
  # sleep 0.5
  # 2. 进入 Build 目录 (必须先进入目录，否则 patchelf 找不到文件)
  #tmux send-keys -t 1 "cd ~/deploy_no_wrist_yaw/robots/g1_29dof/build" C-m
  # 3. 执行 patchelf
  #tmux send-keys -t 1 "patchelf --set-rpath '../../../thirdparty/onnxruntime-linux-aarch64-1.22.0/lib/' g1_ctrl" C-m
  # 4. 运行
  #tmux send-keys -t 1 "./g1_ctrl" C-m

  # ==========================================
  # Pane 2: Image Server (下右)
  # ==========================================
  # 1. ROS 选择
  tmux send-keys -t 1 "$ROS_SELECTION" C-m
  sleep 0.5
  # 2. 激活环境 (解决 ModuleNotFoundError: cv2)
  tmux send-keys -t 1 "conda activate teleop" C-m
  # 3. 确保在主目录 (根据报错日志，image_server 似乎在 ~ 下)
  tmux send-keys -t 1 "cd ~" C-m
  # 4. 运行
  tmux send-keys -t 1 "python image_server/image_server.py" C-m

  # 选中第一个窗格
  tmux select-pane -t 0
fi

# 进入会话
tmux attach-session -t $SESSION