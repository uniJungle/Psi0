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
  # Pane 0: Brainco Hand (上半部分)
  # ==========================================
  # 1. 先发送 ROS 选择
  tmux send-keys -t 0 "$ROS_SELECTION" C-m
  # 2. 等待一下 shell 加载
  sleep 0.5
  # 3. 运行
  tmux send-keys -t 0 "sudo systemctl restart brainco_hand" C-m

  # ==========================================
  # Pane 1: SONIC Image Server (下半部分)
  # ==========================================
  # 1. ROS 选择
  tmux send-keys -t 1 "$ROS_SELECTION" C-m
  sleep 0.5
  # 2. 激活环境
  tmux send-keys -t 1 "conda activate vision" C-m
  # 3. 进入 SONIC 发布目录
  tmux send-keys -t 1 "cd ~/SONIC_psi0_release" C-m
  # 4. 发布双目流：ego_view_left / ego_view_right，各 640×480
  tmux send-keys -t 1 "python -m gear_sonic.camera.composed_camera --ego-view-camera usb_stereo --ego-view-device-id 0 --port 5555" C-m

  # 选中第一个窗格
  tmux select-pane -t 0
fi

# 进入会话
tmux attach-session -t $SESSION
