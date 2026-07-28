# Psi-0 ACT 训练

## 启动（tmux）

```bash
tmux new -s train_act
cd /sh/zzy/Psi0
export WANDB_API_KEY='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'
wandb login
bash train_act.sh
```

## tmux

```bash
# detach
Ctrl-b d

# reattach
tmux attach -t train_act

# list
tmux ls

# kill session
tmux kill-session -t train_act
```

## 停止训练

```bash
pkill -KILL -f 'real_act_config' || true
```
