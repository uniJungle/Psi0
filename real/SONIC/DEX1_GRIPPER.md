# SONIC 数采：Dex1 夹爪接入说明

本文说明在 Psi0 + SONIC 全身遥操作 / 数采流程中，如何安装与启用 Unitree Dex1 双夹爪控制。  
主流程仍见 [teleop_guide.md](../../teleop_guide.md)。Brainco 双手见 [BRAINCO_HAND.md](BRAINCO_HAND.md)。

## 通信机制

Dex1 **不经过** C++ `deploy` 的手部通路，而是由 PICO 进程直接经 Unitree DDS 控夹爪：

```text
PICO 左右 trigger  [0,1]
    → pico_manager_thread_server.py
    → 映射 cmd = (1 - trigger) * 5.5          # 松扳机≈张开(5.5)，扣满≈闭合(0)
    → eef.dex1.Dex1.set_gripper_ratios(l, r)
    → DDS 发布  rt/dex1/{left,right}/cmd      # MotorCmds_
    → 机器人 dex1_gripper.service
    → 真机左右夹爪

机器人同时回传：
    rt/dex1/{left,right}/state  ← Dex1 驱动订阅（就绪检测 / 读状态）
```

与 Brainco 的差异：

| 项目 | Dex1 | Brainco |
|------|------|---------|
| 每手维度 | **1**（夹爪开合） | **2**（`[thumb_aux, others]`） |
| 指令范围 | 约 **0–5.5**（反扳机映射） | **0–1**（0=开，1=合） |
| DDS topic | `rt/dex1/{left,right}/{cmd,state}` | `rt/brainco/{left,right}/{cmd,state}` |
| 机器人服务 | `dex1_gripper` | `brainco_hand` |
| 驱动 API | `set_gripper_ratios(l, r)` | `set_gripper_targets(l, r)` |

要点：

| 对应脚本 | 虚拟环境 | 说明 |
|----------|----------|------|
| `pico_manager_thread_server.py` | 工作站 `.venv_teleop` | 读 PICO trigger，经 `eef.dex1` 发 DDS；`--eef dex1 --dds-interface`（常用 `enp5s0`） |
| `dex1_gripper.service` | 机器人侧 systemd | DDS ↔ 夹爪硬件桥；停掉后夹爪不动 |
| `./deploy.sh` | `gear_sonic_deploy` | 全身 WBC；夹爪指令不走 C++ |
| `run_data_exporter.py` | `.venv_data_collection` | `--eef dex1` 时写入 1D `teleop.*_hand_joints` |
| `scripts/replay/replay_real.py` | Psi0 `.venv-psi` | `--eef dex1` 回放时同样经 DDS 控夹爪 |

## 相关路径（worktree）

| 路径 | 说明 |
|------|------|
| `Psi0/` | Psi0 主仓库 |
| `Psi0/third_party/GR00T-WholeBodyControl/` | SONIC / GR00T 工作树（建议 `g1_setup`） |
| `.../eef/dex1/` | Dex1 Python 驱动（`Dex1` / `Dex1_1_Gripper_Controller`） |
| `.../gear_sonic/scripts/pico_manager_thread_server.py` | PICO manager，`--eef dex1` 接入点 |
| `Psi0/real/SONIC/scripts/collect_psi0-sonic-data-manual.sh` | 数采启动封装（支持 `--eef dex1`） |
| `Psi0/real/SONIC/scripts/sonic_start_teleop_dex1.sh` | 机载：重启 `dex1_gripper` + 四路相机 |
| `Psi0/scripts/replay/replay_real.py` | 真机回放（含 Dex1 DDS） |

参考实现来源：`wbc_pico_record/eef/dex1/`。

## Psi / SONIC 环境新增依赖

在 **工作站** teleop 环境（`.venv_teleop`）中：

| 依赖 | 用途 |
|------|------|
| `eef` 包（`third_party/GR00T-WholeBodyControl/eef`） | `from eef.dex1.dex1 import Dex1` |
| `unitree_sdk2py`（`external_dependencies/unitree_sdk2_python`） | DDS `MotorCmds_` / `MotorStates_` |
| CycloneDDS | 与机器人同一 DDS 域通信 |
| `logging-mp==0.1.6` | Dex1 控制器日志（需 **0.1.x** API：`basic_config` / `get_logger`；0.2+ 不兼容） |
| `rich` | `logging-mp` 依赖 |

`gear_sonic/pyproject.toml` 已将包发现扩展为 `gear_sonic*` + `eef*`，安装 teleop extra 后可从仓库根导入 `eef`。

机器人侧依赖已有 systemd 服务即可，**不必**在 Psi0 的 venv 里再装夹爪驱动包：

```bash
ssh unitree@192.168.123.164
sudo systemctl status dex1_gripper
# 需要时：
sudo systemctl start dex1_gripper
# 或：
sudo systemctl restart dex1_gripper
```

## 安装命令（工作站）

在 GR00T / SONIC 根目录执行（会创建或刷新 `.venv_teleop`）：

```bash
cd third_party/GR00T-WholeBodyControl
bash install_scripts/install_pico.sh
source .venv_teleop/bin/activate
python -c "from eef.dex1.dex1 import Dex1; print('OK')"
```

若环境已存在，确保 editable 安装包含 `eef`、`unitree_sdk2_python`，以及兼容的 `logging-mp`：

```bash
cd third_party/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
uv pip install -e external_dependencies/unitree_sdk2_python
uv pip install -e "gear_sonic[teleop]"
uv pip install 'logging-mp==0.1.6' rich
```

冒烟导入失败 `No module named 'logging_mp'` 或 `basic_config` 报错时，检查是否装成了 0.2.x（API 变为 `basicConfig` / `getLogger`）。

## 启动命令

### 机器人（夹爪服务 + 相机）

推荐用机载脚本（tmux：上栏重启夹爪服务，下栏发四路图）：

```bash
# 工作站先拷贝一次
scp real/SONIC/scripts/sonic_start_teleop_dex1.sh unitree@192.168.123.164:~/

ssh unitree@192.168.123.164
bash ./sonic_start_teleop_dex1.sh
```

仅确认 / 重启夹爪服务：

```bash
ssh unitree@192.168.123.164 \
  'sudo systemctl restart dex1_gripper && systemctl is-active dex1_gripper'
```

### 工作站（PICO）

```bash
cd ~/ycb_ws/Psi0
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico \
    --eef dex1 \
    --dds-interface enp5s0
```

关闭手控：

```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico --eef none
```

### 数采 exporter（1D 手部写入 LeRobot）

```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
    --task-prompt "This is a test" \
    --task-name "test_dex1" \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/sonic \
    --use-stereo-camera \
    --use-wrist-cameras \
    --dds-interface enp5s0 \
    --eef dex1
```

落盘字段：`teleop.left_hand_joints` / `teleop.right_hand_joints` 各 **1** 维（指令约 0–5.5）。

### 真机回放

回放时不要同时跑 pico。需机器人 `dex1_gripper` active，工作站：

```bash
cd ~/ycb_ws/Psi0
source .venv-psi/bin/activate
python scripts/replay/replay_real.py \
  --input_type zmq_manager \
  --dds-interface enp5s0 \
  --zmq_port 5556 \
  --eef dex1 \
  --mode token \
  --data_dir /home/karthus_chen/ycb_ws/datasets/sonic/test_dex1 \
  --episode_idx 0
```

`--eef` 需显式指定：`dex1` / `brainco` / `none`（默认 `none`）。

## 操作与排障

- **左右 trigger**：分别控制左右夹爪；终端应周期性打印 `[Dex1] cmd L=.. R=..`
- 有日志但夹爪不动 → 查 `dex1_gripper` 是否 active、`--dds-interface` 是否为连机器人的网卡、topic 是否为 `rt/dex1/*/cmd`
- `ImportError: Dex1 driver not available` → 多为缺 `logging_mp` 或版本不对，见上文依赖
- 回放夹爪不动 → 确认未与 pico 抢 DDS，且 `--eef dex1` / `--dds-interface` 与采数一致
