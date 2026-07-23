# SONIC 数采：Brainco 灵巧手接入说明

本文说明在 Psi0 + SONIC 全身遥操作 / 数采流程中，如何安装与启用 Brainco 双手开合控制。  
主流程仍见 [teleop_guide.md](../../teleop_guide.md)。

## 通信机制

Brainco **不经过** C++ `deploy` 的 Dex3 通路，而是由 PICO 进程直接经 Unitree DDS 控手：

```text
PICO 左右 trigger
    → pico_manager_thread_server.py
    → eef.brainco.Brainco.set_gripper_targets(l, r)   # 0=张开, 1=闭合
    → DDS 发布  rt/brainco/{left,right}/cmd
    → 机器人 brainco_hand.service (brainco_hand_server)
    → /dev/ttyUSB* 串口 → 真机手指

机器人同时回传：
    rt/brainco/{left,right}/state  ← Brainco 驱动订阅（就绪检测 / 可选读状态）
```

要点：

| 对应脚本 | 虚拟环境 | 说明 |
|----------|----------|------|
| `pico_manager_thread_server.py`| 工作站 `.venv_teleop` | 读 PICO 左右 trigger，经 `eef.brainco` 发 DDS `rt/brainco/*/cmd`；`--dds-interface` 默认 `enp4s0` |
| `brainco_hand.service`（`brainco_hand_server`）| 机器人侧 systemd | DDS ↔ `/dev/ttyUSB*` 串口桥；停掉后手不动 |
| `./deploy.sh`| `gear_sonic_deploy` 本地环境 | 全身 WBC；本阶段不写 Brainco 到 LeRobot |

## 相关路径（worktree）

| 路径 | 说明 |
|------|------|
| `Psi0/` | Psi0 主仓库 |
| `Psi0/third_party/GR00T-WholeBodyControl/` | SONIC / GR00T 工作树（建议 `g1_setup`） |
| `.../eef/brainco/` | Brainco Python 驱动（`Brainco` / `Brainco_Controller`） |
| `.../gear_sonic/scripts/pico_manager_thread_server.py` | PICO manager，`--eef brainco` 接入点 |
| `Psi0/real/SONIC/scripts/collect_psi0-sonic-data-manual.sh` | 数采启动封装（pico 默认启用 Brainco） |
| 机器人 `/home/unitree/brainco_hand_service/` | `brainco_hand_server` 与 systemd 单元 |

参考实现来源：`wbc_pico_record/eef/brainco/`。

## Psi / SONIC 环境新增依赖

在 **工作站** teleop 环境（`.venv_teleop`）中：

| 依赖 | 用途 |
|------|------|
| `eef` 包（`third_party/GR00T-WholeBodyControl/eef`） | `from eef.brainco.brainco import Brainco` |
| `unitree_sdk2py`（`external_dependencies/unitree_sdk2_python`） | DDS `MotorCmds_` / `MotorStates_` |
| CycloneDDS（随 SDK / 系统） | 与机器人同一 DDS 域通信 |

`gear_sonic/pyproject.toml` 已将包发现扩展为 `gear_sonic*` + `eef*`，安装 teleop extra 后可从仓库根导入 `eef`。

机器人侧依赖已有 systemd 服务即可，**不必**在 Psi0 的 venv 里再装手部驱动包：

```bash
ssh unitree@192.168.123.164
sudo systemctl status brainco_hand
# 需要时：
sudo systemctl start brainco_hand
```

## 安装命令（工作站）

在 GR00T / SONIC 根目录执行（会创建或刷新 `.venv_teleop`）：

```bash
cd third_party/GR00T-WholeBodyControl
bash install_scripts/install_pico.sh
source .venv_teleop/bin/activate
python -c "from eef.brainco.brainco import Brainco; print('OK')"
```

若环境已存在，只需确保 editable 安装包含 `eef`，并已安装 `unitree_sdk2_python`：

```bash
cd third_party/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
uv pip install -e external_dependencies/unitree_sdk2_python
uv pip install -e "gear_sonic[teleop]"
```

## 启动命令

```bash
# 机器人：确认桥接服务
ssh unitree@192.168.123.164 'sudo systemctl start brainco_hand && systemctl is-active brainco_hand'

# 工作站：PICO（默认 --eef brainco --dds-interface enp4s0）
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico

# 显式指定
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico \
    --eef brainco --dds-interface enp4s0

# 关闭手控
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico --eef none
```

冒烟：启动后按 trigger，终端应出现 `[Brainco] trigger L=.. R=..`，双手开合。  
有日志但手不动 → 查 `brainco_hand` 是否 active、网卡是否正确。

## 当前限制

- 本阶段 **不把 Brainco 关节写入** SONIC LeRobot（exporter 手字段仍可能是 Dex3/占位）。
- `deploy` 仍可能向 Dex3 topic 发命令；无 Dex3 硬件时可忽略。
