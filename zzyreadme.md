# TODO
1. 在simple中收集仿真数据[x]
  没必要，有官方的开源数据，调通遥操就完事[x]
2. 数据转换，微调不同模型，在仿真中进行推理，形成可复现实验报告
  用官方的数据训练就行，
  当然没卡的话直接用官方的ckpt试也可以。当然选用一个全身locomotion任务最好[x]
  感觉都太简单了，没什么意思，环境不够真实，最好还能放到lehome那种环境中，但是那样有需要自己采数。
3. 使用sonic收集真机数据
  这个是对的。
4. 真机数据微调

5. 真机部署

6. 实验对比分析，总结写作




# 1. 环境安装
```bash
git clone git@github.com:physical-superintelligence-lab/Psi0.git 
cd Psi0
```
请先安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

配置 $\Psi_0$ 环境：
```bash
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate

git submodule update --init 

最初 git submodule update --init --recursive 失败是因为 third_party/SIMPLE 使用了 git@github.com（SSH 协议），而您的机器没有配置对应的 GitHub SSH 公钥权限（报错 Permission denied (publickey)）。
把 third_party/SIMPLE 内部的 .gitmodules 里的 SSH 链接全部一并替换为了 HTTPS 链接。
重新执行了 git submodule sync --recursive 和 git submodule update --init --recursive，确保它下面嵌套的所有子模块（包括 curobo、AMO、openpi-client 等）都已经成功下载完整。

GIT_LFS_SKIP_SMUDGE=1 uv sync \
  --group serve \
  --group viz \
  --group psi \
  --index-strategy unsafe-best-match \
  --active

uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```

测试安装，应该会显示一个版本号。
```bash
python -c "import psi;print(psi.__version__);"
```

> 如果你想支持 `SIMPLE` 的评估，可以使用以下命令将 `SIMPLE` 与 `Psi0` 一起安装。也可参考 [快速入门](examples/quick_start/psi.md)。

```bash
git submodule update --init --recursive

MAX_JOBS=2 GIT_LFS_SKIP_SMUDGE=1 uv sync \
  --all-groups \
  --index-strategy unsafe-best-match \
  --active \
  --no-build-isolation-package nvidia-curobo

uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```

验证 `SIMPLE` 是否安装成功
``` bash
python -c "import simple; print(simple.__version__)"
```

验证共享的 `lerobot` 栈是否可导入。
```bash
python -c "from psi.data.lerobot.compat import LEROBOT_LAYOUT; print(LEROBOT_LAYOUT)"
# datasets 
```

# 2. 数据采集
## 2.1 环境准备
real/README.md

```bash
conda env create -f psi_deploy_env.yaml
conda activate psi_deploy

git clone https://github.com/physical-superintelligence-lab/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
cd ..

pip install -e .
```
安装PICO相关：
```bash
conda activate psi_deploy

git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git
cd XRoboToolkit-PC-Service-Pybind

mkdir -p tmp
cd tmp
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
bash build.sh
cd ../../../..


mkdir -p lib
mkdir -p include
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
# rm -rf tmp

# Build the project
conda install -c conda-forge pybind11
pip uninstall -y xrobotoolkit_sdk
python setup.py install
```
## 2.2 研究pico数采：
```bash
python main.py --robot g1 --pico_streamer --pico_ip 192.168.0.128 (YOUR_PICO_IP)`
```
### 主函数：real/teleop/main.py：参数分解分配，调用遥操管理器
---
### 遥操管理器：real/teleop/manager.py
---
1. 利用 Python 的 multiprocessing.Manager 创建了一个可以跨进程共享的字典和几个事件（Event）。这些事件就像“信号枪”，用于在不同进程间同步状态，比如：通知子进程何时开始记录 (session_start_event)、何时发生错误 (failure_event) 或者何时退出程序 (kill_event)。
2. 动态计算状态空间维度 (根据 H1 或 G1)：根据传入的机器人型号（h1 或 g1），把该机器人各部分的自由度（腿、手臂、灵巧手、IMU、里程计等）加起来，计算出表示机器人完整状态总共需要多少个浮点数（totalsize）。
3. 开辟超高速“共享内存” (Shared Memory)
#### 4. run_taskmaster：
--- 
主控模块进程（Taskmaster Process）的启动入口。由于遥操作的数据读取、机器人控制和 IK（逆运动学）解算需要高频且稳定地运行，系统将这些核心控制逻辑放入了一个独立的子进程（即 self.taskmaster_proc = Process(target=run_taskmaster)）中运行。
- 也就是说初始化他时就初始化了机器人控制器，eef控制器。
- 调用 taskmaster.start()内部会做以下非常复杂的核心工作：
  - 后台站立维稳：启动一个名为 maintain_standing 的守护线程，在未开始遥操作时，通过 IK 算力维持机器人的站立平衡姿态。
  - 等待启动信号：主线程等待 session_start_event（当你在终端输入 s 开始时触发）。
  - 进入高频控制循环 (run_session)：
    - 读取状态：从共享内存极速获取遥操作设备发来的目标位姿数据，以及机器人自身的关节、IMU 等状态数据。
    - IK 与神经网络解算：结合传统的 Inverse Kinematics（全身逆运动学）和加载的 PyTorch 神经网络（adapter_jit.pt），计算躯干、四肢乃至灵巧手的目标关节角度和前馈力矩。
    - 下发指令执行：调用安全限幅检测 (safelySetMotor)，调用底层的机器人控制器（如 body_ctrl.ctrl_whole_body）控制实体机器人运动。
#### 5. run_dataworker()
---
与之前的 run_taskmaster 非常类似，它是另一个独立子进程（即 数据与视觉工作进程 Dataworker Process）的启动入口。

如果说 Taskmaster 是控制机器人的“大脑和小脑”（负责运动控制和逆运动学），那么 Dataworker 就是机器人的“眼睛”和“录像机”。它负责处理所有和视觉、传感器数据存储、以及 VR 眼镜双向通信相关的繁重任务。
- 它会再启动一个专门的子进程 (TeleoperatorProcess) 用于和 VR 头显（比如 Pico 或 Vuer）通信。向上传：把机器人的第一人称相机画面传给 VR 眼镜，让人类操作者能看到机器人眼前的画面。向下取：高频获取操作者的头部姿态、双手位姿和手柄按键，并将这些数据实时写入到共享内存 teleop_shm_array 中（也就是刚才 Taskmaster 正在疯狂读取的那个数组）。
- 内部建立了一个 ZeroMQ (ZMQ) 客户端，通过局域网不断向远程的相机服务器拉取多模态视觉数据，包括：彩色图 (RGB)、红外图 (IR) 和深度图 (Depth)。
- 数据对齐与录制落盘：在遥操作录制期间（进入 run_session 后），它会以严格的 30 帧/秒 (约 33 毫秒/帧，_sleep_until_mod33) 的频率进行循环。每次循环，它会从 robot_shm_array 中读取 Taskmaster 刚刚更新的最新机器人状态（关节角度、IMU、里程计、甚至手部触觉压力）。然后，它将这些机器人状态与刚才收到的相机画面“打包对齐”，并将彩色图片、压缩后的深度图、以及写有状态参数的 JSONL 文件异步地保存到本地硬盘上。





## 2.3 仿真数采
### 环境配置
```bash
cd third_party/SIMPLE
make live # 构建文档
source .venv/bin/activate
python -c "import simple; print(simple.__version__)" # 0.1.0

# 安装curobo
./scripts/install_curobo.sh
# 测试curobo安装：第一次启动稍慢，拉起isaacsim4.5，两个ur5e机械臂，拖动trarget cube，机械臂自动抓取
python examples/multi_arm_reacher.py

# 测试simple安装
# 在后续你进行“仿真数据收集”、“模型推理评估”时，你需要把这些字符串作为参数传给代码，告诉系统你要在哪个具体的仿真场景下测试或训练机器人。
python scripts/list_env.py
# 我这里就主要考虑G1相关的了
(simple) zzz@unitree:~/zzy/Psi0/third_party/SIMPLE$ python scripts/list_env.py

simple/G1TabletopGraspMP-v0
simple/G1TabletopPickNPlaceMP-v0
simple/G1InspireTabletopGraspMP-v0
simple/G1TabletopHandoverMP-v0
simple/G1WholebodyLocomotionMP-v0
simple/G1WholebodyPickNPlaceMP-v0
simple/G1WholebodyLocomotionPickBetweenTablesMP-v0
simple/G1WholebodySitMP-v0
simple/G1WholebodyBendPickAndPlaceMP-v0
simple/G1WholebodyBendPickMP-v0
simple/G1WholebodyBendHandoverMP-v0
simple/G1WholebodyBendPickAndPlaceOnSofaMP-v0
simple/G1WholebodyTabletopGraspMP-v0
simple/G1InspireWholebodyLocomotionMP-v0
simple/G1InspireWholebodyPickNPlaceMP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant1MP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant2MP-v0
simple/G1WholebodyPickNPlaceVariant1MP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant3MP-v0
simple/G1WholebodyTurnPickMP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant4MP-v0
simple/G1WholebodyXMoveAndPickMP-v0
simple/G1WholebodyXMoveAndPickNPlaceMP-v0
simple/G1WholebodyXMoveBendPickMP-v0
simple/G1WholebodyXMoveBendPickNPlaceMP-v0
simple/G1WholebodyYMoveAndPickMP-v0
simple/G1WholebodyXMoveAndPickVariant1MP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant5MP-v0
simple/G1WholebodyXMoveAndHandoverMP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant6MP-v0
simple/G1WholebodyYMoveAndHandoverMP-v0
simple/G1WholebodyYMoveBendPickMP-v0
simple/G1WholebodyPickAndBendPlaceMP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant7MP-v0
simple/G1WholebodyXMoveAndPickVariant2MP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant8MP-v0
simple/G1WholebodyBendPickVariant1MP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant9MP-v0
simple/G1WholebodyTurnPickVariant1MP-v0
simple/G1WholebodyTabletopGraspVariant1MP-v0
simple/G1WholebodyBendPickAndPlaceOnSofaVariant1MP-v0
simple/G1WholebodyBendPickAndPlaceOnSofaVariant2MP-v0
simple/G1WholebodyYMoveAndPickVariant1MP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant10MP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant11MP-v0
simple/G1WholebodyXMoveAndPickVariant3MP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant12MP-v0
simple/G1WholebodyXMoveAndHandoverVariant1MP-v0
simple/G1WholebodyYMoveAndPickVariant2MP-v0
simple/G1WholebodyTabletopGraspVariant2MP-v0
simple/G1WholebodyTurnXMoveAndPickMP-v0
simple/G1WholebodyTurnXMoveAndPickVariant1MP-v0
simple/G1WholebodyTurnXMoveAndPickVariant2MP-v0
simple/G1WholebodyXMoveBendHandoverMP-v0
simple/G1WholebodyTurnYMoveAndPickMP-v0
simple/G1WholebodyTurnYMoveAndPickVariant1MP-v0
simple/G1WholebodyTurnXMoveAndHandoverMP-v0
simple/G1WholebodyTurnXMoveAndHandoverVariant1MP-v0
simple/G1WholebodyTurnXMoveAndBendPickMP-v0
simple/G1WholebodyTurnYMoveAndBendPickMP-v0
simple/G1WholebodyTurnXMoveAndBendHandoverMP-v0
simple/G1WholebodyYMoveAndHandoverVariant1MP-v0
simple/G1WholebodyTabletopGraspVariant3MP-v0
simple/G1WholebodyTabletopHandoverMP-v0
simple/G1WholebodyLocomotionPickBetweenTablesVariant13MP-v0
simple/G1WholebodyXMovePickTeleop-v0
simple/G1WholebodyXMoveBendPickTeleop-v0
simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0
simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0
simple/G1WholebodyHandoverTeleop-v0
simple/G1WholebodyCloseDoorTeleop-v0
simple/G1WholebodyOpenOvenTeleop-v0
simple/G1WholebodyOpenFaucetTeleop-v0
simple/G1WholebodyPushOfficeChairTeleop-v0
simple/G1WholebodyOpenTrashCanTeleop-v0

# 下载资源
scripts/pre-minimal-download.sh --cleanup
```
### 运行环境
```bash
uv pip install -e ".[rlds]"
sudo apt-get update
sudo apt-get install -y libgmpxx4ldbl

# 实际上就是让一个虚拟机器人在仿真环境里“胡乱动一通”，并在后台把画面录制下来。
python scripts/test_env.py --help --max-episode-steps 1000
```
### VR遥操
1. 安装xr
```bash
cd third_party/SIMPLE/third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/

mkdir -p tmp
cd tmp
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK 
bash build.sh
cd ../../../..

mkdir -p lib
mkdir -p include
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
# rm -rf tmp

# Build the project
uv pip install pybind11

pip uninstall -y xrobotoolkit_sdk
python setup.py install
```

uv pip install -e "third_party/gear_sonic[sim,teleop]" -e "third_party/decoupled_wbc[full]" -e third_party/unitree_sdk2_python

2. 尝试遥操：
```bash
# --target=graspnet1b:0 指定抓取目标物体。这表示使用 graspnet1b（一个常用的开源3D抓取物体数据集）里的第 0 号物体。
python src/simple/cli/teleop_decoupled_wbc.py simple/G1WholebodyXMoveBendPickTeleop-v0 \
--target=graspnet1b:0 --sim-mode=mujoco --no-headless

python src/simple/cli/teleop_decoupled_wbc.py simple/G1WholebodyXMoveBendPickTeleop-v0 \
--target=graspnet1b:0 --sim-mode=mujoco --record --no-headless
```
启动后机器人被吊起，按下右遥杆键机器人会缓慢放下。

### Pico VR 手柄按键映射说明

**系统控制**：
- **放下机器人（Drop Robot）**：右摇杆垂直按下（Right Joystick Click）。**注意：** 环境启动后机器人会悬在空中，必须按此键让其落地，落地后才能开始运动！
- **重置场景（Reset Env）**：同时按住左侧握把键（Left Grip）+ 右侧握把键（Right Grip）。

**全身遥操激活**：
- **接管手臂（激活遥操）**：按住 **左摇杆垂直按下（Left Joystick Click）** 的同时，扣动 **右手的扳机键（Right Trigger）**。终端会打印 `Starting teleop policy`。
- *(注：在接管前，请尽量把你的双手放在身体两侧与机器人默认姿态对齐，以免激活瞬间手臂产生剧烈跳变。再次执行该组合键可取消接管)*

**底盘与身体控制**：
- **前后/平移**：左摇杆（上下控制前进后退，左右控制平移）。
- **转向（Yaw）**：右摇杆（左右拨动控制原地转向）。
- **升高底盘**：左手 Y 键。
- **降低底盘**：左手 X 键。

**手部抓取（非菜单键模式下）**：
- **捏合（食指闭合）**：仅扣动扳机键（Trigger）。
- **握持（中指闭合）**：同时扣动扳机键（Trigger）和握把键（Grip）。
- **其它动作**：仅按握把键（Grip）闭合无名指。

**数据录制控制**（需带 `--record` 参数启动）：
- **开始/保存本条录制**：右手 A 键。
- **放弃当前录制（标记失败）**：右手 B 键。

# 3. 仿真模型微调


# 4. 仿真模型部署

# 5. sonic部署
```bash
mkdir third_party
cd third_party
git clone https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl.git
cd Gtab
git lfs pull
bash install_scripts/install_pico.sh              # .venv_teleop          — VR 遥操作
bash install_scripts/install_data_collection.sh   # .venv_data_collection — LeRobot 录制器
bash install_scripts/install_mujoco_sim.sh        # .venv_sim             — MuJoCo 仿真

python download_from_hf.py                         # SONIC 策略 + 规划器 ONNX 模型

cd gear_sonic_deploy
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh
source scripts/setup_env.sh
echo "source $(pwd)/scripts/setup_env.sh" >> ~/.bashrc

```
## 5.1 仿真遥操作测试
psi原生自动化方式：
```bash
# MuJoCo
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh sim
# C++ 控制器（仿真）
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy sim
# PICO 流媒体
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico
```
好像不是键盘控制，按`]`没用，这个是直接用PICO控制的，ABXY启动，AX开始/暂停遥操。

试试sonic官方控制：
```bash
# term1
cd thir  tab
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# term2
bash gear_sonic_deploy/deploy.sh sim
```
启动控制：
  以下两个谁先谁后都行。
    1. 在终端 2（deploy.sh）中，等到 `init done` 出现就可以按 `]` 启动策略。
    2. 点击 MuJoCo 查看器窗口，按 `9` 将机器人放到地面。
  返回终端 2。按 `T` 播放当前参考动作 — 机器人将执行完整动作。 
  按 `N` 或 `P` 切换到下一个或上一个动作序列。
  再次按 `T` 播放新动作。
  动作完成后可以再次按 `T` 重放相同动作。如果想停止并回到当前动作的第一帧，按 `R` 从头重启。这可用于停止动作而不终止策略。
  完成后或需要紧急停止时，按 `O` 停止控制并退出。

这是对的，这个没问题。

## 5.2 真机遥操测试
```bash
# C++ 控制器
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy
# PICO 流媒体
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico
# 数据导出器（录制）
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter

# 可选：覆盖任务提示词 / 保存路径
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
    --task-prompt "Pick bottle and pour into cup." \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC
```

## 5.3 数据回放
回放应该使用和推理时一样的逻辑，所以微调就先不看了，直接去看他怎么进行的推理，模型输出是怎么下发到机器人的。

1. 在机器人上启动全身控制器（保持运行）：
```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-robot.sh
```
2. 工作站上启动策略客户端：
```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-client.sh
```
内部启动的脚本：
```sh
PORT=8014
# INSTRUCTION="Spray the bowl and wipe it and stack it up."
# INSTRUCTION="Pick toys into box and lift and turn and put on the chair new"
INSTRUCTION="pick up the green grapes and place it into the green bowl"

# psi_rtc_sonic_client.py lives at the GR00T-WholeBodyControl (sonic) submodule root
cd "$(dirname "$0")/../../third_party/GR00T-WholeBodyControl"

# Run in SONIC's .venv_teleop (has gear_sonic + cv2 + zmq + msgpack; websocket-client added for the psi0 RTC client)
./.venv_teleop/bin/python psi_rtc_sonic_client.py \
    --port "$PORT" \
    --instruction "$INSTRUCTION"
```

那么就看一下 psi_rtc_sonic_client.py 是怎么获取到动作然后发布的。

### 仿真回放

需要先在mujoco中启动仿真，启动控制器，最后启动回放数据集

```bash
# MuJoCo
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh sim

# C++ 控制器（仿真）
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy sim

# 回放数据集
python scripts/replay/replay_sim.py 
```
通信建立正常，但是直接发送action.wbc，机器人动作不对。

研究发现，客户端发送的是action_motion_token，这个是发给底层控制器的东西，而不是直接发送关节。
```bash
python scripts/replay/replay_sim.py --mode token --episode_idx 0
```
一切正常，只不过需要预加载数据集到内存中，这个可能是因为要过一层NAS，所以比较慢。

### 真机回放

推荐用采数同款控制器（`--input-type zmq_manager`）。回放脚本会 **PUB bind** `tcp://*:5556`，C++ 默认 `--zmq-host localhost` SUB connect；两端都在工作站跑。

```bash
# 1) 工作站：只启动一个 deploy（关掉其它占用 5556 / DDS 的旧进程）
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy
# 等到 Init Done / [ZMQManager] Host: localhost:5556

# 2) 工作站：token 回放（先 planner=True 进 CONTROL，再切 STREAMED_MOTION）
python scripts/replay/replay_real.py \
  --mode token \
  --episode_idx 0 \
  --input_type zmq_manager \
  --data_dir /home/karthus_chen/ycb_ws/datasets/SONIC/test/2026-07-23/
```

成功时 deploy 端应依次出现：
- `[ZMQManager] Planner enabled` / `motion name is planner_motion`
- `[Control] ... transitioning to CONTROL state`
- `[ZMQManager] Switched to: STREAMED MOTION` / `ZMQ STREAMING MODE: ENABLED`
- `[ZMQEndpointInterface] Protocol v4: Received 64D token ...`

注意：
- 不要用 `deploy_psi0-sonic-rtc-robot.sh`（默认 `input-type=manager` / InterfaceManager，默认 KEYBOARD）做 token 回放。
- 不要边开 pico 边回放（pico 也会 bind 5556，端口冲突）。
- 若只看到 token 日志但机器人不动，多半是 `start` 握手失败（必须先 `planner=True` 再切 streamed）；当前 `replay_real.py` 已按该顺序发送。

## 5.4 暂停恢复逻辑

### 现在的录制中暂停逻辑

首先是录制过程：

```bash
# SONIC C++ 控制器
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy

#启动 PICO 遥操作系统
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico \
    --eef brainco \
    --dds-interface enp4s0

# 数据录制导出逻辑
# 双目：ego_view_left / ego_view_right
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter \
    --task-prompt "Pick bottle and pour into cup." \
    --task-name "test" \
    --root-output-dir /home/karthus_chen/ycb_ws/datasets/SONIC \
    --use-stereo-camera \
    --eef brainco
```

但是当前的录制逻辑，我在录制中会按B暂停机器人动作，然后人调整位置，录制下来的数据会把暂停过程同样保留下来，这是我不希望的，我希望暂停时和恢复后中间这一段的图像和数据都不要保存，恢复后正常继续保存。

### 需要
一个是我暂停之后数据，录制同样暂停，但是这个录制的信息有很多，可能需要对齐一下，每种数据恢复之后怎么处理。

还有就是，我暂停之后，人的位置和朝向都移动了，我恢复遥操时机器人会不会产生动作和朝向的跳变。