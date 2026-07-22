<h1 align="center">[RSS26'] Ψ₀: 面向通用人形机器人的移动操作开源基础模型<br/> </h1>

<p align="center">
  <img src="assets/media/teaser.jpg" alt="Psi0 teaser image" />
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.12263-df2a2a.svg)](https://arxiv.org/abs/2603.12263)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://psi-lab.ai/Psi0)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/USC-PSI-Lab/psi-model)
[![Data](https://img.shields.io/badge/Hugging%20Face-Data-pink)](https://huggingface.co/datasets/USC-PSI-Lab/psi-data)
[![License](https://img.shields.io/badge/License-Apache2.0-blue.svg)](./LICENSE)

</div>

贡献者：[Songlin Wei](https://songlin.github.io/), [Hongyi Jing](https://hongyijing.me/), [Boqian Li](https://boqian-li.github.io/), [Zhenyu Zhao](https://zhenyuzhao.com/), [Jiageng Mao](https://pointscoder.github.io/), [Zhenhao Ni](https://nizhenhao-3.github.io/) , [Sicheng He](https://hesicheng.net/), [Jie Liu](https://jie0530.github.io/), [Xiawei Liu](https://www.xiaweiliu.com/), Kaidi Kang, Sheng Zang,[Weiduo Yuan](https://weiduoyuan.com/), [Marco Pavone](https://profiles.stanford.edu/marco-pavone), Di Huang, [Yue Wang](https://yuewang.xyz/)

-------

$\Psi_0$ 是一个面向灵巧人形机器人移动操作（loco-manipulation）的开源视觉-语言-动作（VLA）模型。我们的模型首先从大规模人类第一人称视频中学习任务语义和视觉表征，然后在少量真实世界的遥操作机器人数据上进行后训练（post-training），以学习具身的通用动力学。

<details>
<summary>[可选] 展开了解更多关于 Ψ₀ 的信息。</summary>

我们的基础模型能够通过仅 80 条轨迹的微调，获得全新的长视界灵巧移动操作技能。***我们的核心发现是：以正确的方式扩展正确的数据。***

在顶层，$\Psi_0$ 模型由两个端到端训练的组件构成：一个视觉-语言主干网络（System-2）和一个多模态扩散 Transformer（System-1）动作专家。主干网络基于 Qwen 的 Qwen3-VL-2B-Instruct，负责从观察和指令中提取视觉-语言特征。这些特征作为基于流的多模态扩散 Transformer（灵感来自 Stable Diffusion 3）的条件。动作专家（约 5 亿参数）预测未来的全身动作块（action chunks），从而实现视觉、语言和动作特征的高效融合。在底层（System-0），基于强化学习的追踪控制器执行预测的下半身动作指令，确保稳定和精确的物理控制。

<p align="center">
  <img src="assets/media/arch.png" alt="Psi0 model" />
</p>
</details>

<p></p>

## 📢 最新动态 (News & Updates)

* [2026-06-13] 发布了 Psi-0 与 SONIC 的集成。
* [2026-06-03] 🎉🎉🎉 Psi-0 在 CVPR 2026 第 2 届 3D-LLM/VLA Workshop 荣获最佳论文奖。



## 目录
- [在 Unitree G1 人形机器人上微调 Ψ₀](#finetune-psi0)
  - [安装](#installation)
  - [数据采集](#data-collection)
  - [微调](#training-real)
  - [开环评估](#open-loop-evaluation)
  - [部署](#deployment)
  - [结合 SONIC 的 Ψ₀](#psi0-sonic)
- [基线模型 (Baselines)](#baselines)
  - [GR00T N1.6](#groot-n16)
  - [OpenPi π0.5](#openpi-05)
  - [InternVLA-M1](#internvla-m1)
  - [H-RDT](#h-rdt)
  - [EgoVLA](#egovla)
  - [Diffusion Policy](#diffusion-policy)
  - [ACT](#act) 
- [仿真 🚀🚀🚀](#simulation)
  - [安装 SIMPLE](#install-simple)
  - [数据生成](#data-generation)
  - [微调](#training-sim)
  - [在 SIMPLE 中评估](#evaluation-in-simple)
- [复现 Ψ₀：预训练与后训练](#pre-post-train)
- [模型权重 (Checkpoints)](#checkpoints)
- [常见问题排查](#troubleshootings)
- [引用](#️-citation)

<a id="finetune-psi0"></a>
## 在 Unitree G1 人形机器人上微调 Ψ₀

### 安装

克隆项目并进入项目根目录：
```bash
git clone git@github.com:physical-superintelligence-lab/Psi0.git 
cd Psi0
```
我们使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 来管理 Python 依赖。如果尚未安装，请先安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

配置 $\Psi_0$ 环境：

> ℹ️ 我们通过 `uv` 管理 $\Psi_0$ 环境以及所有的基线模型，它们都共享相同的 `src/` 代码。更多细节请参阅 [环境管理](baselines/README.md)。

```
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync \
  --group serve \
  --group viz \
  --group psi \
  --index-strategy unsafe-best-match \
  --active
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```

> 如果你想支持 `SIMPLE` 的评估，可以使用以下命令将 `SIMPLE` 与 `Psi0` 一起安装。也可参考 [快速入门](examples/quick_start/psi.md)。

```
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match --active --no-build-isolation-package nvidia-curobo
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
UV_PROJECT_ENVIRONMENT=${pwd}/.venv-psi ./scripts/install_curobo.sh
```

测试安装，应该会显示一个版本号。
```bash
python -c "import psi;print(psi.__version__);"
```

验证 `SIMPLE` 是否安装成功
``` bash
python -c "import simple; print(simple.__version__)"
```

验证共享的 `lerobot` 栈是否可导入。
```bash
python -c "from psi.data.lerobot.compat import LEROBOT_LAYOUT; print(LEROBOT_LAYOUT)"
```

### 数据采集
> 📂 我们开源了所有的 9 个真实世界任务。您可以直接下载数据并跳到 [微调](#training-real) 部分。

在此查看详细的遥操作指南：  
[真实世界部署指南](real/README.md#real-world-deployment)

#### 预处理：将原始数据转换为 LeRobot 格式

```
export task=Hug_box_and_move

hf download USC-PSI-Lab/psi-data \
  g1_real_raw/$task.zip \
  --local-dir=$PSI_HOME/data/real_teleop_g1 \
  --repo-type=dataset

unzip $PSI_HOME/data/real_teleop_g1/g1_real_raw/$task.zip -d $PSI_HOME/data/real_teleop_g1/g1_real_raw/$task
```
你应该能看到类似的文件夹结构：

```
g1_real_raw
└── Hug_box_and_move
    ├── episode_0
    │   ├── color
    │   │   ├── frame_000000.jpg
    │   │   └── ...
    │   └── data.json
    └── ...
```

使用以下格式编辑任务描述文件，例如：
```
vim scripts/data/task_description_dict.json
```
```
{
  "Hug_box_and_move": "Hug box and move."
}
```

运行转换脚本：
```
python scripts/data/raw_to_lerobot.py \
  --data-root=$PWD/data/real_teleop_g1/g1_real_raw \
  --work-dir=$PWD/data/real \
  --repo-id=psi0-real-g1 \
  --robot-type=g1 \
  --task=$task
```

计算统计信息：
```
python scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/real \
  --task=$task
```

创建 **$\Psi_0$** 格式的统计信息（目前仅为复制）：
```
cp $PSI_HOME/data/real/$task/meta/stats.json $PSI_HOME/data/real/$task/meta/stats_psi0.json
```

现在就可以微调 $\Psi_0$ 了。

> ✈️ 如果训练环境已经配置好，可以直接通过 `scripts/train/psi0/finetune-real-psi0.sh $task` 启动训练。


<a id="training-real"></a>
### 微调

> ✔️ 假设数据已经采集并处理完毕。现在我们可以开始微调 $\Psi_0$ 模型。

> 加载真实数据时存在一个 [已知问题](https://github.com/physical-superintelligence-lab/Psi0/issues/3)，请先应用此修复程序：`python scripts/data/patch_lerobot_meta.py $PSI_HOME/data/real/$task`

> 📝 这里我们通过使用来自 [Huggingface psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/real) 的预采集数据来进行说明。

根据 `.env.sample` 配置环境变量。Python 中的 `dotenv.load_dotenv()` 将加载这些环境变量。
```
cp .env.sample .env
# 并编辑以下环境变量 
# HF_TOKEN=<YOUR HF READ TOKEN>
# WANDB_API_KEY=<API KEY for wandb logging>
# WANDB_ENTITY=<wandb entity>
# PSI_HOME=<根据惯例存放 PSI 缓存/权重/数据的路径>

source .env
echo $PSI_HOME
```

下载采集的真实世界数据并解压：
```
export task=Pick_bottle_and_turn_and_pour_into_cup

hf download USC-PSI-Lab/psi-data \
  real/$task.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset

unzip $PSI_HOME/data/real/$task.zip -d $PSI_HOME/data/real
```
> 👀 如果你想可视化回放数据，请参阅 examples 中的 [数据可视化](examples/visualize.md)。

启动训练脚本：
```
scripts/train/psi0/finetune-real-psi0.sh $task
```

> 🖥️ 你随时可以更改使用的 GPU，例如 `CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train/...`。  

> ⚠️ 请尽量保持一个合理的全局批大小 (global batch size) = device batch size x number of GPUs x gradient accumulation step。在所有的真实世界和仿真实验中，我们统一使用全局批大小 128。


### 开环评估
> 请按照 `examples/simple/openloop_eval.ipynb` 中的步骤进行。

加载训练数据集，并运行模型推理以查看模型对训练数据的拟合情况。

### 部署

#### 启动 $\Psi_0$ 服务端 (RTC 模式)

```bash
bash ./scripts/deploy/serve_psi0-rtc.sh
```

#### 启动 $\Psi_0$ 客户端 (RTC 模式)

```bash
bash ./real/scripts/deploy_psi0-rtc.sh
```

有关真实的部署环境设置，请参阅专用文档：

[真实世界遥操作指南](real/README.md)


<a id="psi0-sonic"></a>
### 结合 SONIC 的 Ψ₀

[SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 是一款强大的人形机器人全身控制器。$\Psi_0$ 现已支持结合 SONIC 进行数据采集、微调与部署。请使用 [我们的分支 (fork)](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl/tree/main) 以避免任何兼容性问题。

首先初始化 SONIC 子模块：

```bash
git submodule update --init --recursive third_party/GR00T-WholeBodyControl
```

有关完整的环境设置（工作站虚拟环境、TensorRT + C++ 构建、PICO/XRoboToolkit 以及机器人端相机服务端），请参阅 **[SONIC 真实世界遥操作指南](real/SONIC/README.md)**。

#### 数据采集

请按照 [SONIC 真实世界遥操作指南](real/SONIC/README.md#data-collection) 来录制演示数据。

数据集以 LeRobot 格式保存在本地目录 `third_party/GR00T-WholeBodyControl/outputs/<dataset-name>/` 下。

#### 预处理：转换为 $\Psi_0$ LeRobot 格式

将使用 SONIC 采集的数据集转换为 $\Psi_0$ LeRobot 格式：

```bash
export task=<dataset-name>

python scripts/data/raw_sonic_to_psi_lerobot.py \
  --data-root=third_party/GR00T-WholeBodyControl/outputs/$task \
  --work-dir=$PSI_HOME/data/sonic/lerobot \
  --repo-id=$task \
  --robot-type=g1
```

计算统计信息：
```bash
python scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/sonic/lerobot \
  --task=$task
```

创建 **$\Psi_0$** 格式的统计信息（目前仅为复制）：
```bash
cp $PSI_HOME/data/sonic/lerobot/$task/meta/stats.json $PSI_HOME/data/sonic/lerobot/$task/meta/stats_psi0.json
```

现在就可以进行微调了。

#### 结合 SONIC 微调 $\Psi_0$

```bash
bash ./scripts/train/psi0/finetune-real-sonic-psi0.sh $task
```

#### 结合 SONIC 部署 $\Psi_0$

详细说明请参阅 [SONIC 真实世界部署指南](real/SONIC/DEPLOYMENT.md)。

##### 启动结合 SONIC 的 $\Psi_0$ 策略服务端 (RTC 模式)

```bash
bash ./scripts/deploy/serve_psi0-rtc-sonic.sh
```

##### 在机器人上启动结合 SONIC 的 $\Psi_0$ 全身控制器 (RTC 模式)

```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-robot.sh
```

##### 启动结合 SONIC 的 $\Psi_0$ 策略客户端 (RTC 模式)

```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-client.sh
```

## 基线模型 (Baselines)

<a id="groot-n16"></a>

### GR00T
安装环境：
```bash
cd src/gr00t; uv sync
```
1. 训练
```bash
cd src/gr00t
./scripts/train_gr00t.sh --dataset-path /your/lerobot/dataset
```
2. 部署权重模型
```bash
cd src/gr00t
./scripts/deploy_gr00t.sh
```

3. 使用 GT 对训练好的模型进行开环评估
```bash
cd src/gr00t
./scripts/openloop_eval.sh
```

<a id="openpi-05"></a>

### OpenPI $\pi_{0.5}$

请在此处查看更详细的说明：[baselines/pi05](baselines/pi05/README.md)。

### InternVLA-M1
安装环境：
```bash
cd src/InternVLA-M1; uv sync --python 3.10
```
1. 训练
```bash
cd src/InternVLA-M1
bash scripts/train_internvla.sh
```
2. 部署权重模型
```bash
cd src/InternVLA-M1
./scripts/deploy_internvla.sh
```

### H-RDT

查看 [baseline/hrdt](examples/quick_start/hrdt.md) 的快速入门文档。

### EgoVLA

查看 [baseline/egovla](examples/quick_start/egovla.md) 的快速入门文档。

### Diffusion Policy
查看专用文档：[baseline/dp](baselines/dp/README.md)

### ACT
查看专用文档：[baseline/act](baselines/act/README.md)

## 仿真

我们使用 [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) 来对 $\Psi_0$ 及所有基线模型进行基准测试。

> 📢 SIMPLE 是一个易于使用的人形机器人基准测试仿真器，基于 MuJoCo 物理引擎和 Isaac Sim 渲染构建。

### 安装 SIMPLE

目前，有两种将 SIMPLE 和 Psi-0 集成的方法。

#### [选项 1] 独立安装 SIMPLE（最适合通过遥操作收集数据）

> 我们建议在配备 NVIDIA GPU (3090/4090/5090) 的独立台式机上安装 [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE)。

请参考 [此处的](https://github.com/physical-superintelligence-lab/SIMPLE) SIMPLE 仓库。

#### [选项 2] 将 SIMPLE 作为第三方依赖项安装（最适合评估 Psi-0 和所有基线）

请参考 [此处的](examples/quick_start/psi.md) 更详细的步骤。

### 数据生成
> 📂 我们在 [Huggingface psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple) 提供了 6 个预先收集的全身人形机器人移动操作任务数据。如果你想直接使用现有的仿真数据，可以跳到 [微调](#training-sim)。

#### 基于运动规划的数据生成
请参考 SIMPLE 文档。

#### 仿真器中的遥操作
请参考 SIMPLE 文档。

<a id="training-sim"></a>
### 微调

> 👉 你可以跳过微调，直接下载我们发布的 [SIMPLE 模型权重](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/simple-checkpoints)。

下载 [SIMPLE 任务数据](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple) 并解压：

> 💡 在执行以下命令之前，别忘了先执行 `source .env`。

```
export task=G1WholebodyXMovePickTeleop-v0

hf download USC-PSI-Lab/psi-data \
  simple/$task.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset

unzip $PSI_HOME/data/simple/$task.zip -d $PSI_HOME/data/simple
```

> 👀 如果你想可视化回放数据，请参阅 examples 中的 [数据可视化](examples/visualize.md)。

启动训练：

> 如果尚未配置环境变量，请参考 [设置环境变量](#training-real)。

```
bash scripts/train/psi0/finetune-simple-psi0.sh $task
```
训练将创建一个运行目录，位于项目根目录的 `.runs` 文件夹下。
如果你的 GPU 显存有限，可以设置 `--train.optimizer-foreach=false` 以降低优化器步骤的内存使用，但这会牺牲一定的训练速度。

### 在 SIMPLE 中评估

#### 启动 $\Psi_0$ 服务端
```
export run_dir=<你的运行目录路径，在 .runs 文件夹下>
export ckpt_step=<模型权重对应的 step>
uv run --active --group psi --group serve serve_psi0 \
  --host 0.0.0.0 \
  --port 22085 \
  --run-dir=$run_dir \
  --ckpt-step=$ckpt_step \
  --action-exec-horizon=24 \
  --rtc
```

运行开环评估（离线）

[examples/simple/openloop_eval.ipynb](examples/simple/openloop_eval.ipynb)

#### 在 SIMPLE 中运行评估

此 `快速入门` 指南假设你在一台配有 NVIDIA GPU 的独立工作站上运行 SIMPLE。

> 我们建议在本地以外的远程服务器上部署 VLA 模型，因为 IsaacSim 同样非常占用资源。 

> 如果服务端部署在远程服务器上，请运行 SSH 端口转发。例如：`ssh -L 22086:localhost:22086 songlin@nebula100`。

> 端口转发完成后，打开一个新终端测试服务端是否启动：`curl -i http://localhost:22085/health`

从 [USC-PSI-Lab/psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval) 下载评估任务。

```
cd /path/to/SIMPLE
export task=G1WholebodyXMovePickTeleop-v0
```

下载评估数据并解压：
```
hf download USC-PSI-Lab/psi-data \
	simple-eval/$task.zip \
	--local-dir=data/evals \
	--repo-type=dataset

unzip data/evals/simple-eval/$task.zip -d data/evals/simple-eval
```

现在在 SIMPLE 环境中启动 SIMPLE 评估：

> 我们为每个任务提供了三个域随机化（Domain Randomization）级别：`level-0`, `level-1`, `level-2`。

```
export dr=level-0
```
我们使用两个不同的入口点来评估不同的任务：

如果评估的任务以 `Teleop` 结尾，表示任务数据是通过遥操作收集的，需将 entrypoint 和 agent 设置为 `eval_decoupled_wbc.py` 和 `psi0_decoupled_wbc`：
```
export entry=eval_decoupled_wbc.py
export agent=psi0_decoupled_wbc
```

如果评估的任务以 `MP` 结尾，表示任务数据是使用 CuRobo 运动规划生成的，需将 entrypoint 和 agent 设置为 `eval.py` 和 `psi0`：
```
export entry=eval.py
export agent=psi0
```

启动评估脚本：
```
python src/simple/cli/$entry \
	simple/$task \
	$agent \
	$dr \
	--host=localhost \
	--port=9000 \
	--sim-mode=mujoco_isaac \
	--no-headless \
	--data-format=lerobot \
	--data-dir=data/evals/simple-eval/$task/$dr
```

策略生成的演示视频将保存在 `third_party/SIMPLE/data/evals/psi0` 文件夹中。

> 单个回合的评估可能长达 6~10 分钟，因为 SIMPLE 在 IsaacSim 中使用了同步渲染 API。详见 [更多解释](#)。

<a id="pre-post-train"></a>
## 复现 Ψ₀：预训练与后训练


### 预训练 VLM

下载并缓存官方的 `Qwen/Qwen3-VL-2B-Instruct` 权重。
```
scripts/predownload_qwen3vl.py
```

在 [EgoDex 数据集](https://github.com/apple/ml-egodex) 上进行预训练

预计算 `48 自由度 EgoDex 动作`：

> 我们重用了来自 [H-RDT EgoDex 预处理](https://github.com/HongzheBi/H_RDT?tab=readme-ov-file#data-preprocessing) 的处理代码。
> 1. 修改 `src/h_rdt/datasets/pretrain/setup_pretrain.sh` 中的路径。
> 2. 如果服务器性能强大，可以微调 `NUM_PROCESSES`（我尝试过的最大值是 64）。
> 3. 如果处理脚本中断，可以设置 `FORCE_OVERWRITE=True`。

```
source src/h_rdt/datasets/pretrain/setup_pretrain.sh
source .venv-psi/bin/activate
bash src/h_rdt/datasets/pretrain/run_pretrain_pipeline.sh
```

> [可选] 如果你还想训练 `FAST` tokenizer，请参阅 [训练 FAST](src/fast/README.md)。

```
bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh 
```

在 [humanoid everyday (HE) 数据集](https://huggingface.co/datasets/USC-GVL/humanoid-everyday) 上进行预训练

> 请在此处下载预处理的 HE 数据：`hf download USC-PSI-Lab/psi-data HE_RAW.zip --repo-type=dataset`

```
bash scripts/train/psi0/pretrain-he-psi0-fast.sh
```

训练完成后保存预训练的权重模型：
```
python scripts/save_pretrain_qwen3vl_backbone.py
```

### 后训练动作专家 (Action Expert)

下载预训练的 `psi-0` VLM 主干网络权重
```
python scripts/data/download.py \
  --repo-id=USC-PSI-Lab/psi-model \
  --remote-dir=psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
  --local-dir=$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
  --repo-type=model
```

在 [humanoid everyday (HE) 数据集](https://huggingface.co/datasets/USC-GVL/humanoid-everyday) 上进行后训练
```
bash scripts/train/psi0/posttrain-he-psi0.sh
```

训练结束后保存经过后训练的动作头 (Action header)：
```
python scripts/save_posttrain_action_expert.py
```

## 模型权重 (Checkpoints)

发布在 [HuggingFace Psi-Model](https://huggingface.co/USC-PSI-Lab/psi-model) 上的模型权重如下所示：

| Checkpoint | 描述 | Remote Directory |
|---|---|---|
| $\Psi_0$ VLM<br/>(Baseline) | 预训练的 VLM 主干网络 (EgoDex 200K steps + HE 30K steps) | `psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k` |
| $\Psi_0$ Action Expert<br/>(Baseline) | 在 HE 上进行后训练的动作专家 | `psi0/postpre.1by1.pad36.2601131206.ckpt.he30k` |

用于消融实验的更多变体：
| Checkpoint | 描述 | Remote Directory |
|---|---|---|
| $\Psi_0$ VLM<br/>(Ablation Study) | 仅在 EgoDex 上预训练的 VLM 主干网络，200K steps | `psi0/pre.fast.egodex.2512241941.ckpt200k` |
| $\Psi_0$ VLM<br/>(Ablation Study) | 仅在 HE 上预训练的 VLM 主干网络，48K steps  | `psi0/pre.abl.only.he.2512311516.48k` |
| $\Psi_0$ VLM<br/>(Ablation Study) | 仅在 10% EgoDex 上预训练的 VLM 主干网络  | `psi0/pre.abl.ego.10per.2602021632.46k` |
| $\Psi_0$ Action Expert<br/>(Ablation Study) | 在 HE 上对预训练变体 `psi0/pre.abl.only.he.2512311516.48k` 进行后训练 | `psi0/postpre.abl.only.he.2602050012` |
| $\Psi_0$ Action Expert<br/>(Ablation Study) | 在 HE 上对预训练变体 `psi0/pre.abl.ego.10per.2602021632.46k` 进行后训练 | `psi0/postpre.abl.ego.10per.2602050006` |


下载选定的模型

> 若有需要，可以编辑 `.env` 以使用 `HF_ENDPOINT=https://hf-mirror.com`。

```
python scripts/data/download.py \
  --repo-id=USC-PSI-Lab/psi-model \
  --remote-dir=<Remote Directory> \
  --local-dir=$PSI_HOME/cache/checkpoints/<Remote Directory> \
  --repo-type=model
```

## 常见问题排查 (Troubleshootings)

1. Lerobot 数据集问题：`stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column`

这通常意味着环境仍在使用旧版的 PSI `lerobot` 栈。重新同步 PSI 环境，使其使用与 SIMPLE 相同的 `lerobot` 和 `datasets` 版本，然后验证导入结构：

```bash
source .venv-psi/bin/activate
uv sync --group psi --active
python -c "from psi.data.lerobot.compat import LEROBOT_LAYOUT; print(LEROBOT_LAYOUT)"
```

2. 无法安装 `evdev`，报错 `src/evdev/input.c:10:10: fatal error: Python.h: No such file or directory`

```
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential \
    linux-headers-$(uname -r)
```

3. `RuntimeError: Could not load libtorchcodec. Likely causes ...`
```
sudo apt-get install ffmpeg
```

4. `ImportError: cannot import name 'Deprecated' from 'wandb.proto.wandb_telemetry_pb2'` 

重新安装 `wandb`
```
source .venv-pusht/bin/activate
uv pip uninstall wandb
uv pip install wandb==0.18.0
```

5. 在类似 `5090` 或 `RTX 6000` 这样的较新 GPU 上支持 `sm_120` 时报错，`UserWarning: Ignoring invalid value for boolean flag CUDA_LAUNCH_BLOCKING: truevalid values are 0 or 1.`

更新 `torch` 和 `flash-attn`
```
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install flash-attn --no-build-isolation
```

6. 下载并构建 `lerobot ...` 失败。使用 `git lfs logs last` 查看日志。

```
GIT_LFS_SKIP_SMUDGE=1 uv ...
```
## 引用

```
@article{wei2026psi0,
  title={{$\Psi_0$}: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation},
  author={Wei, Songlin and Jing, Hongyi and Li, Boqian and Zhao, Zhenyu and Mao, Jiageng and Ni, Zhenhao and He, Sicheng and Liu, Jie and Liu, Xiawei and Kang, Kaidi and others},
  journal={arXiv preprint arXiv:2603.12263},
  year={2026}
}
```

## 许可证 (License)

本项目采用 Apache License 2.0 许可证。

有关详细信息，请参阅 [LICENSE](LICENSE) 文件。
