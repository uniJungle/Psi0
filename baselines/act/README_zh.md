## ACT (Action Chunking Transformer)

### 环境配置

```bash
uv venv .venv-act --python 3.10
source .venv-act/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group psi --group serve --group viz --active --frozen
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv-act/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```


### 下载 Psi0 任务数据

下载任务数据，例如：

```bash
export task=G1WholebodyXMovePick-v0
hf download USC-PSI-Lab/psi-data simple/$task.zip --local-dir=$PSI_HOME/data --repo-type=dataset
unzip "$PSI_HOME/data/simple/$task.zip" -d "$PSI_HOME/data/simple"
```

在 `src/psi/config/train/real_act_config.py` 中为该任务创建新的 `TrainConfig`：

> 如果你微调的是 $Psi_0$ 提供的相同 SIMPLE/real 任务，可以跳过此步骤。


### 训练 ACT

启动训练脚本：

```bash
bash baselines/act/train_act_g1_real.sh $task  # 真机实验训练
bash baselines/act/train_act_g1_simple.sh $task # SIMPLE 仿真训练
```


### 评估 ACT

```bash
export RUN_DIR=xxxx
export CKPT_STEP=40000
bash baselines/act/serve_act_g1_real.sh $RUN_DIR $CKPT_STEP # 真机实验训练
bash baselines/act/serve_act_g1_simple.sh $RUN_DIR $CKPT_STEP # SIMPLE 仿真训练
```


### 在 SIMPLE 中评估

TODO: 使用 SIMPLE third_party 迁移以下说明

```bash
cd <SIMPLE 项目根目录>
source .venv/bin/activate
```

```bash
export task=G1WholebodyXMovePick-v0
```

下载评估数据并解压：

```bash
hf download USC-PSI-Lab/psi-data \
	simple-eval/$task.zip \
	--local-dir=data/evals \
	--repo-type=dataset

unzip data/evals/simple-eval/$task.zip -d data/evals/simple-eval
```

现在在 SIMPLE 环境中启动评估：

> 我们为每个任务提供了三个域随机化级别：`level-0`、`level-1`、`level-2`

```bash
export dr=level-0
```

我们使用两个不同的入口来进行不同任务的评估：

如果评估的任务以 `Teleop` 结尾（即使用遥操作采集的任务数据），则设置入口和智能体为 `eval_decoupled_wbc.py` 和 `act_decoupled_wbc`：

```bash
export entry=eval_decoupled_wbc.py
export agent=act_decoupled_wbc
```

如果评估的任务以 `MP` 结尾（即使用 CuRobo 运动规划生成的任务数据），则设置入口和智能体为 `eval.py` 和 `act_g1`：

```bash
export entry=eval.py
export agent=act_g1
```

```bash
python src/simple/cli/$entry \
	simple/$task \
	$agent \
	$dr \
	--host=localhost \
	--port=22085 \
	--sim-mode=mujoco_isaac \
	--no-headless \
	--data-format=lerobot \
	--data-dir=data/evals/simple-eval/$task/$dr
```
