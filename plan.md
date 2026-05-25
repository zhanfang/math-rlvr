# math-rlvr 学习计划

## 项目目的

本项目用于学习数学题 RLVR（Reinforcement Learning with Verifiable Rewards）的基本工程闭环：

```text
数学题 prompt
  -> 模型生成推理和答案
  -> 程序抽取最终答案
  -> 自动验证正确性
  -> 根据 reward 更新模型
  -> 独立评估效果
```

第一目标不是追求 SOTA，也不是一次性搭好完整训练平台，而是一步一步理解并跑通一个小而完整的单机实验。

## 基本原则

1. 只做单机版本。
   - 初始版本只面向一台本地机器。
   - 不引入 Ray、DeepSpeed、FSDP、verl、OpenRLHF 或多机多卡训练。
   - vLLM 只作为后续提速方向，不进入第一阶段。

2. 每一步都要可验证。
   - 每个阶段都应该有明确产物和检查命令。
   - 前一步没有跑通，不进入下一步。
   - 先确认环境，再写数据、reward、训练和评估代码。

3. 把学习路径拆小。
   - 不把所有脚本一次性写完。
   - 不在第一阶段下载模型或数据集。
   - 先建立稳定环境，再逐步加入 RLVR 组件。

## 当前 OpenSpec 变更

阶段 1 已完成，对应 OpenSpec change：

```text
openspec/changes/single-machine-learning-rlvr/
```

这个 change 的边界是：

- 重新规划项目为单机学习版。
- 第一阶段只实现环境安装和 smoke check。
- 后续 GRPO 训练、奖励函数、评估、Math-Verify、vLLM、多卡框架都放入后续阶段。

阶段 2 已完成，对应 OpenSpec change：

```text
openspec/changes/stage-2-gsm8k-data-baseline/
```

这个 change 的边界是：

- 加载 GSM8K 小子集并观察数据结构。
- 实现 GSM8K 标准答案抽取。
- 提供离线抽取验证。
- 不下载模型权重，不训练，不做 reward 或评估打分。

阶段 3 已完成，对应 OpenSpec change：

```text
openspec/changes/stage-3-reward-functions/
```

这个 change 的边界是：

- 规定模型输出格式为 `<reasoning>...</reasoning><answer>...</answer>`。
- 实现模型答案抽取、轻量数值等价、正确性 reward 和格式 reward。
- 用本地硬编码样例做离线验证。
- 不下载模型权重，不运行模型生成，不做 GRPO 训练或正式评估。

## 阶段 1：安装环境

这是当前唯一应该优先实现的阶段。

### 目标

创建一个可复现的 Python 环境，并确认核心机器学习依赖可以正常导入。

### 建议依赖

```text
python >= 3.10
torch
transformers
datasets
accelerate
peft
trl
```

### 产物

```text
environment.yml
requirements.txt
scripts/smoke_check_env.py
README.md
```

### 安装命令

当前仓库已使用 `.venv` 跑通阶段 1，优先使用这条路径：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果不激活环境，也可以直接使用：

```bash
.venv/bin/python scripts/smoke_check_env.py
```

conda 保留为可选路径：

```bash
conda env create -f environment.yml
conda activate math-rlvr
```

如果 conda 环境已存在，可以更新：

```bash
conda env update -f environment.yml --prune
conda activate math-rlvr
```

### 检查命令

```bash
python scripts/smoke_check_env.py
```

如果当前 shell 只有 `python3` 命令，可以运行：

```bash
python3 scripts/smoke_check_env.py
```

### 验收标准

运行 smoke check 后应能看到：

```text
Python 版本
PyTorch 版本
Transformers / Datasets / Accelerate / PEFT / TRL 导入成功
可用计算后端：CPU / CUDA / MPS
```

### 第一阶段暂不做

```text
不下载 Qwen 模型
不下载 GSM8K 数据集
不写 GRPO 训练脚本
不写完整评估脚本
不接入 wandb
不安装 math-verify
不安装 vLLM
不使用多卡框架
```

## 阶段 2：数据与基线检查

进入条件：阶段 1 的环境检查通过。

目标：

- 加载 GSM8K 的一个小子集。
- 了解数据结构中的 question 和 answer。
- 实现标准答案抽取，例如 `#### 42` -> `42`。

暂不训练，不下载模型，只观察数据结构和答案抽取问题。

### 检查命令

离线检查答案抽取：

```bash
.venv/bin/python scripts/verify_gsm8k_answers.py
```

检查 GSM8K 小子集：

```bash
.venv/bin/python scripts/cache_gsm8k_dataset.py
```

```bash
.venv/bin/python scripts/inspect_gsm8k_dataset.py --split train --limit 3
```

数据集默认初始化到项目内：

```text
data/hf_datasets/
```

这个目录会被 `.gitignore` 忽略，不应该提交进 git。第一次初始化 GSM8K 可能需要联网下载 Hugging Face 数据集缓存。这个阶段不做模型生成、训练、reward 计算或评估打分。

## 阶段 3：奖励函数

进入条件：能稳定读取数据并抽取标准答案。

目标：

- 规定模型输出格式，例如 `<reasoning>...</reasoning><answer>...</answer>`。
- 实现模型答案抽取。
- 实现数值等价判断。
- 实现正确性 reward 和轻量格式 reward。
- 用手写样例测试 reward 是否符合预期。

### 检查命令

```bash
.venv/bin/python scripts/verify_answer_rewards.py
```

预期结果：

- 使用本地硬编码样例，不下载数据集或模型权重。
- 打印每个样例的模型答案抽取结果、标准答案、正确性 reward 和格式 reward。
- 覆盖正确答案、错误答案、分数与小数等价、逗号数字等价、缺少标签和空标签。

这一阶段仍然可以不训练。

## 阶段 4：最小 GRPO 训练

进入条件：reward 函数能在样例上正确工作。

目标：

- 使用 TRL `GRPOTrainer`。
- 模型先选小模型，例如 `Qwen/Qwen2.5-0.5B-Instruct`。
- 数据先选 GSM8K 小子集。
- 训练步数先设很小，只确认训练闭环能跑通。

### 检查命令

先运行 dry-run，不下载模型、不训练：

```bash
.venv/bin/python scripts/run_minimal_grpo_training.py --dry-run
```

dry-run 通过后，如果本机内存和网络允许，再运行极小真实训练：

```bash
.venv/bin/python scripts/run_minimal_grpo_training.py --max-steps 1 --train-limit 2
```

验收重点：

- 能看到 reward 日志。
- 能保存 LoRA adapter。
- 能解释每个 reward 的作用。
- 输出目录为 `outputs/stage-5-real-rlvr/`，不进入 git。

## 阶段 5：真实单机 RLVR 训练

进入条件：最小训练能完成。

目标：

- 把默认参数切到更高吞吐配置。
- 至少跑一轮中等规模真实训练。
- 保存更有分析价值的 LoRA adapter。
- 记录真实耗时和吞吐。

推荐默认值：

- `train_limit=200`
- `max_steps=100`
- `per_device_train_batch_size=8`
- `max_completion_length=48`
- `logging_steps=10`
- `save_steps=100`

重点不是立刻冲全集，而是先确认：

- 答案抽取是否可靠。
- reward 是否被模型钻空子。
- 训练集表现和 held-out 表现是否一致。
- pass@1 / pass@k 如何变化。

## 后续路线

这些内容等单机最小闭环稳定后再考虑：

```text
Math-Verify：支持 LaTeX、分数、符号表达式等验证
MATH / MATH-500：更复杂的数学数据
vLLM：加速 rollout
wandb：记录实验
verl / OpenRLHF：多卡和更大规模训练
```

## 下一步

下一步只做一件事：

```text
执行阶段 4：最小 GRPO 训练 dry-run 与极小训练闭环
```

对应 OpenSpec 任务见：

```text
openspec/changes/stage-4-minimal-grpo-training/tasks.md
```

阶段 4 仍应保持单机、小模型、小数据、小步数，只先跑通最小训练闭环。
