## Context

阶段 1 已经建立 `.venv` 并安装 `torch`、`transformers`、`datasets`、`accelerate`、`peft`、`trl`。阶段 2 已经把 GSM8K 缓存放到项目内 `data/hf_datasets/`，并实现了标准答案抽取。阶段 3 已经实现 `<reasoning>...</reasoning><answer>...</answer>` 的模型答案抽取、轻量数值等价、正确性 reward 和格式 reward。

本地环境当前包含 TRL `1.4.0`、Transformers `5.9.0`、PEFT `0.19.1`、Torch `2.12.0`，`GRPOTrainer` 支持传入模型名、reward 函数、`GRPOConfig`、训练数据集和 `peft_config`。当前机器未检测到 CUDA 或 MPS，因此阶段 4 必须保持非常小，并提供不下载模型的 dry-run/preflight。

## Goals / Non-Goals

**Goals:**

- 新增一个最小 GRPO 训练入口，能在单机上用 GSM8K 小子集跑很少步数。
- 默认使用小模型和 LoRA，降低显存或内存压力。
- 将 GSM8K 样例转换为带 prompt 和标准答案的数据结构。
- 复用阶段 3 的正确性 reward 和格式 reward，适配 TRL `GRPOTrainer` reward 函数接口。
- 支持 dry-run，验证依赖、数据、prompt、reward 和训练配置，不下载模型、不训练。
- 真实训练时能看到 reward 日志，并将 LoRA adapter 和配置保存到 ignored 输出目录。
- 更新文档，说明命令、资源预期、输出目录、验收标准和失败排查方向。

**Non-Goals:**

- 不追求分数提升或稳定收敛。
- 不做 base model 与训练后模型的独立评估。
- 不引入 wandb、vLLM、Ray、DeepSpeed、FSDP、verl、OpenRLHF 或多机多卡。
- 不加入 Math-Verify、SymPy 或复杂数学答案验证。
- 不实现推理服务、批量评测、checkpoint 管理平台或实验追踪系统。

## Decisions

1. 使用一个脚本作为阶段 4 入口。

   新增 `scripts/train_minimal_grpo.py`，集中暴露阶段 4 的学习入口。它支持 `--dry-run`、`--model-name`、`--split`、`--train-limit`、`--max-steps`、`--output-dir`、`--cache-dir` 等参数。默认值必须保守，目标是帮助学习者看到训练闭环，而不是获得好成绩。

   备选方案是拆成多个训练、数据和配置脚本。那更接近正式工程，但会让阶段 4 的学习路径变散。

2. 数据准备复用阶段 2 缓存和答案抽取。

   训练数据从 `openai/gsm8k` 项目本地缓存读取，默认只取少量 train 样例。每条样例转换为至少包含 `prompt` 和 `answer` 的记录，`answer` 使用阶段 2 的标准答案抽取函数。prompt 明确要求模型输出 `<reasoning>` 和 `<answer>` 标签。

   备选方案是复制一份 JSONL 训练数据。它更稳定，但会把数据来源拆开，后续容易和 GSM8K 缓存不同步。

3. Reward 适配层薄而透明。

   新增 TRL reward 适配函数，例如 `grpo_correctness_rewards(completions, answer, **kwargs)` 和 `grpo_format_rewards(completions, **kwargs)`，内部调用阶段 3 的 `correctness_reward` 和 `format_reward`。这样阶段 4 可以保留 reward 组件的可解释性，也便于后续调试模型是否在钻格式漏洞。

   备选方案是把 reward 直接写在训练脚本里。这样实现快，但会破坏阶段 3 已经建立的模块边界。

4. 默认启用 LoRA。

   训练脚本使用 PEFT `LoraConfig` 传给 `GRPOTrainer`，默认保存 adapter 到输出目录。LoRA 让阶段 4 的产物更小，也更贴近后续可继续实验的路径。

   备选方案是全量微调。它概念更直接，但资源需求更高，不适合当前单机学习阶段。

5. dry-run 是必须的第一道验收。

   `--dry-run` 需要检查依赖导入、数据读取、prompt 构造、标准答案字段、reward 函数和 GRPO 配置，但不能加载模型权重、下载模型或执行训练。真实训练命令可以在资源或网络不足时失败，但 dry-run 必须能帮助学习者定位准备工作是否正确。

   备选方案是只提供真实训练命令。那会让网络、模型下载、设备和训练 API 问题纠缠在一起，很难学习。

6. 输出目录统一放在 ignored 路径。

   默认输出到 `outputs/stage-4-minimal-grpo/`，并确保 `outputs/` 被 `.gitignore` 忽略。训练脚本保存参数配置、adapter 和必要的日志提示。不要提交模型权重、adapter、缓存或训练日志。

   备选方案是输出到仓库根目录。那容易污染 git 状态，也不利于后续多次实验。

## Risks / Trade-offs

- [Risk] 当前机器没有 CUDA/MPS，真实训练可能非常慢或内存不足 -> Mitigation: 默认极小 `train-limit` 和 `max-steps`，提供 dry-run，并在文档中说明 CPU 训练只用于闭环验证。
- [Risk] 首次真实训练需要联网下载模型权重 -> Mitigation: dry-run 不下载模型；文档明确真实训练可能需要网络，并保持模型名可配置。
- [Risk] TRL `GRPOTrainer` API 后续可能变化 -> Mitigation: 以本地 TRL `1.4.0` 为实现目标，并在训练脚本预检中打印版本。
- [Risk] 二值 reward 信号稀疏，训练结果可能无提升 -> Mitigation: 阶段 4 的验收只要求闭环、日志和 adapter，不要求准确率提升。
- [Risk] 模型可能输出不符合标签格式的 completion -> Mitigation: prompt 明确输出格式，并保留格式 reward；失败样例分析留给阶段 5。
