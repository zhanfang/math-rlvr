## Why

阶段 1 到阶段 3 已经分别跑通环境、GSM8K 数据读取和 reward 函数原型。现在需要进入最小训练闭环：用 TRL `GRPOTrainer` 在单机上跑一个极小 GSM8K 子集，确认模型生成、reward 计算、参数更新和 LoRA 保存可以连起来。

## What Changes

- 新增阶段 4 最小 GRPO 训练入口，默认使用小模型、小数据、小步数和 LoRA。
- 新增训练数据准备逻辑，将 GSM8K 样例转换为 GRPO 可用 prompt，并携带标准答案用于 reward。
- 新增 GRPO reward 适配器，复用阶段 3 的正确性 reward 和格式 reward。
- 新增训练前预检或 dry-run 模式，用于检查 TRL/PEFT/Transformers、数据、reward 和训练配置，不下载模型或训练。
- 新增本地输出约定，将 LoRA adapter、训练日志和配置保存到 ignored 输出目录。
- 更新文档，说明阶段 4 的命令、资源预期、验收标准和边界。

## Capabilities

### New Capabilities
- `minimal-grpo-training`: 覆盖阶段 4 的单机最小 GRPO 训练脚本、数据准备、reward 接入、dry-run 预检、训练输出和文档化验收。

### Modified Capabilities
- None.

## Impact

- 可能新增 `src/` 下的训练数据与 GRPO reward 适配模块。
- 可能新增 `scripts/run_minimal_grpo_training.py` 作为阶段 4 训练入口。
- 可能更新 `.gitignore`，忽略训练输出目录，例如 `outputs/`。
- 可能更新 `README.md`、`plan.md` 和 `AGENTS.md`。
- 首次真实训练可能需要联网下载模型权重；dry-run 不应下载模型权重。
- 不引入 vLLM、Ray、DeepSpeed、FSDP、verl、OpenRLHF、多机多卡、wandb、Math-Verify 或正式评估脚本。
