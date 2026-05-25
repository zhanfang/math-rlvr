## Why

阶段 4 已经验证最小 GRPO 训练闭环可以跑通，但当前训练规模只有极少样本和极少步数，产出的 LoRA adapter 主要用于联调，不足以说明 RLVR 已经真正跑起来。阶段 5 需要把训练推进到真实但仍可控的中等规模，在单机上产出一个更有分析价值的 adapter，并记录训练耗时与资源表现。

## What Changes

- 将阶段 5 定义为真实单机 RLVR 训练，而不是独立评估分析。
- 更新训练脚本默认参数，优先采用当前机器上吞吐更高的配置。
- 新增阶段 5 中等规模训练目标，至少运行一轮比阶段 4 明显更大的真实训练。
- 新增阶段 5 训练产物约定，保存更有分析价值的 LoRA adapter、配置摘要和训练日志。
- 新增阶段 5 训练验收要求，记录实际耗时、训练吞吐和 reward 日志。
- 更新文档，说明阶段 5 的默认参数、推荐命令、输出目录和资源边界。

## Capabilities

### New Capabilities
- `real-rlvr-training`: 覆盖阶段 5 的真实单机 RLVR 训练默认参数、中等规模训练执行、训练产物保存和训练耗时记录。

### Modified Capabilities
- None.

## Impact

- 可能更新 `src/` 与 `scripts/run_minimal_grpo_training.py` 的默认训练参数和输出约定。
- 可能新增或更新阶段 5 训练输出目录，例如 `outputs/stage-5-real-rlvr/`。
- 可能更新 `README.md`、`plan.md` 和 `AGENTS.md`，说明阶段 5 的训练规模与资源预期。
- 阶段 5 会继续复用本地缓存模型和 GSM8K 数据，不引入 wandb、vLLM、Math-Verify、Ray、DeepSpeed、FSDP、verl、OpenRLHF 或多机多卡。
