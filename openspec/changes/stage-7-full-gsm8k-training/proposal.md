## Why

阶段 6 已经证明了单机小规模 GSM8K RLVR 可以产生可验证的 base vs adapter 对比，但在 Apple M4 Pro 本机上继续全量训练 GSM8K 的 rollout 成本过高，迭代会非常慢。现在需要把项目推进到“本机负责开发验证，云端 CUDA 负责全量训练”的阶段，同时保持学习项目的渐进式结构和可复现实验产物。

## What Changes

- 新增阶段 7 全量 GSM8K 训练路径，覆盖全训练集配置、远程 CUDA 环境说明、vLLM rollout 加速开关和独立评估命令。
- 将 RunPod L40S 48GB 作为阶段 7 推荐云端目标，固化推荐 GPU、内存、磁盘、CUDA 镜像和 vLLM 模式。
- 扩展训练脚本参数，使其能在支持的 TRL 版本上启用 vLLM，并在不支持时给出清晰降级提示。
- 新增可选 SFT warm-start 路径，用 GSM8K 标准答案先进行监督微调，再接 RLVR，降低从 base 直接做全量 RLVR 的探索成本。
- 新增全量训练实验 profile，保留本机 smoke / small-run 验证和云端 full-run 配置的明确边界。
- 新增运行文档和验收标准，要求产出训练摘要、adapter、held-out 评估、case jsonl 和 base vs adapter 对比。

## Capabilities

### New Capabilities
- `full-gsm8k-training`: 覆盖阶段 7 的全量 GSM8K 训练工作流，包括云端 CUDA/vLLM 配置、可选 SFT warm-start、全训练集 RLVR、断点续训、独立评估和结果复盘。

### Modified Capabilities
- None.

## Impact

- 可能更新 `src/training/` 下的训练 profile、GRPO 配置适配和数据格式化逻辑。
- 可能更新 `scripts/run/minimal_grpo_training.py`，增加 vLLM、全量 profile、断点续训和初始 adapter 相关参数。
- 可能新增 `scripts/run/gsm8k_sft.py` 或等价入口，用于可选 SFT warm-start。
- 可能更新 `scripts/run/gsm8k_eval.py`，增强全量实验评估复现参数。
- 可能更新 `README.md`、`AGENTS.md` 和 `plan.md`，说明本机验证、云端训练、依赖安装、推荐命令和验收指标。
- 运行产物建议写入 `outputs/stage-7-full-gsm8k-training/`，其中模型权重、dataset cache 和评估 jsonl 继续保持 git 忽略。
