## Why

阶段 5 已经把真实单机 RLVR 训练跑起来，但距离 `plan.md` 里“完整 GSM8K 小实验”的目标还有明显缺口：目前还没有独立 held-out 评估、base vs adapter 的稳定对比、case 级失败样例导出，也没有一轮更接近 `train 2000+ / steps 500+` 的实验记录来判断 pass@1 是否真正提升。现在需要把“训练能跑”推进到“实验可验证、可对比、可复盘”。

## What Changes

- 新增完整 GSM8K 小实验入口，覆盖更大规模的训练配置、独立 held-out 评估和实验摘要输出。
- 新增 base model 与 RLVR adapter 的独立评估流程，支持 `pass@1` 和 `pass@k`。
- 新增 case 级 jsonl 导出，保存题目、参考答案、模型输出、抽取答案和正确性结果。
- 新增实验结果对比摘要，汇总训练配置、训练耗时、base 指标、adapter 指标和主要失败类型。
- 新增阶段 6 文档和验收标准，使其对齐 `plan.md` 中的 Milestone 2 成功判据。

## Capabilities

### New Capabilities
- `complete-gsm8k-experiment`: 覆盖阶段 6 的更大规模 GSM8K 训练、独立 held-out 评估、pass@k 统计、case 导出和实验结果对比汇总。

### Modified Capabilities
- None.

## Impact

- 可能新增 `src/` 下的评估、结果汇总和失败分析模块。
- 可能新增 `scripts/` 下的独立评估或完整实验入口脚本。
- 可能更新 `scripts/run/minimal_grpo_training.py`，以支持阶段 6 推荐训练配置或实验元数据落盘。
- 可能新增或更新 `outputs/stage-6-gsm8k-experiment/` 一类的 git 忽略目录。
- 可能更新 `README.md`、`plan.md` 和 `AGENTS.md`，说明完整 GSM8K 小实验的命令、输出与成功标准。
