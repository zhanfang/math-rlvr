## Context

阶段 5 已经具备真实单机 RLVR 训练能力，并且默认参数已经从极小闭环提升到中等规模。但 `plan.md` 中 Milestone 2 的目标不只是“训练能跑”，还包括更接近完整 GSM8K 小实验的训练规模、独立 held-out 评估、`pass@1` / `pass@k` 对比，以及 case 级失败样例导出。当前仓库仍缺少这些能力，因此还无法用结构化结果回答“RLVR 是否真的比 base model 更好”。

这个阶段仍然保持单机、本地缓存、Qwen 0.5B + LoRA 的边界，不引入 Math-Verify、vLLM 或多卡框架。目标是先把第一轮完整实验闭环建立起来，为后续更强验证器和更高吞吐框架提供稳定基线。

## Goals / Non-Goals

**Goals:**
- 提供一轮更接近 `plan.md` Milestone 2 的 GSM8K 小实验入口。
- 支持将训练规模提升到千样本级、数百步级的推荐配置。
- 提供独立于训练日志的 held-out 评估脚本，支持 `pass@1` 与 `pass@k`。
- 导出 case 级 jsonl 明细，便于人工查看格式失败、抽取失败与算术错误。
- 生成实验摘要，统一记录训练配置、训练耗时、base 指标、adapter 指标和主要失败类型。

**Non-Goals:**
- 不引入 Math-Verify、LaTeX 复杂等价验证、`\boxed{}` 输出等更强验证路径。
- 不引入 vLLM、Ray、DeepSpeed、FSDP、verl、OpenRLHF 或多机多卡。
- 不将 GSM8K 全集训练作为必须完成的强制条件。
- 不追求正式 benchmark 或 SOTA，只要求形成稳定、可复盘的小实验闭环。

## Decisions

1. 使用独立评估脚本，而不是把评估继续塞进训练脚本。

   训练脚本已经承担 dry-run、真实训练、配置摘要和产物保存职责。阶段 6 新增 `scripts/run/gsm8k_eval.py` 一类的独立评估入口，更利于清晰地区分训练与评估，也更符合 `plan.md` 中“能独立 eval，不依赖训练日志”的目标。

   备选方案是给现有训练脚本增加 `--eval` 或 `--compare` 模式。这样入口更少，但会让训练和评估分支继续膨胀。

2. 评估结果按“单模型结果 + 对比摘要 + case 明细”三层组织。

   单模型结果负责记录某一侧模型的 `pass@1` / `pass@k` 和原始输出；对比摘要负责并列展示 base 与 adapter；case 明细负责保留题目、参考答案、原始 completion、抽取答案和正确性结果。这种分层更容易满足训练成功判据中的“独立 eval”和“打开 jsonl 看失败案例”。

   备选方案是只打印控制台汇总。这样实现更快，但很难复盘和人工查错。

3. 阶段 6 通过推荐实验预设来扩大训练规模，而不是直接改成全集默认值。

   当前阶段 5 默认值已经适合作为真实单机基线。阶段 6 更适合引入“完整 GSM8K 小实验”推荐配置，例如 `train_limit=2000`、`max_steps=500`、`num_generations=4`，并允许用户显式覆盖。这样既能对齐 `plan.md` 的实验目标，也不会让默认命令突然变成耗时过长的全集训练。

   备选方案是直接把训练脚本默认值改成 Milestone 2 级别。这样更激进，但会恶化日常迭代体验。

4. 失败样例按规则分类，而不是引入复杂分析器。

   现阶段最关键的是识别 `bad_format`、`extractor_miss`、`arithmetic_error`、`question_misread` 这类明显问题。基于已有答案抽取和正确性判断就足够支持第一轮人工分析。

   备选方案是引入模型辅助的自动失败归因。那更复杂，也超出了当前阶段边界。

5. 实验输出目录独立于阶段 5。

   阶段 6 输出建议保存到 `outputs/stage-6-gsm8k-experiment/`，下分 `train/`、`eval-base/`、`eval-adapter/`、`compare/` 等目录。这样能够清楚区分阶段 5 的中等规模训练基线和阶段 6 的完整实验记录。

   备选方案是继续使用 `outputs/stage-5-real-rlvr/`。这样改动较少，但会混淆不同实验目标。

## Risks / Trade-offs

- [Risk] 千样本级、数百步级训练在本机上耗时明显增加 -> Mitigation: 通过推荐预设和摘要记录给出可预期的时间估算，并保留参数覆盖能力。
- [Risk] `pass@1` 提升不稳定，实验可能没有明显收益 -> Mitigation: 允许同时输出 `pass@4` 和失败样例，避免只看单一指标。
- [Risk] 仅用当前答案抽取规则会低估复杂输出 -> Mitigation: 在 jsonl 中保留原始 completion 和抽取答案，方便人工复核。
- [Risk] 独立评估和训练输出目录分离后，路径配置可能变复杂 -> Mitigation: 在实验脚本中自动串联默认模型路径、adapter 路径和输出目录。
