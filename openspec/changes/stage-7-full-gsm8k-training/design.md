## Context

阶段 6 已经完成本机可复现的小规模 GSM8K RLVR 闭环：Qwen2.5-0.5B-Instruct + LoRA、GSM8K 子集训练、独立 held-out 评估、case jsonl 和 base vs adapter 摘要。当前瓶颈不在功能闭环，而在全量 GSM8K 训练的 rollout 吞吐：Apple M4 Pro 的 MPS 可以完成学习级实验，但全训练集、多 completion、多步 GRPO 会非常耗时。

阶段 7 将训练路径拆成两条：本机继续做 smoke / small-run 验证，云端 CUDA 机器负责 full-run。云端默认目标固定为 RunPod on-demand Pod + NVIDIA L40S 48GB，推荐 16 vCPU、64GB RAM、200GB disk、Ubuntu + CUDA/PyTorch 镜像；单卡优先使用 vLLM colocate，双卡时再切到 server 模式。实现上继续复用现有 `scripts/run/minimal_grpo_training.py`、`src/training/` profile、reward adapter 和 `scripts/run/gsm8k_eval.py`，只在需要的位置扩展全量 profile、vLLM 配置和 SFT warm-start。

## Goals / Non-Goals

**Goals:**
- 新增 `stage7-full-gsm8k-training` profile，用于全量 GSM8K train split 的 RLVR 训练。
- 在 stage 7 profile 和 run summary 中记录 RunPod L40S 48GB 推荐云端目标。
- 支持在 TRL `GRPOConfig` 可用时透传 vLLM 相关参数，覆盖 colocate 和 server 两类常见部署方式。
- 保留无 vLLM 的降级路径，使本机和普通单卡 CUDA 仍能运行 dry-run 或小规模训练。
- 新增可选 SFT warm-start，把 GSM8K 标准推理过程转换成项目当前 XML 输出格式，并保存 LoRA adapter。
- 支持 GRPO 从 SFT adapter 继续训练，或者从 trainer checkpoint 断点续训。
- 文档化本机验证命令、云端依赖安装命令、全量训练命令、评估命令和验收产物。

**Non-Goals:**
- 不在本机强制跑完全量 GSM8K 训练。
- 不引入 Ray、DeepSpeed、FSDP、verl、OpenRLHF 或多机多卡框架。
- 不切换到更大模型作为默认目标；阶段 7 默认仍围绕 Qwen2.5-0.5B-Instruct 学习闭环。
- 不引入 Math-Verify、MATH-500 或复杂 LaTeX 等价验证，这些属于后续更强验证阶段。
- 不把 wandb 或远程实验平台作为必需依赖。

## Decisions

1. 将阶段 7 做成新的 training profile，而不是覆盖阶段 6 默认值。

   阶段 6 的 profile 已经是可在本机复现的有效基线。阶段 7 新增 `stage7-full-gsm8k-training`，默认输出到 `outputs/stage-7-full-gsm8k-training/train`，使用全量 GSM8K train split、较长 completion、较低学习率和保守 KL，并把 RunPod L40S 48GB 云端目标写入 profile summary。这样本机开发和云端 full-run 的命令边界清晰。

   备选方案是把 stage 6 参数直接扩大到全量。这样更少代码，但会破坏已有的本机验证路径。

2. vLLM 参数通过训练脚本显式暴露，并在 `build_grpo_config` 中继续使用签名过滤。

   当前 `build_grpo_config` 已经按本地 TRL `GRPOConfig` signature 过滤字段。阶段 7 延续这个模式，新增 `--use-vllm`、`--vllm-mode`、`--vllm-server-host`、`--vllm-server-port`、`--vllm-gpu-memory-utilization`、`--vllm-tensor-parallel-size`、`--vllm-max-model-length` 等参数。支持的字段会进入 GRPOConfig，不支持的字段不阻断 dry-run，但摘要中必须记录用户请求的 vLLM 配置。

   备选方案是要求固定 TRL/vLLM 版本并直接硬编码参数。这样运行环境更一致，但学习项目会更难在本机和云端之间切换。

3. SFT warm-start 作为可选脚本独立实现。

   新增 `scripts/run/gsm8k_sft.py`，把 GSM8K 的 `answer` 字段拆成 reasoning 和 final answer，并格式化成 `<reasoning>...</reasoning><answer>...</answer>`。脚本使用现有模型、LoRA 参数和数据缓存，输出 `sft_config.json`、`train_metrics.json` 和 `adapter/`。它不替代 RLVR，只作为降低冷启动探索成本的前置步骤。

   备选方案是把 SFT 模式塞进 GRPO 脚本。这样入口更少，但会让训练脚本同时承担两种训练范式，后续维护更混乱。

4. GRPO 支持从初始 adapter 继续训练。

   对于 `--initial-adapter-path`，训练入口加载 base model 后用 PEFT 以 trainable adapter 方式恢复，再交给 GRPOTrainer；未提供时保持现有 `model=args.model_name + peft_config=lora_config` 路径。`--resume-from-checkpoint` 继续表示 trainer checkpoint 级续训，不与初始 SFT adapter 混用语义。

   备选方案是要求用户手动 merge SFT adapter。这样实现更简单，但会引入额外权重产物并降低 LoRA 学习路径的透明度。

5. 全量训练的成功标准仍由独立 eval 判定。

   阶段 7 不把训练 loss 或 reward 曲线当作唯一结论。全量训练完成后必须运行 `scripts/run/gsm8k_eval.py`，生成 base / adapter case jsonl、summary、failure report 和 experiment summary。默认评估可以先用固定 limit 做快速门禁，再用更大的 held-out limit 做正式复盘。

   备选方案是在训练末尾自动评估。这样更自动化，但云端训练失败恢复、分批评估和本机复查都会变得不灵活。

## Risks / Trade-offs

- [Risk] 云端 CUDA/vLLM 版本组合与本机 `.venv` 不一致 -> Mitigation: 训练脚本记录 dependency versions 和 vLLM 请求配置，文档给出推荐安装路径，并保留无 vLLM 降级命令。
- [Risk] SFT warm-start 可能让输出格式变好但 RLVR 提升不稳定 -> Mitigation: SFT adapter 与 GRPO adapter 分目录保存，最终只用独立 held-out eval 判断是否采用。
- [Risk] 全量训练时间和费用仍可能偏高 -> Mitigation: 文档保留 `--train-limit`、`--max-steps`、`--num-generations`、`--max-completion-length` 的分级命令，先小跑再 full-run。
- [Risk] vLLM server 模式需要额外进程，端口或模型路径配置容易错 -> Mitigation: CLI 分别暴露 colocate/server 参数，并在 README 中给出 server 启动与训练连接示例。
- [Risk] LoRA adapter 加载路径和 trainer checkpoint 续训语义混淆 -> Mitigation: 使用 `--initial-adapter-path` 表示 SFT/已有 adapter 起点，使用 `--resume-from-checkpoint` 表示同一次 trainer run 的 checkpoint 恢复。

## Migration Plan

1. 先实现 stage 7 profile、vLLM 参数透传和 dry-run 摘要，不改变 stage 5/6 默认行为。
2. 新增 SFT warm-start 脚本及其 dry-run/小样本验证。
3. 新增 GRPO `--initial-adapter-path` 路径，并用小样本 adapter 验证能继续训练和保存新 adapter。
4. 更新 README、AGENTS 和 plan，写清本机验证、云端安装、云端训练和评估命令。
5. 运行 py_compile、离线 reward/data checks、stage7 dry-run、SFT dry-run、GRPO 小样本 dry-run、OpenSpec strict validate。

Rollback 策略：所有新增能力都通过新 profile、新参数或新脚本暴露；如果 stage 7 出现问题，用户仍可继续使用 stage 5/6 命令。

## Open Questions

- 全量正式评估的 held-out limit 可以先使用 200/500 做快速复盘，再在预算允许时跑完整 test split。
