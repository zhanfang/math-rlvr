## 1. 训练数据准备

- [x] 1.1 新增阶段 4 数据准备模块或函数，将 GSM8K 样例转换为 GRPO 训练记录。
- [x] 1.2 每条训练记录包含 `prompt` 和 `answer` 字段，`answer` 复用阶段 2 标准答案抽取。
- [x] 1.3 prompt 明确要求模型输出 `<reasoning>...</reasoning><answer>...</answer>`。
- [x] 1.4 支持 `split`、`train-limit` 和项目本地 `cache-dir` 参数。
- [x] 1.5 为数据准备增加轻量离线检查，确认少量样例可构造成功。

## 2. GRPO Reward 适配

- [x] 2.1 新增 TRL reward 适配函数，接收 completions 和标准答案并返回浮点分数列表。
- [x] 2.2 正确性 reward 适配函数复用阶段 3 的 `score_answer_correctness`。
- [x] 2.3 格式 reward 适配函数复用阶段 3 的 `score_answer_format`。
- [x] 2.4 添加本地硬编码检查，确认 reward 适配函数返回长度和分数符合预期。

## 3. 最小训练脚本

- [x] 3.1 新增 `scripts/run_minimal_grpo_training.py` 作为阶段 4 入口。
- [x] 3.2 支持 `--help` 展示模型名、split、训练样本数、最大步数、输出目录、缓存目录和 dry-run 参数。
- [x] 3.3 支持 `--dry-run`，只检查依赖、数据、prompt、reward 和配置，不下载模型、不训练。
- [x] 3.4 创建 TRL `GRPOConfig`，默认使用很小训练步数、batch size、generation 数和日志间隔。
- [x] 3.5 创建 PEFT `LoraConfig`，真实训练时通过 `GRPOTrainer` 使用 LoRA。
- [x] 3.6 真实训练路径创建 `GRPOTrainer`，传入模型、训练数据、reward 函数、GRPO 配置和 LoRA 配置。
- [x] 3.7 真实训练完成后保存 LoRA adapter，并打印输出目录。
- [x] 3.8 训练失败时输出清楚提示，区分模型下载、设备内存、数据准备和 TRL API 问题。

## 4. 输出与忽略规则

- [x] 4.1 将默认训练输出目录设为 `outputs/stage-4-minimal-grpo/`。
- [x] 4.2 更新 `.gitignore`，确保 `outputs/` 不进入 git。
- [x] 4.3 dry-run 或真实训练打印配置摘要，包括模型名、split、训练样本数、最大步数、缓存目录和输出目录。

## 5. 文档更新

- [x] 5.1 更新 `README.md`，加入阶段 4 dry-run 命令、真实训练命令和预期输出。
- [x] 5.2 更新 `plan.md`，将当前阶段推进到最小 GRPO 训练，并保留阶段 5 独立评估作为后续目标。
- [x] 5.3 更新 `AGENTS.md`，说明阶段 4 可以下载小模型并训练，但仍禁止 vLLM、多卡、wandb、Math-Verify 和正式评估扩展。
- [x] 5.4 文档说明当前机器可能只有 CPU，真实训练可能很慢，应先运行 dry-run。

## 6. 验证

- [x] 6.1 运行 Python 编译检查，确认新增模块和脚本语法正确。
- [x] 6.2 运行阶段 2 答案抽取离线检查。
- [x] 6.3 运行阶段 3 reward 离线检查。
- [x] 6.4 运行阶段 4 dry-run，确认不下载模型且预检通过。
- [x] 6.5 如本机网络和资源允许，运行极小真实训练并确认输出目录中保存 LoRA adapter；当前机器未检测到 CUDA/MPS，且模型下载需要网络，因此本轮不运行真实训练。
- [x] 6.6 运行 `openspec validate stage-4-minimal-grpo-training --strict`。
