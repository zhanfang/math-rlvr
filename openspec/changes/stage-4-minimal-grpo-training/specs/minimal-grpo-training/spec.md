## ADDED Requirements

### Requirement: 阶段 4 训练入口
系统 SHALL 提供一个阶段 4 最小 GRPO 训练入口，用于在单机上运行小模型、小数据和小步数的训练闭环。

#### Scenario: 查看训练命令帮助
- **WHEN** 学习者运行阶段 4 训练脚本并请求帮助
- **THEN** 系统展示模型名、split、训练样本数、最大训练步数、输出目录、缓存目录和 dry-run 参数

#### Scenario: 默认配置保持小规模
- **WHEN** 学习者运行阶段 4 训练脚本且不提供规模参数
- **THEN** 系统 MUST 使用很小的训练样本数和很小的最大训练步数
- **AND** 默认配置 MUST 面向学习闭环验证，而不是面向效果优化

### Requirement: Dry-run 预检
系统 MUST 提供 dry-run 模式，用于验证阶段 4 训练准备工作且不下载模型权重或执行训练。

#### Scenario: 运行 dry-run
- **WHEN** 学习者运行阶段 4 训练脚本并启用 dry-run
- **THEN** 系统检查 TRL、Transformers、PEFT、Torch、GSM8K 数据读取、prompt 构造、标准答案字段和 reward 函数
- **AND** 系统打印将使用的训练配置摘要
- **AND** 系统不加载模型权重、不下载模型权重、不执行训练、不保存 LoRA adapter

#### Scenario: 数据或 reward 预检失败
- **WHEN** dry-run 无法读取 GSM8K 样例或 reward 检查失败
- **THEN** 系统以非零退出码退出
- **AND** 输出可操作的错误信息，提示先完成数据初始化或阶段 3 reward 检查

### Requirement: GSM8K GRPO 数据准备
系统 SHALL 将 GSM8K 小子集转换为 GRPO 训练可用的数据记录。

#### Scenario: 构造训练记录
- **WHEN** 系统读取一条 GSM8K 样例
- **THEN** 系统生成包含 `prompt` 和 `answer` 字段的训练记录
- **AND** `answer` 字段 MUST 使用阶段 2 的 GSM8K 标准答案抽取结果

#### Scenario: Prompt 包含输出格式要求
- **WHEN** 系统构造 GRPO prompt
- **THEN** prompt MUST 要求模型使用 `<reasoning>...</reasoning><answer>...</answer>` 输出格式
- **AND** prompt MUST 包含原始数学问题文本

#### Scenario: 限制训练样本数
- **WHEN** 学习者提供训练样本数量参数
- **THEN** 系统 MUST 只取不超过该数量的 GSM8K 样例

### Requirement: GRPO reward 接入
系统 SHALL 提供 TRL `GRPOTrainer` 可调用的 reward 适配函数，并复用阶段 3 reward 逻辑。

#### Scenario: 正确性 reward 适配
- **WHEN** GRPOTrainer 将 completions 和标准答案传给正确性 reward 适配函数
- **THEN** 适配函数返回与 completions 数量一致的浮点 reward 列表
- **AND** 每个分数 MUST 基于阶段 3 的正确性 reward 计算

#### Scenario: 格式 reward 适配
- **WHEN** GRPOTrainer 将 completions 传给格式 reward 适配函数
- **THEN** 适配函数返回与 completions 数量一致的浮点 reward 列表
- **AND** 每个分数 MUST 基于阶段 3 的格式 reward 计算

#### Scenario: Reward 可解释性
- **WHEN** 学习者阅读训练脚本或文档
- **THEN** 系统 MUST 明确说明正确性 reward 和格式 reward 各自奖励什么行为

### Requirement: 最小 GRPO 训练执行
系统 SHALL 使用 TRL `GRPOTrainer` 执行最小 GRPO 训练，并使用 PEFT LoRA 降低训练产物和资源需求。

#### Scenario: 创建 GRPOTrainer
- **WHEN** 学习者运行真实训练命令
- **THEN** 系统创建 `GRPOTrainer`
- **AND** 系统向 trainer 传入模型、GRPO 配置、训练数据、reward 函数和 LoRA 配置

#### Scenario: 运行有限步数训练
- **WHEN** 训练开始
- **THEN** 系统 MUST 按配置中的最大训练步数停止
- **AND** 日志中 SHOULD 能看到 reward 相关信息或训练进度信息

#### Scenario: 资源不可用时失败清楚
- **WHEN** 模型下载、内存、设备或训练 API 导致训练失败
- **THEN** 系统 MUST 输出清楚的失败原因或排查方向
- **AND** 文档 MUST 提醒先运行 dry-run 来区分准备问题和资源问题

### Requirement: 训练输出保存
系统 SHALL 将阶段 4 训练产物保存到 git 忽略的输出目录。

#### Scenario: 保存 LoRA adapter
- **WHEN** 真实训练成功完成
- **THEN** 系统保存 LoRA adapter 到输出目录
- **AND** 输出目录路径会在命令输出中显示

#### Scenario: 保存配置摘要
- **WHEN** dry-run 或真实训练运行
- **THEN** 系统保存或打印足够复现实验的配置摘要，包括模型名、数据 split、训练样本数、最大步数和输出目录

#### Scenario: 输出目录不进入 git
- **WHEN** 训练输出目录存在
- **THEN** `.gitignore` MUST 忽略训练输出目录

### Requirement: 阶段 4 边界
系统 SHALL 保持阶段 4 为最小单机训练闭环，不扩展为正式评估或分布式训练平台。

#### Scenario: 运行阶段 4 默认命令
- **WHEN** 学习者运行阶段 4 默认命令
- **THEN** 系统不使用 vLLM、Ray、DeepSpeed、FSDP、verl、OpenRLHF 或多机多卡
- **AND** 系统不运行正式评估集打分、不接入 wandb、不引入 Math-Verify

#### Scenario: 文档描述阶段 4
- **WHEN** 学习者阅读 README 或计划文档
- **THEN** 文档明确说明阶段 4 的目标是最小 GRPO 训练闭环
- **AND** 文档把独立评估、失败分析、复杂数学验证和训练提速标记为后续阶段
