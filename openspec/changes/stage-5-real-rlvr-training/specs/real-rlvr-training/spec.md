## ADDED Requirements

### Requirement: 阶段 5 真实 RLVR 训练入口
系统 SHALL 将阶段 5 定义为真实单机 RLVR 训练阶段，并在现有训练入口上支持比阶段 4 明显更大的真实训练。

#### Scenario: 查看训练命令帮助
- **WHEN** 学习者运行阶段 5 训练脚本并请求帮助
- **THEN** 系统展示模型路径、训练样本数、最大步数、输出目录和关键训练参数

#### Scenario: 运行阶段 5 默认训练
- **WHEN** 学习者运行阶段 5 默认训练命令
- **THEN** 系统执行真实 RLVR 训练
- **AND** 系统不再停留在仅用于联调的极小训练规模
- **AND** 系统保存阶段 5 的训练产物

### Requirement: 高吞吐默认参数
系统 SHALL 为阶段 5 提供更适合当前单机 MPS 环境的默认训练参数。

#### Scenario: 默认 batch size 提升
- **WHEN** 学习者使用阶段 5 默认参数运行训练
- **THEN** 系统 MUST 使用高于阶段 4 极小闭环的默认 batch size
- **AND** 默认值 MUST 以本机实测吞吐结果为依据

#### Scenario: 默认 completion 长度与日志频率优化
- **WHEN** 学习者使用阶段 5 默认参数运行训练
- **THEN** 系统 MUST 使用较短的默认 completion 长度和较低频的日志/保存间隔
- **AND** 默认值 MUST 优先减少无效开销并提升样本吞吐

### Requirement: 中等规模真实训练
系统 SHALL 在阶段 5 至少完成一轮中等规模真实训练，而不是只运行一步或两条样本的最小闭环。

#### Scenario: 中等规模训练样本数
- **WHEN** 学习者运行阶段 5 的中等规模训练
- **THEN** 系统 MUST 使用明显大于阶段 4 的训练样本数
- **AND** 训练样本数 MUST 至少达到百样本级别

#### Scenario: 中等规模训练步数
- **WHEN** 学习者运行阶段 5 的中等规模训练
- **THEN** 系统 MUST 使用明显大于阶段 4 的训练步数
- **AND** 训练步数 MUST 至少达到百步级别或同量级真实训练

### Requirement: 训练耗时与吞吐记录
系统 SHALL 在阶段 5 训练结束后输出和保存可用于估算更大规模训练的耗时信息。

#### Scenario: 输出训练耗时摘要
- **WHEN** 阶段 5 训练完成
- **THEN** 系统输出训练总时长、samples per second 和 steps per second
- **AND** 这些信息可用于估算更大规模训练所需时间

#### Scenario: 保存训练配置摘要
- **WHEN** 阶段 5 训练运行
- **THEN** 系统保存完整训练配置摘要
- **AND** 摘要中包含模型路径、训练样本数、最大步数、batch size 和 completion 长度

### Requirement: 阶段 5 训练产物保存
系统 SHALL 将阶段 5 的训练输出保存到与阶段 4 区分的 git 忽略目录。

#### Scenario: 保存阶段 5 adapter
- **WHEN** 阶段 5 真实训练成功完成
- **THEN** 系统保存 LoRA adapter
- **AND** adapter 路径会在命令输出中显示

#### Scenario: 阶段 5 输出目录不进入 git
- **WHEN** 阶段 5 训练输出目录存在
- **THEN** `.gitignore` MUST 忽略该目录
- **AND** 输出目录 MUST 与阶段 4 最小闭环输出区分开

### Requirement: 阶段 5 边界
系统 SHALL 保持阶段 5 为真实单机 RLVR 训练阶段，不扩展为全集强制训练或分布式训练平台。

#### Scenario: 运行阶段 5 默认流程
- **WHEN** 学习者运行阶段 5 默认命令
- **THEN** 系统保持单机、LoRA、本地缓存模型与数据的训练方式
- **AND** 系统不接入 wandb、vLLM、Math-Verify、Ray、DeepSpeed、FSDP、verl、OpenRLHF 或多机多卡

#### Scenario: 阶段 5 不强制全集训练
- **WHEN** 学习者阅读阶段 5 文档或任务
- **THEN** 文档明确说明阶段 5 的必达目标是中等规模真实训练
- **AND** 文档不将 GSM8K 全集训练定义为阶段 5 的必达验收条件
