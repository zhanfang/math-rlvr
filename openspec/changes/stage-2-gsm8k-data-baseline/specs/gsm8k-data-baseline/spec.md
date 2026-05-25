## ADDED Requirements

### Requirement: GSM8K 小子集加载
系统 SHALL 提供一个阶段 2 数据检查入口，用于加载 GSM8K 的小子集并展示样例结构。

#### Scenario: 默认加载少量样例
- **WHEN** 学习者运行阶段 2 数据检查命令且不提供自定义数量
- **THEN** 系统加载 GSM8K 的默认 split 中的少量样例
- **AND** 输出样例数量、字段名、问题文本、原始答案和抽取后的最终答案

#### Scenario: 指定 split 和数量
- **WHEN** 学习者提供 split 和 limit 参数
- **THEN** 系统按照指定 split 和 limit 加载 GSM8K 样例
- **AND** limit MUST 控制展示数量，避免默认打印完整数据集

### Requirement: GSM8K 标准答案抽取
系统 SHALL 提供可复用的 GSM8K 标准答案抽取函数，将 answer 字段中的最终答案提取为干净字符串。

#### Scenario: 抽取带最终标记的答案
- **WHEN** 输入答案文本包含 `#### 42`
- **THEN** 抽取函数返回 `42`

#### Scenario: 清理数字格式
- **WHEN** 输入最终答案包含逗号或首尾空白
- **THEN** 抽取函数返回去除逗号和首尾空白后的答案字符串

#### Scenario: 缺少最终标记
- **WHEN** 输入答案文本不包含 `####`
- **THEN** 抽取函数返回清理后的完整答案文本

### Requirement: 离线抽取验证
系统 MUST 提供不依赖网络或数据集下载的检查方式，用于验证 GSM8K 标准答案抽取逻辑。

#### Scenario: 运行离线测试
- **WHEN** 学习者运行答案抽取测试或等价检查命令
- **THEN** 系统使用本地硬编码样例验证常见 GSM8K 答案格式
- **AND** 测试不下载数据集、不下载模型权重

### Requirement: 阶段 2 边界
系统 SHALL 保持阶段 2 为数据理解和轻量 baseline inspection，不进入模型训练或奖励设计。

#### Scenario: 运行阶段 2 命令
- **WHEN** 学习者运行阶段 2 的默认命令
- **THEN** 系统不下载模型权重
- **AND** 系统不执行模型生成、GRPO 训练、reward 计算或评估打分

#### Scenario: 文档描述阶段 2
- **WHEN** 学习者阅读 README 或计划文档
- **THEN** 文档明确说明阶段 2 的目标是 GSM8K 数据加载、样例观察和标准答案抽取
- **AND** 文档把模型生成、reward、训练和正式评估标记为后续阶段
