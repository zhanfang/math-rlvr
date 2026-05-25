## ADDED Requirements

### Requirement: 模型输出格式约定
系统 SHALL 在阶段 3 规定模型 completion 的目标格式为 `<reasoning>...</reasoning><answer>...</answer>`，其中 `<answer>` 内文本用于最终答案比较。

#### Scenario: 完整格式输出
- **WHEN** completion 同时包含非空 `<reasoning>` 段和非空 `<answer>` 段
- **THEN** 系统将该 completion 视为符合阶段 3 输出格式

#### Scenario: 缺少答案标签
- **WHEN** completion 不包含 `<answer>...</answer>`
- **THEN** 系统 MUST 将该 completion 视为不符合阶段 3 输出格式

#### Scenario: 答案标签为空
- **WHEN** completion 包含 `<answer></answer>` 或答案标签中只有空白
- **THEN** 系统 MUST 将该 completion 视为不符合阶段 3 输出格式

### Requirement: 模型答案抽取
系统 SHALL 提供可复用的模型答案抽取函数，从 completion 的 `<answer>` 标签中提取最终答案文本。

#### Scenario: 抽取标签内答案
- **WHEN** completion 包含 `<answer>42</answer>`
- **THEN** 答案抽取函数返回 `42`

#### Scenario: 清理标签内空白
- **WHEN** completion 包含 `<answer>  1,000  </answer>`
- **THEN** 答案抽取函数返回 `1,000`

#### Scenario: 缺少答案标签时返回空字符串
- **WHEN** completion 不包含完整 `<answer>...</answer>` 标签
- **THEN** 答案抽取函数返回空字符串

### Requirement: 轻量数值等价判断
系统 SHALL 提供轻量数值等价判断，用于比较模型答案和标准答案的基础数值等价关系。

#### Scenario: 整数等价
- **WHEN** 比较 `42` 和 `42`
- **THEN** 数值等价判断返回 true

#### Scenario: 带逗号数字等价
- **WHEN** 比较 `1,000` 和 `1000`
- **THEN** 数值等价判断返回 true

#### Scenario: 分数和小数等价
- **WHEN** 比较 `1/2` 和 `0.5`
- **THEN** 数值等价判断返回 true

#### Scenario: 不等价数值
- **WHEN** 比较 `41` 和 `42`
- **THEN** 数值等价判断返回 false

#### Scenario: 无法解析时回退到清理字符串比较
- **WHEN** 两个答案都无法解析为基础数值
- **THEN** 系统 MUST 比较清理后的字符串是否完全相同

### Requirement: 正确性 reward
系统 SHALL 提供正确性 reward，根据模型答案与 GSM8K 标准答案的轻量数值等价结果返回分数。

#### Scenario: 正确答案获得满分
- **WHEN** completion 的 `<answer>` 内容与标准答案数值等价
- **THEN** 正确性 reward 返回 `1.0`

#### Scenario: 错误答案获得零分
- **WHEN** completion 的 `<answer>` 内容与标准答案不等价
- **THEN** 正确性 reward 返回 `0.0`

#### Scenario: 缺少模型答案获得零分
- **WHEN** completion 缺少可抽取的 `<answer>` 内容
- **THEN** 正确性 reward 返回 `0.0`

### Requirement: 格式 reward
系统 SHALL 提供格式 reward，根据 completion 是否符合阶段 3 输出格式返回分数。

#### Scenario: 完整格式获得满分
- **WHEN** completion 同时包含非空 `<reasoning>` 段和非空 `<answer>` 段
- **THEN** 格式 reward 返回 `1.0`

#### Scenario: 不完整格式获得零分
- **WHEN** completion 缺少 `<reasoning>` 段、缺少 `<answer>` 段或任一段为空
- **THEN** 格式 reward 返回 `0.0`

### Requirement: 离线 reward 验证
系统 MUST 提供不依赖网络、数据集下载或模型权重下载的阶段 3 检查方式，用于验证 reward 函数行为。

#### Scenario: 运行离线 reward 检查
- **WHEN** 学习者运行阶段 3 reward 检查命令
- **THEN** 系统使用本地硬编码样例验证模型答案抽取、数值等价、正确性 reward 和格式 reward
- **AND** 检查不下载数据集、不下载模型权重、不运行模型生成或训练

#### Scenario: 离线样例覆盖边界
- **WHEN** 阶段 3 reward 检查命令运行
- **THEN** 样例 MUST 覆盖正确答案、错误答案、分数与小数等价、逗号数字等价、缺失标签和格式不完整

### Requirement: 阶段 3 边界
系统 SHALL 保持阶段 3 为 reward 函数原型验证，不进入模型训练或正式评估。

#### Scenario: 运行阶段 3 默认命令
- **WHEN** 学习者运行阶段 3 的默认命令
- **THEN** 系统不下载模型权重
- **AND** 系统不执行模型生成、GRPO 训练、LoRA 保存或评估打分

#### Scenario: 文档描述阶段 3
- **WHEN** 学习者阅读 README 或计划文档
- **THEN** 文档明确说明阶段 3 的目标是模型答案抽取、数值等价、正确性 reward、格式 reward 和离线样例验证
- **AND** 文档把 GRPO 训练、模型下载、Math-Verify 和独立评估标记为后续阶段
