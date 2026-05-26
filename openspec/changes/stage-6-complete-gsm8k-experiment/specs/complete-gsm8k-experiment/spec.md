## ADDED Requirements

### Requirement: 更大规模 GSM8K 训练实验
系统 SHALL 提供一轮比阶段 5 更接近 `plan.md` Milestone 2 的 GSM8K 训练实验配置。

#### Scenario: 运行阶段 6 推荐训练配置
- **WHEN** 学习者运行阶段 6 的推荐训练命令
- **THEN** 系统使用明显大于阶段 5 默认值的训练样本数或训练步数
- **AND** 推荐配置面向千样本级、数百步级的 GSM8K 小实验

#### Scenario: 保留显式参数覆盖能力
- **WHEN** 学习者需要根据机器资源调整实验规模
- **THEN** 系统仍允许覆盖训练样本数、最大步数、completion 长度和 batch size
- **AND** 系统不会把 GSM8K 全集训练设为强制默认行为

### Requirement: 独立 Held-out 评估
系统 SHALL 提供独立于训练脚本的 held-out 评估入口，用于比较 base model 与 RLVR adapter。

#### Scenario: 评估 base model
- **WHEN** 学习者运行评估脚本并只指定 base model
- **THEN** 系统在 held-out GSM8K 样例上生成结果
- **AND** 系统输出 `pass@1` 或请求的 `pass@k` 指标

#### Scenario: 评估 adapter model
- **WHEN** 学习者运行评估脚本并提供 LoRA adapter 路径
- **THEN** 系统在同一批 held-out GSM8K 样例上评估 adapter
- **AND** 系统输出与 base model 可对齐的指标和结果文件

### Requirement: Case 级结果导出
系统 SHALL 导出 case 级 jsonl 结果，以支持独立复盘和失败分析。

#### Scenario: 导出单条评估结果
- **WHEN** 系统完成一条 held-out 样例评估
- **THEN** 系统保存题目、参考答案、原始 completion、抽取答案和正确性结果
- **AND** 系统保留模型来源与 case 标识

#### Scenario: 保存 jsonl 文件
- **WHEN** 一轮评估运行结束
- **THEN** 系统将全部 case 明细保存为 jsonl
- **AND** 输出目录位于 git 忽略的阶段 6 实验目录下

### Requirement: Base 与 Adapter 对比摘要
系统 SHALL 输出 base 与 adapter 的对比摘要，用于判断完整 GSM8K 小实验是否达到目标。

#### Scenario: 汇总 pass 指标
- **WHEN** base 与 adapter 都完成评估
- **THEN** 系统输出两侧的 `pass@1` 和请求的 `pass@k`
- **AND** 系统展示 adapter 相对 base 的差异

#### Scenario: 记录实验元数据
- **WHEN** 阶段 6 实验完成
- **THEN** 系统保存训练配置、训练耗时、评估样本数和结果路径
- **AND** 系统生成可复盘的实验摘要文件

### Requirement: 失败案例分类
系统 SHALL 提供基于规则的失败案例分类，以帮助学习者分析 held-out 结果。

#### Scenario: 识别常见失败类型
- **WHEN** 系统分析评估结果
- **THEN** 系统至少识别格式失败、答案抽取失败、算术错误和题意理解错误这类常见问题
- **AND** 系统支持查看每类问题的代表性样例

#### Scenario: 对应训练成功判据
- **WHEN** 学习者回看阶段 6 输出
- **THEN** 系统提供足够信息来判断是否满足“能独立 eval”和“能打开 jsonl 看失败案例”
- **AND** 系统不要求仅依赖训练 reward 进行结论判断
