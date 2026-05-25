## Why

阶段 1 已经完成本地 `.venv` 环境安装和 smoke check，下一步需要确认项目能安全接触真实数学数据。阶段 2 的目标是先理解 GSM8K 数据结构、答案格式和抽取规则，而不是直接进入训练。

## What Changes

- 新增一个轻量的数据检查入口，用于加载 GSM8K 小子集并打印样例结构。
- 新增 GSM8K 标准答案抽取逻辑，将 `#### 42` 形式的最终答案抽成可比较的字符串。
- 新增基础数据检查或单元测试，覆盖常见答案抽取格式。
- 新增一个只读的 baseline inspection 入口，用于可选地观察模型前数据准备状态；默认不下载模型权重、不训练、不做 GRPO。
- 更新文档，说明阶段 2 的目标、命令、验收标准和边界。

## Capabilities

### New Capabilities
- `gsm8k-data-baseline`: 覆盖 GSM8K 小子集加载、样例检查、标准答案抽取和阶段 2 的轻量 baseline inspection。

### Modified Capabilities
- None.

## Impact

- 可能新增 `src/` 下的数据处理模块和/或检查脚本。
- 可能新增 `scripts/` 下的阶段 2 运行脚本。
- 可能新增本地测试文件，用于验证答案抽取逻辑。
- 需要使用阶段 1 已安装的 `datasets` 依赖；如果首次运行加载 GSM8K，可能需要联网下载数据集缓存。
- 不引入模型下载、训练脚本、reward 函数、Math-Verify、vLLM 或多卡框架。
