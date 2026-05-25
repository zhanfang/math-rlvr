## Why

阶段 2 已经能读取 GSM8K 并抽取标准答案，下一步需要把“答案可验证”转化为可复用的 reward 函数。阶段 3 的目标是先在本地硬编码样例上验证 reward 逻辑，避免还没训练就把格式抽取、数值比较和正确性判断混在一起。

## What Changes

- 新增模型输出格式约定，使用 `<reasoning>...</reasoning><answer>...</answer>` 承载推理和最终答案。
- 新增模型答案抽取逻辑，从模型 completion 中提取 `<answer>` 内的最终答案。
- 新增轻量数值等价判断，支持整数、小数、分数和带逗号数字的基础比较。
- 新增正确性 reward：将模型答案与 GSM8K 标准答案进行数值等价比较。
- 新增格式 reward：检查模型输出是否符合阶段 3 约定格式。
- 新增离线检查脚本，用手写样例验证答案抽取、数值等价、正确性 reward 和格式 reward。
- 更新文档，说明阶段 3 的命令、验收标准和边界。

## Capabilities

### New Capabilities
- `reward-function-prototype`: 覆盖阶段 3 的模型答案抽取、数值等价判断、正确性 reward、格式 reward 和离线 reward 验证。

### Modified Capabilities
- None.

## Impact

- 可能新增 `src/` 下的 reward 模块，复用阶段 2 的 GSM8K 标准答案抽取逻辑。
- 可能新增 `scripts/` 下的阶段 3 离线检查脚本。
- 可能更新 `README.md`、`plan.md` 和 `AGENTS.md` 中的阶段说明。
- 不新增模型权重下载，不运行模型生成，不执行 GRPO 训练，不引入 Math-Verify、vLLM、Ray、DeepSpeed、FSDP、verl、OpenRLHF 或多 GPU 框架。
