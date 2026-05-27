# math-rlvr

单机起步、面向数学题的 RLVR（Reinforcement Learning with Verifiable Rewards）学习项目。

## 一、项目思想

> 用一个开源小模型，在 GSM8K 上跑通：
> 模型生成推理与答案 → 程序抽取答案 → 自动验证正确性 → 用 GRPO 更新模型。

核心思路：

- **可验证奖励**：数学题答案可程序判定，无需训练 reward model，也不依赖人工标注。
- **GRPO 而非 PPO**：对同一 prompt 采样多个 completion，按组内 reward 差异更新策略，省掉 critic。
- **格式 + 正确性双 reward**：正确性 reward 占主导，格式 reward 辅助；避免模型只学格式不学做题。
- **LoRA 微调**：在小模型 + 单卡条件下保持显存可控，便于反复实验。
- **分阶段递进**：从环境到云上全量训练，每个阶段只引入一个新关注点。

固定输出格式：

```xml
<reasoning>
推理过程
</reasoning>
<answer>
最终答案
</answer>
```

## 二、整体流程

```text
Stage 1  环境就绪  → .venv + requirements.txt
Stage 2  数据基线  → GSM8K 缓存、抽取 #### 答案
Stage 3  奖励原型  → correctness / format reward 离线验证
Stage 4  最小 GRPO → 双 reward 烟雾测试，打通 TRL 链路
Stage 5  真实 RLVR → 单机吞吐导向的 GRPO 训练
Stage 6  完整实验  → GSM8K 训练 + 独立 held-out 评估
Stage 7  云上全量  → RunPod L40S + vLLM，可选 SFT warm-start
```

每个阶段都有：

- 配置入口在 [src/training/profiles.py](file:///Users/bytedance/Documents/code/github/math-rlvr/src/training/profiles.py)
- 可执行脚本在 [scripts/run/](file:///Users/bytedance/Documents/code/github/math-rlvr/scripts/run)
- 离线检查在 [scripts/check/](file:///Users/bytedance/Documents/code/github/math-rlvr/scripts/check)

## 三、目录结构

- [src/training/](file:///Users/bytedance/Documents/code/github/math-rlvr/src/training)：prompt、profile、dataset、reward adapter、runtime、SFT 等训练侧公共模块。
- [src/answer_rewards.py](file:///Users/bytedance/Documents/code/github/math-rlvr/src/answer_rewards.py)：答案抽取与 correctness / format reward。
- [src/gsm8k_dataset.py](file:///Users/bytedance/Documents/code/github/math-rlvr/src/gsm8k_dataset.py)：GSM8K 加载与 `####` 答案抽取。
- [src/minimal_grpo_training.py](file:///Users/bytedance/Documents/code/github/math-rlvr/src/minimal_grpo_training.py)：训练入口的兼容门面。
- [scripts/run/](file:///Users/bytedance/Documents/code/github/math-rlvr/scripts/run)：训练 / 评估 / SFT / smoke test 入口。
- [scripts/check/](file:///Users/bytedance/Documents/code/github/math-rlvr/scripts/check)：环境与离线 reward 检查。
- [scripts/runpod/](file:///Users/bytedance/Documents/code/github/math-rlvr/scripts/runpod)：Stage 7 云上 preflight / smoke / 全量启动脚本。
- [docker/runpod-stage7/](file:///Users/bytedance/Documents/code/github/math-rlvr/docker/runpod-stage7)：Stage 7 训练镜像。

## 四、环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

.venv/bin/python scripts/check/smoke_check_env.py
```

要求 Python ≥ 3.10，`torch / transformers / datasets / accelerate / peft / trl` 可正常导入。Stage 1～6 不强依赖 CUDA/vLLM；Stage 7 支持两条云上链路：RunPod L40S + vLLM，或 Merlin A10 + no-vLLM。

## 五、各阶段速查

### Stage 2：GSM8K 数据

```bash
.venv/bin/python scripts/check/verify_gsm8k_answers.py
.venv/bin/python scripts/data/cache_gsm8k_dataset.py
.venv/bin/python scripts/data/inspect_gsm8k_dataset.py --split train --limit 3
```

数据缓存在 `data/hf_datasets/`（git ignored）。

### Stage 3：奖励函数

```bash
.venv/bin/python scripts/check/verify_answer_rewards.py
```

校验 `<reasoning>...<answer>...` 解析、数值等价比较、格式 reward 的边界。

### Stage 4：最小 GRPO

```bash
# 不下载模型权重的纯配置 dry-run
.venv/bin/python scripts/run/stage4_dual_reward_smoke_test.py --dry-run

# 真实双 reward 烟雾训练（需要本地或可访问的小模型）
.venv/bin/python scripts/run/stage4_dual_reward_smoke_test.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct
```

目标：在真实 GRPO 日志里同时看到 correctness 与 format reward 为正。

### Stage 5：真实 RLVR

```bash
.venv/bin/python scripts/run/minimal_grpo_training.py
```

吞吐导向的默认参数，输出至 `outputs/stage-5-real-rlvr/`。

### Stage 6：完整 GSM8K 实验

```bash
.venv/bin/python scripts/run/minimal_grpo_training.py \
  --experiment-profile stage6-gsm8k-experiment \
  --model-name <local-or-hub-path>

.venv/bin/python scripts/run/gsm8k_eval.py \
  --base-model <local-or-hub-path> \
  --adapter-path outputs/stage-6-gsm8k-experiment/train/adapter \
  --mode both --limit 50 --k 1 \
  --train-run-config outputs/stage-6-gsm8k-experiment/train/run_config.json \
  --train-metrics outputs/stage-6-gsm8k-experiment/train/train_metrics.json
```

产出：训练侧 `run_config.json` / `train_metrics.json` / `adapter/`；评估侧 `eval/{base,adapter}/cases.jsonl`、`eval/compare/summary.json`、`eval/experiment_summary.json`。

### Stage 7：云上全量训练（RunPod L40S 48GB）

```bash
# 本地仅做配置 dry-run
.venv/bin/python scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-full-gsm8k-training --dry-run --train-limit 8
.venv/bin/python scripts/run/gsm8k_sft.py --dry-run --train-limit 8

# 镜像由 GitHub Actions 或 x86 builder 构建（Apple Silicon 不支持）
scripts/runpod/build_image.sh <registry>/math-rlvr-stage7:cu128-vllm018

# Pod 上分阶段执行
bash scripts/runpod/preflight.sh
bash scripts/runpod/smoke.sh
nohup bash scripts/runpod/run_stage7_full.sh \
  > outputs/stage-7-full-gsm8k-training/logs/full_pipeline.log 2>&1 &
```

可选 SFT warm-start 后接 GRPO：

```bash
.venv/bin/python scripts/run/gsm8k_sft.py --output-dir outputs/stage-7-full-gsm8k-training/sft
.venv/bin/python scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-full-gsm8k-training \
  --initial-adapter-path outputs/stage-7-full-gsm8k-training/sft/adapter \
  --use-vllm --vllm-mode colocate
```

双卡时 GPU 0 训练、GPU 1 跑 `trl vllm-serve`。

### Stage 7：云上训练（Merlin Devbox A10 24GB）

这条链路固定使用：

- 模型：开发机共享目录下的 `Qwen2.5-1.5B-Instruct`
- 训练：`stage7-merlin-a10-qwen15b-no-vllm`
- 环境：项目内 `.venv` + [requirements-merlin-stage7.txt](file:///Users/bytedance/Documents/code/github/math-rlvr/requirements-merlin-stage7.txt)
- 生成：**不启用 vLLM**，避免 Merlin 上的 `trl / vllm / torch` 兼容问题

第一次在开发机上执行：

```bash
cd /mlx_devbox/users/zhanfang.128/playground/math-rlvr
bash scripts/check/merlin_stage7_preflight.sh
```

约定目录：

- 项目：`/mlx_devbox/users/<user>/playground/math-rlvr`
- 模型：`/mlx_devbox/users/<user>/playground/models/Qwen2.5-1.5B-Instruct`
- HF 缓存：`/mlx_devbox/users/<user>/playground/hf_home`

50 步验证版：

```bash
cd /mlx_devbox/users/zhanfang.128/playground/math-rlvr
RUN_NAME=train-50 TRAIN_LIMIT=256 MAX_STEPS=50 LOGGING_STEPS=5 SAVE_STEPS=25 \
  bash scripts/run/merlin_stage7_qwen15b_no_vllm.sh
```

400 步正式版：

```bash
cd /mlx_devbox/users/zhanfang.128/playground/math-rlvr
RUN_NAME=train MAX_STEPS=400 TRAIN_LIMIT=7473 \
  bash scripts/run/merlin_stage7_qwen15b_no_vllm.sh
```

训练完成后会自动跑评估，产出位于：

- 训练：`outputs/stage-7-merlin-a10-qwen15b/<run_name>/`
- 评估：`outputs/stage-7-merlin-a10-qwen15b/<run_name>/eval/`

## 六、训练成功的判据

- correctness reward 能从近 0 稳步上升，不被 format reward 压过。
- 训练日志里的 `frac_reward_zero_std` 不持续偏高，说明 GRPO 有学习信号。
- held-out `pass@1` 优于 base，并能保存可复跑的 LoRA adapter。
- 评估 `cases.jsonl` 可被打开，能定位失败样例与 reward hacking 来源。

## 七、后续方向

- Math-Verify / `\boxed{}`：支持 LaTeX、分数、集合、区间等强等价判定。
- 更大数据：MATH、DeepMath、OpenR1-Math、自建题库。
- 多卡框架：verl / OpenRLHF + Ray + vLLM + DeepSpeed/FSDP，迁移到 1.5B / 3B / 7B。
