可以。你的判断是对的：**如果最终目标是实现“数学题 RLVR 验证 + 训练”能力，就不应该继续围绕 nanoGPT 改造**。nanoGPT 适合学 Transformer、预训练、SFT loss，但不适合做 RLVR，因为 RLVR 需要高效 rollout、多样本采样、奖励函数、LoRA/PEFT、分布式/显存优化、日志监控和稳定的生成训练闭环。

下面给你一套可以直接落地的工程方案。目标是：

> 用一个开源小模型，在 GSM8K 数学题上训练 RLVR：
> 模型生成推理与答案 → 程序抽取答案 → 自动验证正确性 → 用 GRPO 更新模型。

DeepSeek-R1 这类工作证明，纯 RL 可以在可验证任务上激发推理能力，尤其是数学、代码、STEM 等有明确验证信号的任务；这正是 RLVR 的适用场景。([arXiv][1])

---

# 一、最终技术路线

你要实现的不是“从零写 RL 框架”，而是先掌握一个可跑通、可扩展的 RLVR 工程闭环。

推荐路线是：

```text
阶段 1：TRL + GRPOTrainer + GSM8K + 自定义 reward
阶段 2：加入 Math-Verify，支持 LaTeX / 分数 / 符号表达式验证
阶段 3：换成 MATH / DeepMath / 自建数学题数据
阶段 4：用 vLLM 提速 rollout
阶段 5：多卡后迁移到 verl 或 OpenRLHF
```

为什么用 GRPO 而不是 PPO？

因为数学 RLVR 通常不需要 reward model，也不一定需要 critic/value model。GRPO 会对同一个 prompt 采样多个 completion，然后根据组内 reward 差异更新模型。TRL 当前提供 `GRPOTrainer`，支持自定义 reward function、多 reward function、额外数据列传入 reward function，并且可以用 vLLM 加速生成。([Hugging Face][2])

---

# 二、工程选型

## 第一版：单机学习版

建议先用：

```text
训练框架：Hugging Face TRL
算法：GRPO
模型：Qwen/Qwen2.5-0.5B-Instruct
数据集：openai/gsm8k
奖励：答案正确性 reward + 格式 reward
微调方式：LoRA
验证方式：正则抽取 + 数值等价比较
```

GSM8K 是约 8.5K 道小学数学文字题数据集，用来训练和评估多步数学推理非常合适。([Hugging Face][3])

## 第二版：更强验证版

加入：

```text
math-verify
latex2sympy2_extended
\boxed{} 答案抽取
分数、小数、集合、矩阵、区间、符号表达式等价判断
```

`math-verify` 是 Hugging Face 的数学答案验证库，提供 `parse` 和 `verify`，能处理 LaTeX、普通数学表达式、集合、区间、矩阵、百分数、符号等价等问题；PyPI 页面显示它要求 Python >= 3.10，当前可通过 `pip install math-verify` 安装。([GitHub][4])

## 第三版：多卡规模化版

等你单机跑通后，再考虑：

```text
verl
OpenRLHF
Ray
vLLM server
DeepSpeed ZeRO
FSDP
```

`verl` 是面向大语言模型 RL 后训练的框架，支持 FSDP、Megatron-LM、vLLM、SGLang 等组件；OpenRLHF 也支持 PPO、REINFORCE++、GRPO、RLOO，并提供 Ray + vLLM 的训练架构。([GitHub][5])

---

# 三、项目目录结构

新建一个项目，不要继续在 nanoGPT 里改：

```text
math-rlvr/
  requirements.txt
  README.md

  src/
    reward_math.py
    data_gsm8k.py
    train_grpo_gsm8k.py
    eval_math.py

  scripts/
    train_gsm8k_0p5b.sh
    eval_gsm8k.sh

  outputs/
  reports/
```

---

# 四、环境安装

建议 Python 3.10 或 3.11。

```bash
conda create -n math-rlvr python=3.11 -y
conda activate math-rlvr

pip install -U torch
pip install -U transformers datasets accelerate peft trl
pip install -U "math-verify[antlr4_13_2]"
pip install -U wandb
```

如果你后面要用 vLLM 加速 rollout，可以额外安装：

```bash
pip install -U "trl[vllm]"
```

TRL 文档说明，GRPO 训练中 generation 往往是瓶颈，可以用 vLLM 加速；它支持 colocate 和 server 两种模式。([Hugging Face][2])

---

# 五、核心概念：RLVR 在数学题上到底训练什么？

每条训练样本是：

```json
{
  "prompt": "数学题",
  "answer": "标准答案"
}
```

训练时不是给模型标准推理过程，而是让模型自己生成多个答案：

```text
问题 q
  → completion 1 → reward 0
  → completion 2 → reward 1
  → completion 3 → reward 0
  → completion 4 → reward 1
```

然后 GRPO 根据同一题内多个 completion 的 reward 差异，增强高 reward 生成路径，削弱低 reward 生成路径。

你要让模型输出固定格式：

```xml
<reasoning>
这里写推理过程
</reasoning>
<answer>
最终答案
</answer>
```

reward 分两部分：

```text
正确性 reward：答案是否等于标准答案，权重大
格式 reward：是否按要求输出标签，权重小
```

一定要让**正确性 reward 占主导**，否则模型可能只学会输出漂亮格式，而不是学会做题。

---

# 六、第一版：可直接运行的训练代码

创建文件：

```text
src/train_grpo_gsm8k.py
```

内容如下：

```python
import argparse
import re
from fractions import Fraction
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = """You are a careful math reasoning assistant.

Solve the problem step by step.

You must respond in exactly this format:

<reasoning>
your reasoning here
</reasoning>
<answer>
final answer only
</answer>

Rules:
- Put only the final numeric answer inside <answer>.
- Do not include units or explanations inside <answer>.
- If the answer is a fraction, use a/b format.
"""


def extract_hash_answer(gsm8k_answer: str) -> str:
    """
    GSM8K answers usually contain a final marker like:
    ... #### 42
    """
    if "####" in gsm8k_answer:
        return gsm8k_answer.split("####")[-1].strip().replace(",", "")
    return gsm8k_answer.strip().replace(",", "")


def completion_to_text(completion: Any) -> str:
    """
    TRL may pass completions as:
    - plain strings for standard format
    - list[dict] for conversational format
    This helper makes the reward functions robust.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        return completion.get("content", "")

    if isinstance(completion, list):
        if not completion:
            return ""
        if isinstance(completion[0], dict):
            return completion[0].get("content", "")
        if isinstance(completion[-1], dict):
            return completion[-1].get("content", "")
        return str(completion)

    return str(completion)


def extract_xml_answer(text: str) -> str:
    """
    Extract answer from:
    <answer>
    42
    </answer>

    Fallbacks:
    - \\boxed{42}
    - last number-like expression
    """
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")

    boxed_match = re.search(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "")

    # Last fallback: use the last simple number / decimal / fraction.
    candidates = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    if candidates:
        return candidates[-1].strip()

    return ""


def to_fraction(x: str):
    """
    Convert common numeric strings to Fraction for exact comparison.
    Examples:
    "3", "-2", "1/2", "0.5"
    """
    x = x.strip().replace(",", "")
    x = x.replace("$", "")
    x = x.replace("\\", "")

    # Remove simple trailing punctuation.
    x = x.strip(". ")

    if not x:
        return None

    try:
        return Fraction(x)
    except Exception:
        return None


def numeric_equal(pred: str, gold: str) -> bool:
    pred_f = to_fraction(pred)
    gold_f = to_fraction(gold)
    if pred_f is None or gold_f is None:
        return pred.strip() == gold.strip()
    return pred_f == gold_f


def correctness_reward_func(completions, answer, log_extra=None, log_metric=None, **kwargs):
    """
    Main RLVR reward.
    Returns 2.0 for exact numeric correctness, 0 otherwise.
    Correctness reward should dominate all auxiliary rewards.
    """
    texts = [completion_to_text(c) for c in completions]
    preds = [extract_xml_answer(t) for t in texts]

    rewards = []
    correct_count = 0

    for pred, gold in zip(preds, answer):
        ok = numeric_equal(pred, gold)
        rewards.append(2.0 if ok else 0.0)
        correct_count += int(ok)

    if log_extra is not None:
        log_extra("pred_answer", preds)
        log_extra("gold_answer", list(answer))

    if log_metric is not None and len(rewards) > 0:
        log_metric("reward_accuracy_rate", correct_count / len(rewards))

    return rewards


def format_reward_func(completions, **kwargs):
    """
    Small reward for following the XML format.
    Keep this much smaller than correctness reward.
    """
    texts = [completion_to_text(c) for c in completions]
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"

    return [
        0.2 if re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE) else 0.0
        for text in texts
    ]


def answer_presence_reward_func(completions, **kwargs):
    """
    Tiny reward for producing a parseable final answer.
    This helps early training when the model does not follow format yet.
    """
    texts = [completion_to_text(c) for c in completions]
    rewards = []

    for text in texts:
        ans = extract_xml_answer(text)
        rewards.append(0.1 if ans else 0.0)

    return rewards


def build_dataset(split: str, limit: int | None = None):
    ds = load_dataset("openai/gsm8k", "main", split=split)

    if limit is not None:
        limit = min(limit, len(ds))
        ds = ds.select(range(limit))

    def convert(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": extract_hash_answer(example["answer"]),
        }

    ds = ds.map(convert, remove_columns=ds.column_names)
    return ds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen25-0p5b-gsm8k-grpo")

    parser.add_argument("--train_limit", type=int, default=2000)
    parser.add_argument("--eval_limit", type=int, default=200)

    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)

    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.0)

    parser.add_argument("--report_to", type=str, default="none")

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    train_ds = build_dataset("train", limit=args.train_limit)
    eval_ds = build_dataset("test", limit=args.eval_limit)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,

        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,

        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,

        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_grad_norm=0.1,

        temperature=0.9,
        top_p=0.95,

        # beta=0 means no reference model KL term, saving memory.
        # You can try beta=0.001 later.
        beta=args.beta,

        # Current TRL default is DAPO-style loss; it is generally better for long completions.
        loss_type="dapo",
        scale_rewards="group",

        # Important: keep extra columns such as "answer" for reward functions.
        remove_unused_columns=False,

        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,

        report_to=args.report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[
            correctness_reward_func,
            format_reward_func,
            answer_presence_reward_func,
        ],
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
```

几个关键点：

TRL 的 `GRPOTrainer` 要求训练数据至少有 `"prompt"` 列；如果 reward function 需要 `"answer"` 这类额外列，配置里要保留额外列。官方文档也说明，额外数据列会传给 reward function，而自定义 reward function 应返回 `list[float]`。([Hugging Face][2])

`num_generations` 是每个 prompt 采样多少个 completion；TRL 文档说明 effective batch size 必须能被 `num_generations` 整除。([Hugging Face][2])

---

# 七、训练命令

创建：

```text
scripts/train_gsm8k_0p5b.sh
```

内容：

```bash
#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false

accelerate launch src/train_grpo_gsm8k.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/qwen25-0p5b-gsm8k-grpo \
  --train_limit 2000 \
  --eval_limit 200 \
  --max_steps 500 \
  --save_steps 100 \
  --batch_size 4 \
  --grad_accum 4 \
  --num_generations 4 \
  --max_prompt_length 512 \
  --max_completion_length 512 \
  --lr 5e-6 \
  --beta 0.0 \
  --report_to none
```

运行：

```bash
chmod +x scripts/train_gsm8k_0p5b.sh
bash scripts/train_gsm8k_0p5b.sh
```

第一轮不要追求高分。你先看这几件事：

```text
1. reward_accuracy_rate 是否从接近 0 慢慢上升
2. format_reward 是否快速上升
3. completion 长度是否失控
4. frac_reward_zero_std 是否过高
5. 是否出现固定答案，比如总是输出 42
```

TRL 会记录 reward、各 reward function 的均值、reward_std、`frac_reward_zero_std`、entropy、completion length 等指标；其中 `frac_reward_zero_std` 很重要，因为它表示同一题多个 completion 的 reward 没有差异，GRPO 此时学习信号很弱。([Hugging Face][2])

---

# 八、评估脚本

训练完之后，你需要独立评估，而不是只看训练 reward。

创建：

```text
src/eval_math.py
```

内容：

```python
import argparse
import json
import re
from fractions import Fraction

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are a careful math reasoning assistant.

Solve the problem step by step.

You must respond in exactly this format:

<reasoning>
your reasoning here
</reasoning>
<answer>
final answer only
</answer>

Rules:
- Put only the final numeric answer inside <answer>.
- Do not include units or explanations inside <answer>.
- If the answer is a fraction, use a/b format.
"""


def extract_hash_answer(gsm8k_answer: str) -> str:
    if "####" in gsm8k_answer:
        return gsm8k_answer.split("####")[-1].strip().replace(",", "")
    return gsm8k_answer.strip().replace(",", "")


def extract_xml_answer(text: str) -> str:
    m = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().replace(",", "")

    m = re.search(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip().replace(",", "")

    candidates = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    if candidates:
        return candidates[-1].strip()

    return ""


def to_fraction(x: str):
    x = x.strip().replace(",", "").replace("$", "").strip(". ")
    if not x:
        return None
    try:
        return Fraction(x)
    except Exception:
        return None


def numeric_equal(pred: str, gold: str) -> bool:
    pred_f = to_fraction(pred)
    gold_f = to_fraction(gold)
    if pred_f is None or gold_f is None:
        return pred.strip() == gold.strip()
    return pred_f == gold_f


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, k: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = k > 1
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": k,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs.update(
            {
                "temperature": 0.7,
                "top_p": 0.95,
            }
        )

    outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    completions = []

    for output in outputs:
        completion_ids = output[prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        completions.append(text)

    return completions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output_jsonl", type=str, default="reports/eval_gsm8k.jsonl")

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    total = 0
    pass_at_k = 0
    rows = []

    for ex in ds:
        question = ex["question"]
        gold = extract_hash_answer(ex["answer"])

        prompt = build_prompt(tokenizer, question)
        completions = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            k=args.k,
        )

        preds = [extract_xml_answer(c) for c in completions]
        correctness = [numeric_equal(p, gold) for p in preds]

        ok = any(correctness)
        total += 1
        pass_at_k += int(ok)

        rows.append(
            {
                "question": question,
                "gold": gold,
                "preds": preds,
                "correctness": correctness,
                "ok": ok,
                "completions": completions,
            }
        )

    score = pass_at_k / total if total else 0.0

    print(f"split={args.split}")
    print(f"limit={total}")
    print(f"pass@{args.k}={score:.4f}")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
```

运行 base model：

```bash
python src/eval_math.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --limit 200 \
  --k 1 \
  --output_jsonl reports/base_gsm8k_pass1.jsonl
```

运行 RLVR 后的 LoRA 模型：

```bash
python src/eval_math.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/qwen25-0p5b-gsm8k-grpo \
  --limit 200 \
  --k 1 \
  --output_jsonl reports/grpo_gsm8k_pass1.jsonl
```

评估 pass@4：

```bash
python src/eval_math.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/qwen25-0p5b-gsm8k-grpo \
  --limit 200 \
  --k 4 \
  --output_jsonl reports/grpo_gsm8k_pass4.jsonl
```

注意：pass@k 只看最终答案可能会高估推理能力，因为模型有可能推理过程错但最终猜对。近期 RLVR 评估研究提出了 CoT-Pass@K，要求最终答案和中间推理都正确，正是为了解决这个问题。([Hugging Face][6])

---

# 九、第二版：加入 Math-Verify

GSM8K 大部分是整数或简单小数，所以第一版用 `Fraction` 就够了。但你后面做 MATH、竞赛题、LaTeX 题时，必须升级验证器。

新增一个函数：

```python
def math_verify_equal(pred: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify

        gold_parsed = parse(gold)
        pred_parsed = parse(pred)

        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return numeric_equal(pred, gold)
```

然后把 `correctness_reward_func` 里的：

```python
ok = numeric_equal(pred, gold)
```

替换成：

```python
ok = math_verify_equal(pred, gold)
```

`math-verify` 支持多种抽取与比较目标，包括 LaTeX 表达式、普通数学表达式和字符串，并且默认会同时使用 LaTeX 与普通表达式抽取。([GitHub][4])

更推荐的输出格式也可以改成：

```text
推理过程若干
Therefore, the final answer is \boxed{42}.
```

TRL 的 GRPO 文档里也给过类似例子：从 completion 中抽取 `\boxed{...}`，然后和 `ground_truth` 对比，正确给 1，错误给 0。([Hugging Face][2])

---

# 十、训练成功的判据

你这一阶段的目标不是达到 SOTA，而是完成完整工程闭环。合格标准如下：

```text
1. 能在 GSM8K 上跑通 GRPO
2. 能看到格式 reward 变好
3. 能看到 held-out pass@1 或 pass@4 有提升
4. 能保存 LoRA adapter
5. 能独立 eval，不依赖训练日志
6. 能打开 jsonl 看失败案例
7. 能解释每个 reward function 的作用
```

一个合理的第一轮实验记录应该长这样：

```text
model: Qwen/Qwen2.5-0.5B-Instruct
dataset: GSM8K train first 2000
eval: GSM8K test first 200
algorithm: GRPO
num_generations: 4
max_steps: 500
LoRA r: 16

base pass@1: ...
grpo pass@1: ...
base pass@4: ...
grpo pass@4: ...

主要失败：
- 没按格式输出
- 答案抽取失败
- 算术错误
- 题目理解错误
- 推理对但最终答案格式错
```

---

# 十一、最常见问题和修复方法

## 1. reward 一直是 0

原因通常是：

```text
模型完全不会按格式输出
模型太弱
题太难
num_generations 太少
答案抽取失败
```

修复：

```text
先加格式 reward
降低题目难度
只用 GSM8K 前 500～2000 条
num_generations 从 4 提到 8
换 1.5B 或 3B 模型
先做少量 SFT warmup
```

## 2. 模型只学会格式，正确率没提升

原因：

```text
格式 reward 太大
正确性 reward 太稀疏
训练题太难
base model 初始正确率太低
```

修复：

```text
correctness reward 设为 2.0 或 5.0
format reward 控制在 0.1～0.2
先用简单题
加入 answer_presence_reward 但权重要小
```

## 3. 模型输出越来越长

原因：

```text
推理标签内啰嗦
reward 没有长度约束
max_completion_length 太大
```

修复：

```text
max_completion_length 从 512 降到 256
加入轻微长度惩罚
使用 loss_type="dapo" 或 "dr_grpo"
```

TRL 文档说明，当前 GRPOConfig 支持 `loss_type`，其中 `"dapo"` 是默认项，用于改善长 completion 场景下的 token 级归一化问题；也支持 `"dr_grpo"` 等变体。([Hugging Face][2])

## 4. 模型总是输出同一个答案

原因：

```text
探索不足
训练数据太少
reward 设计太粗
温度太低
```

修复：

```text
temperature 提到 0.9～1.0
top_p 设为 0.95
扩大训练集
增加 num_generations
检查 reward 是否错误地把坏答案判成正确
```

## 5. eval 分数比训练 reward 差很多

原因：

```text
训练集过拟合
reward hacking
抽取器训练/评估不一致
只看了训练 reward
```

修复：

```text
固定 held-out eval
保存每次输出 jsonl
人工抽查 50 个失败样本
训练和评估使用同一个 answer extractor
```

---

# 十二、推荐学习里程碑

## Milestone 1：跑通最小闭环

```text
模型：Qwen2.5-0.5B-Instruct
数据：GSM8K 500 条
训练：max_steps=100
目标：能看到 reward 日志，能保存 adapter
```

## Milestone 2：完整 GSM8K 小实验

```text
模型：Qwen2.5-0.5B-Instruct
数据：GSM8K 2000～7000 条
训练：max_steps=500～2000
评估：test 200 / full test
目标：pass@1 有稳定提升
```

当前仓库对这个里程碑的实现方式是：

```bash
.venv/bin/python scripts/run/minimal_grpo_training.py \
  --experiment-profile stage6-gsm8k-experiment \
  --model-name /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775

.venv/bin/python scripts/run/gsm8k_eval.py \
  --base-model /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --adapter-path outputs/stage-6-gsm8k-experiment/train/adapter \
  --mode both \
  --limit 200 \
  --k 4 \
  --train-run-config outputs/stage-6-gsm8k-experiment/train/run_config.json \
  --train-metrics outputs/stage-6-gsm8k-experiment/train/train_metrics.json
```

阶段 6 的验收重点是：

- 能完成比阶段 5 更大的 GSM8K 训练实验。
- 能独立运行 held-out 评估，而不是只看训练 reward。
- 能同时得到 `base` 与 `adapter` 的 `pass@1` / `pass@k` 对比。
- 能打开 `cases.jsonl` 和 `failure_report.json` 看失败样例。
- 能从 `experiment_summary.json` 里一次看到训练配置、训练耗时和评估结果路径。

## Milestone 3：加入 Math-Verify

```text
数据：MATH-500 或 DeepMath 子集
输出：\boxed{}
验证：math-verify
目标：支持分数、根式、集合、区间等答案
```

## Milestone 4：vLLM 提速

```text
use_vllm=True
或 vllm_mode="server"
目标：提升 rollout 吞吐
```

TRL 文档里说明 vLLM 可以在 GRPO 中用于生成加速，并支持 colocate 和 server 模式；server 模式适合有独立推理 GPU 的情况。([Hugging Face][2])

当前仓库的阶段 7 将这个里程碑落到 RunPod L40S 48GB：

```text
云平台：RunPod on-demand Pod
GPU：1 x NVIDIA L40S 48GB
资源：16 vCPU+ / 64GB RAM+ / 200GB+ disk
镜像：Ubuntu + CUDA 12.x + PyTorch
默认：vLLM colocate
双卡可选：GPU0 训练，GPU1 跑 trl vllm-serve
```

本机只做配置验证：

```bash
.venv/bin/python scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-full-gsm8k-training \
  --dry-run \
  --train-limit 8
```

RunPod L40S 上执行全量训练：

```bash
.venv/bin/python scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-full-gsm8k-training \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --allow-dataset-download \
  --use-vllm \
  --vllm-mode colocate \
  --vllm-gpu-memory-utilization 0.35 \
  --vllm-max-model-length 1024
```

如果想降低 RLVR 冷启动难度，先做一次 SFT warm-start：

```bash
.venv/bin/python scripts/run/gsm8k_sft.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --allow-dataset-download \
  --output-dir outputs/stage-7-full-gsm8k-training/sft
```

然后从 SFT adapter 接 GRPO：

```bash
.venv/bin/python scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-full-gsm8k-training \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --allow-dataset-download \
  --initial-adapter-path outputs/stage-7-full-gsm8k-training/sft/adapter \
  --use-vllm \
  --vllm-mode colocate \
  --vllm-gpu-memory-utilization 0.35 \
  --vllm-max-model-length 1024
```

## Milestone 5：迁移多卡框架

```text
verl 或 OpenRLHF
Ray / vLLM / DeepSpeed
更大模型：1.5B、3B、7B
更大数据：DeepMath / OpenR1-Math / 自建题库
```

---

# 十三、你真正要掌握的能力清单

最终你不是只会“跑一个脚本”，而是要能自己设计数学 RLVR 系统。

你要掌握：

```text
1. 如何构造数学 prompt
2. 如何规定最终答案格式
3. 如何抽取模型答案
4. 如何做数值等价判断
5. 如何做符号等价判断
6. 如何避免格式 reward 压过正确性 reward
7. 如何判断 reward 是否有学习信号
8. 如何看 reward_std / entropy / completion length
9. 如何做 pass@1 / pass@k
10. 如何分析 reward hacking
11. 如何从 GSM8K 迁移到 MATH
12. 如何从 TRL 迁移到 verl / OpenRLHF
```
