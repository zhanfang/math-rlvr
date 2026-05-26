"""Prompt-building helpers shared by train and evaluation flows."""

from __future__ import annotations

from typing import Any


CHAT_SYSTEM_PROMPT = (
    "You are a careful grade-school math assistant. Follow the requested XML output format exactly."
)
CHAT_USER_TEMPLATE = """Solve the problem.

Your response must contain exactly two XML sections in this order:
<reasoning>your concise arithmetic reasoning</reasoning>
<answer>final numeric answer only</answer>

Rules:
- Include both opening and closing tags.
- Close </reasoning> before opening <answer>.
- Put only the final number inside <answer>.
- Do not use boxed notation.
- Do not copy placeholder text such as "your concise arithmetic reasoning".

Problem:
{question}
"""
_FALLBACK_CHAT_TEMPLATE = (
    "<|im_start|>system\n"
    "{system}\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{user}\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_grpo_chat_messages(question: str) -> list[dict[str, str]]:
    """Build the shared chat-style prompt payload used by train and eval."""
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": CHAT_USER_TEMPLATE.format(question=question.strip())},
    ]


def render_chat_prompt(messages: list[dict[str, str]], tokenizer: Any | None = None) -> str:
    """Render a chat prompt with the tokenizer template when available."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return _FALLBACK_CHAT_TEMPLATE.format(
        system=messages[0]["content"],
        user=messages[1]["content"],
    )


def build_grpo_training_prompt(question: str, tokenizer: Any | None = None) -> str:
    """Build the shared chat-style prompt for GRPO train/eval runs."""
    return render_chat_prompt(build_grpo_chat_messages(question), tokenizer=tokenizer)


def maybe_load_prompt_tokenizer(model_name: str, local_files_only: bool) -> Any | None:
    """Best-effort tokenizer load for prompt rendering without forcing a hard dependency."""
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception:
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
