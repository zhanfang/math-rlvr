## 1. Full GSM8K Training Profile

- [x] 1.1 Add a `stage7-full-gsm8k-training` profile in `src/training/profiles.py` with full GSM8K train-split defaults and isolated stage 7 output paths.
- [x] 1.2 Ensure the training dry-run summary records stage 7 profile defaults, loaded example count, LoRA settings, reward weights, KL beta, sampling settings, and output directory.
- [x] 1.3 Verify stage 5 and stage 6 profiles keep their existing defaults and output paths.

## 2. vLLM GRPO Configuration

- [x] 2.1 Add training CLI arguments for `--use-vllm`, `--vllm-mode`, server host/port, GPU memory utilization, tensor parallel size, max model length, and related supported GRPOConfig fields.
- [x] 2.2 Extend `build_grpo_config` to pass supported vLLM fields through TRL signature filtering while keeping non-vLLM local training unchanged.
- [x] 2.3 Extend run summaries and failure hints so vLLM requests, server connection settings, and likely vLLM setup errors are visible.

## 3. SFT Warm-Start

- [x] 3.1 Add shared GSM8K SFT formatting utilities that convert raw GSM8K solutions into prompt/completion records using `<reasoning>` and `<answer>` tags.
- [x] 3.2 Add `scripts/run/gsm8k_sft.py` with dry-run, dataset cache controls, train limit, output directory, LoRA configuration, and summary writing.
- [x] 3.3 Implement real SFT LoRA training with TRL SFTTrainer and save `adapter/`, `sft_config.json`, and `train_metrics.json`.
- [x] 3.4 Add validation that SFT dry-run does not load model weights or save adapter artifacts.

## 4. GRPO From Existing Adapter

- [x] 4.1 Add `--initial-adapter-path` to the GRPO training entrypoint and run summary.
- [x] 4.2 Implement base model plus trainable PEFT adapter loading when `--initial-adapter-path` is provided.
- [x] 4.3 Preserve current fresh-LoRA behavior when no initial adapter is provided and preserve `--resume-from-checkpoint` as trainer checkpoint resume.
- [x] 4.4 Run a tiny local GRPO smoke path from an SFT adapter or test adapter to verify adapter loading and saving.

## 5. Evaluation And Documentation

- [x] 5.1 Update stage 7 documentation in `README.md` with local dry-run, SFT warm-start, cloud CUDA setup, vLLM colocate/server examples, full GRPO training, resume, and evaluation commands.
- [x] 5.2 Update `AGENTS.md` so future agents understand stage 7 allows cloud CUDA/vLLM while stage 6 remains the last validated local result.
- [x] 5.3 Update `plan.md` to mark stage 7 as the next full GSM8K training milestone and document the local-vs-cloud workflow.
- [x] 5.4 Ensure `scripts/run/gsm8k_eval.py` examples show stage 7 output paths and attach training metadata to `experiment_summary.json`.

## 6. Verification

- [x] 6.1 Run py_compile for touched `src/` and `scripts/run/` modules.
- [x] 6.2 Run `scripts/check/verify_gsm8k_answers.py` and `scripts/check/verify_answer_rewards.py`.
- [x] 6.3 Run `scripts/run/minimal_grpo_training.py --experiment-profile stage7-full-gsm8k-training --dry-run`.
- [x] 6.4 Run `scripts/run/gsm8k_sft.py --dry-run` with a small train limit.
- [x] 6.5 Run `openspec validate stage-7-full-gsm8k-training --strict`.
