# Agent Notes

## Environment

Use the local `.venv` first. This repository has already been validated with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For non-interactive commands, prefer the explicit interpreter:

```bash
.venv/bin/python scripts/check/smoke_check_env.py
.venv/bin/python scripts/check/verify_gsm8k_answers.py
.venv/bin/python scripts/data/cache_gsm8k_dataset.py
.venv/bin/python scripts/data/inspect_gsm8k_dataset.py --split train --limit 3
.venv/bin/python scripts/check/verify_answer_rewards.py
.venv/bin/python scripts/run/minimal_grpo_training.py --dry-run
.venv/bin/python scripts/run/stage4_dual_reward_smoke_test.py --dry-run
```

Do not assume `conda` is installed. `environment.yml` is kept as an optional conda path, but `.venv` is the current working setup.

## Stage Boundary

Stage 1 environment installation and validation is complete. Stage 2 GSM8K data loading, sample inspection, and standard answer extraction is complete. Stage 3 reward-function prototyping is complete. Stage 4 minimal single-machine GRPO training is complete. Stage 5 real single-machine RLVR training is complete. Stage 6 complete GSM8K experiment is the last locally validated result. The active stage is stage 7: full GSM8K training on cloud CUDA.

The stage 2 GSM8K initialization command stores the dataset cache under `data/hf_datasets/`, which is ignored by git. It may download the GSM8K dataset cache on first run.

Stage 3 reward checks use local hard-coded examples through `.venv/bin/python scripts/check/verify_answer_rewards.py`.

Stage 4 keeps a dedicated dual-reward smoke test through `.venv/bin/python scripts/run/stage4_dual_reward_smoke_test.py`, which uses a tiny built-in case set to verify both correctness reward and format reward become positive in real GRPO training logs.

Stage 6 keeps `.venv/bin/python scripts/run/minimal_grpo_training.py` for training, but adds the `stage6-gsm8k-experiment` profile plus `.venv/bin/python scripts/run/gsm8k_eval.py` for independent held-out evaluation. Keep stage 6 single-machine: no vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, multi-GPU requirements, wandb, Math-Verify, or formal evaluation scoring. Outputs belong under `outputs/stage-6-gsm8k-experiment/`, which is ignored by git.

Stage 7 adds the `stage7-full-gsm8k-training` profile for cloud full-run experiments. The recommended target is a RunPod on-demand Pod with `1 x NVIDIA L40S 48GB`, `16 vCPU+`, `64GB RAM+`, `200GB+` disk, and vLLM colocate mode. Use a prebuilt project image from `docker/runpod-stage7/Dockerfile`, based on `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, rather than installing open-ended dependencies on a paid GPU Pod. Build that image on a native `linux/amd64` builder; Apple Silicon local builds are too slow for the Stage 7 image and are guarded by `scripts/runpod/build_image.sh`. `.github/workflows/build-runpod-stage7.yml` provides a manual GitHub Actions builder that pushes to Docker Hub when `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets are configured. Two-GPU RunPod runs may use GPU 0 for GRPO training and GPU 1 for `trl vllm-serve`. Stage 7 allows CUDA and vLLM; it still does not require Ray, DeepSpeed, FSDP, verl, OpenRLHF, wandb, Math-Verify, or larger models.

Stage 7 also has a Merlin devbox fallback chain for environments where Hugging Face is reachable but `trl/vllm` compatibility is unstable. Use the `stage7-merlin-a10-qwen15b-no-vllm` profile with `Qwen2.5-1.5B-Instruct` stored under `/mlx_devbox/users/<user>/playground/models/Qwen2.5-1.5B-Instruct`, a project-local `.venv`, and `requirements-merlin-stage7.txt`. The reusable entrypoints are:

```bash
bash scripts/check/merlin_stage7_preflight.sh
RUN_NAME=train-50 TRAIN_LIMIT=256 MAX_STEPS=50 LOGGING_STEPS=5 SAVE_STEPS=25 bash scripts/run/merlin_stage7_qwen15b_no_vllm.sh
RUN_NAME=train MAX_STEPS=400 TRAIN_LIMIT=7473 bash scripts/run/merlin_stage7_qwen15b_no_vllm.sh
```

The Merlin chain uses no vLLM, keeps `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, `steps_per_generation=4`, and `num_generations=4`, and writes outputs under `outputs/stage-7-merlin-a10-qwen15b/`.

Stage 7 also includes optional SFT warm-start through `.venv/bin/python scripts/run/gsm8k_sft.py`. Use `--initial-adapter-path outputs/stage-7-full-gsm8k-training/sft/adapter` to continue GRPO from that SFT adapter. Dry-runs must not load model weights or save adapters.

For RunPod, use the staged scripts in order:

```bash
bash scripts/runpod/preflight.sh
bash scripts/runpod/smoke.sh
nohup bash scripts/runpod/run_stage7_full.sh > outputs/stage-7-full-gsm8k-training/logs/full_pipeline.log 2>&1 &
```

Do not run `pip install -U "trl[vllm]"` directly on the training Pod; use `requirements-runpod-stage7.txt` during image build.

## Verification

Before reporting stage-6 experiment changes as complete, run:

```bash
.venv/bin/python -m py_compile src/gsm8k_dataset.py src/answer_rewards.py src/minimal_grpo_training.py src/gsm8k_experiment.py src/training/__init__.py src/training/datasets.py src/training/profiles.py src/training/prompts.py src/training/reward_adapters.py src/training/runtime.py scripts/check/verify_gsm8k_answers.py scripts/check/verify_answer_rewards.py scripts/run/minimal_grpo_training.py scripts/run/gsm8k_eval.py scripts/run/stage4_dual_reward_smoke_test.py scripts/data/cache_gsm8k_dataset.py scripts/data/inspect_gsm8k_dataset.py scripts/check/smoke_check_env.py
.venv/bin/python scripts/check/verify_gsm8k_answers.py
.venv/bin/python scripts/check/verify_answer_rewards.py
.venv/bin/python scripts/run/minimal_grpo_training.py --experiment-profile stage6-gsm8k-experiment --dry-run
.venv/bin/python scripts/run/minimal_grpo_training.py --experiment-profile stage6-gsm8k-experiment --model-name /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --output-dir outputs/stage-6-gsm8k-experiment/train-iter4 --train-limit 2000 --max-steps 300 --per-device-train-batch-size 4 --steps-per-generation 4 --num-generations 4 --max-completion-length 192 --learning-rate 5e-7 --correctness-reward-weight 1 --format-reward-weight 0 --beta 0.02
.venv/bin/python scripts/run/gsm8k_eval.py --base-model /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --adapter-path outputs/stage-6-gsm8k-experiment/train-iter4/adapter --mode both --limit 50 --k 1 --train-run-config outputs/stage-6-gsm8k-experiment/train-iter4/run_config.json --train-metrics outputs/stage-6-gsm8k-experiment/train-iter4/train_metrics.json
openspec validate stage-6-complete-gsm8k-experiment --strict
```

Before reporting stage-7 cloud configuration or SFT changes as complete, run the relevant lightweight checks:

```bash
.venv/bin/python -m py_compile src/training/sft.py scripts/run/gsm8k_sft.py scripts/run/minimal_grpo_training.py src/training/runtime.py src/training/__init__.py
.venv/bin/python scripts/check/verify_gsm8k_answers.py
.venv/bin/python scripts/check/verify_answer_rewards.py
.venv/bin/python scripts/run/minimal_grpo_training.py --experiment-profile stage7-full-gsm8k-training --dry-run --train-limit 8
.venv/bin/python scripts/run/gsm8k_sft.py --dry-run --train-limit 8
openspec validate stage-7-full-gsm8k-training --strict
```

Before reporting the Merlin stage-7 fallback chain as complete, run:

```bash
.venv/bin/python -m py_compile src/training/profiles.py scripts/run/minimal_grpo_training.py scripts/run/gsm8k_eval.py
bash scripts/check/merlin_stage7_preflight.sh
RUN_NAME=train-50 TRAIN_LIMIT=256 MAX_STEPS=50 LOGGING_STEPS=5 SAVE_STEPS=25 bash scripts/run/merlin_stage7_qwen15b_no_vllm.sh
```

Expected stage-6 checks:

```text
GSM8K extraction check passed.
Reward check passed.
Dry-run passed.
Change 'stage-6-complete-gsm8k-experiment' is valid
```

The default stage-6 training and eval commands are:

```bash
.venv/bin/python scripts/run/minimal_grpo_training.py --experiment-profile stage6-gsm8k-experiment --model-name /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
.venv/bin/python scripts/run/gsm8k_eval.py --base-model /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --adapter-path outputs/stage-6-gsm8k-experiment/train-iter4/adapter --mode both --limit 50 --k 1 --train-run-config outputs/stage-6-gsm8k-experiment/train-iter4/run_config.json --train-metrics outputs/stage-6-gsm8k-experiment/train-iter4/train_metrics.json
```
