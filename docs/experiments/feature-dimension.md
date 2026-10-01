# Experiment: feature dimension 32 in stage 1

**Status:** Stage 1 with feature dimension 32 trained on HF Jobs. The MMLU subset evaluation ran on 2026-10-01. PIQA is not evaluated yet.

## Question

Does a feature dimension of 32 give a better stage 1 than the current feature dimension of 128?

The [hyperparameters](../hyperparameters.md) document gives the background:

- The current config uses `feature_dim: 128`. That is 2× the head dimension of Llama-3.2-1B (64).
- The LoLCATs rule is 0.5 × the head dimension, which gives 32 for Llama-3.2-1B.
- The Lizard paper uses 128 for all experiments. For its 8B models, 128 is 1× the head dimension.

## What changes, and what stays the same

| Setting | Current (Run 1, stage 1) | This experiment |
|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd128_m4` | `distill_llama3_2_1b_lizard_w128_fd32_m4` |
| `feature_dim` | 128 | **32** |
| Features per query or key | 256 | 64 |
| Weights of `phi_q` and of `phi_k`, per layer | 8,192 each | 2,048 each |
| Lizard parameters per layer, and in total | 18,437 and 294,992 | 6,149 and 98,384 |
| Distill config, data and seed | `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b`, Alpaca-cleaned, seed 0 | The same |
| Stages | Stage 1 and stage 2 | Stage 1 only |

Only the feature dimension changes. The recipe stays the LoLCATs recipe of Run 1 ([document 4](../04-training-runs.md)). Thus the result shows the effect of the feature dimension alone. This experiment does not test the recipe of the paper.

## How to run

There are two ways to train stage 1: on HF Jobs (option A) or on the A10 machine (option B). Both write the same checkpoint name.

### Option A: HF Jobs (H200)

1. **Build a new Docker image.** Merge the PR with the new config first. Then select Actions → "Docker image" → Run workflow. HF Jobs runs only the code inside the image ([document 3](../03-infrastructure.md)).
2. **Train stage 1 on HF Jobs:**

   ```bash
   make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=2h \
     ARGS="--model_config distill_llama3_2_1b_lizard_w128_fd32_m4 --no_finetune"
   ```

   - `ARGS` adds a second `--model_config` after the default one. argparse keeps the last value.
   - `--no_finetune` stops the run after stage 1. The run name still contains the finetune config, so the checkpoint gets the standard name.
   - Run 1 needed approximately 50 minutes for stage 1 on the H200 ([document 2](../02-compute-and-cost.md)). With the setup, this run probably takes approximately 1 hour and costs approximately $5. These values are estimates.
   - The job pushes the stage 1 checkpoint to `checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt`. The folder and the name are different from those of the fd128 checkpoints, so no checkpoint gets overwritten.
### Option B: the A10 machine, with conda

This option needs no Docker and no HF Jobs. It uses the `lolcats-env` conda environment, as `eval.sh` and `compare_stages.sh` do. It still downloads Llama-3.2-1B and Alpaca-cleaned from the Hub, so it needs `HF_TOKEN`.

```bash
conda activate lolcats-env
export HF_TOKEN=hf_...
make distill-local ARGS="--max_steps 10"             # short test: memory and downloads
nohup make distill-local > distill-fd32.log 2>&1 &   # full stage 1
tail -f distill-fd32.log
```

- `make distill-local` trains stage 1 only, with `LOCAL_MODEL_CONFIG` (default `distill_llama3_2_1b_lizard_w128_fd32_m4`) on `GPU` (default 0, in the order of `nvidia-smi`).
- `CACHE_DIR` (default `~/.cache/huggingface/hub`) replaces the `cache_dir` of the model config, `/workspace/lolcats/scratch/`, which a normal user cannot create.
- The checkpoints go to `checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4/` on the machine. To push them to the Hub too, add `HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b`. This needs a token with write access.
- The short test has `-ms=10` in its run name, so its checkpoint does not replace the checkpoint of the full run.
- **Memory:** The memory of the A10 may not be sufficient. [Document 2](../02-compute-and-cost.md) estimated 22–25 GB for stage 1, including a margin for CUDA overhead, and the A10 has 23,028 MiB. Thus run the short test first. If it stops with an out-of-memory error, use option A.
- **Time:** The H200 needed approximately 50 minutes for stage 1. The A10 has much less compute and memory bandwidth, so the full run probably takes several hours. This value is an estimate.
- If `make` is not installed on the machine, `make -n distill-local` on another machine prints the command.

### Evaluation

**Evaluate the stage 1 model on the A10:**

   ```bash
   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4 \
     MODELS=stage1 TASKS="mmlu_subset piqa" scripts/compare_stages.sh
   ```

   `compare_stages.sh` builds the checkpoint names from `MODEL_CONFIG`. It downloads the stage 2 checkpoint only when `MODELS` contains `stage2`, so it works without a stage 2 checkpoint. If a checkpoint is already on the machine (option B), it uses that file and does not download it.

## How to compare

The values for feature dimension 128 come from the [stage difference experiment](stage-difference.md) (quick checks 1 and 2).

| Measure | Feature dimension 128 | Feature dimension 32 |
|---|---|---|
| Stored stage 1 validation loss (`distill/eval/loss`) | 3.2549 at step 1,100 | 3.4219 at step 1,100 |
| PIQA accuracy after stage 1 | 57.6 ± 1.2 | Not measured yet |
| MMLU-subset accuracy after stage 1 | 22.5 ± 2.5 | 23.2 ± 2.5 |
| Share of "A" answers on the MMLU subset | 95.4% | 66.0% |
| Layers with γ above 0.999 for 100.0% of the tokens (after rounding) | 15 of 16 | 14 of 16 (layer 15: 99.9%) |

Both runs use the same stage 1 loss (1000 × MSE on the same layer outputs) and the same validation data. Thus their validation losses are directly comparable.

How to read the outcome:

| Outcome | Conclusion |
|---|---|
| Lower validation loss and higher scores with 32 | 128 is too large for Llama-3.2-1B with this recipe. |
| Approximately the same values | The feature dimension is not a cause of the gap. |
| Higher validation loss and lower scores with 32 | The larger feature dimension helps. |

## Code changes

- `configs/model/distill_llama3_2_1b_lizard_w128_fd32_m4.yaml`: a copy of the fd128 config with `feature_dim: 32`.
- `distill_llama.py`: the new flag `--no_finetune` stops the run after stage 1.
- `distill_llama.py`: the new argument `--cache_dir` replaces the `cache_dir` of the model config.
- `compare_stages.sh`: it downloads the stage 2 checkpoint only when `MODELS` contains `stage2`. It uses a checkpoint that is already on the machine and does not download it.
- `Makefile`: the new target `distill-local` (option B), and a usage line for a stage-1-only run on HF Jobs.

## Tests

These tests ran on CPU with a tiny Llama and synthetic data. They used the real `distill_llama.main()` with the arguments of `make lizard ARGS="..."`.

- **Stage 1 only:** With `--model_config distill_llama3_2_1b_lizard_w128_fd32_m4 --no_finetune`, the run finished with exit 0 and skipped stage 2. The checkpoint name contains `-m=distill_llama3_2_1b_lizard_w128_fd32_m4-f=finetune_lora_qkvo_alpaca_clean_1b`. `phi_q` and `phi_k` have the shape 32 × 16, the feature dimension times the head dimension of the tiny model. The run wrote no stage 2 file.
- **No change without the flag:** Without `--no_finetune`, stage 2 ran and wrote its `_ft.pt` checkpoint.
- **Evaluation:** `compare_stages.sh` with this model config and `MODELS=stage1` finished with exit 0. It requested only the stage 1 checkpoint. All 10 Lizard parameters of the tiny model loaded. With `stage2` in `MODELS`, it requests both checkpoints.
- **`make distill-local`:** The test ran the command that `make -n distill-local` prints. The model config of the test had `cache_dir: /workspace/lolcats/scratch/`. With `CACHE_DIR` set, the model config and the tokenizer config of the dataset used the new folder, and `/workspace` did not appear in the log. The run skipped stage 2 and wrote the checkpoint to `checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4/`.
- **Local checkpoints in `compare_stages.sh`:** The test used a fake Hub without the stage 1 checkpoint. The script used the local stage 1 file and downloaded the stage 2 file. A checkpoint that is missing on the machine and on the Hub still stops the script with an error.
- **Test-only changes for the CPU:** the optimizer `adamw_torch` (the fused optimizer needs CUDA), float32, and `low_cpu_mem_usage` off for the old torch version of the test environment.

## Results

### Run 1: MMLU subset (2026-10-01)

- Training: stage 1 only, on HF Jobs (option A). The job pushed the checkpoint to `nanoman1/lolcats-lizard-llama-3.2-1b`. This document does not record the job ID or the training time.
- Evaluation command: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4 MODELS=stage1 TASKS=mmlu_subset scripts/compare_stages.sh`
- Run directory: `results/stages/20261001-183601`, on `student06`, GPU 0 (A10)
- Code: lolcats `92010bc`, harness `b281b09`
- Software: Python 3.11.16, torch 2.5.1 (CUDA 12.4), transformers 4.43.1, peft 0.9.0

#### Checkpoint

| Checkpoint | SHA-256 | Size | Parameters | Dtype | Bytes per parameter | Stored step | Stored loss |
|---|---|---|---|---|---|---|---|
| Stage 1, fd32 (`..._distill.pt`) | `768986c87b4ce09f8c503a21dcbb5db6b94b4c59fabf339dd83617869192c918` | 244,370 B | 98,384 | bf16 (80 tensors) | 2.48 | 1100 | `distill/eval/loss` = 3.4219 |

- **The load is complete.** All 80 Lizard parameters (16 layers × 5) are in the file and hold their values after the load.
- The parameter count agrees with the value that the table at the top of this document predicts (98,384).
- The number of bytes for each parameter is higher than for fd128 (2.16), because the tensors are smaller. The file has a constant overhead for each tensor.
- The best checkpoint comes from step 1,100, the same step as for fd128.
- The Hub also has `..._distill_1000.pt`, the periodic save at step 1,000. The Hub has no stage 1 results CSV for this run under the expected name. Thus the summary has no validation curve.

#### Scores and answer letters

| Model | Right answers | Accuracy | "A" | "B" | "C" | "D" | Accuracy if always "A" |
|---|---|---|---|---|---|---|---|
| Stage 1, fd128 ([stage difference](stage-difference.md), quick check 2) | 64 / 285 | 22.5 ± 2.5 | 95.4% | 2.8% | 1.8% | 0.0% | 24.2 |
| Stage 1, fd32 | 66 / 285 | **23.2 ± 2.5** | 66.0% | 22.1% | 0.7% | 11.2% | 24.2 |
| Teacher ([document 7](../07-results.md)) | 96 / 285 | 33.7 ± 2.8 | – | – | – | – | – |

The difference between fd32 and fd128 is +0.7 points (2 questions). The unpaired SE of the difference is approximately 3.5, so z ≈ 0.2. Thus the two accuracies are not clearly different.

| Model | Mass of the four letters | Confidence | Entropy over the four letters |
|---|---|---|---|
| Stage 1, fd128 ([temperature](temperature.md), T = 1) | 0.018 | 0.586 | 1.550 bits |
| Stage 1, fd32 | 0.024 | 0.476 | 1.681 bits |

#### Gates and Lizard parameters

The gate values come from the same 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens) as in the stage difference experiment.

| Measure | fd128 | fd32 |
|---|---|---|
| Layers 1–15: γ above 0.999 | 100.0% in all 15 layers | 100.0% in layers 1–14, 99.9% in layer 15 |
| Layers 1–15: weight kept after 512 tokens | Rounds to 1.0 | Rounds to 1.0 |
| Layer 0: γ above 0.999 | 58.0% | 55.3% |
| Layer 0: weight kept after 128 / 256 / 512 tokens | 0.35 / 0.11 / 0.0059 | 0.71 / 0.53 / 0.25 |
| α | 0.042–0.648 | 0.063–0.656 |
| Sink logits exactly 0.5, 1 or 2 | 13 of 16 layers | 10 of 16 layers |
| ‖W_γ‖ | 3.59–5.13 | 3.53–4.93 |
| Feature-map weight RMS | 0.15–0.22 | 0.16–0.22 |

#### Findings

**Finding 1: feature dimension 32 gives a higher stage 1 validation loss**. The loss is 3.4219 against 3.2549 for fd128, at the same step. That is 0.167 higher (+5.1%). Both runs use the same loss and the same validation data. Thus the attention approximation with 32 is a little worse than with 128.

**Finding 2: the MMLU-subset accuracy does not change**. fd32 gets 23.2 and fd128 gets 22.5 (z ≈ 0.2). Both are near the accuracy of "always A" (24.2) and far below the teacher (33.7).

**Finding 3: the answers are less concentrated on "A", but not more often right.**

- fd32 selects "A" for 66.0% of the questions, against 95.4% for fd128. It selects "B" for 22.1% and "D" for 11.2%.
- The accuracy stays at the level of chance. Thus the other letters do not come from a better understanding of the questions.
- The letter mass is still very small (0.024 against 0.018). The model still almost does not follow the 5-shot format.

**Finding 4: the gate saturates the same way**. In layers 1–15, γ is above 0.999 for 99.9–100.0% of the tokens. The weight kept after 512 tokens rounds to 1.0. Only layer 0 has a decay, and it decays more slowly than with fd128. Thus the feature dimension does not cause the saturation of the gate.

**Finding 5: the sink logits again stop at powers of two in most layers**. In 10 of 16 layers, the value is exactly 0.5, 1 or 2. This agrees with the bf16 rounding of the stage difference experiment (finding 3 there).

### Interpretation

- With the outcome table above, the result is between two rows. The validation loss is higher with 32, but the MMLU-subset accuracy is approximately the same.
- **The feature dimension is not a cause of the gap on the MMLU subset**. A 4× smaller feature map changes the validation loss by only 5%, and changes the accuracy by less than one SE.
- The gate saturation and the "A" preference do not depend on the feature dimension. This agrees with section 13 of the [gap analysis](../11-gap-analysis.md), which ranks the gate saturation as the main defect.
- For the [math against code](../math-code-discrepancy.md) document: this result makes D4 (the meaning of "feature dimension 128") an improbable cause of the gap. It does not test D2 (one feature map for each head).
- This result is preliminary. It uses only the MMLU subset. The comparison with fd128 uses unpaired SEs.

### Open items

1. **PIQA:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4 MODELS=stage1 TASKS=piqa scripts/compare_stages.sh`. Compare with 57.6 ± 1.2 for fd128.
2. **Paired comparison:** `compare_stages.py` compares models only inside one run, and fd32 and fd128 need different values of `MODEL_CONFIG`. A paired comparison needs the per-question logs of both runs.
