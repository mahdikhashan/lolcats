# Experiment: feature dimension 32 in stage 1

**Status:** The config and the tools are ready, with a CPU test on a tiny model. No stage 1 training with this config has run yet.

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
| Stored stage 1 validation loss (`distill/eval/loss`) | 3.2549 at step 1,100 | Not measured yet |
| PIQA accuracy after stage 1 | 57.6 ± 1.2 | Not measured yet |
| MMLU-subset accuracy after stage 1 | 22.5 ± 2.5 | Not measured yet |
| Share of "A" answers on the MMLU subset | 95.4% | Not measured yet |
| Layers with γ above 0.999 for 100.0% of the tokens (after rounding) | 15 of 16 | Not measured yet |

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

Not run yet.
