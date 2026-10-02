# Experiment: second round, with β2 and the minimum learning rate of the paper

**Status:** The code, the config and the CPU tests are ready. No stage 1 training with this config has run yet.

## Question

The [float32 experiment](float32.md) gave a stage 1 validation loss of 4.9478. With the LoLCATs recipe, the loss was 3.4219. Three settings of the optimizer were still different from the paper (Table 13):

| Setting | Float32 run | Paper |
|---|---|---|
| AdamW β2 | 0.999 (the torch default) | 0.99 |
| Learning rate at the end of the cosine schedule | 0 | 0.1 × the peak (1e-4) |
| Gradient clipping | None | 1.0 |

This experiment changes β2 and the minimum learning rate to the values of the paper. Does stage 1 then become better?

**Gradient clipping** is in a separate change. That change adds the clipping to the trainer, and a second config with all three settings. Thus the comparison of the two configs shows the effect of clipping alone.

## What changes, and what stays the same

| Setting | Float32 run | This experiment |
|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd32_m4_fp32` | The same |
| Distill config | `distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b` | **`distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b`** |
| Precision | float32 for the whole model | The same |
| Peak learning rate, warmup | 1e-3, 118 steps (10%), linear | The same |
| Schedule | Cosine to 0 (`cosine_warmup`) | **Cosine to 1e-4 (`cosine_warmup_min_lr`, `min_lr_rate: 0.1`)** |
| AdamW betas, eps | (0.9, 0.999), 1e-8 | **(0.9, 0.99)**, 1e-8 |
| Gradient clipping | None | None |
| Feature dimension, window, sinks | 32, 128, 4 | The same |
| Loss | 1000 × MSE (`mse_factor` 1000) | The same |
| Data, steps, seed | Alpaca-cleaned, 1,178 steps, seed 0 | The same |

After this change, the stage 1 recipe agrees with Table 13 of the paper, except for three settings:

- **No gradient clipping**, not 1.0. The separate change adds it.
- **Feature dimension 32**, not 128. The [feature dimension experiment](feature-dimension.md) found a small effect: +5% validation loss with 32 under the LoLCATs recipe.
- **The scale of the loss**, which the paper does not give clearly ([math against code](../math-code-discrepancy.md), D11). For Adam, a constant factor on the loss has no effect, except through eps (factor 8 of section 12 of the [gap analysis](../11-gap-analysis.md)).

## Predictions

1. **Minimum learning rate:** the learning rate stays at 1e-4 or more after the warmup, not near 0 at the end. In the float32 run, the validation loss still decreased at step 1,100 (finding 4 there). Thus the late steps can lower the loss more.
2. **β2 = 0.99:** the second-moment estimate follows approximately the last 100 steps, not 1,000. When the gradients become smaller, the steps become larger sooner.
3. Together: the validation loss is lower than 4.9478, and the best checkpoint comes from the end of the run.

## How to run

### Option A: HF Jobs (H200), recommended

1. **Build a new Docker image.** Merge the PR with the code and the config first. Then select Actions → "Docker image" → Run workflow. HF Jobs runs only the code inside the image ([document 3](../03-infrastructure.md)).
2. **Train stage 1 on HF Jobs:**

   ```bash
   make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=3h \
     ARGS="--model_config distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b --no_finetune"
   ```

The run needs approximately the same time and memory as the float32 run.

### Checkpoint

```
checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
```

The folder is the same as for the float32 run. The name contains the new distill config. Thus no earlier checkpoint gets overwritten.

### Evaluation on the A10

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b \
MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 \
MODELS=stage1 TASKS="mmlu_subset piqa arc_easy" \
scripts/compare_stages.sh 2>&1 | tee eval-second-round.log
```

## How to compare

All runs use the same validation data and the same loss (`mse_factor` 1000). Thus their validation losses are directly comparable.

| Measure | fd32, LoLCATs recipe, bf16 | fd32, recipe of the paper, float32 | Second round (β2, minimum learning rate) |
|---|---|---|---|
| Stored stage 1 validation loss, and its step | 3.4219 at step 1,100 | 4.9478 at step 1,100 | Not measured yet |
| MMLU-subset accuracy | 23.2 ± 2.5 | 24.6 ± 2.5 | Not measured yet |
| Share of "A" answers | 66.0% | 53.0% | Not measured yet |
| PIQA accuracy | Not measured yet | 57.7 ± 1.2 | Not measured yet |
| ARC-Easy accuracy | Not measured yet | 35.7 ± 1.0 | Not measured yet |
| Layers 1–15 with γ above 0.999 for 100.0% of the tokens | 14 of 15 | 0 of 15 (3 layers at 99.9%) | Not measured yet |
| α | 0.063–0.656 | 0.602–0.786 | Not measured yet |

How to read the outcome:

| Outcome | Conclusion | Next step |
|---|---|---|
| The validation loss is lower than 3.4219 | With β2 and the minimum learning rate of the paper, stage 1 is better than with the LoLCATs recipe | Run stage 2 with the recipe of the paper. Then try feature dimension 128. |
| The validation loss is between 3.4219 and 4.9478 | The two settings help, but do not close the difference | Run the config with gradient clipping. Examine the gate state and the normalization of the gated branch (D1). |
| The validation loss is approximately 4.9478 or higher | The two settings have no large effect | Run the config with gradient clipping. Examine D1 and the sink hypothesis (section 13.2 of the gap analysis). |

## Code changes

These changes keep the behavior of all existing configs:

- `src/trainer/optim.py`:
  - `get_optimizer` changes `betas` from the list of the YAML config to a tuple (lines 14–15). Configs without `betas` keep the torch default (0.9, 0.999).
  - `get_scheduler` has the new type `cosine_warmup_min_lr` (line 45). It uses `get_cosine_with_min_lr_schedule_with_warmup` from `transformers.optimization` (in transformers 4.43.1, it is not in the top-level package). With `min_lr_rate: 0.1`, the cosine decays to 0.1 × the peak.
- `configs/experiment/distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b.yaml`: a copy of `distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b.yaml` with two changes:
  - `optimizer.betas: [0.9, 0.99]`,
  - `lr_scheduler`: `cosine_warmup_min_lr` with `min_lr_rate: 0.1`.

## Tests

These tests ran on CPU with a tiny Llama.

- **Optimizer and schedule.** A script built the optimizer and the scheduler from each config and ran 1,178 steps:

  | Config | AdamW betas | Learning rate at steps 0 / 59 / 118 / 589 / 1,100 / 1,177 | Minimum after the warmup |
  |---|---|---|---|
  | `..._lr1e-3_cosine_1b` (float32 run) | (0.9, 0.999) | 0 / 5.0e-4 / 1.0e-3 / 5.9e-4 / 1.3e-5 / 2.2e-9 | 2.2e-9 |
  | `..._lr1e-3_paper_noclip_1b` (this experiment) | (0.9, 0.99) | 0 / 5.0e-4 / 1.0e-3 / 6.3e-4 / 1.1e-4 / 1.0e-4 | 1.0e-4 |

- **Existing configs.** The LoLCATs config (`..._lr1e-2_1b`) still builds a ReduceLROnPlateau scheduler with betas (0.9, 0.999) and a learning rate of 0.01.
- **Stage 1 on CPU.** The real `distill_llama.main()` ran with the arguments of option A, the float32 model config and synthetic data. It finished with exit 0 and skipped stage 2. The log shows `cosine_warmup_min_lr`, `min_lr_rate: 0.1` and the betas in the config.
- The CPU tests used the optimizer `adamw_torch`, because the fused optimizer needs CUDA.

## Results

Not run yet.
