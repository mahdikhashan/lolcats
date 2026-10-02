# Experiment: second round, with β2 and the minimum learning rate of the paper

**Status:** Config 1 (`..._paper_noclip_1b`, without clipping) finished stage 1 training and evaluation on 2026-10-02 (MMLU subset, PIQA and ARC-Easy). The validation loss is 3.9764, between the float32 run (4.9478) and the LoLCATs recipe (3.4219). The run with gradient clipping is the [gradient clipping experiment](gradient-clipping.md). Its validation loss is 3.5092.

## Question

The [float32 experiment](float32.md) gave a stage 1 validation loss of 4.9478. With the LoLCATs recipe, the loss was 3.4219. Three settings of the optimizer were still different from the paper (Table 13):

| Setting | Float32 run | Paper |
|---|---|---|
| AdamW β2 | 0.999 (the torch default) | 0.99 |
| Learning rate at the end of the cosine schedule | 0 | 0.1 × the peak (1e-4) |
| Gradient clipping | None | 1.0 |

This experiment changes β2 and the minimum learning rate to the values of the paper. Does stage 1 then become better?

**Gradient clipping** is a separate experiment, the [gradient clipping experiment](gradient-clipping.md). It adds the clipping to the trainer, and a second config with all three settings. Thus the comparison of the two configs shows the effect of clipping alone.

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

- **No gradient clipping**, not 1.0. The [gradient clipping experiment](gradient-clipping.md) adds it.
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
| Stored stage 1 validation loss, and its step | 3.4219 at step 1,100 | 4.9478 at step 1,100 | **3.9764 at step 1,100** |
| MMLU-subset accuracy | 23.2 ± 2.5 | 24.6 ± 2.5 | 26.7 ± 2.6 |
| Share of "A" answers | 66.0% | 53.0% | 51.2% |
| PIQA accuracy | Not measured yet | 57.7 ± 1.2 | 57.3 ± 1.2 |
| ARC-Easy accuracy | Not measured yet | 35.7 ± 1.0 | 36.5 ± 1.0 |
| Layers 1–15 with γ above 0.999 for 100.0% of the tokens | 14 of 15 | 0 of 15 (3 layers at 99.9%) | 0 of 15 (3 layers at 99.9%) |
| α | 0.063–0.656 | 0.602–0.786 | 0.472–0.644 |

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

In this section, config 1 is `distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b`, the config of this document. Config 2 is the config with gradient clipping of the [gradient clipping experiment](gradient-clipping.md).

### Run 1: config 1, without clipping (2026-10-02)

- Training: stage 1 only, with the command of option A. The evaluation took the checkpoint from `nanoman1/lolcats-lizard-llama-3.2-1b`. This document does not record the job ID or the training time.
- Evaluation command: the command in "Evaluation on the A10" above, with `TASKS="mmlu_subset piqa arc_easy"`
- Run directory: `results/stages/20261002-102425`, on `student06`, GPU 0 (A10). Start: 2026-10-02 08:24:45 UTC.
- Code: lolcats `680742e` (the merge of the code and the config of this experiment), harness `b281b09`
- Software: Python 3.11.16, torch 2.5.1 (CUDA 12.4), transformers 4.43.1, peft 0.9.0

#### Checkpoint

| Checkpoint | SHA-256 | Size | Parameters | Dtype | Bytes per parameter | Stored step | Stored loss |
|---|---|---|---|---|---|---|---|
| Stage 1, config 1 (`..._paper_noclip_1b-..._distill.pt`) | `386ecbf405b349ff88cf691dcb03b07a55848cd2aa9e7a7f0120920e7589b530` | 442,490 B | 98,384 | float32 (80 tensors) | 4.50 | **1100** | `distill/eval/loss` = **3.9764** |

- **The load is complete**. In all three evaluations, all 80 Lizard parameters (16 layers × 5) are in the file and hold their values after the load.
- **The best checkpoint comes from step 1,100**, the last evaluation of the run. In the float32 run, it also came from step 1,100.
- The summary has no validation curve, because the evaluation did not download a stage 1 results CSV.

#### Scores

| Task | Accuracy | Normalized accuracy | n |
|---|---|---|---|
| MMLU subset (5-shot, 5 questions for each subject) | **26.7 ± 2.6** (76 right) | – | 285 |
| PIQA (0-shot) | **57.3 ± 1.2** | 56.6 ± 1.2 | 1,838 |
| ARC-Easy (0-shot) | **36.5 ± 1.0** | 36.4 ± 1.0 | 2,376 |

Comparison with the earlier fd32 runs:

| Model | Validation loss | MMLU subset | PIQA | ARC-Easy |
|---|---|---|---|---|
| fd32, LoLCATs recipe, bf16 | 3.4219 | 23.2 ± 2.5 | Not measured | Not measured |
| fd32, recipe of the paper, bf16 | 8.1641 | 25.3 ± 2.6 | 55.8 ± 1.2 | 34.1 ± 1.0 |
| fd32, recipe of the paper, float32 | 4.9478 | 24.6 ± 2.5 | 57.7 ± 1.2 | 35.7 ± 1.0 |
| **Config 1 (β2 0.99, minimum learning rate)** | **3.9764** | **26.7 ± 2.6** | **57.3 ± 1.2** | **36.5 ± 1.0** |

| Difference of config 1 | Validation loss | MMLU subset | PIQA | ARC-Easy |
|---|---|---|---|---|
| Against the float32 run | −19.6% | +2.1 points (6 questions), z ≈ 0.6 | −0.4 points (approximately 7 questions), z ≈ −0.2 | +0.8 points (approximately 20 questions), z ≈ 0.6 |
| Against the LoLCATs recipe, fd32 | +16.2% | +3.5 points (10 questions), z ≈ 1.0 | – | – |

The z values use unpaired SEs. The normalized accuracies of the float32 run were 56.3 on PIQA and 34.7 on ARC-Easy.

#### Answer letters on the MMLU subset

| Model | "A" | "B" | "C" | "D" | Letter mass | Confidence | Entropy |
|---|---|---|---|---|---|---|---|
| fd32, LoLCATs recipe, bf16 | 66.0% | 22.1% | 0.7% | 11.2% | 0.024 | 0.476 | 1.681 bits |
| fd32, recipe of the paper, float32 | 53.0% | 2.1% | 12.6% | 32.3% | 0.018 | 0.492 | 1.677 bits |
| Config 1 | 51.2% | 3.9% | 3.9% | 41.1% | 0.019 | 0.473 | 1.726 bits |

The right answers are "A" 24.2%, "B" 24.9%, "C" 25.3% and "D" 25.6%. If the model always selected "A", the accuracy would be 24.2.

#### Gates and Lizard parameters

The gate values come from the same 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens) as in the earlier runs. The last column gives the weight kept after 512 tokens in the float32 run, for comparison.

| Layer | γ mean | γ min | γ above 0.999 | Kept after 128 | Kept after 512 | α | Sink logit (all 4 equal) | ‖W_γ‖ | Float32 run: kept after 512 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.997 | 0.978 | 33.4% | 0.73 | 0.27 | 0.472 | 0.52 | 4.38 | 7.8e-10 |
| 1 | 0.987 | 0.834 | 90.8% | 0.18 | 3.8e-4 | 0.505 | 0.44 | 2.71 | 9.2e-09 |
| 2 | 0.989 | 0.814 | 76.3% | 0.22 | 7.6e-4 | 0.504 | 0.46 | 1.89 | 0.0048 |
| 3 | 0.999 | 0.941 | 89.1% | 0.92 | 0.70 | 0.530 | 0.46 | 1.93 | 0.47 |
| 4 | 1.000 | 0.950 | 94.6% | 0.96 | 0.76 | 0.577 | 0.37 | 1.91 | 0.58 |
| 5 | 1.000 | 0.941 | 99.9% | 1.00 | 0.97 | 0.624 | 0.41 | 1.83 | 0.85 |
| 6 | 1.000 | 0.949 | 99.9% | 0.98 | 0.92 | 0.589 | 0.44 | 1.83 | 0.84 |
| 7 | 0.997 | 0.939 | 11.4% | 0.75 | 0.35 | 0.603 | 0.43 | 2.06 | 0.50 |
| 8 | 0.999 | 0.929 | 37.0% | 0.78 | 0.43 | 0.644 | 0.37 | 1.81 | 0.15 |
| 9 | 0.999 | 0.908 | 43.8% | 0.79 | 0.48 | 0.629 | 0.43 | 1.81 | 0.57 |
| 10 | 0.999 | 0.874 | 84.5% | 0.92 | 0.72 | 0.615 | 0.38 | 1.76 | 0.68 |
| 11 | 1.000 | 0.899 | 99.9% | 0.98 | 0.93 | 0.558 | 0.43 | 1.82 | 0.90 |
| 12 | 1.000 | 0.876 | 99.7% | 0.98 | 0.94 | 0.560 | 0.41 | 1.97 | 0.92 |
| 13 | 1.000 | 0.870 | 97.5% | 0.97 | 0.88 | 0.543 | 0.44 | 2.04 | 0.70 |
| 14 | 0.999 | 0.786 | 65.1% | 0.90 | 0.66 | 0.545 | 0.47 | 1.88 | 0.50 |
| 15 | 0.999 | 0.779 | 77.6% | 0.95 | 0.81 | 0.576 | 0.45 | 1.86 | 0.56 |

No layer has γ below 1e-3 for any token. The feature-map weights have an RMS of 0.108–0.143 for φq and 0.093–0.115 for φk, and a maximum absolute value of 0.25–0.49. In the float32 run, the RMS was 0.086–0.131. In the runs with the LoLCATs recipe, it was 0.15–0.22.

#### Check of the predictions

| Prediction | Result | Holds |
|---|---|---|
| 1. The learning rate stays at 1e-4 or more after the warmup, and the late steps lower the loss more | The summary does not show the learning rate of the run. The CPU test of the schedule gives a minimum of 1.0e-4. The best checkpoint is the last evaluation (step 1,100). | Probably. The run does not measure the learning rate. |
| 2. With β2 = 0.99, the steps become larger sooner | The summary does not show the step sizes. α, the sink logits, ‖W_γ‖ and the feature-map RMS moved farther from their initial values than in the float32 run. | Not measured directly. The parameters agree with larger steps. |
| 3. The validation loss is lower than 4.9478, and the best checkpoint comes from the end of the run | 3.9764 (−19.6%), at step 1,100 | Yes |

#### Findings

**Finding 1: the two settings lower the validation loss, but do not close the difference to the LoLCATs recipe**. The loss is 3.9764, against 4.9478 for the float32 run (−19.6%) and 3.4219 for fd32 with the LoLCATs recipe (+16.2%). The run changes β2 and the minimum learning rate together. Thus it cannot show the effect of each setting alone.

**Finding 2: the validation loss still decreased at the end**. The best step is the last evaluation step, as in the float32 run. Thus a longer run could lower the loss more. This is probable, not proved.

**Finding 3: the parameters moved farther from their initial values in every layer**. Against the float32 run:

- α is lower in all 16 layers, by 0.13–0.17. Its initial value is 1.
- The sink logits are higher in all 16 layers, by 0.13–0.17.
- ‖W_γ‖ is larger in all 16 layers: 1.76–4.38, against 1.33–2.33.
- The RMS of the feature-map weights is a little larger: 0.093–0.143, against 0.086–0.131. It is still smaller than with the LoLCATs recipe (0.15–0.22).

This agrees with larger effective steps (prediction 2) or with more learning in the late steps (prediction 1). The run cannot separate the two causes.

**Finding 4: the gate keeps more history than in the float32 run, but does not saturate fully**:

- The weight kept after 512 tokens is higher in 13 of the 16 layers. Layers 2, 7 and 9 keep less.
- Layers 5, 6, 11, 12 and 13 keep 0.88–0.97 of the weight after 512 tokens. Their γ is above 0.999 for 97.5–99.9% of the tokens.
- Layers 1 and 2 still decay fast. They keep 7.6e-4 or less after 512 tokens.
- Layer 0 changed most. It keeps 0.27 after 512 tokens, against 7.8e-10 in the float32 run. Its ‖W_γ‖ is the largest (4.38).
- No layer of 1–15 has γ above 0.999 for 100.0% of the tokens. With the LoLCATs recipe, 14 of the 15 layers did.

**Finding 5: MMLU stays at the level of chance**. The accuracy is 26.7, the highest of the stage 1 models so far. But it is only 0.65 SEs above 25%, and only z ≈ 0.6 above the float32 run. The model selects "A" or "D" for 92.3% of the questions, and "B" and "C" for 3.9% each. The letter mass is 0.019. In all stage 1 models so far, the accuracy is between 22.5 and 26.7.

**Finding 6: PIQA and ARC-Easy do not change clearly**. PIQA is 57.3, against 57.7 in the float32 run (z ≈ −0.2). ARC-Easy is 36.5, against 35.7 (z ≈ 0.6). Thus the lower validation loss gives no measurable change on these tasks after stage 1.

### Interpretation

- **The outcome is the second row of the outcome table**. The validation loss is between 3.4219 and 4.9478. Thus β2 and the minimum learning rate of the paper help, but do not close the difference to the LoLCATs recipe.
- **The stage 1 recipe progression with fd32** is now: 8.1641 (recipe of the paper, bf16), 4.9478 (float32), 3.9764 (float32, β2 0.99, minimum learning rate). The LoLCATs recipe gives 3.4219 with fd32 and 3.2549 with fd128.
- **The remaining difference has three possible explanations**. This run cannot separate them:
  - **Gradient clipping** is the last optimizer setting of Table 13 that is still different. The run with clipping (config 2) is the [gradient clipping experiment](gradient-clipping.md). It lowered the validation loss to 3.5092.
  - **The gate state**. The runs with the lowest loss (LoLCATs recipe) have a saturated gate. In this run, the gate keeps more history than in the float32 run, and the loss is lower. This agrees with the hypothesis of section 13.2 of the [gap analysis](../11-gap-analysis.md). In that hypothesis, a normalized gated branch reaches a lower loss when it can reach the BOS tokens. The agreement does not prove the hypothesis.
  - **Less training**. The loss still decreased at the last evaluation (finding 2), and the feature-map weights are still smaller than with the LoLCATs recipe (finding 3).
- **The validation loss does not predict the accuracy after stage 1**. The fd32 stage 1 models have validation losses from 3.42 to 8.16. Their accuracies stay in small ranges: 23.2–26.7 on the MMLU subset, 55.8–57.7 on PIQA and 34.1–36.5 on ARC-Easy. With unpaired SEs, none of the differences is clear.
- This result is preliminary. It comes from one seed, and the comparisons use unpaired SEs.

### Open items

1. **Config 2 (gradient clipping)**: done. The [gradient clipping experiment](gradient-clipping.md) gives a validation loss of 3.5092. The logged gradient norm of the whole run is still missing. It shows how often a limit of 1.0 clips with `mse_factor` 1000.
2. **Separate the effects of β2 and the minimum learning rate**, if the difference is necessary for the thesis. Each needs one more run with only one of the two settings.
3. **The gate state and the normalization of the gated branch** (D1 in [math against code](../math-code-discrepancy.md)). Test them with the single-layer bench of section 13.5 of the gap analysis (step 5).
4. **More sensitive measures for stage 1**: the error of each layer against the teacher, and the KL divergence of the logits. Steps 1 and 3 of section 13.5 of the gap analysis describe them. Paired comparisons on the same questions would also help.
5. **Feature dimension 128** with this recipe, after the run with clipping.
6. **Validation curve**: the evaluation did not download a stage 1 results CSV. A curve would show how strongly the loss still decreased at the end.
