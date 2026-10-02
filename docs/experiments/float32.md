# Experiment: float32 in stage 1, with the learning rate of the paper and feature dimension 32

**Status:** Stage 1 trained on HF Jobs. The evaluations on the MMLU subset, PIQA and ARC-Easy ran on 2026-10-01.

## Question

Does float32 fix the problems of the [paper-LR run](paper-lr.md)? That run used the learning rate and the schedule of the paper, with all weights in bf16. It had these problems:

- α stayed at exactly 1.000 in all 16 layers (finding 2 there).
- The stage 1 validation loss was 8.1641, against 3.4219 with the LoLCATs recipe (finding 3).
- The validation loss did not improve after step 700 (finding 4).

The same run showed that the learning rate of the paper stops the saturation of the gate (finding 1). This experiment asks if the gate stays unsaturated and if the attention approximation becomes better than with the LoLCATs recipe, when bf16 rounding is absent.

Background:

- Factor 0 of section 12 of the [gap analysis](../11-gap-analysis.md) found that bf16 storage discards small updates. At a learning rate of 1e-3, it discards all updates of α.
- [Math against code](../math-code-discrepancy.md) lists three more effects of bf16:
  - the gate logit in bf16 (D9),
  - the target of stage 1 in bf16 (D10),
  - the prediction rounded to bf16 before the loss (D13).

## What changes, and what stays the same

| Setting | Paper-LR run | This experiment |
|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd32_m4` | **`distill_llama3_2_1b_lizard_w128_fd32_m4_fp32`** |
| `torch_dtype` (all weights of the model) | bf16 | **float32** |
| `attn_implementation` of the loaded model | `flash_attention_2` | **`eager`** |
| Trainable Lizard weights and their AdamW states | bf16 | **float32** |
| Gate logit $\mathbf{W}_\gamma \mathbf{x}$ (D9) | bf16 | **float32** |
| Target of stage 1 (D10) and prediction before the loss (D13) | bf16 | **float32** |
| Calculation of the Lizard attention | float32 (`upcast`) | float32 |
| Distill config | `distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b` | The same |
| Learning rate, schedule, loss factor | 1e-3 peak, cosine, 118 warmup steps, `mse_factor` 1000 | The same |
| Feature dimension, window, sinks | 32, 128, 4 | The same |
| Data, steps, seed | Alpaca-cleaned, 1,178 steps, seed 0 | The same |

Only the precision changes against the paper-LR run. Thus the comparison with that run shows the effect of bf16 alone.

**`attn_implementation`:** FlashAttention-2 supports only fp16 and bf16. The setting has no effect on stage 1. Lizard attention replaces every attention layer, and the target of stage 1 comes from `softmax_attention`, not from FlashAttention. The setting has an effect only on an evaluation of the teacher with this config.

### Why the whole model, and not only the trainable weights

- The model config gives one dtype for the whole model. Lizard attention copies the dtype of `q_proj` (`src/model/linear_attention/lizard_attention.py`, lines 148–149). AdamW creates its states with the dtype of each weight. Thus `torch_dtype: float32` makes all of these float32 without a code change.
- The trainer flag `bf16: true` of the distill config has no effect. The trainer has no autocast (factor 0 of section 12).
- float32 for the whole model also removes D9, D10 and D13. A code change could keep only the approximately 100,000 trainable weights in float32. That would be faster, but it would keep D9, D10 and D13.
- **Cost:** The weights of the model need approximately 5 GB in float32, against approximately 2.5 GB in bf16. The matrix multiplications of the frozen model become slower. The Lizard calculations were already float32.

### Differences from the paper that stay

- The paper gives bf16 as its precision (Table 13). It trained with FSDP-2, which normally keeps float32 master weights and calculates in bf16. This run calculates in float32 too. Thus it is more precise than the probable setup of the paper.
- These settings also stay different from the paper:
  - feature dimension 32, not 128,
  - β2 = 0.999, not 0.99,
  - no gradient clipping, not 1.0,
  - a decay to 0, not to 0.1 × the peak.

  The [paper-LR document](paper-lr.md) explains these differences.

## Predictions

If bf16 rounding caused the problems of the paper-LR run:

1. α moves away from 1.000 in most layers.
2. The best checkpoint comes from a later step than 700, because the updates after step 700 are no longer lost.
3. The validation loss is lower than 8.1641.
4. The gate stays unsaturated, because the learning rate and the schedule do not change.

## How to run

### Option A: HF Jobs (H200), recommended

1. **Build a new Docker image.** Merge the PR with the new config first. Then select Actions → "Docker image" → Run workflow. HF Jobs runs only the code inside the image ([document 3](../03-infrastructure.md)).
2. **Train stage 1 on HF Jobs:**

   ```bash
   make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=3h \
     ARGS="--model_config distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b --no_finetune"
   ```

   - Stage 1 in bf16 needed approximately 50 minutes on the H200 ([document 2](../02-compute-and-cost.md)). float32 makes the frozen model slower. Thus `HF_TIMEOUT=3h` gives a margin. This document does not have a measured time for float32.

### Option B: a local GPU

```bash
nohup make distill-local LOCAL_MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 \
  DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b > distill-fp32.log 2>&1 &
```

[Document 2](../02-compute-and-cost.md) estimated 22–25 GB for stage 1 in bf16. float32 needs more. Thus the A10 (23,028 MiB) probably does not have sufficient memory. Use option A, or a GPU with more memory.

### Checkpoint

```
checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
```

The checkpoint stores the 98,384 Lizard parameters in float32. Thus it has approximately 4 bytes for each parameter, against approximately 2.5 in the bf16 runs.

### Evaluation on the A10

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b \
MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 \
MODELS=stage1 TASKS="mmlu_subset piqa arc_easy" \
scripts/compare_stages.sh 2>&1 | tee eval-fp32.log
```

The evaluation loads the model config from the checkpoint name. Thus the evaluation also runs in float32. The weights need approximately 5 GB, and the evaluation has no gradients and no optimizer states.

## How to compare

All runs use the same validation data and the same loss (`mse_factor` 1000). Thus their validation losses are directly comparable.

| Measure | fd32, LoLCATs recipe, bf16 | fd32, recipe of the paper, bf16 | fd32, recipe of the paper, float32 |
|---|---|---|---|
| Stored stage 1 validation loss, and its step | 3.4219 at step 1,100 | 8.1641 at step 700 | **4.9478 at step 1,100** |
| MMLU-subset accuracy | 23.2 ± 2.5 | 25.3 ± 2.6 | 24.6 ± 2.5 |
| Share of "A" answers | 66.0% | 21.8% | 53.0% |
| Letter mass | 0.024 | 0.009 | 0.018 |
| PIQA accuracy | Not measured yet | 55.8 ± 1.2 | 57.7 ± 1.2 |
| ARC-Easy accuracy | Not measured yet | 34.1 ± 1.0 | 35.7 ± 1.0 |
| Layers 1–15 with γ above 0.999 for 100.0% of the tokens | 14 of 15 | 0 of 15 | 0 of 15 (3 layers at 99.9%) |
| α | 0.063–0.656 | 1.000 in all 16 layers | **0.602–0.786** |
| Sink logits exactly at a power of two | 10 of 16 layers | 16 of 16 layers | No pattern (2 of 16 print as 0.25) |

How to read the outcome:

| Outcome | Conclusion | Next step |
|---|---|---|
| The validation loss is lower than 3.4219, and the gate does not saturate | The recipe of the paper with float32 gives a better stage 1 than the LoLCATs recipe | Run stage 2 with the recipe of the paper, also in float32 |
| The validation loss is lower than 8.1641 but higher than 3.4219, and the gate does not saturate | bf16 caused part of the problem. A decaying gate still gives a higher loss than a saturated gate. | Test the normalization of the gated branch (D1 in [math against code](../math-code-discrepancy.md)) and the sink hypothesis of section 13.2 of the gap analysis |
| The validation loss stays near 8.1641 | bf16 is not the main cause of the high loss | Examine the start value of the gate and the normalization of the gated branch (section 13.4 of the gap analysis) |

## Code changes

- `configs/model/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32.yaml`: a copy of `distill_llama3_2_1b_lizard_w128_fd32_m4.yaml` with two changes:
  - `torch_dtype: float32`,
  - `attn_implementation: eager`.

There is no change to the code.

## Tests

These tests ran on CPU with a tiny Llama.

- **Precision and one AdamW step**. A script loaded the tiny Llama with each model config. It used the loader of the repository (`get_pretrained_loader`, and `load_and_convert_attns` in stage 1 mode). Then it gave every trainable weight a gradient of 1 and took one AdamW step at a learning rate of 1e-3:

  | Model config | Frozen weights | Trainable Lizard weights | AdamW states | Target and prediction of stage 1 | α after one step |
  |---|---|---|---|---|---|
  | `..._fd32_m4` (bf16) | bf16 | bf16 | bf16 | bf16 | 1.000000 (no change) |
  | `..._fd32_m4_fp32` | float32 | float32 | float32 | float32 | 0.999000 |

  With bf16, rounding discards the step of α. This agrees with the α of exactly 1.000 in the paper-LR run. With float32, α moves by the full step.
- **Stage 1 on CPU.** The real `distill_llama.main()` ran with the arguments of option A and synthetic data. It finished with exit 0 and skipped stage 2. The log shows `torch_dtype: float32` and `attn_implementation: eager`, and the learning rates of gradient steps 1 and 2 agree with the warmup. The run name contains `-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32`.
- **Evaluation.** `scripts/compare_stages.sh` ran with this model config on a float32 test checkpoint with the name above. It finished with exit 0. The summary shows the dtype float32, and all trainable parameters loaded.
- The CPU tests used `attn_implementation: eager` and the optimizer `adamw_torch`, because FlashAttention-2 and the fused optimizer need CUDA.

## Results

### Run 1: MMLU subset (2026-10-01)

- Training: stage 1 only, on HF Jobs (option A). The job pushed the checkpoint to `nanoman1/lolcats-lizard-llama-3.2-1b`. This document does not record the job ID or the training time.
- Evaluation command: the command in "Evaluation on the A10" above, with `TASKS=mmlu_subset`
- Run directory: `results/stages/20261002-013135`, on `student06`, GPU 0 (A10)
- Code: lolcats `c2293a4`, harness `b281b09`
- Software: Python 3.11.16, torch 2.5.1 (CUDA 12.4), transformers 4.43.1, peft 0.9.0

#### Checkpoint

| Checkpoint | SHA-256 | Size | Parameters | Dtype | Bytes per parameter | Stored step | Stored loss |
|---|---|---|---|---|---|---|---|
| Stage 1, fd32, recipe of the paper, float32 (`..._distill.pt`) | `75e6eddb923bf62ac736154a13b1d12df875f39c3705d0fa55d6dddfb1a03090` | 441,986 B | 98,384 | **float32** (80 tensors) | 4.49 | **1100** | `distill/eval/loss` = 4.9478 |

- **The load is complete**. All 80 Lizard parameters (16 layers × 5) are in the file and hold their values after the load.
- **The checkpoint is float32**. It has 4.49 bytes for each parameter, against 2.49 for the same model in bf16. The difference is the 2 extra bytes of each float32 value.
- **The best checkpoint comes from step 1,100**, the last evaluation of the run. In the bf16 run with the same recipe, it came from step 700.
- The Hub also has `..._distill_1000.pt`, the periodic save at step 1,000. The Hub has no stage 1 results CSV for this run under the expected name. Thus the summary has no validation curve.

#### Scores and answer letters

| Model | Right answers | Accuracy | "A" | "B" | "C" | "D" | Letter mass | Confidence | Entropy |
|---|---|---|---|---|---|---|---|---|---|
| fd32, LoLCATs recipe, bf16 | 66 / 285 | 23.2 ± 2.5 | 66.0% | 22.1% | 0.7% | 11.2% | 0.024 | 0.476 | 1.681 bits |
| fd32, recipe of the paper, bf16 | 72 / 285 | 25.3 ± 2.6 | 21.8% | 2.8% | 36.8% | 38.6% | 0.009 | 0.494 | 1.664 bits |
| fd32, recipe of the paper, float32 | 70 / 285 | **24.6 ± 2.5** | 53.0% | 2.1% | 12.6% | 32.3% | 0.018 | 0.492 | 1.677 bits |
| Teacher ([document 7](../07-results.md)) | 96 / 285 | 33.7 ± 2.8 | – | – | – | – | – | – | – |

The right answers are "A" 24.2%, "B" 24.9%, "C" 25.3% and "D" 25.6%. Against the fd32 run with the LoLCATs recipe, the difference is +1.4 points (z ≈ 0.4). Against the bf16 run with the same recipe, it is −0.7 points (z ≈ −0.2). These use unpaired SEs. Thus the three accuracies are not clearly different.

#### Gates and Lizard parameters

The gate values come from the same 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens) as in the earlier runs.

| Layer | γ mean | γ min | γ above 0.999 | Kept after 128 | Kept after 512 | α | Sink logit (all 4 equal) | ‖W_γ‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.961 | 0.827 | 0.0% | 0.0052 | 7.8e-10 | 0.602 | 0.39 | 2.33 |
| 1 | 0.972 | 0.693 | 53.0% | 0.016 | 9.2e-09 | 0.659 | 0.28 | 2.13 |
| 2 | 0.990 | 0.449 | 12.9% | 0.26 | 0.0048 | 0.658 | 0.30 | 1.33 |
| 3 | 0.998 | 0.806 | 22.9% | 0.81 | 0.47 | 0.693 | 0.30 | 1.40 |
| 4 | 0.999 | 0.875 | 69.6% | 0.86 | 0.58 | 0.747 | 0.20 | 1.47 |
| 5 | 1.000 | 0.905 | 99.9% | 0.97 | 0.85 | 0.779 | 0.25 | 1.53 |
| 6 | 1.000 | 0.923 | 99.9% | 0.96 | 0.84 | 0.748 | 0.28 | 1.57 |
| 7 | 0.999 | 0.898 | 26.1% | 0.86 | 0.50 | 0.761 | 0.26 | 1.75 |
| 8 | 0.998 | 0.894 | 22.8% | 0.66 | 0.15 | 0.786 | 0.21 | 1.57 |
| 9 | 0.999 | 0.859 | 73.7% | 0.87 | 0.57 | 0.779 | 0.27 | 1.55 |
| 10 | 0.999 | 0.761 | 97.5% | 0.89 | 0.68 | 0.766 | 0.22 | 1.38 |
| 11 | 1.000 | 0.804 | 99.9% | 0.97 | 0.90 | 0.715 | 0.27 | 1.42 |
| 12 | 0.999 | 0.740 | 96.8% | 0.98 | 0.92 | 0.720 | 0.25 | 1.46 |
| 13 | 0.999 | 0.720 | 90.5% | 0.92 | 0.70 | 0.702 | 0.28 | 1.45 |
| 14 | 0.998 | 0.642 | 6.7% | 0.83 | 0.50 | 0.697 | 0.33 | 1.36 |
| 15 | 0.998 | 0.833 | 21.7% | 0.87 | 0.56 | 0.728 | 0.30 | 1.45 |

The feature-map weights have an RMS of 0.086–0.131 and a maximum absolute value of 0.23–0.54. In the bf16 run with the same recipe, the RMS was 0.075–0.115. In the runs with the LoLCATs recipe, it was 0.15–0.22.

#### Check of the predictions

| Prediction | Result | Holds |
|---|---|---|
| 1. α moves away from 1.000 in most layers | α is 0.602–0.786 in all 16 layers | Yes |
| 2. The best checkpoint comes from a later step than 700 | Step 1,100 | Yes |
| 3. The validation loss is lower than 8.1641 | 4.9478 (−39%) | Yes |
| 4. The gate stays unsaturated | No layer has γ above 0.999 for all tokens. The weight kept after 512 tokens is below 0.95 in every layer. But 3 layers have 99.9% of the tokens above 0.999. | Mostly |

#### Findings

**Finding 1: float32 removes the symptoms of bf16 rounding**. α moves to 0.60–0.79 in every layer. The validation loss improves until the last evaluation (step 1,100). The sink logits no longer stop at powers of two. Thus bf16 rounding caused the frozen α and the early stop of the paper-LR run.

**Finding 2: the validation loss is much lower than with bf16, but higher than with the LoLCATs recipe**. It is 4.9478, against 8.1641 for the same recipe in bf16 (−39%) and 3.4219 for fd32 with the LoLCATs recipe (+45%). Thus float32 explains a large part of the high loss of the paper-LR run, but not all of the difference to the LoLCATs recipe.

**Finding 3: the gate takes an intermediate state**. With the LoLCATs recipe, the gate saturated in 14 or 15 of the layers 1–15. With the recipe of the paper in bf16, it decayed fast in every layer. In this run:

- Layers 0–2 decay fast. The weight kept after 512 tokens is 0.0048 or less.
- Layers 5, 6, 11 and 12 keep 0.84–0.92 of the weight after 512 tokens. Their γ is above 0.999 for 96.8–99.9% of the tokens.
- The other layers keep 0.15–0.70 after 512 tokens.

**Finding 4: the validation loss still decreased at the end**. The best step is the last evaluation step. Thus a longer run, or a higher average learning rate, could lower the loss more. This is probable, not proved. The cosine schedule decays to 0, so the learning rate is very small in the last part of the run.

**Finding 5: MMLU stays at the level of chance**. The accuracy is 24.6, and the letter mass is 0.018. The model selects "A" for 53.0% of the questions and almost never selects "B" (2.1%). In all four stage 1 models so far, the accuracy is between 22.5 and 25.3.

### Run 2: PIQA and ARC-Easy (2026-10-01)

- Evaluation command: the command in "Evaluation on the A10" above, with `TASKS="piqa arc_easy"`
- Run directory: `results/stages/20261002-014647`, on `student06`, GPU 0 (A10)
- Code, software and checkpoint: the same as in run 1. The checkpoint has the same SHA-256 (`75e6eddb…`). All 80 Lizard parameters loaded in both evaluations. The gate table is identical to run 1.

| Task (0-shot) | Accuracy | Normalized accuracy | n |
|---|---|---|---|
| PIQA | **57.7 ± 1.2** | 56.3 ± 1.2 | 1,838 |
| ARC-Easy | **35.7 ± 1.0** | 34.7 ± 1.0 | 2,376 |

Comparison with the other stage 1 models and the references:

| Model | PIQA | ARC-Easy | Source |
|---|---|---|---|
| fd128, LoLCATs recipe, bf16 | 57.6 ± 1.2 | Not measured | [Stage difference](stage-difference.md), quick check 1 |
| fd32, recipe of the paper, bf16 | 55.8 ± 1.2 | 34.1 ± 1.0 | [Paper-LR experiment](paper-lr.md), run 2 |
| fd32, recipe of the paper, float32 | **57.7 ± 1.2** | **35.7 ± 1.0** | This run |
| fd128, after stage 2 (the Lizard model) | 67.95 ± 1.09 | 54.8 | [Document 7](../07-results.md) |
| Teacher, in the paper | 74.1 | 65.4 | Table 9 of the paper, newer harness |
| Chance | 50.0 | Approximately 25 | 2 choices on PIQA. Mostly 4 choices on ARC-Easy. |

| Difference | PIQA | ARC-Easy |
|---|---|---|
| float32 against bf16, same recipe | +1.9 points (approximately 35 questions), z ≈ 1.2 | +1.6 points (approximately 38 questions), z ≈ 1.2 |
| float32 against fd128 with the LoLCATs recipe | +0.1 points, z ≈ 0.1 | – |

The z values use unpaired SEs. A paired comparison on the same questions would be more sensitive, but `compare_stages.py` compares models only inside one run.

**Finding 6: PIQA and ARC-Easy are a little higher than with bf16, but the difference is not clear**. PIQA is 57.7 against 55.8, and ARC-Easy is 35.7 against 34.1 (z ≈ 1.2 for both). Both differences have the same direction as the lower validation loss (4.95 against 8.16).

**Finding 7: PIQA is the same as with the LoLCATs recipe**. This run gets 57.7, and fd128 with the LoLCATs recipe got 57.6 after stage 1. Thus the lower validation loss of the LoLCATs recipe (3.25 for fd128, 3.42 for fd32) does not give a higher PIQA accuracy after stage 1.

**Finding 8: all stage 1 models stay far below the stage 2 model and the teacher**. PIQA is 10 points below the stage 2 model (67.95) and 16 points below the teacher (74.1 in the paper). ARC-Easy is 19 points below the stage 2 model (54.8) and 30 points below the teacher (65.4).

### Interpretation

- **The outcome is the second row of the outcome table**. The validation loss is lower than 8.1641 but higher than 3.4219, and no layer of 1–15 saturates fully. Thus bf16 caused part of the problem. The recipe of the paper with float32 still gives a higher loss than the LoLCATs recipe.
- **Three explanations for the remaining difference** are possible. This run cannot separate them:
  - **The gate state**. The LoLCATs runs, with a saturated gate, have the lowest loss. This agrees with the hypothesis of section 13.2 of the [gap analysis](../11-gap-analysis.md). In that hypothesis, a normalized gated branch reaches a lower loss when it can reach the BOS tokens.
  - **Less training**. At a peak learning rate of 1e-3 with cosine decay, the weights can move much less in 1,178 steps than at a constant 1e-2. Factor 1 of section 12 gives this estimate. The feature-map weights are smaller than with the LoLCATs recipe, and the loss still decreased at the end (finding 4).
  - **The other differences from the paper**: β2 = 0.999, no gradient clipping, the decay to 0 and feature dimension 32.
- The removal of the bf16 rounding of the target (D10) and of the prediction (D13) can also lower the loss a little. Thus a small part of the improvement from 8.16 to 4.95 can come from the precision of the loss, not from better weights. This document has no measurement of this part.
- **MMLU does not follow the validation loss**. The four stage 1 models have validation losses from 3.25 to 8.16, but their MMLU-subset accuracies are all near chance. Thus the MMLU subset does not separate these stage 1 models.
- **PIQA and ARC-Easy separate the stage 1 models only a little**. On PIQA, all measured values are in a range of 2 points (55.8–57.7). On ARC-Easy, the range is 1.6 points (34.1–35.7). None of the differences is clear with unpaired SEs. Thus none of the recipes gives a stage 1 model that is clearly better on these tasks. But their validation losses differ by up to 2.5×.
- **The stage 1 model alone is far from the teacher on every task** (finding 8). With the current recipes, stage 2 recovers most of the measured accuracy. A better stage 1 recipe must show its effect after stage 2, or with a more sensitive measure than these tasks. Examples are the error of each layer against the teacher, and the KL divergence of the logits (section 13.5 of the [gap analysis](../11-gap-analysis.md)).
- This result is preliminary. It uses only the MMLU subset, and the comparisons use unpaired SEs.

### Open items

1. **More sensitive measures for stage 1**: the error of each layer against the teacher, and the KL divergence of the logits. Steps 1 and 3 of section 13.5 of the gap analysis describe them. Paired comparisons on the same questions would also help.
2. **Separate the explanations of the remaining difference:**
   - The normalization of the gated branch (D1 in [math against code](../math-code-discrepancy.md)) and the sink hypothesis. Test them with the single-layer bench of section 13.5 of the gap analysis (step 5).
   - A float32 run with the LoLCATs recipe. It shows whether float32 changes the result of the high learning rate too.
   - The remaining differences from the paper: β2 = 0.99, gradient clipping 1.0, and a minimum learning rate of 0.1 × the peak. Clipping and the minimum learning rate need code changes.
3. **Validation curve:** the Hub has no results CSV for this run under the expected name. A curve would show whether the loss still decreased strongly at the end.
