# Experiment: gradient clipping in stage 1

**Status:** Stage 1 training with config 2 is running (2026-10-02). In epoch 0, the logged gradient norm is approximately 3.6, so clipping acts on these steps. The run has no validation loss yet.

## Question

The [second round](second-round.md) changed β2 and the minimum learning rate to the values of the paper (config 1). The stage 1 validation loss dropped from 4.9478 ([float32 experiment](float32.md)) to 3.9764. With the LoLCATs recipe, the loss was 3.4219.

One optimizer setting of the paper (Table 13) is still different:

| Setting | Config 1 (second round) | Paper |
|---|---|---|
| Gradient clipping | None | 1.0 |

This experiment adds gradient clipping at 1.0 to config 1. The result is config 2. Does stage 1 then become better? Clipping is the only difference between the two configs. Thus their comparison shows the effect of clipping alone.

## What changes, and what stays the same

| Setting | Config 1 (second round) | Config 2 (this experiment) |
|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd32_m4_fp32` | The same |
| Distill config | `distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b` | **`distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_1b`** |
| Gradient clipping | None | **Global norm 1.0 (`trainer.max_grad_norm: 1.0`)** |
| Precision | float32 for the whole model | The same |
| Peak learning rate, warmup | 1e-3, 118 steps (10%), linear | The same |
| Schedule | Cosine to 1e-4 (`cosine_warmup_min_lr`, `min_lr_rate: 0.1`) | The same |
| AdamW betas, eps | (0.9, 0.99), 1e-8 | The same |
| Feature dimension, window, sinks | 32, 128, 4 | The same |
| Loss | 1000 × MSE (`mse_factor` 1000) | The same |
| Data, steps, seed | Alpaca-cleaned, 1,178 steps, seed 0 | The same |

With config 2, the stage 1 recipe agrees with Table 13 of the paper, except for two settings:

- **Feature dimension 32**, not 128. The [feature dimension experiment](feature-dimension.md) found a small effect: +5% validation loss with 32 under the LoLCATs recipe.
- **The scale of the loss**, which the paper does not give clearly ([math against code](../math-code-discrepancy.md), D11). The next section explains why this scale matters for clipping.

### Gradient clipping depends on the scale of the loss

Adam does not change when the loss gets a constant factor, except through eps (factor 8 of section 12 of the [gap analysis](../11-gap-analysis.md)). Gradient clipping does change, because the threshold 1.0 is an absolute value:

- If the global gradient norm is always above 1.0, clipping acts on every step. Then each step uses the direction of the gradient with the norm 1.0. Adam then sees gradients of equal size in every step.
- If the norm is always below 1.0, clipping has no effect.
- The norm depends on `mse_factor`. Our loss uses the mean of the squared errors in each layer, × 1000. The written loss of the paper uses the sum, which is 4,194,304 / 1000 ≈ 4,194 times larger. Thus the same threshold of 1.0 can act differently in the paper and in this run.

To show which case occurs, the trainer now logs the global gradient norm before clipping (`grad norm` in the progress line, `train/grad_norm` in W&B). In the CPU test with a tiny random model, the norm was approximately 3,856. That value does not predict the norm of the real model.

How to read the logged gradient norm:

| Logged gradient norm | Meaning |
|---|---|
| Above 1.0 in almost every step | Clipping acts on every step. Each update uses a gradient with the norm 1.0. |
| Above 1.0 only at the start, or only in some steps | Clipping limits only the large steps. |
| Below 1.0 in almost every step | Clipping has almost no effect. Config 2 then gives approximately the result of config 1. |

## Predictions

1. **The gradient norm:** the logged norm is above 1.0 in most steps, so clipping acts on most steps. This prediction is weak. It comes only from the factor 1000 on the loss and from the CPU test with a tiny random model.
2. **The validation loss:** the effect of clipping is small. The loss of config 2 is within approximately 5% of 3.9764. Reason: clipping multiplies all gradients of one step by the same factor. Adam divides each gradient by the root of its second-moment estimate. Thus a constant factor on all steps has no effect. Clipping changes only the relative weights of the steps in the moment estimates of Adam. The effect is largest when the gradient norm changes much from step to step.
3. **The best checkpoint** comes from the end of the run (step 1,100), as for config 1.
4. **The accuracies** on the MMLU subset, PIQA and ARC-Easy do not change clearly.

## How to run

### Option A: HF Jobs (H200), recommended

1. **Build a new Docker image.** Merge the PR with the code and the config first. Then select Actions → "Docker image" → Run workflow. HF Jobs runs only the code inside the image ([document 3](../03-infrastructure.md)).
2. **Train stage 1 on HF Jobs:**

   ```bash
   make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=3h \
     ARGS="--model_config distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_1b --no_finetune"
   ```

3. **Record the gradient norm.** The log of the job (`hf jobs logs <job id>`) shows `grad norm: …` in the progress line of each step. Record a few values from the start, the middle and the end of the run.

The run needs approximately the same time and memory as config 1. Clipping adds one norm calculation for each optimizer step.

### Checkpoint

```
checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
```

The folder is the same as for config 1. The name contains the distill config of this experiment. Thus no earlier checkpoint gets overwritten.

### Evaluation on the A10

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_1b \
MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 \
MODELS=stage1 TASKS="mmlu_subset piqa arc_easy" \
scripts/compare_stages.sh 2>&1 | tee eval-gradient-clipping.log
```

## How to compare

All runs use the same validation data and the same loss (`mse_factor` 1000). Thus their validation losses are directly comparable.

| Measure | fd32, LoLCATs recipe, bf16 | fd32, recipe of the paper, float32 | Config 1 (second round) | Config 2 (this experiment) |
|---|---|---|---|---|
| Stored stage 1 validation loss, and its step | 3.4219 at step 1,100 | 4.9478 at step 1,100 | 3.9764 at step 1,100 | Not measured yet |
| MMLU-subset accuracy | 23.2 ± 2.5 | 24.6 ± 2.5 | 26.7 ± 2.6 | Not measured yet |
| Share of "A" answers | 66.0% | 53.0% | 51.2% | Not measured yet |
| PIQA accuracy | Not measured yet | 57.7 ± 1.2 | 57.3 ± 1.2 | Not measured yet |
| ARC-Easy accuracy | Not measured yet | 35.7 ± 1.0 | 36.5 ± 1.0 | Not measured yet |
| Layers 1–15 with γ above 0.999 for 100.0% of the tokens | 14 of 15 | 0 of 15 (3 layers at 99.9%) | 0 of 15 (3 layers at 99.9%) | Not measured yet |
| α | 0.063–0.656 | 0.602–0.786 | 0.472–0.644 | Not measured yet |
| Logged gradient norm before clipping | Not logged | Not logged | Not logged | Not measured yet |

How to read the outcome:

| Outcome | Conclusion | Next step |
|---|---|---|
| The validation loss is lower than 3.4219 | With the full optimizer recipe of the paper, stage 1 is better than with the LoLCATs recipe | Run stage 2 with config 2. Then try feature dimension 128. |
| The validation loss is between 3.4219 and approximately 3.78 (5% below config 1) | Clipping helps, but does not close the difference | Try feature dimension 128 with config 2. Examine the gate state and the normalization of the gated branch (D1). |
| The validation loss is within approximately 5% of 3.9764 | Clipping has no large effect. The remaining difference does not come from the optimizer settings of Table 13. | Examine D1 and the sink hypothesis (section 13.2 of the gap analysis). Try feature dimension 128. |
| The validation loss is higher than approximately 4.18 (5% above config 1) | Clipping at 1.0 makes stage 1 worse at this loss scale | Examine the logged gradient norm. Keep config 1 as the stage 1 recipe. |

The 5% limits are a rule of thumb. This project has only one seed for each config. Thus it has no measurement of the variation between seeds.

## Code changes

These changes keep the behavior of all existing configs:

- `src/trainer/default_lm.py`:
  - The trainer has the new argument `max_grad_norm` (line 54). The default is `None`, which means no clipping, as before.
  - With a value, the trainer clips the global gradient norm of all weights with a gradient (lines 178–180). The clip occurs after the accumulation of all gradients of an optimizer step, and before `optimizer.step()`.
  - The trainer writes the norm before clipping into the progress line (`grad norm`) and into `train/grad_norm` for W&B (lines 196–197 and 208–209).
- `configs/experiment/distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_1b.yaml` (config 2): config 1 plus `trainer.max_grad_norm: 1.0`.

## Tests

These tests ran on CPU with a tiny Llama.

- **Gradient clipping in the real training loop.** The real `distill_llama.main()` ran with the arguments of option A, the float32 model config and synthetic data. A test wrapper recorded every call of `clip_grad_norm_` and the gradient norm after it:

  | Distill config | Calls | Norm before clipping | Norm after clipping |
  |---|---|---|---|
  | Config 2 (`max_grad_norm: 1.0`) | 2 (1 for each optimizer step) | 3,856 | 1.000 |
  | Test copy of config 2 with `max_grad_norm: 0.001` | 2 | 3,856 | 0.001 |
  | Config 1 (no `max_grad_norm`) | 0 | – | – |

  All three runs finished with exit 0 and skipped stage 2. The log shows `grad norm: 3856.276` only for the runs with clipping. The two optimizer steps have the same norm, because the first step of the warmup has a learning rate of 0. Thus the weights do not change between the two steps of this short test.
- The CPU tests used the optimizer `adamw_torch`, because the fused optimizer needs CUDA.

## Results

### Run 1, in progress: the training log of epoch 0 (2026-10-02)

This section uses nine progress lines from the job log of epoch 0, at gradient steps 181 and 182 (micro-batches 1,453–1,459 of 4,714). The lines show six different micro-batches.

#### How to read the progress line

| Field | Meaning (from `src/trainer/default_lm.py` and `distill_attention_xent_mse.py`) | Values in the log |
|---|---|---|
| `loss` | The mean over the epoch so far of the micro-batch loss **divided by 8** (`gradient_accumulation_steps`). It is not on the scale of the validation loss. | 4.324 → 4.312. × 8 = 34.6 → 34.5. |
| `loss_mse` | 1000 × the mean MSE over the 16 layers, for the current micro-batch only. It is on the same scale as `distill/eval/loss`. | 6.777–7.628 (mean 7.24) |
| `lr` | The learning rate after the last optimizer step | 0.00099. The cosine schedule gives 9.92e-4 at step 181. |
| `gradient step` | The number of optimizer steps. One optimizer step uses 8 micro-batches. Thus one epoch has 589 optimizer steps. | 181, 182 |
| `grad norm` | The global gradient norm **before** clipping, at the last optimizer step. All lines between two optimizer steps show the same value. | 3.578 (step 181), 3.574 (step 182) |

The `loss` field changes slowly, because it is a mean over all micro-batches of the epoch. Its mean of 34.6 is much higher than the current `loss_mse` of approximately 7. Thus the first micro-batches of the epoch had a much higher loss. The progress line does not show these values.

#### Findings so far

**Finding 1: clipping acts on these steps**. The gradient norm is 3.578 and 3.574, above the threshold 1.0. Clipping multiplies the gradient by 1 / 3.58 ≈ 0.28 before the optimizer step. This agrees with prediction 1, but only for 2 of the 1,178 optimizer steps.

**Finding 2: the norm is small, compared to the earlier estimates**. The CPU test with a tiny random model gave 3,856. The real model gives approximately 3.6. Thus the CPU value does not predict the real norm, as the section on the loss scale says. The norm is only approximately 3.6 × the threshold. If the norm decreases later in the run, it can go below 1.0. Then clipping stops on those steps.

**Finding 3: clipping by an almost constant factor has almost no effect on Adam**. From step 181 to step 182, the norm changes by 0.1%. Adam divides each gradient by the root of its second-moment estimate. Thus the same factor on all steps cancels. The per-weight gradients after clipping are much larger than eps: with a global norm of 1.0 and 98,384 weights, their RMS is approximately 0.003. Clipping can change the steps of Adam only when the norm changes much from step to step, or when it goes below 1.0. This agrees with prediction 2 (a small effect), but the validation loss must show it.

**Finding 4: one micro-batch does not show the difference between config 1 and config 2**. The six values of `loss_mse` vary from 6.78 to 7.63 (±6%). The outcome table uses limits of 5%. Thus only the validation loss can separate the two configs. The log has no value of `distill/eval/loss` yet.

**Finding 5: the duration**. The progress bar shows 23:08 for 1,453 micro-batches, which is 1.05 micro-batches for each second. At this rate, one epoch needs approximately 75 minutes, and two epochs need approximately 2.5 hours. The command of option A has `HF_TIMEOUT=3h`. If the job runs with that command, the margin for the start, the last evaluation and the upload is approximately 30 minutes.

#### What to record next

- The gradient norm in epoch 1 and at the end of the run (near step 1,178). A norm below 1.0 means that clipping stopped acting.
- The values of `distill/eval/loss` at each evaluation (every 100 optimizer steps), and the stored best step and loss.
- If W&B is on: `train/grad_norm` gives the norm at almost every optimizer step. The logging condition in `train_step` skips only the multiples of `logging_steps` (100).
