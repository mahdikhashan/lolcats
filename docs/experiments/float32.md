# Experiment: float32 in stage 1, with the learning rate of the paper and feature dimension 32

**Status:** The config is ready, with CPU tests. No stage 1 training with this config has run yet.

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
| Stored stage 1 validation loss, and its step | 3.4219 at step 1,100 | 8.1641 at step 700 | Not measured yet |
| MMLU-subset accuracy | 23.2 ± 2.5 | 25.3 ± 2.6 | Not measured yet |
| Share of "A" answers | 66.0% | 21.8% | Not measured yet |
| Letter mass | 0.024 | 0.009 | Not measured yet |
| PIQA accuracy | Not measured yet | 55.8 ± 1.2 | Not measured yet |
| ARC-Easy accuracy | Not measured yet | 34.1 ± 1.0 | Not measured yet |
| Layers 1–15 with γ above 0.999 for 100.0% of the tokens | 14 of 15 | 0 of 15 | Not measured yet |
| α | 0.063–0.656 | 1.000 in all 16 layers | Not measured yet |
| Sink logits exactly at a power of two | 10 of 16 layers | 16 of 16 layers | Not measured yet |

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

Not run yet.
