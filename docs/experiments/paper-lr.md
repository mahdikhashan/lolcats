# Experiment: learning rate and schedule of the paper in stage 1, with feature dimension 32

**Status:** Stage 1 trained on HF Jobs. The evaluations on the MMLU subset, PIQA and ARC-Easy ran on 2026-10-01.

## Question

Does stage 1 with the learning rate and the schedule of the Lizard paper give a better attention approximation than the LoLCATs recipe? Does it stop the saturation of the gate?

Background:

- Section 1 of the [gap analysis](../11-gap-analysis.md) gives the stage 1 recipe as the most probable cause of the gap. Factors 1 and 2 of section 12 give the mechanism: a 10× higher learning rate with no warmup and no decay.
- Factor 1 of section 12 predicted a saturated gate. The [stage difference experiment](stage-difference.md) found it: γ is above 0.999 in layers 1–15.
- Section 13.4 ranks the stage 1 hyperparameters as root cause 1.
- The [feature dimension experiment](feature-dimension.md) found that feature dimension 32 does not change the MMLU-subset result or the saturation. This experiment keeps feature dimension 32, so it changes only the learning rate and the schedule against that run.

## What changes, and what stays the same

| Setting | Run 1, stage 1 | Feature dimension run | This experiment | Paper (Table 13) |
|---|---|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd128_m4` | `distill_llama3_2_1b_lizard_w128_fd32_m4` | `distill_llama3_2_1b_lizard_w128_fd32_m4` | – |
| Feature dimension | 128 | 32 | 32 | 128 |
| Distill config | `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b` | The same as Run 1 | **`distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b`** | – |
| Peak learning rate | 1e-2 | 1e-2 | **1e-3** | 1e-3 |
| Schedule | ReduceLROnPlateau, almost constant | The same as Run 1 | **Cosine** | Cosine |
| Warmup | None | None | **118 steps, linear (10%)** | 10%, linear |
| Learning rate at the end | 1e-2 or 1e-3 | The same as Run 1 | **0** | 0.1 × peak |
| AdamW betas | (0.9, 0.999) | The same | The same | (0.9, 0.99) |
| Gradient clipping | None | None | None | 1.0 |
| Storage of the trainable weights | bf16 | bf16 | bf16 | Not given (probably float32, see below) |
| Data, loss, steps, seed | Alpaca-cleaned, 1000 × MSE, 1,178 steps, seed 0 | The same | The same | Alpaca-cleaned, 1,178 steps |

Only the learning rate and the schedule change against the feature dimension run. Thus the comparison with that run shows their effect alone.

### Differences from the paper that stay

- **Feature dimension 32**, not 128. The feature dimension experiment found no large effect of this setting.
- **AdamW β2 = 0.999**, not 0.99 (factor 6 of section 12).
- **No gradient clipping**, not 1.0 (factor 5). Clipping needs a code change (section 11, PR A).
- **The schedule decays to 0**, not to 0.1 × the peak. The scheduler type `cosine_warmup` uses `transformers.get_cosine_schedule_with_warmup`, which has no minimum. A minimum needs a code change.
- **The trainer stores the trainable weights in bf16.** The next section explains the risk.

### Risk: bf16 storage at learning rate 1e-3

Factor 0 of section 12 found that bf16 rounding discards small updates. At a learning rate of 1e-3, its table gives these values:

- `alpha_blend` starts at 1.0. Near 1.0, the step between two bf16 values is 0.0039 below 1 and 0.0078 above 1. An Adam step of approximately 1e-3 is less than half of this step. Thus rounding discards **all** updates of α, and α probably stays at exactly 1.0 in every layer.
- Feature-map weights that grew to approximately 0.2 lose approximately half of their updates.
- After approximately step 600, the cosine schedule is below 5e-4. Then rounding discards a larger part of all updates.

Thus this run can show a worse result than the recipe of the paper would give with float32 weights. The paper trained with FSDP-2, which normally keeps float32 master weights. The paper does not say so.

**Side measurement:** If α is exactly 1.000 in all 16 layers after this run, the result agrees with the prediction of factor 0.

## How to run

### Option A: HF Jobs (H200)

1. **Build a new Docker image.** Merge the PR with the new config first. Then select Actions → "Docker image" → Run workflow. HF Jobs runs only the code inside the image ([document 3](../03-infrastructure.md)).
2. **Train stage 1 on HF Jobs:**

   ```bash
   make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=2h \
     ARGS="--model_config distill_llama3_2_1b_lizard_w128_fd32_m4 --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b --no_finetune"
   ```

   - `make hf-job` does not give `DISTILL_CONFIG` to the job. Thus `ARGS` gives the configs. argparse keeps the last value.
   - `--no_finetune` stops the run after stage 1.

### Option B: the A10 machine, with conda

```bash
conda activate lolcats-env
export HF_TOKEN=hf_...
nohup make distill-local DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b > distill-paper-lr.log 2>&1 &
```

`LOCAL_MODEL_CONFIG` has the default `distill_llama3_2_1b_lizard_w128_fd32_m4`. The [feature dimension experiment](feature-dimension.md) describes the memory risk on the A10 (option B there).

### Checkpoint

Both options write this checkpoint. Option A also pushes it to the Hub.

```
checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
```

The name contains the new distill config. Thus no earlier checkpoint gets overwritten.

### Evaluation on the A10

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b \
MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4 \
MODELS=stage1 TASKS=mmlu_subset \
scripts/compare_stages.sh 2>&1 | tee mmlu-paper-lr.log
```

`scripts/compare_stages.sh` builds the checkpoint name from `DISTILL_CONFIG` and `MODEL_CONFIG`. It downloads the checkpoint from the Hub if the file is not on the machine. Use `TASKS="piqa arc_easy"` for PIQA and ARC-Easy.

## How to compare

The validation losses are directly comparable, because all three runs use the same loss and the same validation data.

| Measure | fd128, LoLCATs recipe | fd32, LoLCATs recipe | fd32, recipe of the paper |
|---|---|---|---|
| Stored stage 1 validation loss, and its step | 3.2549 at step 1,100 | 3.4219 at step 1,100 | **8.1641 at step 700** |
| MMLU-subset accuracy | 22.5 ± 2.5 | 23.2 ± 2.5 | 25.3 ± 2.6 |
| Share of "A" answers | 95.4% | 66.0% | 21.8% |
| Letter mass | 0.018 | 0.024 | 0.009 |
| Layers 1–15 with γ above 0.999 for 100.0% of the tokens | 15 of 15 | 14 of 15 (layer 15: 99.9%) | **0 of 15** |
| Layer 0: weight kept after 512 tokens | 0.0059 | 0.25 | 9.3e-25 |
| α | 0.042–0.648 | 0.063–0.656 | **1.000 in all 16 layers** |
| PIQA accuracy | 57.6 ± 1.2 | Not measured yet | 55.8 ± 1.2 |
| ARC-Easy accuracy | Not measured yet | Not measured yet | 34.1 ± 1.0 |

How to read the outcome:

| Outcome | Conclusion | Next step |
|---|---|---|
| The gate does not saturate, the validation loss is lower, and the MMLU subset is better or less concentrated on one letter | The stage 1 recipe causes the saturation (root cause 1 of section 13.4) | Run stage 2 with the recipe of the paper |
| The gate still saturates, and the validation loss is approximately the same | The learning rate and the schedule alone do not fix stage 1 | Keep the trainable weights in float32 (factor 0). Then test the normalization of the gated branch (D1 in [math against code](../math-code-discrepancy.md)). |
| The validation loss is higher, and α is exactly 1.000 in all layers | bf16 rounding limits this run (factor 0) | Keep the trainable weights in float32, then run this experiment again |

## Code changes

- `configs/experiment/distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b.yaml`: a copy of `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b.yaml` with two changes:
  - `optimizer.lr: 0.001`,
  - `lr_scheduler`: `cosine_warmup` with `num_warmup_steps: 118` and `num_training_steps: 1178`.

The trainer already supports `cosine_warmup` (`src/trainer/optim.py`). It steps the scheduler once after each optimizer step (`src/trainer/default_lm.py`). Thus the step counts are optimizer steps. [Document 2](../02-compute-and-cost.md) gives 2 epochs × 589 = 1,178 optimizer steps for each stage. W&B reported step 1,178 at the end of stage 1 of Run 1 ([document 4](../04-training-runs.md)).

## Tests

- **Schedule:** A script built the optimizer and the scheduler from the new config and ran 1,178 steps. The learning rate of each optimizer step:

  | Optimizer step | 0 | 1 | 59 | 117 | 118 | 300 | 589 | 900 | 1,100 | 1,177 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Learning rate | 0 | 8.5e-6 | 5.0e-4 | 9.9e-4 | **1.0e-3** | 9.3e-4 | 5.9e-4 | 1.6e-4 | 1.3e-5 | 0 |

  The maximum is 1e-3 at step 118. The first optimizer step has a learning rate of 0, as in every schedule of `transformers` with warmup.
- **Stage 1 on CPU:** The real `distill_llama.main()` ran with the arguments of option A, a tiny Llama and synthetic data. It finished with exit 0 and skipped stage 2. The log shows `lr: 0.001` and `cosine_warmup` in the config, and the learning rates of gradient steps 1 and 2 agree with the warmup. The run name starts with `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4`.
- The CPU test used the optimizer `adamw_torch`, because the fused optimizer needs CUDA. This adds `-o=adamw_torch` to the run name of the test only. The test had 2 optimizer steps, and the trainer saves checkpoints only at multiples of 100 steps. Thus the test wrote no checkpoint.

## Results

### Run 1: MMLU subset (2026-10-01)

- Training: stage 1 only, on HF Jobs (option A). The job pushed the checkpoint to `nanoman1/lolcats-lizard-llama-3.2-1b`. This document does not record the job ID or the training time.
- Evaluation command: the command in "Evaluation on the A10" above, without changes
- Run directory: `results/stages/20261001-214440`, on `student06`, GPU 0 (A10)
- Code: lolcats `43026af`, harness `b281b09`
- Software: Python 3.11.16, torch 2.5.1 (CUDA 12.4), transformers 4.43.1, peft 0.9.0

#### Checkpoint

| Checkpoint | SHA-256 | Size | Parameters | Dtype | Bytes per parameter | Stored step | Stored loss |
|---|---|---|---|---|---|---|---|
| Stage 1, fd32, recipe of the paper (`..._distill.pt`) | `189cf94589e642bdf2363331a39067f04d5844c5c98d19e707460f54253f5c51` | 244,958 B | 98,384 | bf16 (80 tensors) | 2.49 | **700** | `distill/eval/loss` = 8.1641 |

- **The load is complete**. All 80 Lizard parameters (16 layers × 5) are in the file and hold their values after the load.
- **The best checkpoint comes from step 700 of 1,178.** The earlier runs had their best checkpoint at step 1,100. Thus the validation loss of this run did not improve after step 700. At step 700, the learning rate of the schedule is 4.2e-4.
- The Hub also has `..._distill_1000.pt`, the periodic save at step 1,000. The Hub has no stage 1 results CSV for this run under the expected name. Thus the summary has no validation curve.

#### Scores and answer letters

| Model | Right answers | Accuracy | "A" | "B" | "C" | "D" | Letter mass | Confidence | Entropy |
|---|---|---|---|---|---|---|---|---|---|
| fd128, LoLCATs recipe | 64 / 285 | 22.5 ± 2.5 | 95.4% | 2.8% | 1.8% | 0.0% | 0.018 | 0.586 | 1.550 bits |
| fd32, LoLCATs recipe | 66 / 285 | 23.2 ± 2.5 | 66.0% | 22.1% | 0.7% | 11.2% | 0.024 | 0.476 | 1.681 bits |
| fd32, recipe of the paper | 72 / 285 | **25.3 ± 2.6** | 21.8% | 2.8% | 36.8% | 38.6% | 0.009 | 0.494 | 1.664 bits |
| Teacher ([document 7](../07-results.md)) | 96 / 285 | 33.7 ± 2.8 | – | – | – | – | – | – | – |

The right answers are "A" 24.2%, "B" 24.9%, "C" 25.3% and "D" 25.6%. The difference to the fd32 run with the LoLCATs recipe is +2.1 points (6 questions). The unpaired SE of the difference is approximately 3.6, so z ≈ 0.6. Thus the two accuracies are not clearly different.

#### Gates and Lizard parameters

The gate values come from the same 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens) as in the earlier runs.

| Layer | γ mean | γ min | γ above 0.999 | Kept after 128 | Kept after 512 | α | Sink logit (all 4 equal) | ‖W_γ‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.901 | 0.719 | 0.0% | 9.1e-07 | 9.3e-25 | 1.000 | 0.25 | 1.59 |
| 1 | 0.944 | 0.555 | 0.0% | 9.0e-05 | 1.3e-17 | 1.000 | 0.25 | 1.55 |
| 2 | 0.986 | 0.425 | 0.0% | 0.15 | 3.1e-04 | 1.000 | 0.25 | 1.20 |
| 3 | 0.995 | 0.613 | 0.0% | 0.49 | 0.071 | 1.000 | 0.25 | 1.18 |
| 4 | 0.994 | 0.670 | 0.2% | 0.39 | 0.023 | 1.000 | 0.125 | 1.22 |
| 5 | 0.994 | 0.768 | 1.3% | 0.34 | 0.0049 | 1.000 | 0.125 | 1.20 |
| 6 | 0.997 | 0.827 | 11.6% | 0.51 | 0.059 | 1.000 | 0.25 | 1.29 |
| 7 | 0.993 | 0.736 | 0.6% | 0.12 | 3.2e-04 | 1.000 | 0.25 | 1.37 |
| 8 | 0.993 | 0.755 | 0.1% | 0.078 | 2.8e-04 | 1.000 | 0.125 | 1.28 |
| 9 | 0.989 | 0.647 | 6.0% | 0.0039 | 2.6e-08 | 1.000 | 0.125 | 1.26 |
| 10 | 0.982 | 0.627 | 11.7% | 3.0e-05 | 8.8e-15 | 1.000 | 0.125 | 1.19 |
| 11 | 0.997 | 0.687 | 68.5% | 0.16 | 0.0073 | 1.000 | 0.25 | 1.21 |
| 12 | 0.998 | 0.624 | 42.4% | 0.36 | 0.066 | 1.000 | 0.125 | 1.24 |
| 13 | 0.997 | 0.588 | 24.0% | 0.25 | 0.023 | 1.000 | 0.25 | 1.24 |
| 14 | 0.994 | 0.474 | 0.0% | 0.30 | 0.020 | 1.000 | 0.25 | 1.17 |
| 15 | 0.998 | 0.257 | 26.9% | 0.71 | 0.36 | 1.000 | 0.25 | 1.24 |

The summary prints the sink logits as 0.25 and 0.12. In bf16, 0.12 is 0.125. The feature-map weights have an RMS of 0.075–0.115 and a maximum absolute value of 0.21–0.50. In the runs with the LoLCATs recipe, the RMS was 0.15–0.22.

#### Findings

**Finding 1: the recipe of the paper stops the saturation of the gate**. No layer of 1–15 has γ above 0.999 for all tokens. In 9 of 16 layers, γ is above 0.999 for at most 1.3% of the tokens. The highest share is 68.5% in layer 11. The weight kept after 512 tokens is between 9.3e-25 (layer 0) and 0.36 (layer 15). Thus the gate decays in every layer. With the LoLCATs recipe, it saturated in 14 or 15 of the layers 1–15. This agrees with the prediction of factor 1 of section 12: the high learning rate causes the saturation.

**Finding 2: α stays at exactly 1.000 in all 16 layers**. This agrees with the prediction of factor 0 of section 12 and of the risk section above. bf16 rounding discards every update of α at this learning rate. In the runs with the LoLCATs recipe, α moved to 0.04–0.66. Thus in this run, the window branch keeps its full start weight in every layer.

**Finding 3: the stage 1 validation loss is much higher**. It is 8.1641, against 3.4219 for fd32 with the LoLCATs recipe (2.4×) and 3.2549 for fd128 (2.5×). All three runs use the same loss and the same validation data. Thus the attention approximation of this run is much worse.

**Finding 4: the validation loss stopped improving after step 700**. After step 700, the learning rate is below 4.2e-4 and decreases to 0. The feature-map weights are smaller than in the earlier runs (RMS approximately 0.1). Near 0.1, the step between two bf16 values is 0.00049. An Adam step of approximately 0.3 × 4.2e-4 is less than half of this step. Thus rounding probably discards most updates after this point. This explanation is probable, not proved.

**Finding 5: the "A" preference is absent, but the accuracy stays at the level of chance**. The model selects "A" for 21.8% of the questions. It prefers "C" (36.8%) and "D" (38.6%), and almost never selects "B" (2.8%). The accuracy (25.3) is not clearly different from the earlier runs (z ≈ 0.6) or from chance (25.0). The letter mass decreases to 0.009. Thus the model follows the 5-shot format even less than before.

**Finding 6: the sink logits again stop at powers of two**. All 16 values are exactly 0.25 or 0.125. They are smaller than with the LoLCATs recipe (0.5–2), because the learning rate is lower.

### Run 2: PIQA and ARC-Easy (2026-10-01)

- Evaluation command: the command in "Evaluation on the A10" above, with `TASKS="piqa arc_easy"`
- Run directory: `results/stages/20261001-220202`, on `student06`, GPU 0 (A10)
- Code, software and checkpoint: the same as in run 1. The checkpoint has the same SHA-256 (`189cf945…`). All 80 Lizard parameters loaded in both evaluations. The gate table is identical to run 1.

| Task (0-shot) | Accuracy | Normalized accuracy | n |
|---|---|---|---|
| PIQA | **55.8 ± 1.2** | 55.1 ± 1.2 | 1,838 |
| ARC-Easy | **34.1 ± 1.0** | 34.0 ± 1.0 | 2,376 |

Reference values:

| Model | PIQA | ARC-Easy | Source |
|---|---|---|---|
| fd128, LoLCATs recipe, after stage 1 | 57.6 ± 1.2 | Not measured | [Stage difference](stage-difference.md), quick check 1 |
| fd128, after stage 2 (the Lizard model) | 67.95 ± 1.09 | 54.8 | [Document 7](../07-results.md) |
| Teacher, in the paper | 74.1 | 65.4 | Table 9 of the paper, newer harness |
| Chance | 50.0 | Approximately 25 | 2 choices on PIQA. Mostly 4 choices on ARC-Easy. |

**Finding 7: PIQA does not improve**. This run gets 55.8, against 57.6 for fd128 with the LoLCATs recipe after stage 1. The difference is −1.8 points (approximately 33 questions). The unpaired SE of the difference is approximately 1.6, so z ≈ −1.1. Thus the two values are not clearly different. Both are only a little above chance (50.0) and far below the teacher (74.1 in the paper).

**Finding 8: ARC-Easy is above chance, but far below the stage 2 model and the teacher**. This run gets 34.1, against approximately 25 for chance, 54.8 for the stage 2 model and 65.4 for the teacher in the paper. No earlier stage 1 model has an ARC-Easy result. Thus this value has no stage 1 reference yet.

### Interpretation

- **The outcome is the third row of the outcome table**: the validation loss is higher, and α is exactly 1.000 in all layers. Thus bf16 rounding limits this run (factor 0). The next step is to keep the trainable weights in float32, then run this experiment again.
- **The gate result is clear**. The learning rate and the schedule of the paper stop the saturation of the gate (finding 1). Thus the LoLCATs recipe causes the saturation in Run 1 (root cause 1 of section 13.4 of the [gap analysis](../11-gap-analysis.md)). The feature dimension does not cause it, and the code does not force it.
- **A decaying gate alone does not give a good approximation**. The validation loss is 2.4× higher. Two effects of bf16 can explain this. α cannot leave 1.0 (finding 2), and the feature-map weights probably stop learning after step 700 (finding 4). This run cannot separate these effects from the effect of the decaying gate itself.
- **Relation to section 13.2 of the gap analysis**. The runs with a saturated gate have a lower validation loss than this run with a decaying gate. This agrees with the hypothesis that a normalized gated branch reaches a lower loss when it can reach the BOS tokens. But the frozen α and the bf16 rounding also affect this run. Thus this result does not test the hypothesis.
- **MMLU stays at the level of chance in all three runs**. The letter pattern changes with the recipe ("A" → "C" and "D"), but the accuracy does not. Thus the "A" preference is not the only problem of the stage 1 model.
- **PIQA and ARC-Easy agree with the MMLU subset**. With the decaying gate, PIQA does not improve (finding 7), and ARC-Easy stays far below the stage 2 model (finding 8). Thus the stage 1 model of this run is not better than the stage 1 model of Run 1 on any of the three tasks. This agrees with its higher validation loss.
- This result is preliminary. The comparison with the earlier runs uses unpaired SEs.

### Open items

1. **float32 storage of the trainable weights** (factor 0 of section 12). Then repeat this experiment. Expected: α moves away from 1.0, and the validation loss improves after step 700.
2. **Stage 1 references for ARC-Easy and PIQA:** evaluate the fd32 checkpoint with the LoLCATs recipe with `TASKS="piqa arc_easy"` (the default `DISTILL_CONFIG`). Then only the learning rate and the schedule differ between the two stage 1 models.
3. **Validation curve:** the trainer writes a results CSV next to the checkpoint. The Hub has none for this run under the expected name. A curve would show the loss after step 700 directly.
