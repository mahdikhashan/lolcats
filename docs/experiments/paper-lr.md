# Experiment: learning rate and schedule of the paper in stage 1, with feature dimension 32

**Status:** The config is ready, with a CPU test. No stage 1 training with this config has run yet.

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

**Side measurement:** If α is exactly 1.000 in all 16 layers after this run, the result confirms the prediction of factor 0.

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

`scripts/compare_stages.sh` builds the checkpoint name from `DISTILL_CONFIG` and `MODEL_CONFIG`. It downloads the checkpoint from the Hub if the file is not on the machine. Add `piqa` to `TASKS` for PIQA.

## How to compare

The validation losses are directly comparable, because all three runs use the same loss and the same validation data.

| Measure | fd128, LoLCATs recipe | fd32, LoLCATs recipe | fd32, recipe of the paper |
|---|---|---|---|
| Stored stage 1 validation loss, and its step | 3.2549 at step 1,100 | 3.4219 at step 1,100 | Not measured yet |
| MMLU-subset accuracy | 22.5 ± 2.5 | 23.2 ± 2.5 | Not measured yet |
| Share of "A" answers | 95.4% | 66.0% | Not measured yet |
| Letter mass | 0.018 | 0.024 | Not measured yet |
| Layers 1–15 with γ above 0.999 for 100.0% of the tokens | 15 of 15 | 14 of 15 (layer 15: 99.9%) | Not measured yet |
| Layer 0: weight kept after 512 tokens | 0.0059 | 0.25 | Not measured yet |
| α | 0.042–0.648 | 0.063–0.656 | Not measured yet |
| PIQA accuracy | 57.6 ± 1.2 | Not measured yet | Not measured yet |

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

Not run yet.
