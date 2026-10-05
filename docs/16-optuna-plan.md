# 16. Hyperparameter search with Optuna: plan

**Status:** A plan only. Nothing has run, and the code changes of section 6 do not exist yet. The plan gives the method, the search space, the cost and the time of a search for stage 1 hyperparameters with Optuna.

Optuna is a Python library for hyperparameter search. A study runs trials. Each trial trains one model with sampled values and returns one score. The default sampler (TPE) uses the scores of the earlier trials to select the next values.

Each number in this document is **measured** (from a run in these notes) or **estimated** (from a calculation). The price of the H200 on Hugging Face Jobs is $5.00 for each hour ([document 2](02-compute-and-cost.md)).

## 1. The score of a trial

The score is an accuracy, not the stage 1 loss.

- Over 6 stage 1 runs, the rank correlation (Spearman) between loss and accuracy is +0.31 for ARC-Easy and −0.06 for PIQA ([document 14](14-liger-gla.md)). With 6 runs, these values are not clear. But neither value shows that a lower loss gives a higher accuracy.
- R1b has the lowest stage 1 loss of all runs (3.0542), but the lowest ARC-Easy (29.9) ([document 13](13-lizard-attention-v2.md)). A study that minimizes the loss would select runs like R1b.
- MMLU stays out of the score. After stage 1, MMLU is at the level of guessing ([document 14](14-liger-gla.md)).

**Score:** the mean of the PIQA accuracy and the ARC-Easy accuracy (0-shot) of the stage 1 model. The study maximizes it. For the three full runs of document 13, the score is 46.9 (config 1), 46.1 (C1) and 42.9 (R1b).

## 2. Time and cost of the parts of a trial

| Part | Time | Cost | Label and source |
|---|---|---|---|
| Stage 1, float32, 2 epochs (1,178 gradient steps) | 2.3–2.5 h | ~$12 | Measured: R1b 2.3 h ([document 13](13-lizard-attention-v2.md)), config 2 2.5 h ([gradient clipping](experiments/gradient-clipping.md)) |
| Speed of stage 1, float32 | 1.05–1.17 micro-batches each second | – | Measured, same runs. One gradient step uses 8 micro-batches. |
| Setup, evaluations every 100 steps, upload | ~10 min | ~$1 | Measured in Run 1 ([document 2](02-compute-and-cost.md)) |
| Stage 2, float32 | ~3 h 17 min | ~$17 | Measured ([stage 2 on config 1](experiments/stage2-config1.md)) |
| PIQA on the A10 | 3 min 11 s | free | Measured on the stage 2 model ([document 2](02-compute-and-cost.md)) |
| ARC-Easy on the A10 | ~8 min | free | Estimated: 2,376 questions × ~4 choices at 19.2 requests each second |
| Download and loading of the model on the A10 | a few minutes | free | Estimated |

## 3. Study options

All options use 25 trials, so that their costs compare directly.

| Option | One trial | Cost of one trial | Cost of 25 trials | Time, one trial after the other | Decision |
|---|---|---|---|---|---|
| A. Single-layer bench on the A10 (layers 0, 1, 8 and 15, [document 11](11-gap-analysis.md), section 13.5) | minutes | $0 | $0 | hours | Not used. Its score is the MSE, and the MSE does not predict the accuracy (section 1). |
| B. Short stage 1 (300 gradient steps), then PIQA and ARC-Easy | ~1 h | ~$4 | ~$100 | ~25 h | **Used in this plan.** Section 4 tests first if its ranking agrees with the full runs. |
| C. Full stage 1, then PIQA and ARC-Easy | ~2.9 h | ~$13 | ~$330 | ~71 h | Used only for the best trials of B (phase 3). |
| D. Stage 1 and stage 2, then PIQA and ARC-Easy | ~6.2 h | ~$30 | ~$750 | ~155 h | Not used. Stage 2 runs only for the final config. |

All costs in this table are estimates from section 2. A short trial has 2,400 micro-batches, thus 34–38 minutes of training. With the setup and the evaluation on the A10, one trial takes approximately 1 hour.

If the Hugging Face account lets several jobs run at the same time, two Optuna workers halve the time. The cost stays the same.

## 4. The plan

| Phase | Purpose | Runs | Cost | Time |
|---|---|---|---|---|
| 0. Check of the short trial | Does a run of 300 steps rank configs like a full run? | Short runs of config 1, C1 and R1b | ~$12 | ~3 h |
| 1. Structure, without Optuna | Which design options help? | Full stage 1 runs on config 1: `window_rope`, `gla_norm: convex`, `gate_per_head` ([document 14](14-liger-gla.md), next step 4) | ~$40 | ~9 h |
| 2. Optuna study | The best values of three continuous hyperparameters | 25 short trials on the best structure of phase 1 | ~$100 | ~25 h |
| 3. Confirmation | Is the best trial really better? | The best 3 trials as full stage 1 runs. The best of them with a second training seed. Then stage 2 of the best config. | ~$70 | ~15 h |
| **Total** | | | **~$220** | **~52 h** |

All values in this table are estimates. The times assume that each job starts when the job before it ends.

### Phase 0: check of the short trial

The short trial is valid only if it gives the same order as the full runs. For the full runs, the score of R1b is 4.0 points below config 1.

- **Rule:** the short trials must give R1b a score at least 2.2 points (2 SE, section 5) below config 1.
- **If the rule fails:** use trials of 1 epoch (approximately 1.25 h, [document 13](13-lizard-attention-v2.md)). This makes phase 2 approximately 2× more expensive.

### Phase 1: structure

Optuna does not select the design options. Each option is a yes or no choice with one seed, and TPE needs many trials to compare such choices. A planned run for each option is less expensive, and its result is easier to explain.

- **Rule:** an option stays only if its score is at least 2.2 points above config 1.
- `gla_norm: convex` needs a code change first. [Document 14](14-liger-gla.md) gives the proposed code.
- If no option stays, phase 2 uses config 1.

### Phase 2: Optuna study

**Search space:**

| Hyperparameter | Range | Scale | Reason for the range |
|---|---|---|---|
| `lr` (peak learning rate of stage 1) | 3e-4 to 1e-2 | log | The paper uses 1e-3 (Table 13). LoLCATs uses 1e-2. [Gap analysis 2](12-gap-analysis-2.md) proposes 3e-3 and 5e-3 (H4). |
| `alpha_init` (start value of α) | 0.1 to 1.0 | log | v1 starts at 1.0. LoLCATs starts its window weight at 0.1, and R1b used 0.1. |
| `gate_bias_init` (start value of the gate bias) | 0 to 4 | linear | A bias of 0 gives the start gate 0.5 of v1. A bias of 4 gives 0.982, near the start gate 0.958 of GLA ([document 14](14-liger-gla.md)). |

- The window size stays at 128, as in the paper. A different window changes the speed and the comparison with the paper.
- `gate_bias_init: 0` is not the same as `null`. The value 0 adds a trainable bias with the start value 0. The value `null` adds no bias.

**Settings:**

| Setting | Value | Reason |
|---|---|---|
| Sampler | TPE, seed 0 | The default sampler of Optuna. A fixed seed makes the study repeatable. |
| Random trials at the start | 10 (the default of TPE) | TPE needs some scores before its model is useful. |
| Trials | 25 | 10 random trials and 15 trials from TPE, for 3 hyperparameters |
| Training seed | 0 for all trials | The scores then differ only because of the hyperparameters and the noise of the benchmark questions. |
| Pruner | None | The score is available only at the end of a trial. The validation loss is not a good signal (section 1). |
| Failed trial (NaN or timeout) | Optuna marks it as failed | A failed trial gives no score. |
| Storage | One SQLite file on the A10 machine | If the driver stops, a new start continues the same study. |

### Phase 3: confirmation

1. Run the best 3 trials again as full stage 1 runs (2 epochs).
2. Run the best full run again with training seed 1. The difference between the two seeds shows the seed noise.
3. Run stage 2 on the best config.
4. Report the full runs, not the short trials (section 5).

## 5. Noise and selection bias

- **Noise of the score.** The SE of PIQA is approximately 1.2 points, and the SE of ARC-Easy is approximately 1.0 point (measured, [document 13](13-lizard-attention-v2.md)). The SE of their mean is approximately 0.78 points. The SE of the difference between two trials is approximately 1.1 points. Thus a difference must be at least approximately **2.2 points** to be clear.
- **The SE covers only the noise of the benchmark questions.** It does not include the noise of the training seed. Phase 3, step 2 gives one measurement of the seed noise.
- **The best of 25 trials is too high.** Some trials get a high score from noise alone, and the study selects such trials. Thus the thesis reports the full runs of phase 3, not the score of the best trial.
- **The study selects with PIQA and ARC-Easy.** Their final scores are thus too high also. For a fair result, the thesis also reports tasks that the study does not use: ARC-Challenge, HellaSwag and WinoGrande. The teacher needs the same tasks, as a reference.

## 6. Code changes before the study

1. **Command-line options `--alpha_init` and `--gate_bias_init`** in `distill_llama.py`, and in the list of `update_model_config_from_args` in `src/utils/setup.py`. `--lr` exists already. The configs are part of the Docker image. Without these options, each trial needs a new image.
2. **A short distill config:** a copy of config 1 with `max_steps: 300`, `num_training_steps: 300` and `num_warmup_steps: 30`. The option `--max_steps` alone changes only the trainer. The schedule then stays at 1,178 steps, and the run stops at approximately 0.94 × the peak learning rate.
3. **A new Docker image** with changes 1 and 2 ([document 3](03-infrastructure.md)).
4. **`scripts/optuna_stage1.py`**, approximately 100 lines. It runs on the A10 machine, with `optuna` in the conda environment. For each trial, it does these steps:
   1. It gets the three values from Optuna.
   2. It starts `make hf-job` with `--no_finetune`, the short distill config, the three values and `--replicate <trial number>`.
   3. It reads the job status until the job ends.
   4. It runs `scripts/compare_stages.sh` with `MODELS=stage1`, `TASKS="piqa arc_easy"` and `DISTILL_CKPT` set to the checkpoint of the trial.
   5. It reads `summary.json` and returns the score to Optuna.
5. **Three tasks in `scripts/compare_stages.sh`** for phase 3: `arc_challenge`, `hellaswag` and `winogrande`. Today the script accepts only `mmlu_subset`, `mmlu`, `piqa` and `arc_easy`.

The option `--replicate` adds the trial number to the run name. Thus each trial has its own checkpoint path. `compare_stages.sh` uses a local checkpoint again if one exists with the same name, so the trials must not share names.

## 7. Limits

- **The short trial can give a different order than a full run.** Phase 0 checks the order with only three configs. In all float32 runs, the validation loss still decreased at the end ([document 13](13-lizard-attention-v2.md)).
- **The order after stage 1 can differ from the order after stage 2.** Stage 2 sets the final scores. Phase 3 runs stage 2 only for the best config.
- **The study tunes three hyperparameters on one structure.** It does not search the design options together with these values.
- **The prices and times can change.** The prices come from `hf jobs hardware` at the time of [document 2](02-compute-and-cost.md). The speeds come from the runs of 2026-09-29 to 2026-10-04.
