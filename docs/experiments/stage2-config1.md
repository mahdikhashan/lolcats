# Experiment: stage 2 on the second-round checkpoint (config 1)

**Status:** No code change is necessary. The command is ready. Stage 2 has not run yet.

## Question

All stage 1 models so far have almost the same accuracies after stage 1. The gap to the paper is a gap after stage 2. In Run 2, stage 2 recovered approximately 10 PIQA points ([stage difference](stage-difference.md)). Thus only a stage 2 run can show if a stage 1 recipe gives a better final model.

This experiment runs stage 2 on the stage 1 checkpoint of [config 1 of the second round](second-round.md). Config 1 uses the recipe of the paper in float32, with β2 = 0.99 and a minimum learning rate. Its stage 1 model has the highest MMLU-subset accuracy so far (26.7).

Does stage 2 on config 1 give a better final model than Run 2, which used the LoLCATs recipe in stage 1?

**Note on the choice of config 1**. [Config 2](gradient-clipping.md) (with clipping) has a lower stage 1 loss: 3.5092 against 3.9764. Its MMLU subset is 23.2 against 26.7, but this difference is not clear (z ≈ −1.0 with unpaired SEs).

## What changes, and what stays the same

Run 2 is the reference: the stage 1 checkpoint of Run 1 (LoLCATs recipe, fd128), then stage 2 ([document 4](../04-training-runs.md)).

| Setting | Run 2 | This experiment |
|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd128_m4` (bf16, feature dimension 128) | **`distill_llama3_2_1b_lizard_w128_fd32_m4_fp32`** (float32, feature dimension 32) |
| Stage 1 checkpoint | LoLCATs recipe (`..._lr1e-2_1b`), validation loss 3.2549 | **Config 1 (`..._lr1e-3_paper_noclip_1b`), validation loss 3.9764** |
| Stage 2 config | `finetune_lora_qkvo_alpaca_clean_1b` | The same |
| Stage 2 recipe | Learning rate 1e-4, ReduceLROnPlateau, no warmup, 2 epochs | The same |
| LoRA | r = 8, α = 16, on q, k, v and o | The same |
| Lizard parameters in stage 2 | Frozen | The same |
| Stage 2 precision | bf16 | bf16. `create_peft_config` casts the whole model to bf16, also with the float32 model config. |
| Data, seed | Alpaca-cleaned, 2 epochs, seed 0 | The same |

- **The stage 2 recipe stays the same, on purpose**. Then only the stage 1 checkpoint changes. A stage 2 with the recipe of the paper (H5 of [gap analysis 2](../12-gap-analysis-2.md)) needs the fix of the bf16 cast first. It comes after this experiment.
- **The stage 1 checkpoint differs from Run 2 in two ways**: the recipe and the feature dimension (32 against 128). With the LoLCATs recipe, feature dimension 32 gave a 5% higher stage 1 loss ([feature dimension](feature-dimension.md)). Thus most of the difference probably comes from the recipe. This run cannot separate the two.

## Predictions

1. **Stage 2 recovers a similar amount as in Run 2**. In Run 2, PIQA rose by approximately 10 points, from 57.6 to 67.95. Thus PIQA probably rises from 57.3 to approximately 66–69.
2. **The final scores are within approximately 2 points of Run 2** on PIQA and ARC-Easy. The two stage 1 models have the same accuracies (PIQA 57.3 and 57.6). Also, gap analysis 2 points at a limit of the architecture, which both recipes share.
3. **The "A" share on MMLU**. In Run 2, the "A" share went from 95.4% after stage 1 to 98.6% after stage 2. Config 1 starts at 51.2%. If stage 2 again selects "A" for almost every question, the "A" collapse does not depend on the stage 1 recipe.
4. **The stage 2 validation loss** is probably near the loss of Run 2 (2.252 at step 1,100).

## How to run

### HF Jobs (H200)

The Docker image that trained config 1 and config 2 already contains everything that this run needs. This includes the float32 model config, the stage 1 config and the stage 2 config. Thus no new image is necessary.

```bash
make hf-job-finetune IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_TIMEOUT=5h \
  DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b \
  DISTILL_CKPT=checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt \
  ARGS="--model_config distill_llama3_2_1b_lizard_w128_fd32_m4_fp32"
```

What the command does:

1. The job downloads `DISTILL_CKPT` (the config 1 checkpoint) from `HF_REPO`.
2. It runs `make lizard` with `--load_distill_checkpoint default`. This skips stage 1 and loads the checkpoint.
3. `make lizard` gives `--model_config distill_llama3_2_1b_lizard_w128_fd128_m4` first. The `--model_config` in `ARGS` comes later and replaces it, because argparse keeps the last value.
4. The run name contains the model config, the distill config, the stage 2 config and the seed. All four agree with the stage 1 run. Thus `default` finds the downloaded checkpoint.
5. Stage 2 trains for 2 epochs and pushes the stage 2 checkpoint to `HF_REPO`.

**Time:** approximately 3 hours. Stage 2 in Run 1 took 1.07–1.08 seconds for each sequence, and 2 epochs have 9,428 sequences. The evaluations every 100 steps add more time. `HF_TIMEOUT=5h` gives a margin.

**W&B (optional):** set `WANDB_API_KEY` in the environment before `make`. Without it, the job runs with `--no_wandb`.

**Record:** the job ID, the stage 2 validation loss at each evaluation (`hf jobs logs <job id> | grep -a "Eval step"`), and the best step.

### Checkpoint

```
checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.pt
```

This is the default stage 2 name of `scripts/compare_stages.sh` for these configs.

### Evaluation on the A10

This command evaluates the teacher, the stage 1 model and the stage 2 model on the same questions. It is the full comparison of section 10 of the [gap analysis](../11-gap-analysis.md), with paired statistics. It also gives the teacher baselines on PIQA and ARC-Easy in this harness (X0 of gap analysis 2).

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b \
MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 \
MODELS="teacher stage1 stage2" TASKS="mmlu_subset piqa arc_easy" \
scripts/compare_stages.sh 2>&1 | tee eval-stage2-config1.log
```

For all MMLU questions, add `mmlu` to `TASKS` (this takes much longer).

## How to compare

| Measure | Run 2 (LoLCATs stage 1, fd128, then stage 2) | Config 1, after stage 1 | Config 1, after stage 2 (this experiment) | Paper, Lizard 1B |
|---|---|---|---|---|
| MMLU subset | 24.9 ± 2.6 | 26.7 ± 2.6 | Not measured yet | – |
| Share of "A" answers | 98.6% | 51.2% | Not measured yet | – |
| MMLU, all questions | 23.3 | Not measured | Not measured yet | 29.8 |
| PIQA | 67.95 ± 1.09 | 57.3 ± 1.2 | Not measured yet | 74.8 |
| ARC-Easy | 54.8 ± 1.0 | 36.5 ± 1.0 | Not measured yet | 65.6 |
| Final stage 2 validation loss, and its step | 2.252 at step 1,100 | – | Not measured yet | – |

The teacher gets 33.7 on the MMLU subset in this harness, and 74.1 on PIQA and 65.4 on ARC-Easy in the paper.

How to read the outcome:

| Outcome on PIQA and ARC-Easy | Conclusion | Next step |
|---|---|---|
| More than approximately 2 points above Run 2 | The stage 1 recipe of the paper gives a better final model, although its stage 1 loss is higher | Stage 2 on config 2. Then stage 2 with the recipe of the paper (H5 of gap analysis 2). |
| Within approximately 2 points of Run 2 | The stage 1 recipe has no large effect on the final model | The gap comes from elsewhere: the limit for each head (section 3 of gap analysis 2) or the stage 2 recipe. Continue with X1 and C1 of gap analysis 2, and with H5. |
| More than approximately 2 points below Run 2 | The LoLCATs stage 1, or feature dimension 128, gives a better final model | Keep the LoLCATs recipe for stage 1. Test stage 2 with the recipe of the paper on the Run 1 checkpoint. |

For MMLU: an "A" share below approximately 60% after stage 2 means that the "A" collapse depends on the stage 1 checkpoint.

The 2-point limits are a rule of thumb. The SE of one PIQA accuracy is approximately 1.1 points. This project has one seed for each run.

## Checks

- **The job command.** `make -n hf-job-finetune` with the arguments above prints the expected job: the download of the config 1 checkpoint, then `make lizard` with `--load_distill_checkpoint default` and the float32 model config.
- **The checkpoint path.** The real argument parsing of `distill_llama.py` ran with the arguments of the inner `make lizard`. The model config is the float32 config, and the `default` path is the same file as `DISTILL_CKPT`.

## Results

Not run yet.
