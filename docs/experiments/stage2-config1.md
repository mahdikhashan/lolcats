# Experiment: stage 2 on the second-round checkpoint (config 1)

**Status:** The stage 2 config (`finetune_lora_qkv_alpaca_clean_paper_noclip_1b`), the float32 change and the CPU tests are ready. Stage 2 has not run yet.

## Question

All stage 1 models so far have almost the same accuracies after stage 1. The gap to the paper is a gap after stage 2. In Run 2, stage 2 recovered approximately 10 PIQA points ([stage difference](stage-difference.md)). Thus only a stage 2 run can show if a stage 1 recipe gives a better final model.

This experiment runs stage 2 on the stage 1 checkpoint of [config 1 of the second round](second-round.md). Config 1 uses the recipe of the paper in float32, with β2 = 0.99, a minimum learning rate and no clipping. Its stage 1 model has the highest MMLU-subset accuracy so far (26.7). Stage 2 uses the stage 2 settings of Table 13 of the paper and the optimizer settings of config 1. It also runs in float32. Thus both stages follow config 1, except for the differences that the next section lists.

Does config 1 in both stages give a better final model than Run 2? How near does it come to the Lizard 1B model of the paper?

**Note on the choice of config 1**. [Config 2](gradient-clipping.md) (with clipping) has a lower stage 1 loss: 3.5092 against 3.9764. But its MMLU subset is lower: 23.2 against 26.7. This difference is not clear (z ≈ −1.0 with unpaired SEs). Its PIQA (57.5) and ARC-Easy (35.6) are almost the same as for config 1. This experiment uses config 1 and its optimizer settings in both stages.

## What changes, and what stays the same

Run 2 is the reference: the stage 1 checkpoint of Run 1 (LoLCATs recipe, fd128), then stage 2 ([document 4](../04-training-runs.md)).

| Setting | Run 2 | This experiment |
|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd128_m4` (bf16, feature dimension 128) | **`distill_llama3_2_1b_lizard_w128_fd32_m4_fp32`** (float32, feature dimension 32) |
| Stage 1 checkpoint | LoLCATs recipe (`..._lr1e-2_1b`), validation loss 3.2549 | **Config 1 (`..._lr1e-3_paper_noclip_1b`), validation loss 3.9764** |
| Stage 2 config | `finetune_lora_qkvo_alpaca_clean_1b` | **`finetune_lora_qkv_alpaca_clean_paper_noclip_1b`** |
| Stage 2 learning rate and schedule | 1e-4, ReduceLROnPlateau, no warmup | **Peak 5e-4, cosine to 5e-5 (`cosine_warmup_min_lr`, `min_lr_rate: 0.1`), 118 warmup steps (10%)** |
| AdamW betas | (0.9, 0.999) | **(0.9, 0.99)** |
| Gradient clipping | None | None, as in config 1 |
| LoRA | r = 8, α = 16, dropout 0, on q, k, v and o (1,703,936 parameters) | **r = 8, α = 16, dropout 0, on q, k and v (1,179,648 parameters)** |
| Tokens | 2 epochs: 2 × 4,714 sequences × 2,048 tokens = 19.3M | The same (Table 13: 20M) |
| Lizard parameters in stage 2 | Frozen | The same |
| Stage 2 precision | bf16 | **float32**: all weights, the LoRA weights and the AdamW states. See "Code change" below. |
| Data, seed | Alpaca-cleaned, 2 epochs, seed 0 | The same |

**Sources of the stage 2 settings**. The stage 2 rows of Table 13 of the paper give five settings. These are the peak learning rate (5e-4), the LoRA rank and α (8 and 16), and the dropout (0). The other two are the projections (Wq, Wk, Wv) and the number of tokens (20M). The schedule and the betas come from the stage 1 config of config 1 (`distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b`). Config 1 also has no clipping. This is H5 of [gap analysis 2](../12-gap-analysis-2.md).

Differences from the paper that stay:

- **Feature dimension 32**, not 128. With the LoLCATs recipe, feature dimension 32 gave a 5% higher stage 1 loss ([feature dimension](feature-dimension.md)).
- **No clipping in either stage**. Table 13 of the paper gives clipping at 1.0. Config 1 has no clipping, and stage 2 follows config 1.
- **The Lizard parameters stay frozen in stage 2**. The paper does not say what it does.

**Two changes against Run 2**: the stage 1 checkpoint and the stage 2 recipe. Thus a comparison with Run 2 cannot separate the two effects. The main reference is now the Lizard 1B model of the paper.

## Code change: stage 2 in the dtype of the model config

`create_peft_config` (`src/model/peft.py`) adds the LoRA weights. Then it casts the whole model to `target_dtype`. The default of `target_dtype` is `'bfloat16'`, and no caller changed it. Thus every stage 2 model was bf16, also with a float32 model config. PEFT gives the LoRA weights the dtype of the base layer, and AdamW gives its states the dtype of the weights. Thus all of them were bf16.

This change gives the dtype of the model config (`model.torch_dtype`) to `target_dtype` in each call:

| File | Call |
|---|---|
| `distill_llama.py` | Stage 2 training, and the load of a stage 2 checkpoint (`load_and_convert_finetune`, 2 calls) |
| `src/model/load_model_for_eval.py` | The evaluation (`scripts/compare_stages.py` and `lm_eval_harness/eval_lm_harness.py`) |
| `scripts/attention_weights.py` | `--stage 2` |

- **A bf16 model config gives the same result as before**, because its dtype is `bfloat16`. Thus Run 2 and the other bf16 runs do not change.
- **A float32 model config now gives float32 in stage 2**. The training and the evaluation use the same dtype.

**Why float32 in stage 2**. With bf16, the learning rate of the late steps loses updates. At 5e-4, bf16 rounding discards no LoRA updates (factor 0 in section 12 of the [gap analysis](../11-gap-analysis.md)). But the cosine schedule goes down to 5e-5. At 1e-4, rounding already discards 43–65% of the noisy LoRA updates. In stage 1, float32 removed the symptoms of bf16 rounding ([float32](float32.md)).

## Predictions

1. **Stage 2 recovers more than in Run 2**. In Run 2, PIQA rose by approximately 10 points, from 57.6 to 67.95. Here the stage 2 learning rate is 5 times higher, and float32 keeps all LoRA updates. Factor 3 of section 12 of the gap analysis rates the stage 2 learning rate as a large factor. Thus PIQA and ARC-Easy probably end above Run 2.
2. **The final scores stay below the paper** (PIQA 74.8, ARC-Easy 65.6). Gap analysis 2 points at a limit for each head in the attention layer. Stage 2 trains only the LoRA weights on q, k and v, so it cannot remove this limit.
3. **The "A" share on MMLU**. In Run 2, the "A" share went from 95.4% after stage 1 to 98.6% after stage 2. Config 1 starts at 51.2%. If stage 2 again selects "A" for almost every question, the "A" collapse does not depend on the stage 1 recipe.
4. **The stage 2 validation loss** is probably lower than the loss of Run 2 (2.252 at step 1,100).
5. **Fewer LoRA parameters have no large effect**. Without o, LoRA has 1,179,648 parameters instead of 1,703,936. Table 8 of the paper shows little sensitivity to the LoRA capacity.

## How to run

### HF Jobs (H200)

1. **Build a new Docker image**. The new stage 2 config and the float32 change must be in the image. Merge the PR with them first. Then select Actions → "Docker image" → Run workflow ([document 3](../03-infrastructure.md)).
2. **Run stage 2 on HF Jobs:**

```bash
make hf-job-finetune IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_TIMEOUT=12h \
  DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b \
  FINETUNE_CONFIG=finetune_lora_qkv_alpaca_clean_paper_noclip_1b \
  DISTILL_CKPT=checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt \
  ARGS="--model_config distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 --load_distill_checkpoint checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt"
```

What the command does:

1. The job downloads `DISTILL_CKPT` (the config 1 checkpoint) from `HF_REPO`.
2. It runs `make lizard` with the new stage 2 config. This skips stage 1 and loads the checkpoint.
3. `make lizard` gives `--model_config distill_llama3_2_1b_lizard_w128_fd128_m4` and `--load_distill_checkpoint default` first. The values in `ARGS` come later and replace them, because argparse keeps the last value.
4. The run name contains the stage 2 config. The stage 1 checkpoint has the name of the old stage 2 config (`finetune_lora_qkvo_alpaca_clean_1b`). Thus `default` would not find it, and `ARGS` gives the path of the checkpoint directly.
5. The float32 model config sets the dtype of stage 2.
6. Stage 2 trains for 2 epochs and pushes the stage 2 checkpoint to `HF_REPO`.

**Time:** not measured for stage 2 in float32. The estimate is 6–9 hours:

- Stage 2 in bf16 took approximately 2 hours 50 minutes ([document 2](../02-compute-and-cost.md)).
- Stage 1 in float32 took approximately 3 times as long as stage 1 in bf16. It needed approximately 0.95 seconds for each sequence, against approximately 0.32 ([gradient clipping](gradient-clipping.md), finding 5).

`HF_TIMEOUT=12h` gives a margin. It is also the default of the Makefile.

**Memory:** not measured. Stage 2 in bf16 needed approximately 41–46 GB ([document 2](../02-compute-and-cost.md)). Float32 needs more, but probably less than the 141 GB of the H200.

**W&B (optional):** set `WANDB_API_KEY` in the environment before `make`. Without it, the job runs with `--no_wandb`.

**Record:** the job ID, the training time, the stage 2 validation loss at each evaluation (`hf jobs logs <job id> | grep -a "Eval step"`), and the best step.

### Checkpoint

```
checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=dacxmldl21lwfmfflqac100_re=0_distill0d-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkv_alpaca_clean_paper_noclip_1b-s=0-se=0-re=0-se=0-re=0_ft.pt
```

With a direct path to the stage 1 checkpoint, `get_run_name_from_checkpoint` shortens the stage 1 part of the name (`dl-d=dacxmldl21lwfmfflqac100_re=0_distill0d`). The CPU test printed this name. It is not the default name of `scripts/compare_stages.sh`. Thus the evaluation gives it with `FT_CKPT`.

### Evaluation on the A10

This command evaluates the teacher, the stage 1 model and the stage 2 model on the same questions. It is the full comparison of section 10 of the [gap analysis](../11-gap-analysis.md), with paired statistics. It also gives the teacher baselines on PIQA and ARC-Easy in this harness (X0 of gap analysis 2).

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b \
MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 \
FT_CKPT="checkpoints/distill_llama3_2_1b_lizard_w128_fd32_m4_fp32/dl-d=dacxmldl21lwfmfflqac100_re=0_distill0d-m=distill_llama3_2_1b_lizard_w128_fd32_m4_fp32-f=finetune_lora_qkv_alpaca_clean_paper_noclip_1b-s=0-se=0-re=0-se=0-re=0_ft.pt" \
MODELS="teacher stage1 stage2" TASKS="mmlu_subset piqa arc_easy" \
scripts/compare_stages.sh 2>&1 | tee eval-stage2-config1.log
```

- The stage 1 checkpoint keeps its default name, because `FINETUNE_CONFIG` stays at its default for the evaluation.
- The evaluation reads the stage 2 config from the `-f=` part of the stage 2 name. Thus it loads LoRA on q, k and v.
- With the float32 change, the evaluation of stage 2 also runs in float32. The weights need approximately 5 GB, as for the stage 1 evaluation of config 1.
- For all MMLU questions, add `mmlu` to `TASKS` (this takes much longer).
- For `attention_weights.py --stage 2`, give `--checkpoint` (the stage 1 path), `--finetune_config finetune_lora_qkv_alpaca_clean_paper_noclip_1b` and `--finetune_checkpoint` (the stage 2 path).

## How to compare

| Measure | Run 2 (LoLCATs stage 1, fd128, then the LoLCATs stage 2) | Config 1, after stage 1 | Config 1, after stage 2 (this experiment) | Paper, Lizard 1B |
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
| Within approximately 2 points of the paper | Config 1 in both stages closes the gap. The recipe was the main cause, also with feature dimension 32 and without clipping. | Repeat with feature dimension 128. Write up the result. |
| More than approximately 2 points above Run 2, but more than 2 points below the paper | Config 1 in both stages helps, but does not close the gap | Separate the two stages: this stage 2 on the Run 1 checkpoint. Continue with X1 and C1 of gap analysis 2. |
| Within approximately 2 points of Run 2 | Config 1 in both stages has no large effect on the final model | The limit for each head (section 3 of gap analysis 2) is the main candidate. Continue with X1, C1 and C2. |
| More than approximately 2 points below Run 2 | The LoLCATs recipe, or feature dimension 128, gives a better final model | Test this stage 2 on the Run 1 checkpoint |

For MMLU: an "A" share below approximately 60% after stage 2 means that the "A" collapse depends on the stage 1 checkpoint.

The 2-point limits are a rule of thumb. The SE of one PIQA accuracy is approximately 1.1 points. This project has one seed for each run.

## Checks

These checks ran on CPU, in a copy of the repository with a tiny Llama (3 layers) and synthetic Alpaca-style data. The test copy of the stage 2 config used short chunks, 1 epoch and fewer steps. The other settings were those of `finetune_lora_qkv_alpaca_clean_paper_noclip_1b`.

- **The job command**. `make -n hf-job-finetune` with the arguments above prints the expected job. The inner `make lizard` gets the new stage 2 config, the float32 model config and the direct path of the stage 1 checkpoint.
- **Stage 2 in the real training loop**. `distill_llama.main()` ran with the arguments of the inner `make lizard` and finished with exit 0:
  - It loaded the stage 1 checkpoint from the direct path.
  - LoRA was only on q, k and v (18 LoRA matrices for 3 layers), with rank 8.
  - AdamW had the betas (0.9, 0.99) and a peak learning rate of 5e-4. The scheduler was the cosine schedule with the minimum (`LambdaLR`).
  - The training did not clip the gradients: `max_grad_norm` was None, and the clipping function had 0 calls.
  - All weights were float32 after `create_peft_config`: the frozen model, the Lizard parameters and the LoRA weights. The test did not read the AdamW states. AdamW creates them with the dtype of the weights, so they are float32 too.
  - The stage 2 checkpoint got the name in the section "Checkpoint". All 18 tensors in it were float32. The real job gives the same name, because the inputs to the name are the same.
- **The evaluation loader**. `load_model_from_checkpoint` loaded the stage 2 checkpoint in float32. With a bf16 copy of the tiny model config, it loaded the model in bf16, as before the change.
- **`attention_weights.py --stage 2`**. With the stage 2 checkpoint: 18 LoRA keys, 0 missing, 0 unexpected, 0 not loaded. The largest error of the Lizard check was 9.9e-8 (float32). Before the change, the bf16 cast gave 1.7e-3 ([sample attention weights](xai-sample-attention-weight.md)). The teacher check was 0.0.

## Results

Not run yet.
