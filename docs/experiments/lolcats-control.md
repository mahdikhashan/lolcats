# Experiment: LoLCATs attention as a control in stage 1

**Status:** Run 1 finished on 2026-10-09. Stage 1 validation loss 0.4442 (Lizard: 3.2549), PIQA 73.5 (Lizard: 57.6), ARC-Easy 62.8, MMLU subset 26.0. On PIQA and ARC-Easy, this model computes almost exactly the teacher attention ("Results", finding 2).

## Question

Does the original LoLCATs attention give a better stage 1 than Lizard, with the same pipeline, data, recipe and harness? This is step A2 of [document 19](../19-next-steps-from-literature.md), for stage 1 only.

The run changes only the attention layer. Thus a clear difference comes from the layer, not from the pipeline. The published LoLCATs results for 1B are after stage 2 (Liger paper: PIQA 74.1, ARC-Easy 63.7, MMLU 23.1). This experiment cannot compare with them. A stage 2 run is a later step.

## What changes, and what stays the same

The reference is the stage 1 checkpoint of Run 1: Lizard with feature dimension 128 and the LoLCATs recipe ([stage difference](stage-difference.md)).

| Part | Lizard, Run 1 stage 1 | This experiment |
|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd128_m4` | `distill_llama3_1_1b_lk_smd_wtk64_fd64_w01` (`make lolcats`) |
| Attention type | `lolcats_llama_lizard` | `lolcats_llama_window_tk` |
| Linear branch | A gate (W_γ), all tokens, its own denominator | No gate, only the tokens outside the window, one denominator shared with the window |
| Feature maps | One map for each layer, shared by the 32 heads | One map for each head (32 × 64 × 128) |
| Window | Sliding, 128 tokens, 4 sinks, no RoPE | Terraced: its own block of 128 tokens and the block before it, thus 128–255 tokens. No sinks, with RoPE. |
| Window weight | α for each layer, start value 1 | σ(a_h) for each head, start value 0.1 |
| Start values of the maps | Random, RMS 0.02 | Identity (`--lk_zero_init`) |
| Trainable parameters | 294,992 | Approximately 8.39M (16 layers × (2 × 32 × 64 × 128 + 32)), an estimate |

**The same in both runs:** Llama-3.2-1B in bf16, the Alpaca-cleaned data in chunks of 2048 tokens, and the stage 1 recipe `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b`. The recipe has learning rate 1e-2, the plateau schedule, no warmup, no clipping, 2 epochs and batch 1 × 8. Both runs use the same loss (1000 × MSE of the head outputs before `o_proj`), the same validation data and the same harness. Thus the two validation losses compare directly.

The file name of the model config says `wtk64_fd64`, but the file sets window 128 and feature dimension 128, as for Lizard. The LoLCATs defaults are 64 and 64. This run keeps the file without changes.

## How to run

### Training: HF Jobs (H200)

The current Docker image already has the config and the `make lolcats` target. Thus no new image is necessary.

```bash
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=2h \
  TARGET=lolcats ARGS="--no_finetune"
```

- The job runs `make lolcats ARGS="--no_wandb --no_finetune"` (with `--no_wandb` only if `WANDB_API_KEY` is not set).
- **Time and cost:** approximately 1 hour and $5. This is an estimate: stage 1 of Run 1 (Lizard, bf16) took approximately 50 minutes ([document 2](../02-compute-and-cost.md)).
- **First check during the run:** `hf jobs logs <job id> | grep -a "Eval step"`. The loss must decrease.
- **Checkpoint on the Hub:** `--lk_zero_init` adds `-lzi=1` to the run name. Thus the path is:

```text
checkpoints/distill_llama3_1_1b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_1_1b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-lzi=1_distill.pt
```

### Evaluation: the A10 machine

The evaluation needs the change to `scripts/compare_stages.py` of this PR ("Code changes"). Thus first run `git pull`.

```bash
cd /system/user/khashan/lolcats && git pull && conda activate lolcats-env
export HF_TOKEN=<token with access to HF_REPO and meta-llama/Llama-3.2-1B>
CKPT=checkpoints/distill_llama3_1_1b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_1_1b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-lzi=1_distill.pt

# 1. MMLU subset, PIQA and ARC-Easy of the LoLCATs stage 1 model
MODELS=stage1 TASKS="mmlu_subset piqa arc_easy" MODEL_CONFIG=distill_llama3_1_1b_lk_smd_wtk64_fd64_w01 \
  DISTILL_CKPT="$CKPT" OUT_DIR=results/stages/$(date +%Y%m%d-%H%M%S)-lolcats-control scripts/compare_stages.sh

# 2. Layer-wise MSE, and a plot against Lizard fd128 with the same recipe
python scripts/layer_mse.py compute results/layer_mse/lolcats_attention_fd128.json --label "LoLCATs attention, fd128" \
  --model_config distill_llama3_1_1b_lk_smd_wtk64_fd64_w01 --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2_1b \
  --checkpoint "$CKPT" --cache_dir ~/.cache/huggingface/hub
python scripts/layer_mse.py plot results/layer_mse/lizard_vs_lolcats_attention \
  docs/experiments/xai-layer-wise-mse/fd128_lolcats.json results/layer_mse/lolcats_attention_fd128.json
```

Note: `fd128_lolcats.json` is Lizard with feature dimension 128 and the LoLCATs recipe. `lolcats_attention_fd128.json` is the LoLCATs attention of this experiment.

**Reference evaluations** (optional, minutes each). They complete the comparison table below.

```bash
# ARC-Easy of the Lizard reference (Run 1 stage 1): the default paths of compare_stages.sh
MODELS=stage1 TASKS=arc_easy scripts/compare_stages.sh
# The teacher on PIQA and ARC-Easy in this harness (X0 of gap analysis 2)
MODELS=teacher TASKS="piqa arc_easy" scripts/compare_stages.sh
```

## How to compare

| Measure | Lizard, Run 1 stage 1 | LoLCATs attention | Limit for a clear difference |
|---|---|---|---|
| Stage 1 validation loss | 3.2549 | 0.4442 | The same loss and data. No SE. |
| PIQA | 57.6 | 73.5 | Approximately 3.2 points (2 unpaired SE) |
| ARC-Easy | Not measured (reference evaluation above) | 62.8 | Approximately 2.8 points |
| MMLU subset | 22.5, with "A" for 95.4% | 26.0, with "A" for 19.3% | Approximately 7 points. This subset cannot separate the two models. |

| Result | Meaning | Next step |
|---|---|---|
| LoLCATs is clearly better on PIQA or ARC-Easy | With the same pipeline and recipe, the Lizard layer is the weaker part. | Stage 2 for LoLCATs, against the target of the Liger paper. Then steps B1 and B5 of [document 19](../19-next-steps-from-literature.md). |
| No clear difference | Stage 1 alone does not separate the two layers on these tasks, as for the recipe experiments. | Stage 2 for LoLCATs. The published LoLCATs results are after stage 2. |
| LoLCATs is clearly worse | Not expected | First check the run: the loss curve and the checkpoint check. |

## Code changes

`scripts/compare_stages.py`: the stage 1 checkpoint check accepted only Lizard parameter names. A LoLCATs checkpoint thus gave 0 expected keys, and the check had no effect. The new tuple `LOLCATS_PARAMS` adds the three LoLCATs names: `feature_map_q.mlp.layer`, `feature_map_k.mlp.layer` and `window_factors`. `layer_mse.py` uses the same check. The Lizard names do not change.

Other parts of the scripts work without a change. `lizard_stats` finds no Lizard layer and writes an empty table. `layer_mse.py` reads `(y_pred, y_true)` from each layer, and the LoLCATs layer gives the same pair.

## Tests

CPU, 2026-10-09. The test ran the exact command of `make lolcats ARGS="--no_wandb --no_finetune ..."` with `distill_llama.main()`. It used a tiny local Llama (3 layers, 4 heads, head dimension 16) and a small Alpaca file. In a copy of the repository, the configs pointed to the tiny model, in float32 and eager attention.

| Check | Result |
|---|---|
| Trainable tensors | Only `feature_map_q.mlp.layer`, `feature_map_k.mlp.layer` and `window_factors` (9 tensors for 3 layers) |
| Start values | The feature maps are identity matrices. σ(window factor) = 0.1 for each head. |
| Training at 1024 tokens, 4 steps, `--gradient_accumulation_steps 1` | The validation loss decreased at each evaluation: 2289, 2224, 2198, 2175. |
| Checkpoint | The name ends in `-se=0-re=0-lzi=1_distill.pt` (the test adds `-gas=1-ms=4`). It holds the 9 tensors. |
| `check_checkpoint` with the change | 9 expected keys, 0 missing, 0 unexpected, 0 not loaded |

**Two notes from the tests:**

- At 256 tokens, the error at the start was approximately 2e-13. The terraced window covers up to 255 tokens, so it covers the whole sequence with exact softmax. The linear branch acts only on longer sequences. The training data has 2048 tokens, so this does not affect the real run.
- At 1024 tokens, on the tiny random model, the error at the start was 2.0–2.4 for LoLCATs and 6.9–7.1 for Lizard. The mean square of the teacher output was 3.2–3.8. The output of LoLCATs is a weighted mean, and the output of Lizard has a total weight above 1 ([document 15](../15-attention-math-side-by-side.md), claim C). These values do not predict the result at 1B.

## Results

### Run 1 (2026-10-09)

- **Training**: stage 1 on HF Jobs (H200) with the command above. These notes do not record the job ID or the time.
- **Evaluation**: `scripts/compare_stages.sh` with `MODELS=stage1`, on `student06`, A10, on 2026-10-09 at 17:50 UTC. Run directory: `results/stages/20261009-194950-lolcats-control`. Code: lolcats `8bd7614`, harness `b281b09`, torch 2.5.1, transformers 4.43.1.

| Checkpoint | SHA-256 | Size | Parameters | Dtype | Stored step | Stored loss |
|---|---|---|---|---|---|---|
| LoLCATs attention, stage 1 | `200d27aa5393c1465c3f203639ad24ea2a791c0a54ee1f2a295776bcc0ee3c08` | 16,808,818 B | 8,389,120 | bfloat16 (48 tensors) | 1100 | **0.4442** |

In all three evaluations, the 48 expected tensors (16 layers × 3) loaded with their values. The stored step is the last evaluation, as in all earlier runs.

**Scores** (z values with unpaired SEs):

| Measure | Lizard, Run 1 stage 1 | LoLCATs attention, stage 1 | Difference |
|---|---|---|---|
| Stage 1 validation loss | 3.2549 | **0.4442** | −86% (7.3× lower) |
| PIQA | 57.6 ± 1.2 | **73.5 ± 1.0** | +15.9 points, z ≈ 10 |
| ARC-Easy | Not measured | **62.8 ± 1.0** (normalized 58.0) | – |
| MMLU subset | 22.5 ± 2.5 | 26.0 ± 2.6 | +3.5 points, z ≈ 1.0 |
| "A" / "B" / "C" / "D" on MMLU | 95.4% / 2.8% / 1.8% / 0.0% | 19.3% / 14.7% / 42.8% / 23.2% | – |
| Letter mass, confidence, entropy | 0.018, 0.586, 1.550 bits ([temperature](temperature.md)) | 0.962, 0.450, 1.746 bits | – |

**Other reference values:**

| Model | PIQA | ARC-Easy |
|---|---|---|
| LoLCATs attention, stage 1 (this run) | 73.5 | 62.8 |
| Lizard model after stage 2 (Run 2, [document 7](../07-results.md)) | 67.95 | 54.8 |
| Lizard, best stage 1 on ARC-Easy (config 1, [second round](second-round.md)) | 57.3 | 36.5 |
| Teacher, Lizard paper (another harness) | 74.1 | 65.4 |
| Lizard 1B, Lizard paper | 74.8 | 65.6 |
| LoLCATs 1B after stage 2, Liger paper | 74.1 | 63.7 |

**Finding 1: the pipeline can train a much better model than the Lizard runs**. LoLCATs after stage 1 is 15.9 points above Lizard after stage 1 on PIQA. Without stage 2, it is also above the final Lizard model on PIQA (+5.5, z ≈ 3.8) and on ARC-Easy (+8.0, z ≈ 5.6). The pipeline, the data, the recipe and the harness are the same. Thus the gap on these tasks comes from the Lizard layer. This is the first row of "How to compare".

**Finding 2: on PIQA and ARC-Easy, this model computes almost exactly the teacher attention**. The terraced window gives exact softmax attention with RoPE over up to 255 tokens. The linear branch acts only on tokens farther back. 95% of the PIQA and ARC-Easy prompts have fewer than approximately 90 tokens ([document 13](../13-lizard-attention-v2.md), an estimate). For such a prompt, the window factor cancels, and the output is the softmax attention of the teacher, up to rounding. Thus these two scores test the window, not the linear attention. They are probably near the teacher scores in this harness. X0 of [gap analysis 2](../12-gap-analysis-2.md) has not run yet and can check this.

**Finding 3: LoLCATs also approximates long sequences much better**. The stage 1 loss uses sequences of 2048 tokens, so the linear branch acts on most positions. The loss is 7.3× lower than for Lizard. Two differences explain a part of this: the larger window (128–255 tokens against 128) and the feature maps for each head (28× more parameters). This run cannot separate the causes.

**Finding 4: MMLU stays near chance, but without the "A" collapse**. The 5-shot prompts are long, so the linear branch acts. The accuracy is 26.0, against 33.7 for the teacher (z ≈ −2.0). This agrees with the Liger paper (LoLCATs 1B: 23.1 after stage 2). But the model puts 96.2% of its probability on the four letters, against 1.8% for Lizard stage 1. Thus the LoLCATs model keeps the answer format, and Lizard stage 1 does not ("Stuck on A", [document 19](../19-next-steps-from-literature.md)).

**Finding 5: the most probable cause of the short-context gap of Lizard**. On a short prompt, Lizard never computes the teacher attention. Its window has no RoPE, and its gated branch adds weight on every token, also inside the window ([document 15](../15-attention-math-side-by-side.md), claims B, C and F). LoLCATs computes the teacher attention on a short prompt, because its window has RoPE and its linear branch excludes the window tokens.

### Next steps

1. X0: the teacher on PIQA and ARC-Easy in this harness (minutes). It checks finding 2.
2. Lizard with `window_rope` in the setup of Run 1 (step B5 of [document 19](../19-next-steps-from-literature.md)). This is the most direct test of finding 5. It needs only a new model config: the Run 1 config with `attention_type: lolcats_llama_lizard_v2` and `window_rope: true`.
3. A Lizard option that keeps the gated branch out of the window, as in LoLCATs (claim A of [document 15](../15-attention-math-side-by-side.md)). It needs a code change.
4. Stage 2 of this LoLCATs model, against the target of the Liger paper (MMLU 23.1).
5. ARC-Easy of Lizard Run 1 stage 1 (the reference evaluation above), to fill the empty cell.
