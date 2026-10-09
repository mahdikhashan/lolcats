# Experiment: LoLCATs attention as a control in stage 1

**Status:** Prepared, not run yet. A CPU test of the training command passes ("Tests").

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

**Per-layer table alone** (no benchmark, a few minutes). The task `layers` loads the model, checks the checkpoint and writes the table "LoLCATs parameters and window share" to `summary.md`. The harness stops before a benchmark.

```bash
MODELS=stage1 TASKS=layers MODEL_CONFIG=distill_llama3_1_1b_lk_smd_wtk64_fd64_w01 \
  DISTILL_CKPT="$CKPT" OUT_DIR=results/stages/$(date +%Y%m%d-%H%M%S)-lolcats-control-layers scripts/compare_stages.sh
```

The table uses one 5-shot MMLU prompt of 2048 tokens. For each layer, it gives these values:

| Column | Meaning |
|---|---|
| Window factor mean, min, max | σ(a_h) over the 32 heads, the weight of the window terms before the normalization. Start value 0.1. |
| Window share, all queries | The share of the attention weight of a query on the keys inside its window, as the mean over the heads and the queries |
| Far queries | The same share, only for the queries with keys outside the window (position 256 or later). There, the linear branch acts. |
| Last query | The same share for the last token of the prompt |
| φq, φk RMS / max | The size of the feature-map weights |
| φq, φk change from identity | ‖W − I‖ / ‖I‖. The maps start as identity matrices (`--lk_zero_init`). |

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
| Stage 1 validation loss | 3.2549 | – | The same loss and data. No SE. |
| PIQA | 57.6 | – | Approximately 3.2 points (2 unpaired SE) |
| ARC-Easy | Not measured (reference evaluation above) | – | Approximately 2.8 points |
| MMLU subset | 22.5, with "A" for 95.4% | – | Approximately 7 points. This subset cannot separate the two models. |

| Result | Meaning | Next step |
|---|---|---|
| LoLCATs is clearly better on PIQA or ARC-Easy | With the same pipeline and recipe, the Lizard layer is the weaker part. | Stage 2 for LoLCATs, against the target of the Liger paper. Then steps B1 and B5 of [document 19](../19-next-steps-from-literature.md). |
| No clear difference | Stage 1 alone does not separate the two layers on these tasks, as for the recipe experiments. | Stage 2 for LoLCATs. The published LoLCATs results are after stage 2. |
| LoLCATs is clearly worse | Not expected | First check the run: the loss curve and the checkpoint check. |

## Code changes

`scripts/compare_stages.py`: the stage 1 checkpoint check accepted only Lizard parameter names. A LoLCATs checkpoint thus gave 0 expected keys, and the check had no effect. The new tuple `LOLCATS_PARAMS` adds the three LoLCATs names: `feature_map_q.mlp.layer`, `feature_map_k.mlp.layer` and `window_factors`. `layer_mse.py` uses the same check. The Lizard names do not change.

Second change to `scripts/compare_stages.py` (2026-10-09): the per-layer table at the end of `summary.md` was empty for LoLCATs, because it read only Lizard parameters. `layer_stats` now also reads LoLCATs layers. It gives the window factors, the window share of the attention weight and the change of the feature maps. The share comes from the attention weights that the layer itself returns, so the output of the model does not change. `compare_stages.sh` has the new task `layers`, which writes only the checkpoint check and this table. For Lizard runs, `summary.md` does not change: the summary of R1b is the same as before.

`layer_mse.py` works without a change. It reads `(y_pred, y_true)` from each layer, and the LoLCATs layer gives the same pair.

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

Not run yet.
