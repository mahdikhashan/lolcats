# Experiment (XAI): MSE by token position

**Status:** The script `scripts/position_mse.py` exists and passed CPU tests on 2026-10-09. It has not run on the real checkpoints yet.

## Question

This is X1 of [document 20](../20-layer-mse-for-piqa-arc.md). In which query positions is the error of the stage 1 attention? Does the MSE of positions 0–127 order the stage 1 checkpoints as PIQA and ARC-Easy do?

## Why the position matters

- **PIQA and ARC-Easy use short prompts.** 95% of the prompts have fewer than approximately 90 tokens ([document 20](../20-layer-mse-for-piqa-arc.md), section 2). Thus for almost every question, all keys are inside the window of 128 tokens.
- **Positions 0–127 have the same case.** The validation data has 16 sequences of 2048 tokens. A query at position i sees the keys 0 to i of its sequence. For i < 128, all these keys are inside the window.
- **The stage 1 loss gives these positions little weight.** Positions 0–127 are 6.25% of the 2048 positions. The other 93.75% have keys outside the window.
- **The stage 1 loss does not predict the accuracy.** [Document 14](../14-liger-gla.md), section 5, compares 6 Lizard runs. The rank correlation between the loss and the accuracy is +0.31 for ARC-Easy and −0.06 for PIQA.

**Hypothesis:** the MSE of positions 0–127 predicts PIQA and ARC-Easy better than the stage 1 loss.

**A difference to the benchmarks.** The data loader packs Alpaca samples into sequences of 2048 tokens. Thus a validation sequence usually starts inside a sample, and its first token is usually not the first token of a text. A PIQA prompt starts at the beginning of its text.

## What the script calculates

`compute` uses the model, the data and the checkpoint of `scripts/layer_mse.py` ([XAI: layer-wise MSE](xai-layer-wise-mse.md)). `--from_json` takes the model config, the distill config, the checkpoint and the label from a result of `layer_mse.py`.

- Each attention layer runs in distillation mode. It calculates the teacher output and the student output from the same q, k and v. Each layer gets the teacher input.
- **Buckets:** `--edges 0,128,512,2048` (the default) gives the query positions 0–127, 128–511 and 512–2047.
- **For each layer and bucket:** the MSE, the mean square of the teacher output, the relative MSE, and the MSE of each head.
- **For each bucket:** 1000 × the mean MSE over the layers (the scale of the stage 1 loss), and the mean relative MSE.
- **Checks:**
  1. For each layer, the bucket MSEs, weighted by their number of positions, give the MSE over all positions. The script stops if the relative difference is above 1e-4.
  2. 1000 × the mean layer MSE over all positions is the stage 1 validation loss. With all batches, it must agree with the loss in the checkpoint, as in `layer_mse.py`.

`rank` reads the results of several checkpoints and the accuracy table [`accuracy.csv`](xai-position-mse/accuracy.csv).

- **Scores:** the loss of each bucket, and the stage 1 loss over all positions for comparison.
- **For each score and task:** the rank correlation (Spearman) ρ, the two-sided permutation p-value, and the number of checkpoints n. A negative ρ means that a lower MSE comes with a higher accuracy.
- **Ties:** tied values get their mean rank. The p-value is exact up to 9 checkpoints.
- `--exclude` removes a checkpoint by its label, for example the LoLCATs control.
- **Output:** `OUT.md` with the tables, and `OUT.png` with the accuracy against the loss of each bucket.

## The accuracy table

| Checkpoint (label) | PIQA | ARC-Easy | Source |
|---|---|---|---|
| fd32, LoLCATs recipe, bf16 | – | – | [Feature dimension](feature-dimension.md) |
| fd128, LoLCATs recipe, bf16 (Lizard Run 1) | 57.6 | – | [Stage difference](stage-difference.md) |
| fd32, paper recipe, bf16 | 55.8 | 34.1 | [Paper LR](paper-lr.md) |
| fd32, paper recipe, float32 | 57.7 | 35.7 | [float32](float32.md) |
| Second round, config 1 | 57.3 | 36.5 | [Second round](second-round.md) |
| Second round, config 2 (clipping) | 57.5 | 35.6 | [Gradient clipping](gradient-clipping.md) |
| v2, C1: alpha per head | 57.5 | 34.6 | [Document 13](../13-lizard-attention-v2.md) |
| v2, R1b: per-head, hybrid | 55.9 | 29.9 | [Document 13](../13-lizard-attention-v2.md) |
| LoLCATs attention | 73.5 | 62.8 | [LoLCATs control](lolcats-control.md) |

Three values are not measured. Two evaluations on the A10 complete the table ("How to run").

## Predictions

1. **LoLCATs:** the relative MSE of positions 0–127 is near 0. The terraced window gives exact softmax attention up to position 255 ([LoLCATs control](lolcats-control.md), finding 2). Only bf16 rounding remains. Positions 128–511 have a larger MSE, because positions 256–511 have keys outside the window.
2. **Lizard:** the relative MSE of positions 0–127 is clearly above that of LoLCATs in every checkpoint. The window has no RoPE, and the gated branch adds weight inside the window ([document 15](../15-attention-math-side-by-side.md), claims B, C and F).
3. **The main test:** if the hypothesis holds, the loss of positions 0–127 has a negative ρ. Its |ρ| is larger than for the stage 1 loss. This must also hold without LoLCATs.

## Limits of the rank test

- **Few checkpoints.** With LoLCATs, PIQA has 8 checkpoints and ARC-Easy 7. Without LoLCATs, they have 7 and 6. For p < 0.05 without ties, |ρ| must be at least 0.738 (n = 8), 0.786 (n = 7) or 0.886 (n = 6).
- **The PIQA values of Lizard are close.** They are 55.8–57.7, a spread of 1.9 points. One SE is 1.2 points, so the SE of a difference is approximately 1.7 points. Thus the PIQA order of the Lizard checkpoints is mostly noise. ARC-Easy spreads over 6.6 points (29.9–36.5), mainly because of R1b.
- **LoLCATs dominates the test with all checkpoints.** LoLCATs has the lowest loss and the highest accuracy. Thus each score that puts LoLCATs lowest gets a negative ρ. The stage 1 loss also puts LoLCATs lowest. Thus the test without LoLCATs is the important one.
- **One data set.** The MSE comes from Alpaca validation data, not from PIQA or ARC-Easy prompts.

## How to run

On the A10 (`student06`, GPU 0), in the `lolcats-env` environment, from the repository root:

```bash
mkdir -p results/position_mse
for name in fd32_lolcats fd128_lolcats fd32_paper_bf16 fd32_paper_fp32 config1 v2_alphahead \
            v2_perhead_hybrid lolcats_attention_fd128; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/position_mse.py compute \
    results/position_mse/$name.json --from_json docs/experiments/xai-layer-wise-mse/$name.json \
    --cache_dir ~/.cache/huggingface/hub
done

# Config 2 (gradient clipping) has PIQA and ARC-Easy, but no result of layer_mse.py yet
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/position_mse.py compute \
  results/position_mse/config2.json --label "Second round, config 2 (clipping)" \
  --model_config distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 \
  --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_1b --cache_dir ~/.cache/huggingface/hub

# Rank correlations, with and without the LoLCATs control
python scripts/position_mse.py rank results/position_mse/rank_all results/position_mse/*.json \
  --accuracy docs/experiments/xai-position-mse/accuracy.csv
python scripts/position_mse.py rank results/position_mse/rank_lizard results/position_mse/*.json \
  --accuracy docs/experiments/xai-position-mse/accuracy.csv --exclude "LoLCATs attention"

zip -r position-mse-$(date +%Y%m%d-%H%M).zip results/position_mse
```

- Each `compute` runs the same forward passes as `layer_mse.py compute`, thus a few minutes for each checkpoint (an estimate).
- The checkpoints are the checkpoints of the [layer-wise MSE](xai-layer-wise-mse.md) runs. A checkpoint that is not local comes from `HF_REPO`.
- With all batches, each result prints the difference to the stored loss. The earlier runs of `layer_mse.py` give the expected values: 0.000% for float32, −0.10% to −0.21% for bf16.

**Evaluations that complete the accuracy table** (optional, minutes each). Add the values to `accuracy.csv`, then run `rank` again.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MODELS=stage1 TASKS=arc_easy scripts/compare_stages.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MODEL_CONFIG=distill_llama3_2_1b_lizard_w128_fd32_m4 \
  MODELS=stage1 TASKS="piqa arc_easy" scripts/compare_stages.sh
```

The first command evaluates Lizard Run 1 stage 1 (fd128). The second command evaluates fd32 with the LoLCATs recipe.

## Tests

CPU, 2026-10-09, in a copy of the repository with tiny random Llama models and a small Alpaca file. The configs of the copy point to these models.

| Check | Result |
|---|---|
| `layer_mse.py` after the change (its loading part is now the function `load`, which `position_mse.py` uses) | The same values for each layer as before the change. Tiny Lizard, 3 layers: loss 6486.9407, equal to the stored loss. |
| `compute` on tiny Lizard (window 16, sequences of 64 tokens, `--edges 0,16,32,64`, `--from_json`) | The loss equals the loss of `layer_mse.py` and the stored loss. Largest difference of the bucket check: 1.1e-8. |
| `compute` on tiny LoLCATs (window 128, sequences of 1024 tokens, `--edges 0,128,256,512,1024`) | 1000 × MSE of approximately 1e-10 at positions 0–127 and 128–255, and 2,909 and 2,896 at positions 256–511 and 512–1023. Thus the error starts at position 256, where the terraced window ends. The loss equals the stored loss (2175.1218). |
| Default edges with sequences of 1024 tokens | Buckets 0–127, 128–511 and 512–1023. The script cuts the edge 2048 to the sequence length. |
| `rank` on synthetic results with the 9 labels of `accuracy.csv` | A score equal to 100 − PIQA gives ρ = −1.00 with p = 9.9e-5. Two PIQA values tie (57.5), so 4 of the 40,320 orders reach \|ρ\| = 1. All ρ values equal `scipy.stats.spearmanr`. The p-values agree with the sampled permutation test of scipy. |
| `rank --exclude "LoLCATs attention"` | 8 of 9 results remain |

The values of the tiny models do not predict the values at 1B.

## Results

Not run yet.

## Decision rules

From [document 20](../20-layer-mse-for-piqa-arc.md), section 4:

- **The hypothesis holds without LoLCATs** (prediction 3). Then use the loss of positions 0–127 as a screen for new stage 1 runs, together with the accuracy.
- **It does not hold:** keep the accuracy as the screen. The position buckets then still show where each change acts.
- **Prediction 2 holds, and the relative MSE of positions 0–127 is largest in the first layers.** This supports `window_rope` as the first change. It is step 2 of [document 20](../20-layer-mse-for-piqa-arc.md), section 5.
