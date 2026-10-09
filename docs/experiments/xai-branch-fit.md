# Experiment (XAI): branch decomposition

**Status:** The script `scripts/branch_fit.py` exists and passed CPU tests on 2026-10-09. It has not run on the real checkpoints yet.

## Question

This is X2 of [document 20](../20-layer-mse-for-piqa-arc.md). Which branch of the Lizard attention limits the fit to the teacher: the window branch without RoPE, or the gated branch? How much of the error at short positions comes from the missing RoPE?

[XAI: MSE by token position](xai-position-mse.md) found that Lizard misses 35–63% of the teacher output at positions 0–127, where all keys are inside the window. X2 asks which part of the attention causes this.

## What the script calculates

`compute` uses the model, the data and the checkpoint of `scripts/layer_mse.py`. Each layer gets the teacher input. For each layer and head, the script calculates these outputs from the same q, k and v, before `o_proj`:

| Name | Output | Source |
|---|---|---|
| y | The teacher: softmax attention with RoPE over all previous keys. This is the stage 1 target. | Teacher |
| G | The gated branch: Hedgehog maps, gate and normalization | Checkpoint |
| W | The window branch, without the factor α. In v1, it has no RoPE and has the sinks. | Checkpoint |
| WR | Softmax attention with RoPE over the same window of keys, without sinks | Oracle, no trained parameters |
| trained | The output of the checkpoint. In v1, it is G + α W. | Checkpoint |

**The fit.** For each set of candidates and each head, the script finds the weights c with the lowest error ‖y − Σ c_s y_s‖². The sum is over the candidates in the set. The error is over the positions of one bucket. The script reports the remaining error divided by ‖y‖².

- **Sets:** G, W, WR, G+W, G+WR, W+WR and G+W+WR.
- **Meaning:** each branch gets one free weight for each head and bucket. Thus no scale of these branches, such as α, gives a lower error.
- **Buckets:** `--edges 0,128,512,2048` (the default) gives the query positions 0–127, 128–511 and 512–2047. The script also fits all positions together.
- **The error of a layer:** the sum of the remaining errors of its heads, divided by the sum of ‖y‖² of its heads. Each head has its own weights.
- **Output:** for each layer and bucket, the error of the layer, the error of each head and the weights of each head. The summary gives the mean over the layers.
- **v2 checkpoints:** the script uses the branch weights of the layer (`branch_weights`). W is the window part divided by α of its head.
- **LoLCATs:** not supported. Its two parts share one denominator, so they are not separate branches of this kind.

**Checks:**

1. G + α W (or the v2 branch weights times v) reproduces the output of the checkpoint.
2. A query before position 128 sees all its keys inside the window. There, WR is the teacher attention, so its error is only rounding.
3. The trained output is one point of the G+W fit (with the weights 1 and α). Thus its error is never below the error of the G+W fit.

`plot` draws one heatmap for each result: the layers in the rows, the candidate sets and `trained` in the columns. `--bucket` selects the positions. All panels use one color scale.

### What each comparison shows

| Comparison | Question |
|---|---|
| W against WR, positions 0–127 | How much error comes from the window branch itself: the missing RoPE and the sinks? |
| G+W against W, positions 0–127 | Does the gated branch help where all keys are inside the window? |
| G+WR against WR, positions 512–2047 | Does the gated branch give the part of the teacher output from keys outside the window? |
| trained against G+W | Does the training reach the best scale of the two branches? The difference is the mixing gap. |

## Predictions

1. **WR at positions 0–127:** only rounding. For bf16 checkpoints, this is approximately 1e-5, as for LoLCATs in [XAI: MSE by token position](xai-position-mse.md), finding 1.
2. **W against WR at positions 0–127:** W has a much larger error, most of all in layers 0–3. The window without RoPE cannot give the local attention of the teacher ([document 15](../15-attention-math-side-by-side.md), claim F). Layers 2 and 3 have the largest error at these positions (X1, finding 3).
3. **Mixing gap:** for the v1 checkpoints, the G+W fit has a clearly lower error than the trained output in some layers. One α for each layer, and the weight 1 of the gated branch, are fewer free weights than the fit has. For C1 (α for each head), the gap is smaller.

## Limits

- **The gated branch is the trained one.** The checkpoint trained G together with W, not with WR. A model with `window_rope` would train another G. Thus G+WR shows what the current G adds to an exact window, not the best possible result.
- **The fit has more freedom than the model.** It has one weight for each branch, head and bucket. Thus its error is a lower limit for these branches. The model itself can be worse.
- **WR is an oracle.** It has no sinks and uses the teacher scores. A trained window with RoPE and sinks can differ.
- **One data set.** The data loader packs the Alpaca validation data into sequences. A validation sequence usually starts inside a sample.

## How to run

On `student06`, in the `lolcats-env` environment, from the repository root. Set the GPU variables in the shell that runs the commands, for example inside the `tmux` session. The check must print one A10.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"  # 1 NVIDIA A10

mkdir -p results/branch_fit
for name in fd32_lolcats fd128_lolcats fd32_paper_bf16 fd32_paper_fp32 config1 config2 \
            v2_alphahead v2_perhead_hybrid; do
  python scripts/branch_fit.py compute results/branch_fit/$name.json \
    --from_json docs/experiments/xai-layer-wise-mse/$name.json --cache_dir ~/.cache/huggingface/hub \
    2>&1 | tee results/branch_fit/$name.log
done

for bucket in 0-127 512-2047 all; do
  python scripts/branch_fit.py plot results/branch_fit/branch_fit_$bucket results/branch_fit/*.json --bucket $bucket
done

zip -r branch-fit-$(date +%Y%m%d-%H%M).zip results/branch_fit
```

- **Checkpoints:** the 8 Lizard checkpoints of [XAI: layer-wise MSE](xai-layer-wise-mse.md). `--from_json` takes their configs, checkpoint paths and labels.
- **Time:** each `compute` runs the forward passes of `layer_mse.py` and some dense matrices for each layer. Thus a few minutes for each checkpoint (an estimate).
- **Check in the log:** the last lines give the mean errors and the largest error of check 1. For bf16 checkpoints, check 1 is of the order of 1e-3, because the checkpoint rounds its output to bf16.

## Tests

CPU, 2026-10-09, in a copy of the repository with tiny random Llama models (3 layers, 4 heads, window 16) and sequences of 64 tokens. The configs of the copy point to these models.

| Check | Result |
|---|---|
| Tiny v1 checkpoint, `--edges 0,16,32,64` | Check 1: 0 (float32). WR at positions 0–15: below 1e-15. The `trained` column equals the relative MSE of `position_mse.py` for each bucket (0.7908, 1.481 and 1.890). |
| numpy `lstsq` on the raw vectors of layer 1, for the sets WR, G+W and G+W+WR, buckets 16–31 and all | The largest relative difference to the JSON is 3.4e-6, from the 6 significant digits in the JSON. For each head, the error of `trained` is at least the error of the G+W fit. |
| Tiny v2 checkpoints: the C1 config (α for each head) and the R1b config (`hybrid`) | Check 1: 1.0e-7 and 8.6e-8. The JSON has α of each head. |
| LoLCATs checkpoint | The script stops with the message that it supports Lizard layers only. |
| `plot` | 3 results with the buckets 0–15 and all, and 8 synthetic results with 16 layers. An unknown bucket stops with a message. |

The values of the tiny models do not predict the values at 1B.

## Results

Not run yet.

## Decision rules

From [document 20](../20-layer-mse-for-piqa-arc.md), section 4:

- **Window:** compare W and WR at positions 0–127. If the error of WR is much lower, the missing RoPE is the cause. Then `window_rope` comes first.
- **Gated branch:** add the gated branch to the window with RoPE. If the error at positions 128–511 and 512–2047 does not decrease, the gated branch or its maps are the limit. Then X3 decides between the maps and the gate.
- **Mixing gap:** the trained output can be far above the G+W fit in a layer. Then the scale of the branches limits that layer, not the branches. α for each head (C1) and `gla_norm: hybrid` (R1b) change this scale. Their results show if these options close the gap.
