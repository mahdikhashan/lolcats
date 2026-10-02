# Experiment (XAI): layer-wise MSE

**Status:** The script `scripts/layer_mse.py` and its CPU test are ready. It has not run on the real stage 1 checkpoints yet.

## Question

In which layers does the stage 1 Lizard attention differ most from the softmax attention of the teacher? Do the stage 1 recipes differ in all layers, or only in some layers?

The stage 1 validation loss (`distill/eval/loss`) is a mean over the 16 layers. It cannot show which layers cause the error. Figure 14 of the LoLCATs paper shows the error for each layer:

- **Figure 14a:** the MSE of each layer, for two attention types (Hedgehog linear attention, and LoLCATs linear + sliding window attention). Each type has one line after attention transfer and one line after LoRA. The model is Llama 3 8B.
- **Figure 14b:** the change of the MSE of each layer from LoRA.
- In that figure, the MSE is small in the first layers and largest in the last layers.

This experiment makes the same plot for the Lizard checkpoints of this project. It belongs to the XAI ideas ([XAI for the distillation](xai.md)), because it shows where in the model the approximation fails.

## Can we make the plot without retraining?

Yes. The layer-wise MSE needs only forward passes:

- **The teacher weights** (`meta-llama/Llama-3.2-1B`) and **the stage 1 checkpoint** (the Lizard weights of each layer).
- **The stage 1 validation data**: the first 200 examples of Alpaca-cleaned, in chunks of 2,048 tokens.
- In distillation mode (`train_attention`), each Lizard layer already calculates two outputs from the same q, k and v ([math formulas](../math-formula.md), section 3):
  - the teacher output: softmax attention with RoPE,
  - the Lizard output: the gated branch plus α × the window branch.
- The next layer gets the teacher output. Thus the error of one layer does not reach the next layer. Each value measures the approximation of one layer alone.

The stage 1 trainer calculates the same values. Then it keeps only 1000 × their mean. Thus the script must reproduce the stored validation loss of each checkpoint. This is a built-in check of the script.

**Stage 2** is also possible without retraining. After LoRA, q, k and v change. Then the teacher output comes from softmax attention with the adapted q, k and v. This document assumes that Figure 14 ("Attn. Transfer + LoRA") uses the same reference. The caption of the paper does not say this exactly. The script measures only stage 1 now.

## What the script calculates

`python scripts/layer_mse.py compute` writes one JSON file for each checkpoint. For each layer, it has:

| Value | Meaning |
|---|---|
| `mse` | MSE between the Lizard output and the teacher output, before `o_proj`. The stage 1 trainer calculates it in the same way (`nn.MSELoss`, mean over all elements). |
| `target_mean_square` | The mean square of the teacher output |
| `relative_mse` | `mse` / `target_mean_square`. The output scale differs between layers. The relative MSE removes this scale. |
| `mse_per_head` | The MSE of each of the 32 query heads |

All values are means over the validation batches. The file also has 1000 × the mean over the layers (`loss`) and the stored loss of the checkpoint. It also has the checkpoint check of `compare_stages.py` (SHA-256, step, missing keys).

`python scripts/layer_mse.py plot` reads several JSON files. It writes a PNG with two panels and a Markdown table with the same values:

- **(a)** the MSE of each layer, one line for each checkpoint. The y-axis shows 1000 × MSE. Thus the mean of each line is the validation loss of the checkpoint.
- **(b)** with 2 or more checkpoints: the change of each layer against the first checkpoint. A bar below 0 means a lower MSE than the first checkpoint. Figure 14b shows the absolute change. This plot keeps the sign.
- With `--relative`, both panels use the relative MSE.

## Predictions

1. **The check:** 1000 × the mean over the layers agrees with the stored loss of each checkpoint. An example is 3.9764 for config 1 of the [second round](second-round.md). Small differences can come from the GPU.
2. **The MSE differs between the layers**. As in Figure 14, the last layers probably have the largest MSE. This prediction is weak, because Figure 14 shows Llama 3 8B, not Llama 3.2 1B.
3. **The relative MSE can give a different order of the layers**, because the output scale differs between layers.
4. **The recipes differ most in the layers where their gates differ most**. The LoLCATs recipe gives the lowest validation loss (3.2549 with fd128, 3.4219 with fd32). Its gate saturates in 14 or 15 of the layers 1–15. With the recipe of the paper, the gate decays fast in layers 0–2 ([float32 experiment](float32.md), finding 3). Section 13.2 of the [gap analysis](../11-gap-analysis.md) gives the gate-state hypothesis. If it is right, panel (b) shows the largest differences in these layers.

## How to run

### Stage 1 checkpoints

| Label | Model config | Distill config | Stored loss |
|---|---|---|---|
| fd128, LoLCATs recipe, bf16 | `distill_llama3_2_1b_lizard_w128_fd128_m4` | `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b` | 3.2549 |
| fd32, LoLCATs recipe, bf16 | `distill_llama3_2_1b_lizard_w128_fd32_m4` | `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b` | 3.4219 |
| fd32, recipe of the paper, bf16 | `distill_llama3_2_1b_lizard_w128_fd32_m4` | `distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b` | 8.1641 |
| fd32, recipe of the paper, float32 | `distill_llama3_2_1b_lizard_w128_fd32_m4_fp32` | `distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b` | 4.9478 |
| Second round, config 1 | `distill_llama3_2_1b_lizard_w128_fd32_m4_fp32` | `distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b` | 3.9764 |
| Gradient clipping, config 2 | `distill_llama3_2_1b_lizard_w128_fd32_m4_fp32` | `distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_1b` | After its training |

### Commands on the A10

The script finds the checkpoint name from the two configs. If the file is not in `checkpoints/`, it downloads it from `HF_REPO` (default `nanoman1/lolcats-lizard-llama-3.2-1b`). Run it from the repository root, in the `lolcats-env` environment:

```bash
conda activate lolcats-env
export HF_TOKEN=<token with access to HF_REPO and meta-llama/Llama-3.2-1B>
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
CACHE=$HOME/.cache/huggingface/hub   # replaces the cache_dir of the model config
run() {  # label, model config, distill config, output name
  python scripts/layer_mse.py compute "results/layer_mse/$4.json" --label "$1" \
    --model_config "$2" --distill_config "$3" --cache_dir "$CACHE" 2>&1 | tee "results/layer_mse/$4.log"
}
mkdir -p results/layer_mse
run "fd128, LoLCATs recipe, bf16" distill_llama3_2_1b_lizard_w128_fd128_m4 distill_alpaca_clean_xent0_mse1000_lr1e-2_1b fd128_lolcats
run "fd32, LoLCATs recipe, bf16" distill_llama3_2_1b_lizard_w128_fd32_m4 distill_alpaca_clean_xent0_mse1000_lr1e-2_1b fd32_lolcats
run "fd32, paper recipe, bf16" distill_llama3_2_1b_lizard_w128_fd32_m4 distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b fd32_paper_bf16
run "fd32, paper recipe, float32" distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 distill_alpaca_clean_xent0_mse1000_lr1e-3_cosine_1b fd32_paper_fp32
run "Second round, config 1" distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b config1

python scripts/layer_mse.py plot results/layer_mse/stage1 \
  results/layer_mse/fd32_lolcats.json results/layer_mse/fd128_lolcats.json \
  results/layer_mse/fd32_paper_bf16.json results/layer_mse/fd32_paper_fp32.json results/layer_mse/config1.json
python scripts/layer_mse.py plot results/layer_mse/stage1_relative --relative \
  results/layer_mse/fd32_lolcats.json results/layer_mse/fd128_lolcats.json \
  results/layer_mse/fd32_paper_bf16.json results/layer_mse/fd32_paper_fp32.json results/layer_mse/config1.json
```

The first file of `plot` is the reference of panel (b). In the commands above, it is fd32 with the LoLCATs recipe, the run with the lowest validation loss at feature dimension 32. Each `compute` log ends with the difference to the stored loss. Check it before you use the plot (prediction 1).

Options:

- `--checkpoint init` uses no checkpoint: the Lizard weights at initialization (seed 0). It shows how much stage 1 lowers the MSE of each layer. The initial MSE can be much larger than the trained MSE. Thus put it in a separate plot.
- `--torch_dtype float32` runs a bf16 model config in float32. Then all runs use the same precision. The result of a bf16 run then does not reproduce its stored loss exactly.
- `--max_batches N` uses only the first N validation batches, for a fast check.

## How to read the result

| Result | Meaning | Next step |
|---|---|---|
| A few layers have most of the MSE | The approximation fails mainly in these layers | Examine the gates, the sinks and `mse_per_head` of these layers. As an ablation, keep softmax attention in these layers (`softmax_attentions` in the model config). |
| The MSE is similar in all layers | A cause that affects all layers, for example the recipe or the normalization of the gated branch (D1 in [math against code](../math-code-discrepancy.md)) | Test D1 with the single-layer bench of section 13.5 of the gap analysis (step 5) |
| The recipes differ in the same layers as their gates | Agrees with the gate-state hypothesis (section 13.2 of the gap analysis) | Compare the gate values and the MSE of these layers directly |

## Tests

These tests ran on CPU, in a copy of the repository with a tiny Llama (3 layers, 4 heads) and synthetic Alpaca-style data:

- **Reproduction of the stored loss**. The real `distill_llama.main()` trained stage 1 for 25 steps and saved the best checkpoint. `layer_mse.py compute` with the default checkpoint name gave 6486.9407. The stored loss is 6486.9407. The difference is 0.000%.
- **Initialization**. With `--checkpoint init`, the script gave 6811.1495. This is the same value as the evaluation of the trainer at step 0.
- **Checkpoint check**. 15 expected Lizard keys (3 layers × 5), 0 missing, 0 unexpected, 0 not loaded.
- **Plots**. `plot` wrote the PNG and the table for 1 result, for 2 results, and with `--relative`. A layout check with 16 layers and 6 synthetic results showed no overlap of the legend and the lines.

## Results

Not run yet.
