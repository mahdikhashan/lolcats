# Experiment (XAI): sample attention weights

**Status:** The script `scripts/attention_weights.py` and its CPU test are ready. It has not run on the real checkpoints yet.

## Question

Stage 1 trains Lizard to reproduce the attention **outputs** of the teacher (MSE loss). Does Lizard then also reproduce the attention **weights** of the teacher? Does it reproduce them over distances outside its window of 128 tokens?

The [layer-wise MSE experiment](xai-layer-wise-mse.md) shows where the outputs differ. Its largest single source of error is heads 14 and 23 of layer 15. A possible cause is attention to tokens far outside the window. The attention weights can test this hypothesis directly.

## The figures of LoLCATs

Appendix E.2 of the LoLCATs paper ("Sample Attention Weights", Figures 18–21) shows sample attention weights of Llama 3 8B:

- **Samples:** held-out packed Alpaca samples of 1024 tokens.
- **Panels:** the last 32 queries against the first 32 and the last 32 keys. A dashed line separates the two key groups. The distance between the last queries and the first keys is much larger than the window of LoLCATs (64 tokens).
- **Rows:** softmax attention, LoLCATs (MSE), LoLCATs (XENT), Hedgehog (MSE), LoLCATs (init.) and Hedgehog (init.).
- **Columns:** layers 0, 8, 16, 24 and 31. Figures 18–21 show heads 0–3, one figure for each head.

The paper reports these observations:

- The training of LoLCATs uses only the attention outputs (MSE). LoLCATs often recovers the softmax attention weights qualitatively, with a quality comparable to an explicit loss on the weights (XENT).
- LoLCATs often recovers the weights over distances outside its window. Thus it learns both the feature maps and the weighting factors.
- Newly initialized LoLCATs layers do not capture the softmax attention weights (init.). Thus attention transfer is necessary.
- Trained LoLCATs layers match the weights better than trained Hedgehog layers (the same feature map, but no sliding window).

## The attention weights of Lizard

**Teacher**. The teacher uses causal softmax attention with RoPE ([math formulas](../math-formula.md), section 3):

```math
A^{T}_{i,t} = \frac{\exp\left( \varphi_R(\mathbf{q}_i)^\top \varphi_R(\mathbf{k}_t) / \sqrt{d} \right)}{\sum_{j \le i} \exp\left( \varphi_R(\mathbf{q}_i)^\top \varphi_R(\mathbf{k}_j) / \sqrt{d} \right)}, \qquad t \le i
```

**Lizard**. The output of a Lizard layer is a weighted sum of the values too. Thus it has an exact weight matrix $A$ with $\hat{\mathbf{y}}_i = \sum_t A_{i,t} \mathbf{v}_t$:

```math
A = A^{gla} + \alpha \, A^{window}
```

```math
A^{gla}_{i,t} = \frac{\phi_q(\mathbf{q}_i)^\top \phi_k(\mathbf{k}_t) \prod_{l=t+1}^{i} \gamma_l}{\sum_{j \le i} \phi_q(\mathbf{q}_i)^\top \phi_k(\mathbf{k}_j) \prod_{l=j+1}^{i} \gamma_l}, \qquad A^{window}_{i,t} = \frac{\exp\left( \mathbf{q}_i^\top \mathbf{k}_t / \sqrt{d} \right)}{\sum_{j} \exp(t_j) + \sum_{t' = i-w+1}^{i} \exp\left( \mathbf{q}_i^\top \mathbf{k}_{t'} / \sqrt{d} \right)}
```

$A^{window}_{i,t}$ is 0 for keys outside the window ($t \le i - w$). The code calculates the same terms (`gla`, `awa` and `sink_softmax` in `lizard_attention.py`).

Two differences to softmax attention matter for the plots:

- **The rows of $A$ do not sum to 1**. $A^{gla}$ sums to 1. The sinks take a part of the window mass. Thus a row of $A$ sums to $1 + \alpha \, (1 - \text{sink mass})$. The plots show $A$ unchanged. The total variation distance (below) normalizes the rows first.
- **Lizard has no RoPE**. Both branches use q and k before RoPE ([math against code](../math-code-discrepancy.md)). Thus Lizard cannot use the relative position of two tokens directly, also inside the window.

## Differences to the figures of LoLCATs

| Item | LoLCATs | This experiment |
|---|---|---|
| Model | Llama 3 8B, 32 layers | Llama 3.2 1B, 16 layers |
| Window | 64 tokens | 128 tokens. The last 32 queries and the first 32 keys are more than 960 tokens apart. |
| Columns | Layers 0, 8, 16, 24, 31 | Default layers 0, 4, 8, 12, 15: the same fractions of the depth |
| Rows | Softmax, then the trained and the initialized attention types | Softmax (teacher), then one row for each result file (stage 1, stage 2 or init.) |
| Color scale | Not given | Shared in each column: 0 to the largest value of the column. Thus the rows of one column are comparable. |
| Layer input | Not given | Default `--inputs teacher`: each Lizard layer gets the input of the same teacher layer. `--inputs own`: the Lizard model runs on its own hidden states. |

## What the script calculates

`python scripts/attention_weights.py compute` writes one JSON file for one checkpoint:

- **Samples:** the stage 1 validation split (the first 200 examples of Alpaca-cleaned, not used in training), packed into chunks of `--seq_len` tokens (default 1024). `--samples` sets the number of chunks (default 1).
- **Checkpoint:** stage 1 (default), stage 2, or the initial Lizard weights (`--checkpoint init`, seed 0). With `--stage 2`, the script loads the stage 1 checkpoint and then the LoRA weights of stage 2.
- **Crops:** the panels of the figure for `--layers` and `--heads` (default heads 0–3). Each crop is the last `--queries` queries (default 32) against the first and the last `--queries` keys. The file has a crop for the teacher, for Lizard ($A$) and for the gated branch ($A^{gla}$).
- **Metrics for all layers and heads:** means over the queries at or after the window (these queries have keys outside the window):

| Metric | Meaning |
|---|---|
| `teacher_first_key`, `lizard_first_key` | The weight on the first key of the sample. In many teacher layers, the first key is an attention sink. |
| `teacher_outside_window`, `lizard_outside_window` | The sum of the weights on keys outside the window. For Lizard, only the gated branch can give such weight. |
| `tv_distance` | The total variation distance between the teacher row and the normalized Lizard row: 0.5 × the sum of the absolute differences. 0 means equal rows. 1 means no overlap. |
| `row_sum` | The sum of a Lizard row: $1 + \alpha \, (1 - \text{sink mass})$ |
| `window_share` | The share of the Lizard row from the window branch: $\alpha \sum_t A^{window}_{i,t}$ / row sum |
| `window_sink_mass` | The mass that the sinks take in the window branch |

`python scripts/attention_weights.py plot` reads one or more JSON files. It writes:

- one PNG for each head (`OUT_head<h>.png`), with the layout of Figures 18–21. The first row is the teacher of the first file.
- `OUT_metrics.png`: for all layers and heads, the mass of the teacher outside the window, and the TV distance of each result.
- `OUT_metrics.md`: the metrics as tables. One table has the means for each layer. One table for each result has the heads with the largest TV distance.

### Checks

The script checks both weight matrices in every layer:

- **Teacher:** $A^{T}$ times v, through `o_proj`, against the output of the teacher layer. A check without RoPE gave a relative error of 0.89. Thus the check finds a wrong teacher matrix.
- **Lizard:** $A$ times v against the output of the Lizard layer (`lizard`).

Each log ends with the largest relative error of the checks.

**Stage 2 runs in bf16**. `create_peft_config` (`src/model/peft.py`) casts the whole model to bf16 by default (`target_dtype='bfloat16'`). The evaluation (`load_model_for_eval.py`) and the stage 2 training call it in the same way. Thus a stage 2 model is bf16, also with a float32 model config. The script does the same. Then the Lizard check has the size of bf16 rounding (approximately 2e-3).

## Predictions

1. **Initialization**: the initialized Lizard layers do not reproduce the teacher weights, as in LoLCATs. At initialization, γ = 0.5. Thus the gated branch keeps only 0.5^10 ≈ 0.001 of a token 10 positions back. The first keys stay empty, and the mass outside the window is approximately 0.
2. **The first key**: in many middle layers, the teacher puts much weight on the first key (as in Figure 18, layers 8–24). Lizard can reach this key only through the gated branch. With the LoLCATs recipe, the gate saturates, so the gated branch can keep the first key. With the recipe of the paper, the gate decays fast in some layers. Thus the LoLCATs-recipe checkpoints probably put more weight on the first key. This relates to the sink hypothesis (section 13.2 of the [gap analysis](../11-gap-analysis.md)).
3. **Inside the window**: trained Lizard reproduces patterns that depend on the content of the tokens. Patterns that depend on the position, such as a sharp diagonal, are harder without RoPE.
4. **Layer 15, heads 14 and 23**: the teacher puts much weight outside the window, and Lizard misses most of it. Then these heads have a large TV distance.

## How to run

Run the commands from the repository root on the A10, in the `lolcats-env` environment. As in `layer_mse.py`, the script finds the checkpoint names from the configs and downloads missing files from `HF_REPO`:

```bash
conda activate lolcats-env
export HF_TOKEN=<token with access to HF_REPO and meta-llama/Llama-3.2-1B>
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
CACHE=$HOME/.cache/huggingface/hub
run() {  # output name, label, model config, distill config, other options
  python scripts/attention_weights.py compute "results/attention_weights/$1.json" --label "$2" \
    --model_config "$3" --distill_config "$4" --cache_dir "$CACHE" "${@:5}" 2>&1 | tee "results/attention_weights/$1.log"
}
mkdir -p results/attention_weights
run fd32_lolcats "fd32, LoLCATs recipe" distill_llama3_2_1b_lizard_w128_fd32_m4 distill_alpaca_clean_xent0_mse1000_lr1e-2_1b
run config1 "Second round, config 1" distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b
run init "Lizard (init.)" distill_llama3_2_1b_lizard_w128_fd32_m4_fp32 distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b --checkpoint init
run fd128_stage2 "fd128, LoLCATs recipe, stage 2" distill_llama3_2_1b_lizard_w128_fd128_m4 distill_alpaca_clean_xent0_mse1000_lr1e-2_1b --stage 2

python scripts/attention_weights.py plot results/attention_weights/heads0-3 \
  results/attention_weights/config1.json results/attention_weights/fd32_lolcats.json \
  results/attention_weights/fd128_stage2.json results/attention_weights/init.json
```

- The first file of `plot` gives the teacher row. The first file above uses a float32 model config, so the teacher row is float32.
- `--stage 2` uses the stage 2 checkpoint name of `compare_stages.sh`. For another name, give `--finetune_checkpoint`.
- For prediction 4, add the crops of the heads with the largest MSE. For example: `--layers 13 14 15 --heads 14 23`, with other output names. The metrics already cover all layers and heads.

## How to read the result

| Result | Meaning |
|---|---|
| A trained row looks like the teacher row, also left of the dashed line | Lizard reproduces the teacher weights, also outside the window, as LoLCATs in Figures 18–21 |
| A trained row matches only right of the dashed line | Lizard reproduces the local weights, but not the weights outside the window. The gated branch does not carry the long-distance part. |
| The teacher has a strong first-key column, and Lizard has none | Lizard misses the attention sink of the teacher (prediction 2) |
| A head has a large TV distance and a large teacher mass outside the window | The error of this head comes from distances outside the window (prediction 4) |

Attention weights help with the diagnosis. But they do not prove which mechanism carries the knowledge ([XAI for the distillation](xai.md), section 9).

## Tests

These tests ran on CPU, in a copy of the repository with a tiny Llama (3 layers, 4 heads) and synthetic Alpaca-style data. The chunks had 64 tokens, the crops 8 queries, and the window 16 tokens:

- **Four results**. The stage 1 checkpoint from 25 training steps, and the initial weights. A stage 2 checkpoint (LoRA with random nonzero `lora_B`), with `--inputs teacher` and with `--inputs own`. All four finished with exit 0.
- **Checks**. Teacher: 0.0 in every layer (float32, eager attention, the same operations). Lizard: at most 1e-7 for stage 1 and the initial weights, and 1.7e-3 for stage 2 (bf16, see "Checks").
- **The checks find errors**. The teacher matrix without RoPE gave a relative error of 0.89.
- **The metrics respond**. With the initial weights, the row sum was 1.99 and the window share 0.50 (α = 1). With γ forced to approximately 0.97, the Lizard mass outside the window rose from 1.5e-5 to 0.19.
- **Checkpoint check**. Stage 1: 15 expected keys. Stage 2: 24 LoRA keys. 0 missing, 0 unexpected, 0 not loaded.
- **Plots**. `plot` wrote the head figures, the metrics figure and the tables for 3 and 4 results. `--layers 2 --heads 1 3` gave crops only for these, and `--layers 7` on the 3-layer model stopped with a clear error.

## Results

Not run yet.
