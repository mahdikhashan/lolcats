# Experiment: `window_rope` with the shared denominator (`gla_norm: hybrid`)

**Status:** A plan from 2026-10-10. The model config exists and passed CPU tests. Stage 1 has not run yet.

## Question

Does the shared denominator of `gla_norm: hybrid` give a better mix of the two branches than `gla_norm: row`, in the `window_rope` setup? Does this mix lower the error at short positions, and increase PIQA and ARC-Easy over `window_rope`?

This is the next run of the decision rules of [RoPE in the window branch](window-rope.md).

## Why this change

| Evidence | Source |
|---|---|
| With RoPE, the window branch gives almost the teacher output at positions 0–127. W alone keeps an error of 0.048 there, with the best weight for each head. | [RoPE in the window branch](window-rope.md), finding 5 |
| At positions 0–127, the trained output has 13× the error of the best mix of the same two branches (0.321 against 0.025) | [RoPE in the window branch](window-rope.md), finding 6 |
| The best weight of the gated branch increases with the position: 0.08 at positions 0–127, 0.65 at 128–511 and 0.79 at 512–2047. With `row`, it is always 1. | [RoPE in the window branch](window-rope.md), finding 7 |
| The loss decreased by 62%, but PIQA closed only 25% of the gap to LoLCATs, and ARC-Easy 18% | [RoPE in the window branch](window-rope.md), finding 2 |
| With `hybrid`, the best weight of the gated branch over all positions was 0.98 (R1b), against 0.55–0.81 with `row` | [XAI: branch decomposition](xai-branch-fit.md), finding 7 |
| LoLCATs uses this scaling, with a window with RoPE. PIQA is 73.5 after stage 1. Its linear branch covers only the keys outside the window. | [LoLCATs control](lolcats-control.md), [document 13](../13-lizard-attention-v2.md) |
| Document 20 recommends the shared denominator together with the window steps | [Document 20](../20-layer-mse-for-piqa-arc.md), section 5 |

## What `hybrid` computes

Let $w_{ij} = \phi(q_i)^\top \phi(k_j)\, \Gamma_{ij}$ be the gated weights over all keys $j \le i$. Let $s_{ij}$ be the window scores with RoPE over the 128 keys of the window. Let $t_k$ be the 4 sink logits. Let $m_i$ be the larger of $\max_j s_{ij}$ and $\max_k t_k$.

- **`row` (`window_rope`):** $y_i = \sum_j \frac{w_{ij}}{\sum_l w_{il}} v_j + \alpha \sum_j p_{ij} v_j$. Here $p_{ij}$ is the softmax over the window scores and the sinks. The gated branch has the weight 1 at each position.
- **`hybrid` (this experiment):** one denominator for both branches and the sinks:

```math
y_i = \frac{\sum_j w_{ij} v_j + |\alpha| \sum_{j \in \text{window}} e^{s_{ij} - m_i} v_j}{\sum_j w_{ij} + |\alpha| \left( \sum_{j \in \text{window}} e^{s_{ij} - m_i} + \sum_k e^{t_k - m_i} \right)}
```

The share of the gated branch is its part of the denominator. In `window_rope`, the gate of layers 1–15 stays above 0.999 for 99.9–100% of the tokens of a 2048-token MMLU prompt. With such a gate, $\sum_j w_{ij}$ has one term for each key, so it grows with the position. The window sum has at most 128 terms. Thus with `hybrid`, the share of the gated branch can increase with the position, as finding 7 asks. At short positions, the share depends on the ratio of $\sum_j w_{ij}$ to $|\alpha|$ times the window sum. The training must set this ratio.

## What changes, and what stays the same

| Setting | `window_rope` | This experiment |
|---|---|---|
| Model config | [`distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope`](../../configs/model/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope.yaml) | [`distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid`](../../configs/model/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid.yaml) |
| **`gla_norm`** | **`row`** | **`hybrid`** |
| **Start value of α (`alpha_init`)** | **1.0** | **0.1** |
| RoPE in the window branch | Yes | Yes |
| Feature dimension, window, sinks | 128, 128, 4 | The same |
| One α, one map and one gate for each layer | Yes | The same |
| Precision, stage 1 recipe, data | bf16, LoLCATs recipe, Alpaca | The same |

**Why the start value of α changes too.** With `hybrid`, α sets the share of the gated branch at the start.

- With α = 0.1, the gated branch has a share of 0.05–0.39 at the start values ([document 13](../13-lizard-attention-v2.md), P6). The measurement used 1B sizes, random q and k, and 1,024 tokens.
- With the same inputs, α = 1.0 gives 0.005–0.06. The share is $r / (r + \alpha)$, where $r$ does not depend on α. Thus the conversion is exact ("Tests").
- With a share near 0, the gated branch has almost no effect on the output at the start. For this reason, run R1 with `joint` stopped ([document 13](../13-lizard-attention-v2.md), P6).
- 0.1 is the start value of the window factor of LoLCATs, and R1b used it with `hybrid`.

Thus this run changes one part, the mix of the branches, with two settings. A run with `hybrid` and α = 1.0 would test the start value alone.

## Predictions

Reference values from `window_rope` ([RoPE in the window branch](window-rope.md), findings 6–9). They convert the relative errors of the branch decomposition into a stage 1 loss, as in [RoPE in the window branch](window-rope.md), "Predictions":

| Mix of the `window_rope` branches | Estimated stage 1 loss |
|---|---|
| The trained output (`row`) | 1.23 |
| The best weights for each head, one weight for all positions | 0.89 |
| The best weights for each head and each position bucket (0–127, 128–511, 512–2047) | 0.71 |
| LoLCATs attention (trained, for comparison) | 0.44 |

1. **Stage 1 loss:** below 1.229. A loss near 0.71 means that `hybrid` finds the best mix in each position bucket. The branches also train, so the loss can also go below 0.71.
2. **Short positions:** the relative MSE of positions 0–127 decreases clearly from 0.321, probably below 0.2. Positions 0–127 are no longer the bucket with the largest error.
3. **Mixing gap at positions 0–127:** the trained output has less than 3× the error of the best G+W mix (`window_rope`: 13×).
4. **α:** at the end, α is above its start value 0.1 in most layers. With RoPE, the window carries most of the teacher output at short positions. In R1b, without RoPE, the mean \|α\| of each layer fell to 0.009–0.072.
5. **Accuracy:** PIQA and ARC-Easy increase over `window_rope` (61.5 and 43.3). A clear gain needs approximately 3.2 points on PIQA and 2.9 points on ARC-Easy (2 unpaired SE).
6. **MMLU subset:** no prediction. The subset cannot separate differences of a few points ([LoLCATs control](lolcats-control.md), "How to compare").

## Risks

- **The window can switch off, as in R1b.** In R1b, the mean \|α\| of each layer fell below its start value 0.1. ARC-Easy fell to 29.9. R1b had no RoPE in the window, so the window was less useful. If α falls below 0.1 here too, this risk is real (prediction 4).
- **The sinks take a part of each row.** With `hybrid`, the sink terms are in the shared denominator. Thus each row sums to less than 1 ([document 13](../13-lizard-attention-v2.md), check 5). With `row`, only the window part has this property.
- **The gated branch still covers the keys inside the window.** `hybrid` can make its share small at short positions, but not 0. Step 3 of [document 20](../20-layer-mse-for-piqa-arc.md) removes these keys from the gated branch.
- **Thesis:** `hybrid` is the scaling of LoLCATs. Report this run as an extension, as `window_rope` ([document 19](../19-next-steps-from-literature.md), part B).

## How to run

### 1. A new Docker image

The job runs the code inside the image, and the image does not have the new config yet ([document 3](../03-infrastructure.md)). After the merge, select Actions → "Docker image" → Run workflow on GitHub.

### 2. Stage 1 on HF Jobs

```bash
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=a100-large HF_TIMEOUT=6h \
  ARGS="--model_config distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid --no_finetune"
```

- **Time and cost:** `window_rope` used the same flavor and timeout. It trained at 1.24 sequences each second, thus approximately 2.5 hours and $6 ([RoPE in the window branch](window-rope.md)). `hybrid` uses the same dense calculation, so the time is probably the same.
- **Early check:** `hf jobs logs <job id> | grep -a "Eval step"`. Compare each value with the curve of `window_rope`:

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 1000 | 1100 |
|---|---|---|---|---|---|---|---|---|
| `window_rope` | 2.0879 | 1.7700 | 1.6123 | 1.5103 | 1.4014 | 1.3315 | 1.2480 | 1.2290 |

- **Checkpoint on the Hub:**

```text
checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
```

### 3. Evaluation and XAI on `student06` (A10)

Set the GPU variables in the shell that runs the commands, for example inside the `tmux` session. The check must print one A10.

```bash
cd /system/user/khashan/lolcats && git pull && conda activate lolcats-env
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"  # 1 NVIDIA A10
MC=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid
DC=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b
CACHE=$HOME/.cache/huggingface/hub
OUT=results/window_rope_hybrid && mkdir -p $OUT

# 1. Benchmarks
MODELS=stage1 TASKS="mmlu_subset piqa arc_easy" MODEL_CONFIG=$MC OUT_DIR=$OUT/stages scripts/compare_stages.sh

# 2. XAI: layer-wise MSE, MSE by token position, branch decomposition
python scripts/layer_mse.py compute $OUT/layer_mse.json --model_config $MC --distill_config $DC \
  --label "v2, window_rope, hybrid, fd128, LoLCATs recipe, bf16" --cache_dir $CACHE
python scripts/position_mse.py compute $OUT/position_mse.json --from_json $OUT/layer_mse.json --cache_dir $CACHE
python scripts/branch_fit.py compute $OUT/branch_fit.json --from_json $OUT/layer_mse.json --cache_dir $CACHE

# 3. Plots against window_rope
WR=docs/experiments
python scripts/layer_mse.py plot $OUT/layer_mse_vs_window_rope $WR/xai-layer-wise-mse/v2_window_rope.json $OUT/layer_mse.json
python scripts/layer_mse.py plot $OUT/layer_mse_vs_window_rope_relative $WR/xai-layer-wise-mse/v2_window_rope.json \
  $OUT/layer_mse.json --relative
python scripts/position_mse.py heatmap $OUT/position_heatmap $WR/xai-position-mse/v2_window_rope.json $OUT/position_mse.json
for bucket in 0-127 all; do
  python scripts/branch_fit.py plot $OUT/branch_fit_$bucket $WR/xai-branch-fit/v2_window_rope.json $OUT/branch_fit.json --bucket $bucket
done

zip -r window-rope-hybrid-$(date +%Y%m%d-%H%M).zip $OUT
```

- `compare_stages.sh` finds the checkpoint from `MODEL_CONFIG` and the default recipe. It downloads a missing checkpoint from `HF_REPO`.
- **Branch decomposition with `hybrid`:** G and W are the two branches with the shared denominator, and W is without |α|. If the trained mix is the best mix, the best weights are 1 for G and |α| for W.

## Tests

CPU, 2026-10-10.

| Check | Result |
|---|---|
| The config against `window_rope` | The model section is equal. In the attention section, only `gla_norm` (`hybrid`) and `alpha_init` (0.1) differ. |
| `pytest tests/test_lizard_attention_v2.py` | 60 passed. They include `hybrid` against the loop reference and the decode form (checks 2 and 3). They also include the gated share at the start values (check 10) and \|α\| (check 11). |
| The share of the gated branch for α = 1.0, from the share for α = 0.1 | One layer of the test model, window 32, 128 positions, the same feature maps. The formula $r / (r + \alpha)$ gives the share for α = 1.0 from the share for α = 0.1. The largest difference is 4.4e-16. |
| Training of a tiny copy of the config (3 layers, window 16, feature dimension 8). The real stage 1 recipe, sequences of 1024 tokens, 8 steps. | The validation loss decreased at each evaluation: 4444, 4271, 4141. α increased from 0.1 to 0.157–0.158 in the 3 layers. The checkpoint has 15 Lizard tensors (3 layers × 5). It loads with 0 missing and 0 unexpected keys, and `layer_mse.py` reproduces its stored loss. `branch_fit.py` reproduces the output of the checkpoint with a relative error of 8.0e-8. |

The values of the tiny model do not predict the values at 1B. After the training, `distill_llama.py` stopped with the known error of the tiny tokenizer in the sample generation (`token_type_ids`, [RoPE in the window branch](window-rope.md), "Tests").

## Results

Not run yet.

## Decision rules

- **PIQA or ARC-Easy increases by 2 SE or more over `window_rope`:** the shared denominator is a real gain. Next: stage 2 for this config. If the relative MSE of positions 0–127 stays above 0.1, also test step 3 of [document 20](../20-layer-mse-for-piqa-arc.md). In step 3, the gated branch gets only the keys outside the window.
- **The loss decreases, but the accuracy does not:** check the relative MSE of positions 0–127 and the best weight of G there. If G still has a large share at short positions, step 3 of document 20 is next.
- **α falls below 0.1 in most layers (the window switches off, as in R1b):** this start value does not work with `hybrid`. Next: step 3 of document 20, or `hybrid` with α = 1.0.
- **The loss stays above 1.229:** `hybrid` does not help in this setup. Keep `row`, and test step 3 of document 20.
