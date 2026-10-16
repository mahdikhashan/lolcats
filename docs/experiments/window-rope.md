# Experiment: RoPE in the window branch (`window_rope`)

**Status:** Stage 1 and stage 2 finished on 2026-10-10. After stage 2: PIQA 73.1 and ARC-Easy 63.8, against 68.0 and 54.8 for Run 2 of Lizard Run 1. MMLU subset 25.6. On all MMLU questions: 27.2, against 31.7 for the teacher ("Results", MMLU, all questions). ARC-Challenge 36.1 and HellaSwag 60.7 (normalized accuracy, 2026-10-11). WinoGrande 59.6 (accuracy, 2026-10-12, "Results", Table 9 tasks). The teacher in this harness gets 36.3, 63.7 and 60.1 on these three tasks (finding 25). The teacher in this harness has 74.4 and 65.3. After stage 1: validation loss 1.2290 (Run 1: 3.2549). PIQA 61.5 (Run 1: 57.6), ARC-Easy 43.3 (Run 1: 39.0), MMLU subset 24.6 (Run 1: 22.5). All six predictions hold. At positions 0–127, the window branch now gives almost the teacher output. There, most of the error comes from the mix of the two branches. The gated branch keeps the weight 1, but its best weight there is 0.08 ("Results", findings 6 and 7).

## Question

Does RoPE in the window branch give a better stage 1, and a higher PIQA and ARC-Easy accuracy, than Lizard Run 1? This is step 2 of [document 20](../20-layer-mse-for-piqa-arc.md), section 5, and step B5 of [document 19](../19-next-steps-from-literature.md).

## Why this change

| Evidence | Source |
|---|---|
| Inside the window, the window branch without RoPE explains only approximately 60% of the teacher output, even with the best weight for each head. A window with RoPE explains all of it. | [XAI: branch decomposition](xai-branch-fit.md), findings 1 and 2 |
| A window with RoPE halves the error of the current branches, and cuts it 4.4–6.9 times in layers 0 and 15 | [XAI: branch decomposition](xai-branch-fit.md), finding 5 |
| Heads 14 and 23 of layer 15 give approximately 21% of the stage 1 loss. With a window with RoPE, their error is 0.017 and 0.134. | [XAI: branch decomposition](xai-branch-fit.md), finding 6 |
| The window of LoLCATs has RoPE. On short prompts, it computes the teacher attention, and PIQA is 73.5 against 57.6. | [LoLCATs control](lolcats-control.md), finding 2 |
| A window without RoPE cannot give the previous-token attention of layer 0 | [Document 15](../15-attention-math-side-by-side.md), claim F |
| The Liger paper writes its window without RoPE, but the Liger code applies RoPE | [Document 15](../15-attention-math-side-by-side.md), table |

The Lizard paper writes the window without RoPE. Thus the thesis must report this run as an extension, not as the reproduction ([document 19](../19-next-steps-from-literature.md), part B).

## What changes, and what stays the same

| Setting | Lizard Run 1 | This experiment |
|---|---|---|
| Model config | `distill_llama3_2_1b_lizard_w128_fd128_m4` | [`distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope`](../../configs/model/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope.yaml) |
| Attention code | v1 (`lizard_attention.py`) | v2 (`lizard_attention_v2.py`). With the default options, v2 gives the outputs of v1 (`test_defaults_match_v1_exactly`). |
| **RoPE in the window branch** | **No** | **Yes (`window_rope: true`)** |
| Gated branch | No RoPE | No RoPE (no change) |
| Feature dimension, window, sinks | 128, 128, 4 | The same |
| α, normalization, feature maps, gate | One α for each layer, `row`, one map and one gate for each layer | The same |
| Precision | bf16 | The same |
| Stage 1 recipe and data | `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b` (LoLCATs recipe), Alpaca | The same |
| Stage 2 | Run 2 | Not in this experiment |

The window branch applies RoPE to its queries and keys with the cos and sin of the teacher (`window_inputs` in `lizard_attention_v2.py`). When the model generates text, the cache holds the keys after RoPE ([document 13](../13-lizard-attention-v2.md), P5).

## RoPE in the window and the linearization

RoPE is only in the window branch. The gated branch has no RoPE.

- `window_inputs` applies RoPE to the queries and keys of the window branch, and only when `window_rope` is true.
- `feature_maps` gets the queries and keys without RoPE. Thus the kernel $\phi(q_i)^\top \phi(k_j)$ of the gated branch depends only on the content of the tokens.
- When the model generates text, the cache holds the window keys after RoPE. The state of the gated branch holds $\phi(k_j)$ without RoPE.

### The window branch

1. **The window branch is not a linear attention.** It is a softmax attention over the last 128 keys and the 4 sinks. The linearization applies only to the gated branch. Thus RoPE in the window does not change the cost or the memory. Each key gets its rotation one time, at its own position, when it goes into the cache. A RoPE score depends only on $i - j$, so the rotated keys in the cache stay correct ([document 13](../13-lizard-attention-v2.md), P5).
2. **Inside the window, the scores are the scores of the teacher.** With RoPE, the window score is $R(q_i)^\top R(k_j) / \sqrt{d}$, as in the teacher. Thus for the keys inside the window, the window weights are proportional to the weights of the teacher. Only the denominator is different: it is the sum over the window and the sinks, not over all earlier keys. For queries at positions 0–127, all keys are inside the window. There, a window with RoPE gives the teacher output up to rounding (1.3e-5 in bf16, [XAI: branch decomposition](xai-branch-fit.md), finding 1).
3. **Without RoPE, the window knows which tokens are inside it, but not their order.** The mask gives the set of keys. The score $q_i^\top k_j$ gives no position. Thus a window without RoPE cannot give the previous-token attention of layer 0 ([document 15](../15-attention-math-side-by-side.md), claim F). Heads 14 and 23 of layer 15 are window heads. In Run 1, their window without RoPE was the largest single source of the stage 1 loss ([XAI: branch decomposition](xai-branch-fit.md), finding 6).
4. **The two branches get separate tasks.** The window gives the local attention of the teacher, with the positions. The gated branch must approximate only the part of the teacher output from keys outside the window. Thus the error of the linearization moves to the long range. One part of the error stays at short positions: with `gla_norm: row`, the gated branch keeps the weight 1 there ("Results", findings 6 and 7).

### The gated branch

- The only position information of the gated branch is the decay $\prod \gamma$. This decay prefers recent tokens and decreases with the distance. It is not a code of the relative position, as RoPE is. For example, it cannot select the token three positions back.
- In `window_rope`, 99.9–100.0% of the tokens in layers 1–15 have γ > 0.999 ("Results", Lizard parameters). If γ > 0.999 for all tokens, the decay is above $0.999^{128} \approx 0.88$ over 128 tokens and above $0.999^{2048} \approx 0.13$ over 2048 tokens. Thus in these layers, the gated branch is almost a mean of the earlier tokens without positions, with weights from the feature similarity.
- Layer 0 is different. Its gate keeps 0.87 of the weight after 128 tokens.

### The methods side by side

| Method | Window branch | Linear or gated branch | Source |
|---|---|---|---|
| LoLCATs | RoPE | RoPE. The feature maps get the queries and keys after RoPE. | [Document 15](../15-attention-math-side-by-side.md), `process_qkv` in `linear_attention.py` |
| Lizard paper | No RoPE | No RoPE. The positions come only from the gate. | [Document 15](../15-attention-math-side-by-side.md), [document 13](../13-lizard-attention-v2.md) |
| Liger | Paper: no RoPE. Code: RoPE. | No RoPE | [Document 15](../15-attention-math-side-by-side.md) |
| `window_rope` | RoPE | No RoPE. The positions come only from the gate. | `lizard_attention_v2.py` |

### RoPE in the gated branch: an option for later

- The gated branch can get RoPE as in LoLCATs: $\phi(R(q_i))^\top \phi(R(k_j))$. The form stays linear and recurrent, because each key gets its rotation at its own position before it goes into the state.
- The Hedgehog map $\phi$ is not linear. Thus this kernel is not a function of $i - j$ only. The feature maps see the absolute angles of the rotation, and RoPE loses its relative-position property.
- Positions after the training length give combinations of angles that the feature maps did not see in training. Thus the results on longer inputs can become worse.
- The code does not have this option. It would be a new experiment with one change (for example `gla_rope: true`), with the recipe of `window_rope`. It is an open question whether this option improves the long range. Outside the window, the scores of the teacher depend on the positions. Now the gated branch must approximate them with features without positions.

## Predictions

[XAI: branch decomposition](xai-branch-fit.md) gives reference values for Run 1. The table converts the relative errors of each layer into a stage 1 loss. The conversion uses the mean square of the teacher output of each layer ([`fd128_lolcats.json`](xai-layer-wise-mse/fd128_lolcats.json)). The conversion of the trained output gives 3.252, against the stored loss 3.2549.

| Fit of the Run 1 branches (positions 0–2047) | Estimated stage 1 loss |
|---|---|
| The trained output of Run 1 | 3.25 |
| G+W: the best weights of the two trained branches | 2.96 |
| WR: a window with RoPE alone, without the gated branch | 2.43 |
| G+WR: the trained gated branch and a window with RoPE, with the best weights | 1.20 |

1. **Stage 1 loss:** clearly below 3.25, and probably below 2.96. A loss below 2.43 means that the trained gated branch adds to a window with RoPE.
2. **Not down to 1.20:** with `gla_norm: row`, the gated branch keeps the weight 1. But its best weight in Run 1 was 0.71–0.92 ([XAI: branch decomposition](xai-branch-fit.md), finding 7). Thus the loss probably stays above 1.20.
3. **Short positions:** the relative MSE of positions 0–127 decreases clearly from 0.463 (Run 1, [XAI: MSE by token position](xai-position-mse.md)). It probably stays above 0, for the reason of prediction 2.
4. **Layers:** layers 0 and 15, and heads 14 and 23 of layer 15, improve most.
5. **α of layer 0:** above 0.042 (Run 1). A window with RoPE can give the local attention of layer 0, as the window of LoLCATs does ([LoLCATs control](lolcats-control.md), finding 8).
6. **Accuracy:** PIQA and ARC-Easy increase. A gain needs at least 3.2 points over the PIQA of Run 1 (57.6), which is approximately 2 unpaired SE. The ARC-Easy of Run 1 stage 1 is not measured yet. The evaluation below measures it.

## How to run

### 1. A new Docker image

The job runs the code inside the image, and the image does not have the new config yet ([document 3](../03-infrastructure.md)). After the merge, select Actions → "Docker image" → Run workflow on GitHub.

### 2. Stage 1 on HF Jobs (H200)

```bash
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=2h \
  ARGS="--model_config distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope --no_finetune"
```

- **Recipe:** the job runs `make lizard`. Its default distill config is the LoLCATs recipe of Run 1. `--model_config` replaces the model config of the target.
- **Time and cost:** approximately 1 hour and $5. This is an estimate: stage 1 of Run 1 (bf16) took approximately 50 minutes ([document 2](../02-compute-and-cost.md)).
- **First check during the run:** `hf jobs logs <job id> | grep -a "Eval step"`. The loss must decrease. Run 1 ended at 3.2549.
- **Checkpoint on the Hub:**

```text
checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
```

### 3. Evaluation and XAI on `student06` (A10)

Set the GPU variables in the shell that runs the commands, for example inside the `tmux` session. The check must print one A10.

```bash
cd /system/user/khashan/lolcats && git pull && conda activate lolcats-env
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"  # 1 NVIDIA A10
MC=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope
DC=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b
CACHE=$HOME/.cache/huggingface/hub
OUT=results/window_rope && mkdir -p $OUT

# 1. Benchmarks of this model, and ARC-Easy of Run 1 stage 1 (the missing reference)
MODELS=stage1 TASKS="mmlu_subset piqa arc_easy" MODEL_CONFIG=$MC OUT_DIR=$OUT/stages-window-rope scripts/compare_stages.sh
MODELS=stage1 TASKS=arc_easy OUT_DIR=$OUT/stages-run1-arc scripts/compare_stages.sh

# 2. XAI: layer-wise MSE, MSE by token position, branch decomposition
python scripts/layer_mse.py compute $OUT/layer_mse.json --model_config $MC --distill_config $DC \
  --label "v2, window_rope, fd128, LoLCATs recipe, bf16" --cache_dir $CACHE
python scripts/position_mse.py compute $OUT/position_mse.json --from_json $OUT/layer_mse.json --cache_dir $CACHE
python scripts/branch_fit.py compute $OUT/branch_fit.json --from_json $OUT/layer_mse.json --cache_dir $CACHE

# 3. Plots against Run 1 stage 1
R1=docs/experiments
python scripts/layer_mse.py plot $OUT/layer_mse_vs_run1 $R1/xai-layer-wise-mse/fd128_lolcats.json $OUT/layer_mse.json
python scripts/layer_mse.py plot $OUT/layer_mse_vs_run1_relative $R1/xai-layer-wise-mse/fd128_lolcats.json $OUT/layer_mse.json --relative
python scripts/position_mse.py heatmap $OUT/position_heatmap $R1/xai-position-mse/fd128_lolcats.json $OUT/position_mse.json
for bucket in 0-127 all; do
  python scripts/branch_fit.py plot $OUT/branch_fit_$bucket $R1/xai-branch-fit/fd128_lolcats.json $OUT/branch_fit.json --bucket $bucket
done

zip -r window-rope-$(date +%Y%m%d-%H%M).zip $OUT
```

- `compare_stages.sh` finds the checkpoint from `MODEL_CONFIG` and the default recipe. It downloads a missing checkpoint from `HF_REPO`.
- In the branch decomposition of this model, W is the window branch with RoPE and the sinks. Thus W at positions 0–127 must be close to WR.

## Tests

CPU, 2026-10-09.

| Check | Result |
|---|---|
| The config against Run 1 | The model section is equal. In the attention section, only `attention_type` (v2) and `window_rope` differ. The v2 options have the same keys as the C1 config. |
| `pytest tests/test_lizard_attention_v2.py` | 60 passed. They include `window_rope` against the loop reference and the decode form, and the v2 defaults against v1. |
| A tiny copy of the config at initialization (3 layers, window 16, `branch_fit.py --checkpoint init`) | At positions 0–15, W has an error of 0.089 with `window_rope`, and 0.414 without. At positions 16–31, W gives 0.371 and WR 0.373. Thus the window branch uses RoPE. The rest of 0.089 comes from the 4 sinks at their start value. |
| Training of the tiny copy: the real stage 1 recipe, sequences of 1024 tokens, 8 steps | The validation loss decreased at each evaluation: 8536, 8148, 7805. The checkpoint has 15 Lizard tensors (3 layers × 5). It loads with 0 missing and 0 unexpected keys, and `layer_mse.py` reproduces its stored loss. After the training, W at positions 0–15 has an error of 0.094. |

**Two notes from the tests:**

- After the training, `distill_llama.py` stopped with an error in the sample generation. The tiny word-level tokenizer gives `token_type_ids`, and `generate` does not accept them. The earlier CPU tests had the same error. The tokenizer of Llama does not give them.
- The trainer evaluates one batch more than `--max_eval_batches` (`default_lm.py`). Thus the stored loss of the test needed 3 batches in `layer_mse.py`. Real runs evaluate all batches, so this has no effect on them.

## Results

### Run 1: stage 1 and evaluation (2026-10-10)

- **Training:** HF Jobs with `HF_FLAVOR=a100-large` and `HF_TIMEOUT=6h`. The job trained at 1.24 sequences each second for 2 epochs.
- **Evaluation:** `student06`, one A10, lolcats `02b4ce9`. The commands are the block of "How to run", section 3. The XAI scripts used all 16 validation batches (32,768 tokens).
- **Checks:**
  1. In each evaluation, all 80 expected Lizard tensors loaded. 0 missing, 0 unexpected.
  2. `layer_mse.py` gives the loss 1.2310, against the stored loss 1.2290 (+0.16%). The earlier bf16 checkpoints gave −0.11% to −0.22%.
  3. The position buckets add up to the MSE over all positions, with a relative difference of 3.6e-8.
  4. The branches reproduce the checkpoint output with a relative error of 1.7e-3. This is the bf16 rounding, as for the bf16 checkpoints of X2.
- **Files in [`window-rope/`](window-rope/):** the summaries of the two evaluations ([this model](window-rope/stages_window_rope.md), [ARC-Easy of Run 1](window-rope/stages_run1_arc.md)) and their lm-eval CSV files. The folder also has the plots with their tables. The JSON files are `v2_window_rope.json` in the folders of the three XAI documents.

#### Validation loss during stage 1

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 1000 | 1100 |
|---|---|---|---|---|---|---|---|---|
| Loss | 2.0879 | 1.7700 | 1.6123 | 1.5103 | 1.4014 | 1.3315 | 1.2480 | **1.2290** |

The values come from the job log. This document does not have the values of steps 700–900. The first epoch ends near step 589. Before that step, the loss decreased by 0.07–0.32 for each 100 steps. In the second epoch, it decreased by approximately 0.02 for each 100 steps. The checkpoint holds step 1100, the last evaluation.

#### Scores

z values with unpaired SEs. The ARC-Easy of Run 1 comes from this evaluation. The other values of Run 1 come from [stage difference](stage-difference.md), and the values of LoLCATs from [LoLCATs control](lolcats-control.md).

| Measure | Lizard Run 1, stage 1 | `window_rope`, stage 1 | Difference | LoLCATs attention, stage 1 |
|---|---|---|---|---|
| Stage 1 validation loss | 3.2549 | **1.2290** | −62% | 0.4442 |
| PIQA | 57.6 ± 1.2 | **61.5 ± 1.1** | +3.9 points, z ≈ 2.4 | 73.5 |
| PIQA, normalized | 55.5 ± 1.2 | 58.8 ± 1.1 | +3.3 points, z ≈ 2.0 | – |
| ARC-Easy | 39.0 ± 1.0 | **43.3 ± 1.0** | +4.3 points, z ≈ 3.0 | 62.8 |
| ARC-Easy, normalized | 37.8 ± 1.0 | 41.0 ± 1.0 | +3.2 points, z ≈ 2.3 | 58.0 |
| MMLU subset | 22.5 ± 2.5 | 24.6 ± 2.5 | +2.1 points, z ≈ 0.6 | 26.0 |
| "A" / "B" / "C" / "D" on MMLU | 95.4% / 2.8% / 1.8% / 0.0% | 40.0% / 20.4% / 16.5% / 23.2% | – | 19.3% / 14.7% / 42.8% / 23.2% |
| Letter mass, confidence, entropy | 0.018, 0.586, 1.550 bits | 0.498, 0.743, 0.963 bits | – | 0.962, 0.450, 1.746 bits |

#### Lizard parameters

From the gate tables of the two evaluation summaries. The tables use one 5-shot MMLU prompt of 2048 tokens.

| Measure | Lizard Run 1 | `window_rope` |
|---|---|---|
| α, layers 0–15 | 0.042–0.648 | 0.293–0.719 |
| α, layer 0 | 0.042 | 0.660 |
| Sink logit, layer 0 | 0.73 | 4.25 |
| Gate, layers 1–15: tokens with γ > 0.999 | 100.0% | 99.9–100.0% |
| Gate, layer 0: weight kept after 128 tokens | 0.35 | 0.87 |

#### Plots

![Layer-wise MSE of window_rope against Lizard Run 1](window-rope/layer_mse_vs_run1.png)

![Layer-wise relative MSE of window_rope against Lizard Run 1](window-rope/layer_mse_vs_run1_relative.png)

![Relative MSE of each layer and position bucket, Lizard Run 1 and window_rope](window-rope/position_heatmap.png)

![Branch decomposition at positions 0–127, Lizard Run 1 and window_rope](window-rope/branch_fit_0-127.png)

![Branch decomposition over all positions, Lizard Run 1 and window_rope](window-rope/branch_fit_all.png)

#### Means over the layers

Relative MSE: the MSE divided by the mean square of the teacher output. Branch fits: the remaining error divided by ‖y‖², with the best weights for each head ([XAI: branch decomposition](xai-branch-fit.md)). "Best weight of G" is the median over the layers of the median weight of the heads, in the G+W fit.

| Measure | Lizard Run 1 | `window_rope` | LoLCATs attention |
|---|---|---|---|
| Relative MSE, positions 0–127 | 0.463 | 0.321 | 0.0000151 |
| Relative MSE, positions 128–511 | 0.407 | 0.168 | 0.035 |
| Relative MSE, positions 512–2047 | 0.341 | 0.141 | 0.067 |
| W alone, positions 0–127 | 0.403 | 0.048 | – |
| G+W, positions 0–127 | 0.314 | 0.025 | – |
| Trained output, positions 0–127 | 0.463 | 0.321 | – |
| G+W, all positions | 0.328 | 0.116 | – |
| Trained output, all positions | 0.363 | 0.159 | – |
| Best weight of G: 0–127 / 128–511 / 512–2047 / all | 0.42 / 0.76 / 0.84 / 0.81 | 0.08 / 0.65 / 0.79 / 0.75 | – |

#### Findings

**Finding 1: RoPE in the window is a real gain on PIQA and ARC-Easy**. PIQA increases by 3.9 points (z ≈ 2.4), and ARC-Easy by 4.3 points (z ≈ 3.0). Both pass the rule of 2 SE. These are the highest PIQA and ARC-Easy scores of all Lizard stage 1 checkpoints. The earlier checkpoints have PIQA 55.8–57.7 and ARC-Easy 29.9–39.0 ([XAI: MSE by token position](xai-position-mse.md), accuracy table).

**Finding 2: the loss closes most of the gap to LoLCATs, but the accuracy closes only a small part**. Between Run 1 and LoLCATs, the stage 1 loss closes 72% of the gap. PIQA closes 25% of its gap, and ARC-Easy 18%.

**Finding 3: the "A" collapse on MMLU is mostly gone, but the accuracy stays at chance**. The share of "A" falls from 95.4% to 40.0%, and the letter mass increases from 0.018 to 0.498. The accuracy is 24.6%. An answer of "A" for every question gives 24.2%.

**Finding 4: the MSE decreases in every layer, most of all in layers 15 and 0**. 1000 × MSE is 1.5–5.3× lower than in Run 1. The largest factors are in layer 15 (from 16.19 to 3.03, 5.3×) and layer 0 (3.7×). Heads 14 and 23 of layer 15 have 13.7× and 11.3× less MSE. In Run 1, they gave 21% of the loss. Now they give 4.4%. Against LoLCATs, the MSE is still 1.7× (layer 5) to 10× (layer 0) higher.

**Finding 5: the window branch now gives almost the teacher output at short positions**. At positions 0–127, W alone keeps a mean error of 0.048 with the best weight for each head (Run 1: 0.403). Layers 2–15 have 0.007–0.038, and layers 0 and 1 have 0.117 and 0.293. The oracle WR gives 1.3e-5. W and WR use the same queries and keys, and only W has the 4 sinks. Thus the sinks cause the remaining error of W. In layer 0, the sink logit increased from 0.73 to 4.25.

**Finding 6: at short positions, the mix of the branches gives most of the error**. At positions 0–127, the trained output has a mean error of 0.321. The best weights of the same two branches give 0.025. Thus the trained mix has 13× the error of the best mix, and 3–89× in the single layers. In Run 1, this factor was 1.1–1.9. At these positions, the median best weight of G is 0.08 (0.03–0.47 in the layers). But with `gla_norm: row`, G keeps the weight 1. The best weight of W is 0.98–1.6, but α is 0.29–0.72.

**Finding 7: the best weight of G depends on the position**. Its median is 0.08 at positions 0–127, 0.65 at positions 128–511 and 0.79 at positions 512–2047. One α for each layer and the constant weight 1 of G cannot fit all three cases. In the stage 1 loss, 94% of the positions are above 127. Thus the training sets the mix for long contexts. PIQA and ARC-Easy prompts are short, so they get a mix that does not fit them. This probably explains finding 2.

**Finding 8: short positions now have the largest error**. At positions 0–127, the relative MSE decreases by 31% (from 0.463 to 0.321). At the other two buckets, it decreases by 59%. Thus the error at positions 0–127 is now 1.9× and 2.3× the error of the other two buckets. Layers 2 and 3 have the largest error there (0.593 and 0.538), only 16–18% less than in Run 1. In these two layers, the trained mix has 89× and 22× the error of the best mix.

**Finding 9: over all positions, the best mix would lower the loss by approximately 28%**. With the best weights of G and W, the estimated stage 1 loss is 0.89, against 1.23. The conversion is the conversion of "Predictions". The trained output of this model gives 1.231 with the same conversion.

#### Check of the predictions

| Prediction | Result | Holds |
|---|---|---|
| 1. Stage 1 loss clearly below 3.25, probably below 2.96 | 1.2290. It is also below 2.43, so the trained gated branch adds to the window with RoPE. | Yes |
| 2. Not down to 1.20, because the gated branch keeps the weight 1 | 1.2290, 2% above 1.20. The best weight of G is below 1 (finding 7). | Yes |
| 3. The relative MSE of positions 0–127 decreases clearly from 0.463, but stays above 0 | 0.321 (−31%), the smallest decrease of the three buckets (finding 8) | Yes |
| 4. Layers 0 and 15, and heads 14 and 23 of layer 15, improve most | Layer 15: 5.3×, layer 0: 3.7×, the two largest factors. Heads 14 and 23: 13.7× and 11.3× (finding 4). | Yes |
| 5. α of layer 0 above 0.042 | 0.660 | Yes |
| 6. PIQA and ARC-Easy increase, PIQA by at least 3.2 points | PIQA +3.9, ARC-Easy +4.3 (finding 1) | Yes |

#### Decision

- **Rule 1 applies.** PIQA and ARC-Easy increase by more than 2 SE. Thus RoPE in the window is a real gain.
- **Next: `window_rope` with `gla_norm: hybrid`** ([experiment](window-rope-hybrid.md)), as the rule says. Findings 6 and 7 support this run. With the shared denominator, the share of G can change with the position. The sum of the gated weights grows with all keys, but the window sum stops at 128 keys. It is not known whether the training sets a small share of G at short positions. In R1b, `hybrid` turned the window almost off (α 0.01–0.07, from the start value 0.1).
- **If the error at positions 0–127 stays high:** step 3 of [document 20](../20-layer-mse-for-piqa-arc.md), section 5. The gated branch then gets only the keys outside the window. With `hybrid`, a prompt shorter than the window then gets only the window and the sinks.
- **Stage 2** waits for the better of these configs.
- **Thesis:** report this run as an extension, not as the reproduction.

### Run 2: stage 2 (2026-10-10)

- **Training:** `make hf-job-finetune`, from the stage 1 checkpoint of Run 1. This is the first run of this target with a v2 config ([`window_rope` with the shared denominator](window-rope-hybrid.md), "Decision").
- **What stage 2 trains:** only the LoRA weights of q, k, v and o (128 tensors, 1,703,936 parameters). The Lizard parameters keep their stage 1 values. α and the sink logits are equal in the two evaluation summaries. The gates differ a little, because LoRA changes the hidden states.
- **Evaluation:** `student06`, one A10, lolcats `84228c5`, `MODELS=stage2`.
- **Checks:** in each evaluation, all 80 stage 1 tensors and all 128 stage 2 tensors loaded. 0 missing, 0 unexpected. The stored step of the stage 2 checkpoint (1100) is the best step.
- **Files in [`window-rope/`](window-rope/):** the [evaluation summary](window-rope/stages_stage2.md), its lm-eval CSV file, and the validation losses of stage 2 (`stage2_training.csv`).

#### Validation loss during stage 2

The causal language-model loss on the Alpaca validation data.

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `window_rope` | 1.3432 | 1.2961 | 1.2792 | 1.2661 | 1.2575 | 1.2533 | 1.2482 | 1.2433 | 1.2413 | 1.2372 | **1.2352** |

Stage 2 of Lizard Run 1 (Run 2) started at 3.516 and ended at 2.252 ([stage difference](stage-difference.md)). Thus the final perplexity, exp(loss), is approximately 3.44 here, against approximately 9.5 for Run 2.

#### Scores

All values come from the same harness. z values with unpaired SEs. Run 2 is the model of [document 7](../07-results.md): the stage 1 checkpoint of Lizard Run 1 and the stage 2 checkpoint of Run 2.

| Measure | Run 2, after stage 2 | `window_rope`, stage 1 | `window_rope`, after stage 2 | Difference to Run 2 | LoLCATs attention, stage 1 |
|---|---|---|---|---|---|
| PIQA | 67.95 ± 1.09 | 61.5 ± 1.1 | **73.1 ± 1.0** | +5.1 points, z ≈ 3.4 | 73.5 |
| PIQA, normalized | 66.6 | 58.8 | 72.5 | +5.9 points | – |
| ARC-Easy | 54.8 ± 1.0 | 43.3 ± 1.0 | **63.8 ± 1.0** | +9.0 points, z ≈ 6.3 | 62.8 |
| ARC-Easy, normalized | 50.1 | 41.0 | 56.9 | +6.9 points | 58.0 |
| MMLU subset | 24.9 ± 2.6 | 24.6 ± 2.5 | 25.6 ± 2.6 | +0.7 points | 26.0 |
| MMLU, all questions (2026-10-11) | 23.3 | – | **27.2 ± 0.4** | +3.9 points | – |
| "A" / "B" / "C" / "D" on MMLU | 98.6% / 1.1% / 0.4% / 0.0% | 40.0% / 20.4% / 16.5% / 23.2% | 41.4% / 37.5% / 20.7% / 0.4% | – | 19.3% / 14.7% / 42.8% / 23.2% |
| Letter mass, confidence, entropy | – | 0.498, 0.743, 0.963 bits | 0.980, 0.348, 1.908 bits | – | 0.962, 0.450, 1.746 bits |

In this harness, the teacher gets 33.7 on the MMLU subset ([document 7](../07-results.md)). It gets 74.4 ± 1.0 on PIQA and 65.3 ± 1.0 on ARC-Easy (X0, 2026-10-10, [summary](../teacher-x0/summary.md)).

Values of the papers, from another version of the harness ([document 7](../07-results.md), Table 9 of the Lizard paper):

| Task | Lizard 1B | LoLCATs 1B | Teacher 1B |
|---|---|---|---|
| PIQA | 74.8 | 74.6 | 74.1 |
| ARC-Easy | 65.6 | 63.0 | 65.4 |

#### Findings of Run 2

**Finding 10: after stage 2, `window_rope` is the best Lizard model of this project**. PIQA is 73.1 and ARC-Easy is 63.8. Against Run 2, PIQA increases by 5.1 points (z ≈ 3.4), and ARC-Easy by 9.0 points (z ≈ 6.3).

**Finding 11: the gain of stage 1 stays after stage 2, and it increases**. After stage 1, `window_rope` had +3.9 points on PIQA and +4.3 points on ARC-Easy against Lizard Run 1 (finding 1). After stage 2, the differences are +5.1 and +9.0. Stage 2 adds 11.6 points on PIQA and 20.5 points on ARC-Easy. For Lizard Run 1, stage 2 added 10.4 and 15.8 points.

**Finding 12: PIQA and ARC-Easy come near the values of the papers, but these values come from another harness**. PIQA is 1.7 points below Lizard 1B of the paper (74.8), and ARC-Easy is 1.8 points below (65.6). ARC-Easy is above LoLCATs 1B of the paper (63.0). The paper used a newer version of the harness. Thus only the teacher in this harness can give the real gap. In this harness, the LoLCATs attention after stage 1 has 73.5 and 62.8.

**X0 (2026-10-10):** in this harness, the teacher has 74.4 on PIQA and 65.3 on ARC-Easy. These values are within 0.3 points of the teacher of the paper (74.1 and 65.4). Thus `window_rope` gets 98.2% and 97.7% of the teacher. The differences are −1.4 points (z ≈ −0.9) and −1.5 points (z ≈ −1.1), with unpaired SEs. Run 2 got 91.3% and 83.9%.

**Finding 13: the stage 2 loss is much lower than in Run 2**. The final validation loss is 1.235, against 2.252 for Run 2. The loss still decreased at the last evaluation (by 0.002 after step 1000).

**Finding 14: MMLU stays at chance**. The accuracy is 25.6%, and the teacher has 33.7%. The "A" collapse of Run 2 (98.6% "A") is gone, and the four letters get 98.0% of the probability. But the probability is spread almost evenly over "A", "B" and "C" (confidence 0.348, entropy 1.908 bits), and "D" gets 0.4% of the answers. Thus the model gives the answer format, but it does not find the right answer. **Update (2026-10-11):** on all questions, the accuracy is 27.2, above chance (finding 15). The teacher has the same even spread over the letters (finding 20). The 5-shot MMLU prompts are long (up to 2048 tokens). Thus MMLU probably depends more on the part of the attention outside the window than PIQA and ARC-Easy do. This is a hypothesis.

#### Decision of Run 2

- **Report:** `window_rope` after stage 2 is the best Lizard model of this project. It is an extension, not the reproduction ([document 19](../19-next-steps-from-literature.md), part B).
- **Done on 2026-10-10:** the teacher on PIQA and ARC-Easy in this harness (X0 of [gap analysis 2](../12-gap-analysis-2.md)), with `MODELS=teacher TASKS="piqa arc_easy" scripts/compare_stages.sh`. It gives 74.4 and 65.3 (finding 12).
- **The remaining gap is MMLU** (27.2 against 31.7 for the teacher on all questions, findings 15–18). Changes to the part outside the window, such as the gated branch and step 3 of [document 20](../20-layer-mse-for-piqa-arc.md), target this gap. Their effect must be checked on MMLU, not only on PIQA and ARC-Easy.

### MMLU, all questions (2026-10-11)

- **Run:** `MODELS="teacher stage2" TASKS=mmlu scripts/compare_stages.sh` on `student06` (one A10, lolcats `c132d29`). 14,042 questions, 5-shot. The teacher took 2 hours 30 minutes, and the stage 2 model 4 hours 31 minutes.
- **Checks:** all 80 stage 1 tensors and all 128 stage 2 tensors loaded. 0 missing, 0 unexpected.
- **Files in [`window-rope/`](window-rope/):** the [summary](window-rope/mmlu_full_summary.md) and its lm-eval CSV file. The write-out files of each question (approximately 175 MB for each model) are not in the repository.
- **Analysis:** the values below the score table come from a one-off analysis of the write-out files, on a CPU. The script `mmlu_items.py` of [document 22](../22-mmlu-xai-methods.md) can make it repeatable.

The harness score is the mean over the 57 subjects. The mean over all questions is also given.

| Measure | Teacher | `window_rope` after stage 2 | Run 2 ([document 7](../07-results.md)) |
|---|---|---|---|
| MMLU, mean over subjects | **31.7 ± 0.4** | **27.2 ± 0.4** | 23.3 |
| MMLU, mean over questions | 31.0 | 26.9 | – |
| "A" / "B" / "C" / "D" | 49.6% / 26.1% / 21.8% / 2.6% | 42.3% / 38.0% / 18.2% / 1.4% | – |
| Letter mass, confidence, entropy | 0.999, 0.342, 1.904 bits | 0.975, 0.354, 1.900 bits | – |

The right answers are "A" for 22.9%, "B" for 24.7%, "C" for 25.5% and "D" for 26.9% of the questions. The Lizard paper gives 31.0 for the teacher, 29.8 for Lizard and 27.3 for LoLCATs.

**Finding 15: on all questions, `window_rope` is above chance**. It gets 27.2, which is approximately 5.9 SE above 25%. Run 2 was 4.8 SE below chance. Thus the subset (25.6) did not show the gain. The value is near LoLCATs in the Lizard paper (27.3), and 2.6 points below Lizard in the paper.

**Finding 16: the gap to the teacher is clear, and it is a third of the knowledge**. On the same questions, the teacher has 4.2 more points (paired SE 0.44, z ≈ 9.5). The teacher alone is right on 2,210 questions, and the student alone on 1,625 (McNemar z ≈ 9.5). The student gets 86% of the teacher score. But it keeps only 33% of the part of the teacher above chance ((27.2 − 25) / (31.7 − 25)).

**Finding 17: the student follows the teacher**. When the teacher is right, the student is right on 49.3% of the questions. When the teacher is wrong, the student is right on only 16.8%. Without a link between the models, both values would be 26.9%. The two models give the same letter on 49.2% of the questions (34.9% by chance, kappa 0.22). The probability of the right letter has a correlation of 0.60 between the models.

**Finding 18: the student loses most where the teacher knows most**.

| Subject group | Questions | Teacher | `window_rope` | Gap | Part of the teacher above chance that the student keeps |
|---|---|---|---|---|---|
| STEM | 3,153 | 28.3 | 25.4 | 2.8 | 12% |
| Humanities | 4,705 | 28.8 | 26.1 | 2.7 | 29% |
| Social sciences | 3,077 | 32.1 | 27.5 | 4.7 | 35% |
| Other | 3,107 | 36.2 | 28.9 | 7.2 | 35% |

The values are means over the questions of each group. Computer security has the largest gap of a single subject (49.0 against 30.0). The next largest are high school European history, high school statistics, security studies, high school geography and world religions.

**Finding 19: the gap does not increase with the length of the prompt**. In the four quarters of the prompt length (in characters), the gap is 5.9, 3.8, 2.4 and 4.5 points. The shortest prompts have the largest gap. The length and the subject are not independent. But this result does not support the hypothesis that the long 5-shot context causes the gap ([document 21](../21-mmlu-options.md), section 2).

**Finding 20: the letter statistics do not separate the two models**. The teacher and the student have almost the same confidence and entropy. Both prefer "A" and almost never give "D". Thus finding 14 is wrong in one point. The even spread over the letters is a property of the 1B teacher in this format, not a sign of damage in the student. The gap is in the choice of the right letter (findings 16–18).

**Finding 21: bf16 ties affect both models**. In both models, the letter log-likelihoods are multiples of 0.125 or 0.25 (bf16 logits). 24.2% of the teacher answers and 20.5% of the student answers have a tie at the maximum. In a tie, the first letter wins. With a random choice in each tie, the mean over questions is 31.7 for the teacher (stored 31.0) and 27.0 for the student (stored 26.9). The letter shares are then 40.0% / 28.6% / 27.8% / 3.5% (teacher) and 35.5% / 38.7% / 23.7% / 2.1% (student). Thus the ties add to the preference for "A", but the rare "D" does not come from the ties. [Document 22](../22-mmlu-xai-methods.md), A6, gives the details.

### Table 9 tasks: ARC-Challenge, HellaSwag and WinoGrande (2026-10-11 and 2026-10-12)

- **Run:** `MODELS=stage2 TASKS="arc_challenge hellaswag winogrande" MODEL_CONFIG=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope OUT_DIR=results/table9_stage2 scripts/compare_stages.sh` on `student06` (one A10, lolcats `ba003ed`, harness `b281b09`). All three tasks are 0-shot.
- **Run history:** the first run gave ARC-Challenge. Then the home folder became full (quota 20 GB) during the download of HellaSwag, and the run stopped. The command `rm -rf ~/.cache/pip` freed 4.8 GB. Then a second run with `TASKS="hellaswag winogrande"` gave HellaSwag. The output that these notes have ends after HellaSwag. Thus WinoGrande had no result. A third run with `TASKS=winogrande` gave WinoGrande on 2026-10-12.
- **Checks:** in all three runs, all 80 stage 1 tensors and all 128 stage 2 tensors loaded. 0 missing, 0 unexpected. The stored step of both checkpoints is 1100, with the losses 1.2290 and 1.2352.
- **Files:** none in the repository. The zip of the run directory failed, because the disk was full. The values come from the console output of the harness.
- **Time:** ARC-Challenge, 4,687 requests in 4 minutes 23 seconds. HellaSwag, 40,145 requests in 47 minutes. WinoGrande, 2,534 requests in 2 minutes 42 seconds.
- **Summary step of the WinoGrande run:** the harness finished, but `compare_stages.py summary` then stopped with `ValueError: could not convert string to float: 'False'`. WinoGrande writes the per-question `acc` as the string `'True'` or `'False'`. `read_run` now reads these strings (`as_float`). The score comes from the harness output: `{'acc': 0.595895816890292, 'acc_stderr': 0.013791610664670858}`.

Accuracy in %, ± the binomial SE. The paper reports the normalized accuracy for ARC-Challenge and HellaSwag, and the accuracy for WinoGrande. The paper values come from Table 9 of the Lizard paper and from another version of the harness.

| Task | Questions | `window_rope`, accuracy | `window_rope`, normalized accuracy | Teacher, this harness, accuracy | Teacher, this harness, normalized accuracy | Paper: teacher 1B | Paper: Lizard 1B | Paper: LoLCATs 1B |
|---|---|---|---|---|---|---|---|---|
| ARC-Challenge | 1,172 | 33.2 ± 1.4 | **36.1 ± 1.4** | 31.4 ± 1.4 | **36.3 ± 1.4** | 36.4 | 36.5 | 35.1 |
| HellaSwag | 10,042 | 47.1 ± 0.5 | **60.7 ± 0.5** | 47.8 ± 0.5 | **63.7 ± 0.5** | 63.8 | 64.1 | 63.7 |
| WinoGrande | 1,267 | **59.6 ± 1.4** | – | **60.1 ± 1.4** | – | 60.0 | 59.7 | 61.5 |

The teacher ran on 2026-10-12 ("Teacher on the Table 9 tasks" below).

**Finding 22: on ARC-Challenge, `window_rope` is at the level of the paper**. The normalized accuracy is 36.1. It is 0.3 points below the teacher of the paper (36.4) and 0.4 points below Lizard of the paper (36.5). The SE is 1.4 points, so the difference is not clear. On PIQA and ARC-Easy, this harness gives the teacher within 0.3 points of the paper (finding 12).

**Finding 23: on HellaSwag, `window_rope` is 3.1 points below the teacher of the paper**. The normalized accuracy is 60.7 ± 0.5, against 63.8 for the teacher and 64.1 for Lizard in the paper. This gap is larger than on PIQA (1.0 points below the teacher of the paper), ARC-Easy (1.6) and ARC-Challenge (0.3). But the teacher has no HellaSwag result in this harness. If the two harness versions agree on HellaSwag as on PIQA and ARC-Easy, `window_rope` keeps approximately 95% of the teacher on HellaSwag (60.7 / 63.8). The teacher run in this harness gives 95.3% (finding 25).

**Finding 24: on WinoGrande, `window_rope` is at the level of the paper**. The accuracy is 59.6 ± 1.4 on 1,267 questions. The paper gives 60.0 for the teacher, 59.7 for Lizard and 61.5 for LoLCATs. Thus `window_rope` is 0.4, 0.1 and 1.9 points below these values. The SE is 1.4 points, so none of these differences is clear. The teacher in this harness gets 60.1 (finding 25).

### Teacher on the Table 9 tasks (2026-10-12)

- **Run:** `OUT_DIR=results/table9/teacher MODEL_CONFIG=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope MODELS=teacher TASKS="arc_challenge hellaswag winogrande" scripts/compare_stages.sh` on `student06` (one A10, lolcats `daa2bbe`, harness `b281b09`, Python 3.11.16, torch 2.5.1, CUDA 12.4, transformers 4.43.1, peft 0.9.0). All three tasks are 0-shot, batch size 1.
- **Time:** ARC-Challenge, 4,687 requests in 2 minutes 6 seconds. HellaSwag, 40,145 requests in 28 minutes 40 seconds. WinoGrande, 2,534 requests in 1 minute 3 seconds.
- **Summary step:** the harness finished all three tasks. Then `compare_stages.py summary` stopped with `ValueError: could not convert string to float: 'True'`, the same WinoGrande problem as above. `as_float` reads these strings. The scores come from the harness output:
  - ARC-Challenge: `acc` 0.31399 (SE 0.01356), `acc_norm` 0.36348 (SE 0.01406)
  - HellaSwag: `acc` 0.47799 (SE 0.00498), `acc_norm` 0.63683 (SE 0.00480)
  - WinoGrande: `acc` 0.60142, SE 0.01376

**Finding 25: in this harness, the teacher gets the values of the paper, and `window_rope` loses only on HellaSwag**. The teacher gets 36.3, 63.7 and 60.1, against 36.4, 63.8 and 60.0 in the paper. Thus the two harness versions agree within 0.1 points on these three tasks, as on PIQA and ARC-Easy (finding 12). Against the teacher of this harness, `window_rope` gets these differences (unpaired SE):

| Task | `window_rope` | Teacher, this harness | Difference | Share of the teacher |
|---|---|---|---|---|
| ARC-Challenge (normalized) | 36.1 | 36.3 | −0.2 points, z ≈ −0.1 | approximately 99.4% |
| HellaSwag (normalized) | 60.7 | 63.7 | −3.0 points, z ≈ −4.4 | approximately 95.3% |
| WinoGrande | 59.6 | 60.1 | −0.5 points, z ≈ −0.3 | approximately 99.2% |

On ARC-Challenge and WinoGrande, the difference is not clear. On HellaSwag, the gap of 3.0 points is clear (|z| ≥ 2). HellaSwag is the only short task where `window_rope` is clearly below the teacher.

### Paired tests against the teacher (2026-10-12)

- **Command:** `scripts/paired_tests.py` on `student06`, CPU only. It read the per-question files of these run directories:
  - `window_rope`, stage 2: `results/window_rope_stage2/stages/stage2` (PIQA, ARC-Easy) and `results/table9_stage2/stage2` (ARC-Challenge, HellaSwag, WinoGrande)
  - teacher: `results/teacher_x0/teacher` (PIQA, ARC-Easy) and `results/table9/teacher/teacher` (ARC-Challenge, HellaSwag, WinoGrande)
- **Method:** McNemar's exact test on the questions that only one model answers right, and the paired SE. Also a paired bootstrap 95% interval (10,000 samples, seed 0) and the Holm correction over the five tasks.
- **File:** [`window-rope/paired_window_rope_vs_teacher.md`](window-rope/paired_window_rope_vs_teacher.md).

| Task (metric) | n | `window_rope` | Teacher | Difference | Paired SE | z | Only `window_rope` / only teacher | McNemar p | Holm p | 95% bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|
| PIQA (acc) | 1,838 | 73.1 | 74.4 | −1.4 | 0.70 | −1.9 | 71 / 96 | 0.063 | 0.24 | [−2.7, +0.0] |
| ARC-Easy (acc) | 2,376 | 63.8 | 65.3 | −1.5 | 0.78 | −1.9 | 155 / 191 | 0.06 | 0.24 | [−3.0, +0.0] |
| ARC-Challenge (acc_norm) | 1,172 | 36.1 | 36.3 | −0.3 | 1.17 | −0.2 | 92 / 95 | 0.88 | 1 | [−2.6, +2.0] |
| HellaSwag (acc_norm) | 10,042 | 60.7 | 63.7 | −3.0 | 0.31 | −9.7 | 336 / 637 | 3.1e-22 | 1.6e-21 | [−3.6, −2.4] |
| WinoGrande (acc) | 1,267 | 59.6 | 60.1 | −0.6 | 1.47 | −0.4 | 169 / 176 | 0.75 | 1 | [−3.3, +2.4] |

**Finding 26: the paired tests keep the conclusions of the X0 comparison and of finding 25, but PIQA and ARC-Easy come near the limit**. The paired SE is about two thirds of the unpaired SE on PIQA and ARC-Easy, so z changes from about −1 to −1.9. The McNemar p-values are 0.063 and 0.06, and the upper ends of the bootstrap intervals are at +0.0. After the Holm correction, both p-values are 0.24, so the differences are not clear. On ARC-Challenge and WinoGrande, the differences stay small and not clear. On HellaSwag, the gap of 3.0 points is clear (z ≈ −9.7, Holm p 1.6e-21).

## Decision rules

- **PIQA or ARC-Easy increases by 2 SE or more:** RoPE in the window is a real gain. Next: `window_rope` with `gla_norm: hybrid`, which corrects the weight of the gated branch ([XAI: branch decomposition](xai-branch-fit.md), finding 7). Then stage 2 for the better config.
- **The loss decreases, but the accuracy does not:** the MSE does not predict the accuracy ([XAI: MSE by token position](xai-position-mse.md), finding 6). Check the MSE of positions 0–127 and the share of "A" in the MMLU subset. Then try `window_rope` with `gla_norm: hybrid`, because the gated branch keeps the weight 1 on short prompts (prediction 2).
- **The loss stays above 2.96:** check in the branch decomposition that W at positions 0–127 is close to WR. If it is close, the training of the other parts limits the result.
