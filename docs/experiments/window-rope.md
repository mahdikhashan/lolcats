# Experiment: RoPE in the window branch (`window_rope`)

**Status:** A plan from 2026-10-09. The model config exists and passed CPU tests. Stage 1 has not run yet.

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
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=4h \
  ARGS="--model_config distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope --no_finetune"
```

- **Recipe:** the job runs `make lizard`. Its default distill config is the LoLCATs recipe of Run 1. `--model_config` replaces the model config of the target.
- **Time and cost:** approximately 2.5 hours and $12. The first run (2026-10-10) trained at 1.24 sequences each second, thus approximately 63 minutes for each epoch. Stage 1 of Run 1 (v1, bf16) had approximately 3.2 sequences each second ([document 2](../02-compute-and-cost.md)). The cause of the difference is not checked. `HF_TIMEOUT=2h` is too short: the job stops before the end of the second epoch.
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

Not run yet.

## Decision rules

- **PIQA or ARC-Easy increases by 2 SE or more:** RoPE in the window is a real gain. Next: `window_rope` with `gla_norm: hybrid`, which corrects the weight of the gated branch ([XAI: branch decomposition](xai-branch-fit.md), finding 7). Then stage 2 for the better config.
- **The loss decreases, but the accuracy does not:** the MSE does not predict the accuracy ([XAI: MSE by token position](xai-position-mse.md), finding 6). Check the MSE of positions 0–127 and the share of "A" in the MMLU subset. Then try `window_rope` with `gla_norm: hybrid`, because the gated branch keeps the weight 1 on short prompts (prediction 2).
- **The loss stays above 2.96:** check in the branch decomposition that W at positions 0–127 is close to WR. If it is close, the training of the other parts limits the result.
