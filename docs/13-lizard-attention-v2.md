# 13. Lizard attention v2

**Status:** Implemented and tested on CPU (37 checks pass, float64). One stage 1 run finished: C1, one α for each head (validation loss 3.8092, −4.2% against config 1, no accuracy gain). See "Experiment C1".

v2 is a new attention file: `src/model/linear_attention/lizard_attention_v2.py`. It has the layer of v1 (`lizard_attention.py`) and a model config option for each reading of the paper. It also has an option for each code change of [gap analysis 2](12-gap-analysis-2.md) (C1–C6). The default options give the outputs of v1 exactly. Thus each experiment changes one option, as section 5.2 of gap analysis 2 requires. v1 does not change.

## Why v2

[Gap analysis 2](12-gap-analysis-2.md) ranks "a limit for each head" as the main cause of the gap. The two XAI experiments measured this limit:

- **Window share at its ceiling**. In layer 15, head 14, the window branch gives 0.365 of the weight. This value is the ceiling α / (1 + α) of the layer ([sample attention weights](experiments/xai-sample-attention-weight.md), finding 6).
- **Local heads match worst**. The more local a teacher head is, the larger its TV distance to Lizard (correlation −0.75 to −0.78).
- **Equal long-range share in a layer**. All heads of a layer get almost the same long-range share (standard deviation 0.034–0.048, teacher 0.126).
- **Two heads give 20% of the loss**. Layer 15, heads 14 and 23 give approximately 20% of the stage 1 loss ([layer-wise MSE](experiments/xai-layer-wise-mse.md), finding 5).

The [math against code](math-code-discrepancy.md) comparison lists the readings of the paper that the code chose: D1 (normalization), D2 (shared maps), D3 (softmax), D5 (sink term). v2 makes these choices options.

## Plan

1. **Check the source**. Mathbox skill `literature-check`: what does version 4 of the paper state for D1, D2, D3, D5 and α?
2. **Check the derivations**. Mathbox skill `proof-audit`: the claims P1–P5 that the options depend on.
3. **Implement**. One option for each reading and each code change. The defaults give v1. The parameter names stay those of v1, so the scripts and the checkpoints still work.
4. **Test**. Mathbox skill `computation-audit`. The parallel form against a loop form of [math formulas](math-formula.md), the decode form against the parallel form, and the defaults against v1. Negative controls show that the checks find errors.
5. **Document**. This document and the model config.

## Options

| Option | Values (default first) | Source | Reading of the paper (version 4) |
|---|---|---|---|
| `gla_norm` | `row`, `none`, `joint` | D1, C3 | `row` is the parallel form of Section 3.1 (v1). `none` is the recurrent form of Section 3.1 and the matrix form of Section 4. `joint` is one denominator for both branches and the sinks, as in LoLCATs. |
| `alpha_per_head` | `false`, `true` | C1, X1 | The paper gives one α. The LoLCATs default has one window factor for each head. |
| `alpha_init`, `train_alpha` | `1.0`, `true` | D14 | The paper does not call α learnable (see "Literature check"). `train_alpha: false` keeps α at `alpha_init`. |
| `feature_map_per_head` | `false`, `true` | D2, C2 | Probably the reading of Appendix B: the LoLCATs default has one map for each head. |
| `feature_activation` | `softmax`, `exp` | D3 | Table 13 gives softmax. Section 4 gives exp. |
| `gate_per_head` | `false`, `true` | C4, X3 | A deviation. Table 4 of the paper prefers one gate for all heads. |
| `gate_bias_init` | `null`, a number | C6 | Not in the paper. With 3.0, the gate starts at γ = σ(3) ≈ 0.953. |
| `window_rope` | `false`, `true` | C5 | A deviation. Lizard has no RoPE. |

These parts stay as in v1:

- the sink term $\sum_j \exp(t_j)$ (D5, see P4),
- one set of 4 sink logits for each layer,
- the gated branch without RoPE,
- the Lizard calculations in at least float32,
- the dense parallel form (D7),
- the loss and the target of stage 1 (D10–D15).

## Literature check

**Source**. Lizard, arXiv:2507.09025, version 4 (18 Apr 2026). The check used a text extraction of the PDF that the user supplied. The equations in the extraction are hard to read. Thus the check compared them with [math formulas](math-formula.md), which comes from images of version 4. The check did not cache the source.

**Checks performed**. Authentication: the arXiv stamp in the extraction (version 4). Extraction: the passages below. Application: one row for each question.

| Question | What version 4 states | Verdict |
|---|---|---|
| D1: denominator of the gated branch | Section 3.1, parallel form: a denominator. Section 3.1, recurrent form: $\mathbf{S}_i = \boldsymbol{\Gamma}_i \mathbf{S}_{i-1} + \phi_k(\mathbf{k}_i)\mathbf{v}_i^\top$, $\hat{\mathbf{y}}_i = \phi_q(\mathbf{q}_i)^\top \mathbf{S}_i$, no denominator. Section 4, matrix form for training: no denominator. | Unverified. The paper has both forms, and they are not equal (P1). v2 offers both. |
| D2: one map for each head | Appendix B: "For the other designs, we adopted the default values used by prior work (Zhang et al., 2025)", which is LoLCATs. LoLCATs uses `untied_head_einsum`, one map for each head. | Conditional: one map for each head, if "other designs" includes the feature map. |
| D3: activation | Section 4: $\phi(\mathbf{x}) = [\exp(\mathbf{x}\mathbf{W}) \oplus \exp(-\mathbf{x}\mathbf{W})]$, and "this exponential-based structure is critical" for the log-space form. Table 13: "Hedgehog Feature Activation: Softmax". | Unverified. The two statements disagree. P3 shows that the log-space form of Section 4 needs exp. |
| D5: sink term | Section 3.1 writes $\sum_{j} t_j$ in the denominator. The text calls $t_j$ "the logit of a meta-memory token". | Conditional: $\sum_j \exp(t_j)$, if "logit" has its usual meaning. |
| α | Section 3.1: $\hat{\mathbf{Y}}_{lizard} = \hat{\mathbf{Y}}_{gate} + \alpha \cdot \hat{\mathbf{Y}}_{anchor}$. No head index. The Figure 1 caption lists the learnable modules as φ, $\mathbf{W}_\gamma$ and $\mathbf{t}$. The loss lists the same three. | Not stated if α is learnable. [Math formulas](math-formula.md) and D14 say that the paper calls α learnable. The text of version 4 does not support this. |
| Sinks | $\mathbf{t} \in \mathbb{R}^m$, no head index. A cache of $w + m$ tokens. | Verified: one set for each layer, as in v1. |

**LoLCATs defaults** (from `linear_window_attention_tk.py` and `feature_map.py` in this repository):

- one window factor $\sigma(a_h)$ for each head, start value σ(−2.197) ≈ 0.1,
- one feature map for each head,
- one denominator for both branches.

The LoLCATs code divides the window weights by $\exp(\max_t s_{it})$, but not the linear weights. Thus its effective window factor changes with the row. The `joint` option of v2 applies the same shift to both branches, so the result is exact.

## Proof audit

Mathbox skill `proof-audit`. Self-review of the revision `a677e3c` (`docs/math-formula.md`, `docs/math-code-discrepancy.md`, gap analysis 2, section 3.1). The derivations are by hand. A float64 script recomputed the smallest cases (6 positions).

| Claim | Verdict | Decisive evidence |
|---|---|---|
| **P1a**: The recurrent form without a denominator equals the parallel form without a denominator | Proved as written (a scalar gate for each position, also one for each head) | Induction: $\mathbf{S}_i = \sum_{t \le i} \prod_{l=t+1}^{i} \gamma_l\, \phi_k(\mathbf{k}_t)\mathbf{v}_t^\top$. Recomputed difference 4.4e-16. The matrix form of Section 4 gives the same weights $c_i / c_t$, if $c_t > 0$. |
| **P1b**: With a second state $\mathbf{z}_i = \gamma_i \mathbf{z}_{i-1} + \phi_k(\mathbf{k}_i)$, the recurrent form equals the normalized parallel form | Proved as written | The same induction for $\mathbf{z}_i$. The denominator is at least $\phi_q(\mathbf{q}_i)^\top\phi_k(\mathbf{k}_i) > 0$, because the features are positive. Recomputed difference 2.2e-16. |
| **P2**: The window share of a head is at most α / (1 + α) | Correct only for α ≥ 0. The code does not limit α. | The gated branch sums to 1, and the window branch sums to $\rho_i \le 1$. The share is $\alpha\rho_i / (1 + \alpha\rho_i)$, which increases with $\alpha\rho_i$. |
| P2 with one α for each head (C1) | The ceiling stays: α_h / (1 + α_h) for each head | The gated branch still adds a weight of exactly 1 to each row. |
| P2 without a denominator (C3-i) | The ceiling disappears | The gated weight of a row, $\mu_i = \sum_t \prod \gamma\, \phi_q^\top\phi_k$, has no constant value. With softmax features, $0 < \phi_q^\top\phi_k \le 2$, so $\mu_i \le 2i$. With exp features, $\mu_i$ has no upper limit. |
| P2 with one joint denominator (C3-ii) | The ceiling disappears. Each row is a weighted mean of values with total weight below 1. This needs α ≥ 0, so `joint` uses max(α, 0). | Test 7 shows a share of more than 0.7 / 1.7 with `joint`. |
| **P3**: The log-space form of Section 4 equals $(\phi(\mathbf{Q}) \odot \mathbf{C})(\phi(\mathbf{K}) / \mathbf{C})^\top$ | Correct only for the exp activation. Refuted for the softmax activation of Table 13. | $\mathrm{softmax}(\mathbf{x}\mathbf{W} + \log c) = \mathrm{softmax}(\mathbf{x}\mathbf{W})$ for a scalar $c$. Thus the gate disappears. Recomputed: equal to the ungated map (1e-16), and 0.71 away from the gated map. |
| P3, numerical range | The form over the whole sequence overflows | At γ = 0.5 (the start value of v1), $1/c_t$ is finite only up to $t = 126$ in float32 and in bfloat16. The paper mentions chunkwise operations. Only a chunkwise form can avoid this. |
| **P4**: The written sink term $\sum_j t_j$ can make the denominator of the window branch 0 or negative | Proved | With $m = 4$ and one key with score 0: $t_j = -1$ gives a denominator of −3, and $t_j = -0.25$ gives 0. With $\sum_j \exp(t_j)$, the denominator is always positive. |
| **P5**: The decode forms of `joint` and `window_rope` equal their parallel forms | Proved, with a condition | `joint`: P1b for the gated sums, and the same window sums over the cached keys. `window_rope`: RoPE scores depend only on $t - i$, so cached keys rotated at their own positions are correct. The condition: the model passes the true positions when it decodes. |

**Remaining gaps**. The paper itself does not settle D1, D3 and α. v2 does not settle them either. It makes them testable.

## Implementation

| Part | File |
|---|---|
| Attention class `LolcatsLizardAttentionV2` | `src/model/linear_attention/lizard_attention_v2.py` |
| Attention type `lolcats_llama_lizard_v2` | `src/model/convert_model.py`, `src/model/linear_attention/__init__.py` |
| Model config, all options at their defaults | `configs/model/distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32.yaml` |
| Tests | `tests/test_lizard_attention_v2.py` |
| Evaluation summary | `scripts/compare_stages.py`: the mean α for α with one value for each head, and `W_gamma.bias` in the checkpoint check |

- **Same parameter names as v1**: `phi_q.weight`, `phi_k.weight`, `W_gamma.weight`, `meta_tokens` and `alpha_blend`. `W_gamma.bias` exists only with `gate_bias_init`. With `feature_map_per_head`, `phi_q.weight` has the shape (32, feature dimension, 64).
- **α with `train_alpha: false`** is a buffer. Thus it is not trained and not in the checkpoint of trainable weights.
- **Generation** uses `LizardAttentionCache` of v1. With `window_rope`, the cache holds the keys after RoPE.
- **XAI**: `branch_weights(q, k, gamma, rope)` returns the dense weights of the two branches, with α in the window weights. Their sum times v is the output of the layer (test 5). `attention_weights.py` and `ablate.py` still calculate the v1 form. For v2 options, they must use `branch_weights`. `layer_mse.py` uses the outputs of the layer itself, so it is correct for all v2 options.

## Computation audit

Mathbox skill `computation-audit`.

**Contract**. Claim: for each option, v2 calculates the formulas of [math formulas](math-formula.md) as the table "Options" reads them. With the default options, it gives the outputs of v1. Assertion tested: on tiny random layers in float64, the relative error is below 1e-10, or the outputs are identical. Non-claims: bfloat16 behavior, layers of the size of Llama-3.2-1B (except the parameter counts), speed, and the effect of an option on the model quality.

**Inputs**. A LlamaAttention with hidden size 32, 4 query heads, 2 key/value heads, head dimension 8, feature dimension 6, window 4 and 11 positions. All weights are random, also the Lizard parameters. The start values (γ = 0.5, the same α for each head) would hide errors in the options for each head. Seeds 0–7 with `torch.manual_seed`.

| # | Check | Options | Result |
|---|---|---|---|
| 1 | Defaults against v1: forward pass, distillation outputs, and prefill with token-by-token decode | Defaults | Identical (`torch.equal`) |
| 2 | Parallel form against the loop form of the math document | 12 sets: each option alone, and 2 combinations | 12 pass, relative error < 1e-10 |
| 3 | Decode against the parallel form: prefill 3 tokens, one call with 2 tokens, then one token at a time past the window | The same 12 sets | 12 pass, relative error < 1e-10 |
| 4 | A change at position 7 does not change the outputs before position 7 | 4 sets | 4 pass |
| 5 | `branch_weights` times v gives the output. The weights are causal, non-negative and inside the window. `row`: the gated weights sum to 1. `joint`: each row sums to less than 1. | 5 sets | 5 pass |
| 6 | Parameter counts at the size of Llama-3.2-1B (feature dimension 32): C1 +31, C2 4,096 → 131,072, C4 +31 × 2,048, C6 +1 for each layer. These agree with the table in section 5.2 of gap analysis 2. | C1, C2, C4, C6 | Pass |
| 7 | Window share: `row` stays at most 0.7 / 1.7. `joint` goes above it. | `row`, `joint` | Pass |
| 8 | In a tiny `LolcatsLlamaForCausalLM` (2 layers): conversion, the teacher mode, the evaluation path with `use_cache=True`, and generation with prefill and decode | `all_joint` combination | Pass, relative error < 1e-6 (the model casts the logits to float32) |

**Negative controls**. Four errors went into a copy of v2, one at a time. Each error made checks fail:

| Error | Failed checks |
|---|---|
| `joint`: no sinks in the denominator | 7 |
| Decode: the gate of head 0 for all heads | 4 |
| `window_rope`: keys without RoPE in the prefill cache | 4 |
| `exp`: the wrong sign in the second half of the map | 2 |

**Command** (from the repository root):

```bash
python -m pytest -q tests/test_lizard_attention_v2.py
```

**Environment**: CPU, Python 3.11, torch 2.0.1, transformers 4.43.1, pytest 9.1.1. Result: 37 passed in approximately 3 seconds. The manifest of the run with the provenance runner of `computation-audit` is in [`lizard-attention-v2/`](lizard-attention-v2/).

**Outcome**: implementation and finite assertion verified in the stated range. The loop reference comes from the math document, not from the code. But it reads the formulas as the table "Options" does. Thus it cannot find a wrong reading of the paper.

## How to run an experiment

1. Copy the v2 model config. Change one option. Put the change in the file name, for example `distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_alphahead.yaml`.
2. Merge the config, then build a new Docker image ([document 3](03-infrastructure.md)).
3. Run stage 1 on HF Jobs:

```bash
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=3h \
  ARGS="--model_config <the new config> --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b --no_finetune"
```

4. Compare with config 1 of the [second round](experiments/second-round.md) (validation loss 3.9764), then run `layer_mse.py` and the MMLU subset, PIQA and ARC-Easy.

The order of section 5.2 of gap analysis 2 applies: C1 (`alpha_per_head`) first, then C2 (`feature_map_per_head`), then C3 (`gla_norm`) or C6 (`gate_bias_init`). C4 (`gate_per_head`) and C5 (`window_rope`) come last, because they deviate most from the paper.

### Experiment C1: one α for each head

**Config**: `distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_alphahead`. It is the v2 config with one change: `alpha_per_head: true`. The stage 1 recipe is config 1 of the [second round](experiments/second-round.md). Thus the only difference from config 1 is one α for each of the 32 heads.

**Stage 1 on HF Jobs (H200)**, after the merge of the config and a new Docker image:

```bash
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=4h \
  ARGS="--model_config distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_alphahead --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b --no_finetune"
```

- **Time**: approximately 2.5 hours, as for stage 1 in float32 ([gradient clipping](experiments/gradient-clipping.md), finding 5). The 31 extra α values for each layer add almost no calculation. `HF_TIMEOUT=4h` gives a margin of approximately 1.5 hours.
- **Checkpoint**:

```
checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_alphahead/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_alphahead-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
```

**Evaluation on the A10**, after `git pull` on `main` (the v2 code must be on the machine):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
DISTILL_CONFIG=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b \
MODEL_CONFIG=distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_alphahead \
MODELS=stage1 TASKS="mmlu_subset piqa arc_easy" \
scripts/compare_stages.sh 2>&1 | tee eval-v2-alphahead.log
```

`summary.md` reports the mean α of each layer. `summary.json` has the 32 values of each layer (`alpha_per_head`).

**CPU check**. `distill_llama.main()` ran stage 1 with this config on a tiny Llama (3 layers, 4 heads) and finished the training:

- The layers were `LolcatsLizardAttentionV2`.
- `alpha_blend` was trainable, with the shape (4,).
- The checkpoint got the name above.
- After 25 steps, the 4 α values of each layer were different (0.9814–0.9819). Thus each head trains its own α.

The final text generation check of `distill_llama.py` then stopped with an error about `token_type_ids`. The tiny test tokenizer gives this field, and the Llama tokenizer does not. The v1 config gave the same error in the same test.

#### Results of the C1 run (2026-10-04)

- **Training**: stage 1 on HF Jobs with the command above. These notes do not record the job ID or the time.
- **Evaluation**: `scripts/compare_stages.sh` with `MODELS=stage1`, on `student06`, GPU 0 (A10). Run directory: `results/stages/20261004-004056-v2-alphahead`. Code: lolcats `8bd2e88`, harness `b281b09`, torch 2.5.1.
- **Comparison values**: config 1 of the [second round](experiments/second-round.md). The z values use unpaired SEs.

| Checkpoint | SHA-256 | Size | Parameters | Dtype | Stored step | Stored loss |
|---|---|---|---|---|---|---|
| C1, stage 1 | `3f359abeed2415efeb4a9603f5ffd31a4e42b345b4deb91742fecd10628a881b` | 445,886 B | 98,880 | float32 (80 tensors) | 1100 | **3.8092** |

The checkpoint has 496 more parameters than config 1 (98,384): 31 extra α values in each of the 16 layers. In all three evaluations, all 80 Lizard tensors loaded with their values.

| Measure | Config 1 (one α for each layer) | C1 (one α for each head) | Difference |
|---|---|---|---|
| Stage 1 validation loss | 3.9764 | **3.8092** | −4.2% |
| MMLU subset | 26.7 ± 2.6 (76 right) | 23.9 ± 2.5 (68 right) | −2.8 points, z ≈ −0.8 |
| PIQA | 57.3 ± 1.2 (normalized 56.6) | 57.5 ± 1.2 (normalized 55.0) | +0.2 points, z ≈ 0.1 |
| ARC-Easy | 36.5 ± 1.0 (normalized 36.4) | 34.6 ± 1.0 (normalized 34.0) | −1.9 points, z ≈ −1.3 |
| "A" / "B" / "C" / "D" on MMLU | 51.2% / 3.9% / 3.9% / 41.1% | 64.9% / 5.3% / 7.0% / 22.8% | – |
| Letter mass, confidence, entropy | 0.019, 0.473, 1.726 bits | 0.015, 0.477, 1.720 bits | – |

**Gates and Lizard parameters** (the same 5-shot prompt as for config 1):

- The mean α of the layers is 0.508–0.666 (config 1: 0.472–0.644). The values of the 32 heads are in `lizard.json` of each task folder (`alpha_per_head`).
- The heads of a layer now have different α values, with a standard deviation of 0.045–0.122. In layer 1, the values go from 0.45 to 1.04. The start value was 1.0 for all heads.
- The sink logits, ‖W_γ‖ and the RMS of the feature maps are almost the same as in config 1.
- The gated branch keeps more weight after 512 tokens in 14 of the 16 layers. Examples: layer 9 keeps 0.79 (config 1: 0.48), and layer 10 keeps 0.91 (0.72). Layers 7 and 13 keep less (0.17 against 0.35, and 0.87 against 0.88). These values come from one prompt.

**Layer-wise MSE** (`layer_mse.py`, 16 validation batches, 32,768 tokens). The recalculated loss is 3.8092, the same as the stored loss (difference +0.000%). The reference is `docs/experiments/xai-layer-wise-mse/config1.json`.

| Layers | MSE against config 1 | Relative MSE (config 1 → C1) |
|---|---|---|
| 0–3 | −0.5% to −3.9% | Layer 2: 1.109 → 1.085, still above 1 |
| 4–9 | −1.2% to −4.4% | 0.29–0.51 → 0.29–0.49 |
| 10–14 | −2.6% to −6.0% | 0.33–0.48 → 0.31–0.47 |
| 15 | −6.6% | 0.623 → 0.582 |

- The MSE decreases in all 16 layers. The largest decreases are in layers 10 and 13–15.
- Layer 15 still gives 28.9% of the loss (config 1: 29.7%).
- In layer 2, the Lizard output is still farther from the teacher than an output of 0 (relative MSE above 1).
- Layer 15, head 14: MSE 0.234 → 0.216 (−7.8%), 11.5% → 11.1% of the loss. Head 23: 0.146 → 0.145 (−0.7%), 7.2% → 7.4% of the loss. Together, the two heads still give 18.5% of the loss (config 1: 18.7%).

**Check against the gain rule of the plan** (section "Experiment plan with fewer runs"):

| Part of the rule | Result | Met |
|---|---|---|
| Validation loss at least 5% below config 1 | −4.2% | No |
| Lower MSE of layer 15, head 14 | Head 14: 0.234 → 0.216 (−7.8%), 11.5% → 11.1% of the loss. Layer 15: −6.6%. | Yes |
| No accuracy loss of more than approximately 2 points | MMLU −2.8 and ARC-Easy −1.9, neither clear | Borderline |

**Finding 1: one α for each head alone gives no gain by the rule of the plan**. The loss is 4.2% lower, below the limit of 5%. The MSE of layer 15, head 14 is 7.8% lower. The accuracies do not improve. As in the earlier runs, a lower loss does not give higher accuracies ([stage 2 on config 1](experiments/stage2-config1.md), finding 3).

**Finding 2: the result agrees with P2**. With one α for each head, each head still has a ceiling, α_h / (1 + α_h). The normalized gated branch still adds a weight of exactly 1 to each row. Thus C1 alone can only change the compromise between the heads. It cannot remove the limit.

**Finding 3: the gate keeps a longer memory in most layers**. This agrees with the mechanism of section 3.3 of gap analysis 2. If local heads can take more from the window branch, the shared gate can stay nearer to 1 for the other heads. This is a hypothesis.

**Finding 5: the local head with the largest error gets the largest α of its layer, but its ceiling stays far below the teacher**:

| Head | Teacher weight inside the window | α (C1) | Rank in the layer | Ceiling α / (1 + α), C1 | Ceiling, config 1 |
|---|---|---|---|---|---|
| Layer 15, head 14 | 0.913 | 0.855 | 1 of 32 (layer mean 0.574) | 0.461 | 0.365 |
| Layer 15, head 23 | 0.61 | 0.761 | Above the mean | 0.432 | 0.365 |
| Layer 0, head 2 (one token back) | Not measured | 0.436 | Below the mean (0.508) | 0.304 | 0.321 |

- **Layer 15, head 14**. Its ceiling rose from 0.365 to 0.461. The teacher puts 0.913 of its weight inside the window. Thus the ceiling still binds, and the MSE of this head decreased by only 7.8%. A larger α would also make the total weight of the row larger than 1, because the gated branch always adds 1 (D1). Thus α = 0.855 is the best compromise for this head, and only a change of the normalization (C3) can remove the ceiling.
- **Layer 0, head 2** got a smaller α than the mean of its layer. This head attends to the previous token. Without RoPE, the window branch cannot find "one token back" (section 3.3 of gap analysis 2). Thus a larger α cannot help this head. This supports C5 (`window_rope`) for this type of head, in R4.

**Finding 4: on MMLU, the answers are still at the level of guessing**. A model that ignores the questions and uses the same letter shares gets 24.6. C1 gets 23.9.

**Decision for the plan**: C1 alone gives no gain. R1 is the next run.

## Experiment plan with fewer runs

**The problem**. v2 has 8 options. All their combinations give 3 × 2⁷ = 384 configs. One option at a time needs 8 stage 1 runs before any combination. Each stage 1 run in float32 takes approximately 2.5 hours on the H200 (approximately $12.50 at $5.00 for each hour).

**Three rules make the plan shorter**:

1. **Drop options without evidence**. Options that no XAI result points at, or that have a known problem, are not in the plan (table "Options not in the plan").
2. **Test a group in one run**. The first run tests the three parts of the per-head limit together (section 3 of gap analysis 2). If the group gives no gain, one run removes all three options.
3. **Remove one option only after a gain**. These runs remove only the deviations from the paper. If a run without a deviation is as good, the plan keeps the config that is closer to the paper.

**Gain**: a stage 1 validation loss at least 5% below the reference run, and a lower MSE of layer 15, head 14 in `layer_mse.py`. The accuracies (MMLU subset, PIQA, ARC-Easy) must not fall by more than approximately 2 points. The project has one seed for each run, so a difference of a few percent is not clear.

| Run | v2 options (all others keep their defaults) | Question | Run only if | Decision |
|---|---|---|---|---|
| B0 | None: config 1 of the second round | Reference | Exists (validation loss 3.9764) | – |
| **R1** | `alpha_per_head`, `feature_map_per_head`, `gla_norm: joint` | Does the removal of the per-head limit lower the loss? | Always | No gain: stop the architecture path and skip R2–R6 |
| R2 | R1 without `gla_norm: joint` | Is the joint denominator (a deviation) necessary? | R1 has a gain | R2 within 2% of R1: keep R2 |
| R3 | R1 without `alpha_per_head` | Is one α for each head (a deviation) necessary? | R1 has a gain | R3 within 2% of R1: keep R3 |
| W | The best config of R1–R3, with the fewest deviations | – | – | – |
| R4 | W + `window_rope` + `gate_bias_init: 3.0` | Do position information for the local heads and a gate start near 1 add a gain? | Always after W | No gain: drop both options |
| R5 | W + `gate_bias_init: 3.0` | Which of the two options gives the gain? The effect of `window_rope` is R4 − R5. | R4 has a gain | Keep the options with a gain |
| R6 | The best config so far + `gate_per_head` | Does one gate for each head add a gain? | X3 predicts a decrease of the loss of 5% or more | Keep only with a gain |
| R7 | Stage 2 of the final config (float32, the stage 2 recipe of [stage 2 on config 1](experiments/stage2-config1.md)) | Final scores | Always, last | Compare with stage 2 on config 1 (PIQA 67.7, ARC-Easy 54.9) and with the paper |

**Number of runs**:

- **R1 has no gain**: 1 stage 1 run. The plan then stops.
- **Normal case**: 4–5 stage 1 runs (R1, R2, R3, R4, maybe R5) and 1 stage 2 run.
- **One option at a time**: at least 8 stage 1 runs, before any combination.

**The C1 run** finished: −4.2% validation loss and no accuracy gain ("Experiment C1"). Thus `alpha_per_head` probably adds little to R1, and R3 probably keeps the gain of R1. R3 must run to show this. A config that removes C1 without a run would be untested.

**Screens without training** (optional, A10, forward passes only). They need a small extension of `layer_mse.py` with `branch_weights`:

- X1: the best α for each head, from the outputs of config 1. It decides R3.
- X2: the best scale of the gated branch for each head. It predicts the effect of `gla_norm`.
- X3: the best constant gate for each head. It decides R6.

**Shorter screens** (optional). R2–R6 can run for 1 epoch (approximately 1.25 hours), and only the final config for 2 epochs. This halves their cost. But in all float32 runs, the validation loss still decreased at the end, so the order of two configs can change.

### Options not in the plan

| Option | Why it is not in the plan | When to run it |
|---|---|---|
| `gla_norm: none` | It removes the ceiling as `joint` does (P2), but the total weight of a row has no constant value (up to 2i with softmax features). `joint` is the LoLCATs default. | Only if the thesis needs the recurrent form of the paper |
| `feature_activation: exp` | Only the log-space form of Section 4 needs it (P3), and v2 uses the dense form. No XAI result points at it. With `gla_norm: none`, its weights have no upper limit. | Only for the reading question D3 |
| `train_alpha: false`, `alpha_init` | It conflicts with `alpha_per_head` in R1. It answers only a reading question: the paper does not call α learnable. | Only for the reading question of α |

### Run R1

**Config**: `distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_joint`. The stage 1 recipe is config 1 of the second round, as in B0.

```bash
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=4h \
  ARGS="--model_config distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_joint --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b --no_finetune"
```

- **Time**: approximately 2.5 hours. A map for each head does the same calculation for each head as the shared map. The joint denominator adds no matrix product.
- **Checkpoint**: `checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_joint/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_joint-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt`
- **Evaluation**: the command of "Experiment C1", with `MODEL_CONFIG=distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_joint`.
- **CPU check**. `distill_llama.main()` trained stage 1 with this config on a tiny Llama (3 layers, 4 heads). The checks:
  - The layers were `LolcatsLizardAttentionV2`.
  - `phi_q.weight` had the shape (4, 8, 16), one map for each head.
  - `alpha_blend` had the shape (4,).
  - The checkpoint got the name above.
  - The final text generation check stopped with the `token_type_ids` error of the tiny test tokenizer, as in "Experiment C1".

## Limits

- **XAI scripts**: `attention_weights.py` and `ablate.py` calculate the v1 form. Their results are correct for v2 only with the default options. `layer_mse.py` is correct for all v2 options. `compare_stages.py` reports the gate of head 0 when `gate_per_head` is true.
- **`gla_norm: none` with `feature_activation: exp`**: the gated weights have no upper limit (P2). Training can become unstable.
- **`gla_norm: joint`** uses max(α, 0). If training pushes α below 0, the window branch stops, and α gets no gradient.
- **Not tested**: bfloat16, sequences of 2,048 tokens, and the speed and memory of the options for each head.
