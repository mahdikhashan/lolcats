# 14. Liger and GLA

**Status:** Math of two related methods, from their official code. Proof audit of five new claims (P7–P11), with float64 checks. A critical review of the stage 1 runs so far, one proposed code change and a new order of next steps. No new training.

- **GLA**: Gated Linear Attention Transformers with Hardware-Efficient Training (Yang, Wang, Shen, Panda, Kim, ICML 2024). Table 4 of the Lizard paper compares the Lizard gate with the gate of this paper.
- **Liger**: Linearizing Large Language Models to Gated Recurrent Structures (Lan, Sun, Hu, Du, Cheng, arXiv:2503.01496). Liger linearizes a pretrained model with gated linear attention and a sliding window, as Lizard does.

## Sources

Mathbox skill `literature-check`.

| Source | Version read | Files read | Status |
|---|---|---|---|
| Liger, arXiv:2503.01496 | Official code [OpenSparseLLMs/Linearization](https://github.com/OpenSparseLLMs/Linearization), commit `0b364eb` (2025-07-08). Its README links the arXiv paper. | `liger/models/liger_gla/modeling_liger_gla.py`, `training/train.py`, `training/trainer.py`, `training/utils.py`, `configs/liger_gla.yaml` | Code read. Paper text not read. |
| GLA, arXiv:2312.06635, OpenReview `ia5XvxFUJT` | Code of the GLA authors: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), commit `3e52d5a` (2026-10-04). Its README cites the GLA paper for `fla/layers/gla.py`. | `fla/layers/gla.py`, `fla/ops/gla/naive.py`, `fla/ops/gla/chunk.py` (docstring) | Code read. Paper text not read. |

**Access**. The network policy of this container blocks arxiv.org, openreview.net, proceedings.mlr.press and huggingface.co. GitHub is open. A web search lists Liger in the ICML 2025 proceedings (PMLR volume 267). This document did not open that page.

**Verdict**. The formulas below are the formulas of the official code (checked). The text of the two papers can say more or differ (unverified). To check the papers, put the two PDFs into the session, as with the result files of document 13.

## 1. GLA

Source: `fla/layers/gla.py` and `fla/ops/gla/naive.py`. One head, position $t$, key dimension $d_k$.

**Gate**. One gate value for each key dimension, from a low-rank projection of the input $\mathbf{x}_t$:

```math
\boldsymbol{\alpha}_t = \sigma\left( \mathbf{x}_t \mathbf{W}_1 \mathbf{W}_2 + \mathbf{b} \right)^{1/\tau}, \qquad \tau = 16, \quad \mathbf{W}_1 \in \mathbb{R}^{d \times 16}, \quad \mathbf{W}_2 \in \mathbb{R}^{16 \times d_k}
```

The code calculates $\log \boldsymbol{\alpha}_t = \mathrm{logsigmoid}(\cdot) / 16$. With a logit of 0, the gate is $0.5^{1/16} = 0.958$. Thus the gate starts with a long memory.

**Recurrent form**. No feature map (the default `feature_map=None`) and no denominator:

```math
\mathbf{S}_t = \mathrm{Diag}(\boldsymbol{\alpha}_t)\, \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top, \qquad \mathbf{o}_t = \mathbf{S}_t^\top \frac{\mathbf{q}_t}{\sqrt{d_k}}
```

**Parallel form** (P11 below). With the cumulative gates $\mathbf{b}_t = \prod_{s=1}^{t} \boldsymbol{\alpha}_s$ as the rows of $\mathbf{B}$:

```math
\mathbf{O} = \left( \left( (\mathbf{Q} \odot \mathbf{B}) \left( \frac{\mathbf{K}}{\mathbf{B}} \right)^\top \right) \odot \mathbf{M} \right) \mathbf{V}
```

The division by $\mathbf{B}$ overflows for long sequences (P3 of document 13). The code (`chunk_gla`) calculates the same output in chunks of the sequence, with the gates in log space.

**Output**. The code normalizes each head (RMSNorm), multiplies the result by an output gate, then applies the output projection:

```math
\mathbf{y}_t = \left( \mathrm{RMSNorm}(\mathbf{o}_t) \odot \mathrm{swish}(\mathbf{x}_t \mathbf{W}_g) \right) \mathbf{W}_O
```

## 2. Liger

Source: `modeling_liger_gla.py` (class `LigerGatedLinearAttention`). Liger keeps the q, k, v and o projections of the pretrained model. Apart from LoRA, it adds no parameters.

**Gated branch**. The feature map is a softmax over the head dimension (64), with no weights. The gate comes from the key. The code applies `AdaptiveAvgPool1d` with an output size equal to the input size, thus the pooling does not change the key.

```math
\phi(\mathbf{q}_t) = \mathrm{softmax}(\mathbf{q}_t), \qquad \phi(\mathbf{k}_t) = \mathrm{softmax}(\mathbf{k}_t), \qquad \boldsymbol{\alpha}_t = \sigma(\mathbf{k}_t)^{1/16}
```

```math
\mathbf{S}_t = \mathrm{Diag}(\boldsymbol{\alpha}_t)\, \mathbf{S}_{t-1} + \phi(\mathbf{k}_t) \mathbf{v}_t^\top, \qquad \mathbf{o}^{gla}_t = \mathbf{S}_t^\top \phi(\mathbf{q}_t)
```

The code calls `fused_chunk_gla` with `scale=1`. There is no denominator.

**Window branch**. The softmax attention of the teacher, with RoPE, over approximately the last 64 positions (FlashAttention with `sliding_window=64`). No sinks.

```math
\mathbf{o}^{swa}_t = \sum_{s=t-w+1}^{t} \frac{\exp\left( \varphi_R(\mathbf{q}_t)^\top \varphi_R(\mathbf{k}_s) / \sqrt{d} \right)}{\sum_{s'=t-w+1}^{t} \exp\left( \varphi_R(\mathbf{q}_t)^\top \varphi_R(\mathbf{k}_{s'}) / \sqrt{d} \right)}\, \mathbf{v}_s, \qquad w = 64
```

**Output**. A constant mix with equal weights:

```math
\mathbf{o}_t = 0.5\, \mathbf{o}^{swa}_t + 0.5\, \mathbf{o}^{gla}_t
```

The GSA variant of the same code has the comment "0.5 is important".

**Training** (`training/train.py`, `configs/liger_gla.yaml`). One stage, with the next-token cross-entropy loss. LoRA with rank 8 on q, k and v. AdamW, learning rate 1e-3, 2 epochs of alpaca-clean, sequences of 1,024 tokens, gradient clipping at 1.0. Liger has no attention transfer stage. The code uses the MSE trainer only for its LoLCATs baseline.

## 3. Lizard against GLA and Liger

| Part | Lizard paper and v1 | v2 options | GLA | Liger |
|---|---|---|---|---|
| Gate | One scalar for each position, shared by all heads: $\sigma(W_\gamma \mathbf{x})$. Start 0.5. | `gate_per_head`, `gate_bias_init` | One value for each head and key dimension: $\sigma(\cdot)^{1/16}$. Start 0.958. | One value for each head and key dimension: $\sigma(\mathbf{k})^{1/16}$. No parameters. |
| Feature map | Hedgehog with learned weights (softmax form, 2 × 32 features) | `feature_map_per_head`, `feature_activation` | None | Softmax over the head dimension, no weights |
| Denominator of the gated branch | Row sum (`row`) | `none`, `joint`, `hybrid` | None | None |
| Window | 128 positions and 4 sinks, no RoPE | `window_rope` | – | 64 positions, RoPE, no sinks |
| Mix of the branches | $\hat{\mathbf{y}}_{gate} + \alpha \hat{\mathbf{y}}_{window}$, α learned | `alpha_per_head`, `hybrid` | – | Constant 0.5 and 0.5 |
| Output normalization and gate | None | – | RMSNorm and swish gate (new weights) | None |
| Training | Stage 1 (MSE to the teacher), then stage 2 (LoRA) | – | Pretraining | One stage (cross-entropy and LoRA) |
| New parameters (Llama-3.2-1B, fd32) | 98,384 | Up to 2,130,496 (R1b) | – | 0 (LoRA only) |

**Note on Table 4 of the Lizard paper** ([math formulas](math-formula.md), section 8). The row "1D-Pooling" is the gate type of Liger: a gate from the keys, with no new parameters. It got the lowest MMLU score (44.1). The paper writes this gate as $\sigma(\mathrm{Pooling}(\mathbf{k}))$ and the GLA gate as $\sigma(\mathbf{x} W_{\gamma_1} W_{\gamma_2})$, both without the exponent 1/16 of the code above. With the exponent, these gates start near 0.96. Without it, they start near 0.5. The paper does not show which form its ablation used. Thus Table 4 does not settle if the gates of GLA and Liger are worse than the Lizard gate.

## 4. Proof audit

Mathbox skill `proof-audit`, self-review of `src/model/linear_attention/lizard_attention_v2.py` at commit `3b13620`. Notation of document 13: $\hat{\mathbf{y}}_g$ is the row-normalized gated output, with weights that sum to 1. $\hat{\mathbf{y}}_w$ is the window output with sinks, with weights that sum to $\rho_i \le 1$.

| Claim | Verdict | Decisive evidence |
|---|---|---|
| **P7**: Sink logits with equal start values stay equal during training | Proved as written | The loss uses the sinks only through $\sum_j \exp(t_j)$. Thus equal logits get equal gradients. Adam updates each element from its own gradient history only, so equal histories give equal updates. By induction, the logits stay equal. Thus 4 sinks act as 1 sink with logit $t + \ln 4$. All runs so far show 4 equal logits. Check: after 50 Adam steps, the difference is exactly 0. |
| **P8a**: In the row form, the total weight of row $i$ is $1 + \alpha\rho_i$ | Proved as written (α ≥ 0) | The gated weights sum to 1, and the window weights sum to $\alpha\rho_i$. Check: error 4.4e-16, total weight up to 1.7. |
| **P8b**: Division by $1 + \alpha$ keeps the share of each branch. The total weight becomes $(1 + \alpha\rho_i) / (1 + \alpha) \le 1$. | Proved as written (α ≥ 0) | Both branches get the same factor. The bound follows from $\rho_i \le 1$. Check: share difference 1.7e-16. |
| **P8c**: In the row and convex forms, the share does not depend on the position | Proved as written | The share $\alpha\rho_i / (1 + \alpha\rho_i)$ depends only on $\rho_i$. The window and sink weights set $\rho_i$, not the number of earlier tokens. Check: the same share (0.4831) at positions 16, 32, 64, 128 and 256 of a periodic input. |
| **P8d**: With α = 1 and no sink weight, the convex form is $0.5\, \hat{\mathbf{y}}_g + 0.5\, \hat{\mathbf{y}}_w$, the mix of Liger | Correct only after a restriction: $\rho_i = 1$ | With sinks, the window part is $0.5\,\rho_i$. Liger also has no denominator in its gated branch and uses RoPE in its window. Thus the mix is the same, but the branches differ. Check: error 7.6e-17. |
| **P9**: With `hybrid` and γ = 1, the window share decreases approximately as 1 / position | Correct only after a restriction: γ = 1 at all positions, and a mean kernel value $\phi_q^\top\phi_k$ above a positive constant | The gated weights of row $i$ sum to approximately $c \cdot i$. The window and sink weights are each at most $\lvert\alpha\rvert$, so their sum is at most $\lvert\alpha\rvert(w + m)$. Check: share × position stays at 0.44–0.45 from position 16 to 256. In R1b, γ is above 0.999 for every token in layers 3, 7 and 10–14. |
| **P10a**: A window branch without RoPE cannot give exact previous-token attention | Correct only after a restriction: inputs without position information. Shown with a repeated token. | Without RoPE, the score $\mathbf{q}_i^\top\mathbf{k}_t$ depends only on $\mathbf{x}_i$ and $\mathbf{x}_t$. If $\mathbf{x}_{i-1} = \mathbf{x}_{i-2}$, the two positions get the same weight. Previous-token attention puts all weight on $i - 1$. Llama has no position embeddings, so the inputs of layer 0 are token embeddings only. In later layers, the inputs can hold position information from earlier layers. Check: weight difference exactly 0 without RoPE, 0.329 with RoPE. |
| **P10b**: The window of Liger gives the weights of the teacher inside the window, divided by their sum over the window | Correct only after a restriction: the q and k projections of the teacher, thus before LoRA training | It uses the scores of the teacher with RoPE. Thus each window weight is $p_{it} / \sum_{t' \in \text{window}} p_{it'}$, with $p$ the weights of the teacher. |
| **P11**: The GLA parallel form equals the recurrent form. With one scalar gate for each position and the features $\phi$, it is the Section 4 form of Lizard and the gated branch of `gla_norm: none` | Proved as written (gates above 0) | $(\mathbf{q}_t \odot \mathbf{b}_t)^\top (\mathbf{k}_s / \mathbf{b}_s) = \sum_d q_{td}\, k_{sd} \prod_{r=s+1}^{t} \alpha_{rd}$. This is the weight of $\mathbf{v}_s$ in the recurrent form. Check: error 2.3e-16 (gate for each dimension) and 9.8e-17 against `branch_weights`. |

**Consequences**:

- **P2 of document 13 stays correct**, but the limit that binds is the total weight (P8a). A larger α gives the window a larger share, and it also makes the output larger than a weighted mean of the values. C1 found this compromise: α = 0.855 for layer 15, head 14 (document 13, finding 5).
- **P10a agrees with finding 3 of R1b**. A short gate weights recent tokens more, thus it gives position information. The window without RoPE cannot. In R1b, the local heads of layers 0 and 15 probably used the short gate, because their α is almost 0 (document 13, finding 3).
- **GLA starts with a long memory**. With the exponent 1/16, the gate 0.128 of layer 15 in R1b needs the logit −32.9. Without the exponent, it needs −1.92. Thus GLA makes short gates hard, and Liger leaves the local work to a window with RoPE.

**Computation check** (Mathbox skill `computation-audit`). [`liger-gla/check_claims.py`](liger-gla/check_claims.py) runs on tiny random layers in float64 on CPU, with the v2 code of commit `3b13620`. Output: [`liger-gla/check_claims.txt`](liger-gla/check_claims.txt), 11 checks pass. Two checks are controls: the window with RoPE (P10a) and the share of the row form (against P9). The checks cover tiny sizes only. They do not show the effect of a claim on the model quality.

## 5. Critical review of the stage 1 runs

All runs use feature dimension 32, except fd128. Values from documents 7–13 and their experiments.

| Run | Stage 1 validation loss | PIQA | ARC-Easy | MMLU subset |
|---|---|---|---|---|
| fd128, LoLCATs recipe (stage 1 of Run 2) | 3.2549 | 57.6 | – | 22.5 |
| fd32, LoLCATs recipe | 3.4219 | – | – | 23.2 |
| fd32, recipe of the paper, bf16 | 8.1641 | 55.8 | 34.1 | 25.3 |
| fd32, recipe of the paper, float32 | 4.9478 | 57.7 | 35.7 | 24.6 |
| Config 1 | 3.9764 | 57.3 | 36.5 | 26.7 |
| Config 2 (clipping) | 3.5092 | 57.5 | 35.6 | 23.2 |
| C1 | 3.8092 | 57.5 | 34.6 | 23.9 |
| R1b | 3.0542 | 55.9 | 29.9 | 24.9 |

1. **The stage 1 loss does not predict the accuracy**. Over the 6 runs with ARC-Easy, the rank correlation (Spearman) between loss and accuracy is +0.31 for ARC-Easy and −0.06 for PIQA. Without R1b, it is −0.20 and −0.36. A positive value means that a lower loss comes with a lower accuracy. With 5–6 runs, none of these values is clear. The run with the lowest loss has the lowest ARC-Easy.
2. **Stage 1 differences did not reach stage 2 either**. Run 2 (stage 1 loss 3.2549) and stage 2 on config 1 (3.9764) end within 0.3 points: PIQA 67.95 and 67.7, ARC-Easy 54.8 and 54.9. The two runs also have different stage 2 recipes.
3. **The MMLU subset gives no information after stage 1**. All runs have 22.5–26.7. A model that always selects "A" gets 24.2.
4. **The noise is large**. PIQA has an SE of 1.2 points, ARC-Easy 0.9–1.0. Each run has one seed. Thus the threshold of 5% on the loss in the plan of document 13 has no measured noise floor.
5. **The loss measures each layer alone**. Each layer gets the input of the teacher. All 16 layers and all positions count the same. Positions 0–127 give 6.25% of the loss, but the prompts of PIQA and ARC-Easy are short. The loss cannot see an error that grows from layer to layer.
6. **The two related methods are simpler**. Liger has no stage 1, no new parameters, a window with RoPE and a constant mix. GLA has no denominator and a gate that starts near 1. Lizard v1 has a window without RoPE. In C1, its previous-token head (layer 0, head 2) got a small α (document 13, finding 5). R1b added 2 million parameters, and its accuracy fell.

**Conclusion**. The plan of document 13 uses the stage 1 loss as its first test. The data does not support this. Stage 1 runs should be compared by ARC-Easy and PIQA, and the final answer comes from stage 2.

## 6. Proposed code changes

Keep the changes small. One new option, three existing options.

**New: `gla_norm: convex`** (P8). The row form divided by $1 + \lvert\alpha\rvert$:

```math
\mathbf{y}_i = \frac{\hat{\mathbf{y}}_g + \lvert\alpha\rvert\, \hat{\mathbf{y}}_w}{1 + \lvert\alpha\rvert}
```

- The output stays a weighted mean of the values (total weight at most 1), for every α.
- The share does not depend on the position (P8c), unlike `hybrid` (P9).
- The start value α = 1 gives the mix 0.5 and 0.5 of Liger (P8d). No new parameters.

The change in `lizard_attention_v2.py`, approximately 6 lines:

```python
GLA_NORMS = ('row', 'none', 'joint', 'hybrid', 'convex')

# lizard_params(): |alpha|, as for hybrid
elif self.gla_norm in ('hybrid', 'convex'):
    alpha = alpha.abs()

# lizard(), lizard_recurrent() and branch_weights(): convex uses the row-normalized gated branch
if self.gla_norm in ('row', 'convex'):
    y_gla = y_gla / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
y = y_gla + alpha * (sink_softmax(scores, meta) @ v)
if self.gla_norm == 'convex':
    y = y / (1 + alpha)
```

Tests: add `convex` to the option sets of `tests/test_lizard_attention_v2.py` (loop reference, decode, gradients). Add one check that α = 1 without sinks gives the mean of the two branches.

**Existing options, supported by GLA and Liger**:

| Option | Support | Expected effect |
|---|---|---|
| `window_rope: true` (C5) | The window of Liger uses RoPE (P10b). P10a explains why the window without RoPE fails for local heads. | Local heads use the window, and the gate can keep a long memory |
| `gate_per_head: true` (C4) | GLA and Liger have a gate for each head (and each key dimension). R1b, finding 4: one gate for 32 heads. | Local and long-range heads get different gates |
| `gate_bias_init: 3.0` (C6) | The GLA gate starts at 0.958. This bias gives 0.953. | A long memory at the start |

**Not proposed** (not simple):

- **A gate for each key dimension** (GLA, Liger). The dense form of v2 needs a weight matrix of size positions × positions for each head. With a gate for each dimension, it needs one for each dimension too, or the chunked kernels of GLA.
- **The feature map and gate of Liger without parameters**. A different method, not a change of Lizard. Table 4 of the Lizard paper does not settle its quality (section 3).
- **Output normalization and gate of GLA**. New weights, and the output no longer matches the teacher in stage 1. Liger does not use them.
- **`num_meta: 1`**. Equal to 4 sinks (P7). Keep 4, so that the checkpoints still load.

## 7. Next steps

This order replaces R2–R6 of the plan in document 13 (a proposal, not decided).

1. **Change the decision rule**. Compare stage 1 runs by ARC-Easy and PIQA, paired against config 1. Keep the loss only as a check that training worked. Skip the MMLU subset after stage 1.
2. **Run the cheap check of document 13 first** (A10, no training): R1b and config 1 with the softmax attention of the teacher in layer 15. It tests the most probable cause of the ARC-Easy loss of R1b.
3. **Add `gla_norm: convex`** with its tests, in one small PR (CPU only).
4. **Three stage 1 runs on config 1, one change each**, in this order: `window_rope`, `gla_norm: convex`, `gate_per_head`. Each takes approximately 2.5 hours on the H200. Keep a change only if ARC-Easy and PIQA do not fall.
5. **Stage 2 of the best config** (R7). Compare with stage 2 on config 1 (PIQA 67.7, ARC-Easy 54.9).
6. **Optional control**: stage 2 directly from the start values, without stage 1. Liger also trains in one stage. If it reaches the scores of stage 2 on config 1, stage 1 adds little, and the thesis can report this.

**On hold**: one feature map for each head (21.7× the parameters, and R1b cannot assign its effect) and `hybrid` (P9 and the ARC-Easy loss of R1b).

## Limits

- **Paper text not read**. The math above is the math of the official code at the stated commits. The papers can describe other variants or settings.
- **Liger window**. `sliding_window=64` in FlashAttention covers approximately 64 positions. The exact count depends on the FlashAttention version.
- **Proof audit**: self-review, not an independent audit. The checks use tiny layers and random inputs.
- **Rank correlations** use 5–6 runs with one seed each. They show only that the data does not support the loss as a selection rule.
