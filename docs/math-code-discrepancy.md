# Math against code: discrepancies

This document compares each formula in [math-formula.md](math-formula.md) with the code, side by side. It lists every difference. The code is `main` at commit `92010bc`. Three files contain the code:

- `src/model/linear_attention/lizard_attention.py` (in the tables: `lizard_attention.py`),
- `src/model/linear_attention/linear_attention.py` (the attention of the teacher in stage 1),
- `src/trainer/distill_attention_xent_mse.py` (the stage 1 loss).

Each difference has one of four types:

| Type | Meaning |
|---|---|
| Reading | The paper is not clear or not consistent. The code had to make a choice. |
| Numerical | The code calculates the same function with a different precision or with a guard against a numerical error. |
| Implementation | The code calculates the same function in a different way, or adds a detail that the paper does not describe. |
| Scope | The code does not implement the formula. |

## Summary

The possible effect is an estimate. No experiment measured it yet.

| # | Formula | Difference | Type | Possible effect |
|---|---|---|---|---|
| D1 | Gated linear attention | The code divides by a denominator in all forms. The recurrent form and the matrix form of the paper have no denominator. | Reading | Possibly large. Section 13.2 of the [gap analysis](11-gap-analysis.md) gives a mechanism. |
| D2 | Hedgehog feature map | One map for all heads of a layer. The paper does not say. The LoLCATs default has one map for each head. | Reading | Possibly large: 16,384 against 524,288 feature-map parameters for each layer |
| D3 | Hedgehog feature map | The code uses softmax. Section 4 writes exp. Table 13 gives softmax. | Reading | Unknown. The two forms give different kernels. |
| D4 | Hedgehog feature map | The code has 128 dimensions in $\mathbf{x}\mathbf{W}$ and 256 in $\phi(\mathbf{x})$. The paper gives "feature dimension 128" without more detail. | Reading | Unknown. If the paper means 128 in $\phi(\mathbf{x})$, the code has twice the features. |
| D5 | Window attention | The code adds $\sum_j \exp(t_j)$ to the denominator. The paper writes $\sum_j t_j$. | Reading | Small. The code reading agrees with the word "logit". |
| D6 | Hardware-aware algorithm | Section 4 uses one $\mathbf{W}$ for queries and keys. The code has separate $\phi_q$ and $\phi_k$, as in Section 3.1. | Reading | None. The code follows Section 3.1. |
| D7 | Hardware-aware algorithm | Not implemented. The code calculates the parallel form in float32 and keeps the full L × L matrix. | Scope | None on the function. Only speed and memory. |
| D8 | Gated linear attention | The denominator has a lower limit (`clamp_min(tiny)`). | Numerical | None, unless all weights of a row underflow |
| D9 | Gate | The code calculates the gate logit $\mathbf{W}_\gamma \mathbf{x}$ in bf16, and the sigmoid in float32. | Numerical | Small. It limits the resolution of γ near 1. |
| D10 | Target of stage 1 | The code calculates the teacher attention in bf16, with only the softmax in float32. | Numerical | Small noise in the target |
| D11 | Loss | The code uses 1000 × the mean over the layers of the mean squared error. The paper uses the mean over the layers of the summed squared error. | Implementation | None for Adam, except through eps (factor 8 of section 12 of the gap analysis) |
| D12 | Loss | The code compares the outputs of each head before `o_proj`. The paper says "attention output". | Reading | Unknown. Before `o_proj`, every head has the same weight in the loss. |
| D13 | Loss | The code rounds the prediction to bf16 before the loss. | Numerical | Small |
| D14 | Loss | Stage 1 also trains α. The loss of the paper lists only $\phi$, $\mathbf{W}_\gamma$ and $\mathbf{t}$. | Reading | Probably none. The paper calls α learnable. |
| D15 | Target of stage 1 | Each layer gets its input from the teacher (teacher forcing). The paper does not say. | Implementation | No effect on one layer. Errors do not add up over the layers during stage 1. |
| D16 | All | Grouped-query attention: the code repeats each key head and value head for 4 query heads. | Implementation | None |
| D17 | All | The code ignores padding masks. | Implementation | None now. The evaluation uses batch size 1, and the training packs sequences without padding. |

## 1. Sliding window attention

| | Paper | Code |
|---|---|---|
| Formula | Section 3.1, "Anchor Window Attention" | `awa` and `sink_softmax` (`lizard_attention.py`, lines 50–63) |
| Window | $t$ from $i - w + 1$ to $i$, thus $w$ positions with the current one | `window_mask`: `(t <= i) & (i - t < window)`. The same. |
| Scores | $\mathbf{q}_i^\top \mathbf{k}_t / \sqrt{d}$, no RoPE | `q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])`, no RoPE. The same. |
| Sink term in the denominator | $\sum_{j=0}^{m-1} t_j$ | $\sum_j \exp(t_j)$ (D5) |
| Values of the sinks | None | None. The same. |
| Sink logits | $\mathbf{t} \in \mathbb{R}^m$, $m = 4$ | `meta_tokens`, 4 scalars for each layer, shared by all heads. The same. |
| Overflow guard | – | `sink_softmax` subtracts the maximum of the row before `exp`. The result does not change. |

**D5, sink term**. With $\sum_j \exp(t_j)$, the sink mass is always positive. With the written $\sum_j t_j$, a negative sum could make the denominator small or negative. At the start, $t_j = 0$. Then the code adds a mass of 4, like 4 extra tokens with score 0. The written form adds 0.

## 2. Gated linear attention

### Parallel form

| | Paper | Code |
|---|---|---|
| Formula | Section 3.1, normalized | `gla` (`lizard_attention.py`, lines 44–47) |
| Kernel | $\phi_q(\mathbf{q}_i)^\top \phi_k(\mathbf{k}_t)$ | `fq @ fk.transpose(-1, -2)`. The same. |
| Gate product | $\prod_{l=t+1}^{i} \boldsymbol{\Gamma}_l$ | `gate_products` (lines 36–41): a reverse cumulative product, shifted by one position. The same. |
| Denominator | $\phi_q(\mathbf{q}_i)^\top \sum_j (\prod \boldsymbol{\Gamma}) \phi_k(\mathbf{k}_j)$ | `w.sum(-1, keepdim=True)`. The same, with a lower limit (D8). |
| Precision | Not given | float32 (`upcast` in `lizard`, line 202) |
| Memory | – | The full L × L weight matrix (D7) |

### Recurrent form

| | Paper | Code |
|---|---|---|
| Formula | $\mathbf{S}_i = \boldsymbol{\Gamma}_i \mathbf{S}_{i-1} + \phi_k(\mathbf{k}_i)\mathbf{v}_i^\top$, $\hat{\mathbf{y}}_i = \phi_q(\mathbf{q}_i)^\top \mathbf{S}_i$ | The decode loop of `lizard_recurrent` (lines 225–247) |
| State update | $\mathbf{S}_i = \boldsymbol{\Gamma}_i \mathbf{S}_{i-1} + \phi_k(\mathbf{k}_i)\mathbf{v}_i^\top$ | `kv_state` = γ · `kv_state` + `fk` vᵀ, with `einsum`. The same. |
| Denominator | None | A second state: `k_state` = γ · `k_state` + `fk`. The code divides the output by `fq` · `k_state` (D1). |
| Prefill | – | The parallel form, then the states from the last row of the gate products (lines 217–224) |

**D1, normalization**. The paper has three forms of the gated branch. Only the parallel form in Section 3.1 has a denominator. The recurrent form (Section 3.1) and the matrix form (Section 4) have none. The code uses a denominator in the parallel form and in the recurrent form. Thus the training and the generation of the code agree with each other. But they agree with only one of the three forms of the paper.

A normalized gated branch always gives a weighted mean of value vectors. It cannot make its output smaller. Section 13.2 of the [gap analysis](11-gap-analysis.md) explains why this can push the gate to 1. Step 5 of section 13.5 tests both forms.

## 3. Attention approximation

### Target of stage 1

| | Paper | Code |
|---|---|---|
| Formula | Softmax attention of the teacher with RoPE (Section 3.1) | `softmax_attention(q_rope, k_rope, v)` (`lizard_attention.py`, line 179, and `linear_attention.py`, lines 62–81) |
| RoPE | $\varphi_R$ on queries and keys | `apply_rotary_pos_emb` with the rotary embedding of the teacher. The same. |
| Causal mask | $j \le i$ | The upper triangle gets `-finfo.max`, not −∞. The result is the same. |
| Precision | Exact softmax | Scores in bf16. Softmax in float32. Weights rounded to bf16. Weights × values in bf16 (D10). |
| Input of the next layer | – | The target output `y_true` (D15) |

**D10, precision of the target**. The code rounds the scores $\mathbf{q}^\top\mathbf{k}/\sqrt{d}$ to bf16 before the softmax. For a score $s$, the rounding error is up to approximately 0.4% of $s$. For $s = 10$, each attention weight can change by up to approximately 4%. For the teacher at inference, the model config selects FlashAttention-2, which calculates the scores in float32. Thus the target of stage 1 is a little different from the real output of the teacher.

### Output of Lizard attention

| | Paper | Code |
|---|---|---|
| Formula | $\hat{\mathbf{Y}}_{lizard} = \hat{\mathbf{Y}}_{gate} + \alpha \cdot \hat{\mathbf{Y}}_{anchor}$ | `gla(...) + alpha * awa(...)` (line 204). The same. |
| α | Learnable | `alpha_blend`, one scalar for each layer, start value 1 |
| RoPE in the Lizard attention | None | None. The same. |

## 4. Loss function

| | Paper | Code |
|---|---|---|
| Formula | $\frac{1}{N} \sum_{l=1}^{N} \Vert \mathbf{Y}^l_{\mathrm{softmax}} - \hat{\mathbf{Y}}^l_{lizard} \Vert_F^2$ | `loss_mse / n_layers * mse_factor`, with `nn.MSELoss(reduction='mean')` for each layer (`distill_attention_xent_mse.py`, lines 29, 66–73) |
| Reduction in a layer | Sum of the squared errors | Mean of the squared errors, then × 1000 (D11) |
| Mean over the layers | $1/N$ | `/ n_layers`. The same. |
| Compared tensors | "Attention output" | The output of each head before `o_proj`, shape (batch, heads, length, head dimension) (D12) |
| Prediction | – | Rounded to bf16 before the loss (`y.to(dtype)`, line 205) (D13) |
| Trained parameters | $\phi$, $\mathbf{W}_\gamma$, $\mathbf{t}$ | `phi_q`, `phi_k`, `W_gamma`, `meta_tokens` and `alpha_blend` (D14) |

**D11, scale**. In a layer, the code divides by the number of elements (batch × 32 heads × 2048 positions × 64) and multiplies by 1000. Thus the code loss is a constant multiple of the loss of the paper. Adam does not change with a constant factor on the loss, except through eps = 1e-8.

## 5. Hardware-aware algorithm

| | Paper | Code |
|---|---|---|
| Matrix form | $((\phi(\mathbf{Q}) \odot \mathbf{C})(\phi(\mathbf{K}) / \mathbf{C})^\top \odot \mathbf{M})\mathbf{V}$, no denominator | Not used (D7). `gla` multiplies the kernel matrix by the gate products directly. |
| Cumulative products | $\mathbf{c}_t = \prod_{j=1}^{t} \boldsymbol{\Gamma}_j$, in log space | Products $\prod_{l=t+1}^{i}$ for each pair of positions, in float32 |
| Feature maps | $\exp(\pm\mathbf{Q}\mathbf{W} + \log\mathbf{C})$ and $\exp(\pm\mathbf{K}\mathbf{W} - \log\mathbf{C})$, one $\mathbf{W}$ (D6) | Separate `phi_q` and `phi_k`, softmax activation |
| Precision | bf16 with Tensor Cores | float32 |

**D7.** The code does not need the reparameterization, because it calculates in float32. In float32, a gate product underflows only below approximately 1e-38. Then the weight of that token becomes 0, which is approximately its true value. The jku-thesis file `kernels/gla_lizard_reparam.py` implements the reparameterization as an experiment.

## 6. Causal softmax attention

| | Paper | Code |
|---|---|---|
| Formula | Section 2, without RoPE | Not used alone. The target of stage 1 (section 3 of this document) is this formula with RoPE. |

The code has no difference here, except D10.

## 7. Hedgehog feature map

| | Paper | Code |
|---|---|---|
| Formula | Section 4: $[\exp(\mathbf{x}\mathbf{W}) \oplus \exp(-\mathbf{x}\mathbf{W})]$. Table 13: activation softmax. | `hedgehog` (lines 25–27): `cat([softmax(xW), softmax(-xW)])` (D3) |
| Weights | $\mathbf{W}$, no bias | `phi_q.weight` and `phi_k.weight`, no bias. The same. |
| Dimension | "Hedgehog Feature Dimension 128" (Table 13) | `Linear(64, 128)`: 128 in $\mathbf{x}\mathbf{W}$, 256 in $\phi(\mathbf{x})$ (D4) |
| Heads | Not given | One map for all 32 query heads, and one for the keys (D2) |
| Separate maps for queries and keys | $\phi_q$, $\phi_k$ (Section 3.1) | `phi_q`, `phi_k`. The same. |
| Start values | Not given | Normal distribution, standard deviation 0.02 |

**D2, shared maps**. Each layer has one map from 64 to 128 dimensions for $\phi_q$ and one for $\phi_k$: 16,384 parameters. The 32 query heads use the same $\phi_q$. The 8 key heads use the same $\phi_k$. Appendix B of the paper says that the other design choices follow the defaults of LoLCATs. The LoLCATs default (`untied_head_einsum`) has one map for each head. At feature dimension 128, that is 524,288 parameters for each layer.

**D3, softmax against exp**. Each softmax divides the exponentials by their sum for one token. The two halves have different sums. Thus the softmax form does not only scale the exp form. It gives a different kernel, also after the normalization of the gated branch.

**D4, dimension**. The LoLCATs configs use `feature_dim` for the dimension of $\mathbf{x}\mathbf{W}$. The code follows this convention. The paper does not say which convention it uses.

## Gate

The gate occurs in sections 2 and 5 of [math-formula.md](math-formula.md).

| | Paper | Code |
|---|---|---|
| Formula | $\boldsymbol{\Gamma}_i = \gamma_i \mathbf{1}_d^\top$, $\gamma_i = \sigma(\mathbf{W}_\gamma \mathbf{x}_i)$, $\mathbf{W}_\gamma \in \mathbb{R}^{d \times 1}$ (Section 5) | `torch.sigmoid(*upcast(self.W_gamma(x)))` (line 170) |
| Input $\mathbf{x}_i$ | The input of the layer | `hidden_states`, the input of the attention module, after the RMSNorm of the decoder layer |
| Shape | One scalar for each position | Shape (batch, length), the same value for all heads. The same. |
| Bias | None | `bias=False`. The same. |
| Start value | Not given | $\mathbf{W}_\gamma = 0$, thus γ = 0.5 |
| Precision | Not given | Logit in bf16, sigmoid in float32 (D9) |

**D9, gate precision**. Near γ = 1, $1 - \gamma \approx e^{-z}$ for the logit $z$. Between 8 and 16, the step between two bf16 values is 0.0625. Thus $1 - \gamma$ can change only in steps of approximately 6%. This limits how exactly the gate can approach 1.

## What agrees

These parts agree with the paper:

- the window of $w$ positions, the scale $1/\sqrt{d}$, and no RoPE in the Lizard attention,
- the range of the gate products, $l$ from $t + 1$ to $i$,
- one scalar gate for each position, shared by all heads, from $\mathbf{W}_\gamma \in \mathbb{R}^{d \times 1}$ without bias,
- separate feature maps for queries and keys,
- the sum of the two branches with α on the window branch,
- the RoPE of the target in stage 1,
- the mean over the layers in the loss.
