# 15. Attention math side by side: Lizard, LoLCATs, GLA and Liger

**Status:** One table compares the attention math of four methods, part by part. A Mathbox proof audit checks eight claims about the differences, by derivation. No new training, no code.

Lizard is the paper of this thesis, and LoLCATs is the code base of this repository. Lizard compares its gate with the gate of GLA. Liger linearizes a model with GLA and a window. [Document 14](14-liger-gla.md) has the proposed changes for Lizard v2.

## Sources

Mathbox skill `literature-check`. This check compared each method with its paper text and with code.

| Method | Paper read | Code read | Verdict |
|---|---|---|---|
| Lizard | arXiv:2507.09025, version 4 (18 Apr 2026), from images of the paper ([math formulas](math-formula.md)) | `lizard_attention.py` (v1), `lizard_attention_v2.py` | Verified. The paper leaves three points open: D1, D3, and if α is a learned parameter ([document 13](13-lizard-attention-v2.md)). |
| LoLCATs | Preprint v0 ([`lolcats_preprint_v0.pdf`](../lolcats_preprint_v0.pdf), 2024-10-14), Sections 2, 3.1 and 3.3 | `linear_window_attention_sw.py`, config `distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml` | Verified. The code makes γ one value for each head, σ(a), with the start value 0.1. |
| GLA | ICML 2024 version, Sections 2.1, 4.1 and 4.4, Table 1 | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), commit `3e52d5a` | Verified. The code adds the query scale 1/√d and uses RMSNorm, where the paper uses LayerNorm. |
| Liger | ICML 2025 version, Sections 3.1–3.3, Table 1 | [OpenSparseLLMs/Linearization](https://github.com/OpenSparseLLMs/Linearization), commit `0b364eb` | Verified. The code adds the gate exponent 1/16 and RoPE in the window. The paper writes neither. |

## Notation

The papers use α and γ for different things. Thus the table uses neutral symbols.

| Symbol | Meaning |
|---|---|
| $i$, $j$, $d$, $D$ | Query position, key position, head dimension, hidden size |
| $q_i$, $k_j$, $v_j$ | Query, key and value after the projections |
| $R(\cdot)$ | RoPE |
| $\phi$ | Feature map of the linear branch |
| $g_i$ | Gate at position $i$. $c_{ij} = \prod_{s=j+1}^{i} g_s$ for a scalar gate, $b_i = \prod_{s \le i} g_s$ for a gate vector. |
| $W_i$ | Window positions $i - w + 1, \ldots, i$ |
| $s_{ij}$ | Window score |
| $u_i$, $e_{ij}$ | Largest window score $u_i = \max_{j \in W_i} s_{ij}$, and $e_{ij} = e^{s_{ij} - u_i}$ |
| $\kappa_{ij}$ | Kernel of the linear branch, $\phi(q_i)^\top \phi(k_j)$ |
| $t_m$ | Sink logits, $m = 1, \ldots, 4$ |
| $a$ | Weight of the window branch: α in Lizard, γ in LoLCATs, β in Liger |
| $\rho_i$ | Sum of the window weights of Lizard (with sinks), at most 1 |
| $\mu_i$ | Sum of the weights of the linear branch of Liger, $\sum_{j \le i} \phi(q_i) \cdot (b_i / b_j) \cdot \phi(k_j)$ |

## Side by side

| Part | Lizard | LoLCATs | GLA | Liger |
|---|---|---|---|---|
| Source | Section 3.1, Section 4, Table 13 | Section 2, Section 3.1, Eq. 7 | Section 2.1, Eq. 3 and 4, Section 4.4 | Eq. 6, 7, 9 and 10 |
| Feature map $\phi(x)$ | $[\mathrm{softmax}(xW) \oplus \mathrm{softmax}(-xW)]$, learned $W$. Section 4 writes exp. | $\mathrm{softmax}(R(x)\tilde W + \tilde b)$, learned, one map for each head | None: $\phi(x) = x$ | $\mathrm{softmax}(x)$ over the head dimension, no weights |
| Gate $g_i$ | $\sigma(W_\gamma x_i)$, one scalar for all heads | None: $g_i = 1$ | $\sigma(x_i W_1 W_2 + b)^{1/16} \in (0,1)^{d_k}$ | Paper: $\sigma(\mathrm{Pooling}(k_i))$. Code: $\sigma(k_i)^{1/16}$. |
| Gate at logit 0 | 0.5 | – | 0.958 | 0.958 (code) |
| Linear branch | $\hat y^{lin}_i = \dfrac{\sum_{j \le i} c_{ij}\, \kappa_{ij}\, v_j}{\sum_{j \le i} c_{ij}\, \kappa_{ij}}$ (parallel form) | $\sum_{j \le i-w} \kappa_{ij}\, v_j$, in a shared fraction (see Output) | $S_i = \mathrm{Diag}(g_i)\, S_{i-1} + k_i^\top v_i$, $o_i = q_i S_i$ | $S_i = \mathrm{Diag}(g_i)\, S_{i-1} + \phi(k_i)^\top v_i$, $\hat y^{lin}_i = \phi(q_i)\, S_i$ |
| Tokens in the linear branch | All $j \le i$ | Only $j \le i - w$ | All $j \le i$ | All $j \le i$ |
| Denominator of the linear branch | Its own row sum. The recurrent form of the paper has none (D1). | Shared with the window | None | None |
| Matrix form | $((\phi(Q) \odot C)(\phi(K) / C)^\top \odot M)\, V$ (Section 4, no denominator) | – | $((Q \odot B)(K / B)^\top \odot M)\, V$, in log space (Eq. 4) | The GLA form, with $\phi(Q)$ and $\phi(K)$ |
| Window scores $s_{ij}$ | $q_i^\top k_j / \sqrt{d}$, no RoPE | $R(q_i)^\top R(k_j) / \sqrt{d}$ | – | Paper: $q_i^\top k_j / \sqrt{d}$. Code: with RoPE. |
| Window size $w$ | 128 | 64 | – | 64 |
| Window branch | $\hat y^{win}_i = \dfrac{\sum_{j \in W_i} e^{s_{ij}}\, v_j}{\sum_m e^{t_m} + \sum_{j \in W_i} e^{s_{ij}}}$ | $\sum_{j \in W_i} a\, e_{ij}\, v_j$, in a shared fraction | – | $\hat y^{win}_i = \dfrac{\sum_{j \in W_i} e^{s_{ij}}\, v_j}{\sum_{j \in W_i} e^{s_{ij}}}$ |
| Sinks | $\sum_{m=1}^{4} e^{t_m}$ (the paper writes $\sum_m t_m$) | None | – | None |
| Output $y_i$ | $\hat y^{lin}_i + a\, \hat y^{win}_i$ | $\dfrac{\sum_{W_i} a\, e_{ij} v_j + \sum_{j \le i-w} \kappa_{ij} v_j}{\sum_{W_i} a\, e_{ij} + \sum_{j \le i-w} \kappa_{ij}}$ | $(\mathrm{LN}(o_i) \odot \mathrm{swish}(x_i W_r))\, W_O$ | $0.5\, \hat y^{lin}_i + 0.5\, \hat y^{win}_i$ (Eq. 9 with $\alpha + \beta = 1$) |
| Window weight $a$ | α, learned, start value 1 | $\sigma(a_h)$ for each head, start value 0.1 (code) | – | 0.5, constant |
| Total weight of row $i$ | $1 + a\rho_i$ | 1 | – | $0.5 + 0.5\mu_i$ |
| Training loss | Stage 1: MSE of each layer. Stage 2: next-token cross-entropy with LoRA. | The same (Eq. 5 and 6, then cross-entropy) | Next-token cross-entropy (pretraining) | Next-token cross-entropy with LoRA, one stage (Eq. 8) |
| Trained parts | $\phi$, $W_\gamma$, $t_m$, α, then LoRA on q, k, v | $\phi$, $a$, then LoRA on q, k, v, o | All weights | LoRA on q, k, v |

## Proof audit

Mathbox skill `proof-audit`, self-review of the formulas in the table above. All evidence is a derivation. The audit proves five claims, and three claims are correct after stated restrictions.

| Claim | Verdict | Derivation |
|---|---|---|
| **A**. LoLCATs equals `hybrid` of Lizard v2 with $g = 1$, no sinks, and linear terms only for $j \le i - w$ | Correct only after these three restrictions | Without sinks, `hybrid` gives $y_i = \frac{\sum_{j \le i} c_{ij} \kappa_{ij} v_j + a \sum_{W_i} e_{ij} v_j}{\sum_{j \le i} c_{ij} \kappa_{ij} + a \sum_{W_i} e_{ij}}$. Set $c_{ij} = 1$, and keep only $j \le i - w$ in the linear sums. The result is the LoLCATs output, term by term. |
| **B**. In Lizard and Liger, a token inside the window gets weight from both branches | Proved as written | Their linear sums use all $j \le i$, and $W_i$ is part of $j \le i$. In LoLCATs, the positions $j \le i - w$ and $W_i$ have no common element. |
| **C**. The total weight of row $i$ in Lizard is $1 + a\rho_i$ | Proved as written ($a \ge 0$) | The weights of $\hat y^{lin}_i$ sum to 1. The weights of $\hat y^{win}_i$ sum to $\rho_i = \frac{\sum_{W_i} e^{s_{ij}}}{\sum_m e^{t_m} + \sum_{W_i} e^{s_{ij}}} \le 1$, then the factor $a$. |
| **D**. The total weight of row $i$ in Liger is $0.5 + 0.5\mu_i$, with $\mu_i \le 1 / (1 - g_{max})$ | Proved as written (all gates at most $g_{max} < 1$) | The window weights sum to 1. Each linear term is $\sum_d \phi(q_i)_d (b_i/b_j)_d\, \phi(k_j)_d \le \max_d (b_i/b_j)_d \le g_{max}^{\,i-j}$, because $\phi(k_j)_d \le 1$ and $\phi(q_i)$ sums to 1. The sum over $j$ is a geometric series. |
| **E**. The parallel form of GLA equals its recurrent form. With a scalar gate and the features $\phi$, it is the Section 4 form of Lizard. | Proved as written (gates above 0) | Eq. 3 gives $S_i = \sum_{j \le i} \mathrm{Diag}(b_i / b_j)\, k_j^\top v_j$. Thus $q_i S_i = \sum_j (q_i \odot b_i) \cdot (k_j / b_j)\, v_j$, the row $i$ of the matrix form. With a scalar gate, $b_i / b_j = c_{ij}$. |
| **F**. The window of Lizard cannot give exact previous-token attention in layer 0 | Correct only after a restriction: inputs without position information | Without RoPE, $s_{ij}$ depends only on $x_i$ and $x_j$. Llama has no position embeddings, so layer 0 gets token embeddings only. If $x_{i-1} = x_{i-2}$, both positions get the same weight. Previous-token attention needs all weight on $i - 1$. |
| **G**. The window of Liger gives the weights of the teacher inside the window, divided by their sum over the window | Correct only after a restriction: RoPE as in the code, and the projections of the teacher (before LoRA) | With the scores of the teacher, each weight is $p_{ij} / \sum_{j' \in W_i} p_{ij'}$, with $p$ the weights of the teacher. |
| **H**. The 4 sinks of Lizard act as one sink with the logit $t + \ln 4$ | Proved as written (equal start values) | The sinks enter only through $\sum_m e^{t_m}$. Thus equal logits get equal gradients and equal Adam updates. Then $4 e^{t} = e^{t + \ln 4}$. |

**Result**. Only LoLCATs gives an exact weighted mean of the values (claims C and D). Only the window of Lizard has no RoPE and has sinks (claims F and H). Lizard and Liger count the window tokens twice, and LoLCATs does not (claim B).
