# 15. Attention math side by side: Lizard, LoLCATs, GLA and Liger

**Status:** A comparison of the attention math of four methods, part by part, with a proof audit of seven claims about their differences. No new training. A version for comments is in a [shared doc](https://claude.ai/code/artifact/e2239248-6bda-40e9-9e65-7e66d5abaf72).

The four methods: Lizard is the paper of this thesis, and LoLCATs is the code base of this repository. Lizard compares its gate with the gate of GLA. Liger linearizes a model with GLA and a window. [Document 14](14-liger-gla.md) has the full math of GLA and Liger, and the proposed changes.

## Sources and checks

Mathbox skill `literature-check`. This check compared each method with its paper text and with code.

| Method | Paper read | Code read | Verdict |
|---|---|---|---|
| Lizard | arXiv:2507.09025, version 4 (18 Apr 2026), from images of the paper ([math formulas](math-formula.md)) | `lizard_attention.py` (v1) and `lizard_attention_v2.py` in this repository | Verified, except three points that the paper leaves open ([document 13](13-lizard-attention-v2.md)). They are the denominator of the gated branch (D1), exp or softmax features (D3), and if the model learns α. |
| LoLCATs | Preprint v0 ([`lolcats_preprint_v0.pdf`](../lolcats_preprint_v0.pdf), 47 pages, 2024-10-14), Sections 2, 3.1 and 3.3, Appendix A | `linear_window_attention_sw.py`, config `distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml` | Verified. The code makes the γ of the paper one value for each head, σ(a), with the start value 0.1. |
| GLA | ICML 2024 version (23 pages), Sections 2, 4.1 and 4.4, Table 1 | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), commit `3e52d5a` | Verified. The code adds the query scale 1/√d and uses RMSNorm, where the paper uses LayerNorm. |
| Liger | ICML 2025 version (15 pages), Sections 2–4, Tables 1, 2, 4 and 6 | [OpenSparseLLMs/Linearization](https://github.com/OpenSparseLLMs/Linearization), commit `0b364eb` | Verified. The code adds the gate exponent 1/16 and RoPE in the window. The paper writes neither. |

The paper texts come from a text extraction of the PDFs. A check script compares LoLCATs Eq. 7 with its code (claim A below).

## Notation

The four papers use α and γ for different things. Thus this document uses one neutral set of symbols. One head, query position $i$, key position $j$, head dimension $d$, hidden size $D$.

| This document | Meaning | Lizard | LoLCATs | GLA | Liger |
|---|---|---|---|---|---|
| $q_i, k_j, v_j$ | Query, key and value after the projections | The same | The same | The same | The same |
| $R(\cdot)$ | RoPE | $\varphi_R$ | RoPE | – | – |
| $\phi$ | Feature map of the linear branch | $\phi_q, \phi_k$ | $\phi_q, \phi_k$ | None | $\phi$ |
| $g_i$ | Gate (decay) at position $i$ | $\Gamma_i = \gamma_i$ | None | $\alpha_t$ | $G_t$, from $\alpha_t$ |
| $b_i$ | Cumulative gate $g_1 \cdots g_i$ | $C$ | – | $b_t$, $B$ | – |
| $w$ | Window size | 128 | 64 | – | 64 |
| $t_m$ | Sink logits ($m = 1 \ldots 4$) | $t_j$ | – | – | – |
| $a$ | Weight of the window branch | $\alpha$ | $\gamma$ | – | $\beta$ ($\alpha$ for the linear branch) |
| $s_{ij}$ | Window score $q_i \cdot k_j / \sqrt{d}$ | The same | The same | – | The same |

Thus the γ of Lizard is a gate, but the γ of LoLCATs is a window weight. The α of GLA is a gate, but the α of Lizard and Liger is a branch weight.

## Feature maps

Lizard and LoLCATs learn a Hedgehog map. GLA uses no map. Liger uses a softmax without weights.

```math
\begin{aligned}
\text{Lizard (Table 13):}\quad & \phi(x) = \left[\mathrm{softmax}(xW) \oplus \mathrm{softmax}(-xW)\right] \quad \text{(Section 4 writes exp)}\\
\text{LoLCATs (Section 3.1):}\quad & \phi(x) = f\left(R(x)\,\tilde W + \tilde b\right),\ f = \text{softmax over the features}\\
\text{GLA (Section 2.1):}\quad & \phi(x) = x\\
\text{Liger (Section 3.2, Eq. 7):}\quad & \phi(x) = \mathrm{softmax}(x) \text{ over the head dimension}
\end{aligned}
```

| | Lizard | LoLCATs | GLA | Liger |
|---|---|---|---|---|
| Learned weights | Yes, $W$ ($d \times 32$ in this project) | Yes, $\tilde W$ ($d \times 64$) | No | No |
| One map for each head | Not clear in the paper (D2). v1 shares one map, v2 has an option. | Yes (`untied_head_einsum`) | – | – |
| After RoPE | No | Yes | No RoPE | No (the code applies the map before RoPE) |
| Upper limit of $\phi(q) \cdot \phi(k)$ | 2 (two softmax halves) | 1 for each softmax | None | 1 |
| Features for each head | 2 × 32 = 64 | 64 | $d_k = D/2$ divided by the heads | 64 |

In the ablation of Liger (Llama-3-8B, Table 6), a learned feature map was much worse in one-stage training: perplexity 9.04 against 2.96.

## Gates

Lizard has one scalar gate for each position, shared by all heads, with the start value 0.5. GLA and Liger have a gate for each head and key dimension, with the start value 0.958. LoLCATs has no gate.

```math
\begin{aligned}
\text{Lizard:}\quad & g_i = \sigma(W_\gamma x_i) \in (0,1), \quad W_\gamma \in \mathbb{R}^{D \times 1}\\
\text{LoLCATs:}\quad & g_i = 1\\
\text{GLA:}\quad & g_i = \sigma\left(x_i W_1 W_2 + b\right)^{1/16} \in (0,1)^{d_k}, \quad W_1 \in \mathbb{R}^{D \times 16},\ W_2 \in \mathbb{R}^{16 \times d_k}\\
\text{Liger:}\quad & g_i = \sigma(k_i)^{1/16} \in (0,1)^{d} \quad \text{(code. The paper writes } \sigma(\mathrm{Pooling}(k_i))\text{)}
\end{aligned}
```

| | Lizard | LoLCATs | GLA | Liger |
|---|---|---|---|---|
| Shape | One scalar for each position | – | One value for each head and key dimension | One value for each head and key dimension |
| Shared by the heads | Yes | – | No | No |
| New parameters for each layer | $D$ (2,048 here) | 0 | $16(D + d_k) + d_k$ | 0 |
| Start value (logit 0) | 0.5 | 1 | 0.958 | 0.958 |
| Logit for $g = 0.128$ | −1.92 | – | −32.9 | −32.9 |

The exponent 1/16 is the main difference. GLA calls it a temperature term "to encourage model to have a slower forgetting rate". Table 4 of Lizard writes the GLA gate and the pooled gate without it. Thus that ablation possibly does not test these gates as GLA and Liger use them. In R1b of this project, layer 15 learned $g \approx 0.128$. This value is easy for the gate of Lizard, but almost impossible to reach with the exponent.

## Linear branch

Lizard normalizes its linear branch by a row sum and uses all earlier tokens. LoLCATs uses only the tokens before the window. GLA and Liger have no denominator.

```math
\begin{aligned}
\text{Lizard (Section 3.1, parallel form):}\quad & \hat y^{lin}_i = \frac{\sum_{j \le i} \left(\prod_{s=j+1}^{i} g_s\right) \phi(q_i)^\top \phi(k_j)\, v_j}{\sum_{j \le i} \left(\prod_{s=j+1}^{i} g_s\right) \phi(q_i)^\top \phi(k_j)}\\
\text{LoLCATs (Eq. 7, linear terms):}\quad & \phi(q_i)^\top \sum_{j \le i-w} \phi(k_j)\, v_j^\top \quad \text{(the denominator is shared with the window)}\\
\text{GLA (Eq. 3, Section 4.1):}\quad & S_i = \mathrm{Diag}(g_i)\, S_{i-1} + k_i^\top v_i,\quad o_i = q_i S_i,\quad O = \left(\left((Q \odot B)(K / B)^\top\right) \odot M\right) V\\
\text{Liger (Eq. 7):}\quad & S_i = \mathrm{Diag}(g_i)\, S_{i-1} + \phi(k_i)^\top v_i,\quad \hat y^{lin}_i = \phi(q_i)\, S_i
\end{aligned}
```

| | Lizard | LoLCATs | GLA | Liger |
|---|---|---|---|---|
| Tokens used | All $j \le i$, also the window tokens | Only $j \le i - w$ | All $j \le i$ | All $j \le i$, also the window tokens |
| Denominator | Its own row sum. The recurrent form of the paper has none (D1). | Shared with the window | None | None |
| Total weight of the branch | 1 | – (one shared sum) | Not limited | Not limited. Approximately 0.3 at random inputs (claim D′). |
| Recurrent state for each head | $S$ and $z$ | $S$ and $z$ | $S$ only | $S$ only |
| Training form | Dense in float32 here. The paper: one GEMM in log space (Section 4). | Dense, or the TK kernel | Chunks in log space (Section 4.2) | `fused_chunk_gla` |

GLA and Lizard use the same log-space form: the weight $q_i \cdot (b_i/b_j) \cdot k_j$ becomes $\exp(\log b_i - \log b_j)$ in the exponent. The GLA paper calls this term "a data-dependent relative position factor". With a softmax feature map, the Section 4 form of Lizard loses the gate ([document 13](13-lizard-attention-v2.md), P3).

## Window branch

Only the window of Lizard has no RoPE, and only Lizard has sinks. LoLCATs and Liger use the RoPE scores of the teacher over 64 tokens.

```math
\begin{aligned}
\text{Lizard (Section 3.1):}\quad & \hat y^{win}_i = \frac{\sum_{j=i-w+1}^{i} e^{s_{ij}}\, v_j}{\sum_{m} e^{t_m} + \sum_{j=i-w+1}^{i} e^{s_{ij}}},\quad s_{ij} = q_i^\top k_j / \sqrt{d},\ w = 128\\
\text{LoLCATs (Eq. 7, window terms):}\quad & \sum_{j=i-w+1}^{i} a\, e^{s_{ij} - c_i}\, v_j,\quad s_{ij} = R(q_i)^\top R(k_j) / \sqrt{d},\ c_i = \max_j s_{ij},\ w = 64\\
\text{Liger (Eq. 10):}\quad & \hat y^{win}_i = \frac{\sum_{j=i-w+1}^{i} e^{s_{ij}}\, v_j}{\sum_{j=i-w+1}^{i} e^{s_{ij}}},\quad s_{ij} = R(q_i)^\top R(k_j) / \sqrt{d} \text{ (RoPE in the code)},\ w = 64
\end{aligned}
```

| | Lizard | LoLCATs | GLA | Liger |
|---|---|---|---|---|
| Window size | 128 | 64 | – | 64 |
| RoPE in the scores | No | Yes | – | Yes (code) |
| Sinks | 4 logits, always equal: one sink with logit $t + \ln 4$ ([document 14](14-liger-gla.md), P7) | No | – | No |
| Own normalization | Yes, with sinks: the weights sum to $\rho_i \le 1$ | No, shared with the linear terms | – | Yes: the weights sum to 1 |
| Teacher weights inside the window | Not exactly: without RoPE, layer 0 cannot find the previous token (document 14, P10a) | Yes, with the shared denominator | – | Yes, divided by their sum over the window (before LoRA) |

The paper writes the sink term as a sum of $t_m$. The code uses $\exp(t_m)$, which keeps the denominator positive (document 13, P4).

## Mixing the branches

Only LoLCATs gives an exact weighted mean of the values. The sum of Lizard is larger than a weighted mean, and the mix of Liger is smaller.

```math
\begin{aligned}
\text{Lizard (Section 3.1):}\quad & y_i = \hat y^{lin}_i + a\, \hat y^{win}_i \quad \text{total weight } 1 + a\rho_i\\
\text{LoLCATs (Eq. 7):}\quad & y_i = \frac{\sum_{j \in W_i} a\, e^{s_{ij} - c_i} v_j + \sum_{j \le i-w} \phi(q_i)^\top \phi(k_j)\, v_j}{\sum_{j \in W_i} a\, e^{s_{ij} - c_i} + \sum_{j \le i-w} \phi(q_i)^\top \phi(k_j)} \quad \text{total weight } 1\\
\text{GLA (Section 4.4):}\quad & y_i = \left(\mathrm{LN}(o_i) \odot \mathrm{swish}(x_i W_r)\right) W_O \quad \text{one branch}\\
\text{Liger (Eq. 9):}\quad & y_i = 0.5\, \hat y^{lin}_i + 0.5\, \hat y^{win}_i \quad \text{total weight } 0.5 + 0.5\,\mu_i
\end{aligned}
```

| | Lizard | LoLCATs | GLA | Liger |
|---|---|---|---|---|
| Window weight $a$ | α, learned, one for each layer, start value 1 | σ($a_h$), one for each head, start value 0.1 | – | 0.5, constant |
| Weighted mean of the values | No, total weight $1 + a\rho_i > 1$ | Yes | – | No, total weight $0.5 + 0.5\mu_i$ (approximately 0.66 at random inputs) |
| Window share depends on the position | No | Yes, the linear sum grows with $i$ | – | Yes, $\mu_i$ grows with the memory |
| Matching option of Lizard v2 | `row` (default) | `hybrid` (but v2 also has a gate and the overlap) | None | `convex` (proposed in document 14) at $a = 1$ |

$\mu_i = \sum_j \phi(q_i) \cdot (b_i/b_j) \cdot \phi(k_j)$ is the total weight of the linear branch of Liger. Liger states that branch weights with the sum 1 are "particularly critical for linearization". But its linear branch has no denominator. Thus the sum of the branch weights sets only the balance at the start.

## Training

Lizard and LoLCATs train in two stages: attention matching, then LoRA. Liger trains in one stage, with LoRA only. GLA trains from scratch.

```math
\begin{aligned}
\text{Stage 1 (Lizard, LoLCATs Eq. 5 and 6):}\quad & \ell_{MSE} = \frac{1}{MH} \sum_{m=1}^{M} \sum_{h=1}^{H} \frac{1}{d} \sum_{n} \left(y_n - \hat y_n\right)^2 \quad \text{(each layer gets the input of the teacher)}\\
\text{Stage 2 (Lizard, LoLCATs) and Liger (Eq. 8):}\quad & \ell_{xent} = -\sum_t \log P_\Theta(u_{t+1} \mid u_{1:t}),\quad \Theta = \Theta_0 + BA \text{ (LoRA)}
\end{aligned}
```

| | Lizard | LoLCATs | GLA | Liger |
|---|---|---|---|---|
| Stages | 2: MSE, then LoRA | 2: MSE, then LoRA | Pretraining | 1: LoRA with the next-token loss |
| Stage 1 trains | φ, gate, sinks, α | φ, window weight | – | – |
| LoRA | q, k, v (Table 13) | q, k, v, o, rank 8, scale 2 | – | q, k, v, rank 8, alpha 8 |
| Learning rate | Stage 1 1e-3, stage 2 5e-4 (Table 13) | Stage 1 1e-2, stage 2 1e-4 | – | 1e-3 |
| Data | Alpaca | Alpaca, sequences of 1,024 tokens, batch 8 | SlimPajama, 15B or 100B tokens | Alpaca, 50,000 samples × 2 epochs (approximately 0.02B tokens) |
| Llama-3.2-1B, mean of 6 tasks / without MMLU | – | 51.1 / 56.7 (Liger, Table 4) | 46.9 / 51.1 (GLA-1B, Liger, Table 4) | 52.9 / 59.0 (teacher 55.1 / 59.9) |

Table 4 of Liger does not list its tasks. Probably they are the 6 tasks of its Table 2. In this project, the stage 1 loss did not predict the accuracy (document 14: Spearman +0.31 for ARC-Easy over 6 runs).

## Proof audit

Mathbox skill `proof-audit`, self-review of `lizard_attention_v2.py` at commit `3b13620` and of the paper formulas above. The audit proves four claims. Two claims are correct after stated restrictions, and one is a computation in a stated range.

| Claim | Verdict | Evidence |
|---|---|---|
| **A**. LoLCATs Eq. 7 is the `hybrid` form of Lizard v2 with $g = 1$, no sinks, and linear terms only for $j \le i - w$ | Correct only after these three restrictions | The same shared denominator, and window terms $a \cdot \exp(s - \max s)$. Check: Eq. 7 against the LoLCATs code, relative error 1.2e-7 (float32). |
| **B**. In Lizard and Liger, a token inside the window gets weight from both branches | Proved as written | Their linear sums use all $j \le i$. LoLCATs removes $j > i - w$ from its linear branch. |
| **C**. The row form of Lizard has the total weight $1 + a\rho_i > 1$ | Proved as written ($a \ge 0$) | The linear weights sum to 1, the window weights to $a\rho_i$. Check: error 4.4e-16 (document 14, P8a). |
| **D**. The mix of Liger has the total weight $0.5 + 0.5\mu_i$, with $\mu_i \le 1 / (1 - g_{max})$ | Proved as written | Each term $\phi(q_i) \cdot (b_i/b_j) \cdot \phi(k_j)$ is at most the largest $(b_i/b_j)_d$, because each $\phi(k)_d \le 1$ and $\phi(q)$ sums to 1. The gates give a geometric series. |
| **D′**. At random inputs, $\mu_i$ is approximately 0.18 at position 16, and 0.31–0.32 from position 64 to 512 | Computed in this range only | 4 heads, $d = 64$, gate $\sigma(k)^{1/16}$. Trained models can differ. |
| **E**. The parallel form of GLA equals its recurrent form. With a scalar gate, it is the Section 4 form of Lizard. | Proved as written (gates above 0) | $(q_i \odot b_i) \cdot (k_j / b_j) = \sum_d q_{id}\, k_{jd} \prod g$. Check: error 2.3e-16 (document 14, P11). |
| **F**. The window of Lizard without RoPE cannot give exact previous-token attention in layer 0 | Correct only after a restriction: inputs without position information | Two equal tokens at $i - 1$ and $i - 2$ get equal weight. Check: difference 0 without RoPE, 0.329 with RoPE (document 14, P10a). |

[`attention-side-by-side/check_side_by_side.py`](attention-side-by-side/check_side_by_side.py) checks A and computes D′, on CPU with the code of this repository. Output: [`check_side_by_side.txt`](attention-side-by-side/check_side_by_side.txt). B, C and E follow from the formulas in the sections above.

## Changes for Lizard v2

The comparison points to three small changes. Each is one config option, and two already exist.

| Change | From | Reason | In v2 |
|---|---|---|---|
| Window with RoPE | LoLCATs, Liger | Both windows give the weights of the teacher inside the window. The window of Lizard cannot find the previous token (claim F). | `window_rope: true` |
| Branch weights with the sum 1 | LoLCATs (exact), Liger (0.5 and 0.5) | The row form of Lizard is larger than a weighted mean (claim C). `convex` keeps a weighted mean, without the position effect of `hybrid`. | `gla_norm: convex`, proposed in document 14 (approximately 6 lines) |
| One gate for each head | GLA, Liger | Both have a gate for each head. In R1b, one shared gate served 2 local heads and made 28 of the 32 heads of layer 15 worse. | `gate_per_head: true` |

Two larger changes stay out, so that the next runs stay simple:

- **A linear branch only before the window** (LoLCATs, claims A and B). It removes the double count, but it changes the gated branch and its recurrent cache.
- **The feature map and gate of Liger without parameters**. A different method. The ablations of Liger and Lizard disagree on it (document 14).

Document 14 has the order of the runs and the code for `convex`.
