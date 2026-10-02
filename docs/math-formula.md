# Math formulas of Lizard attention

This document collects the equations of the paper ([arXiv:2507.09025](https://arxiv.org/abs/2507.09025), version 4). The formulas come from images of the paper. A formula that comes from the text of the paper, and not from an image, has a note. Each section gives the source in the paper and the matching code. The code is in `src/model/linear_attention/lizard_attention.py`, unless the section gives a different file.

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{q}_i, \mathbf{k}_t, \mathbf{v}_t$ | Query of position $i$, key and value of position $t$ |
| $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ | Queries, keys and values of all positions, one row for each position |
| $\mathbf{x}_i$ | Input of the attention layer at position $i$ |
| $d$ | Head dimension |
| $\varphi_R$ | The RoPE transformation |
| $\phi, \phi_q, \phi_k$ | Feature maps of linear attention. Lizard uses separate maps for queries ($\phi_q$) and keys ($\phi_k$). |
| $\mathbf{W}$ | Weight matrix of a Hedgehog feature map |
| $\boldsymbol{\Gamma}_l$ | Gate of position $l$. The paper defines $\boldsymbol{\Gamma}_i = \mathrm{sigmoid}(\mathbf{W}_\gamma \mathbf{x}_i)$ (Section 3.1). By default, it is one scalar $\gamma_i$ for each position (Section 5). |
| $\mathbf{S}_i$ | State of the gated branch after position $i$ |
| $\mathbf{C}$ | Matrix of cumulative gate products, with rows $\mathbf{c}_t$ |
| $\mathbf{M}$ | Causal mask |
| $w$ | Window size (128) |
| $m$, $t_j$ | Number of meta-memory tokens (4), and the learnable logit of meta-memory token $j$ |
| $\alpha$ | Learnable weight of the window branch |
| $N$, $\mathbf{Y}^l$ | Number of attention layers, and the attention output of layer $l$ |
| $\odot$, $\oplus$ | Element-wise product, and concatenation |

## 1. Sliding window attention (SWA)

Source: Section 3.1 of the paper, "Anchor Window Attention for Local Precision". The window covers the last $w$ positions. The $m$ meta-memory tokens add only their logits $t_j$ to the denominator. They add no values. The paper marks $t_j$ in red. This document uses red too.

```math
\hat{\mathbf{y}}_i = \frac{\sum_{t=i-w+1}^{i} \exp\left( \mathbf{q}_i^\top \mathbf{k}_t / \sqrt{d} \right) \mathbf{v}_t}{\sum_{j=0}^{m-1} {\color{red} t_j} + \sum_{t=i-w+1}^{i} \exp\left( \mathbf{q}_i^\top \mathbf{k}_t / \sqrt{d} \right)}
```

The paper calls $t_j$ "a learnable scalar parameter representing the logit of a meta-memory token".

In code: `awa` and `sink_softmax`.

**Note:** The code uses $\sum_j \exp(t_j)$ in the denominator, not $\sum_j t_j$. This is the reading that agrees with the word "logit" ([document 9](09-paper-comparison.md)).

## 2. Gated linear attention (GLA)

### Linear attention, without a gate

Source: Section 2 of the paper, "Linear Attention and Linearization". The kernel $\phi(\mathbf{q})^\top \phi(\mathbf{k})$ replaces the exponential of softmax attention.

```math
\hat{\mathbf{y}}_i = \frac{\phi(\mathbf{q}_i)^\top \left( \sum_{t=1}^{i} \phi(\mathbf{k}_t)\, \mathbf{v}_t^\top \right)}{\phi(\mathbf{q}_i)^\top \left( \sum_{j=1}^{i} \phi(\mathbf{k}_j) \right)}
```

### Gated linear attention, parallel form

Source: Section 3.1 of the paper, "Learnable Gating for Adaptive Memory Control and Length Extrapolation". The paper marks the gate product in red. This document uses red too.

```math
\hat{\mathbf{y}}_i = \frac{\phi_q(\mathbf{q}_i)^\top \left( \sum_{t=1}^{i} {\color{red} \left( \prod_{l=t+1}^{i} \boldsymbol{\Gamma}_l \right)} \phi_k(\mathbf{k}_t)\, \mathbf{v}_t^\top \right)}{\phi_q(\mathbf{q}_i)^\top \left( \sum_{j=1}^{i} {\color{red} \left( \prod_{l=j+1}^{i} \boldsymbol{\Gamma}_l \right)} \phi_k(\mathbf{k}_j) \right)}
```

In code: `gla`, with the gate products from `gate_products`.

### Gated linear attention, recurrent form

Source: Section 3.1 of the paper. The state $\mathbf{S}_i$ holds the history up to position $i$. Thus inference does not need to store all keys and values.

```math
\mathbf{S}_i = \boldsymbol{\Gamma}_i \mathbf{S}_{i-1} + \phi_k(\mathbf{k}_i)\, \mathbf{v}_i^\top, \qquad \hat{\mathbf{y}}_i = \phi_q(\mathbf{q}_i)^\top \mathbf{S}_i
```

In code: the decode loop of `lizard_recurrent`, which also keeps a state for the denominator.

**Note:** The recurrent form has no denominator. The parallel form has one. The matrix form in section 5 of this document also has no denominator. The code uses the parallel form with the denominator. Section 13.4 of the [gap analysis](11-gap-analysis.md) discusses this difference (point (a)).

## 3. Attention approximation

### Target of stage 1: softmax attention with RoPE

Source: Section 3.1 of the paper, "First Stage: Approximating Softmax Attention for Unbounded Context". In the teacher, RoPE transforms the queries and the keys before the attention. Stage 1 trains the Lizard attention, which has no RoPE, to give this output.

```math
\mathbf{y}_i = \sum_{t=1}^{i} \frac{\exp\left( \varphi_R(\mathbf{q}_i)^\top \varphi_R(\mathbf{k}_t) / \sqrt{d} \right)}{\sum_{j=1}^{i} \exp\left( \varphi_R(\mathbf{q}_i)^\top \varphi_R(\mathbf{k}_j) / \sqrt{d} \right)}\, \mathbf{v}_t
```

**Note:** In the image, the numerator shows $\varphi_R(\mathbf{k}t)$. This document writes $\varphi_R(\mathbf{k}_t)$, as in the denominator.

In code: `y_true` in `LolcatsLizardAttention.forward`, when `train_attention` is true.

### Output of Lizard attention

Source: Section 3.1 of the paper, "Attention Approximation". The output adds the gated branch (section 2) and the window branch (section 1), with the learnable weight $\alpha$ on the window branch.

```math
\hat{\mathbf{Y}}_{lizard} = \hat{\mathbf{Y}}_{gate} + \alpha \cdot \hat{\mathbf{Y}}_{anchor}
```

**Note:** This formula comes from the text of the paper. It is not in the images.

In code: `LolcatsLizardAttention.lizard`, with $\alpha$ in `alpha_blend`.

## 4. Loss function

Source: Section 3.1 of the paper. Stage 1 trains the feature maps $\phi$, the gate weights $\mathbf{W}_\gamma$ and the meta-memory logits $\mathbf{t}$. For each layer, the loss takes the squared Frobenius norm of the difference between the teacher output and the Lizard output. Then it takes the mean over the $N$ attention layers.

```math
\mathcal{L}_{\mathrm{MSE}}(\phi, \mathbf{W}_\gamma, \mathbf{t}) = \frac{1}{N} \sum_{l=1}^{N} \left\| \mathbf{Y}^l_{\mathrm{softmax}} - \hat{\mathbf{Y}}^l_{lizard} \right\|_F^2
```

In code: `src/trainer/distill_attention_xent_mse.py`. The code uses 1000 × the mean over the layers of the mean squared error of each layer. The two forms differ only by a constant factor. For Adam, this factor has no effect (factor 8 in section 12 of the [gap analysis](11-gap-analysis.md)).

**Notes:**

- The loss lists $\phi$, $\mathbf{W}_\gamma$ and $\mathbf{t}$. The code also trains $\alpha$ (`alpha_blend`) in stage 1.
- The code compares the outputs before `o_proj`.

## 5. Hardware-aware algorithm for efficient training

Source: Section 4 of the paper, "Hardware-Aware Algorithm for Efficient Training".

### Matrix form of gated linear attention

```math
\hat{\mathbf{Y}}_{gate} = \left( \left( \left( \phi(\mathbf{Q}) \odot \mathbf{C} \right) \left( \frac{\phi(\mathbf{K})}{\mathbf{C}} \right)^\top \right) \odot \mathbf{M} \right) \mathbf{V}
```

Each row of $\mathbf{C}$ is a cumulative gate product:

```math
\mathbf{c}_t = \prod_{j=1}^{t} \boldsymbol{\Gamma}_j
```

### Problem

The paper gives these reasons:

- The products $\mathbf{c}_t$ can become very small. In bfloat16, this causes underflow and instability during training.
- Thus the matrix form needs a fallback to float32. The paper says that float32 is 2–3× slower, uses more memory, and cannot use the Tensor Cores.

### Reparameterization in log space

The Hedgehog feature map is strictly non-negative and exponential (section 7). Thus the paper moves the gate term $\mathbf{C}$ into the exponents of the query and key features:

```math
\widetilde{\mathbf{Q}} = \left[ \exp\left( \mathbf{Q}\mathbf{W} + \log \mathbf{C} \right) \oplus \exp\left( -\mathbf{Q}\mathbf{W} + \log \mathbf{C} \right) \right]
```

```math
\widetilde{\mathbf{K}} = \left[ \exp\left( \mathbf{K}\mathbf{W} - \log \mathbf{C} \right) \oplus \exp\left( -\mathbf{K}\mathbf{W} - \log \mathbf{C} \right) \right]
```

With these features, the gated branch becomes one standard matrix multiplication (GEMM):

```math
\hat{\mathbf{Y}}_{gate} = \left( \left( \widetilde{\mathbf{Q}} \widetilde{\mathbf{K}}^\top \right) \odot \mathbf{M} \right) \mathbf{V}
```

**Note:** The last formula comes from the text of the paper. It is not in the images.

In code: this project does not use this form for training. `lizard_attention.py` calculates the parallel form of section 2 in float32 (`upcast`, `gate_products`, `gla`). The jku-thesis file `kernels/gla_lizard_reparam.py` implements the reparameterization as an experiment.

## 6. Causal softmax attention

Source: Section 2 of the paper, "Causal Softmax Attention". This is the attention of the teacher, without the RoPE transformation.

```math
\mathbf{y}_i = \sum_{t=1}^{i} \frac{\exp\left( \mathbf{q}_i^\top \mathbf{k}_t / \sqrt{d} \right)}{\sum_{j=1}^{i} \exp\left( \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d} \right)}\, \mathbf{v}_t
```

## 7. Hedgehog feature map

Source: Section 4 of the paper, which cites Hedgehog (Zhang et al.). The map concatenates the exponential of $\mathbf{x}\mathbf{W}$ and the exponential of $-\mathbf{x}\mathbf{W}$. Thus every feature is positive.

```math
\phi(\mathbf{x}) = \left[ \exp(\mathbf{x}\mathbf{W}) \oplus \exp(-\mathbf{x}\mathbf{W}) \right]
```

**Note:** Table 13 of the paper gives "Hedgehog Feature Activation: Softmax" and feature dimension 128. The code uses the softmax form:

```math
\phi(\mathbf{x}) = \left[ \mathrm{softmax}(\mathbf{x}\mathbf{W}) \oplus \mathrm{softmax}(-\mathbf{x}\mathbf{W}) \right]
```

Each softmax divides the exponentials by their sum for one token. The paper does not write the softmax form as a formula. This document writes it from Table 13 and from the code.

In code: `hedgehog`, with $\mathbf{W}$ in `phi_q` for queries and `phi_k` for keys.

## 8. Gating designs (Table 4 of the paper)

Source: Section 5.4 of the paper, "Gated Structures Design", Table 4. The caption of the table is "Performance comparison of different gating designs and their parameterizations." The paper reports the ablations of Section 5.4 for Llama-3-8B. The value 61.2 is also the MMLU 5-shot result of the full Lizard model of Llama-3-8B in Table 1.

```math
\begin{array}{llll}
\textbf{Model} & \textbf{Gating parameterization} & \textbf{Learnable parameters} & \textbf{MMLU 5-shot} \\
\hline
\text{Lizard (ours)} & \boldsymbol{\Gamma}_i = \gamma_i \mathbf{1}_d^\top,\ \gamma_i = \sigma(W_\gamma \mathbf{x}_i) & W_\gamma \in \mathbb{R}^{d \times 1} & 61.2 \\
\text{Mamba-2 (Dao and Gu, 2024)} & \boldsymbol{\Gamma}_i = \gamma_i \mathbf{1}_d^\top,\ \gamma_i = \exp\left( -\mathrm{softplus}(\mathbf{x}_i W_\gamma) \cdot \exp(a) \right) & W_\gamma \in \mathbb{R}^{d \times 1},\ a \in \mathbb{R} & 57.6 \\
\text{GLA (Yang et al., 2024)} & \boldsymbol{\Gamma}_i = \sigma\left( \mathbf{x}_i W_{\gamma_1} W_{\gamma_2} \right) & W_{\gamma_1} \in \mathbb{R}^{d \times 16},\ W_{\gamma_2} \in \mathbb{R}^{16 \times d} & 53.5 \\
\text{1D-Pooling} & \boldsymbol{\Gamma}_i = \sigma\left( \mathrm{Pooling}(\mathbf{k}_t) \right) & \text{N/A} & 44.1 \\
\end{array}
```

LaTeX source of the same table for the thesis (needs the package `booktabs`):

```latex
\begin{table}[t]
  \centering
  \begin{tabular}{llll}
    \toprule
    \textbf{Model} & \textbf{Gating Parameterization} & \textbf{Learnable Parameters} & \textbf{MMLU 5-shot} \\
    \midrule
    \textbf{Lizard (Ours)} & $\Gamma_i = \gamma_i \mathbf{1}_d^\top,\ \gamma_i = \sigma(W_\gamma \mathbf{x}_i)$ & $W_\gamma \in \mathbb{R}^{d \times 1}$ & 61.2 \\
    Mamba-2 (Dao and Gu, 2024) & $\Gamma_i = \gamma_i \mathbf{1}_d^\top,\ \gamma_i = \exp\left(-\mathrm{softplus}(\mathbf{x}_i W_\gamma) \cdot \exp(a)\right)$ & $W_\gamma \in \mathbb{R}^{d \times 1},\ a \in \mathbb{R}$ & 57.6 \\
    GLA (Yang et al., 2024) & $\Gamma_i = \sigma\left(\mathbf{x}_i W_{\gamma_1} W_{\gamma_2}\right)$ & $W_{\gamma_1} \in \mathbb{R}^{d \times 16},\ W_{\gamma_2} \in \mathbb{R}^{16 \times d}$ & 53.5 \\
    1D-Pooling & $\Gamma_i = \sigma\left(\mathit{Pooling}(\mathbf{k}_t)\right)$ & N/A & 44.1 \\
    \bottomrule
  \end{tabular}
  \caption{Performance comparison of different gating designs and their parameterizations.}
  \label{tab:gating-designs}
\end{table}
```

**Notes:**

- The paper writes $W_\gamma \mathbf{x}_i$ for Lizard, but $\mathbf{x}_i W_\gamma$ for Mamba-2 and GLA. With $W_\gamma \in \mathbb{R}^{d \times 1}$ and $\mathbf{x}_i$ as a row vector, $\mathbf{x}_i W_\gamma$ is the order that gives a scalar. Both forms mean one scalar for each position.
- The Lizard row is the default gate of the paper (Section 5). The code uses this gate: `W_gamma = Linear(hidden, 1, bias=False)` and a sigmoid ([math against code](math-code-discrepancy.md), section "Gate").
- The GLA row gives a gate for each dimension ($W_{\gamma_2}$ maps to $d$ outputs). The Lizard row and the Mamba-2 row give one scalar for each position.
- In the 1D-Pooling row, the gate has no learnable parameters. It pools over the key vectors $\mathbf{k}_t$.
