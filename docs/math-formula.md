# Math formulas of Lizard attention

This document collects the equations of the paper ([arXiv:2507.09025](https://arxiv.org/abs/2507.09025), version 4). The formulas come from images of the paper. Each section gives the source in the paper and the matching function in `src/model/linear_attention/lizard_attention.py`.

**Status:** sections 2, 3 and 6 are complete. Sections 1, 4, 5 and 7 wait for the second set of images.

## Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{q}_i, \mathbf{k}_t, \mathbf{v}_t$ | Query of position $i$, key and value of position $t$ |
| $d$ | Head dimension |
| $\varphi_R$ | The RoPE transformation |
| $\phi, \phi_q, \phi_k$ | Feature maps of linear attention. Lizard uses separate maps for queries ($\phi_q$) and keys ($\phi_k$). |
| $\boldsymbol{\Gamma}_l$ | Gate of position $l$ |
| $\mathbf{S}_i$ | State of the gated branch after position $i$ |

## 1. Sliding window attention (SWA)

Waits for the second set of images.

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

**Note:** The recurrent form has no denominator. The parallel form has one. The code uses the parallel form with the denominator. Section 13.4 of the [gap analysis](11-gap-analysis.md) discusses this difference (point (a)).

## 3. Attention approximation

### Target of stage 1: softmax attention with RoPE

Source: Section 3.1 of the paper, "First Stage: Approximating Softmax Attention for Unbounded Context". In the teacher, RoPE transforms the queries and the keys before the attention. Stage 1 trains the Lizard attention, which has no RoPE, to give this output.

```math
\mathbf{y}_i = \sum_{t=1}^{i} \frac{\exp\left( \varphi_R(\mathbf{q}_i)^\top \varphi_R(\mathbf{k}_t) / \sqrt{d} \right)}{\sum_{j=1}^{i} \exp\left( \varphi_R(\mathbf{q}_i)^\top \varphi_R(\mathbf{k}_j) / \sqrt{d} \right)}\, \mathbf{v}_t
```

**Note:** In the image, the numerator shows $\varphi_R(\mathbf{k}t)$. This document writes $\varphi_R(\mathbf{k}_t)$, as in the denominator.

In code: `y_true` in `LolcatsLizardAttention.forward`, when `train_attention` is true.

## 4. Loss function

Waits for the second set of images.

## 5. Hardware-aware algorithm for efficient training

Waits for the second set of images.

## 6. Causal softmax attention

Source: Section 2 of the paper, "Causal Softmax Attention". This is the attention of the teacher, without the RoPE transformation.

```math
\mathbf{y}_i = \sum_{t=1}^{i} \frac{\exp\left( \mathbf{q}_i^\top \mathbf{k}_t / \sqrt{d} \right)}{\sum_{j=1}^{i} \exp\left( \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d} \right)}\, \mathbf{v}_t
```

## 7. Hedgehog feature map

Waits for the second set of images.
