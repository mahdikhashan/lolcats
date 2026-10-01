# 9. Comparison with the paper

The paper: *Lizard: An Efficient Linearization Framework for Large Language Models* ([arXiv:2507.09025](https://arxiv.org/abs/2507.09025), version 4).

## Architecture: the implementation matches the paper

| Component | Paper | This implementation |
|---|---|---|
| Positional encoding | **No RoPE, by design** (Section 3.1). The paper intends the decay of the gate to learn relative position in place of RoPE, for length extrapolation. RoPE has an effect only through the target of the teacher. | The same. The code uses RoPE only to calculate the teacher output in distillation. |
| Feature map | Hedgehog, feature dimension 128, softmax activation (Table 13): φ(x) = [softmax(xW) ⊕ softmax(−xW)] | The same (`feature_dim: 128`, `phi_q`/`phi_k` shared by all heads) |
| Gate | The default is one scalar gate for each token: Γᵢ = γᵢ 1ᵀ, γᵢ = σ(W_γ xᵢ), W_γ ∈ ℝ^{d×1} (Section 5) | The same: `W_gamma = Linear(hidden, 1, bias=False)`. All heads share one γ for each token. |
| Gated linear attention | Normalized: ŷᵢ = φ_q(qᵢ)ᵀ Σₜ (Π Γ) φ_k(kₜ) vₜ / φ_q(qᵢ)ᵀ Σⱼ (Π Γ) φ_k(kⱼ) | The same (`gla`) |
| Window attention | Window w = 128 with m = 4 meta tokens. The meta tokens are learnable scalar logits tⱼ in the denominator only. They do not add values. | The same (`awa`, `window_size: 128`, `num_meta: 4`) |
| Combination | Ŷ = Ŷ_gate + α · Ŷ_anchor, α learnable | The same (`alpha_blend`) |
| Stage 1 loss | MSE between the outputs of softmax attention and Lizard attention, over the layers | MSE in the LoLCATs form: 1000 × mean squared error for each layer. For Adam, the scale factor has no effect. |

In two small points, the paper is ambiguous or gives no information:

- **Sink term.** The equation of the paper has Σⱼ tⱼ in the denominator, but the paper calls tⱼ "the logit of a meta-memory token". The implementation uses Σⱼ exp(tⱼ). This is the reading that agrees with the word "logit".
- **Gate initialization.** The paper does not give it. The implementation sets the initial value of W_γ to zero. Thus γ = 0.5 for every token at the start.

## Recipe: differences between the runs and the paper

The table compares the recipe of the paper (Section 5 and Table 13) with the recipe of the runs. The runs used the LoLCATs configs ([document 4](04-training-runs.md)).

| Setting | Paper | Runs of this project |
|---|---|---|
| Data | Alpaca-cleaned, 50K examples, 2 epochs, ~20M tokens per stage, 2048 tokens per example | The same |
| Global batch | 8 (micro batch 1) | The same (1 × 8 accumulation) |
| Precision | bf16 | The same (Lizard calculations in float32) |
| **Stage 1 learning rate** | **1e-3** | **1e-2 (10× higher)** |
| Stage 2 learning rate | 5e-4 | 1e-4 (5× lower) |
| **Schedule** | **Cosine, 10% linear warmup, minimum 0.1× peak** | **Constant (ReduceLROnPlateau with patience 10 evaluations), no warmup** |
| **Gradient clipping** | **1.0** | **None** |
| AdamW betas, eps | (0.9, 0.99), 1e-8 | (0.9, 0.999), 1e-8 (defaults) |
| LoRA | r = 8, α = 16, dropout 0, on W_Q, W_K, W_V | r = 8, α = 16, dropout 0, on q, k, v **and o** |
| Lizard parameters in stage 2 | No information | Frozen (only LoRA trains) |

The jku-thesis `config.py` and `train.py` already use exactly the recipe of the paper:

- learning rates 1e-3 and 5e-4,
- 10% warmup, then cosine decay to 0.1×,
- betas (0.9, 0.99),
- clipping at 1.0,
- LoRA on q/k/v.

The jku-thesis code also keeps the Lizard parameters trainable in stage 2. The thesis pipeline runs in float32, not in bf16.

**Hypothesis:** The stage 1 learning rate is 10× higher, with no warmup and no clipping. It trains new feature maps and gates from their initial values. This combination gives a poor approximation of the attention. Stage 2 cannot compensate for it, because its learning rate is 5× lower than in the paper. The checks found no error in the code ([document 8](08-verification.md)). Thus the recipe is the main remaining difference from the paper. This hypothesis has no proof yet ([document 10](10-open-issues-and-next-steps.md)).

## Reference values of the paper

### Small models (Table 9)

| Model | PIQA (acc) | ARC-e (acc) | ARC-c (acc_norm) | HellaSwag (acc_norm) | WinoGrande (acc) | MMLU 5-shot | Avg. |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 74.1 | 65.4 | 36.4 | 63.8 | 60.0 | 31.0 | 59.9 |
| → LoLCATs | 74.6 | 63.0 | 35.1 | 63.7 | 61.5 | 27.3 | 59.6 |
| → Lizard | 74.8 | 65.6 | 36.5 | 64.1 | 59.7 | 29.8 | 60.1 |
| Llama-3.2-3B | 76.4 | 74.7 | 46.0 | 73.6 | 69.9 | 56.2 | 68.1 |
| → Lizard | 76.8 | 75.2 | 45.2 | 74.3 | 69.3 | 53.4 | 68.2 |

At 1B, MMLU is near chance for every model (teacher 31.0). Thus MMLU improvements are small in absolute values.

### Ablations on Llama-3-8B (MMLU 5-shot)

| Table | Variant | MMLU |
|---|---|---|
| 6 | Full Lizard | 61.2 |
| 6 | Without sliding window attention | 39.7 |
| 6 | Without the gated module | 42.2 |
| 6 | Without stage 1 (attention approximation) | 50.8 |
| 6 | Full finetuning in place of LoRA | 61.4 |
| 4 | Gate σ(W_γ x), W_γ ∈ ℝ^{d×1} (default) | 61.2 |
| 4 | Gate in the style of Mamba-2 | 57.6 |
| 4 | Low-rank gate in the style of GLA (d×16, 16×d) | 53.5 |
| 4 | 1D pooling gate without parameters | 44.1 |
| 7 | w = 32 / 64 / 128 / 256, m = 4 | 52.4 / 57.6 / 61.2 / 44.6 |
| 7 | m = 2 / 4 / 6, w = 128 | 58.6 / 61.2 / 60.8 |
| 8 | LoRA rank 4 / 8 / 16 / 32 / 64 | 59.7 / 61.2 / 60.6 / 61.0 / 59.2 |

The paper gives "local attention dominance" as the reason for the decrease at w = 256. With a large window, the exact local softmax controls most of the gradients, and the gated module does not learn its dynamics.

## Suggestions from before the paper was available, now withdrawn

Before the text of the paper was available, this project suggested four architecture changes for MMLU. Each change goes against the paper:

| Suggestion | Reason for the withdrawal |
|---|---|
| RoPE in the window branch, as in the original LoLCATs attention | The paper uses no RoPE on purpose (Section 3.1). |
| One gate for each head, with a bias near 1 at initialization | The default of the paper is the scalar gate that the code already has. Gates with more parameters get lower scores (Table 4). |
| A larger window (256 or 512) | w = 256 decreases MMLU from 61.2 to 44.6 (Table 7). |
| Sink logits for each head | The paper uses m scalars for each layer (t ∈ ℝ^m), as the code does. |

The lesson for the thesis: before an architecture change, check the ablations of the paper. The remaining difference is the recipe.
