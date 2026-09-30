# 9. Comparison with the paper

Paper: *Lizard: An Efficient Linearization Framework for Large Language Models*
([arXiv:2507.09025](https://arxiv.org/abs/2507.09025), v4).

## Architecture: the implementation matches the paper

| Component | Paper | This implementation |
|---|---|---|
| Positional encoding | **RoPE-free by design** (Sec. 3.1): the gate's decay is meant to learn relative position instead of RoPE, for length extrapolation. RoPE only enters through the teacher's target | Same: RoPE is used only to compute the teacher output in distillation |
| Feature map | Hedgehog, feature dimension 128, softmax activation (Table 13): φ(x) = [softmax(xW) ⊕ softmax(−xW)] | Same (`feature_dim: 128`, `phi_q`/`phi_k` shared across heads) |
| Gate | Default scalar gate per token: Γᵢ = γᵢ 1ᵀ, γᵢ = σ(W_γ xᵢ), W_γ ∈ ℝ^{d×1} (Sec. 5) | Same: `W_gamma = Linear(hidden, 1, bias=False)`, one γ per token shared by all heads |
| Gated linear attention | Normalized: ŷᵢ = φ_q(qᵢ)ᵀ Σₜ (Π Γ) φ_k(kₜ) vₜ / φ_q(qᵢ)ᵀ Σⱼ (Π Γ) φ_k(kⱼ) | Same (`gla`) |
| Window attention | Window w = 128 with m = 4 meta tokens: learnable scalar logits tⱼ in the denominator only; they don't contribute values | Same (`awa`, `window_size: 128`, `num_meta: 4`) |
| Combination | Ŷ = Ŷ_gate + α · Ŷ_anchor, α learnable | Same (`alpha_blend`) |
| Stage 1 loss | MSE between the softmax and Lizard attention outputs, over layers | MSE (LoLCATs form: 1000 × mean squared error per layer; scale-insensitive for Adam) |

Two small points where the paper is ambiguous or silent:

- **Sink term.** The paper's equation writes Σⱼ tⱼ in the denominator but calls tⱼ "the logit of a
  meta-memory token". The implementation uses Σⱼ exp(tⱼ), the consistent reading of a logit.
- **Gate initialization.** The paper doesn't state it. The implementation zero-initializes W_γ, so
  γ = 0.5 for every token at the start.

## Training recipe: where the run differed from the paper

The paper's recipe (Section 5 and Table 13), compared with what the run used (the LoLCATs configs,
[document 4](04-training-runs.md)):

| Setting | Paper | This run |
|---|---|---|
| Data | Alpaca-cleaned, 50K examples, 2 epochs, ~20M tokens per stage, 2048 tokens per example | Same |
| Global batch | 8 (micro batch 1) | Same (1 × 8 accumulation) |
| Precision | bf16 | Same (Lizard math in float32) |
| **Stage 1 learning rate** | **1e-3** | **1e-2 (10× higher)** |
| Stage 2 learning rate | 5e-4 | 1e-4 (5× lower) |
| **Schedule** | **Cosine, 10% linear warmup, minimum 0.1× peak** | **Constant (ReduceLROnPlateau with patience 10 evals), no warmup** |
| **Gradient clipping** | **1.0** | **none** |
| AdamW betas, eps | (0.9, 0.99), 1e-8 | (0.9, 0.999), 1e-8 (defaults) |
| LoRA | r = 8, α = 16, dropout 0, on W_Q, W_K, W_V | r = 8, α = 16, dropout 0, on q, k, v **and o** |
| Lizard parameters in stage 2 | Not stated | Frozen (only LoRA trains) |

The jku-thesis `config.py` and `train.py` already implement the paper's recipe exactly (learning
rates 1e-3 / 5e-4, 10% warmup, cosine to 0.1×, betas (0.9, 0.99), clipping 1.0, LoRA on q/k/v), and
keep the Lizard parameters trainable in stage 2. The thesis pipeline runs in float32 rather than bf16.

**Hypothesis:** a 10× higher stage 1 learning rate, with no warmup or clipping, on freshly initialized
feature maps and gates gives a poor attention approximation, which stage 2 (at a 5× lower rate than
the paper's) cannot make up for. The code is verified ([document 8](08-verification.md)), so this is
the main remaining difference from the paper. It is not yet proven; see
[document 10](10-open-issues-and-next-steps.md).

## The paper's reference numbers

### Small models (Table 9)

| Model | PIQA (acc) | ARC-e (acc) | ARC-c (acc_norm) | HellaSwag (acc_norm) | WinoGrande (acc) | MMLU 5-shot | Avg. |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 74.1 | 65.4 | 36.4 | 63.8 | 60.0 | 31.0 | 59.9 |
| → LoLCATs | 74.6 | 63.0 | 35.1 | 63.7 | 61.5 | 27.3 | 59.6 |
| → Lizard | 74.8 | 65.6 | 36.5 | 64.1 | 59.7 | 29.8 | 60.1 |
| Llama-3.2-3B | 76.4 | 74.7 | 46.0 | 73.6 | 69.9 | 56.2 | 68.1 |
| → Lizard | 76.8 | 75.2 | 45.2 | 74.3 | 69.3 | 53.4 | 68.2 |

At 1B, MMLU is close to chance for every model (teacher 31.0), so MMLU gains are small in absolute terms.

### Ablations on Llama-3-8B (MMLU 5-shot)

| Table | Variant | MMLU |
|---|---|---|
| 6 | Full Lizard | 61.2 |
| 6 | without sliding window attention | 39.7 |
| 6 | without the gated module | 42.2 |
| 6 | without stage 1 (attention approximation) | 50.8 |
| 6 | full finetuning instead of LoRA | 61.4 |
| 4 | Gate σ(W_γ x), W_γ ∈ ℝ^{d×1} (default) | 61.2 |
| 4 | Mamba-2-style gate | 57.6 |
| 4 | GLA-style low-rank gate (d×16, 16×d) | 53.5 |
| 4 | Parameter-free 1D pooling gate | 44.1 |
| 7 | w = 32 / 64 / 128 / 256, m = 4 | 52.4 / 57.6 / 61.2 / 44.6 |
| 7 | m = 2 / 4 / 6, w = 128 | 58.6 / 61.2 / 60.8 |
| 8 | LoRA rank 4 / 8 / 16 / 32 / 64 | 59.7 / 61.2 / 60.6 / 61.0 / 59.2 |

The paper explains the drop at w = 256 as "local attention dominance": with a large window, the exact
local softmax dominates the gradients, and the gated module fails to learn its dynamics.

## Suggestions made before reading the paper, and withdrawn

Before the paper text was available, four architecture changes were suggested for MMLU. Each goes
against the paper:

| Suggestion | Why it was withdrawn |
|---|---|
| Apply RoPE in the window branch (as the original LoLCATs attention does) | The paper is deliberately RoPE-free (Sec. 3.1) |
| Per-head gates with a bias initialized near 1 | The paper's default is the scalar gate the code already has; heavier gates score lower (Table 4) |
| Larger window (256 or 512) | w = 256 drops MMLU from 61.2 to 44.6 (Table 7) |
| Per-head sink logits | The paper uses m scalars per layer (t ∈ ℝ^m), as the code does |

The lesson for the thesis: check the paper's own ablations before changing the architecture. The
remaining difference is the training recipe.
