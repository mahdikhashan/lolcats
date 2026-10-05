Yes — and after searching specifically for **attention-layer reproduction, Transformer architecture reproduction, and LLM training reproduction**, I found several that are much closer to your Lizard work than the generic reproducibility papers I listed before.

The **2026 paper on partial residual ablations** is especially close to what you are doing: it reproduces architectural experiments in GPT-style Transformers, finds a discrepancy, identifies a measurement confound, fixes it, and then reruns controlled experiments. ([arXiv][1])

## Most relevant papers

| Paper                                                                                   | Model / architecture                      | What is reproduced                                                                | What they actually investigate                                                                                                  | Methodology                                                                                                                                                                                        | What you could borrow for Lizard                                                                                                                                                                       |
| --------------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **A Reproducibility Study of Partial Residual Ablations in Pre-LN Transformers** (2026) | GPT-style Pre-LN Transformers, 10M & 124M | Architectural ablation results concerning **attention and FFN residual pathways** | Whether reported architectural effects reproduce across scale and seeds                                                         | Reimplemented experiments; 4 residual configurations; 8-seed controlled study; discovered a measurement confound; corrected it; repeated experiments; released failed runs + checkpoints + configs | **★★★★★ Best match.** Use the exact pattern: reproduce → observe discrepancy → identify implementation/measurement cause → correct → rerun → distinguish confirmed vs unresolved results. ([arXiv][1]) |
| **RoBERTa: A Robustly Optimized BERT Pretraining Approach** (2019)                      | BERT                                      | BERT's pretraining recipe                                                         | Which supposedly minor training choices actually explain performance differences                                                | Systematically varied training data, batch size, training duration, masking, learning rate, etc.                                                                                                   | **★★★★★** Make a Lizard *training-factor ablation matrix*: initialization, LR, warmup, tokens, batch size, masking, distillation temperature, etc. ([Hugging Face][2])                                 |
| **Attention Is All You Need — independent reproductions**                               | Original Transformer                      | **Multi-head attention + full Transformer architecture + training recipe**        | Whether the architecture can be reconstructed from the paper and achieve comparable behavior                                    | Implement attention, positional encoding, encoder/decoder, optimizer and Noam LR schedule; train and compare                                                                                       | **★★★★☆** Very useful for your **equation → PyTorch implementation → numerical verification → end-to-end evaluation** workflow. ([GitHub][3])                                                          |
| **AttentionSmithy: A Modular Framework for Rapid Transformer Development**              | Vanilla Transformer + variants            | Original Transformer architecture/training setup                                  | Whether a modular implementation can reproduce the original Transformer while supporting controlled architectural modifications | Reimplemented Transformer on WMT14; tested context-length truncation and alternative architectures                                                                                                 | **★★★★☆** Good model for building a controlled baseline before replacing attention with GLA/SWA. ([PubMed Central (PMC)][4])                                                                           |
| **The MultiBERTs: BERT Reproductions for Robustness Analysis**                          | BERT-base                                 | BERT pretraining repeatedly                                                       | How much results depend on random seed/data ordering rather than architecture                                                   | 25 independently trained BERT models + many intermediate checkpoints                                                                                                                               | **★★★★★** Run multiple Lizard seeds and compare *training trajectories*, not only final MMLU.                                                                                                          |
| **OpenLLaMA: An Open Reproduction of LLaMA**                                            | LLaMA                                     | LLaMA architecture + training recipe                                              | Whether LLaMA can be independently recreated using open resources                                                               | Reimplemented architecture/training pipeline and trained from scratch                                                                                                                              | **★★★★☆** Best precedent for your overall "independent LLM reproduction" claim.                                                                                                                        |
| **TinyLlama**                                                                           | Llama-2 architecture                      | Llama architecture at 1.1B scale                                                  | Whether a Llama-like model can be independently trained at manageable scale                                                     | Open training data + Llama-compatible architecture + long pretraining                                                                                                                              | **★★★☆☆** Useful for validating architectural modifications cheaply before Llama-3.2-1B.                                                                                                               |

---

# 1. The 2026 residual-ablation paper is *very* close to your thesis

This is the one I would read first.

**A Reproducibility Study of Partial Residual Ablations in Pre-LN Transformers**.

The authors study GPT-style Transformers and remove different residual pathways:

```text
Normal:

x → Attention → + residual
  → FFN       → + residual


Ablation 1:

x → Attention → + residual
  → FFN       → no residual


Ablation 2:

x → Attention → no residual
  → FFN       → + residual
```

They test models at **10M and 124M parameters**.

The really interesting part isn't merely the ablation.

They report that they:

1. attempted reproduction,
2. obtained unexpected results,
3. investigated the discrepancy,
4. found a **measurement confound**,
5. corrected the experiment,
6. reran it,
7. used multiple seeds,
8. distinguished reproducible findings from unresolved findings,
9. released even the failed intermediate experiments. ([arXiv][1])

That is almost exactly the methodology I'd recommend for your Lizard problem.

### Their experimental logic

```text
Published claim
      ↓
Independent implementation
      ↓
Does result reproduce?
      │
    NO
      ↓
Investigate discrepancy
      ↓
Implementation?
Measurement?
Training?
Randomness?
      ↓
Correct
      ↓
Controlled replication
      ↓
Does claim now reproduce?
```

You could use almost the same structure:

```text
Lizard paper
     ↓
Your implementation
     ↓
MMLU = 22–23%
vs reported / teacher = ~32%
     ↓
Why?
 ┌───┼────┬────┬─────┐
 ↓   ↓    ↓    ↓     ↓
math code init train eval
     ↓
controlled experiment
     ↓
corrected reproduction
     ↓
GLA + SWA modification
```

That is a **very defensible thesis methodology**.

---

# 2. RoBERTa is extremely useful for your training-reproduction problem

RoBERTa is technically a BERT replication study rather than an attention-layer replication.

But this is precisely why it is valuable.

The authors asked:

> How much of BERT's reported performance is actually determined by the architecture versus seemingly secondary training choices?

They systematically investigated things such as:

* training duration
* batch size
* learning rate
* training data
* masking strategy
* data size
* optimization choices

and showed that BERT was substantially undertrained. ([Hugging Face][2])

This gives you a powerful argument for Lizard:

### Don't immediately assume:

```text
MMLU gap
   ↓
architecture implementation is wrong
```

Instead:

```text
MMLU gap
│
├── architecture
│   ├── Q/K/V
│   ├── GLA
│   ├── SWA
│   ├── sink tokens
│   ├── normalization
│   └── residual
│
├── initialization
│
├── optimization
│   ├── LR
│   ├── warmup
│   ├── Adam β
│   └── weight decay
│
├── training
│   ├── number of tokens
│   ├── batch size
│   ├── sequence length
│   └── checkpoint
│
└── evaluation
    ├── MMLU prompt
    ├── 5-shot examples
    ├── tokenizer
    ├── answer extraction
    └── lm-eval version
```

This is exactly the kind of decomposition RoBERTa gives you precedent for.

---

# 3. There are also actual attention-layer reproductions

For **attention specifically**, the original *Attention Is All You Need* architecture has numerous independent implementations/reproductions.

For example, one reproduction explicitly reconstructs:

* scaled dot-product attention
* multi-head attention
* masking
* positional encoding
* encoder
* decoder
* residual connections
* layer normalization
* FFN
* label smoothing
* Noam learning-rate schedule

and compares the resulting model against the original Transformer setup. ([GitHub][3])

Another reproduction takes a more explicit **scaled reproduction** approach and tests the individual attention implementation, masking, complete forward pass, and gradient flow. ([GitHub][5])

These are not all peer-reviewed papers, so I would **not use the GitHub repositories as your primary scholarly citations**. But they are useful as implementation references.

---

# 4. There is an interesting distinction for your thesis

I'd separate your work into three levels:

### Level 1 — Mathematical reproduction

```text
Paper equation
      ↓
PyTorch implementation
      ↓
Numerical equivalence
```

For example:

$$
A(Q,K,V)=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Then test:

```text
paper equation
       ≈
reference implementation
       ≈
your implementation
```

This is where your current **math-vs-code verification** work fits extremely well.

---

### Level 2 — Architectural reproduction

```text
Paper
 ↓
attention module
 ↓
Transformer block
 ↓
full LLM
```

Validate:

* tensor shapes
* parameter count
* masks
* normalization
* residual paths
* initialization
* positional encoding
* attention outputs
* gradients

---

### Level 3 — Training reproduction

```text
architecture
     +
data
     +
tokenizer
     +
optimizer
     +
LR schedule
     +
initialization
     +
training tokens
     ↓
trained model
     ↓
evaluation
```

This is where OpenLLaMA / RoBERTa / MultiBERTs become relevant.

---

# 5. The paper I'd use as the methodological template for Lizard

I'd actually combine **three papers**:

| Your Lizard stage                                | Best precedent                                           |
| ------------------------------------------------ | -------------------------------------------------------- |
| **Verify attention equations/code**              | *Attention Is All You Need* reproductions                |
| **Reproduce Lizard training**                    | OpenLLaMA + RoBERTa                                      |
| **Investigate why reproduction differs**         | **2026 Partial Residual Ablation Reproducibility Study** |
| **Determine whether differences are stochastic** | MultiBERTs                                               |
| **Analyze training trajectory**                  | Pythia                                                   |

This gives you a very clean methodology:

```text
                  LIZARD
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
   Mathematical           Training
   reproduction           reproduction
          │                   │
   equations ↔ code      recipe ↔ code
          │                   │
          └─────────┬─────────┘
                    ↓
             discrepancy
                    ↓
          systematic analysis
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
 architecture    training    evaluation
       │            │            │
       └────────────┼────────────┘
                    ↓
             corrected baseline
                    ↓
             GLA + SWA model
                    ↓
           controlled comparison
```

And the **2026 residual-ablation paper is particularly valuable** because it demonstrates that an architectural reproduction paper can legitimately contain:

> **failed reproduction → diagnosis → corrected reproduction → new architectural insight**

rather than pretending that the first implementation was perfect. ([arXiv][1])

That is arguably **the closest existing methodological precedent to what you're currently doing with Lizard**.

[1]: https://arxiv.org/abs/2608.14689?utm_source=chatgpt.com "A Reproducibility Study of Partial Residual Ablations in Pre-LN Transformers"
[2]: https://huggingface.co/papers/1907.11692?utm_source=chatgpt.com "Paper page - RoBERTa: A Robustly Optimized BERT Pretraining Approach"
[3]: https://github.com/PolarisAI-Implementations/paper-1706-03762?utm_source=chatgpt.com "GitHub - PolarisAI-Implementations/paper-1706-03762: Polaris reproduction for arXiv 1706.03762 · GitHub"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12987691/?utm_source=chatgpt.com "AttentionSmithy: A Modular Framework for Rapid Transformer Development - PMC"
[5]: https://github.com/Lavanya-Jothivel/transformer-from-scratch?utm_source=chatgpt.com "GitHub - Lavanya-Jothivel/transformer-from-scratch: A scaled reproduction of \"Attention Is All You Need\", implementing the Transformer architecture from scratch in PyTorch with experiments and ablations. · GitHub"
