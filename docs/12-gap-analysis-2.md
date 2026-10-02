# 12. Gap analysis 2

This document analyzes the gap to the paper again, with the results that came after [document 11](11-gap-analysis.md). It uses five sources:

- the [gap analysis](11-gap-analysis.md) (document 11),
- the [layer-wise MSE](experiments/xai-layer-wise-mse.md) experiment,
- the [sample attention weights](experiments/xai-sample-attention-weight.md) experiment,
- the [math formulas](math-formula.md),
- the [math against code](math-code-discrepancy.md) comparison (D1–D17).

It also uses the results of the stage 1 recipe experiments ([paper LR](experiments/paper-lr.md), [float32](experiments/float32.md), [second round](experiments/second-round.md), [feature dimension](experiments/feature-dimension.md)) and of the [stage difference](experiments/stage-difference.md) experiment.

**Basis**. The ranking and the mechanism in section 3 come from measurements on saved checkpoints. No intervention tested the mechanism yet. Section 5 proposes the experiments that can test it. The gradient clipping run ([config 2](experiments/gradient-clipping.md)) finished after this analysis, with a validation loss of 3.5092 (sections 1 and 2).

## The gap

| Benchmark | Lizard model (after stage 2) | Paper, Lizard 1B | Gap |
|---|---:|---:|---:|
| MMLU 5-shot | 23.3 | 29.8 | **−6.5** |
| PIQA 0-shot | 68.0 | 74.8 | **−6.8** |
| ARC-Easy 0-shot | 54.8 | 65.6 | **−10.8** |

The table is the same as in document 11. All stage 1 models so far get 22.5–26.7 on the MMLU subset, 55.8–57.7 on PIQA and 34.1–36.5 on ARC-Easy.

## 1. New evidence since document 11

| Experiment | Result | Effect on the hypotheses of document 11 |
|---|---|---|
| [Stage difference](experiments/stage-difference.md) | PIQA: 57.6 after stage 1, 68.0 after stage 2. MMLU subset: 22.5 after stage 1 ("A" for 95.4%), 24.9 after stage 2 (teacher 33.7). | The drop starts in stage 1. Stage 2 recovers a part of it. |
| Checkpoint checks (`compare_stages.py`, `attention_weights.py`) | Every evaluation with these scripts loaded all 80 Lizard parameters. The stage 2 checkpoint loaded all 128 LoRA weights. | Cause 3 of document 11 (missing keys) does not occur. |
| [Paper LR](experiments/paper-lr.md), [float32](experiments/float32.md) | With the learning rate of the paper, bf16 storage froze α at 1.000 (validation loss 8.1641). float32 fixed it (4.9478). | Factor 0 of document 11 (precision) holds for stage 1. |
| [Second round](experiments/second-round.md) | β2 = 0.99 and a minimum learning rate lowered the validation loss to 3.9764. The LoLCATs recipe gives 3.4219 (fd32) and 3.2549 (fd128). | The recipe of the paper, in float32, does not beat the LoLCATs recipe. |
| [Gradient clipping](experiments/gradient-clipping.md) | Clipping at 1.0 lowered the validation loss from 3.9764 to 3.5092. α is lower, and the gate saturates more. The accuracies did not change clearly. | With all optimizer settings of Table 13, the recipe of the paper comes near the LoLCATs recipe, but does not pass it. |
| All stage 1 recipes | The accuracies after stage 1 stay in small ranges, also with validation losses from 3.25 to 8.16. | The stage 1 recipe does not explain the stage 1 drop on the tasks. |
| [Layer-wise MSE](experiments/xai-layer-wise-mse.md) | Layer 15 gives 24–31% of the loss. Its heads 14 and 23 alone give up to 21%. With the LoLCATs recipe, the error has 24–56% of the power of the teacher output in each layer. | Even the best recipe approximates each layer only roughly. |
| [Sample attention weights](experiments/xai-sample-attention-weight.md) | Mean TV distance to the teacher: 0.31–0.33 (initial weights 0.71). The more local a teacher head is, the worse Lizard matches it (correlation −0.75 to −0.78). | The error has a structure: it depends on the head, not mainly on the recipe. |

### Two causes of document 11 that the evidence now excludes or weakens

- **Missing checkpoint keys (cause 3)**. The checks of `compare_stages.py` and `attention_weights.py` found 0 missing and 0 unexpected keys in every checkpoint.
- **Checkpoint selection (factor 9)**. The stored step is 1,100, the last evaluation, in every float32 run and in the LoLCATs runs. Only the bf16 paper-LR run stopped at step 700, because of the frozen α.

## 2. The stage 1 recipe is not the main cause

Document 11 stated this hypothesis: the code is right, but the LoLCATs recipe gives a much worse Lizard state than the recipe of the paper. The recipe experiments test it:

| Stage 1 run | Validation loss | MMLU subset | PIQA | ARC-Easy |
|---|---:|---:|---:|---:|
| fd128, LoLCATs recipe, bf16 | 3.2549 | 22.5 | 57.6 | – |
| fd32, LoLCATs recipe, bf16 | 3.4219 | 23.2 | – | – |
| fd32, recipe of the paper, bf16 | 8.1641 | 25.3 | 55.8 | 34.1 |
| fd32, recipe of the paper, float32 | 4.9478 | 24.6 | 57.7 | 35.7 |
| fd32, second round, config 1, float32 | 3.9764 | 26.7 | 57.3 | 36.5 |
| fd32, gradient clipping, config 2, float32 | 3.5092 | 23.2 | 57.5 | 35.6 |

- **The recipe of the paper approaches the LoLCATs recipe, but does not pass it**. Each change to the recipe of the paper lowered the loss. But the lowest loss is still the loss of the LoLCATs recipe.
- **The tasks do not follow the loss**. With unpaired SEs, none of the differences in the accuracies is clear.
- **The LoLCATs recipe saturates the gate, but still has the lowest loss**. Its gate keeps γ above 0.999 for all tokens in 14 of the layers 1–15. Section 3 explains why this state can be the best state for the current architecture.

**Conclusion**. The recipe and the precision matter, but they do not explain the size of the stage 1 drop. One recipe cell is still missing: the LoLCATs recipe in float32 (H1 in section 5).

## 3. The new main finding: a limit for each head

### 3.1 A ceiling on the window share

A Lizard row is $A_{i,\cdot} = A^{gla}_{i,\cdot} + \alpha A^{window}_{i,\cdot}$ ([sample attention weights](experiments/xai-sample-attention-weight.md)). The gated branch always sums to 1, because the code normalizes it (D1). The window branch sums to $1 - \text{sink mass} \le 1$. Thus the window share of every head has a ceiling:

```math
\text{window share}_i = \frac{\alpha \sum_t A^{window}_{i,t}}{1 + \alpha \sum_t A^{window}_{i,t}} \le \frac{\alpha}{1 + \alpha}
```

α is one number for each layer. Thus all 32 heads of a layer have the same ceiling. The measurements reach it:

| Layer, head | Checkpoint | α of the layer | Ceiling α / (1 + α) | Measured window share | Teacher mass inside the window |
|---|---|---:|---:|---:|---:|
| 15, head 14 | Second round, config 1 | 0.576 | 0.365 | 0.365 | 0.913 |
| 0, all heads | fd32, LoLCATs recipe | 0.063 | 0.059 | 0.059 (mean) | 0.489 (mean) |

- Layer 15, head 14 puts 91% of its teacher weight inside the window. Lizard can take at most 36.5% of this head from the exact window branch. The other 63.5% comes from the gated branch, which spreads over the whole prefix in this layer.
- In layer 0, the LoLCATs recipe trained α down to 0.063. Then no head of layer 0 can take more than 6% from the window branch. Layer 0 has local heads, for example head 2, which attends to the previous token.

### 3.2 The heads of a layer share all its Lizard parameters

| Parameter | Shape in the code | Shared by the 32 heads | The paper |
|---|---|---|---|
| `phi_q`, `phi_k` | One map from 64 to the feature dimension | Yes | Not given. The LoLCATs default has one map for each head (D2). |
| `W_gamma` | One gate value for each token | Yes | Yes: one scalar for each position (Section 5, Table 4) |
| `alpha_blend` | One scalar | Yes | One learnable α, no more detail |
| `meta_tokens` | 4 scalars | Yes | $\mathbf{t} \in \mathbb{R}^m$, no head index |

Only q, k and v differ between the heads. Stage 1 does not train them. Thus each head can adapt only through its own q, k and v, which come from the teacher.

### 3.3 The gated branch cannot be local in one head only

The gated branch could reproduce a local head with a fast decay. But three properties prevent this for one head only:

- **One gate for all heads**. If the other heads of the layer need far tokens, γ stays near 1 for all heads. The LoLCATs recipe saturates the gate in 14 of the layers 1–15.
- **One feature map for all heads**. The kernel $\phi_q(\mathbf{q})^\top \phi_k(\mathbf{k})$ uses the same map for every head.
- **There is no RoPE**. Neither branch knows the relative position of two tokens. Thus the window branch cannot make a sharp diagonal, such as "one token back".

### 3.4 Consequences in the measurements

- **Local heads match worst**. Teacher heads with less than 30% of their weight outside the window have a mean TV distance of 0.56. Heads with more than 70% have 0.21–0.25.
- **The long-range share hardly changes between the heads of a layer**. Inside a layer, the standard deviation over the heads is 0.126 for the teacher, but only 0.034–0.048 for Lizard.
- **Training makes some local heads worse than the initial weights**. In 8–14 of the 512 heads, the trained checkpoint is farther from the teacher than the initial weights.
- **α is a compromise for the whole layer**. 345 of the 512 teacher heads put at least half of their weight outside the window. For these heads, a small α is better. Thus training probably lowers α for the majority of the heads, and the local heads lose. α fell from 1 to 0.47–0.64 (config 1) and to 0.06–0.66 (LoLCATs recipe).
- **The largest single error of stage 1 is a local head**. Layer 15, head 14 gives approximately 20% of the stage 1 loss. It is at the ceiling of section 3.1.
- **The "A" collapse fits this finding**. Copying "Answer: X" from the 5-shot examples needs heads that find the previous token and its copy. These are local, position-dependent heads. Section 3.3 explains why Lizard probably cannot reproduce them. This is a hypothesis. Experiment X4 can test it.

### 3.5 Relation to the readings of the paper

Document 11 (section 13.4) listed two readings of the paper with a possibly large effect. The new evidence points at both:

- **D2, shared feature maps**. The LoLCATs default gives each head its own map. Appendix B of the paper says that the other design choices follow the LoLCATs defaults. If the paper used one map for each head, the code has 32 times fewer feature-map parameters than the paper.
- **D1, normalization of the gated branch**. Without the denominator, the gated branch has no fixed row sum of 1. Then the ceiling of section 3.1 does not exist, and a head can make its gated output small.

The gate is different. The paper chose the shared scalar gate on purpose. Table 4 of the paper found it best on MMLU for Llama-3-8B ([math formulas](math-formula.md), section 8). Thus a gate for each head is not an obvious fix.

### 3.6 Limits of this evidence

- The attention weights come from one sample of 1,024 tokens. The metrics average over 896 queries for each head.
- Each Lizard layer got the input of the teacher layer (teacher forcing). The model with its own hidden states can behave differently.
- The mechanism of sections 3.1–3.3 agrees with all measurements. But no experiment changed the architecture yet.

## 4. Updated ranking of the causes

| Rank | Cause | Stage | Evidence | Status |
|---|---|---|---|---|
| 1 | **A limit for each head**: shared Lizard parameters (D2, α, gate), a normalized gated branch (D1), no RoPE | Stage 1, and thus everything after it | Heads reach the window-share ceiling. Local heads match worst (correlation −0.75). The largest error is a local head. The relative MSE is 0.24–0.56 also with the best recipe. | Measured pattern. The mechanism is a hypothesis. |
| 2 | **Stage 2 recipe and precision** | Stage 2 | Stage 2 recovers 10.4 PIQA points. It has never run with the recipe of the paper. `create_peft_config` cast the stage 2 model to bf16, also for float32 configs. [Stage 2 on config 1](experiments/stage2-config1.md) fixes this. | Not tested. The run is ready. |
| 3 | **Stage 1 recipe** | Stage 1 | The recipe of the paper in float32 does not beat the LoLCATs recipe. Clipping lowered the loss to 3.5092, 2.6% above the LoLCATs recipe, with no clear change in the accuracies. The LoLCATs recipe in float32 is not measured. | Partly tested. Smaller than expected. |
| 4 | **Stage 1 too short** | Stage 1 | In the float32 runs, the best step is the last evaluation. The loss still decreased. | Probable, not measured |
| 5 | **The "A" collapse on MMLU** | Downstream | Probably a symptom of rank 1: no copy heads (section 3.4) | Hypothesis |
| 6 | **Data, packing and harness** | Both | No new evidence. The teacher is not measured on PIQA and ARC-Easy in this harness yet. | Open, inexpensive to check |

## 5. Proposed experiments

### 5.1 XAI experiments (forward passes only, on the A10)

These experiments estimate the gain of the code changes in section 5.2 before any training.

| # | Experiment | Method | Question | Cost |
|---|---|---|---|---|
| X0 | Teacher baselines | The teacher on PIQA and ARC-Easy with `compare_stages.sh` (`MODELS=teacher`) | What is the target in this harness? | Minutes |
| X1 | Best α for each head | For each head $h$, find the α that minimizes $\Vert \mathbf{y}^{teacher} - \mathbf{y}^{gla} - \alpha_h \mathbf{y}^{window} \Vert^2$. The closed form is $\alpha_h = \langle \mathbf{y}^{teacher} - \mathbf{y}^{gla}, \mathbf{y}^{window} \rangle / \Vert \mathbf{y}^{window} \Vert^2$. Then calculate the new MSE of each layer. | How much of the loss can one α for each head remove (code change C1)? | Approximately 1 hour |
| X2 | Best scale of the gated branch for each head | The same least squares with two numbers for each head: $\beta_h \mathbf{y}^{gla} + \alpha_h \mathbf{y}^{window}$. $\beta_h < 1$ makes the gated output smaller. | How much can a gated branch without a fixed row sum remove (code change C3)? | Approximately 1 hour |
| X3 | Best constant gate for each head | For each layer and head, the constant γ from {0.5, 0.9, 0.99, 0.999, 1} with the lowest MSE. This extends step 2 of section 13.5 of document 11 to each head. | Would a gate for each head help (code change C4)? | A few hours |
| X4 | Repeat test | A random token sequence, then the same sequence again. Measure the loss on the second copy for the teacher, the stage 1 model and the stage 2 model. | Can Lizard copy? A failure supports the explanation of the "A" collapse in section 3.4. | Minutes |
| X5 | Classes of teacher heads | For each teacher head: the weight on the previous token, on the first key, inside 8, 32 and 128 tokens, and the mean distance. Crops of layers 13–15, heads 14 and 23. | Which classes of heads does Lizard miss (local, previous token, sink, long-range)? | Approximately 1 hour, with `attention_weights.py` |

X1 and X2 can use the stored outputs of the layer-wise MSE pipeline. They need the two branch outputs $\mathbf{y}^{gla}$ and $\mathbf{y}^{window}$ separately. Thus they need a small extension of `scripts/layer_mse.py`.

### 5.2 Code changes in the attention layer

Each change keeps the old behavior as the default, behind a model config option. Each change first runs on the single-layer bench of section 13.5 of document 11 (layers 0, 1, 8 and 15). Then it runs as a full stage 1 run of approximately 50 minutes. The checks after each run: the stage 1 validation loss, `layer_mse.py` and `attention_weights.py`.

| # | Change | New parameters (fd32, all 16 layers) | Reading of the paper | Prediction |
|---|---|---|---|---|
| C1 | **One α for each head** (`alpha_blend` with shape (32,)) | +496 | The paper gives one α. A deviation, but a small one. | Local heads get a larger window share. The TV distance of local heads and the MSE of layer 15 decrease. |
| C2 | **One feature map for each head** (D2, the LoLCATs default) | 4,096 → 131,072 for each layer | Probably the reading of the paper (Appendix B) | The kernel can differ between heads. The relative MSE decreases in every layer. |
| C3 | **Normalization of the gated branch** (D1). Variant (i): no denominator, as in the recurrent form of the paper. Variant (ii): one denominator for both branches, as in LoLCATs. | 0 | (i) is a reading of the paper. (ii) is a deviation. | Without the fixed row sum, the ceiling of section 3.1 disappears. The gate saturates less, because it does not need the first key as a sink. |
| C4 | **One gate for each head** (`W_gamma` with 32 outputs) | +31 × 2,048 for each layer | A deviation. Table 4 of the paper prefers the shared gate. | Local heads can decay fast, and long-range heads can keep γ ≈ 1 |
| C5 | **RoPE in the window branch only** | 0 | A deviation. Lizard has no RoPE. | The sharp diagonals of the local heads (layer 0, head 2) appear |
| C6 | **Gate start near 1** (a bias in `W_gamma`, start value approximately +3, thus γ ≈ 0.95) | +16 | Not given in the paper | Less early drift to saturation (section 13.4 of document 11) |

**Order**: C1 first, because it is the smallest change and X1 measures its upper limit before training. Then C2, because it is probably the reading of the paper. Then C3 and C6. C4 and C5 deviate most from the paper. Thus they come last, as thesis contributions.

### 5.3 Hyperparameter experiments

| # | Experiment | Reason | Cost |
|---|---|---|---|
| H1 | **The LoLCATs recipe in float32** (learning rate 1e-2, plateau schedule, the float32 model config) | The missing cell of the 2 × 2 grid of document 11 (section 12): {bf16, float32} × {LoLCATs recipe, recipe of the paper} | Approximately 50 minutes |
| H2 | **Gradient clipping, config 2** | Trained and evaluated: validation loss 3.5092 ([gradient clipping](experiments/gradient-clipping.md)). `layer_mse.py` and `attention_weights.py` have not run on it yet. | XAI only |
| H3 | **A longer stage 1** (4 epochs instead of 2) with the best recipe | In the float32 runs, the validation loss still decreased at the last evaluation | Approximately 100 minutes |
| H4 | **Stage 1 learning rate sweep in float32**: 3e-3 and 5e-3, with the cosine schedule and the minimum learning rate | Between the paper (1e-3) and LoLCATs (1e-2). Appendix B of the paper used such a sweep. | Approximately 50 minutes each |
| H5 | **Stage 2 with the recipe of the paper** on the best stage 1: learning rate 5e-4, cosine, LoRA on q, k, v. The bf16 cast of `create_peft_config` has a fix ([stage 2 on config 1](experiments/stage2-config1.md)). Variants: Lizard parameters frozen or trainable. | Stage 2 decides the final scores. It has never run with the recipe of the paper. | Approximately 3 hours each in bf16. Estimate in float32: 6–9 hours each. |

### 5.4 Sequence and decision rules

| Step | Experiments | Decision |
|---|---|---|
| 1 | X0, H2, X4, X5 | X4 fails for Lizard and not for the teacher: the "A" collapse comes from missing copy heads. Then C1–C5 are the main path. |
| 2 | X1, X2, X3 | Choose the code changes with the largest predicted decrease of the loss. A change that X1–X3 predict below approximately 5% of the loss comes last. |
| 3 | H1 | float32 with the LoLCATs recipe clearly beats config 1: keep the high learning rate for the code changes. Otherwise use config 1 or config 2. |
| 4 | C1, then C2, then C3 or C6 | Keep a change only if the stage 1 loss and the TV distance of the local heads decrease |
| 5 | H3 with the best architecture | The loss still decreases at the end: use the longer run |
| 6 | H5 on the best stage 1 | Success: PIQA and ARC-Easy within approximately 2 points of the teacher in this harness, MMLU at 27 or more, and no single-letter pattern |

## 6. What stays from document 11

- **Measure the attention approximation directly**. Use MMLU only as the final check.
- **Change one thing at a time**.
- **Keep the trainable weights in float32**. For stage 2, `create_peft_config` now keeps the dtype of the model config ([stage 2 on config 1](experiments/stage2-config1.md)).
- **The architecture changes are now justified**. Document 11 advised against them before the recipe tests. The recipe tests finished (section 2), and the measurements point at the architecture (section 3). C1 and C2 stay close to the paper. C3–C6 are thesis contributions, and the thesis must report them as deviations from the paper.
