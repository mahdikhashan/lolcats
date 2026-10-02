# Experiment ideas: XAI for the distillation

**Status:** ideas only. No part of this document has run yet.

## Question

Does the distillation of Llama-3.2-1B into the Lizard model work as expected? The question has four parts:

1. Do the model and the new Lizard parameters learn?
2. Do the distributions of the layers match the teacher?
3. Does the initialization affect the result, and would another initialization method help?
4. Which interpretability (XAI) methods for knowledge distillation can answer these questions?

The architecture change: each attention layer has two branches, the gated branch (GLA) and the window branch (SWA) with sink logits.

| Model | MMLU 5-shot | Source |
|---|---|---|
| Teacher, this harness | 33.7 (MMLU subset) | [Document 7](../07-results.md) |
| Teacher, Meta model card | 32.2 | Reference 1 |
| Teacher, Lizard paper | 31.0 | Table 9 of the paper |
| Lizard model after stage 1 | 22.5 (MMLU subset) | [Stage difference](stage-difference.md), quick check 2 |
| Lizard model after stage 2 | 24.9 (MMLU subset), 23.3 (all questions) | Stage difference, quick check 2, and document 7 |

The ideas below come from an external review. A paragraph that starts with **Repository check** compares an idea with the code or with the results of this project. These checks occurred during the writing of this document.

## Main idea

Treat interpretability as a debugging and evaluation layer for the distillation, not only as a way to explain single predictions. An MMLU score of 22% does not immediately mean that the distillation failed. The task is to find where the model loses information.

## 1. Output level: does the student reproduce the teacher?

Do not start with MMLU. Give the same input sequences to the teacher and to the student, and compare their logits with these metrics:

- KL divergence,
- cross-entropy,
- top-1 agreement,
- top-5 agreement,
- cosine similarity of the logits,
- Jensen-Shannon divergence,
- the probability that the student gives to the top-1 token of the teacher.

How to read the result:

- If the agreement is good but MMLU stays at 22%, the distillation is probably not the main problem. Then the evaluation, the prompts or the generalization of the architecture can be the cause.
- If the agreement is poor, continue with the next sections.

## 2. Layer-by-layer representation matching

The review calls this probably the most important experiment for the thesis.

- Calculate the cosine similarity cos(h_T^l, h_S^k) between the hidden states of every teacher layer l and every student layer k.
- Show the result as a heatmap (teacher layers × student layers).
- Do not assume that teacher layer i matches student layer i. The GLA + SWA architecture can produce a different sequence of representations.
- A diagonal heatmap means that each student layer matches the same teacher layer.
- A shifted pattern is also possible, for example teacher layer 3 → student layer 2 and teacher layer 12 → student layer 10. Such a pattern would show that the architecture moves information to a different depth.

There is precedent for distillation through intermediate representations: CoDIR (reference 2).

## 3. Logit lens and tuned lens

Take the hidden state after every layer and decode it into vocabulary predictions. This shows at which layer the information becomes predictive.

- Compare the entropy of these predictions layer by layer, for the teacher and for the student.
- Compare the predictions themselves. For example, for "The capital of France is", the teacher can predict "Paris" at layer 8. The student can predict "London" at layer 8 and "Paris" only at layer 12. Such a result means that the architecture delays the formation of useful representations.
- The tuned lens is a variant of the logit lens that learns a small decoder for each layer.
- DistillLens (reference 3) applies a logit-lens approach to knowledge distillation. It aligns intermediate teacher and student representations in vocabulary space. The review names it as one of the most relevant papers for this experiment.

**Repository check:** In Llama, the LM head reads the hidden state after the final RMSNorm (`self.norm` in `LolcatsLlamaModel`, `src/model/modeling_llama.py`). Thus a logit lens applies `model.norm` and then `lm_head` to each hidden state.

## 4. Attribution between the gated branch and the window branch

Find out which branch learns. Record these values for each layer:

- the output norm of each branch,
- the gradient norm of each branch,
- the update norm of the parameters of each branch.

Three outcomes are possible:

- **A:** The window branch does not learn. The architecture becomes mainly a gated-branch model.
- **B:** The gated branch does not learn.
- **C:** One branch dominates, for example 95% against 5%. Then the two-branch attention is present, but in function it is close to one branch.

**Repository check:**

- The two branches use the same q/k/v projections. The parameters of the gated branch are `phi_q`, `phi_k` and `W_gamma`. The window branch has the sink logits (`meta_tokens`). α (`alpha_blend`) sets the weight of the window branch against the gated branch.
- [Document 7](../07-results.md) disabled each branch at inference on PIQA. Both branches are in use: without the window branch, PIQA is 57.18, and without the gated branch, 52.34 (full model 67.95).
- The [stage difference experiment](stage-difference.md) found α between 0.042 and 0.648 in all layers. Thus the window branch has less weight than the gated branch in every layer. It also found the gate near 1 in layers 1–15, so the decay of the gated branch has almost no effect there.
- Output norms and gradient norms of the branches are not measured yet.

## 5. Ablations of the branches and the sinks

Run the trained student in five modes:

| Mode | Description |
|---|---|
| A | Gated branch + window branch + sinks (the trained model) |
| B | Gated branch only |
| C | Window branch only |
| D | Gated branch + window branch, without sinks |
| E | Original attention (the teacher) |

Evaluate all five modes on the validation loss, the KL divergence to the teacher, MMLU, PIQA, ARC-Easy and the perplexity. The review gave example values for such a table. They are illustrations, not measurements, so this document does not copy them.

**Repository check:** The project measured part of this table on PIQA, with the stage 2 model ([document 7](../07-results.md), `ablate.py`):

| Mode | PIQA accuracy |
|---|---|
| A. Trained model | 67.95 |
| B. Gated branch only, (1 + α) · GLA | 57.18 |
| C. Window branch only, (1 + α) · AWA | 52.34 |
| D. Without sinks | Not measured. `ablate.py` has no such mode yet. |
| E. Teacher | Not measured in this harness. The paper reports 74.1. |

Mode D needs a new mode in `ablate.py`. One way is to set the sink logits to a very large negative value at inference, so that they have no effect in the denominator.

## 6. Initialization

The review puts initialization high on the list of things to examine. The changed attention computes a different function, and pretraining optimized the Llama weights for the original function. New parameters with random values can damage the function, even if most weights come from pretraining. The review proposes three experiments:

- **A. Random initialization:** the new parameters start with random values. The review assumed this as the current baseline.
- **B. Function-preserving initialization:** the new layer starts with student(x) ≈ teacher(x). For example, one branch reproduces the original attention, and the other branch starts near zero and learns later.
- **C. Teacher-informed initialization:** teacher activations initialize or train the new parameters.

**Repository check:**

- The current initialization is not fully random. q/k/v/o come from the teacher. Of the five Lizard parameters, only `phi_q` and `phi_k` start random (normal distribution, standard deviation 0.02). `W_gamma` starts at 0 (γ = 0.5), the sink logits start at 0, and α starts at 1 ([document 1](../01-lizard-in-lolcats.md)).
- Stage 1 is already a teacher-informed initialization (experiment C). It trains the Lizard parameters to reproduce the attention output of each teacher layer before stage 2 starts. The paper reports that MMLU of its 8B model decreases from 61.2 to 50.8 without stage 1 (Table 6).
- An exactly function-preserving start (experiment B) is not possible with this architecture. The Lizard layer has no RoPE, and the window covers only 128 tokens. The teacher attention uses RoPE over the full sequence ([document 9](../09-paper-comparison.md)).
- With an unbounded window and no sinks, the window branch equals causal softmax attention, but without RoPE ([document 8](../08-verification.md)). Thus the window branch, not the gated branch, is the closer start point for experiment B.
- The gate start value (W_γ = 0) is a choice of this implementation, because the paper does not give one (document 9). The stage difference experiment found the gate near 1 after stage 1. A stage 1 run with another gate start value can show if the start value or the recipe causes this saturation.

## 7. Parameter movement

Measure the relative change ‖W_after − W_before‖ / ‖W_before‖ of each parameter group. Record these values for each group:

- the weight norm,
- the gradient norm,
- the update norm,
- the ratio of update to weight.

The groups are q_proj, k_proj, v_proj, o_proj, the parameters of each branch, the sink logits, the MLP and the RMSNorm layers. The gradient norms are especially important: a parameter can have a value that is not zero and still receive almost no useful gradient.

**Repository check:**

- In this project, most groups are frozen by design. Stage 1 trains only the Lizard parameters. Stage 2 trains only LoRA on q/k/v/o. The embeddings, the MLP, the RMSNorm layers and the LM head never change.
- The relative change is not defined for parameters that start at 0 (`W_gamma`, the sink logits). For them, use the absolute norm.
- The stage difference experiment already logs the change from the initial values:
  - φ weight RMS: 0.02 → 0.15–0.22,
  - ‖W_γ‖: 0 → 3.59–5.13,
  - α: 1 → 0.042–0.648,
  - sink logits: 0 → 0.50–2.00.
- The trainer does not log gradient norms ([document 10](../10-open-issues-and-next-steps.md)).
- Factor 0 of section 12 of the [gap analysis](../11-gap-analysis.md) found that bf16 rounding can discard updates. Thus an additional metric is useful: the fraction of updates that rounding discards.

## 8. Activation statistics

Collect these statistics for each layer and each branch: mean, standard deviation, L2 norm, maximum, minimum, RMS and the percentage of zeros. Compare them between the teacher and the student. They can show these problems:

- exploding activations,
- vanishing activations,
- bad normalization,
- imbalance between the branches,
- a mismatch in the residual stream.

## 9. Comparison of attention patterns

Compare the attention maps of the teacher, of the gated branch and of the window branch with a similarity measure. Do not overinterpret attention weights as explanations. Attention maps help with diagnosis, but they are not sufficient evidence that a mechanism contains the knowledge. For this architecture, the review prefers causal interventions (section 10).

**Repository check:** The implementation is dense and calculates the full L × L weight matrix of both branches ([document 2](../02-compute-and-cost.md)). Thus the attention maps of both branches are available directly. The teacher attention uses RoPE, and the Lizard branches do not. `attention_weights.py` plots the weights of the teacher and of Lizard and measures their distance ([XAI: sample attention weights](xai-sample-attention-weight.md)).

## 10. Activation patching

Replace one activation of the student with the matching activation of the teacher, and let the student continue. If the output improves much, that layer is a bottleneck.

- Do this for each layer in sequence, and measure MMLU accuracy each time.
- For example, if a patch of layer 8 increases MMLU much more than a patch of the other layers, examine layer 8.
- With two branches, patch the gated branch, the window branch, and both branches of one layer. This can show which branch loses the knowledge.

**Repository check:**

- The LoLCATs wrapper already has a switch for each layer. With `train_attention = True`, a layer outputs softmax attention with RoPE from its own q/k/v projections, as the teacher does (`LolcatsLizardAttention.forward`, [document 1](../01-lizard-in-lolcats.md)). Thus a patch of one attention layer is one attribute change on that layer. This is not tested yet.
- For the stage 1 model, this gives exactly the teacher attention of that layer for the current input. For the stage 2 model, LoRA changes q/k/v, so the patched attention differs from the teacher.
- The teacher has one attention output for each layer, not one for each branch. Thus a patch of one branch needs a definition, for example "replace the gated branch with teacher attention, and keep the window branch".

## Proposed diagnostic script

The review proposes one script, `analyze_distillation.py`, with this output:

```text
results/
├── logits/
│   ├── kl.json
│   ├── topk_agreement.json
├── representations/
│   ├── teacher_student_cosine.npy
│   ├── layer_similarity.png
├── activations/
│   ├── norms.json
│   ├── activation_statistics.json
├── attention/
│   ├── gla_attention.npy
│   ├── swa_attention.npy
├── parameters/
│   ├── gradient_norms.json
│   ├── update_norms.json
├── logit_lens/
│   ├── teacher.json
│   ├── student.json
└── ablations/
    ├── gla_only.json
    ├── swa_only.json
    ├── no_sink.json
```

Its summary has these parts:

1. MMLU 5-shot of the teacher and the student.
2. Output alignment: KL(student ‖ teacher), top-1 agreement, top-5 agreement.
3. Representation alignment: the best teacher → student layer mapping, and the mean cosine similarity.
4. Branch use: the contribution of each branch.
5. Parameter learning: the ratio of update to weight for the gated branch, the window branch and the sink logits.
6. Activation statistics: the activation RMS of each branch.
7. Logit lens: the first predictive layer of the teacher and of the student.
8. Conclusion: the probable bottleneck.

The review gave example values for this summary. They are illustrations, not measurements.

**Repository check:** `compare_stages.py` already logs part of this: the gate statistics, the Lizard parameter values and the checkpoint check. Branch ablations are in `ablate.py`. A new script can use the same loading path (`load_model_from_checkpoint`). The MSE of each layer against the teacher attention is in `layer_mse.py` ([XAI: layer-wise MSE](xai-layer-wise-mse.md)).

## Check the MMLU evaluation first

Before an interpretation of the 22%, check the evaluation. MMLU results are sensitive to the prompt, the answer extraction and the evaluation code. An independent replication reported values from 15.7% to 42.9% for Llama-3.2-1B-Instruct, depending on these choices (reference 4).

| Model | This harness | Reference |
|---|---|---|
| Teacher (original Llama-3.2-1B) | 33.7 (MMLU subset, [document 7](../07-results.md)) | 32.2 (Meta), 31.0 (paper) |
| Lizard model after stage 1 | 22.5 (MMLU subset) | Not measured |
| Lizard model after stage 2 | 24.9 (MMLU subset), 23.3 (all questions) | 29.8 (paper, its own Lizard model) |

How to read the result:

- If the original Llama also gets approximately 22–25% in this harness, the problem is mainly the evaluation.
- If the original Llama gets approximately 32% and the Lizard model 22–23%, the architecture change damages the capability. Then the next steps are initialization and function preservation.

**Repository check:** The second case applies. In the same harness, the teacher gets 33.7 on the MMLU subset. Thus the harness is not the cause of the gap ([document 8](../08-verification.md)).

## Priority order

The review recommends this sequence and does not start with advanced XAI methods.

| Step | Method | Status in this project |
|---|---|---|
| 1 | Check the MMLU evaluation | Done for the teacher on the MMLU subset (33.7) |
| 2 | Compare teacher and student logits on the same text (section 1) | Not done |
| 3 | Layer × layer representation similarity (section 2) | Not done |
| 4 | Gradient norms and update norms of each branch (sections 4 and 7) | Partly done: parameter values against initial values. No gradient norms. |
| 5 | Activation statistics (section 8) | Not done |
| 6 | Logit lens or tuned lens (section 3) | Not done |
| 7 | Ablation of each branch (section 5) | Partly done: PIQA, stage 2 model ([document 7](../07-results.md)) |
| 8 | Function-preserving initialization (section 6) | Not done. An exact version is not possible with this architecture. |
| 9 | Activation patching (section 10) | Not done |

This sequence should answer the main question in two parts. At which point does the Lizard model stop behaving like the original Llama? Is the cause the initialization, the optimization, a representation mismatch or the new attention?

**Note from this project:** The [stage difference experiment](stage-difference.md) places the damage in stage 1 and found a saturated gate. Thus steps 2, 3, 6 and 9 are most useful on the stage 1 model against the teacher. On the stage 1 model, LoRA does not change q/k/v, so a patch with teacher attention is exact.

## References

The review gave these references. This project did not check them.

1. Meta, Llama 3.2 model card: base Llama-3.2-1B, MMLU 5-shot 32.2. <https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md>
2. CoDIR: Contrastive Distillation on Intermediate Representations for Language Model Compression. <https://arxiv.org/abs/2009.14167>
3. DistillLens: Symmetric Knowledge Distillation Through Logit Lens (2026). <https://arxiv.org/abs/2602.13567>
4. An attempt to replicate the MMLU results of Llama-3.2-1B-Instruct. <https://github.com/jaceroldan/llama-3.2-benchmark>
