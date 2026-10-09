# 20. Lower layer-wise MSE for PIQA and ARC-Easy

**Status:** An analysis and a plan from 2026-10-09. Nothing has run. The analysis uses the figure [`lizard_vs_lolcats_attention.png`](experiments/xai-layer-wise-mse/lizard_vs_lolcats_attention.png), the stored PIQA and ARC-Easy prompts, and a web search. The session could not open arXiv. Thus the values from papers come from search snippets. Check each value against the paper before the thesis cites it.

## 1. What the figure shows

Panel (a) gives 1000 × MSE for each layer: Lizard Run 1 stage 1 (fd128, LoLCATs recipe) and the LoLCATs attention. Panel (b) gives the difference LoLCATs − Lizard.

| Layers | Lizard − LoLCATs (1000 × MSE) | Share of the total difference |
|---|---|---|
| 0–3 | 0.18, 0.32, 0.34, 1.30 | 4.8% |
| 4–9 | 2.28 to 3.54 | 38.6% |
| 10–14 | 1.81 to 2.44 | 22.6% |
| 15 | 15.26 | 34.0% |

- LoLCATs has a lower MSE in every layer ([XAI: layer-wise MSE](experiments/xai-layer-wise-mse.md), Run 2).
- Panel (b) uses absolute values. Thus it hides layer 0, which has the largest ratio (36.9×). The [relative plot](experiments/xai-layer-wise-mse/lizard_vs_lolcats_attention_relative.png) shows it.
- The MSE comes from Alpaca sequences of 2048 tokens. 94% of the query positions (128 to 2047) have keys outside the window of 128 tokens.

## 2. What PIQA and ARC-Easy test

### The papers

| Benchmark | Paper | Content | Set in this harness |
|---|---|---|---|
| PIQA | Bisk et al., "PIQA: Reasoning about Physical Commonsense in Natural Language", AAAI 2020 | A goal and two solutions. One solution is correct. Humans get approximately 95%, and the models of 2019 approximately 77%. | Validation, 1,838 questions, 0-shot |
| ARC-Easy | Clark et al., "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge", 2018 | Grade-school science questions with usually 4 answers. 7,787 questions: the Easy set has 5,197, the Challenge set 2,590. | Easy test set, 2,376 questions, 0-shot |

### How the harness scores them

The stored write-out files of the [LoLCATs control](experiments/lolcats-control.md) show the prompt of each choice:

```text
Question: Remove seeds from  strawberries
Answer: Blend the strawberries, pour the mixture through a fine-mesh strainer ...
```

The harness calculates the log-likelihood of each answer after "Answer:". The answer with the highest value counts (`acc`). `acc_norm` divides by the length of the answer. In PIQA, the two answers often differ in a few words only, for example "Blend" against "Chop up". Thus the score depends on the next-token probabilities over the whole answer.

### The prompts are short

From the write-out files, the longest choice of each question:

| Task | Median | 95th percentile | Maximum | Longer than 512 characters |
|---|---|---|---|---|
| PIQA | 132 characters | 355 characters | 1,106 characters | 1.1% |
| ARC-Easy | 139 characters | 311 characters | 802 characters | 0.3% |

With approximately 4 characters for each token, the 95th percentile is approximately 88 tokens (PIQA) and 77 tokens (ARC-Easy). Thus for almost every question, all keys are inside the window of 128 tokens. Then the three models compute these outputs:

| Model | Output on a prompt shorter than the window |
|---|---|
| Teacher | Softmax attention with RoPE over all tokens |
| LoLCATs | The same, up to rounding ([LoLCATs control](experiments/lolcats-control.md), finding 2) |
| Lizard | The gated branch over all tokens, plus α × the window branch without RoPE over all tokens. The total weight of a row is 1 + αρ ([document 15](15-attention-math-side-by-side.md), claims B and C). |

**Consequence 1.** On PIQA and ARC-Easy, the window branch of Lizard must do the work of the teacher. Without RoPE, it cannot give the same weights (claim F). The gated branch adds weight to tokens that the window already counts.

**Consequence 2.** The stage 1 loss gives each query position the same weight. Positions 0–127 are only 6.25% of the 2048 positions. Thus the stage 1 loss mostly measures the long-sequence case, which PIQA and ARC-Easy do not use. This can explain why the stage 1 loss does not predict PIQA and ARC-Easy over the Lizard runs ([document 16](16-optuna-plan.md), section 1). This is a hypothesis. X1 in section 4 tests it. **Result (2026-10-09):** the hypothesis does not hold. The MSE of positions 0–127 does not predict PIQA or ARC-Easy over the Lizard checkpoints either ([XAI: MSE by token position](experiments/xai-position-mse.md), finding 6).

## 3. Is the window, the gated branch or Hedgehog wrong?

The code passes 26 tests against the jku-thesis reference, and the decode output equals the dense output ([document 8](08-verification.md)). Thus a code bug is improbable. "Wrong" here means a design or a reading of the paper that does not work at 1B.

| Part | Difference to the teacher and to LoLCATs | Evidence in these notes | Isolating test |
|---|---|---|---|
| **Window branch (SWA)** | No RoPE (claim F). The 4 sinks act as one sink (claim H). The window weight α is one value for all heads. | Local heads have the worst match, and layer 0 is the worst layer ([sample attention weights](experiments/xai-sample-attention-weight.md), findings 4 and 7). The previous-token head of layer 0 gets a small α ([document 13](13-lizard-attention-v2.md)). In layer 0, α is 0.042, but LoLCATs keeps its window ([LoLCATs control](experiments/lolcats-control.md), finding 8). | X2 (window with and without RoPE), then `window_rope` |
| **Gated branch (GLA)** | One scalar gate for all heads. GLA has a gate for each head and each key dimension. The gate starts at 0.5, the GLA gate at approximately 0.96. The branch covers the window tokens too (claim B). Its own row sum is 1, so a head cannot switch off its far weight. | In R1b, the shared gate of layer 15 fell to 0.128. Heads 14 and 23 improved, and 28 of 32 heads got worse ([document 13](13-lizard-attention-v2.md), finding 4). | X2 (gated branch alone and in combination), then the gated branch outside the window, `gate_per_head` |
| **Hedgehog maps** | One map for the 32 heads of a layer. LoLCATs has one map for each head, with 28× more parameters at fd128. The maps start from random values (standard deviation 0.02). Thus at the start, the kernel is almost constant. The LoLCATs maps start as the identity. The maps learn only from the output MSE. | R1b added a map for each head and lowered the loss by 23.2%. But ARC-Easy fell by 6.6 points ([document 13](13-lizard-attention-v2.md)). | X3 (monotonicity and entropy) |

**A note on the paper.** The Liger paper writes its window without RoPE, but the Liger code applies RoPE ([document 15](15-attention-math-side-by-side.md), table). Thus a formula without RoPE in a paper does not prove that the code has no RoPE. The Lizard paper has no public code. Thus the Lizard window can also have RoPE in the code of the authors. This is a possibility, not a fact. Step A4 of [document 19](19-next-steps-from-literature.md) asks the authors.

**Ranking for PIQA and ARC-Easy.** The window branch is the most probable cause, because short prompts use almost only the window (section 2). The gated branch is second, because it adds weight inside the window. The Hedgehog maps act only through the gated branch. They matter more for long sequences and for MMLU.

## 4. XAI methods

| # | Method | Question | Source | Status | Cost |
|---|---|---|---|---|---|
| X1 | MSE for each bucket of query positions: 0–127, 128–511, 512–2047 | In which case is the error? Does the MSE of positions 0–127 predict PIQA and ARC-Easy over the 7 Lizard checkpoints and LoLCATs? | New | Done on 2026-10-09 ([XAI: MSE by token position](experiments/xai-position-mse.md)). Lizard misses 35–63% of the teacher output at positions 0–127, most of all in layers 2 and 3. This MSE does not predict the accuracy. | Forward passes on the A10, minutes for each checkpoint |
| X2 | Branch decomposition with the best mixing weights | For each head, fit the teacher output with the least-squares weights of each candidate branch. Candidates: the window without RoPE, the window with RoPE (teacher softmax over the window), the gated branch, and pairs of these. The remaining error is the lowest error that a training of α can give. | New, from the claims of [document 15](15-attention-math-side-by-side.md) | Script `branch_fit.py` ([XAI: branch decomposition](experiments/xai-branch-fit.md)). Not run yet. | Forward passes, approximately 1 hour |
| X3 | Spikiness and monotonicity of the gated weights | Is the entropy of the gated weights near the teacher? Do the kernel values φ(q)·φ(k) keep the order of the teacher scores q·k (rank correlation for each row)? | Hedgehog paper (Zhang et al., ICLR 2024) | Not done. `attention_weights.py` already has the weights. | Minutes |
| X4 | Activation patching: teacher attention in all layers outside a set S | Which layers cost the accuracy? Use the log-likelihood difference between the right and the wrong answer, not only the accuracy. | Heimersheim and Nanda 2024. [Faster feedback loop](experiments/fast-feedback-loop.md), step 1. | Planned | Approximately 4 hours on the A10 |
| X5 | Head types: previous-token, induction, sink, local and long-range heads | Which head types does Lizard miss? Previous-token heads feed the induction heads of later layers. | Olsson et al. 2022 | In part ([sample attention weights](experiments/xai-sample-attention-weight.md), finding 4) | Minutes |
| X6 | Logit lens or tuned lens on PIQA prompts | In which layer does the prediction of Lizard leave the teacher? | Belrose et al. 2023. Paulo et al. 2024 used it on RNNs. | Idea ([XAI ideas](experiments/xai.md), section 3) | Hours, with a trained lens |

Done already:

- the MSE of each layer and head ([XAI: layer-wise MSE](experiments/xai-layer-wise-mse.md))
- the attention weights ([sample attention weights](experiments/xai-sample-attention-weight.md))
- the ablation of one branch at inference ([document 7](07-results.md), section 5)

### Decision rules for X1 and X2

- **X1:** if the MSE of positions 0–127 orders the 8 checkpoints as PIQA does, use this MSE as the screen for stage 1 runs. Otherwise, keep the accuracy as the screen.
- **X2, window:** compare the window with RoPE and the window without RoPE at positions 0–127. If the error with RoPE is much lower, the missing RoPE is the cause. Then `window_rope` comes first.
- **X2, gated branch:** add the gated branch to the window with RoPE. If the error at positions 128–2047 does not decrease, the gated branch or its maps are the limit. Then X3 decides between the maps and the gate.

## 5. What to do, in order

| # | Change | Part | Target | Code | Cost |
|---|---|---|---|---|---|
| 1 | X1 and X2 | – | Select the part to change | Small | A10, approximately 1 hour, free |
| 2 | `window_rope` in the setup of Run 1 | Window | Positions 0–127, thus PIQA and ARC-Easy. Layer 0. | None, the option exists ([document 13](13-lizard-attention-v2.md)) | One stage 1: approximately 50 minutes and $5 on the H200 (bf16). Or a few layers on the A10 ([faster feedback loop](experiments/fast-feedback-loop.md), step 4). |
| 3 | The gated branch only for keys outside the window | Gated branch | Short prompts: no double weight inside the window | A new option. Claim A of [document 15](15-attention-math-side-by-side.md) gives the form. | As step 2 |
| 4 | A larger weight for the loss of positions 0–127, or a part of the data in shorter chunks | Loss | The case of PIQA and ARC-Easy | A new option in the trainer | As step 2 |
| 5 | `gate_per_head` | Gated branch | Layer 15 (34% of the MSE difference) | None, the option exists | As step 2 |
| 6 | A map for each head with an identity start, or a loss on the attention weights as in Hedgehog | Hedgehog | Long sequences, MMLU | A new option | As step 2 |

- Use steps 2 and 3 with the shared denominator (`gla_norm: hybrid`). Then a prompt shorter than the window gets the teacher weights, multiplied by the window share ρ. The sinks get the rest. If the sinks get almost no weight, the output is the teacher output. LoLCATs has this property, and it is the probable cause of its PIQA score (finding 2 of the control).
- Steps 2 and 3 move Lizard toward LoLCATs and Liger. Thus the thesis must report them as extensions, not as the reproduction ([document 19](19-next-steps-from-literature.md), part B).
- Change one part for each run. Evaluate each run on PIQA and ARC-Easy, not only on the MSE ([document 16](16-optuna-plan.md), section 1).

## 6. Sources

The search ran on 2026-10-09. The session could not open the full texts.

- PIQA: Bisk et al., AAAI 2020, [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/6239), [arXiv 1911.11641](https://arxiv.org/abs/1911.11641v1)
- ARC: Clark et al. 2018, [arXiv 1803.05457](https://arxiv.org/abs/1803.05457), [AI2 data page](https://allenai.org/data/arc)
- Hedgehog: Zhang et al., ICLR 2024, [arXiv 2402.04347](https://arxiv.org/pdf/2402.04347)
- GLA: Yang et al., ICML 2024, [PMLR](https://proceedings.mlr.press/v235/yang24ab.html)
- Based (linear attention plus a sliding window for local token shifts): Arora et al. 2024, [arXiv 2402.18668](https://arxiv.org/pdf/2402.18668)
- Attention sinks (StreamingLLM): Xiao et al., ICLR 2024, [arXiv 2309.17453](https://arxiv.org/pdf/2309.17453)
- Multiple-choice normalization in the harness: [EleutherAI blog](https://blog.eleuther.ai/multiple-choice-normalization/)
- Activation patching: Heimersheim and Nanda 2024, [arXiv 2404.15255](https://arxiv.org/pdf/2404.15255)
- Induction heads: Olsson et al. 2022, [Transformer Circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/)
- Tuned lens: Belrose et al. 2023, [arXiv 2303.08112](https://arxiv.org/pdf/2303.08112v5)
- Lenses on RNNs: Paulo et al. 2024, "Does Transformer Interpretability Transfer to RNNs?", [arXiv 2404.05971](https://arxiv.org/pdf/2404.05971)
