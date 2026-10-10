# 21. Options for MMLU

**Status:** An analysis and a plan from 2026-10-10. No new runs. It uses a web search on 2026-10-10 and the sources of [document 19](19-next-steps-from-literature.md). The session could not open arXiv, the ACL Anthology or PMLR. Thus most values from papers come from search snippets. Check each value against the paper before the thesis cites it.

## 1. Where MMLU stands

All values of this project come from the same harness (`b281b09`).

| Model | MMLU subset (285 questions) | MMLU, all questions | PIQA | ARC-Easy |
|---|---|---|---|---|
| Teacher (Llama-3.2-1B) | 33.7 | Not measured | 74.4 | 65.3 |
| `window_rope` after stage 2 ([experiment](experiments/window-rope.md)) | 25.6 | Not measured | 73.1 | 63.8 |
| Lizard model of [document 7](07-results.md) (Run 2) | 24.9 | 23.3 | 67.95 | 54.8 |
| LoLCATs attention, stage 1 only ([control](experiments/lolcats-control.md)) | 26.0 | Not measured | 73.5 | 62.8 |

Values of other papers for Llama-3.2-1B, from [document 19](19-next-steps-from-literature.md):

| Source | Teacher | Fully linear methods | Methods with some exact softmax attention |
|---|---|---|---|
| Liger paper, Table 4 | 31.0 | LoLCATs 23.1, Liger 22.4 | – |
| Lizard paper, Table 9 | 31.0 | LoLCATs 27.3, Lizard 29.8 | – |
| STILL | 31.9 | Liger 22.4 | STILL 29.8 |

**Three facts:**

1. **The gap is now specific to MMLU.** On PIQA and ARC-Easy, `window_rope` gets 98.2% and 97.7% of the teacher. On the MMLU subset, it is at chance.
2. **MMLU gives a small signal at 1B.** The teacher is only 6–9 points above chance (25%). On the subset, one SE is approximately 2.6 points. Thus the subset cannot separate differences below approximately 7 points. On all 14,042 questions, one SE is approximately 0.4 points.
3. **At 1B, fully linear methods usually stay near chance.** The Liger paper measured 23.1 for LoLCATs and 22.4 for Liger. Only the Lizard paper (29.8) and STILL (29.8) report values near the teacher. STILL keeps softmax attention over selected tokens.

## 2. Why MMLU is different from PIQA and ARC-Easy

- **The answer format.** In MMLU, the score is the likelihood of one letter ("A" to "D"). The model must find the option that agrees with its knowledge and give the letter of that option. In PIQA and ARC-Easy, the score is the likelihood of the full answer text. Thus a model can know the answer and still fail on the letter step.
- **A few heads do the letter step.** Bick et al. (ICML 2025) call these heads Gather-and-Aggregate (G&A) heads. In a pruned Llama-3.1-8B, one disabled G&A head drops MMLU from 66% to 25%. Recurrent models give smoother attention patterns, without the sharp transitions that G&A needs. Hybrid models give the G&A role to their attention layers.
- **The data of this project agree with this.** The model gives a letter, but it does not select one. After stage 2, `window_rope` gives "A" / "B" / "C" / "D" for 41.4% / 37.5% / 20.7% / 0.4% of the questions. The letter mass is 0.980, but the confidence is only 0.348, and the entropy is 1.908 bits (maximum 2).
- **The context is long.** The 5-shot prompts have up to 2048 tokens. The question and the options are usually in the last approximately 100 tokens, inside the window ([document 7](07-results.md), section 3). But with `gla_norm: row`, the gated branch adds an average over all earlier tokens with the weight 1. A smooth average can blur the sharp attention of the letter step. This is a hypothesis.
- **The knowledge is probably not lost.** The knowledge of a language model is mostly in its MLP layers. Neither stage changes the MLP layers. Stage 2 trains LoRA weights only in q, k, v and o. This is also a hypothesis. Check M3 in section 3 tests it.

## 3. Checks first (no training)

| # | Check | Question | Cost | Code |
|---|---|---|---|---|
| M0 | MMLU, all questions, for the teacher and for `window_rope` after stage 2 | What is the exact gap? | Hours on the A10 (not measured) | None: `TASKS=mmlu` |
| M1 | The MMLU subset 0-shot, for both models | Do the 5 examples before the question cause the failure? | Minutes | One task line in `compare_stages.sh` (`--num_shots 0`) |
| M2 | The MMLU subset with BOS, for both models | Does the missing BOS in the evaluation matter ([document 18](18-gap-timeline.md), section 2)? | Minutes | A small change in `lm_eval_harness/models_huggingface.py`. It does not accept `add_special_tokens=True` for causal models. |
| M3 | MMLU with rotated answer options ([document 19](19-next-steps-from-literature.md), A3) | Does the model know the answer but fail on the letter? | Minutes | A small script |
| M4 | The MMLU subset with the gated branch switched off or scaled down at inference | Does the gated branch blur the letter step? | Minutes | `scripts/ablate.py` supports only v1. It needs v2 support. |
| M5 | The G&A heads of the teacher: switch off one layer, then one head, on a small MMLU set | Which heads must stay exact? | 16 layer runs, then 32 head runs for each selected layer | A new script |

**How to read the results:**

| Result | Meaning | Next option (section 4) |
|---|---|---|
| M1: 0-shot near the teacher, 5-shot at chance | The long context causes the failure | T5, then T6 |
| M3: the same answer content at each position, but a wrong letter | The knowledge is there. The letter step fails. | T1 or T2, T3, T4 |
| M3: a fixed letter preference at each position | The letter step fails, and the knowledge cannot be checked | T1 or T2, T3 |
| M4: MMLU increases without the gated branch | The gated branch blurs the letter step | T5 |
| M5: the G&A heads are in one or two layers | These layers must stay exact | T1 with these layers, or T2 with these heads |

## 4. Options that change the model (after the checks)

All options change the method of the paper. Thus the thesis must report them as extensions ([document 19](19-next-steps-from-literature.md), part B).

| # | Option | Evidence | Expected effect on MMLU | Cost | Code |
|---|---|---|---|---|---|
| T1 | **Keep one or two layers as softmax** (`softmax_attentions`), selected by M5 or by the KL divergence ([document 19](19-next-steps-from-literature.md), B1 and B2) | LoLCATs at 8B: 50% softmax layers give 65.8 against 52.8 (teacher 66.6). Lizard at 8B with 50% softmax layers: 65.1 (teacher 66.1 or 66.6, the versions of the paper differ). Bick et al.: hybrids give the G&A role to attention. | The largest, if the G&A heads are in these layers | One stage 1 (approximately 2.5 hours on the A100) and one stage 2 (approximately 3 hours on the H200) | None, a config option |
| T2 | **Softmax attention only for the G&A heads**, and Lizard for the other heads of the layer | DuoAttention: a few retrieval heads need all tokens. The other heads work with a window and sinks. Bick et al. | Near T1, with less memory | As T1 | A new head mask in v2 |
| T3 | **Stage 2 with the KL divergence to the teacher logits** ([document 19](19-next-steps-from-literature.md), B3) | RADLADS, MOHAWK and Mamba in the Llama use the teacher logits | Medium. It transfers the answer distribution of the teacher, also over the letters. | Approximately 3–4 hours | A new loss in the stage 2 trainer |
| T4 | **A short stage on the answer format**, with multiple-choice data that is not MMLU ([document 19](19-next-steps-from-literature.md), B6) | "Stuck on A" (from document 19). This search could not find this paper again. | Helps only if M3 shows that the knowledge is there | Approximately 1,000 steps | A new data loader |
| T5 | **Less gated weight at long positions:** step 3 of [document 20](20-layer-mse-for-piqa-arc.md), or `hybrid` with the gate start at approximately 0.95 (`gate_bias_init: 3.0`) | The best weight of the gated branch is 0.79 at positions 512–2047 ([RoPE in the window branch](experiments/window-rope.md), finding 7). With `hybrid`, the layers with an open gate had a lower MSE ([experiment](experiments/window-rope-hybrid.md), finding 4). | Helps if M1 or M4 show the long context as the cause | One stage 1 and one stage 2 | Step 3: a new option. Gate start: a config. |
| T6 | **A sparse global cache** (LoLA) | No training. It adds a small cache of the tokens that linear attention cannot recall, for window and linear models such as LoLCATs. Pass-key retrieval at 4K tokens: 0.6% to 97.4%. | Probably small. The MMLU options are inside the window. | Inference code only | New |
| T7 | **Other data:** pretraining data, more tokens, or few-shot-like sequences ([document 19](19-next-steps-from-literature.md), B7) | LoLCATs: RedPajama in place of Alpaca adds approximately 2 MMLU points. RADLADS uses 350–700M tokens. | Small to medium | 10× or more of the current cost | A new data loader |

**The cost of T1 and T2.** A softmax layer or head keeps all keys and values. Its memory grows with the sequence length. One layer of 16 is 6.25% of the attention memory of the teacher. The thesis must give this cost with the result.

## 5. Recommendation

1. **M0, M1 and M2 in one session on the A10.** M1 needs one task line. M2 needs a small change in the model wrapper of the harness. M0 gives the exact gap before any training.
2. **M3 and M4.** Each needs a small script. They separate "the knowledge is lost" from "the letter step fails" and show the effect of the gated branch.
3. **M5, then T1 with one layer.** Bick et al. and the hybrid results of LoLCATs and Lizard make T1 the option with the largest expected effect. M5 selects the layer. Without M5, the KL selection of [document 19](19-next-steps-from-literature.md) (B2) can select it.
4. **T3 as the change of the recipe.** It can run alone or together with T1.

**A realistic target at 1B:** approximately 28–30 on all MMLU questions, as the Lizard paper (29.8) and STILL (29.8). The teacher has 31.0–31.9 in the papers and 33.7 on the subset of this project.

**Stop rules:**

- Do not compare runs on the MMLU subset alone. Its SE is 2.6 points. Use all questions (M0) for the final values.
- Do not stop at the MMLU value. Check PIQA and ARC-Easy too, because T1–T5 also change the short-context behavior.

## 6. Sources

The search ran on 2026-10-10. The session could not open the full texts.

- Bick, Xing and Gu, "Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism", ICML 2025: [PMLR](https://proceedings.mlr.press/v267/bick25a.html), [arXiv 2504.18574](https://arxiv.org/abs/2504.18574)
- DuoAttention, ICLR 2025: [arXiv 2410.10819](https://arxiv.org/abs/2410.10819), [project page](https://hanlab.mit.edu/projects/duo-attention)
- LoLA, "Low-Rank Linear Attention With Sparse Caching": [arXiv 2505.23666](https://arxiv.org/pdf/2505.23666)
- Lizard: [arXiv 2507.09025](https://arxiv.org/pdf/2507.09025), [ACL 2026](https://aclanthology.org/2026.acl-long.1613/), [Hugging Face](https://huggingface.co/papers/2507.09025)
- LoLCATs: [arXiv 2410.10254](https://arxiv.org/pdf/2410.10254)
- Liger: [arXiv 2503.01496](https://arxiv.org/pdf/2503.01496)
- STILL: [arXiv 2602.02180](https://arxiv.org/abs/2602.02180)
- KL-guided layer selection, ICLR 2026: [arXiv 2512.20569](https://arxiv.org/html/2512.20569v1)
- RADLADS: [arXiv 2505.03005](https://arxiv.org/html/2505.03005v4)
- Mamba in the Llama: [arXiv 2408.15237](https://arxiv.org/pdf/2408.15237)
- Llamba: [arXiv 2502.14458](https://arxiv.org/pdf/2502.14458)
- Based, the recall and memory trade-off: [blog](https://hazyresearch.stanford.edu/blog/2024-03-03-based)
- OLMES, the multiple-choice and cloze formats: [arXiv 2406.08446](https://arxiv.org/html/2406.08446v1)
