# 19. Next steps from the literature

**Status:** A web search on 2026-10-08 for the next steps of the reproduction and for ways to improve the scores. No new runs, no code. The session could not open arXiv, the ACL Anthology or most paper pages. Thus most values below come from search snippets. Check each value against the paper before the thesis cites it.

## 1. What other work says about this gap

| Source | Finding | Meaning for this project |
|---|---|---|
| Liger paper, Table 4 (Llama-3.2-1B) | The teacher gets 31.0 on MMLU. LoLCATs gets 23.1, and Liger gets 22.4. On PIQA, LoLCATs gets 74.1 and Liger 75.0. On ARC-Easy, LoLCATs gets 63.7 and Liger 65.4. | The MMLU result of this project (23.3) is the same as two other methods at 1B in an independent paper. The PIQA (68.0) and ARC-Easy (54.8) results are much lower than both. |
| Lizard paper, Table 9 (arXiv version) | Teacher: PIQA 74.1, ARC-Easy 65.4, MMLU 31.0. Lizard: 74.8, 65.6 and 29.8. LoLCATs: MMLU 27.3. | The Lizard paper reports 27.3 for LoLCATs, but the Liger paper reports 23.1. The two papers measure differently. |
| Lizard paper, ACL 2026 version (long paper, pages 34935–34948) | A new version exists. Its abstract gives gains of "9.4–24.5 points" on MMLU 5-shot. The search found no public code. | Check Table 9 of the ACL version. Ask the authors for the 1B configs and code. |
| STILL (arXiv 2602.02180, NeurIPS 2026) | For the 1B teacher, it gives MMLU 31.9. It reports Liger at 22.4 and STILL at 29.8. STILL keeps softmax attention over selected tokens inside each layer. | Methods with some exact softmax attention reach the MMLU of the Lizard paper. Fully linear methods stay near chance. |
| Bick et al., "Gather-and-Aggregate" (ICML 2025) | MMLU needs a few retrieval heads. In a pruned Llama-3.1-8B, one disabled head drops MMLU from 66% to 25%. Hybrid models give this task to their attention layers. | This agrees with the "A" collapse ([document 7](07-results.md)): the answer letter needs a retrieval step. |
| "Stuck on A" (arXiv 2608.02689) | A linearized 0.6B model has a perplexity near the teacher, but chance accuracy on multiple choice, with "A" for 81% of the questions. A short training stage on the answer format added 12.48 points. bf16 updates of the optimizer were lost. | The same symptoms as this project ([paper LR](experiments/paper-lr.md)). The "A" collapse can be an injury of the answer interface, not only of the knowledge. |
| LoLCATs paper | On Llama 3 8B, 50% softmax layers close 94.2% of the MMLU gap (65.8 against 52.8, teacher 66.6). RedPajama data instead of Alpaca adds approximately 2 MMLU points. | A few softmax layers can recover MMLU. The data has a smaller effect. |
| KL-guided layer selection (ICLR 2026, code `fla-org/hybrid-distillation`) | It keeps the top-K layers as softmax. The score of a layer is the KL divergence to the teacher after a short distillation. | A method to select the softmax layers, if this project tests a hybrid |
| RADLADS (arXiv 2505.03005) | Step 1: alignment of the attention outputs, 100M tokens, sequence length 512, learning rate from 1e-3 to approximately 1e-5. Step 2: KL divergence to the teacher logits, constant learning rate 1e-5. In total 350–700M tokens. It reports LoLCATs with an MMLU below chance. | Stage 2 of Lizard uses cross-entropy on Alpaca, not the teacher logits. A KL stage is the main difference to newer recipes. |
| "What Matters in Linearizing Language Models?" (arXiv 2504.14366) | Plain linear attention recovers the least of seven architectures. Gated delta rules recover the most. The gaps stay up to 10B tokens of training. | The architecture limits the recovery more than the training budget. |
| Taylor-Calibrate (arXiv 2606.16429) | The teacher projections do not set the decay and the gates of the student. A start from teacher statistics needs 4.9–9.2× fewer tokens. | The gate of this project starts at 0.5 in every layer. A start from the attention distance of each teacher head is an option. |
| "Untangling Component Imbalance" (arXiv 2510.05901) | Hybrid conversions can ignore the linear branch and use only the window. | This project uses both branches ([document 7](07-results.md), section 5). No action is necessary. |
| "Sliding-window beats linear attention" (arXiv 2608.28444) | A window with sinks is as good as post-trained linear attention on many tasks, and 2–10× better on long context. | Supports `window_rope`: the window carries most of the score on short tasks. |

## 2. The plan

### Part A: complete the reproduction (first)

| # | Step | Reason | Cost | Decision |
|---|---|---|---|---|
| A1 | The teacher on PIQA, ARC-Easy, ARC-Challenge, HellaSwag and WinoGrande in this harness (X0) | The target in this harness is not measured. The Lizard paper, STILL and Meta give 31.0, 31.9 and 32.2 for MMLU. **Done in part (2026-10-10):** PIQA 74.4 and ARC-Easy 65.3. ARC-Challenge, HellaSwag and WinoGrande are still open. | Minutes on the A10 | Use these values as the target for every later result. |
| A2 | **Positive control:** the original LoLCATs attention on Llama-3.2-1B, with the same pipeline and harness (`make lolcats`) | The Liger paper gives a target for LoLCATs at 1B: PIQA 74.1, ARC-Easy 63.7, MMLU 23.1. This control separates the pipeline from the Lizard layer. | Approximately 4 hours and $19 on the H200 (estimate from [document 2](02-compute-and-cost.md)) | Near the target: the gap is specific to Lizard. Near 68 / 55: the cause is in the pipeline, the data or the harness. |
| A3 | MMLU with rotated answer options | It separates the knowledge from the letter preference ("Stuck on A", Bick et al.). | Minutes and a small script | The same answer content at every position: the model knows the answer. The same letter at every position: an interface injury. |
| A4 | The ACL version of the Lizard paper, and a message to the authors | The search found no code. The abstract of the paper changed between the versions. | No runs | Write down each answer, and each change in Table 9. |
| A5 | A second seed for the final model | One seed for each run ([document 16](16-optuna-plan.md)) | Approximately $13 for each stage 1 run | Report the mean and the spread. |

The model config [`distill_llama3_1_1b_lk_smd_wtk64_fd64_w01.yaml`](../configs/model/distill_llama3_1_1b_lk_smd_wtk64_fd64_w01.yaml) of `make lolcats` uses Llama-3.2-1B. Its window is 128 and its feature dimension is 128, against 64 in the LoLCATs defaults. For A2, keep the config, but write down this difference.

### Part B: improve the scores (after part A)

These steps change the method of the paper. Thus the thesis must report them as extensions, not as the reproduction.

| # | Step | Reason | Cost | Code |
|---|---|---|---|---|
| B1 | **Keep one or two layers as softmax:** `softmax_attentions: [15]`, then `[0, 15]` | LoLCATs, STILL and Bick et al. In this project, layer 15 gives 24–31% of the stage 1 loss, and layer 0 has the worst match of the attention weights. | Approximately $13 for each stage 1 run, and the evaluation | None. `convert_model.py` already keeps these layers and freezes them. |
| B2 | Select the softmax layers by KL divergence | KL-guided layer selection | One short run for each layer | A new script |
| B3 | Stage 2 with the KL divergence to the teacher logits | RADLADS, MOHAWK and Mamba in the Llama use the teacher logits. | Approximately 3–4 hours for each run | A new loss in the stage 2 trainer |
| B4 | A start value of the gate for each layer, from the attention distance of the teacher heads | Taylor-Calibrate. X3 and X5 of [gap analysis 2](12-gap-analysis-2.md) give the statistics. | Minutes for the statistics, then one stage 1 run | Small: `gate_bias_init` for each layer |
| B5 | `window_rope` | Liger uses RoPE in the window, and claim F of [document 15](15-attention-math-side-by-side.md) | Approximately $13 for each stage 1 run | None, the option exists ([document 13](13-lizard-attention-v2.md)) |
| B6 | A short training stage on the answer format, with multiple-choice data that is not MMLU | "Stuck on A" | Approximately 1,000 steps | A new data loader |
| B7 | More data, or pretraining data in place of Alpaca | RADLADS uses 350–700M tokens. LoLCATs saw approximately 2 MMLU points from RedPajama. | 10× or more of the current cost | A new data loader |

### Stop

- **Do not tune the stage 1 recipe for a lower loss.** The loss does not predict the accuracy ([document 14](14-liger-gla.md)). The literature points to the architecture and to the stage 2 objective as the limits.
- **Do not use MMLU alone to compare runs.** At 1B, two other methods also stay near chance on MMLU (Liger paper). Use PIQA, ARC-Easy and the rotation test of A3 too.

## 3. Sources

All sources come from a web search on 2026-10-08. The session could not open the full texts.

- Lizard: [arXiv 2507.09025](https://arxiv.org/abs/2507.09025), [ACL 2026](https://aclanthology.org/2026.acl-long.1613/), [Adobe Research](https://research.adobe.com/publication/lizard-an-efficient-linearization-framework-for-large-language-models)
- Liger: [arXiv 2503.01496](https://arxiv.org/pdf/2503.01496), code [OpenSparseLLMs/Linearization](https://github.com/opensparsellms/Linearization)
- STILL: [arXiv 2602.02180](https://arxiv.org/html/2602.02180), code [ZacharyMeng/STILL](https://github.com/ZacharyMeng/STILL)
- Gather-and-Aggregate: [ICML 2025](https://proceedings.mlr.press/v267/bick25a.html)
- Stuck on "A": [arXiv 2608.02689](https://arxiv.org/abs/2608.02689)
- LoLCATs: [arXiv 2410.10254](https://arxiv.org/pdf/2410.10254)
- KL-guided layer selection: [arXiv 2512.20569](https://arxiv.org/pdf/2512.20569), code [fla-org/hybrid-distillation](https://github.com/fla-org/hybrid-distillation)
- RADLADS: [arXiv 2505.03005](https://arxiv.org/pdf/2505.03005)
- What Matters in Linearizing Language Models: [arXiv 2504.14366](https://arxiv.org/pdf/2504.14366)
- Taylor-Calibrate: [arXiv 2606.16429](https://arxiv.org/pdf/2606.16429)
- Untangling Component Imbalance: [arXiv 2510.05901](https://arxiv.org/pdf/2510.05901)
- Sliding-window beats linear attention: [arXiv 2608.28444](https://arxiv.org/pdf/2608.28444)
- Mamba in the Llama: [arXiv 2408.15237](https://arxiv.org/pdf/2408.15237)
- Llamba: [arXiv 2502.14458](https://arxiv.org/pdf/2502.14458)
- When Perplexity Lies: [arXiv 2603.26556](https://arxiv.org/abs/2603.26556)
