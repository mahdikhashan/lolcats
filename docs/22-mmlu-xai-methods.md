# 22. XAI methods for MMLU: review of a ChatGPT answer

**Status:** A review of an answer from ChatGPT (2026-10-10) about papers on gated linear attention (GLA) and MMLU. The review compares the answer with these notes and gives an XAI method and a script for each proposed analysis. One check ran on existing evaluation files (section 3, A6). No new runs and no new code. The session could not open arXiv. Thus paper details come from search snippets.

## 1. The answer in short

- **Three kinds of papers:** reports of MMLU for GLA models, explanations of the MMLU behavior of GLA, and studies that change only the attention. The answer says that the third kind is the hardest to find.
- **Two papers.** The GLA paper (Yang et al., ICML 2024) is the baseline for GLA, but not an analysis of MMLU. "Gated Attention for Large Language Models" (NeurIPS 2025) studies a gate on softmax attention, not recurrent GLA.
- **Six analyses for the Lizard model:** MMLU for each subject, gate dynamics, memory retention, distillation effects, an architecture ablation, and calibration.
- **A caution.** An MMLU of 22–23% alone does not show that GLA is the cause. Scale, pretraining, fine-tuning, the evaluation and the interaction of GLA with the window can all contribute.

## 2. Claims of the answer against these notes

| Claim | Check | Verdict |
|---|---|---|
| The GLA paper is the baseline for GLA, and it does not analyze MMLU | The search shows the paper: 340M and 1.3B models on 15B and 100B tokens, against a Llama-style Transformer, RetNet and Mamba. The search found no MMLU table. | Correct as far as checked. The gated branch of Lizard is not the GLA of this paper. It has Hedgehog maps, one scalar gate for each layer and normalized rows ([document 15](15-attention-math-side-by-side.md)). Thus the results of the GLA paper do not transfer directly. |
| "Gated Attention for Large Language Models" is about a gate on softmax attention, not GLA | The search shows a sigmoid gate for each head after the softmax attention. The gate also reduces attention sinks. | Correct. Its only link to this project is the attention sink. The window of Lizard has sink tokens. Low priority. |
| No paper explains why GLA fails on MMLU | Bick et al. (ICML 2025) explain why recurrent models fail on the letter step of MMLU: a few Gather-and-Aggregate (G&A) heads need sharp attention ([document 21](21-mmlu-options.md)). | Partly wrong. The paper is not about GLA alone, but it explains the failure for recurrent attention in general. The answer misses it. |
| 22–23% alone does not show that GLA is the cause | The LoLCATs control has no gated branch, and it gets 26.0 on the subset after stage 1 ([control](experiments/lolcats-control.md)). At 1B, the Liger paper gives 23.1 for LoLCATs and 22.4 for Liger. | Correct. MMLU near chance at 1B is common to linear attention, not specific to GLA. Also, `window_rope` reaches 98% of the teacher on PIQA and ARC-Easy. Thus the failure is specific to MMLU. |
| The most valuable experiment is a controlled replacement study with MMLU for each subject and an error analysis for each question | This project trained variants with the same pipeline, recipe and data: Lizard Run 1, `window_rope`, `window_rope` with `hybrid`, and the LoLCATs control. | Partly done. Inference-time ablations and the error analysis for each question are missing (A5 and A4 below). |

## 3. XAI methods and scripts, one for each analysis

The lm-eval output of each MMLU run is in `*_write_out_info.json`. For each question, it has the prompts, the log-likelihood of each letter (`logit_0` to `logit_3`) and the right answer. Thus A1, A4 (letters) and A6 need no new forward passes.

| # | Analysis of the answer | XAI method | Question | Data | Script | Cost |
|---|---|---|---|---|---|---|
| A1 | MMLU for each subject | Accuracy for each subject and for each group (STEM, humanities, social sciences, other), for two models. Paired differences. Accuracy against the prompt length in tokens. | Do long prompts or some subject groups fail more? | The write-out files of two runs. All questions (M0 of [document 21](21-mmlu-options.md)) give an SE of 3–5 points for each subject. The subset gives approximately 20 points. | New: `scripts/mmlu_items.py subjects` | CPU, seconds |
| A2 | Gate dynamics | γ for each layer and token, grouped by the segment of the prompt: the 5 examples, the question, the options and "Answer:". The gated weight that reaches the last token from each segment. | Does the gated branch keep the question, or does it mix it with the examples? | Forward passes on 50–100 MMLU prompts | Extension of `scripts/attention_weights.py` (an MMLU input mode) | A10, minutes |
| A3 | Memory retention | (a) The attention row of the last token, for each head, Lizard against the teacher. It gives the mass on the options, on the right option and on the examples. (b) Logit lens: the letter probabilities after each layer. (c) A linear probe for each layer that predicts the right option from the hidden state of the last token. | In which layer does the student lose the answer? Is the answer in the hidden state, but not in the output? | Forward passes on 500–1,000 MMLU questions, teacher and student | (a) the same extension of `attention_weights.py`. (b), (c) new: `scripts/mmlu_residual.py` | A10, approximately 1 hour (estimate) |
| A4 | Distillation effects | The KL divergence between teacher and student over the four letters, for each question. A 2 × 2 table: teacher right or wrong, student right or wrong, with the McNemar test. The KL over the full vocabulary at the last token needs a forward pass. | Does the student fail on the questions that the teacher answers right, or at random? | The write-out files of both models (letters). A forward pass for the full vocabulary. | New: `scripts/mmlu_items.py divergence` | CPU, seconds (letters) |
| A5 | Architecture ablation | (a) The gated branch or the window switched off or scaled at inference (M4 of [document 21](21-mmlu-options.md)). (b) **Layer restoration:** one layer at a time goes back to softmax attention at inference, then the MMLU accuracy and the letter KL. (c) The trained variants of this project in one table. | Which branch blurs the letter step? Which layer gives the most MMLU if it stays softmax (T1 of [document 21](21-mmlu-options.md))? | MMLU subset or a sample of 1,000 questions | (a) `scripts/ablate.py` with v2 support. (b) new: `scripts/mmlu_restore.py`. (c) no script. | A10, 16 runs of minutes |
| A6 | Calibration | The probability of the right letter and the margin to the best wrong letter. The expected calibration error, the prior of each letter, the entropy, and the ties at the maximum. | Is the model unsure, or sure and wrong? Does the evaluation add a bias? | The write-out files | New: `scripts/mmlu_items.py calibration` | CPU, seconds |

**Notes on A5 (b).** In the stage 1 checkpoint, q, k, v and o are the weights of the teacher. Thus a restored layer is exactly the teacher layer. In the stage 2 checkpoint, LoRA changed q, k, v and o. A restored layer then uses softmax attention with the changed weights, so it is only an approximation. The model already has a teacher mode for each layer, which test 8 of [document 13](13-lizard-attention-v2.md) uses.

#### A first result of A6: bf16 ties

In the write-out files of `window_rope` after stage 2, every difference between two letter log-likelihoods is a multiple of 0.125. This agrees with logits in bf16 at a magnitude of approximately 16–32. On the MMLU subset, 63 of 285 questions (22.1%) have a tie at the maximum. In a tie, the first letter wins: "A" in 44 of the 63 ties. With a random choice in each tie, the accuracy is 24.7, against the stored 25.6.

| Run (MMLU subset) | Questions with a tie at the maximum | Tie includes the right letter | Stored accuracy | Accuracy with random choice in ties |
|---|---|---|---|---|
| `window_rope`, stage 1 | 4 (1.4%) | 0 | 24.6 | 24.6 |
| `window_rope` with `hybrid`, stage 1 | 16 (5.6%) | 11 | 25.3 | 25.0 |
| `window_rope`, after stage 2 | 63 (22.1%) | 31 | 25.6 | 24.7 |

- The ties change the letter shares more than the accuracy. A part of the 41.4% "A" after stage 2 comes from the ties.
- A model with small letter margins gets more ties. After stage 2, the confidence is 0.348.
- PIQA and ARC-Easy sum the log-likelihoods of many tokens. Thus ties are rare there.
- **A change for the evaluation:** compute the output layer (`lm_head`) in float32 before the log-softmax. Then evaluate the MMLU subset again. The teacher write-out files are not in this repository, so the ties of the teacher are not measured yet.

## 4. Two scripts cover most of the analyses

| Script | Commands | Input | Output | Order |
|---|---|---|---|---|
| `scripts/mmlu_items.py` | `subjects` (A1), `divergence` (A4, letters), `calibration` (A6, with the ties) | The write-out directories of one or two runs (`<run>/<model>/mmlu_subset` or `mmlu`) | A JSON file, a Markdown table and plots. They show the accuracy for each subject and group, and the paired 2 × 2 table. They also show the reliability curve and the letter shares with and without ties. | 1. CPU only, from existing files |
| `scripts/mmlu_restore.py` | `layers` (A5 b), `branches` (A5 a, for v2) | A checkpoint and an MMLU sample | Accuracy and letter KL for each restored layer or ablated branch | 2. Predicts T1 without training |
| `scripts/mmlu_residual.py` | `lens` (A3 b), `probe` (A3 c) | Teacher and student, 500–1,000 MMLU questions | Letter probabilities and probe accuracy for each layer | 3. |
| `scripts/attention_weights.py` (extension) | `--inputs mmlu` for A2 and A3 (a) | Teacher and student, 50–100 MMLU prompts | Attention mass and gate values for each prompt segment | 4. |

**Data before the scripts:** the write-out files of the teacher on the MMLU subset (`MODELS=teacher TASKS=mmlu_subset scripts/compare_stages.sh`, minutes). Later, all questions for both models (M0).

## 5. What the answer adds, and what it misses

- **Adds:** a clear list of analyses for each question. A4 (the paired error analysis) and A6 (calibration) were not in [document 21](21-mmlu-options.md).
- **Misses:** the Gather-and-Aggregate result of Bick et al., the 1B values of the Liger paper, and the controls of this project (section 2).
- **The bf16 ties:** only A6 shows them, because it looks at the letter log-likelihoods of each question.

## 6. Sources

- Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training", ICML 2024: [arXiv 2312.06635](https://arxiv.org/abs/2312.06635), [PMLR](https://proceedings.mlr.press/v235/yang24ab.html)
- Qiu et al., "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free", NeurIPS 2025: [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html), [arXiv 2505.06708](https://huggingface.co/papers/2505.06708)
- Bick, Xing and Gu, Gather-and-Aggregate, ICML 2025: [PMLR](https://proceedings.mlr.press/v267/bick25a.html)
- The other sources of [document 21](21-mmlu-options.md), section 6
