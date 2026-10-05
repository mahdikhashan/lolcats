# 17. Reproduction studies: review of a ChatGPT answer

**Status:** A review of an answer from ChatGPT (2026-10-05) about earlier reproduction studies of language models. This document lists the claims of the answer and compares them with these notes. It also adapts the proposed research questions to this project. No new runs, no code. This session could not open arXiv or the project pages, so this document does not check the cited papers.

## 1. The answer in short

- **Two kinds of studies.** Some studies train a model again from a published description (OpenLLaMA, TinyLlama, OPT, OLMo). Other studies repeat published experiments under control (Pythia, reproduction reports). The answer says that both kinds are useful for this thesis.
- **Nine works**, ranked by relevance. The top three are OpenLLaMA, Pythia and the reproduction study of Behavior Transformers.
- **The Pythia idea.** Save checkpoints during training. Then measure the scores, the branches, the representations and the attention at each checkpoint.
- **A tree of causes for the MMLU gap.** Split the gap into an evaluation gap and a model gap before blaming the architecture.
- **Four research questions:** reproduction, discrepancy, architecture and interpretability.

## 2. Sources

| Work | Reference in the answer | Check in this document |
|---|---|---|
| OpenLLaMA | GitHub README of `openlm-research/open_llama` | Not checked |
| Pythia | arXiv:2304.01373 | Not checked. The identifier matches the title (from memory). |
| OLMo | None | Not checked |
| TinyLlama | None | Not checked |
| OPT | arXiv:2205.01068 | Not checked. The identifier matches the title (from memory). |
| Reproducibility Issues for BERT-based Evaluation Metrics | ACL Anthology 2022.emnlp-main.192 | Not checked |
| Pre-trained Transformers: An Empirical Comparison | None | Not checked |
| [Re] Reproducibility Study of Behavior Transformers | None | Not checked |
| Pythia with several seeds | None | Not checked. The answer gives no source for the seeds 1–9. |

Before the thesis cites a work, the thesis author must read it and check the claim. The star ranking of the answer is an opinion, not a measurement.

## 3. Claims of the answer against these notes

| # | Claim of the answer | What these notes show | Verdict |
|---|---|---|---|
| 1 | The MMLU result is "22% → 23%", with a gap of 9–10 points. | All MMLU questions: 23.3 for the Lizard model, 29.8 for Lizard 1B in the paper, thus 6.5 points ([document 7](07-results.md)). MMLU subset: 24.9 against 33.7 for the teacher, thus 8.8 points. Stage 1 runs: 22.5–26.7 on the MMLU subset. | **Not exact.** Use the numbers of document 7. |
| 2 | GLA plus a sliding window is "the proposed modification" (research question 3). | The gated branch plus the window with sinks **is** Lizard, the architecture of the paper ([document 15](15-attention-math-side-by-side.md)). The changes of this project are the v2 options C1–C6 ([document 13](13-lizard-attention-v2.md)) and the proposals of [document 14](14-liger-gla.md). | **Wrong premise.** Section 4 gives a new research question 3. |
| 3 | Check the evaluation before blaming the model. | Done in part. The attention code passes the checks of [document 8](08-verification.md). In this harness, the teacher gets 33.7 ± 2.8 on the MMLU subset. The paper gives 31.0 for all MMLU questions. Thus the harness probably does not make the scores low. **Open:** the teacher on PIQA and ARC-Easy in this harness (X0 in [gap analysis 2](12-gap-analysis-2.md), minutes on the A10). | **Correct.** One check of a few minutes is still open. |
| 4 | Run several seeds and report mean ± std. | Each run has one seed ([document 13](13-lizard-attention-v2.md)). On MMLU, the stage 1 models stay at the level of guessing. A model that always selects "A" gets 24.2. Thus more seeds on MMLU only measure noise near chance. Seeds matter for PIQA and ARC-Easy. [Document 16](16-optuna-plan.md) plans a second seed for the best config. | **Correct for PIQA and ARC-Easy.** |
| 5 | Measure the weight norms of each branch, ‖W_GLA‖ and ‖W_SWA‖. | The window branch has no weight matrix of its own. It uses q, k and v of the teacher, plus α and four sink logits. The notes already record the signals of the branches. These are α of each head, gate statistics, sink logits and the window share of each head ([document 13](13-lizard-attention-v2.md), [gap analysis 2](12-gap-analysis-2.md)). | **Does not fit Lizard.** Use the window share. |
| 6 | Measure the KL divergence between the logits of the teacher and the student. | Stage 1 does not use logits. Its loss is the MSE of the attention outputs, and `xent_factor` is 0. The notes have no KL measurement. | **New and cheap.** A diagnostic only, not a training change. |
| 7 | Compare hidden states of teacher and student (CKA, cosine, SVCCA). | The layer-wise MSE compares the attention outputs, with the input of the teacher for each layer ([XAI: layer-wise MSE](experiments/xai-layer-wise-mse.md)). No document compares the hidden states of the full student model, where errors add up from layer to layer. Document 14 names this error growth as a blind spot of the stage 1 loss. | **New and useful.** It measures the error growth. |
| 8 | Measure attention entropy, sink mass, local and global attention, and head types. | Done in part. [XAI: sample attention weights](experiments/xai-sample-attention-weight.md) gives the TV distance to the teacher, the window share and the local heads. No document gives the entropy. | **Mostly done.** |
| 9 | Save checkpoints during training, as Pythia does, and compare them with a trained baseline. | The trainer keeps only the best stage 1 checkpoint (one file, `load_best_model_at_end`). The R1b stage 1 checkpoint has 8,574,050 bytes ([document 13](13-lizard-attention-v2.md)). Thus one checkpoint at each evaluation (every 100 of 1,178 steps) needs approximately 100 MB. The teacher does not train, so the baseline is the frozen teacher. | **New and cheap.** It shows when a change starts, for example the fall of the gate in layer 15. |
| 10 | Test the branches first on a small model, as TinyLlama did. | TinyLlama trained from the start. This thesis linearizes a pretrained model. Llama-3.2-1B is already the smallest Llama 3 model. This project has a small controlled test: the single-layer bench ([document 11](11-gap-analysis.md), section 13.5). | **Low value.** Use the single-layer bench. |
| 11 | Make a reproduction matrix, as OpenLLaMA did: each part of the recipe, same or different. | Three documents have the parts. [Document 9](09-paper-comparison.md) has a recipe table, [math against code](math-code-discrepancy.md) has 17 differences, and [document 11](11-gap-analysis.md) ranks the causes. No single table exists. | **Useful.** One table, no runs. |
| 12 | Make each experiment auditable: config, commit, data version, checkpoint, evaluation script and W&B run. | Done in part. The R1b folder has a manifest with SHA-256 values, and `compare_stages.sh` writes `env.txt`. But some notes say: "These notes do not record the job ID" ([document 13](13-lizard-attention-v2.md), [stage 2 on config 1](experiments/stage2-config1.md)). | **Useful.** The same run record for each job. |
| 13 | Show that the training dynamics match the original Lizard implementation. | These notes have no code and no training curves from the authors of Lizard. | **Not possible** with the sources of these notes. |

## 4. Research questions for this project

The four questions of the answer give a clear frame for the thesis. Question 3 needs a change (claim 2). The table maps each question to the documents of these notes.

| # | Question | Status | Documents |
|---|---|---|---|
| RQ1 | Can the Lizard results of the paper for Llama-3.2-1B be reproduced? | Not reproduced so far. MMLU 23.3 against 29.8, PIQA 68.0 against 74.8, ARC-Easy 54.8 against 65.6. | [4](04-training-runs.md), [7](07-results.md), [9](09-paper-comparison.md) |
| RQ2 | Which differences in the implementation, the recipe and the evaluation explain the gap? | Open, partly explained. Candidates: the recipe, the open points D1–D5 of the paper, the limit for each head, the window without RoPE. | [11](11-gap-analysis.md), [12](12-gap-analysis-2.md), [13](13-lizard-attention-v2.md), [math against code](math-code-discrepancy.md) |
| RQ3 | Do changes to the Lizard attention close the gap? The changes are the v2 options and ideas from Liger and LoLCATs. | C1 and R1b done, with no accuracy gain. [Document 14](14-liger-gla.md) plans `window_rope`, `convex` and `gate_per_head`. | [13](13-lizard-attention-v2.md), [14](14-liger-gla.md), [15](15-attention-math-side-by-side.md), [16](16-optuna-plan.md) |
| RQ4 | Do the layer errors, attention weights, gates and branch shares explain the scores? | Partly done. The trajectory of checkpoints and the hidden-state distance are not done. | [12](12-gap-analysis-2.md), [13](13-lizard-attention-v2.md), [XAI](experiments/xai.md) |

The answer proposes a thesis structure from the reproduction study of Behavior Transformers: claim, reproduction, discrepancy, investigation, conclusion. RQ1–RQ4 follow that order.

## 5. Proposed additions, least expensive first

| # | Addition | Cost | Answers |
|---|---|---|---|
| 1 | The teacher on PIQA and ARC-Easy in this harness (X0) | Minutes on the A10 | RQ1, RQ2 |
| 2 | One reproduction matrix: each part of the recipe, paper against this project, with "same", "different" or "unknown" | No runs | RQ2 |
| 3 | A run record for each job: commit, image digest, configs, job ID, GPU flavor, seed, checkpoint SHA-256, harness commit | No runs | All |
| 4 | Save the stage 1 checkpoint at each evaluation, then run `layer_mse.py` on each | Small code change. Approximately 100 MB for each run. | RQ4 |
| 5 | Hidden-state distance between teacher and student after each layer, in the full model | A new script, minutes for each checkpoint | RQ2, RQ4 |
| 6 | KL divergence of the logits on the validation set | A new script, minutes for each checkpoint | RQ4 |
| 7 | A second seed for the best config | ~$13 for each stage 1 run ([document 16](16-optuna-plan.md)) | RQ3 |

Not adopted: a small model trained from the start (claim 10) and weight norms for each branch (claim 5).

## 6. A second answer

A second answer from ChatGPT (2026-10-05) lists more reproduction studies. The folder `reproduction-studies` keeps [the second answer](reproduction-studies/second-answer.md) without changes. This document does not check its sources.

- **New works:** MultiBERTs, the BERT4Rec replicability study, Mieskes (2022), A Primer in BERTology, Belz et al. (2021), ReproNLP 2024, Vaugrante et al. and ReproEvalCard.
- **BERT4Rec applies to stage 1.** According to the answer, the default config reproduced the paper only after much longer training. In all float32 runs of this project, the stage 1 validation loss still decreased at the end. H3 of [gap analysis 2](12-gap-analysis-2.md) (4 epochs) tests this.
- **MultiBERTs agrees with claims 4 and 9.** One checkpoint describes one run, not the recipe. Thus more seeds and checkpoints during training are necessary.
- **Its reproduction package agrees with addition 3 of section 5.** The run record needs the commit, configs, data order, seeds and harness version.
- **It repeats two errors of the first answer.** The MMLU values "23% instead of 32%" are not in these notes. The teacher gets 33.7 on the MMLU subset in this harness, and 31.0 in the paper (claim 1). It also calls GLA plus the window "the proposed architecture" (claim 2).

## 7. A third answer

A third answer from ChatGPT (2026-10-05) lists reproduction studies of attention layers and training recipes. The folder `reproduction-studies` keeps [the third answer](reproduction-studies/third-answer.md) without changes. This document does not check its sources.

- **New works:** a reproduction study of partial residual ablations (2026), RoBERTa, independent code of "Attention Is All You Need" and AttentionSmithy. The answer itself says that the GitHub projects are not peer reviewed.
- **The three levels fit these notes.** Level 1 (math against code) is [math against code](math-code-discrepancy.md) and the float64 tests against the jku-thesis reference. Level 2 (the layer inside the model) is [document 8](08-verification.md). Level 3 (training) is [document 4](04-training-runs.md) and the experiments.
- **RoBERTa agrees with the BERT4Rec point of section 6.** A recipe can train too little. The recipe experiments of these notes already form a part of the proposed matrix, for example the 2 × 2 grid of [document 11](11-gap-analysis.md).
- **The residual study gives a pattern: discrepancy, cause in the measurement, correction, new run.** These notes have such a case. The stage 1 loss is a measurement that does not predict the accuracy ([document 14](14-liger-gla.md)).
- **Some items do not apply.** Stage 1 uses the MSE, so it has no distillation temperature. Lizard replaces only the attention, and the residual paths and the norms of Llama stay the same.
- **It repeats the two errors of the first answer.** The MMLU values "22–23%" and "~32%" do not match these notes (claim 1). Its diagrams put "GLA + SWA" after the reproduction, as a change of Lizard (claim 2).

## 8. Limits

- This document does not check the cited papers. Section 2 lists each source and its status.
- The comparison in section 3 uses only these notes. A claim of the answer can be correct for the cited work and still not apply to this project.
