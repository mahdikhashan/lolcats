# 7. Results

All evaluations used the pinned harness (`b281b09`) with batch size 1 and no beginning-of-text token ([document 6](06-evaluation-setup.md)). Two models appear in this document:

- The **Lizard model** is the stage 1 checkpoint of Run 1 plus the stage 2 checkpoint of Run 2 (`...-s=0-se=0-re=0-se=0-re=0_ft.pt`). [Document 4](04-training-runs.md) describes the runs.
- The **teacher** is the unmodified `meta-llama/Llama-3.2-1B`.

The standard errors (SE) are binomial, `sqrt(p (1 − p) / n)`, as the harness reports them.

## Summary

| Task | Lizard model | Teacher (same harness) | Paper: Lizard 1B | Paper: LoLCATs 1B | Paper: teacher 1B |
|---|---|---|---|---|---|
| MMLU 5-shot, all questions | **23.3** | not run yet | 29.8 | 27.3 | 31.0 |
| MMLU 5-shot, MMLU subset (285 questions) | 24.9 | 33.7 | – | – | – |
| PIQA, 0-shot (accuracy) | **68.0 ± 1.1** | not run yet | 74.8 | 74.6 | 74.1 |
| ARC-Easy, 0-shot (accuracy) | **54.8 ± 1.0** | not run yet | 65.6 | 63.0 | 65.4 |

The paper values come from its Table 9. The paper used a newer harness version. Thus the exact gap needs an evaluation of the teacher in this harness ([document 10](10-open-issues-and-next-steps.md)).

**Update (2026-10-10).** A later model with RoPE in the window branch gives PIQA 73.1 and ARC-Easy 63.8 after stage 2. Section 6 compares it with the Lizard model of this document.

## 1. MMLU subset (`--limit 5`)

The MMLU subset has 5 questions from each of the 57 subjects. The harness shuffles each subject with a fixed seed (42) before it applies the limit. Thus both models received exactly the same questions.

| Model | Right answers | Accuracy |
|---|---|---|
| Lizard model | 71 / 285 | 0.249 |
| Teacher | 96 / 285 | 0.337 |

- The gap is 8.8 points (25 questions). The unpaired SE of the difference is 0.038, z = 2.31, and the 95% interval is 1.3 to 16.2 points. A paired test would give a narrower interval. But it needs the outputs for each question, and the log does not contain them.
- The teacher result of 0.337 is near the paper value of 31.0 for Llama-3.2-1B. This shows that the harness configuration works.
- The Lizard result of 0.249 on 285 questions has an SE of approximately 2.6 points. This result alone could not separate a broken model from a model that is only somewhat weaker. For this reason, the next step was an evaluation on all questions.

## 2. MMLU, all questions (5-shot, 14,042 questions)

**Lizard model: 0.2326** (unweighted mean over the 57 subjects).

- With approximately 14,000 questions, the SE at p = 0.25 is 0.37 points. Thus 0.2326 is approximately 4.8 SE **below chance** (0.25).
- 39 subjects are below 0.25, 4 subjects are exactly at 0.25, and 14 subjects are above 0.25. No subject is clearly above chance.
  - The highest subjects are world religions 32.2, computer security 32.0, machine learning 31.2, medical genetics 31.0, human aging 30.5 and business ethics 30.0.
  - Each of these subjects has 100–250 questions, so the SE is 3–5 points.
- The lowest subjects are high school statistics 15.3, high school geography 17.2, management 17.5, astronomy 17.8, global facts 18.0 and professional medicine 18.4.

## 3. Answer-letter analysis (`letters.py`)

On questions with 4 choices, a score that is reliably below chance suggests this behavior: the model ignores the question and prefers one letter. In that case, its accuracy in each subject is equal to the frequency of that letter as the right answer. `letters.py` compares the accuracy of each subject with the frequency of each letter among the right answers.

| If the model always gave | Macro-average score | Mean \|acc − freq\| over subjects | Correlation with the accuracy per subject |
|---|---|---|---|
| **"A"** | **0.231** | **0.004** | **+0.98** |
| "B" | 0.245 | 0.039 | +0.29 |
| "C" | 0.254 | 0.051 | −0.05 |
| "D" | 0.270 | 0.087 | −0.65 |

The real macro average is 0.233.

**Finding: on MMLU, the Lizard model selects "A" for almost every question.**

- Its choice does not depend on the question or the options. After "Answer:", it uses a fixed preference for " A".
- For most questions, the question and the options are in the last ~100 tokens, inside the 128-token window. Thus this is not only a failure on long contexts.
- The teacher itself is only approximately 6–9 points above chance on MMLU. Thus a model with moderate damage can already show a letter preference. This result alone does not show how wide the damage is.

## 4. PIQA and ARC-Easy (0-shot)

For these tasks, the score uses the likelihood of the full answer text. Thus a letter preference cannot affect them. Most prompts are shorter than the 128-token window.

| Task | Accuracy | Normalized accuracy | Paper: Lizard 1B (accuracy) |
|---|---|---|---|
| PIQA (1,838 questions) | 67.95 ± 1.09 | 66.59 ± 1.10 | 74.8 |
| ARC-Easy (2,376 questions) | 54.80 ± 1.02 | 50.08 ± 1.03 | 65.6 |

The paper reports the accuracy (`acc`) for both tasks.

- Both results are far above chance (50% for PIQA, 25% for ARC-Easy). Thus the model still works as a language model.
- Both results are clearly below the paper: approximately 6.3 SE below on PIQA and 10.6 SE below on ARC-Easy. They are also below the LoLCATs 1B values of the paper (74.6, 63.0).
- **Finding: the damage is wide**. It is not limited to MMLU or to long contexts. Lizard attention uses no RoPE. Thus, even over short spans, the model needs the trained feature maps, the gate and α to get the token order. It also needs them to get the attention patterns of the teacher.

## 5. Disabled branches during inference (`ablate.py`, PIQA)

| PIQA | Accuracy | Normalized accuracy | Accuracy difference from the full model |
|---|---|---|---|
| Full Lizard model | 67.95 ± 1.09 | 66.59 | – |
| Window branch disabled: (1 + α) · GLA | 57.18 ± 1.15 | 54.95 | −10.8 (z ≈ 6.8) |
| Gated branch disabled: (1 + α) · AWA | 52.34 ± 1.17 | 51.80 | −15.6 (z ≈ 9.8) |
| Chance | 50.0 | 50.0 | |

**Findings:**

- **The model uses both branches.** A disabled branch causes a decrease of 11–16 points.
- **Without the gated branch, the model is almost at chance** (52.3, approximately 2 SE above 50). This agrees with the design of the paper. Without RoPE, the window branch cannot find the token order inside its 128 tokens. The decay of the gate is the only source of position information in the model.
- **The gated branch alone keeps some ability** (57.2). But it does not have the sharp local attention of the window branch.
- This result excludes an earlier hypothesis: that the gated branch did not learn, and that the model used only the window branch.
- This method measures how much the trained model depends on each branch. It does not measure if the architecture needs the branch.
  - The paper (Table 6) trains a new model without the branch.
  - In that ablation, MMLU of the 8B model decreases from 61.2 to 39.7 without the window branch, and to 42.2 without the gate.

## 6. Update: `window_rope` after stage 2 (2026-10-10)

The Lizard attention v2 with RoPE in the window branch (`window_rope`), after stage 1 and stage 2. The harness, the tasks and the questions are the same as above. [RoPE in the window branch](experiments/window-rope.md), Run 2, gives all details.

| Task | Lizard model (this document) | `window_rope` after stage 2 | Difference | Paper: Lizard 1B |
|---|---|---|---|---|
| MMLU subset (285 questions) | 24.9 | 25.6 | +0.7 points | – |
| PIQA (accuracy) | 67.95 ± 1.09 | **73.1 ± 1.0** | +5.1 points, z ≈ 3.4 | 74.8 |
| ARC-Easy (accuracy) | 54.8 ± 1.0 | **63.8 ± 1.0** | +9.0 points, z ≈ 6.3 | 65.6 |
| Stage 2 validation loss | 2.252 | 1.235 | – | – |

- The "A" collapse on MMLU is gone ("A" for 41.4% of the questions). But the MMLU subset stays at chance, and the teacher has 33.7.
- The paper values come from a newer harness. The teacher in this harness on PIQA and ARC-Easy is still not measured.

## Interpretation so far

- The checks show no error in the attention code, the connection into the model or the harness ([document 8](08-verification.md)). In the same harness, the teacher gets the expected score.
- Both components of the architecture are active. But together they get only 68.0 on PIQA, against 74.8 in the paper. On MMLU, the model gives a constant answer.
- Of the remaining explanations, the recipe has the largest difference from the paper ([document 9](09-paper-comparison.md)). [Document 10](10-open-issues-and-next-steps.md) describes the controlled test.

## Appendix: MMLU per subject, all questions (Lizard model, 5-shot, accuracy in %)

| Subject | Accuracy |
|---|---|
| abstract algebra | 22.0 |
| anatomy | 21.5 |
| astronomy | 17.8 |
| business ethics | 30.0 |
| clinical knowledge | 21.1 |
| college biology | 25.0 |
| college chemistry | 20.0 |
| college computer science | 25.0 |
| college mathematics | 22.0 |
| college medicine | 20.8 |
| college physics | 21.6 |
| computer security | 32.0 |
| conceptual physics | 26.4 |
| econometrics | 23.7 |
| electrical engineering | 24.1 |
| elementary mathematics | 20.9 |
| formal logic | 29.4 |
| global facts | 18.0 |
| high school biology | 18.4 |
| high school chemistry | 18.7 |
| high school computer science | 24.0 |
| high school european history | 21.8 |
| high school geography | 17.2 |
| high school government and politics | 19.7 |
| high school macroeconomics | 20.3 |
| high school mathematics | 21.1 |
| high school microeconomics | 21.0 |
| high school physics | 19.2 |
| high school psychology | 19.3 |
| high school statistics | 15.3 |
| high school us history | 25.0 |
| high school world history | 27.0 |
| human aging | 30.5 |
| human sexuality | 26.0 |
| international law | 24.0 |
| jurisprudence | 25.9 |
| logical fallacies | 23.9 |
| machine learning | 31.2 |
| management | 17.5 |
| marketing | 28.6 |
| medical genetics | 31.0 |
| miscellaneous | 23.5 |
| moral disputes | 24.9 |
| moral scenarios | 23.8 |
| nutrition | 22.2 |
| philosophy | 18.6 |
| prehistory | 21.6 |
| professional accounting | 23.4 |
| professional law | 24.6 |
| professional medicine | 18.4 |
| professional psychology | 25.0 |
| public relations | 21.8 |
| security studies | 18.8 |
| sociology | 24.4 |
| us foreign policy | 27.0 |
| virology | 27.7 |
| world religions | 32.2 |
| **Macro average** | **23.3** |
