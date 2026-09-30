# 7. Results

All evaluations used the pinned LM Evaluation Harness (`b281b09`) with batch size 1 and no
beginning-of-text token ([document 6](06-evaluation-setup.md)). "Lizard" is the model from
[Run 2](04-training-runs.md): Run 1's stage 1 checkpoint plus the finetune-only stage 2 checkpoint
(`...-s=0-se=0-re=0-se=0-re=0_ft.pt`). "Teacher" is the unmodified `meta-llama/Llama-3.2-1B`.
Standard errors (SE) are binomial, `sqrt(p (1 − p) / n)`, as reported by the harness.

## Summary

| Task | Lizard | Teacher (same harness) | Paper: Lizard 1B | Paper: LoLCATs 1B | Paper: teacher 1B |
|---|---|---|---|---|---|
| MMLU 5-shot, full | **23.3** | not yet run | 29.8 | 27.3 | 31.0 |
| MMLU 5-shot, 285-question subset | 24.9 | 33.7 | – | – | – |
| PIQA, 0-shot (acc) | **68.0 ± 1.1** | not yet run | 74.8 | 74.6 | 74.1 |
| ARC-Easy, 0-shot (acc) | **54.8 ± 1.0** | not yet run | 65.6 | 63.0 | 65.4 |

The paper numbers are from its Table 9. The paper used a newer harness version, so the exact gap
needs the teacher evaluated in this harness ([document 10](10-open-issues-and-next-steps.md)).

## 1. MMLU on a 285-question subset (`--limit 5`)

5 questions from each of the 57 subjects. The harness shuffles each subject with a fixed seed (42)
before applying the limit, so both models saw exactly the same questions.

| Model | Correct | Accuracy |
|---|---|---|
| Lizard | 71 / 285 | 0.249 |
| Teacher | 96 / 285 | 0.337 |

- Gap: 8.8 points (25 questions). Unpaired SE of the difference 0.038, z = 2.31, 95% interval
  1.3 to 16.2 points. A paired test would be tighter but needs per-question outputs, which the log
  doesn't contain.
- The teacher's 0.337 is close to the paper's 31.0 for Llama-3.2-1B, so the harness setup works.
- On its own, 0.249 on 285 questions (SE about 2.6 points) could not distinguish a broken model from a
  somewhat weaker one. That motivated the full run.

## 2. Full MMLU (5-shot, 14,042 questions)

**Lizard: 0.2326** (unweighted mean over 57 subjects).

- With about 14,000 questions the SE at p = 0.25 is 0.37 points, so 0.2326 is about 4.8 SE
  **below chance** (0.25).
- 39 subjects are below 0.25, 4 exactly at 0.25, and 14 above. None is clearly above chance: the
  highest are world religions 32.2, computer security 32.0, machine learning 31.2, medical genetics
  31.0, human aging 30.5 and business ethics 30.0, on 100–250 questions each (SE 3–5 points).
- The lowest are high school statistics 15.3, high school geography 17.2, management 17.5,
  astronomy 17.8, global facts 18.0 and professional medicine 18.4.

## 3. Answer-letter analysis (`letters.py`)

A score reliably below chance on 4-choice questions suggests the model ignores the question and
prefers one letter, so that its accuracy in each subject equals how often that letter is correct.
`letters.py` compares each subject's accuracy with each letter's frequency among the correct answers:

| If the model always answered | Macro-average score | Mean \|acc − freq\| over subjects | Correlation with per-subject accuracy |
|---|---|---|---|
| **"A"** | **0.231** | **0.004** | **+0.98** |
| "B" | 0.245 | 0.039 | +0.29 |
| "C" | 0.254 | 0.051 | −0.05 |
| "D" | 0.270 | 0.087 | −0.65 |

Actual macro average: 0.233.

**Finding: on MMLU the Lizard model answers "A" almost regardless of the question.** Its choice
doesn't depend on the question or the options; it falls back to a fixed preference for " A" after
"Answer:". For most questions the question and options fit in the last ~100 tokens, inside the
128-token window, so this is not only a long-context failure. Because the teacher itself is only
about 6–9 points above chance on MMLU, a moderately degraded model can already fall back to a letter
preference, so this result alone doesn't show how broad the damage is.

## 4. PIQA and ARC-Easy (0-shot)

These are scored on the likelihood of the full answer text, so a letter preference can't affect
them, and most prompts fit inside the 128-token window.

| Task | acc | acc_norm | Paper: Lizard 1B (acc) |
|---|---|---|---|
| PIQA (1,838 questions) | 67.95 ± 1.09 | 66.59 ± 1.10 | 74.8 |
| ARC-Easy (2,376 questions) | 54.80 ± 1.02 | 50.08 ± 1.03 | 65.6 |

The paper reports `acc` for both tasks.

- Both are far above chance (50% for PIQA, 25% for ARC-Easy): the model still works as a language model.
- Both are clearly below the paper: about 6.3 SE below on PIQA and 10.6 SE on ARC-Easy. They are also
  below the paper's LoLCATs 1B numbers (74.6, 63.0).
- **Finding: the degradation is broad, not limited to MMLU or to long context.** Lizard is RoPE-free,
  so even over short spans it depends on the trained feature maps, gate and α to recover token order
  and the teacher's attention patterns.

## 5. Removing a branch at inference time (`ablate.py`, PIQA)

| PIQA | acc | acc_norm | Change in acc vs full |
|---|---|---|---|
| Full Lizard model | 67.95 ± 1.09 | 66.59 | – |
| Window branch off: (1 + α) · GLA | 57.18 ± 1.15 | 54.95 | −10.8 (z ≈ 6.8) |
| Gated branch off: (1 + α) · AWA | 52.34 ± 1.17 | 51.80 | −15.6 (z ≈ 9.8) |
| Chance | 50.0 | 50.0 | |

**Findings:**

- **Both branches are in use.** Removing either one costs 11–16 points.
- **Without the gated branch the model is essentially at chance** (52.3, about 2 SE above 50). This
  fits the paper's design: without RoPE, the window branch can't tell token order within its 128
  tokens, and the gate's decay is the model's only source of position information.
- **The gated branch alone keeps some ability** (57.2), but lacks the window branch's sharp local
  attention.
- This rules out an earlier hypothesis that the gated branch had not learned and the model ran on the
  window branch alone.
- This measures how much the trained model relies on each branch, not whether the architecture needs
  it (the paper's Table 6 retrains without the branch; there, 8B MMLU drops from 61.2 to 39.7 without
  the window branch and to 42.2 without the gate).

## Interpretation so far

- The attention code, the model wiring and the evaluation harness are verified
  ([document 8](08-verification.md)); the teacher scores as expected in the same harness.
- Both architectural components are active, but together they reach only 68.0 on PIQA against the
  paper's 74.8, and MMLU has collapsed to a constant answer.
- Of the remaining explanations, the training recipe differs most from the paper
  ([document 9](09-paper-comparison.md)). The controlled test is described in
  [document 10](10-open-issues-and-next-steps.md).

## Appendix: full MMLU per subject (Lizard, 5-shot, accuracy in %)

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
