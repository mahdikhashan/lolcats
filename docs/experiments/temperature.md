# Experiment: temperature on the MMLU subset

**Status:** Run 1 finished: the stage 1 model at the temperatures 0.1, 0.5, 1 and 2. The prediction holds. The stage 2 model has not run yet.

## Question

Does the temperature of the model change the MMLU-subset accuracy of the Lizard model after stage 1 and after stage 2?

The idea comes from the llama-3.2-benchmark notebook ([references](../references.md)). In that notebook, temperature 0.1 increased the MMLU accuracy of Llama-3.2-1B-Instruct from 15.7% (run 2) to 30.5% (run 3).

## How this harness scores MMLU

- The harness (`b281b09`) sends four log-likelihood requests for each question. Each request has the same 5-shot prompt and one of the continuations " A", " B", " C" and " D" ([document 6](../06-evaluation-setup.md)).
- The predicted letter is the continuation with the highest log-likelihood. The harness does not sample text.
- The notebook generates text with sampling, so the temperature changes its answers. This harness has no such sampling step.

## Temperature in this pipeline

`compare_stages.py` divides the logits by the temperature T before the harness applies `log_softmax`. The patch is in `AutoCausalLM._model_call`, so it applies to the teacher, to the stage 1 model and to the stage 2 model. The gate measurement of `compare_stages.py` calls the model directly, so the temperature has no effect on it.

## Prediction

For a continuation of one token after the same prompt, the log-likelihood of letter c at temperature T is:

log p_T(c) = z_c / T − log Σ_v exp(z_v / T)

Here z_c is the logit of letter c, and the sum runs over the full vocabulary.

- The second term is the same for the four letters.
- Thus, for every T > 0, the four log-likelihoods have the same order as at T = 1.
- **Prediction:** the predicted letters, the accuracy and the letter shares stay the same at every temperature.
- Only the probabilities change: the mass on the four letters, the confidence and the entropy over the four letters. A lower T gives a higher confidence and a lower entropy.
- **Condition:** each letter is one token. The run records the number of tokens of " A" to " D". If a letter has more than one token, the temperature can change the order.

## Purpose of the experiment

1. It checks the prediction on the real checkpoints. If the prediction holds, the temperature cannot explain the gap or the collapse to "A" in this harness.
2. It measures how confident the "A" answers are, at T = 1 and at the other temperatures.
3. It separates the result of the notebook (generated text with sampling) from the method of this project (log-likelihoods).

## Method

| Setting | Value |
|---|---|
| Models | B. Lizard model after stage 1, without LoRA. C. Lizard model after stage 2. |
| Task | MMLU subset (5-shot, 5 questions per subject, 285 questions) |
| Temperatures | 0.1, 0.5, 1 and 2 by default. T = 1 is the reference. |
| Tool | `scripts/temperature.sh`, which runs `scripts/compare_stages.sh` once for each temperature |

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 scripts/temperature.sh   # all temperatures, models B and C
TEMPERATURES="0.1 1" MODELS=stage2 scripts/temperature.sh                     # a part of it
```

The full run has 8 evaluations of the MMLU subset (4 temperatures × 2 models). It probably takes 40–60 minutes on the A10. This value is an estimate from the earlier runs on the MMLU subset.

The run writes `results/temperature/<time>/`:

- `T=<t>/` for each temperature: a full run directory of `compare_stages.sh`. Its `env.txt` records the temperature.
- `summary.md` and `summary.json`: a comparison of the temperatures. For each model and temperature, it gives these values:
  - the accuracy and the letter shares,
  - the number of predictions that differ from T = 1,
  - the tokens per letter,
  - the mass, the confidence and the entropy.

## What to check

| Check | Expected value |
|---|---|
| Tokens per letter | 1 for each letter |
| Predictions that differ from T = 1 | 0 of 285, for each model and each temperature |
| Accuracy and letter shares | The same at each temperature. At T = 1: 22.5 (stage 1) and 24.9 (stage 2), as in quick check 2 of the [stage difference experiment](stage-difference.md). |
| Confidence | Increases when T decreases |
| Entropy | Decreases when T decreases |

If a check fails:

- If a letter has more than one token, the temperature can change the predictions. Then a change is a real effect of the temperature.
- If predictions change although each letter has one token, the cause is probably numerical. For example, float32 rounding can make two almost equal logits equal after the division.

## Code changes

- `compare_stages.py`:
  - The environment variable `TEMPERATURE` (default 1) sets the temperature.
  - `results.json` records the temperature and the number of tokens of each answer letter.
  - `summary.md` has a new table with the choice probabilities (mass, confidence and entropy) for MMLU runs.
  - A new step, `python scripts/compare_stages.py temperatures ROOT`, compares the runs in `ROOT/T=<t>/`.
- `compare_stages.sh`: `env.txt` records the temperature.
- `temperature.sh`: new.

## Tests

The test used the CPU setup of the `compare_stages.sh` tests. It has a tiny Llama, and fake stage 1 and stage 2 checkpoints from the loaders of the repository. Offline stand-ins replace the MMLU tasks in the pinned harness.

- `temperature.sh` with the temperatures 0.5, 1 and 2, for models B and C: exit 0, and 0 changed predictions.
- For each question, the distribution over the four letters at temperature T equals softmax(log-likelihoods at T = 1 / T). The largest difference is 6e-8. Thus the script applies the temperature exactly as intended.
- A lower T gave a higher confidence and a lower entropy, for both models.
- A run of the teacher with the temperatures 0.25 and 1 passed.
- `TEMPERATURE=0` stops with an error message.
- The summary step still works on result folders from before this change.

## Results

### Run 1: stage 1 model (2026-10-01)

- Run directory: `results/temperature/20261001-130135`
- Model: B, the Lizard model after stage 1, without LoRA (`MODELS=stage1`)
- Temperatures: 0.1, 0.5, 1 and 2 (the default values)
- Machine: `student06`, A10

| Temperature | Accuracy | "A" | "B" | "C" | "D" | Changed | Mass | Confidence | Entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 22.5 | 95.4% | 2.8% | 1.8% | 0.0% | 0 of 285 | 0.006 | 0.981 | 0.070 |
| 0.5 | 22.5 | 95.4% | 2.8% | 1.8% | 0.0% | 0 of 285 | 0.025 | 0.804 | 0.855 |
| 1 | 22.5 | 95.4% | 2.8% | 1.8% | 0.0% | 0 of 285 | 0.018 | 0.586 | 1.550 |
| 2 | 22.5 | 95.4% | 2.8% | 1.8% | 0.0% | 0 of 285 | 0.002 | 0.415 | 1.879 |

Each answer letter is one token.

| Check | Expected value | Result |
|---|---|---|
| Tokens per letter | 1 for each letter | 1. Pass. |
| Predictions that differ from T = 1 | 0 of 285 | 0 of 285 at each temperature. Pass. |
| Accuracy and letter shares | The same at each temperature, and 22.5 at T = 1 | 22.5 and the same letter shares at each temperature. These are also the values of quick check 2 of the [stage difference experiment](stage-difference.md). Pass. |
| Confidence | Increases when T decreases | 0.415 at T = 2, 0.981 at T = 0.1. Pass. |
| Entropy | Decreases when T decreases | 1.879 bits at T = 2, 0.070 bits at T = 0.1. Pass. |

**Finding 1: the temperature does not change the MMLU result of the stage 1 model.** The prediction holds on the real checkpoint. Thus, in this harness, the temperature cannot explain the gap or the collapse to "A".

**Finding 2: the stage 1 model puts little probability on the answer letters.**

- At T = 1, the four letters " A" to " D" together get 1.8% of the probability of the next token. This value is a mean over the questions.
- At T = 0.1, they get only 0.6%. A low temperature moves the probability to the most probable token. If a letter were the most probable token for a question, its probability at T = 0.1 would be near 1.
- Thus, for almost all questions, the most probable next token after "Answer:" is probably not an answer letter. The model does not follow the answer format of the 5-shot prompt well.
- The harness compares only the four letters, so it still gets an answer for each question.
- The mass is largest at T = 0.5 (0.025). This agrees with the explanation above. A lower temperature moves the probability to the most probable token, which is not a letter. A higher temperature spreads it over the full vocabulary.
- The value of the teacher is not measured yet. Without it, this experiment cannot show how much of this result is specific to the Lizard model.

**Finding 3: the preference for "A" is weak for each question, but constant.**

- At T = 1, the mean confidence over the four letters is 0.586. The mean entropy is 1.55 bits, and the maximum is 2 bits.
- But the model selects "A" for 95.4% of the questions.
- This agrees with a small, constant advantage for " A" that decides most questions. It also agrees with little information about the question in the four letter probabilities. This is an interpretation, not a measurement.

### Open items

1. **Stage 2 model:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MODELS=stage2 scripts/temperature.sh`.
2. **Teacher at T = 1:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MODELS=teacher TEMPERATURES=1 scripts/temperature.sh`. It gives the mass, the confidence and the entropy of the teacher, for a comparison with findings 2 and 3.

### Appendix: summary.md of run 1

The text below is the `summary.md` of run 1, without changes.

```markdown
# Temperature on the MMLU subset

Run directory: `results/temperature/20261001-130135`. Each temperature divides the logits before the harness takes log_softmax. "Changed" counts the questions whose predicted letter differs from the run at temperature 1 of the same model.

| Temperature | Model | Tokens per letter | Accuracy | "A" | "B" | "C" | "D" | Changed | Mass | Confidence | Entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | B. Lizard after stage 1 (no LoRA) | 1 | 22.5 | 95.4 | 2.8 | 1.8 | 0.0 | 0 of 285 | 0.006 | 0.981 | 0.070 |
| 0.5 | B. Lizard after stage 1 (no LoRA) | 1 | 22.5 | 95.4 | 2.8 | 1.8 | 0.0 | 0 of 285 | 0.025 | 0.804 | 0.855 |
| 1 | B. Lizard after stage 1 (no LoRA) | 1 | 22.5 | 95.4 | 2.8 | 1.8 | 0.0 | 0 of 285 | 0.018 | 0.586 | 1.550 |
| 2 | B. Lizard after stage 1 (no LoRA) | 1 | 22.5 | 95.4 | 2.8 | 1.8 | 0.0 | 0 of 285 | 0.002 | 0.415 | 1.879 |

Prediction: with one token per answer letter, the predicted letters and the accuracy are the same at every temperature above 0. Only the mass, the confidence and the entropy change.
```
