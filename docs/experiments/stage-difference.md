# Experiment: stage difference

**Question:** Where does the gap start: in stage 1 (attention distillation) or in stage 2 (LoRA finetuning)? Section 10 of [the gap analysis](../11-gap-analysis.md) describes this experiment. The review gives it a higher priority than almost all other steps.

## Method

The experiment evaluates three models with the same harness and the same questions:

| | Model | Checkpoints |
|---|---|---|
| A | Teacher (`meta-llama/Llama-3.2-1B`) | None |
| B | Lizard model after stage 1, without LoRA | Stage 1 checkpoint of Run 1 |
| C | Lizard model after stage 2 | Stage 1 checkpoint of Run 1 and stage 2 checkpoint of Run 2. This is the model of [document 7](../07-results.md). |

The tasks are the MMLU subset (5-shot, 285 questions), PIQA (0-shot) and ARC-Easy (0-shot).

The tool is `compare_stages.sh` (PR #11, commit `ce6685a`). It runs on a GPU machine with the `lolcats-env` conda environment:

```bash
MODELS=stage1 TASKS=piqa ./compare_stages.sh   # quick check: model B on PIQA
./compare_stages.sh                            # full run: models A, B and C on the three tasks
```

Each run writes `results/stages/<time>/` with `summary.md`, `summary.json`, `env.txt`, the training results CSVs from the Hub, and the raw logs of each model and task.

**How to read the outcome:** the table in section 10 of the gap analysis gives three outcomes. The script compares the models question by question. A drop counts as clear when it is more than 2 paired SE.

## Runs

| Run | Date (UTC) | Models | Tasks | Machine | Status |
|---|---|---|---|---|---|
| Quick check | 2026-10-01 08:32 | B | PIQA | `student06` (A10) | Done |
| Full run | – | A, B, C | MMLU subset, PIQA, ARC-Easy | – | Not run yet |

## Quick check (2026-10-01)

- Command: `MODELS=stage1 TASKS=piqa ./compare_stages.sh`
- Run directory: `results/stages/20261001-103238`
- Code: lolcats `681a481` (`main` after PR #11), harness `b281b09`
- Software: Python 3.11.16, torch 2.5.1 (CUDA 12.4), transformers 4.43.1, peft 0.9.0
- GPUs in the machine: one A10, one V100 and two P40. `env.txt` does not show which GPU ran the evaluation.

### PIQA score

| Model | Run | Accuracy | Normalized accuracy | n |
|---|---|---|---|---|
| B. Lizard model after stage 1 | This quick check | **57.6 ± 1.2** | 55.5 ± 1.2 | 1,838 |
| C. Lizard model after stage 2 | [Document 7](../07-results.md) | 67.95 ± 1.09 | 66.59 ± 1.10 | 1,838 |
| Teacher, in the paper | Table 9 of the paper, newer harness | 74.1 | – | – |
| Chance | – | 50.0 | 50.0 | – |

**Finding: on PIQA, stage 2 does not cause the drop.** Model C is approximately 10.4 points above model B. The unpaired SE of this difference is approximately 1.6, so z ≈ 6.4. Thus stage 2 recovers approximately 10 points.

The teacher is not measured in this harness yet. Thus the size of the stage 1 drop is not known. With the teacher value of the paper (74.1), the stage 1 drop would be approximately 16.5 points. This preliminary result agrees with the outcome "stage 1 failure, and stage 2 recovers part of it". The full run must check it with the teacher and with paired statistics.

### Checkpoint check (sections 3 and 9 of the gap analysis)

| Checkpoint | SHA-256 | Size | Parameters | Dtype | Bytes per parameter | Stored step | Stored loss |
|---|---|---|---|---|---|---|---|
| Stage 1, Run 1 (`..._distill.pt`) | `7610c8976811267563f790070dbdb3575a778311138817e8fd725d5b0c56c815` | 637,670 B | 294,992 | bf16 (80 tensors) | 2.16 | 1100 | `distill/eval/loss` = 3.2549 |

- **The stage 1 load is complete**. All 80 expected Lizard parameters (16 layers × 5) are in the file and hold their values after the load. No key is missing or unexpected. Thus the loading hole of section 3 does not affect stage 1.
- The best stage 1 checkpoint comes from step 1,100 of 1,178.
- The parameter count, the dtype and the 2.16 bytes per parameter agree with [document 3](../03-infrastructure.md) and with factor 0 of section 12.
- The check of the stage 2 checkpoint needs model C, so it waits for the full run.

### Validation loss of Run 2 (section 9 of the gap analysis)

The Hub has the stage 2 results CSV of Run 2. It has no stage 1 results CSV under the expected name.

| Stage | Evaluations | First: step, loss | Best: step, loss | Last: step, loss |
|---|---|---|---|---|
| Stage 2, Run 2 | 11 | 100, 3.516 | 1100, 2.252 | 1100, 2.252 |

- **The final stage 2 validation loss of Run 2 is 2.252, at step 1,100** (perplexity approximately 9.5). Documents 4 and 10 list this value as missing.
- The best step is the last step. Thus the evaluated stage 2 checkpoint is the best checkpoint (section 9), and the validation loss still decreased at the last evaluation. This agrees with factor 3 of section 12 (a stage 2 learning rate 5× lower than in the paper), but it does not prove it.

### Gates and Lizard parameters (section 6, and factors 0 and 1 of section 12)

The script measured the gate values on one 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens). "Kept after w" is the weight that the gated branch keeps on a token w positions before the last token. The initial values are γ = 0.5, α = 1, sink logits 0 and feature-map weight RMS 0.02.

| Layer | γ mean | γ min | γ > 0.999 | Kept after 128 | Kept after 512 | α | Sink logit (all 4 equal) | ‖W_γ‖ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.990 | 0.543 | 58.0% | 0.35 | 0.0059 | 0.042 | 0.73 | 5.13 |
| 1 | 1.000 | 1.000 | 100.0% | 1.0 | 1.0 | 0.184 | 0.50 | 3.93 |
| 2 | 1.000 | 0.779 | 100.0% | 1.0 | 1.0 | 0.156 | 0.50 | 3.63 |
| 3 | 1.000 | 0.835 | 100.0% | 1.0 | 1.0 | 0.303 | 1.00 | 3.68 |
| 4 | 1.000 | 0.799 | 100.0% | 1.0 | 1.0 | 0.371 | 0.50 | 3.66 |
| 5 | 1.000 | 0.769 | 100.0% | 1.0 | 1.0 | 0.570 | 1.00 | 3.59 |
| 6 | 1.000 | 0.619 | 100.0% | 1.0 | 1.0 | 0.467 | 1.00 | 3.85 |
| 7 | 1.000 | 0.763 | 100.0% | 1.0 | 1.0 | 0.559 | 1.93 | 4.26 |
| 8 | 1.000 | 0.799 | 100.0% | 1.0 | 1.0 | 0.613 | 1.01 | 4.05 |
| 9 | 1.000 | 0.663 | 100.0% | 1.0 | 1.0 | 0.648 | 2.00 | 4.02 |
| 10 | 1.000 | 0.685 | 100.0% | 1.0 | 1.0 | 0.582 | 1.00 | 3.92 |
| 11 | 1.000 | 0.667 | 100.0% | 1.0 | 1.0 | 0.439 | 1.00 | 3.89 |
| 12 | 1.000 | 0.650 | 100.0% | 1.0 | 1.0 | 0.393 | 0.50 | 4.03 |
| 13 | 1.000 | 0.686 | 100.0% | 1.0 | 1.0 | 0.361 | 0.50 | 4.05 |
| 14 | 1.000 | 0.553 | 100.0% | 1.0 | 1.0 | 0.420 | 1.00 | 3.79 |
| 15 | 1.000 | 0.424 | 100.0% | 1.0 | 1.0 | 0.578 | 2.00 | 3.71 |

The feature-map weights have an RMS of 0.15–0.22 and a maximum absolute value of 0.80–2.30 (the full table is in the appendix).

**Finding 1: the gate saturates at 1 in layers 1–15.**

- In layers 1–15, γ is above 0.999 for 100.0% of the 2048 tokens after rounding. Thus at most one token per layer has a lower γ, which gives the minimum values of 0.42–0.84. The standard deviation is 0.013 or less.
- The weight kept after 512 tokens rounds to 1.0, so it is at least 0.95. Thus the gated branch keeps almost the full weight of all earlier tokens, and its decay has almost no effect.
- Only layer 0 has a decay: it keeps 0.35 of the weight after 128 tokens, and 0.0059 after 512 tokens.
- Factor 1 of section 12 predicted this failure mode. Its words: "`gates.py` should show γ near 0 or 1 in many layers, with little variation from token to token". The prediction holds for γ near 1.
- [Document 7](../07-results.md) calls the gate decay the only source of position information in the model. With γ ≈ 1, this source is absent in 15 of 16 layers. This can explain part of the damage on PIQA and ARC-Easy. This experiment does not prove that.
- These values come from model B. Stage 2 keeps `W_gamma` frozen, but LoRA changes the input of the gate in layers 1–15. The full run gives the same table for model C.

**Finding 2: α decreased from 1 in every layer.** The values are 0.042–0.648. In layer 0, α = 0.042, so the window branch has almost no weight there. α moved far from its initial value, as factor 0 expects at learning rate 1e-2.

**Finding 3: the sink logits look like the result of bf16 rounding.**

- In each layer, the four sink logits are equal. They start equal at 0 and receive equal gradients. Thus they act as one sink with the logit t + ln 4 ([document 8](../08-verification.md)).
- In 13 of 16 layers, the value is exactly 0.50, 1.00 or 2.00. In layer 8, it is 1.01, which is the next bf16 value above 1.0 (1.0078).
- bf16 spacing doubles at each power of two. Thus an Adam step that is large enough to reach a power of two can be too small to go past it. Then the value stops there.
- A simulation shows the same behavior. It used one bf16 scalar, AdamW at learning rate 1e-2, 1,178 steps, and a synthetic noisy gradient that pushes the value up. The bf16 value stopped at or just above a power of two (4.0, 2.06, 1.125 and 0.547). It ended at 47–66% of the float32 value of the same simulation.
- Factor 0 of section 12 says that stage 1 at learning rate 1e-2 loses no updates to rounding. That statement used Adam steps of size lr or 0.3 lr. The sink logits suggest that smaller steps are lost even at 1e-2. This is probable, not proved.

## Interpretation so far

- The stage 1 checkpoint loads completely. Thus the quick check measures the trained stage 1 state, not a loading error.
- On PIQA, the stage 1 model is approximately 10 points below the stage 2 model. Thus the damage on PIQA is already present after stage 1, and stage 2 recovers part of it. The teacher baseline in this harness is still necessary to measure the full stage 1 drop.
- The gate of the stage 1 model does not decay in 15 of 16 layers. This is a concrete defect of the stage 1 state. It agrees with the main hypothesis of the gap analysis: the stage 1 recipe gives a poor Lizard state.
- The run-2 validation loss still decreased at the end of stage 2, and the sink logits show signs of bf16 rounding. Both agree with section 12, but neither is proof.

## Open items

1. **Full run:** `./compare_stages.sh` for models A, B and C on the MMLU subset, PIQA and ARC-Easy. It gives the teacher baseline, the paired statistics and the stage 2 checkpoint check.
2. **GPU choice:** The machine has four GPUs. For the full run, select the A10 with `CUDA_VISIBLE_DEVICES`, and record it.
3. **Gate table of model C:** Compare it with the table above.
4. **Other documents:** Document 4 and document 10 can now record the final stage 2 validation loss of Run 2 (2.252 at step 1,100).

## Appendix: summary.md of the quick check

The text below is the `summary.md` of the quick check, without changes.

````markdown
# Stage comparison: where the gap starts

This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `results/stages/20261001-103238`.

```
date (UTC): 2026-10-01 08:32:56
lolcats commit: 681a481957dea7c92ce9afc7ad9ed95374e9fbd4
harness commit: b281b0921b636bc36ad05c0b0b0763bd6dd43463
models: stage1
tasks: piqa
HF_REPO: nanoman1/lolcats-lizard-llama-3.2-1b
stage 1 checkpoint: checkpoints/distill_llama3_2_1b_lizard_w128_fd128_m4/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
stage 2 checkpoint: checkpoints/distill_llama3_2_1b_lizard_w128_fd128_m4/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.pt
python 3.11.16, torch 2.5.1 (CUDA 12.4), transformers 4.43.1, peft 0.9.0
NVIDIA A10, 23028 MiB, 580.126.09
Tesla V100-PCIE-16GB, 16384 MiB, 580.126.09
Tesla P40, 24576 MiB, 580.126.09
Tesla P40, 24576 MiB, 580.126.09
```

## Scores

Accuracy in %, ± the binomial SE. n is the number of questions.

| Task | B. Lizard after stage 1 (no LoRA) |
|---|---|
| PIQA (0-shot) | 57.6 ± 1.2 (n = 1838) |
| PIQA (0-shot), normalized | 55.5 ± 1.2 (n = 1838) |

## Checkpoints

Sections 3 and 9 of the gap analysis.

| Stage | File | SHA-256 | Size | Parameters | Dtypes | Bytes per parameter | Stored step | Stored losses |
|---|---|---|---|---|---|---|---|---|
| stage1 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt` | `7610c8976811267563f790070dbdb3575a778311138817e8fd725d5b0c56c815` | 637,670 B | 294,992 | {'torch.bfloat16': 80} | 2.16 | 1100 | {'distill/eval/loss': 3.2548828125} |

Trainable keys: for each evaluation, each expected trainable parameter of the model must be in the checkpoint and hold its value after the load.

| Evaluation | Checkpoint | Expected | Missing | Unexpected | Not loaded |
|---|---|---|---|---|---|
| stage1/piqa | stage1 | 80 | 0 | 0 | 0 |

Result: every expected trainable parameter loaded from its checkpoint.

## Validation loss during training

From the training results CSVs in `training/`. The trainer saves the checkpoint at the best step, so the stored step of the checkpoint should equal the best step.

| Stage | File | Evaluations | First: step, loss | Best: step, loss | Last: step, loss | Stored step of the checkpoint |
|---|---|---|---|---|---|---|
| stage2 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.csv` | 11 | 100, 3.5160 | 1100, 2.2520 | 1100, 2.2520 | – |

## Lizard parameters and gates: B. Lizard after stage 1 (no LoRA)

Gate values on one 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens). Kept after w: the weight that the gated branch keeps on a token w positions before the last token. Initial values: gamma = 0.5, alpha = 1, feature-map weight RMS 0.02.

| Layer | Gamma mean | Std | Min | Max | < 1e-3 | > 0.999 | Kept after 128 | Kept after 256 | Kept after 512 | Alpha | Sink logits | φq RMS / max | φk RMS / max | ‖W_γ‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.990 | 0.028 | 0.543 | 1.000 | 0.0% | 58.0% | 3.5e-01 | 1.1e-01 | 5.9e-03 | 0.042 | 0.73, 0.73, 0.73, 0.73 | 0.182 / 1.10 | 0.180 / 1.07 | 5.13 |
| 1 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.184 | 0.50, 0.50, 0.50, 0.50 | 0.193 / 1.05 | 0.148 / 0.80 | 3.93 |
| 2 | 1.000 | 0.005 | 0.779 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.156 | 0.50, 0.50, 0.50, 0.50 | 0.201 / 0.99 | 0.182 / 0.84 | 3.63 |
| 3 | 1.000 | 0.004 | 0.835 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.303 | 1.00, 1.00, 1.00, 1.00 | 0.200 / 1.25 | 0.171 / 1.48 | 3.68 |
| 4 | 1.000 | 0.004 | 0.799 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.371 | 0.50, 0.50, 0.50, 0.50 | 0.212 / 1.31 | 0.205 / 1.11 | 3.66 |
| 5 | 1.000 | 0.005 | 0.769 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.570 | 1.00, 1.00, 1.00, 1.00 | 0.217 / 1.22 | 0.193 / 1.05 | 3.59 |
| 6 | 1.000 | 0.008 | 0.619 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.467 | 1.00, 1.00, 1.00, 1.00 | 0.212 / 1.02 | 0.198 / 0.86 | 3.85 |
| 7 | 1.000 | 0.005 | 0.763 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.559 | 1.93, 1.93, 1.93, 1.93 | 0.206 / 1.01 | 0.206 / 1.02 | 4.26 |
| 8 | 1.000 | 0.004 | 0.799 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.613 | 1.01, 1.01, 1.01, 1.01 | 0.216 / 1.48 | 0.203 / 1.04 | 4.05 |
| 9 | 1.000 | 0.007 | 0.663 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.648 | 2.00, 2.00, 2.00, 2.00 | 0.204 / 1.26 | 0.196 / 1.23 | 4.02 |
| 10 | 1.000 | 0.007 | 0.685 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.582 | 1.00, 1.00, 1.00, 1.00 | 0.203 / 1.17 | 0.201 / 1.08 | 3.92 |
| 11 | 1.000 | 0.007 | 0.667 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.439 | 1.00, 1.00, 1.00, 1.00 | 0.201 / 1.18 | 0.185 / 1.00 | 3.89 |
| 12 | 1.000 | 0.008 | 0.650 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.393 | 0.50, 0.50, 0.50, 0.50 | 0.197 / 1.23 | 0.174 / 0.85 | 4.03 |
| 13 | 1.000 | 0.007 | 0.686 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.361 | 0.50, 0.50, 0.50, 0.50 | 0.182 / 1.19 | 0.174 / 1.15 | 4.05 |
| 14 | 1.000 | 0.010 | 0.553 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.420 | 1.00, 1.00, 1.00, 1.00 | 0.207 / 1.16 | 0.185 / 1.11 | 3.79 |
| 15 | 1.000 | 0.013 | 0.424 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.578 | 2.00, 2.00, 2.00, 2.00 | 0.196 / 1.91 | 0.195 / 2.30 | 3.71 |
````
