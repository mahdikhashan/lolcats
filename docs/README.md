# Lizard on Llama-3.2-1B with LoLCATs: project notes

These notes record the steps, decisions, experiments and results of this project. The project added Lizard attention ([arXiv:2507.09025](https://arxiv.org/abs/2507.09025)) to this LoLCATs repository, trained Llama-3.2-1B with it on Hugging Face Jobs, and evaluated the result. The notes cover the work of 2026-09-29 and 2026-09-30. They are source material for the thesis.

Each number in these notes has one of two labels. A **measured** number comes from a run or a test. An **estimated** number comes from a calculation.

## Documents

| # | Document | Contents |
|---|---|---|
| 1 | [Lizard in LoLCATs](01-lizard-in-lolcats.md) | How the Lizard layer from jku-thesis went into LoLCATs, the configs and the Makefile, and the differences from the jku-thesis pipeline |
| 2 | [Compute and cost](02-compute-and-cost.md) | Memory measurements, GPU choice, time and cost estimates, and the speeds of the real runs |
| 3 | [Infrastructure](03-infrastructure.md) | Docker image, conda environment, image builds, Hugging Face Jobs targets, W&B, and checkpoint names |
| 4 | [Training runs](04-training-runs.md) | The three training runs: Run 0, Run 1 and Run 2 |
| 5 | [NaN crash in stage 2](05-nan-crash.md) | Why Run 1 stopped, the numerical fix in the attention, and the parts that the fix leaves out |
| 6 | [Evaluation setup](06-evaluation-setup.md) | Harness setup, the problems found, `eval.sh`, and the analysis scripts |
| 7 | [Results](07-results.md) | All evaluation results with statistics |
| 8 | [Checks of the attention layer](08-verification.md) | Tests that show that the attention code calculates the equations of the paper and connects correctly into the model |
| 9 | [Comparison with the paper](09-paper-comparison.md) | Architecture against the paper, recipe differences, and the 1B results of the paper |
| 10 | [Open issues and next steps](10-open-issues-and-next-steps.md) | Known code issues, hypotheses, planned experiments and decision rules |
| 11 | [Gap analysis](11-gap-analysis.md) | Root cause analysis of the gap to the paper. Ranked causes, a hole in the check of checkpoint loading, and a debugging sequence. A plan to close the gap, the effect of each hyperparameter, and a ranking of the causes of the "A" collapse with a debugging plan |
| – | [Hyperparameters](hyperparameters.md) | Hyperparameters of the Lizard model with their sources, starting with the feature dimension of the feature maps |
| – | [Math formulas](math-formula.md) | The equations of the paper, each with its source and its code. They include the gated branch, the window branch, the stage 1 loss and the hardware-aware algorithm. |
| – | [References](references.md) | External sources, each with a link and a short description: the MMLU evaluation details of Meta for Llama 3.2, and an MMLU replication notebook |

## Experiments

Each experiment has its own document in `experiments/`, with its method, its runs and its results.

| Experiment | Question | Status |
|---|---|---|
| [Stage difference](experiments/stage-difference.md) | Does the gap start in stage 1 or in stage 2? Section 10 of the gap analysis. | Two quick checks done: stage 1 on PIQA, and stages 1 and 2 on the MMLU subset. Both point to stage 1. The full run with the teacher is not done yet. |
| [XAI for the distillation](experiments/xai.md) | Which interpretability methods can show if the distillation works, if the new parameters learn, and if the layers match the teacher? | Ideas only. Nothing has run yet. |
| [Temperature on the MMLU subset](experiments/temperature.md) | Does the temperature change the MMLU-subset accuracy after stage 1 and after stage 2? | Run 1 done (stage 1 at T = 0.1, 0.5, 1 and 2): the accuracy does not change, as predicted. Stage 2 not run yet. |
| [Feature dimension 32 in stage 1](experiments/feature-dimension.md) | Does a feature dimension of 32 (the LoLCATs rule) give a better stage 1 than 128? | Config and tools ready and tested on CPU. Stage 1 not trained yet. |

## Glossary

These notes use each term below with one meaning only.

| Term | Meaning |
|---|---|
| teacher | The unmodified `meta-llama/Llama-3.2-1B`. |
| Lizard attention | The attention of the paper: a gated branch plus a window branch. In code, `LolcatsLizardAttention`. |
| gated branch | Gated linear attention (GLA) with Hedgehog feature maps and one gate value per token. In code, `gla`. |
| window branch | Softmax attention over the last 128 tokens, with 4 sink logits. The paper calls it Anchor Window Attention (AWA). In code, `awa`. |
| feature maps | The Hedgehog maps `phi_q` and `phi_k`: φ(x) = [softmax(xW) ⊕ softmax(−xW)]. |
| gate | `W_gamma`. It gives the gate value γ = sigmoid(W_γ x) for each token. |
| sink logits | Four learnable scalars that appear only in the denominator of the window branch. In code, `meta_tokens`. |
| α | The learnable weight of the window branch: output = gated branch + α · window branch. In code, `alpha_blend`. |
| Lizard parameters | The five parameters above that Lizard adds to each layer: `phi_q`, `phi_k`, `W_gamma`, `meta_tokens`, `alpha_blend`. |
| stage 1 | Attention distillation. Only the Lizard parameters train. The loss compares the output of Lizard attention with the output of the teacher attention. |
| stage 2 | LoRA finetuning with the next-token cross-entropy loss. |
| recipe | The set of training hyperparameters of a run. |
| Run 0, Run 1, Run 2 | The three training runs. [Document 4](04-training-runs.md) describes them. |
| job | One container run on Hugging Face Jobs. |
| Lizard model | The model that the evaluation used: the stage 1 checkpoint of Run 1 plus the stage 2 checkpoint of Run 2. |
| harness | The LM Evaluation Harness at commit `b281b09`. |
| MMLU subset | The 285 MMLU questions that `--limit 5` selects: 5 questions from each of the 57 subjects. |
| accuracy, normalized accuracy | The harness metrics `acc` and `acc_norm`. |
| SE | Standard error. |
| point | One percentage point. |
| the paper | *Lizard: An Efficient Linearization Framework for Large Language Models*, arXiv:2507.09025, version 4. |

## Summary

- **Implementation.** Lizard attention became the LoLCATs attention type `lolcats_llama_lizard`. Its calculations match the jku-thesis reference to less than 1e-10 in float64. Its connection into a model gives the expected outputs. [Document 8](08-verification.md) gives the checks.
- **Training.** Stage 1 and stage 2 ran on one H200 on Hugging Face Jobs. The recipe came from the existing LoLCATs configs. Run 1 stopped in stage 2 with a NaN ([document 5](05-nan-crash.md)). After a numerical fix, Run 2 repeated stage 2 from the stage 1 checkpoint of Run 1 and finished ([document 4](04-training-runs.md)).
- **Results** ([document 7](07-results.md)):

  | Task | Lizard model | Teacher, this project | Paper, Lizard 1B | Paper, teacher 1B |
  |---|---|---|---|---|
  | MMLU 5-shot, all questions | 23.3 | not run yet | 29.8 | 31.0 |
  | MMLU 5-shot, MMLU subset | 24.9 | 33.7 | – | – |
  | PIQA (accuracy) | 68.0 | not run yet | 74.8 | 74.1 |
  | ARC-Easy (accuracy) | 54.8 | not run yet | 65.6 | 65.4 |

- **MMLU behavior.** On MMLU, the Lizard model selects "A" for almost every question. The per-subject accuracy has a correlation of +0.98 with the frequency of "A" as the right answer.
- **Both branches are active.** A disabled window branch drops PIQA accuracy to 57.2. A disabled gated branch drops it to 52.3.
- **Diagnosis so far.** The checks exclude the attention code, the connection into the model and the harness as causes. The recipe of the runs is different from the recipe of the paper. The stage 1 learning rate is 10× higher, with no warmup and no gradient clipping, and the schedules are constant. This difference is the most probable cause. A controlled run must test it ([document 10](10-open-issues-and-next-steps.md)).

## Timeline

| When (UTC) | Step | Details |
|---|---|---|
| 2026-09-29, approximately 11:00 | Lizard attention goes into LoLCATs | PR #1, [document 1](01-lizard-in-lolcats.md) |
| 2026-09-29, approximately 11:30 | GPU choice, memory estimates and cost estimates | [document 2](02-compute-and-cost.md) |
| 2026-09-29, approximately 13:00 | Training runs in Docker and pushes checkpoints to the Hub | PR #2, [document 3](03-infrastructure.md) |
| 2026-09-29, approximately 15:00 | GitHub Actions builds the image | PR #3 |
| 2026-09-29 15:44 | Run 0 (10 steps per stage) passes on an H200 | [document 4](04-training-runs.md) |
| 2026-09-29, approximately 15:45 | W&B entity fix | PR #4 |
| 2026-09-29 16:13 | Run 1 starts. Stage 1 completes. Stage 2 stops at step 386 | [document 5](05-nan-crash.md) |
| 2026-09-29 18:35–19:40 | NaN fix in the attention | PR #5 |
| 2026-09-29, approximately 20:10 | `make hf-job-finetune` repeats stage 2 only | PR #6 |
| 2026-09-30, before 01:40 | Run 2 finishes | [document 4](04-training-runs.md) |
| 2026-09-30 02:16 | The first evaluation job fails with `No module named 'src'` | [document 6](06-evaluation-setup.md) |
| 2026-09-30, approximately 13:45 | `eval.sh` for a GPU machine with conda only | PR #7 |
| 2026-09-30 14:25–14:40 | MMLU subset: Lizard model 24.9, teacher 33.7 | [document 7](07-results.md) |
| 2026-09-30 15:00–15:20 | Comparison with the paper, and checks of the attention code | documents 8 and 9 |
| 2026-09-30 19:10–19:20 | All of MMLU: 23.3. Answer-letter analysis | [document 7](07-results.md) |
| 2026-09-30 19:40 | PIQA 68.0, ARC-Easy 54.8 | [document 7](07-results.md) |
| 2026-09-30 20:00 | Disabled branches on PIQA | [document 7](07-results.md) |

## Pull requests

All these pull requests merged into `main` of `mahdikhashan/lolcats`.

| PR | Commits | Change |
|---|---|---|
| #1 | `c2dd4f0` | Adds Lizard attention, its model config, and a Makefile in place of `run.sh` |
| #2 | `f890c21`, `96e2f79` | Training in Docker: a conda environment from `environment.yaml`, a flash-attn wheel, checkpoint pushes to the Hub, no final evaluation, builds for linux/amd64 |
| – | `312616c` | Direct commit: longer conda download timeouts and more retries in the Dockerfile |
| #3 | `7f2d9cd` | A GitHub Actions workflow that builds and pushes the Docker image |
| #4 | `75d9408` | W&B runs go to the default entity of the API key |
| #5 | `cb8bc80` | Stops NaN in Lizard attention from `exp` overflow and from 0 / 0 |
| #6 | `47be168` | `make hf-job-finetune`: repeats stage 2 on Hugging Face Jobs from the stage 1 checkpoint |
| #7 | `a9bfe49` | `eval.sh`: evaluation on any GPU machine with conda. Configurable evaluation paths and cache directory |
| – | `baa7b3e`, `ee8b8c4` | Direct commits: `letters.py` (answer-letter analysis) and `ablate.py` (disabled branches) |
| #8 | `c3adf0b` | These notes |

## Key identifiers

- Code: `mahdikhashan/lolcats`, branch `main`. The thesis reference code is in `mahdikhashan/jku-thesis`.
- Scripts for evaluation and debugging: `scripts/`. [scripts/README.md](../scripts/README.md) lists them with their documents.
- Docker image: `mahdikhashan/lolcats` on Docker Hub.
- Checkpoints: the private Hugging Face repository `nanoman1/lolcats-lizard-llama-3.2-1b`.
- W&B: entity `nano-apps`, project `lolcats-personal`. The W&B run of Run 1 is `zyt5syvy`.
- Teacher: `meta-llama/Llama-3.2-1B`.
