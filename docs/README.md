# Lizard on Llama-3.2-1B with LoLCATs: project notes

These notes record every step, decision, experiment and result from porting Lizard attention
([arXiv:2507.09025](https://arxiv.org/abs/2507.09025)) into this LoLCATs repository, training
Llama-3.2-1B with it on Hugging Face Jobs, and evaluating the result. They cover the work done
on 2026-09-29 and 2026-09-30 and are meant as source material for the thesis.

Numbers are marked as **measured** (from a run or a test) or **estimated** (from a calculation).

## Documents

| # | Document | Contents |
|---|---|---|
| 1 | [Lizard in LoLCATs](01-lizard-in-lolcats.md) | How the Lizard layer from jku-thesis was added to LoLCATs, the configs and Makefile, and how it differs from the thesis pipeline |
| 2 | [Compute and cost](02-compute-and-cost.md) | Memory measurements, GPU choice, time and cost estimates, and the speeds measured on real runs |
| 3 | [Infrastructure](03-infrastructure.md) | Docker image, conda environment, image builds, Hugging Face Jobs targets, W&B, checkpoint naming |
| 4 | [Training runs](04-training-runs.md) | Every training job: the smoke test, the first full run, and the finetune-only rerun |
| 5 | [NaN crash in stage 2](05-nan-crash.md) | Why the first full run crashed, the numerical fix in the attention, and what was left out |
| 6 | [Evaluation setup](06-evaluation-setup.md) | LM Evaluation Harness setup, every pitfall found, `eval.sh`, and the analysis scripts |
| 7 | [Results](07-results.md) | All evaluation results with statistics: MMLU, the answer-letter analysis, PIQA, ARC-Easy, branch removal |
| 8 | [Verifying the attention layer](08-verification.md) | Tests showing the attention code computes the paper's equations and is wired correctly |
| 9 | [Comparison with the paper](09-paper-comparison.md) | Architecture vs the paper, training recipe differences, the paper's 1B targets |
| 10 | [Open issues and next steps](10-open-issues-and-next-steps.md) | Known code issues, hypotheses, planned experiments and decision rules |

## Summary

- **Implementation.** Lizard (gated linear attention with Hedgehog feature maps, plus sliding-window
  softmax attention with 4 sink logits) was added as the LoLCATs attention type
  `lolcats_llama_lizard`. Its math matches the jku-thesis reference to below 1e-10 in float64, and
  swapping it into a model is wired correctly ([document 8](08-verification.md)).
- **Training.** Stage 1 (attention distillation) and stage 2 (LoRA finetuning) ran on a single H200
  on Hugging Face Jobs, using the LoLCATs hyperparameters from the existing configs. The first
  full run crashed in stage 2 with a NaN ([document 5](05-nan-crash.md)). After a numerical fix,
  stage 2 was rerun from the saved stage 1 checkpoint and finished ([document 4](04-training-runs.md)).
- **Results** ([document 7](07-results.md)):

  | Task | Lizard (this project) | Teacher (this project) | Paper, Lizard 1B | Paper, teacher 1B |
  |---|---|---|---|---|
  | MMLU 5-shot, full | 23.3 | not yet run | 29.8 | 31.0 |
  | MMLU 5-shot, 285-question subset | 24.9 | 33.7 | – | – |
  | PIQA (acc) | 68.0 | not yet run | 74.8 | 74.1 |
  | ARC-Easy (acc) | 54.8 | not yet run | 65.6 | 65.4 |

  On MMLU the model answers "A" regardless of the question (per-subject accuracy correlates +0.98
  with how often "A" is correct). Removing either attention branch at inference time drops PIQA
  to 57.2 (window branch off) or 52.3 (gated branch off), so both branches are in use.
- **Diagnosis so far.** The attention code, the model wiring and the evaluation harness have been
  checked and ruled out. The run used a training recipe that differs from the paper's (stage 1
  learning rate 10× higher, no warmup, no gradient clipping, constant schedules). This is the
  leading explanation, to be tested with a controlled rerun ([document 10](10-open-issues-and-next-steps.md)).

## Timeline

| When (UTC) | Step | Details |
|---|---|---|
| 2026-09-29 ~11:00 | Lizard added to LoLCATs | PR #1 ([document 1](01-lizard-in-lolcats.md)) |
| 2026-09-29 ~11:30 | GPU choice, memory and cost estimates | [document 2](02-compute-and-cost.md) |
| 2026-09-29 ~13:00 | Training dockerized, checkpoints pushed to the Hub | PR #2 ([document 3](03-infrastructure.md)) |
| 2026-09-29 ~15:00 | Image build moved to GitHub Actions | PR #3 |
| 2026-09-29 15:44 | Smoke test on H200 passes (10 steps per stage) | [document 4](04-training-runs.md) |
| 2026-09-29 ~15:45 | W&B entity fix | PR #4 |
| 2026-09-29 16:13 | First full run starts; stage 1 completes, stage 2 crashes at step 386 | [document 5](05-nan-crash.md) |
| 2026-09-29 18:35–19:40 | NaN fix in the attention | PR #5 |
| 2026-09-29 ~20:10 | `make hf-job-finetune` to rerun stage 2 only | PR #6 |
| 2026-09-30 before 01:40 | Stage 2 rerun finishes | [document 4](04-training-runs.md) |
| 2026-09-30 02:16 | First eval attempt on HF Jobs fails (`No module named 'src'`) | [document 6](06-evaluation-setup.md) |
| 2026-09-30 ~13:45 | `eval.sh` for a conda-only GPU machine | PR #7 |
| 2026-09-30 14:25–14:40 | MMLU on a 285-question subset: Lizard 24.9, teacher 33.7 | [document 7](07-results.md) |
| 2026-09-30 15:00–15:20 | Comparison with the paper; attention code verified | documents 8 and 9 |
| 2026-09-30 19:10–19:20 | Full MMLU 23.3; answer-letter analysis | [document 7](07-results.md) |
| 2026-09-30 19:40 | PIQA 68.0, ARC-Easy 54.8 | [document 7](07-results.md) |
| 2026-09-30 20:00 | Branch removal on PIQA | [document 7](07-results.md) |

## Pull requests

All merged into `main` of `mahdikhashan/lolcats`.

| PR | Commit(s) | Change |
|---|---|---|
| #1 | `c2dd4f0` | Add Lizard attention, its model config, and a Makefile replacing `run.sh` |
| #2 | `f890c21`, `96e2f79` | Dockerize training (conda env from `environment.yaml`, flash-attn wheel), push checkpoints to the Hub, drop the final eval; build for linux/amd64 |
| – | `312616c` | (direct commit) Longer conda download timeouts and retries in the Dockerfile |
| #3 | `7f2d9cd` | GitHub Actions workflow that builds and pushes the Docker image |
| #4 | `75d9408` | Log W&B runs to the API key's default entity |
| #5 | `cb8bc80` | Prevent NaN in Lizard attention from `exp` overflow and 0 / 0 |
| #6 | `47be168` | `make hf-job-finetune`: rerun stage 2 on HF Jobs from the stage 1 checkpoint |
| #7 | `a9bfe49` | `eval.sh`: evaluate on any conda GPU machine; configurable eval paths and cache directory |
| – | `baa7b3e`, `ee8b8c4` | (direct commits) `letters.py` (answer-letter analysis) and `ablate.py` (branch removal) |

## Key identifiers

- Code: `mahdikhashan/lolcats`, branch `main`; thesis reference code in `mahdikhashan/jku-thesis`.
- Docker image: `mahdikhashan/lolcats` on Docker Hub.
- Checkpoints: private Hugging Face repo `nanoman1/lolcats-lizard-llama-3.2-1b`.
- W&B: entity `nano-apps`, project `lolcats-personal`; first full run `zyt5syvy`.
- Teacher: `meta-llama/Llama-3.2-1B`.
