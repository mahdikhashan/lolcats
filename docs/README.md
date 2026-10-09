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
| 12 | [Gap analysis 2](12-gap-analysis-2.md) | The gap analysis again, with the recipe, layer-wise MSE and attention weight results. A limit for each head as the new main cause, an updated ranking, and new XAI, code and hyperparameter experiments |
| 13 | [Lizard attention v2](13-lizard-attention-v2.md) | A second Lizard attention file with an option for each reading of the paper (D1, D2, D3, α) and each code change C1–C6. Literature check, proof audit and computation audit with the Mathbox skills. The defaults give v1. A plan with fewer runs (R1–R7). The first stage 1 run, C1 (one α for each head), gave a validation loss of 3.8092 (−4.2% against config 1). The MSE of layer 15, head 14 fell by 7.8%, with no accuracy gain. R1 (`joint`) stopped, because its gated branch had almost no weight. R1b (C1, C2, `hybrid`) gave 3.0542 (−23.2%), the lowest stage 1 loss so far. But ARC-Easy fell from 36.5 to 29.9. In layer 15, the shared gate fell to 0.128: heads 14 and 23 improved by 76–78%, but 28 of the 32 heads got worse. Next: checks of layer 15 and of the positions. |
| 14 | [Liger and GLA](14-liger-gla.md) | The math of Liger and GLA, from their papers and official code, and a comparison with Lizard. Proof audit P7–P11 with float64 checks. Four equal sinks act as one sink, and the row form has a total weight above 1. With `hybrid`, the window share falls as 1 / position. A window without RoPE cannot find the previous token. Critical review: the stage 1 loss does not predict the accuracy. Proposal: `gla_norm: convex` (the mix 0.5 and 0.5 of Liger at α = 1), then `window_rope` and `gate_per_head` on config 1. |
| 15 | [Attention math side by side](15-attention-math-side-by-side.md) | One table with the attention math of Lizard, LoLCATs, GLA and Liger, part by part, from the papers and the code. A proof audit of eight claims, by derivation. Only LoLCATs gives an exact weighted mean of the values, and only the window of Lizard has no RoPE. |
| 16 | [Optuna plan](16-optuna-plan.md) | A plan for a search of stage 1 hyperparameters with Optuna. The score is the mean of PIQA and ARC-Easy, not the loss. The plan has four phases. First, a check of short trials against full runs. Then planned runs for the design options. Then 25 short trials for the learning rate, the start value of α and the gate bias. Last, full runs of the best trials. Approximately $220 and 52 hours (estimates). |
| 17 | [Reproduction studies](17-reproduction-studies.md) | A review of a ChatGPT answer about earlier reproduction studies of language models (OpenLLaMA, Pythia and others), against these notes. The answer treats GLA plus the window as a change, but it is Lizard itself. New and cheap: checkpoints at each evaluation, the hidden-state distance and the KL of the logits. Four research questions for the thesis, mapped to these documents. The cited papers are not checked. |
| 18 | [Timeline of the work on the gap](18-gap-timeline.md) | One table with every try and idea against the gap, in the order of time: the activity, the result, the challenges and the document. A list of the ideas that have not run yet, and a short summary. |
| 19 | [Next steps from the literature](19-next-steps-from-literature.md) | A web search for the next steps. At 1B, two other methods also get MMLU near chance (Liger paper: LoLCATs 23.1, Liger 22.4). Plan part A completes the reproduction: the teacher baseline, a LoLCATs control run, an MMLU test with rotated options and a message to the authors. Part B lists extensions: softmax layers, KL to the teacher logits, a gate start from teacher statistics and `window_rope`. |
| – | [Hyperparameters](hyperparameters.md) | Hyperparameters of the Lizard model with their sources, starting with the feature dimension of the feature maps |
| – | [Math formulas](math-formula.md) | The equations of the paper, each with its source and its code. They include the gated branch, the window branch, the stage 1 loss and the hardware-aware algorithm. |
| – | [Math against code](math-code-discrepancy.md) | Each formula of the paper next to the code, with a list of 17 differences, their types and their possible effects |
| – | [References](references.md) | External sources, each with a link and a short description: the MMLU evaluation details of Meta for Llama 3.2, and an MMLU replication notebook |
| – | [Definitions](appendix-definitions.md) | An appendix for the thesis. It gives short definitions of fine-tuning, causal language modeling, RLHF, packing, the EOS token, padding, context size and batch. An answer from ChatGPT, kept without changes. |
| – | [Evaluation](evaluation.md) | An answer from ChatGPT about the metrics for the evaluation of Lizard, kept without changes. It covers perplexity, benchmarks, BERTScore, KL divergence of the logits and efficiency. Its example tables do not contain results of this project. |

## Experiments

Each experiment has its own document in `experiments/`, with its method, its runs and its results.

| Experiment | Question | Status |
|---|---|---|
| [Stage difference](experiments/stage-difference.md) | Does the gap start in stage 1 or in stage 2? Section 10 of the gap analysis. | Two quick checks done: stage 1 on PIQA, and stages 1 and 2 on the MMLU subset. Both point to stage 1. The full run with the teacher is not done yet. |
| [XAI for the distillation](experiments/xai.md) | Which interpretability methods can show if the distillation works, if the new parameters learn, and if the layers match the teacher? | Ideas only. Nothing has run yet. |
| [Temperature on the MMLU subset](experiments/temperature.md) | Does the temperature change the MMLU-subset accuracy after stage 1 and after stage 2? | Run 1 done (stage 1 at T = 0.1, 0.5, 1 and 2): the accuracy does not change, as predicted. Stage 2 not run yet. |
| [Feature dimension 32 in stage 1](experiments/feature-dimension.md) | Does a feature dimension of 32 (the LoLCATs rule) give a better stage 1 than 128? | Stage 1 trained. MMLU subset: 23.2 against 22.5 for 128, with a higher validation loss (3.42 against 3.25) and the same gate saturation. Not a cause of the gap. PIQA not run yet. |
| [Learning rate of the paper in stage 1](experiments/paper-lr.md) | Does the stage 1 recipe of the paper (peak 1e-3, cosine schedule, 10% warmup) give a better stage 1 and stop the gate saturation? Feature dimension 32. | Stage 1 trained. The gate does not saturate, but α stays at 1.000 (bf16) and the validation loss is 2.4× higher (8.16 against 3.42). MMLU subset 25.3 with no "A" preference, PIQA 55.8, ARC-Easy 34.1. Next: float32 trainable weights. |
| [float32 in stage 1](experiments/float32.md) | Does float32 (all weights, the AdamW states and the stage 1 target) fix the frozen α and the high validation loss of the paper-LR run? | Stage 1 trained. α moves (0.60–0.79) and the validation loss drops from 8.16 (bf16) to 4.95, but stays above 3.42 (LoLCATs recipe). MMLU subset 24.6, PIQA 57.7, ARC-Easy 35.7: not clearly better than the other stage 1 models. |
| [Second round: β2 and minimum learning rate](experiments/second-round.md) | Does stage 1 become better when β2 and the minimum learning rate also follow the paper? Gradient clipping is a separate experiment. | Config 1 trained and evaluated on 2026-10-02. Validation loss 3.9764 (float32 run: 4.9478, LoLCATs recipe: 3.4219). MMLU subset 26.7, PIQA 57.3, ARC-Easy 36.5. |
| [Gradient clipping in stage 1](experiments/gradient-clipping.md) | Does gradient clipping at 1.0 (Table 13 of the paper) change stage 1, on top of the second round (config 1)? The trainer logs the gradient norm. | Config 2 trained and evaluated on 2026-10-02. Validation loss 3.5092 (config 1: 3.9764, LoLCATs recipe: 3.4219). MMLU subset 23.2, PIQA 57.5, ARC-Easy 35.6. The gate saturates more, and α is lower. |
| [Stage 2 on config 1](experiments/stage2-config1.md) | Does config 1 in both stages give a better final model than Run 2? Stage 2 on the second-round checkpoint (config 1), in float32. It uses the stage 2 settings of Table 13 (5e-4, LoRA on q, k, v) and the optimizer settings of config 1 (no clipping). | Stage 2 trained and evaluated. PIQA 67.7 and ARC-Easy 54.9, within 0.3 points of Run 2 (67.95, 54.8). Validation loss 1.9158 (Run 2: 2.252). MMLU subset 23.5, with 77.5% "A" answers. Stage 2 shortens the memory of the gate in layers 9–15. |
| [XAI: layer-wise MSE](experiments/xai-layer-wise-mse.md) | In which layers does the stage 1 Lizard attention differ most from the teacher attention? A plot like Figure 14 of the LoLCATs paper, from saved checkpoints without retraining. | Run 1 done (five stage 1 checkpoints): the script reproduces each stored loss. Layer 15 gives 24–31% of the loss, and its heads 14 and 23 alone give up to 21%. The recipes of the paper differ most from the LoLCATs recipe in layers 0–3 and 13–15. Run 2 (the LoLCATs attention as a control): its MSE is at least 3.0× lower in every layer, against all 7 Lizard checkpoints. Heads 14 and 23 of layer 15 give only 1.9% of its loss. Thus the cause is in the Lizard attention, not in the teacher heads. |
| [XAI: sample attention weights](experiments/xai-sample-attention-weight.md) | Does Lizard reproduce the attention weights of the teacher, also outside its window? Plots like Figures 18–21 of the LoLCATs paper, for stage 1, stage 2 and the initial weights. | Run 1 done (four checkpoints): mean TV distance 0.31–0.33 (initial weights 0.71). Local teacher heads match worst, because all heads of a layer get almost the same long-range share. Layer 15, head 14 is a local head. |
| [LoLCATs attention as a control in stage 1](experiments/lolcats-control.md) | Does the original LoLCATs attention give a better stage 1 than Lizard, with the same pipeline, data, recipe and harness? Step A2 of document 19, stage 1 only. | Run 1 done (2026-10-09). Validation loss 0.4442 against 3.2549 for Lizard. PIQA 73.5 (Lizard stage 1: 57.6, Lizard after stage 2: 68.0), ARC-Easy 62.8, MMLU subset 26.0 without the "A" collapse. On short prompts, the LoLCATs window computes the teacher attention. Per-layer table: on far queries, the window keeps 31–79% of the weight. Layer 0 keeps its window, but Lizard turned its window almost off there. Layer-wise MSE: 3.7–36.9× lower than Lizard Run 1 in each layer, with the largest difference in layer 0. Next: the teacher baseline, then Lizard with `window_rope`. |

## Figures

The folder `figures/` keeps the figures for the thesis. [figures/README.md](figures/README.md) gives each figure with its caption and a check against these notes.

| # | Figure | Contents |
|---|---|---|
| 1 | [Stage 1](figures/lizard-stage1.webp) | Stage 1 of the Lizard paper: the teacher, the data, the trained parts and the MSE loss |
| 2 | [Stage 2](figures/lizard-stage2.webp) | Stage 2 of the Lizard paper: LoRA, the causal language modeling loss and the settings of both stages |

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
