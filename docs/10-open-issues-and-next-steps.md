# 10. Open issues and next steps

## Known code issues (no fix yet)

| Issue | Location | Effect | Possible fix |
|---|---|---|---|
| `breakpoint()` after a failed backward pass, inside anomaly detection | `src/trainer/default_lm.py`, `train_step` | A NaN in a job without an interactive terminal stops the job with `bdb.BdbQuit`. | Remove it, or skip the batch and write it to the log. |
| No gradient clipping | `src/trainer/default_lm.py` | Different from the paper (clipping 1.0). No protection against loss spikes. | An optional `max_grad_norm` in the trainer config. The first version of PR #5 had it. The PR removed it on request. |
| The W&B step counter starts again at 0 in stage 2 | `default_lm.py`, `wandb.log(..., step=self.grad_step)` | W&B ignores all stage 2 metrics. | Let W&B count the steps, and log `train/step` as a metric. |
| The evaluation results CSV records MMLU as 0 | `lm_eval_harness/eval_lm_harness.py`: `save_results_to_dict` uses a local variable of `main()` | The CSV value for MMLU is wrong. The printed `MMLU RESULT` line is right. | Give the MMLU accuracies to the function as an argument. |
| Stage 2 copies the full logits to the CPU at every step | `default_lm.py` (`outputs.cpu()`) | Stage 2 takes approximately two times the estimated time. | Remove the copy. |
| The default W&B entity of the evaluation script is `hazy-research` | `eval_lm_harness.py` | W&B fails without `--no_wandb`. | Use the default entity, as in PR #4. |

## Missing measurements

1. **The teacher on PIQA and ARC-Easy in the same harness.** This gives the exact gap, because the paper used a newer harness. **Done on 2026-10-10:** PIQA 74.4, ARC-Easy 65.3 ([document 7](07-results.md)). Use the teacher command in [document 6](06-evaluation-setup.md) with `--task piqa` or `--task arc_easy`, with `--num_shots 0`, and without `--limit`.
2. **The teacher on all MMLU questions**, and `scripts/letters.py` on its log. The teacher should show no strong correlation with one letter. This would be different from the constant answer of the Lizard model.
3. **Gate statistics** with `gates.py` ([document 6](06-evaluation-setup.md)). They show how much information the gated branch keeps from tokens beyond the window.
4. **The stage 2 validation loss of Run 2**, from its results CSV on the Hub.
5. Optional: the remaining tasks of the paper. These are ARC-Challenge (normalized accuracy), HellaSwag (normalized accuracy) and WinoGrande (accuracy).

## Main hypothesis and the controlled test

**Hypothesis:** The recipe causes the gap to the paper, mainly the recipe of stage 1 ([document 9](09-paper-comparison.md)).

**Evidence so far:**

- The checks found no error in the attention calculations or in the connection into the model ([document 8](08-verification.md)).
- The harness works. The teacher gets 33.7 on the MMLU subset (285 questions), near the paper value of 31.0.
- Both branches are active (disabled branches on PIQA). Thus no component is missing or inactive.
- The damage to the model is wide, not only on MMLU. Against the paper, PIQA is 7 points lower and ARC-Easy is 11 points lower.
- The recipe is different from the paper in four settings: the stage 1 learning rate (10×), the schedule, the warmup and the clipping.

**Controlled test:** Change only the recipe. Keep the code, the data and the seed.

1. **Stage 1 with the recipe of the paper.** In lolcats, this needs a new distill config with these settings:
   - `lr: 1e-3`,
   - `lr_scheduler_type: cosine_warmup` with `num_warmup_steps: 118` and `num_training_steps: 1178` (2 epochs × 4,714 sequences ÷ 8),
   - `betas: [0.9, 0.99]` under `optimizer`,
   - gradient clipping at 1.0 in the trainer.

   The cosine schedule of HF decays to 0, not to 0.1×. This is a small difference at the end of the schedule. Give the new config file a new name, so that its checkpoints do not overwrite the current checkpoints.
2. **Compare the two stage 1 results before stage 2 starts.** Use held-out text for these measurements:
   - The relative error for each layer, ‖Ŷ − Y‖² / ‖Y‖², between the outputs of Lizard attention and teacher attention. Measure it for the current stage 1 and for the new stage 1.
   - Two reference points that need no training: Lizard attention at initialization, and the window branch alone (no gated branch). A well-trained stage 1 should be far below both.
   - The perplexity of the model with Lizard attention, against the perplexity of the teacher.
3. **Stage 2 with the recipe of the paper:** learning rate 5e-4, cosine schedule with 10% warmup, clipping at 1.0, LoRA on q/k/v. Also decide if the Lizard parameters stay trainable, as in jku-thesis. In lolcats, this needs `trainable_weights: [phi_q, phi_k, W_gamma, meta_tokens, alpha_blend]` under `finetune:`.
4. **Evaluate** these tasks:
   - MMLU with `--limit 20`: 1,140 questions, approximately ±1.3 points, the same questions for every model,
   - PIQA and ARC-Easy,
   - then all MMLU questions for the final model.

   The targets from the paper are PIQA ~74, ARC-Easy ~65 and MMLU ~30.

**How to read the outcome:**

| Outcome | Conclusion |
|---|---|
| The new stage 1 has a clearly lower error for each layer, and the final model gets near the values of the paper. | The recipe was the cause. |
| The new stage 1 is not better. | Examine the data and the sequence handling next: for example, how LoLCATs concatenates and chunks Alpaca, and how it normalizes the loss. |
| Stage 1 is better, but the final model is not better. | Examine stage 2: the learning rate and the trainable Lizard parameters. |

## Other options from the discussion

- Use the jku-thesis `train.py` directly, in place of new lolcats configs. It already uses the recipe of the paper.
- A hybrid variant that keeps some softmax layers (`softmax_attentions` in the model config). It limits the cost of the linearized layers. Table 10 of the paper reports these MMLU values on Llama-3-8B: 62.8 with 25% of the softmax layers kept, and 65.1 with 50%.

## How to reproduce the current results

1. Checkpoints: `nanoman1/lolcats-lizard-llama-3.2-1b`. The stage 1 file is `...-s=0-se=0-re=0_distill.pt`. The stage 2 file is `...-s=0-se=0-re=0-se=0-re=0_ft.pt`. [Document 3](03-infrastructure.md) gives the full names.
2. Environment: `CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yaml`.
3. Evaluation:
   - `./eval.sh` (MMLU),
   - `TASK=piqa NUM_SHOTS=0 ./eval.sh`,
   - `TASK=arc_easy NUM_SHOTS=0 ./eval.sh`,
   - `scripts/letters.py` and `scripts/ablate.py` as in [document 6](06-evaluation-setup.md).
4. Training from the start: `make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=<repo> HF_FLAVOR=h200 HF_TIMEOUT=6h`. Use a Docker image built from `main` ([document 3](03-infrastructure.md)).
