# 10. Open issues and next steps

## Known code issues (not fixed)

| Issue | Where | Effect | Possible fix |
|---|---|---|---|
| `breakpoint()` after a failed backward, inside anomaly detection | `src/trainer/default_lm.py`, `train_step` | Any NaN in a non-interactive job stops it with `bdb.BdbQuit` | Remove, or skip the batch and log it |
| No gradient clipping | `src/trainer/default_lm.py` | Differs from the paper (clipping 1.0); no protection against loss spikes | Optional `max_grad_norm` in the trainer config (was in the first version of PR #5, removed on request) |
| W&B step counter restarts at 0 in stage 2 | `default_lm.py`, `wandb.log(..., step=self.grad_step)` | All stage 2 metrics are dropped by W&B | Let W&B count steps and log `train/step` as a metric |
| MMLU recorded as 0 in the eval results CSV | `lm_eval_harness/eval_lm_harness.py`, `save_results_to_dict` uses a variable local to `main()` | CSV is wrong for MMLU; the printed `MMLU RESULT` line is correct | Pass the MMLU accuracies into the function |
| Full logits copied to CPU every step in stage 2 | `default_lm.py` (`outputs.cpu()`) | Stage 2 about twice as slow as estimated | Remove the copy |
| Eval script default W&B entity `hazy-research` | `eval_lm_harness.py` | W&B fails unless `--no_wandb` | Use the default entity, as in PR #4 |

## Measurements still missing

1. **Teacher on PIQA and ARC-Easy in the same harness**, for the exact gap (the paper used a newer
   harness). Command: the teacher command in [document 6](06-evaluation-setup.md) with
   `--task piqa` / `--task arc_easy` and `--num_shots 0`, without `--limit`.
2. **Teacher on full MMLU**, and `letters.py` on its log. The teacher should show no strong
   correlation with any single letter, which contrasts with Lizard's constant answer.
3. **Gate statistics** with `gates.py` ([document 6](06-evaluation-setup.md)): how much the gated
   branch keeps beyond the window.
4. **Stage 2 validation loss of Run 2**, from its results CSV on the Hub.
5. Optionally the remaining paper tasks: ARC-Challenge (acc_norm), HellaSwag (acc_norm), WinoGrande (acc).

## Main hypothesis and the controlled test

**Hypothesis:** the gap to the paper comes from the training recipe, mainly stage 1
([document 9](09-paper-comparison.md)).

**Evidence so far:**

- The attention math and the wiring are verified ([document 8](08-verification.md)).
- The harness works: the teacher scores 33.7 on the 285-question subset, near the paper's 31.0.
- Both branches are active (branch removal on PIQA), so no component is missing or dead.
- The model is degraded broadly (PIQA −7, ARC-Easy −11 points against the paper), not only on MMLU.
- The recipe differs from the paper in stage 1 learning rate (10×), schedule, warmup and clipping.

**Controlled test:** change only the recipe, keep the code, data and seed.

1. **Stage 1 with the paper's recipe.** In lolcats this needs a new distill config: `lr: 1e-3`,
   `lr_scheduler_type: cosine_warmup` with `num_warmup_steps: 118` and `num_training_steps: 1178`
   (2 epochs × 4,714 sequences ÷ 8), and `betas: [0.9, 0.99]` under `optimizer`; plus gradient
   clipping at 1.0 in the trainer. (HF's cosine schedule decays to 0 rather than 0.1×; a small
   difference at the end.) Use a new config file name so the checkpoints don't overwrite the
   current ones.
2. **Compare stage 1 before spending time on stage 2**, on held-out text:
   - per-layer relative error ‖Ŷ − Y‖² / ‖Y‖² between Lizard and teacher attention outputs, for the
     current and the new stage 1;
   - two reference points that need no training: Lizard at initialization, and the window branch
     alone (no gated branch). A well-trained stage 1 should be far below both;
   - perplexity of the model with Lizard swapped in, against the teacher.
3. **Stage 2 with the paper's recipe** (lr 5e-4, cosine with 10% warmup, clipping 1.0, LoRA on q/k/v),
   and decide whether to keep the Lizard parameters trainable, as jku-thesis does (in lolcats:
   `trainable_weights: [phi_q, phi_k, W_gamma, meta_tokens, alpha_blend]` under `finetune:`).
4. **Evaluate** MMLU with `--limit 20` (1,140 questions, about ±1.3 points, the same questions for
   every model), PIQA and ARC-Easy, then full MMLU for the final model.
   Targets from the paper: PIQA ~74, ARC-Easy ~65, MMLU ~30.

**Reading the outcome:**

| Outcome | Conclusion |
|---|---|
| New stage 1 has clearly lower per-layer error, and the final model approaches the paper's numbers | The recipe was the cause |
| New stage 1 is not better | Look at data and sequence handling next (e.g., how LoLCATs concatenates and chunks Alpaca, the loss normalization) |
| Stage 1 is better but the final model isn't | Look at stage 2 (learning rate, trainable Lizard parameters) |

## Other options discussed

- Use the jku-thesis `train.py` directly, which already follows the paper's recipe, instead of new
  lolcats configs.
- A hybrid variant that keeps some softmax layers (`softmax_attentions` in the model config), to
  bound what the linearized layers cost; the paper's Table 10 reports 62.8 MMLU with 25% softmax
  layers kept on Llama-3-8B, and 65.1 with 50%.

## Reproducing the current results

1. Checkpoints: `nanoman1/lolcats-lizard-llama-3.2-1b` (stage 1 `...-s=0-se=0-re=0_distill.pt`,
   stage 2 `...-s=0-se=0-re=0-se=0-re=0_ft.pt`; see [document 3](03-infrastructure.md)).
2. Environment: `CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yaml`.
3. Evaluation: `./eval.sh` (MMLU), `TASK=piqa NUM_SHOTS=0 ./eval.sh`,
   `TASK=arc_easy NUM_SHOTS=0 ./eval.sh`; `letters.py` and `ablate.py` as in
   [document 6](06-evaluation-setup.md).
4. Training from scratch: `make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=<repo> HF_FLAVOR=h200 HF_TIMEOUT=6h`
   with the Docker image built from `main` ([document 3](03-infrastructure.md)).
