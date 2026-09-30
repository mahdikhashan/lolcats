# 4. Training runs

All runs used the Docker image `mahdikhashan/lolcats` on a single H200 on Hugging Face Jobs.

## Training recipe used

These are the existing LoLCATs configs, not the Lizard paper's recipe
(see [document 9](09-paper-comparison.md) for the comparison).

| Setting | Stage 1: distillation (`distill_alpaca_clean_xent0_mse1000_lr1e-2_1b`) | Stage 2: finetuning (`finetune_lora_qkvo_alpaca_clean_1b`) |
|---|---|---|
| Data | yahma/alpaca-cleaned, concatenated into 2048-token sequences | same |
| Trainable | Lizard parameters (294,992) | LoRA r=8, α=16, dropout 0 on q/k/v/o (1,703,936); Lizard parameters frozen |
| Loss | 1000 × MSE between Lizard and teacher attention outputs, per layer | Next-token cross-entropy |
| Optimizer | AdamW (fused), lr 1e-2, weight decay 0 | AdamW (fused), lr 1e-4, weight decay 0 |
| Schedule | ReduceLROnPlateau (factor 0.1, patience 10 evals, min 1e-5); no warmup | same |
| Gradient clipping | none | none |
| Batch | 1 sequence × 8 gradient accumulation | same |
| Epochs | 2 (1,178 gradient steps) | 2 (1,178 gradient steps) |
| Precision | bf16 model, Lizard math in float32 | same |
| Checkpoint selection | best validation loss, evaluated every 100 steps | same |

## Run 0: smoke test (2026-09-29)

`make hf-job HF_FLAVOR=h200 HF_TIMEOUT=45m ARGS="--max_steps 10 --max_finetune_steps 10 --eval_steps 5"`

- **First attempt** (job `6abbdbaf4c46ef19870303d4`) failed at W&B login: the entity problem
  fixed in PR #4 ([document 3](03-infrastructure.md)).
- **Second attempt** (started 15:44:40, with `--no_wandb`) passed end to end:
  - Llama-3.2-1B downloaded (2.47 GB in about 7 s); Alpaca loaded (51,560 train and 200 validation
    examples).
  - The 16 `LlamaFlashAttention2` layers were replaced by `LolcatsLizardAttention`.
  - Stage 1 trained 294,992 parameters (0.024%); stage 2 trained 1.7M LoRA parameters (0.14%).
  - Checkpoints and the results CSV were pushed to the Hub. No evaluation ran, as intended.

| Stage 1 step | Distillation loss (validation) |
|---|---|
| 0 | 76.9 |
| 5 | 33.3 |
| 10 | 17.1 |

| Stage 2 step | Validation loss | Perplexity |
|---|---|---|
| 0 | 8.85 | 7,207 |
| 5 | 7.57 | 1,989 |
| 10 | 6.87 | 996 |

Note on the stage 2 progress bar: the `loss` shown there is the loss divided by the 8 accumulation
steps (1.14 × 8 ≈ 9.1), so the validation numbers are the ones to read.

Harmless messages in the log:

- `No module named 'causal_attention_cuda'` and `Failed to import ThunderKittens`: optional kernels
  for other LoLCATs attention types; Lizard doesn't use them.
- `Bad LlamaTokenizer ... But resolved with AutoTokenizer`: LoLCATs first tries a local tokenizer
  folder, then downloads it.
- `FutureWarning: torch.load ... weights_only`: a PyTorch notice.

## Run 1: first full run (2026-09-29, started 16:13:04)

Launched with `make hf-job` on an H200 (the stage 2 speed, 1.07–1.08 s per sequence, matched the
smoke test), with W&B enabled: run `zyt5syvy` in `nano-apps/lolcats-personal`.

- **Stage 1 completed** all 1,178 gradient steps (W&B's "current step 1178"). Its best checkpoint
  (`..._distill.pt`) is the stage 1 checkpoint used for everything afterwards.
- **Stage 2 started from a much better model than in the smoke test**, showing stage 1 worked:

  | Stage 2 starting point | Validation loss | Perplexity |
  |---|---|---|
  | Smoke test (stage 1 ran 10 steps) | 8.85 | 7,207 |
  | Run 1 (stage 1 ran fully) | 5.42 | 233 |

  Training perplexity then fell (190, 146, 128, 115 at steps 5, 6, 8, 9), and was about 12–21
  shortly before the crash.
- **W&B dropped every stage 2 metric** because of the step counter restarting at 0
  ([document 3](03-infrastructure.md)).
- **Stage 2 crashed** in epoch 0 at iteration ~3,094 of 4,714, gradient step 386, lr 1e-4: a NaN
  in the forward pass, followed by a debugger breakpoint in a non-interactive job. Details in
  [document 5](05-nan-crash.md). The best stage 2 checkpoint up to that point (step ≤ 386) was
  already on the Hub.

## Run 2: stage 2 rerun from Run 1's stage 1 checkpoint (2026-09-29/30)

After the NaN fix (PR #5) was merged and the image rebuilt, stage 2 was rerun on its own with
`make hf-job-finetune` (PR #6). The job downloaded
`..._distill.pt` from the Hub and ran `make lizard ARGS="--load_distill_checkpoint default"`, which
skips distillation. It finished before 2026-09-30 01:40 UTC and wrote
`...-s=0-se=0-re=0-se=0-re=0_ft.pt` (3.49 MB). This is the model evaluated in
[document 7](07-results.md).

The final stage 2 validation loss of this run is not recorded in these notes. It is in the results
CSV `results/distill_llama3_2_1b_lizard_w128_fd128_m4/...-s=0-se=0-re=0-se=0-re=0_ft.csv` on the Hub.

What was carried over from Run 1, and what changed:

| | Run 1 stage 2 | Run 2 stage 2 |
|---|---|---|
| Stage 1 weights | Trained in the same job | Run 1's best stage 1 checkpoint |
| Attention code | Before the NaN fix | With the NaN fix (mathematically identical, overflow-safe) |
| Optimizer, schedule, data order | – | Fresh start of the 2-epoch schedule |
| Trainer | Anomaly detection + breakpoint on a failed backward (unchanged) | same |
