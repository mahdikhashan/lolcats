# 4. Training runs

All runs used the Docker image `mahdikhashan/lolcats` on one H200 on Hugging Face Jobs.

## Recipe of the runs

The runs used the existing LoLCATs configs, not the recipe of the paper. [Document 9](09-paper-comparison.md) compares the two recipes.

| Setting | Stage 1: distillation (`distill_alpaca_clean_xent0_mse1000_lr1e-2_1b`) | Stage 2: finetuning (`finetune_lora_qkvo_alpaca_clean_1b`) |
|---|---|---|
| Data | yahma/alpaca-cleaned, concatenated into 2048-token sequences | Same as stage 1 |
| Trainable parameters | Lizard parameters (294,992) | LoRA r=8, α=16, dropout 0, on q/k/v/o (1,703,936). The Lizard parameters are frozen. |
| Loss | For each layer, 1000 × MSE between the outputs of Lizard attention and teacher attention | Next-token cross-entropy |
| Optimizer | AdamW (fused), learning rate 1e-2, weight decay 0 | AdamW (fused), learning rate 1e-4, weight decay 0 |
| Schedule | ReduceLROnPlateau (factor 0.1, patience 10 evaluations, minimum 1e-5). No warmup. | Same as stage 1 |
| Gradient clipping | None | None |
| Batch | 1 sequence × 8 gradient accumulation steps | Same as stage 1 |
| Epochs | 2 (1,178 gradient steps) | 2 (1,178 gradient steps) |
| Precision | bf16 model, Lizard calculations in float32 | Same as stage 1 |
| Checkpoint selection | Best validation loss. The validation runs every 100 steps. | Same as stage 1 |

## Run 0: smoke test (2026-09-29)

`make hf-job HF_FLAVOR=h200 HF_TIMEOUT=45m ARGS="--max_steps 10 --max_finetune_steps 10 --eval_steps 5"`

- **First attempt** (job `6abbdbaf4c46ef19870303d4`): It failed at the W&B login because of the entity problem. PR #4 fixed this problem ([document 3](03-infrastructure.md)).
- **Second attempt** (start 15:44:40, with `--no_wandb`): It passed from start to end.
  - The job downloaded Llama-3.2-1B (2.47 GB in approximately 7 s). It loaded Alpaca: 51,560 training examples and 200 validation examples.
  - `LolcatsLizardAttention` replaced the 16 `LlamaFlashAttention2` layers.
  - Stage 1 trained 294,992 parameters (0.024%). Stage 2 trained 1.7M LoRA parameters (0.14%).
  - The job pushed the checkpoints and the results CSV to the Hub. As intended, no evaluation ran.

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

A note about the stage 2 progress bar: it shows the loss divided by the 8 accumulation steps (1.14 × 8 ≈ 9.1). Thus read the validation values, not the value on the progress bar.

The log contains these messages. They do not show a problem.

- `No module named 'causal_attention_cuda'` and `Failed to import ThunderKittens`: These are optional kernels for other LoLCATs attention types. Lizard does not use them.
- `Bad LlamaTokenizer ... But resolved with AutoTokenizer`: LoLCATs first tries a local tokenizer folder. When that fails, it downloads the tokenizer.
- `FutureWarning: torch.load ... weights_only`: This is a notice from PyTorch.

## Run 1: first full run (2026-09-29, start 16:13:04)

`make hf-job` started this run on an H200, with W&B enabled: run `zyt5syvy` in `nano-apps/lolcats-personal`. The stage 2 speed (1.07–1.08 s per sequence) was the same as in the smoke test.

- **Stage 1 completed** all 1,178 gradient steps (W&B reported "current step 1178"). All later steps used its best checkpoint (`..._distill.pt`) as the stage 1 checkpoint.
- **Stage 2 started from a much better model than in the smoke test.** This shows that stage 1 had an effect.

  | Start of stage 2 | Validation loss | Perplexity |
  |---|---|---|
  | Smoke test (stage 1 ran 10 steps) | 8.85 | 7,207 |
  | Run 1 (stage 1 ran all steps) | 5.42 | 233 |

  After the start, the training perplexity decreased: 190, 146, 128 and 115 at steps 5, 6, 8 and 9. A short time before the crash, it was approximately 12–21.
- **W&B ignored every stage 2 metric**, because the step counter started again at 0 ([document 3](03-infrastructure.md)).
- **Stage 2 crashed** in epoch 0, at iteration ~3,094 of 4,714, gradient step 386, learning rate 1e-4. The forward pass gave a NaN. Then a debugger breakpoint stopped the job, which had no interactive terminal. [Document 5](05-nan-crash.md) gives the details. The best stage 2 checkpoint until that point (step ≤ 386) was already on the Hub.

## Run 2: stage 2 again, from the stage 1 checkpoint of Run 1 (2026-09-29/30)

PR #5 (the NaN fix) merged, and a new build of the image followed. Then `make hf-job-finetune` (PR #6) ran stage 2 alone.

1. The job downloaded `..._distill.pt` from the Hub.
2. It ran `make lizard ARGS="--load_distill_checkpoint default"`. This command does not run the distillation.
3. It finished before 2026-09-30 01:40 UTC and wrote `...-s=0-se=0-re=0-se=0-re=0_ft.pt` (3.49 MB).

[Document 7](07-results.md) gives the evaluation of this model.

These notes do not record the final stage 2 validation loss of this run. The value is in the results CSV on the Hub: `results/distill_llama3_2_1b_lizard_w128_fd128_m4/...-s=0-se=0-re=0-se=0-re=0_ft.csv`.

The table shows which parts Run 2 kept from Run 1, and which parts it changed.

| | Run 1, stage 2 | Run 2, stage 2 |
|---|---|---|
| Stage 1 weights | Trained in the same job | The best stage 1 checkpoint of Run 1 |
| Attention code | Before the NaN fix | With the NaN fix (mathematically identical, no overflow) |
| Optimizer, schedule, data order | – | New start of the 2-epoch schedule |
| Trainer | Anomaly detection and a breakpoint after a failed backward pass (no change) | Same as Run 1 |
