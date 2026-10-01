# 2. Compute and cost

## Memory (measured on CPU, before the GPU runs)

The measurement ran the training code on CPU, with the layer sizes of Llama-3.2-1B, random weights, 2048-token sequences and batch size 1. It used models with 1 layer and 2 layers. A linear extrapolation of these two results gave the value for 16 layers. The values in the table include a margin for CUDA overhead.

| Stage | Lizard | Previous LoLCATs config (Hedgehog + TK window) |
|---|---|---|
| Stage 1: attention distillation | ~22–25 GB | ~37–40 GB |
| Stage 2: LoRA finetuning | **~41–46 GB** | ~41–45 GB |

- **Stage 2 sets the GPU size.** The LoRA gradients go back through every layer, and LoLCATs does not use gradient checkpointing in stage 2.
- **Why a 1B model needs this much memory.** This Lizard implementation is a dense reference. For both branches, it makes full L × L matrices (L = 2048) in float32.
- **Error in the time measurement.** The timing used a PyTorch dispatch mode. This mode forces the slow composite backward of `cumprod`: 6.31 s and 20,516 operations, against 0.09 s without the mode. The time estimates do not include this difference.

## GPU choice (estimated)

| HF Jobs flavor | VRAM | Is the memory sufficient? | Estimated compute time, both stages |
|---|---|---|---|
| `t4-*`, `l4x1`, `a10g-*` | 16–24 GB | No | – |
| `l40sx1` | 48 GB | Borderline for stage 2 | ~7 h |
| `a100-large` | 80 GB | Yes | ~5.5 h |
| `rtx-pro-6000` | 96 GB | Yes | ~3.5 h |
| `h200` | 141 GB | Yes | ~2 h |

- **Flavors with more than one GPU do not help.** `distill_llama.py` trains on one GPU only.
- **The A100 is slower than expected for this code.** The Lizard calculations use float32, and LoLCATs never enables TF32. Thus these matrix multiplications run at the plain FP32 speed of the A100. The H200 has approximately 2.4× the memory bandwidth and 3.4× the FP32 throughput of the A100.
- **Data size in the estimate.** The estimate used approximately 9K sequences per stage: 2 epochs × 4–5K sequences of 2048 tokens. The smoke test later showed 4,714 sequences per epoch.

### Cost estimate before the first run

The prices come from `hf jobs hardware`: `a100-large` costs $2.50/h, and `h200` costs $5.00/h. The estimates include approximately 15 minutes of setup. They do not include evaluation.

| Flavor | Best case | Probable range | Cost |
|---|---|---|---|
| a100-large | ~5.7 h | ~7–11 h | ~$14–28 |
| h200 | ~2.2 h | ~2.7–4 h | ~$11–20 |

The project selected the H200. It is approximately 2.7× faster and 20–25% less expensive in total, although its price per hour is two times higher.

## Measured speed (H200, smoke test and first full run)

| Quantity | Value |
|---|---|
| Alpaca-cleaned examples | 51,560 for training, 200 for validation |
| Sequences per epoch (2048 tokens, concatenated) | 4,714 (≈ 9.7M tokens) |
| Gradient steps per stage | 2 epochs × 589 = 1,178 (8 sequences per step) |
| Stage 1 speed | ~0.32 s per sequence (~3.2 sequences/s) → **~50 min** for the stage |
| Stage 2 speed | ~1.06–1.08 s per sequence → **~2 h 50 min** for the stage |
| Setup, evaluations every 100 steps, uploads | ~10 min |
| **Full run on the H200** | **≈ 3 h 50 min, approximately $19** |

- Stage 1 took almost exactly the estimated time. Stage 2 took approximately two times the estimated time.
- **Suspected cause.** In stage 2, LoLCATs copies the full model output (~1 GB of logits) from the GPU to the CPU at every step. The line is `outputs.cpu()` in `src/trainer/default_lm.py`. Stage 2 will probably be faster without this copy. The code does not have this change yet.
- **Timeout.** With `HF_TIMEOUT=6h`, approximately 2 hours of margin remained.

## Evaluation speed (measured, A10 24 GB)

The evaluation ran on a university GPU machine with one A10, at batch size 1 ([document 6](06-evaluation-setup.md)).

| Run | Requests | Time | Rate |
|---|---|---|---|
| PIQA, full Lizard model | 3,676 | 3 min 11 s | 19.2 requests/s |
| PIQA, window branch removed | 3,676 | 2 min 25 s | 25.4 requests/s |
| PIQA, gated branch removed | 3,676 | 2 min 16 s | 27.0 requests/s |

MMLU 5-shot has approximately 56,000 requests: 14,042 questions × 4 answer letters. Its prompts are much longer, and the harness runs the longest prompts first. At batch size 1, the 24 GB of the A10 were sufficient. No out-of-memory error occurred.
