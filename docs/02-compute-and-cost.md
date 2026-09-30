# 2. Compute and cost

## Memory (measured on CPU, before any GPU run)

Peak memory was measured by running the actual training code on CPU at Llama-3.2-1B layer sizes
(random weights, 2048-token sequences, batch 1). Models with 1 and 2 layers were measured and the
result extrapolated linearly to 16 layers, with headroom added for CUDA overhead.

| Stage | Lizard | Previous LoLCATs config (Hedgehog + TK window) |
|---|---|---|
| Stage 1: attention distillation | ~22–25 GB | ~37–40 GB |
| Stage 2: LoRA finetuning | **~41–46 GB** | ~41–45 GB |

- **Stage 2 decides the GPU.** LoRA gradients flow back through every layer, and LoLCATs does not
  use gradient checkpointing there.
- **Why so much memory for a 1B model:** this Lizard implementation is a dense reference that
  materializes full L × L matrices (L = 2048) for both branches, in float32.
- **Measurement artifact.** Timing was measured with a PyTorch dispatch mode that forces the slow
  composite `cumprod` backward (0.09 s normally vs 6.31 s and 20,516 ops with the mode). This was
  subtracted from the time estimates.

## GPU choice (estimated)

| HF Jobs flavor | VRAM | Fits? | Estimated compute, both stages |
|---|---|---|---|
| `t4-*`, `l4x1`, `a10g-*` | 16–24 GB | No | – |
| `l40sx1` | 48 GB | Borderline for stage 2 | ~7 h |
| `a100-large` | 80 GB | Yes | ~5.5 h |
| `rtx-pro-6000` | 96 GB | Yes | ~3.5 h |
| `h200` | 141 GB | Yes | ~2 h |

- **Multi-GPU flavors don't help:** `distill_llama.py` trains on a single GPU.
- **The A100 is slower here than expected:** the Lizard math runs in float32 and LoLCATs never
  enables TF32, so those matmuls run at the A100's plain FP32 speed. The H200 has about 2.4× the
  memory bandwidth and 3.4× the FP32 throughput.
- **Assumed data size:** about 9K sequences per stage (2 epochs × 4–5K sequences of 2048 tokens),
  confirmed later by the smoke test (4,714 sequences per epoch).

### Cost estimate before the first run

Prices from `hf jobs hardware`: `a100-large` $2.50/h, `h200` $5.00/h. The estimates include about
15 minutes of setup and no evaluation.

| Flavor | Best case | Likely range | Cost |
|---|---|---|---|
| a100-large | ~5.7 h | ~7–11 h | ~$14–28 |
| h200 | ~2.2 h | ~2.7–4 h | ~$11–20 |

The H200 was chosen: about 2.7× faster and 20–25% cheaper despite twice the hourly price.

## Measured speed (H200, from the smoke test and the first full run)

| Quantity | Value |
|---|---|
| Alpaca-cleaned examples | 51,560 train, 200 validation |
| Sequences per epoch (2048 tokens, concatenated) | 4,714 (≈ 9.7M tokens) |
| Gradient steps per stage | 2 epochs × 589 = 1,178 (8 sequences per step) |
| Stage 1 speed | ~0.32 s per sequence (~3.2 sequences/s) → **~50 min** for the stage |
| Stage 2 speed | ~1.06–1.08 s per sequence → **~2 h 50 min** for the stage |
| Setup, evals every 100 steps, uploads | ~10 min |
| **Full run on H200** | **≈ 3 h 50 min, about $19** |

- Stage 1 matched the estimate almost exactly. Stage 2 was about twice as slow as estimated.
- **Suspected cause:** in stage 2 LoLCATs copies the full model output (~1 GB of logits) from GPU
  to CPU on every step (`outputs.cpu()` in `src/trainer/default_lm.py`). Removing it should speed
  up stage 2. Not changed so far.
- **Timeout:** `HF_TIMEOUT=6h` left about 2 hours of margin.

## Evaluation throughput (measured, A10 24 GB)

The evaluation ran on a university GPU machine with an A10, batch size 1
(see [document 6](06-evaluation-setup.md)):

| Run | Requests | Time | Rate |
|---|---|---|---|
| PIQA, full Lizard model | 3,676 | 3 min 11 s | 19.2 requests/s |
| PIQA, window branch removed | 3,676 | 2 min 25 s | 25.4 requests/s |
| PIQA, gated branch removed | 3,676 | 2 min 16 s | 27.0 requests/s |

MMLU 5-shot has about 56,000 requests (14,042 questions × 4 answer letters), with much longer
prompts, and runs longest prompts first. At batch size 1 on the A10, 24 GB was enough: no
out-of-memory error occurred.
