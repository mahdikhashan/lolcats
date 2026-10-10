# Stage comparison: where the gap starts

This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `results/window_rope/stages-run1-arc`.

```
date (UTC): 2026-10-10 02:07:24
lolcats commit: 02b4ce9d30ca9233b552c98381f6368c19c26526
harness commit: b281b0921b636bc36ad05c0b0b0763bd6dd43463
models: stage1
tasks: arc_easy
temperature: 1
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
| ARC-Easy (0-shot) | 39.0 ± 1.0 (n = 2376) |
| ARC-Easy (0-shot), normalized | 37.8 ± 1.0 (n = 2376) |

## Checkpoints

Sections 3 and 9 of the gap analysis.

| Stage | File | SHA-256 | Size | Parameters | Dtypes | Bytes per parameter | Stored step | Stored losses |
|---|---|---|---|---|---|---|---|---|
| stage1 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt` | `7610c8976811267563f790070dbdb3575a778311138817e8fd725d5b0c56c815` | 637,670 B | 294,992 | {'torch.bfloat16': 80} | 2.16 | 1100 | {'distill/eval/loss': 3.2548828125} |

Trainable keys: for each evaluation, each expected trainable parameter of the model must be in the checkpoint and hold its value after the load.

| Evaluation | Checkpoint | Expected | Missing | Unexpected | Not loaded |
|---|---|---|---|---|---|
| stage1/arc_easy | stage1 | 80 | 0 | 0 | 0 |

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
