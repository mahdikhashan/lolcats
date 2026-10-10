# Stage comparison: where the gap starts

This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `results/window_rope_hybrid/stages`.

```
date (UTC): 2026-10-10 13:54:02
lolcats commit: 84228c550b815e3e94c3a78dda6d41f4e9805cf4
harness commit: b281b0921b636bc36ad05c0b0b0763bd6dd43463
models: stage1
tasks: mmlu_subset piqa arc_easy
temperature: 1
HF_REPO: nanoman1/lolcats-lizard-llama-3.2-1b
stage 1 checkpoint: checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
stage 2 checkpoint: checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.pt
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
| MMLU subset (5-shot, 5 questions per subject) | 25.3 ± 2.6 (n = 285) |
| PIQA (0-shot) | 58.9 ± 1.1 (n = 1838) |
| PIQA (0-shot), normalized | 58.5 ± 1.1 (n = 1838) |
| ARC-Easy (0-shot) | 36.0 ± 1.0 (n = 2376) |
| ARC-Easy (0-shot), normalized | 35.3 ± 1.0 (n = 2376) |

## Answer letters: MMLU subset (5-shot, 5 questions per subject)

Share of each predicted letter, in %. On MMLU, the Lizard model of Run 2 selected "A" for almost every question (section 4 of the gap analysis).

| Model | A | B | C | D | Accuracy | Accuracy if always "A" |
|---|---|---|---|---|---|---|
| B. Lizard after stage 1 (no LoRA) | 26.3 | 11.9 | 27.0 | 34.7 | 25.3 | 24.2 |

Right answers: A 24.2%, B 24.9%, C 25.3%, D 25.6%

Choice probabilities, as means over the questions. Mass: the probability of " A" to " D" together. Confidence: the largest of the four probabilities after normalization over the four letters. Entropy: over the four letters, in bits (maximum 2).

| Model | Temperature | Tokens per letter | Mass | Confidence | Entropy (bits) |
|---|---|---|---|---|---|
| B. Lizard after stage 1 (no LoRA) | 1 | 1 | 0.743 | 0.472 | 1.706 |

## Checkpoints

Sections 3 and 9 of the gap analysis.

| Stage | File | SHA-256 | Size | Parameters | Dtypes | Bytes per parameter | Stored step | Stored losses |
|---|---|---|---|---|---|---|---|---|
| stage1 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope_hybrid-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt` | `345d5ef72cdf4737953f7e7d56116f06d37faf39a7f1622943c1506f25e4001f` | 639,434 B | 294,992 | {'torch.bfloat16': 80} | 2.17 | 1100 | {'distill/eval/loss': 1.6171875} |

Trainable keys: for each evaluation, each expected trainable parameter of the model must be in the checkpoint and hold its value after the load.

| Evaluation | Checkpoint | Expected | Missing | Unexpected | Not loaded |
|---|---|---|---|---|---|
| stage1/mmlu_subset | stage1 | 80 | 0 | 0 | 0 |
| stage1/piqa | stage1 | 80 | 0 | 0 | 0 |
| stage1/arc_easy | stage1 | 80 | 0 | 0 | 0 |

Result: every expected trainable parameter loaded from its checkpoint.

## Lizard parameters and gates: B. Lizard after stage 1 (no LoRA)

Gate values on one 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens). Kept after w: the weight that the gated branch keeps on a token w positions before the last token. Initial values: gamma = 0.5, alpha = 1, feature-map weight RMS 0.02.

| Layer | Gamma mean | Std | Min | Max | < 1e-3 | > 0.999 | Kept after 128 | Kept after 256 | Kept after 512 | Alpha | Sink logits | φq RMS / max | φk RMS / max | ‖W_γ‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.036 | -1.07, -1.07, -1.07, -1.07 | 0.212 / 1.05 | 0.193 / 0.88 | 9.86 |
| 1 | 1.000 | 0.016 | 0.293 | 1.000 | 0.0% | 99.6% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.151 | 1.99, 1.99, 1.99, 1.99 | 0.293 / 3.66 | 0.250 / 1.89 | 10.66 |
| 2 | 0.000 | 0.009 | 0.000 | 0.401 | 100.0% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.196 | 2.55, 2.55, 2.55, 2.55 | 0.167 / 2.36 | 0.148 / 2.00 | 2.90 |
| 3 | 0.000 | 0.008 | 0.000 | 0.365 | 100.0% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.204 | 3.09, 3.09, 3.09, 3.09 | 0.217 / 1.73 | 0.191 / 1.98 | 3.11 |
| 4 | 0.000 | 0.014 | 0.000 | 0.629 | 100.0% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.200 | 2.67, 2.67, 2.67, 2.67 | 0.243 / 1.75 | 0.171 / 1.40 | 2.93 |
| 5 | 0.000 | 0.003 | 0.000 | 0.133 | 100.0% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.194 | 2.95, 2.95, 2.95, 2.95 | 0.281 / 4.00 | 0.181 / 1.26 | 2.94 |
| 6 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.028 | 1.54, 1.54, 1.54, 1.54 | 0.308 / 2.00 | 0.401 / 4.00 | 10.67 |
| 7 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.039 | 1.58, 1.58, 1.58, 1.58 | 0.292 / 1.66 | 0.286 / 1.61 | 11.09 |
| 8 | 0.001 | 0.022 | 0.000 | 0.997 | 99.2% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.111 | 2.09, 2.09, 2.09, 2.09 | 0.285 / 2.28 | 0.234 / 2.06 | 4.40 |
| 9 | 1.000 | 0.001 | 0.971 | 1.000 | 0.0% | 99.1% | 9.4e-01 | 9.4e-01 | 9.4e-01 | 0.022 | 1.15, 1.15, 1.15, 1.15 | 0.312 / 2.14 | 0.373 / 3.83 | 8.55 |
| 10 | 0.000 | 0.016 | 0.000 | 0.729 | 100.0% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.215 | 2.14, 2.14, 2.14, 2.14 | 0.281 / 2.36 | 0.201 / 1.32 | 3.88 |
| 11 | 0.000 | 0.001 | 0.000 | 0.053 | 100.0% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.177 | 2.00, 2.00, 2.00, 2.00 | 0.238 / 1.48 | 0.169 / 1.49 | 3.23 |
| 12 | 0.000 | 0.000 | 0.000 | 0.001 | 100.0% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.201 | 2.00, 2.00, 2.00, 2.00 | 0.228 / 2.27 | 0.174 / 1.31 | 2.87 |
| 13 | 0.000 | 0.000 | 0.000 | 0.002 | 100.0% | 0.0% | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.201 | 2.34, 2.34, 2.34, 2.34 | 0.228 / 3.36 | 0.177 / 1.38 | 3.76 |
| 14 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.061 | 0.35, 0.35, 0.35, 0.35 | 0.269 / 1.42 | 0.220 / 1.67 | 5.75 |
| 15 | 1.000 | 0.007 | 0.756 | 1.000 | 0.0% | 95.6% | 9.8e-01 | 9.7e-01 | 9.6e-01 | 0.094 | 2.00, 2.00, 2.00, 2.00 | 0.221 / 1.39 | 0.205 / 1.64 | 4.86 |
