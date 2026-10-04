# Stage comparison: where the gap starts

This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `results/stages/20261004-050624-v2-perhead-hybrid`.

```
date (UTC): 2026-10-04 03:06:35
lolcats commit: 722ec8426580dbb53509aea6437e5998ee57d728
harness commit: b281b0921b636bc36ad05c0b0b0763bd6dd43463
models: stage1
tasks: piqa arc_easy mmlu_subset
temperature: 1
HF_REPO: nanoman1/lolcats-lizard-llama-3.2-1b
stage 1 checkpoint: checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_hybrid/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_hybrid-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
stage 2 checkpoint: checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_hybrid/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_hybrid-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.pt
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
| MMLU subset (5-shot, 5 questions per subject) | 24.9 ± 2.6 (n = 285) |
| PIQA (0-shot) | 55.9 ± 1.2 (n = 1838) |
| PIQA (0-shot), normalized | 54.1 ± 1.2 (n = 1838) |
| ARC-Easy (0-shot) | 29.9 ± 0.9 (n = 2376) |
| ARC-Easy (0-shot), normalized | 30.3 ± 0.9 (n = 2376) |

## Answer letters: MMLU subset (5-shot, 5 questions per subject)

Share of each predicted letter, in %. On MMLU, the Lizard model of Run 2 selected "A" for almost every question (section 4 of the gap analysis).

| Model | A | B | C | D | Accuracy | Accuracy if always "A" |
|---|---|---|---|---|---|---|
| B. Lizard after stage 1 (no LoRA) | 90.2 | 2.1 | 3.5 | 4.2 | 24.9 | 24.2 |

Right answers: A 24.2%, B 24.9%, C 25.3%, D 25.6%

Choice probabilities, as means over the questions. Mass: the probability of " A" to " D" together. Confidence: the largest of the four probabilities after normalization over the four letters. Entropy: over the four letters, in bits (maximum 2).

| Model | Temperature | Tokens per letter | Mass | Confidence | Entropy (bits) |
|---|---|---|---|---|---|
| B. Lizard after stage 1 (no LoRA) | 1 | 1 | 0.032 | 0.605 | 1.460 |

## Checkpoints

Sections 3 and 9 of the gap analysis.

| Stage | File | SHA-256 | Size | Parameters | Dtypes | Bytes per parameter | Stored step | Stored losses |
|---|---|---|---|---|---|---|---|---|
| stage1 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd32_m4_fp32_perhead_hybrid-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt` | `03833215dfe3100e82422eba041a933efecfa873ca77e951a7edcc8962e476e6` | 8,574,050 B | 2,130,496 | {'torch.float32': 80} | 4.02 | 1100 | {'distill/eval/loss': 3.0541732609272003} |

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
| 0 | 0.698 | 0.100 | 0.322 | 0.965 | 0.0% | 0.0% | 2.8e-22 | 2.8e-42 | 1.6e-84 | 0.024 | -0.12, -0.12, -0.12, -0.12 | 0.090 / 0.69 | 0.086 / 0.65 | 2.00 |
| 1 | 0.999 | 0.008 | 0.820 | 1.000 | 0.0% | 96.9% | 9.4e-01 | 9.4e-01 | 7.8e-01 | 0.023 | 0.05, 0.05, 0.05, 0.05 | 0.079 / 0.63 | 0.080 / 0.53 | 3.61 |
| 2 | 1.000 | 0.001 | 0.975 | 1.000 | 0.0% | 97.8% | 9.9e-01 | 9.9e-01 | 9.5e-01 | 0.011 | 0.16, 0.16, 0.16, 0.16 | 0.076 / 0.60 | 0.076 / 0.47 | 3.85 |
| 3 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.010 | 0.08, 0.08, 0.08, 0.08 | 0.092 / 0.63 | 0.085 / 0.54 | 2.76 |
| 4 | 1.000 | 0.001 | 0.992 | 1.000 | 0.0% | 85.4% | 7.9e-01 | 6.8e-01 | 5.0e-01 | 0.024 | 0.01, 0.01, 0.01, 0.01 | 0.094 / 0.59 | 0.088 / 0.55 | 2.39 |
| 5 | 0.997 | 0.005 | 0.957 | 1.000 | 0.0% | 37.4% | 2.6e-01 | 1.2e-01 | 2.8e-02 | 0.024 | 0.30, 0.30, 0.30, 0.30 | 0.109 / 0.66 | 0.100 / 0.53 | 2.09 |
| 6 | 1.000 | 0.000 | 0.998 | 1.000 | 0.0% | 98.5% | 9.2e-01 | 8.7e-01 | 8.0e-01 | 0.037 | 0.09, 0.09, 0.09, 0.09 | 0.107 / 0.62 | 0.091 / 0.50 | 2.12 |
| 7 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 9.8e-01 | 9.7e-01 | 9.6e-01 | 0.068 | 0.25, 0.25, 0.25, 0.25 | 0.104 / 0.62 | 0.094 / 0.53 | 2.41 |
| 8 | 0.999 | 0.001 | 0.992 | 1.000 | 0.0% | 89.4% | 8.5e-01 | 7.5e-01 | 6.0e-01 | 0.027 | 0.03, 0.03, 0.03, 0.03 | 0.114 / 0.62 | 0.100 / 0.59 | 2.04 |
| 9 | 0.995 | 0.008 | 0.873 | 0.999 | 0.0% | 3.5% | 7.0e-01 | 5.4e-01 | 2.9e-01 | 0.009 | 0.35, 0.35, 0.35, 0.35 | 0.108 / 0.73 | 0.102 / 0.62 | 2.51 |
| 10 | 1.000 | 0.000 | 0.999 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.071 | 0.06, 0.06, 0.06, 0.06 | 0.100 / 0.68 | 0.103 / 0.61 | 2.28 |
| 11 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.045 | 0.08, 0.08, 0.08, 0.08 | 0.101 / 0.58 | 0.097 / 0.60 | 2.32 |
| 12 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.037 | 0.10, 0.10, 0.10, 0.10 | 0.096 / 0.60 | 0.106 / 0.56 | 2.61 |
| 13 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.038 | 0.16, 0.16, 0.16, 0.16 | 0.091 / 0.58 | 0.094 / 0.52 | 2.48 |
| 14 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.067 | 0.18, 0.18, 0.18, 0.18 | 0.095 / 0.56 | 0.093 / 0.58 | 2.68 |
| 15 | 0.128 | 0.179 | 0.002 | 0.867 | 0.0% | 0.0% | 9.8e-206 | 0.0e+00 | 0.0e+00 | 0.042 | 0.62, 0.62, 0.62, 0.62 | 0.096 / 0.59 | 0.087 / 0.69 | 2.58 |
