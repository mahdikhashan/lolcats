# Stage comparison: where the gap starts

This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `results/window_rope_stage2/stages`.

```
date (UTC): 2026-10-10 18:09:12
lolcats commit: 84228c550b815e3e94c3a78dda6d41f4e9805cf4
harness commit: b281b0921b636bc36ad05c0b0b0763bd6dd43463
models: stage2
tasks: mmlu_subset piqa arc_easy
temperature: 1
HF_REPO: nanoman1/lolcats-lizard-llama-3.2-1b
stage 1 checkpoint: checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt
stage 2 checkpoint: checkpoints/distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.pt
python 3.11.16, torch 2.5.1 (CUDA 12.4), transformers 4.43.1, peft 0.9.0
NVIDIA A10, 23028 MiB, 580.126.09
Tesla V100-PCIE-16GB, 16384 MiB, 580.126.09
Tesla P40, 24576 MiB, 580.126.09
Tesla P40, 24576 MiB, 580.126.09
```

## Scores

Accuracy in %, ± the binomial SE. n is the number of questions.

| Task | C. Lizard after stage 2 |
|---|---|
| MMLU subset (5-shot, 5 questions per subject) | 25.6 ± 2.6 (n = 285) |
| PIQA (0-shot) | 73.1 ± 1.0 (n = 1838) |
| PIQA (0-shot), normalized | 72.5 ± 1.0 (n = 1838) |
| ARC-Easy (0-shot) | 63.8 ± 1.0 (n = 2376) |
| ARC-Easy (0-shot), normalized | 56.9 ± 1.0 (n = 2376) |

## Answer letters: MMLU subset (5-shot, 5 questions per subject)

Share of each predicted letter, in %. On MMLU, the Lizard model of Run 2 selected "A" for almost every question (section 4 of the gap analysis).

| Model | A | B | C | D | Accuracy | Accuracy if always "A" |
|---|---|---|---|---|---|---|
| C. Lizard after stage 2 | 41.4 | 37.5 | 20.7 | 0.4 | 25.6 | 24.2 |

Right answers: A 24.2%, B 24.9%, C 25.3%, D 25.6%

Choice probabilities, as means over the questions. Mass: the probability of " A" to " D" together. Confidence: the largest of the four probabilities after normalization over the four letters. Entropy: over the four letters, in bits (maximum 2).

| Model | Temperature | Tokens per letter | Mass | Confidence | Entropy (bits) |
|---|---|---|---|---|---|
| C. Lizard after stage 2 | 1 | 1 | 0.980 | 0.348 | 1.908 |

## Checkpoints

Sections 3 and 9 of the gap analysis.

| Stage | File | SHA-256 | Size | Parameters | Dtypes | Bytes per parameter | Stored step | Stored losses |
|---|---|---|---|---|---|---|---|---|
| stage1 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0_distill.pt` | `4aa0a3d1a64a61fa9b648af28da253b78e2bcceb3e2963d85864889bc9f7472b` | 638,846 B | 294,992 | {'torch.bfloat16': 80} | 2.17 | 1100 | {'distill/eval/loss': 1.22900390625} |
| stage2 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.pt` | `c828d6b3c790e83d1aa21b6547e0a4842c075b7b0c80423a460f5f8e5e4e475d` | 3,491,070 B | 1,703,936 | {'torch.bfloat16': 128} | 2.05 | 1100 | {'eval/loss': 1.2352412939071655} |

Trainable keys: for each evaluation, each expected trainable parameter of the model must be in the checkpoint and hold its value after the load.

| Evaluation | Checkpoint | Expected | Missing | Unexpected | Not loaded |
|---|---|---|---|---|---|
| stage2/mmlu_subset | stage1 | 80 | 0 | 0 | 0 |
| stage2/mmlu_subset | stage2 | 128 | 0 | 0 | 0 |
| stage2/piqa | stage1 | 80 | 0 | 0 | 0 |
| stage2/piqa | stage2 | 128 | 0 | 0 | 0 |
| stage2/arc_easy | stage1 | 80 | 0 | 0 | 0 |
| stage2/arc_easy | stage2 | 128 | 0 | 0 | 0 |

Result: every expected trainable parameter loaded from its checkpoint.

## Validation loss during training

From the training results CSVs in `training/`. The trainer saves the checkpoint at the best step, so the stored step of the checkpoint should equal the best step.

| Stage | File | Evaluations | First: step, loss | Best: step, loss | Last: step, loss | Stored step of the checkpoint |
|---|---|---|---|---|---|---|
| stage2 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.csv` | 11 | 100, 1.3432 | 1100, 1.2352 | 1100, 1.2352 | 1100 |

## Lizard parameters and gates: C. Lizard after stage 2

Gate values on one 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens). Kept after w: the weight that the gated branch keeps on a token w positions before the last token. Initial values: gamma = 0.5, alpha = 1, feature-map weight RMS 0.02.

| Layer | Gamma mean | Std | Min | Max | < 1e-3 | > 0.999 | Kept after 128 | Kept after 256 | Kept after 512 | Alpha | Sink logits | φq RMS / max | φk RMS / max | ‖W_γ‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.999 | 0.001 | 0.983 | 1.000 | 0.0% | 65.6% | 8.7e-01 | 7.5e-01 | 5.6e-01 | 0.660 | 4.25, 4.25, 4.25, 4.25 | 0.140 / 0.75 | 0.144 / 1.12 | 5.18 |
| 1 | 1.000 | 0.001 | 0.960 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.416 | 0.50, 0.50, 0.50, 0.50 | 0.186 / 1.20 | 0.156 / 1.02 | 4.23 |
| 2 | 1.000 | 0.004 | 0.808 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.293 | 0.55, 0.55, 0.55, 0.55 | 0.193 / 1.59 | 0.180 / 0.96 | 3.87 |
| 3 | 1.000 | 0.003 | 0.864 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.508 | 1.00, 1.00, 1.00, 1.00 | 0.198 / 1.24 | 0.186 / 2.00 | 3.85 |
| 4 | 1.000 | 0.003 | 0.869 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.570 | 0.52, 0.52, 0.52, 0.52 | 0.200 / 1.05 | 0.197 / 1.07 | 3.91 |
| 5 | 1.000 | 0.003 | 0.844 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.711 | 1.46, 1.46, 1.46, 1.46 | 0.197 / 1.04 | 0.188 / 0.97 | 3.91 |
| 6 | 1.000 | 0.003 | 0.856 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.609 | 0.81, 0.81, 0.81, 0.81 | 0.196 / 1.03 | 0.190 / 0.84 | 3.99 |
| 7 | 1.000 | 0.005 | 0.794 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.590 | 1.04, 1.04, 1.04, 1.04 | 0.190 / 0.96 | 0.201 / 1.41 | 4.29 |
| 8 | 1.000 | 0.004 | 0.820 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.660 | 0.50, 0.50, 0.50, 0.50 | 0.194 / 1.07 | 0.199 / 1.00 | 4.09 |
| 9 | 1.000 | 0.006 | 0.736 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.688 | 0.64, 0.64, 0.64, 0.64 | 0.191 / 1.17 | 0.193 / 1.09 | 4.10 |
| 10 | 1.000 | 0.007 | 0.672 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.660 | 1.00, 1.00, 1.00, 1.00 | 0.215 / 1.41 | 0.208 / 1.28 | 3.96 |
| 11 | 1.000 | 0.007 | 0.679 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.598 | 0.25, 0.25, 0.25, 0.25 | 0.196 / 1.23 | 0.200 / 1.91 | 3.90 |
| 12 | 1.000 | 0.008 | 0.650 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.660 | 0.93, 0.93, 0.93, 0.93 | 0.200 / 2.42 | 0.191 / 1.95 | 4.05 |
| 13 | 1.000 | 0.009 | 0.615 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.527 | 0.50, 0.50, 0.50, 0.50 | 0.195 / 1.26 | 0.199 / 0.97 | 4.00 |
| 14 | 1.000 | 0.010 | 0.535 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.562 | 1.00, 1.00, 1.00, 1.00 | 0.196 / 0.91 | 0.196 / 0.93 | 3.79 |
| 15 | 1.000 | 0.014 | 0.358 | 1.000 | 0.0% | 100.0% | 1.0e+00 | 1.0e+00 | 1.0e+00 | 0.719 | 1.23, 1.23, 1.23, 1.23 | 0.191 / 2.00 | 0.191 / 1.51 | 3.68 |
