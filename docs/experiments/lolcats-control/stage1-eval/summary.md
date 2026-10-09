# Stage comparison: where the gap starts

This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `results/stages/20261009-194950-lolcats-control`.

```
date (UTC): 2026-10-09 17:50:10
lolcats commit: 8bd761425793adc9095585c51622cfc9871ed8b8
harness commit: b281b0921b636bc36ad05c0b0b0763bd6dd43463
models: stage1
tasks: mmlu_subset piqa arc_easy
temperature: 1
HF_REPO: nanoman1/lolcats-lizard-llama-3.2-1b
stage 1 checkpoint: checkpoints/distill_llama3_1_1b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_1_1b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-lzi=1_distill.pt
stage 2 checkpoint: checkpoints/distill_llama3_1_1b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_1_1b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.pt
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
| MMLU subset (5-shot, 5 questions per subject) | 26.0 ± 2.6 (n = 285) |
| PIQA (0-shot) | 73.5 ± 1.0 (n = 1838) |
| PIQA (0-shot), normalized | 73.5 ± 1.0 (n = 1838) |
| ARC-Easy (0-shot) | 62.8 ± 1.0 (n = 2376) |
| ARC-Easy (0-shot), normalized | 58.0 ± 1.0 (n = 2376) |

## Answer letters: MMLU subset (5-shot, 5 questions per subject)

Share of each predicted letter, in %. On MMLU, the Lizard model of Run 2 selected "A" for almost every question (section 4 of the gap analysis).

| Model | A | B | C | D | Accuracy | Accuracy if always "A" |
|---|---|---|---|---|---|---|
| B. Lizard after stage 1 (no LoRA) | 19.3 | 14.7 | 42.8 | 23.2 | 26.0 | 24.2 |

Right answers: A 24.2%, B 24.9%, C 25.3%, D 25.6%

Choice probabilities, as means over the questions. Mass: the probability of " A" to " D" together. Confidence: the largest of the four probabilities after normalization over the four letters. Entropy: over the four letters, in bits (maximum 2).

| Model | Temperature | Tokens per letter | Mass | Confidence | Entropy (bits) |
|---|---|---|---|---|---|
| B. Lizard after stage 1 (no LoRA) | 1 | 1 | 0.962 | 0.450 | 1.746 |

## Checkpoints

Sections 3 and 9 of the gap analysis.

| Stage | File | SHA-256 | Size | Parameters | Dtypes | Bytes per parameter | Stored step | Stored losses |
|---|---|---|---|---|---|---|---|---|
| stage1 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_1_1b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-lzi=1_distill.pt` | `200d27aa5393c1465c3f203639ad24ea2a791c0a54ee1f2a295776bcc0ee3c08` | 16,808,818 B | 8,389,120 | {'torch.bfloat16': 48} | 2.00 | 1100 | {'distill/eval/loss': 0.4442138671875} |

Trainable keys: for each evaluation, each expected trainable parameter of the model must be in the checkpoint and hold its value after the load.

| Evaluation | Checkpoint | Expected | Missing | Unexpected | Not loaded |
|---|---|---|---|---|---|
| stage1/mmlu_subset | stage1 | 48 | 0 | 0 | 0 |
| stage1/piqa | stage1 | 48 | 0 | 0 | 0 |
| stage1/arc_easy | stage1 | 48 | 0 | 0 | 0 |

Result: every expected trainable parameter loaded from its checkpoint.

## Lizard parameters and gates: B. Lizard after stage 1 (no LoRA)

Gate values on one 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens). Kept after w: the weight that the gated branch keeps on a token w positions before the last token. Initial values: gamma = 0.5, alpha = 1, feature-map weight RMS 0.02.

| Layer | Gamma mean | Std | Min | Max | < 1e-3 | > 0.999 | Kept after 128 | Kept after 256 | Kept after 512 | Alpha | Sink logits | φq RMS / max | φk RMS / max | ‖W_γ‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
