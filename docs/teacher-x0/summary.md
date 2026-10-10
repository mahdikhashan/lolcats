# Stage comparison: where the gap starts

This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `results/teacher_x0`.

```
date (UTC): 2026-10-10 19:04:52
lolcats commit: c132d2901051cb7fbef7b32bae8a37fb17ccc2f2
harness commit: b281b0921b636bc36ad05c0b0b0763bd6dd43463
models: teacher
tasks: piqa arc_easy
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

| Task | A. Teacher |
|---|---|
| PIQA (0-shot) | 74.4 ± 1.0 (n = 1838) |
| PIQA (0-shot), normalized | 74.5 ± 1.0 (n = 1838) |
| ARC-Easy (0-shot) | 65.3 ± 1.0 (n = 2376) |
| ARC-Easy (0-shot), normalized | 60.3 ± 1.0 (n = 2376) |

## Validation loss during training

From the training results CSVs in `training/`. The trainer saves the checkpoint at the best step, so the stored step of the checkpoint should equal the best step.

| Stage | File | Evaluations | First: step, loss | Best: step, loss | Last: step, loss | Stored step of the checkpoint |
|---|---|---|---|---|---|---|
| stage2 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-se=0-re=0_ft.csv` | 11 | 100, 3.5160 | 1100, 2.2520 | 1100, 2.2520 | – |
