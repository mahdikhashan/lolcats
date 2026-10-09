# Stage comparison: where the gap starts

This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `results/stages/20261009-202521-lolcats-control-layers`.

```
date (UTC): 2026-10-09 18:25:33
lolcats commit: e5e0e9d52b06af3dca69630abce53f0247a2595b
harness commit: b281b0921b636bc36ad05c0b0b0763bd6dd43463
models: stage1
tasks: layers
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

## Checkpoints

Sections 3 and 9 of the gap analysis.

| Stage | File | SHA-256 | Size | Parameters | Dtypes | Bytes per parameter | Stored step | Stored losses |
|---|---|---|---|---|---|---|---|---|
| stage1 | `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_1_1b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0-lzi=1_distill.pt` | `200d27aa5393c1465c3f203639ad24ea2a791c0a54ee1f2a295776bcc0ee3c08` | 16,808,818 B | 8,389,120 | {'torch.bfloat16': 48} | 2.00 | 1100 | {'distill/eval/loss': 0.4442138671875} |

Trainable keys: for each evaluation, each expected trainable parameter of the model must be in the checkpoint and hold its value after the load.

| Evaluation | Checkpoint | Expected | Missing | Unexpected | Not loaded |
|---|---|---|---|---|---|
| stage1/layers | stage1 | 48 | 0 | 0 | 0 |

Result: every expected trainable parameter loaded from its checkpoint.

## LoLCATs parameters and window share: B. Lizard after stage 1 (no LoRA)

Window share on one 5-shot prompt of `hendrycksTest-high_school_us_history` (2048 tokens): the share of the attention weight of a query on the keys inside its window, as the mean over the heads. Far queries: the queries with keys outside the window, where the linear branch acts. Window factor: sigmoid(a_h) of each head, the weight of the window terms before the normalization. Change from identity: ‖W − I‖ / ‖I‖ of the feature-map weights. Initial values: window factor 0.1, feature maps identity (--lk_zero_init).

| Layer | Window factor mean | Min | Max | Window share, all queries | Far queries | Last query | φq RMS / max | φk RMS / max | φq change from identity | φk change from identity |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.116 | 0.102 | 0.134 | 0.818 | 0.792 | 0.757 | 0.146 / 1.80 | 0.138 / 1.46 | 1.325 | 1.191 |
| 1 | 0.092 | 0.066 | 0.110 | 0.439 | 0.359 | 0.336 | 0.165 / 3.31 | 0.169 / 2.08 | 1.628 | 1.651 |
| 2 | 0.096 | 0.080 | 0.107 | 0.399 | 0.313 | 0.335 | 0.185 / 3.72 | 0.159 / 1.86 | 1.906 | 1.525 |
| 3 | 0.101 | 0.051 | 0.128 | 0.516 | 0.447 | 0.336 | 0.240 / 3.48 | 0.151 / 2.03 | 2.576 | 1.432 |
| 4 | 0.107 | 0.070 | 0.124 | 0.623 | 0.569 | 0.509 | 0.240 / 4.12 | 0.157 / 2.03 | 2.585 | 1.497 |
| 5 | 0.109 | 0.060 | 0.125 | 0.755 | 0.720 | 0.692 | 0.259 / 4.28 | 0.188 / 4.09 | 2.811 | 1.896 |
| 6 | 0.113 | 0.102 | 0.129 | 0.782 | 0.751 | 0.756 | 0.232 / 3.06 | 0.184 / 3.14 | 2.467 | 1.850 |
| 7 | 0.111 | 0.085 | 0.129 | 0.802 | 0.774 | 0.644 | 0.249 / 4.03 | 0.169 / 2.09 | 2.696 | 1.695 |
| 8 | 0.111 | 0.078 | 0.129 | 0.772 | 0.739 | 0.777 | 0.236 / 4.00 | 0.176 / 4.00 | 2.534 | 1.794 |
| 9 | 0.106 | 0.071 | 0.118 | 0.768 | 0.735 | 0.717 | 0.226 / 3.55 | 0.166 / 3.38 | 2.416 | 1.620 |
| 10 | 0.100 | 0.050 | 0.130 | 0.687 | 0.642 | 0.583 | 0.227 / 3.50 | 0.254 / 3.52 | 2.427 | 2.723 |
| 11 | 0.104 | 0.084 | 0.118 | 0.609 | 0.553 | 0.549 | 0.257 / 3.00 | 0.170 / 2.03 | 2.770 | 1.691 |
| 12 | 0.095 | 0.059 | 0.110 | 0.520 | 0.451 | 0.440 | 0.230 / 4.22 | 0.178 / 2.86 | 2.451 | 1.791 |
| 13 | 0.103 | 0.076 | 0.135 | 0.521 | 0.453 | 0.429 | 0.218 / 4.06 | 0.170 / 2.55 | 2.327 | 1.702 |
| 14 | 0.101 | 0.075 | 0.129 | 0.558 | 0.495 | 0.613 | 0.228 / 4.38 | 0.173 / 2.08 | 2.436 | 1.762 |
| 15 | 0.097 | 0.067 | 0.130 | 0.562 | 0.499 | 0.490 | 0.228 / 3.16 | 0.176 / 2.00 | 2.429 | 1.767 |
