# Hyperparameters

This document lists hyperparameters of the Lizard model in this project and compares them with reference values. Each row gives the source of its value.

## Model dimensions of Llama-3.2-1B

| Quantity | Value | Source |
|---|---|---|
| Hidden size | 2048 | Model config of `meta-llama/Llama-3.2-1B` |
| Attention heads | 32 | Model config of `meta-llama/Llama-3.2-1B` |
| Head dimension | 64 (2048 / 32) | `head_dim = hidden // heads` in `src/model/linear_attention/lizard_attention.py` |

## Feature dimension of the feature maps

`feature_dim` is the output size of the feature-map projections `phi_q` and `phi_k`. Each projection is a linear layer from the head dimension to `feature_dim`, shared by all heads. The Hedgehog feature map then concatenates softmax(xW) and softmax(−xW). Thus each query and each key gets 2 × `feature_dim` features.

| Setting | Feature dimension | Ratio to the head dimension | Features per query or key | Source |
|---|---|---|---|---|
| LoLCATs rule, applied to Llama-3.2-1B | 32 (0.5 × 64) | 0.5 | 64 | LoLCATs paper |
| LoLCATs config for Llama-3-8B | 64 | 0.5 (head dimension 128) | 128 | `configs/model/distill_llama3_8b_lk_smd_wtk64_fd64_w01.yaml` |
| Lizard paper | 128 | 1 for its 8B models (head dimension 128) | 256 | Table 13 of the paper |
| **This project** | **128** | **2** | **256** | `feature_dim: 128` in `configs/model/distill_llama3_2_1b_lizard_w128_fd128_m4.yaml` |
| jku-thesis | 128 | 2 | 256 | `LIZARD = {..., "feature_dim": 128}` in `config.py` |

**The feature dimension of this project is 128.** It is 4× the value of the LoLCATs rule for Llama-3.2-1B (32), and 2× the head dimension.

Notes:

- **Origin of the value:** The value comes from Table 13 of the Lizard paper, through `jku-thesis/config.py`. Table 13 gives one value (128) for all experiments. It does not say if the 1B models of the paper used the same value.
- **Ratio:** For the 8B models of the paper, 128 is 1× the head dimension. The same ratio gives 64 for Llama-3.2-1B.
- **Parameter count:** `phi_q` and `phi_k` each have 64 × 128 = 8,192 weights. Thus each layer has 18,437 Lizard parameters ([document 1](01-lizard-in-lolcats.md)). With a feature dimension of 32, each projection would have 2,048 weights, and each layer 6,149 Lizard parameters.
- **The old LoLCATs config for Llama-3.2-1B:** `configs/model/distill_llama3_1_1b_lk_smd_wtk64_fd64_w01.yaml` (`make lolcats`) also sets `feature_dim: 128`, although its name contains "fd64". Its comment says "LoLCATs default is 64".
- **Gap analysis:** Section 12 of the [gap analysis](11-gap-analysis.md) lists the feature dimension as equal to the paper. This is true for the value (128). It is possibly not true for the ratio to the head dimension. The paper possibly used 128 only because its 8B models have a head dimension of 128.
