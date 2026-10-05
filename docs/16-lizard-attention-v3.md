# 16. Lizard attention v3

**Status:** v3 is in the repository without changes, connected to the model code, with a config and 35 CPU checks. No GPU run yet.

v3 is a third Lizard attention file. The user supplied the code (generated with ChatGPT). Its file is `src/model/linear_attention/lizard_attention_v3.py`, and its attention type is `lolcats_llama_lizard_v3`. This change keeps every line of the supplied code.

## What v3 is

v3 has the math of v2 ([document 13](13-lizard-attention-v2.md)), with these differences:

| Part | v2 | v3 |
|---|---|---|
| α by default | Trained (`train_alpha: true`) | Constant at `alpha_init` = 1.0, a buffer (`train_alpha: false`). The paper lists only φ, W_γ and t as learnable. |
| Calculation of the gated weights | Dense weights | `gla_impl`: `direct` (dense weights, as in v2) or `reparam` (the Section 4 form, only with `feature_activation: exp`) |
| `gla_norm` values | `row`, `none`, `joint`, `hybrid` | The same |
| Gate | Sigmoid in float32 | Sigmoid in the dtype of the model |
| `feature_dim` | Features in each Hedgehog half | The same. The docstring states it. |

**Consequence**. With its defaults, v3 calculates the same function as v2 with `train_alpha: false` and `alpha_init: 1.0`. At the start values, it also gives the outputs of v1 (check 1). In training, the only difference from config 1 is that α stays at 1.

## Connection to the model code

| File | Change |
|---|---|
| `src/model/linear_attention/lizard_attention_v3.py` | New: the supplied code |
| `src/model/linear_attention/__init__.py` | Imports `LolcatsLizardAttentionV3` |
| `src/model/convert_model.py` | The attention type `lolcats_llama_lizard_v3` gives v3 |
| `configs/model/distill_llama3_2_1b_lizard_v3_w128_fd32_m4_fp32.yaml` | New config: the v3 defaults, feature dimension 32, window 128, float32 |
| `tests/test_lizard_attention_v3.py` | New: 35 CPU checks |

These parts work without change:

- **Cache**: the name contains `llama_lizard`, so `get_attention_cache` gives `LizardAttentionCache`.
- **Scripts**: `compare_stages.py` reads the trainable parameters with `named_parameters`. Thus it does not expect α (a buffer) in the checkpoint. `layer_mse.py` uses the outputs of each layer.

## Experiment E3: α constant at 1

**Question**. Does a constant α = 1 change stage 1 against config 1, where α is trained? This is the reading question of α in document 13 (table "Options not in the plan").

- **The only difference from config 1**: α stays at 1. In config 1, α went to 0.47–0.64.
- **Expectation** ([document 14](14-liger-gla.md), P8a): with α = 1, the total weight of row $i$ is $1 + \rho_i$, larger than in config 1. Thus the stage 1 loss is probably higher than 3.9764. This expectation is not checked.

**Stage 1 on HF Jobs (H200)**, after the merge and a new Docker image:

```bash
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=nanoman1/lolcats-lizard-llama-3.2-1b HF_FLAVOR=h200 HF_TIMEOUT=4h \
  ARGS="--model_config distill_llama3_2_1b_lizard_v3_w128_fd32_m4_fp32 --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-3_paper_noclip_1b --no_finetune"
```

- **Time**: approximately 2.5 hours, as config 1 and C1.
- **GPU memory**: probably approximately 29 GB, as C1. Both use the `row` form.
- **Evaluation on the A10**: the command of "Experiment C1" in document 13, with `MODEL_CONFIG=distill_llama3_2_1b_lizard_v3_w128_fd32_m4_fp32`. Then `layer_mse.py` with the same config.

## Checks

| Check | Result |
|---|---|
| The defaults against v1 with the same weights (float64) | Relative error below 1e-12. α is not a parameter. |
| 15 option sets against the loop form of [math formulas](math-formula.md) | All pass, relative error below 1e-10 |
| Decode (`LizardAttentionCache`) against the parallel form, the same 15 sets | All pass, relative error below 1e-10 |
| Stage 1 parameters of the defaults | φq, φk, W_γ and the sinks get a gradient. α is not trained. |
| Model conversion with the v3 config, teacher mode and cached decode | Pass |
| Tiny stage 1 run (`distill_llama.main()`, 3 layers) | v3 layers, 12 trainable tensors (φq, φk, W_γ, sinks). The validation loss fell from 6811 to 6598 in 25 steps. The checkpoint has the expected name and no α. |
| `check_checkpoint` of `compare_stages.py` on that checkpoint | 12 expected keys, 0 missing, 0 unexpected, all loaded |
| The 60 checks of v2 | Still pass |

The tiny run stops at the end with the `token_type_ids` error of the tiny test tokenizer, as with v1 and v2. Command for the 35 checks: `python -m pytest -q tests/test_lizard_attention_v3.py`.

## Notes on the code (not changed)

1. **`gla_impl: reparam` overflows in float32 for long sequences**. At the start gate 0.5, the outputs contain inf or NaN from 260 tokens. At 250 tokens, the outputs agree with `direct` (relative error 1.3e-6). The training chunks have 2,048 tokens. The default `direct` is not affected.
2. **The gate is calculated in the dtype of the model**. v2 calculates it in float32. With a bfloat16 model, gates near 1 are rounded. The E3 config is float32, so it has no effect there.

Both notes need a change of the supplied code. They stay as they are until the user decides.
