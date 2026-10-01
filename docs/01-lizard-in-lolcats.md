# 1. Lizard in LoLCATs

**Goal:** Train Llama-3.2-1B with the Lizard attention from the thesis repository (`jku-thesis/lizard_attention.py`). Use the two-stage pipeline of LoLCATs: attention distillation first, then LoRA finetuning. Replace the old `run.sh` with a Makefile.

PR #1 (commit `c2dd4f0`) merged this work.

## Added and removed files

| File | Purpose |
|---|---|
| `src/model/linear_attention/lizard_attention.py` | The Lizard functions and layer, the LoLCATs wrapper, and a cache for generation |
| `src/model/convert_model.py` | Registers the attention type `lolcats_llama_lizard` (in `get_attention`) and its cache (in `get_attention_cache`) |
| `src/model/linear_attention/__init__.py` | Exports `LolcatsLizardAttention` and `LizardAttentionCache` |
| `configs/model/distill_llama3_2_1b_lizard_w128_fd128_m4.yaml` | Model config. Its `model:` section is the same as in the old `run.sh` config. Its `attention:` section selects Lizard. |
| `Makefile` | `make lolcats` runs the old `run.sh` command. `make lizard` runs the same configs with Lizard attention. |
| `run.sh` | Removed |

### The Lizard code

The functions `hedgehog`, `window_mask`, `gate_products`, `gla`, `awa` and the class `LizardAttention` came from jku-thesis without changes. A byte comparison showed that the two copies were identical at that time. The port does not include `from_llama`, because the wrapper builds the layer from the original attention module. Later, the NaN fix of PR #5 changed these functions ([document 5](05-nan-crash.md)).

Lizard adds five parameters to each layer. The teacher q/k/v/o projections stay as they are.

| Parameter | Shape | Function |
|---|---|---|
| `phi_q`, `phi_k` | Linear 64 → 128, shared by all heads | Feature maps: φ(x) = [softmax(xW) ⊕ softmax(−xW)], with 256 features |
| `W_gamma` | Linear 2048 → 1, no bias, initial value 0 | Gate: one gate value γ = sigmoid(W_γ x) for each token. At initialization, γ = 0.5. |
| `meta_tokens` | 4 scalars | Sink logits. They appear only in the denominator of the window softmax. |
| `alpha_blend` | 1 scalar, initial value 1 | α, the weight of the window branch: y = GLA + α · AWA |

Each layer has 18,437 Lizard parameters. The 16 layers have **294,992 in total**. This number is the same as the trainable parameter count that stage 1 reported (0.024% of the model).

### The LoLCATs wrapper (`LolcatsLizardAttention`)

- **Construction.** The wrapper is a subclass of `LizardAttention`. It takes the teacher `LlamaAttention` as `base_attn` and uses its q/k/v/o projection modules again. Thus only the five Lizard parameters are new. The wrapper moves the module to the device and dtype of the teacher weights.
- **Distillation mode (`train_attention=True`).** Under `no_grad`, the wrapper applies RoPE to q and k. It then calculates the teacher softmax attention as the target `y_true`.
  - It calculates the Lizard output `y_pred`.
  - It returns `((None, None), (y_pred, y_true))` in place of the attention weights. The LoLCATs distillation trainer reads these values.
  - It gives `y_true` to the next layer as its input (teacher forcing).
- **Student mode.** The wrapper calculates `gla(...) + alpha * awa(...)` in float32 or a more precise dtype (`upcast`). It then casts the result back to the model dtype. In training, the model dtype is bf16.
- **Generation.** The sample evaluations and the final evaluations of LoLCATs call `model.generate(use_cache=True)`. Without a cache that matches Lizard, the outputs of these calls are wrong.
  - For each layer, `LizardAttentionCache` keeps the state of the gated branch: `kv_state` with shape (b, h, f, d), and `k_state`.
  - It also keeps the keys and values of the last `window` tokens.
  - The wrapper processes the prompt in one dense pass (`lizard_recurrent`, the prefill). After the prefill, it uses the recurrent form for each new token.
- **RoPE and masks.** Lizard attention uses no RoPE and ignores padding masks. The thesis code does the same.

## Configuration

`distill_llama3_2_1b_lizard_w128_fd128_m4.yaml` contains these settings:

- `model:` `meta-llama/Llama-3.2-1B`, bfloat16, `rope_theta: 500000`, `attn_implementation: flash_attention_2`.
- `attention:` `attention_type: lolcats_llama_lizard`, `feature_dim: 128`, `window_size: 128`, `num_meta: 4`, `softmax_attentions: []`, `train_attention: true`, `remove_base_attn: true`. The empty `softmax_attentions` list makes Lizard replace every layer. The Lizard sizes come from `jku-thesis/config.py`.

An intermediate version set `attn_implementation: sdpa`. With that setting, the environment did not need flash-attn, because Lizard replaces every attention layer. When training moved into Docker, the config went back to `flash_attention_2` on request ([document 3](03-infrastructure.md)).

On the current `main`, `make lizard` runs this command:

```bash
python distill_llama.py --model_config distill_llama3_2_1b_lizard_w128_fd128_m4 \
  --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2_1b \
  --finetune_config finetune_lora_qkvo_alpaca_clean_1b \
  --no_init_eval --verbose --seed 0 --replicate 0 $(ARGS)
```

The distill config and the finetune config are the existing LoLCATs configs, with no changes. The new command does not have `--lk_zero_init` from the old command. That flag applies only to the learned Hedgehog kernel of the old config. PR #1 also gave `--eval_config eval_alpaca_clean`. PR #2 removed this flag and the final evaluation, and added `--no_init_eval`.

## Differences from the jku-thesis pipeline

| Aspect | jku-thesis `train.py` | This LoLCATs configuration |
|---|---|---|
| Stage 1 loss | Summed squared error for each layer | The LoLCATs loss: 1000 × mean squared error on the output of each layer before `o_proj` (`mse_factor: 1000`) |
| Cross-entropy distillation term | – | Must stay 0 (`xent_factor: 0`), because Lizard returns no attention weights |
| Trainable parameters in stage 2 | LoRA on q/k/v **and** the Lizard parameters | LoRA on q/k/v/o only. The Lizard parameters are frozen. |
| Precision | float32 | bf16 model. The Lizard calculations upcast to float32. |
| Hyperparameters | The recipe of the paper ([document 9](09-paper-comparison.md)) | The LoLCATs configs: stage 1 learning rate 1e-2, stage 2 learning rate 1e-4, plateau scheduler |

The thesis code keeps the Lizard parameters trainable in stage 2. LoLCATs can do the same with a new line under `finetune:` in the finetune config: `trainable_weights: [phi_q, phi_k, W_gamma, meta_tokens, alpha_blend]`. PR #1 did not add this line, because the request was to keep the existing configs.

## Tests at that time (CPU, tiny random Llama)

No GPU was available. Thus these tests used a tiny Llama with random weights.

1. In float64, the new layer matched the jku-thesis layer and its `reference.lizard_loop` with an error less than 1e-10. A later version gave less than 1e-12.
2. In distillation mode, the target was equal to the attention output of the original model. Only the five Lizard parameters received gradients.
3. Generation with the cache gave the same tokens and logits as a full recalculation of the sequence. This test included prompts longer than the window, and chunks of more than one token on an existing cache.
4. A bf16 forward pass and backward pass gave only finite values.
5. A simulation of the full `make lizard` flow ran from start to end on the tiny model with synthetic data. The flow included the arguments, the configs, the attention swap, distillation, checkpoint save and load, LoRA finetuning and generation.

The tests found two bugs. Both bugs got a fix during development.

- A float64 mismatch of 8.7e-8 came from `.float()`, which changed float64 inputs to float32. The wrapper now uses `upcast`, which gives float32 or a more precise dtype.
- `generate(use_cache=False)` stopped with the error `'DynamicCache' object has no attribute 'kv_states'`. The forward pass now uses the recurrent path only when it receives a `LizardAttentionCache`.

[Document 8](08-verification.md) describes more complete checks on the merged code.
