# 1. Lizard in LoLCATs

**Goal:** train Llama-3.2-1B with the Lizard attention from the thesis repository
(`jku-thesis/lizard_attention.py`) using the LoLCATs two-stage pipeline (attention distillation,
then LoRA finetuning), and replace the old `run.sh` with a Makefile.

Merged in PR #1 (commit `c2dd4f0`).

## What was added

| File | Purpose |
|---|---|
| `src/model/linear_attention/lizard_attention.py` | The Lizard functions and layer, plus the LoLCATs wrapper and a generation cache |
| `src/model/convert_model.py` | Registers the attention type `lolcats_llama_lizard` (in `get_attention`) and its cache (in `get_attention_cache`) |
| `src/model/linear_attention/__init__.py` | Exports `LolcatsLizardAttention` and `LizardAttentionCache` |
| `configs/model/distill_llama3_2_1b_lizard_w128_fd128_m4.yaml` | Model config: the same `model:` section as the old `run.sh` config, with a Lizard `attention:` section |
| `Makefile` | `make lolcats` (the old `run.sh` command) and `make lizard` (same configs, Lizard attention) |
| `run.sh` | Removed |

### The Lizard code

The functions `hedgehog`, `window_mask`, `gate_products`, `gla`, `awa` and the class
`LizardAttention` were copied unchanged from jku-thesis; a byte comparison confirmed they were
identical at the time. `from_llama` was not copied, because the wrapper builds the layer from the
original attention module itself. (The functions were changed later by the NaN fix in PR #5; see
[document 5](05-nan-crash.md).)

Per layer, Lizard adds five parameters on top of the teacher's q/k/v/o projections:

| Parameter | Shape | Role |
|---|---|---|
| `phi_q`, `phi_k` | Linear 64 → 128 (shared by all heads) | Hedgehog feature maps: φ(x) = [softmax(xW) ⊕ softmax(−xW)], 256 features |
| `W_gamma` | Linear 2048 → 1, no bias, zero init | Scalar gate per token: γ = sigmoid(W_γ x), so γ = 0.5 at initialization |
| `meta_tokens` | 4 scalars | Sink logits that appear only in the denominator of the window softmax |
| `alpha_blend` | 1 scalar, init 1 | Weight of the window branch: y = GLA + α · AWA |

That is 18,437 parameters per layer, **294,992 in total** over 16 layers, which matches the
trainable parameter count reported by the stage 1 run (0.024% of the model).

### The LoLCATs wrapper (`LolcatsLizardAttention`)

- **Construction.** It subclasses `LizardAttention`, takes the teacher's `LlamaAttention` as
  `base_attn`, and reuses its q/k/v/o projection modules, so only the five Lizard parameters are
  new. The module is moved to the teacher weights' device and dtype.
- **Distillation mode (`train_attention=True`).** Under `no_grad`, it applies RoPE to q and k and
  computes the teacher's softmax attention as the target `y_true`. It computes Lizard's output
  `y_pred`, returns `((None, None), (y_pred, y_true))` as the attention weights for the LoLCATs
  distillation trainer, and passes `y_true` on to the next layer (teacher forcing).
- **Student mode.** It computes `gla(...) + alpha * awa(...)` in at least float32 (`upcast`), then
  casts back to the model dtype (bf16 in training).
- **Generation.** LoLCATs' sample and final evals call `model.generate(use_cache=True)`, which
  would give wrong outputs without a proper cache. `LizardAttentionCache` stores, per layer, the
  gated state of the linear branch (`kv_state` of shape (b, h, f, d) and `k_state`) and the keys
  and values of the last `window` tokens. The prompt is processed in one dense pass
  (`lizard_recurrent`, prefill); each new token then uses the recurrent form.
- **RoPE and masks.** As in the thesis code, Lizard uses no RoPE and ignores padding masks.

## Configuration

`distill_llama3_2_1b_lizard_w128_fd128_m4.yaml`:

- `model:` `meta-llama/Llama-3.2-1B`, bfloat16, `rope_theta: 500000`, `attn_implementation: flash_attention_2`.
- `attention:` `attention_type: lolcats_llama_lizard`, `feature_dim: 128`, `window_size: 128`,
  `num_meta: 4`, `softmax_attentions: []` (every layer is replaced), `train_attention: true`,
  `remove_base_attn: true`. The Lizard sizes come from `jku-thesis/config.py`.

(An intermediate version set `attn_implementation: sdpa` to avoid installing flash-attn, since
every attention layer is replaced anyway. It was reverted to `flash_attention_2` when training
was dockerized, as requested; see [document 3](03-infrastructure.md).)

`make lizard` runs (current `main`):

```bash
python distill_llama.py --model_config distill_llama3_2_1b_lizard_w128_fd128_m4 \
  --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2_1b \
  --finetune_config finetune_lora_qkvo_alpaca_clean_1b \
  --no_init_eval --verbose --seed 0 --replicate 0 $(ARGS)
```

The distill and finetune configs are the existing LoLCATs ones, unchanged. `--lk_zero_init` from the
old command was dropped, because it only applies to the Hedgehog learned kernel of the old config.
(PR #1 also passed `--eval_config eval_alpaca_clean`; PR #2 removed it together with the final
evaluation and added `--no_init_eval`.)

## Differences from the jku-thesis pipeline

| Aspect | jku-thesis `train.py` | This LoLCATs setup |
|---|---|---|
| Stage 1 loss | Per-layer summed squared error | LoLCATs: 1000 × mean-squared error on each layer's output before `o_proj` (`mse_factor: 1000`) |
| Cross-entropy distillation term | – | Must stay 0 (`xent_factor: 0`), because Lizard returns no attention weights |
| Stage 2 trainable parameters | LoRA on q/k/v **and** the Lizard parameters | LoRA on q/k/v/o only; the Lizard parameters are frozen |
| Precision | float32 | bf16 model; Lizard math upcast to float32 |
| Hyperparameters | The paper's recipe (see [document 9](09-paper-comparison.md)) | The LoLCATs configs (stage 1 lr 1e-2, stage 2 lr 1e-4, plateau scheduler) |

Keeping the Lizard parameters trainable in stage 2, as the thesis does, is possible by adding
`trainable_weights: [phi_q, phi_k, W_gamma, meta_tokens, alpha_blend]` under `finetune:` in the
finetune config. It was left out because the request was to keep the existing configs.

## Tests at the time (CPU, tiny random Llama)

No GPU was available, so these used a tiny randomly initialized Llama:

1. In float64, the new layer matched the jku-thesis layer and its `reference.lizard_loop` to below
   1e-10 (later below 1e-12).
2. In distillation mode, the target equalled the original model's attention output, and only the
   five Lizard parameters received gradients.
3. Generating with the cache gave the same tokens and logits as recomputing the full sequence,
   including prompts longer than the window and multi-token chunks on an existing cache.
4. A bf16 forward and backward pass stayed finite.
5. A simulation of the whole `make lizard` flow (arguments, configs, attention swap, distillation,
   checkpoint save and reload, LoRA finetuning, generation) ran end to end on the tiny model with
   synthetic data.

Two bugs were found and fixed during development:

- A float64 mismatch of 8.7e-8 came from `.float()` downcasting float64 inputs. The wrapper now
  uses `upcast`, which promotes to *at least* float32.
- `generate(use_cache=False)` crashed with `'DynamicCache' object has no attribute 'kv_states'`.
  The forward now uses the recurrent path only when it gets a `LizardAttentionCache`.

These tests were extended later with a more complete verification on the merged code; see
[document 8](08-verification.md).
