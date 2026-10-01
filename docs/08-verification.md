# 8. Checks of the attention layer

**Question:** Does a bug in the attention layer cause the poor evaluation results, or does a different problem cause them? This document checks the code that trained and evaluated the model. That code is `main` at that time, with the NaN fix from PR #5.

All checks ran on CPU with PyTorch 2.0.1, transformers 4.43.1 (the version that lolcats pins) and Python 3.11.

## Level 1: the calculations match the reference formulas

jku-thesis has a test suite (`test_lizard.py`). It compares the dense implementation of jku-thesis with slow loop and recurrent versions of every formula (`reference.py`). The tests use float64 and require a relative error less than 1e-10. A small shim module with the name `lizard_attention.py` made these tests run against the **lolcats** functions:

```python
# Shim: run the jku-thesis tests against the lolcats implementation
import os, sys
sys.path.insert(0, os.environ['LOLCATS_DIR'])
from src.model.linear_attention.lizard_attention import (  # noqa: F401
    hedgehog, window_mask, gate_products, gla, awa, LizardAttention)
```

```bash
mkdir lzcheck && cp jku-thesis/{test_lizard.py,reference.py,config.py,pytest.ini} lzcheck/
# save the shim above as lzcheck/lizard_attention.py
cd lzcheck && LOLCATS_DIR=/path/to/lolcats python -m pytest -q test_lizard.py \
  --deselect test_lizard.py::test_equals_llama_attention_in_the_softmax_limit \
  --deselect test_lizard.py::test_whole_model_logits_unchanged_after_swap
```

**Result: 26 passed, 2 deselected.** The tests cover these properties:

- The feature maps match their formula. Both halves are positive, and each half has a sum of 1.
- The gated branch matches the loop formula and the recurrent form. A gate of zero keeps only the current token. Constant values stay constant.
- The window branch with sinks matches the loop formula. Without sinks and with a window of unlimited size, it is equal to causal softmax. More than one sink gives the same result as one sink at their logsumexp.
- In the two branches, future tokens give no gradient. In the window branch, gradients come only from tokens inside the window.
- The gradients match finite differences. The full layer matches the loop formula. The layer is causal, and the items in a batch are independent. Every Lizard parameter receives a gradient.

The two deselected tests need a transformers version newer than 4.43 (`LlamaRotaryEmbedding(config=...)`). In this environment, the same failure occurs on the original jku-thesis code. Level 2 checks the same behavior that these two tests check.

## Level 2: the LoLCATs wrapper and the connection into the model

The Level 1 tests do not cover three parts: `LolcatsLizardAttention`, the conversion inside a real model, and the cached paths that evaluation and generation use. The script below checks these parts inside `LolcatsLlamaForCausalLM`. It uses a tiny Llama with random weights in float64, with these properties:

- grouped-query attention with 4 query heads and 2 key/value heads, as in Llama-3.2,
- a window of 8 tokens,
- sequences of 24 tokens.

```python
"""
Check lolcats' LolcatsLizardAttention inside LolcatsLlamaForCausalLM on a tiny random Llama (float64)
Run from the lolcats repo root
"""
import sys
sys.path.insert(0, '.')

import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig

from src.model.modeling_llama import LolcatsLlamaForCausalLM
from src.model.convert_model import convert_attention, toggle_attention
from src.model.linear_attention.lizard_attention import LizardAttention, LizardAttentionCache

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
WINDOW, LENGTH = 8, 24  # sequence longer than the window, so the decode cache has to slide

config = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2,  # GQA, as in Llama 3.2
                     max_position_embeddings=128, attn_implementation='eager', initializer_range=0.3)
model = LolcatsLlamaForCausalLM(config).eval()
ids = torch.randint(0, config.vocab_size, (2, LENGTH))


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


with torch.no_grad():
    teacher_logits = model(ids, use_cache=False).logits

    # 1. Teacher mode: the layer outputs softmax attention from its own copied projections
    attention_config = OmegaConf.create({'attention_type': 'lolcats_llama_lizard', 'window_size': WINDOW,
                                         'num_meta': 4, 'feature_dim': 16, 'softmax_attentions': []})
    model = convert_attention(model, attention_config, train_attention=True, remove_base_attn=True)
    for layer in model.model.layers:  # random Lizard params, so student mode is not trivial
        attn = layer.self_attn
        for p in (attn.phi_q.weight, attn.phi_k.weight, attn.W_gamma.weight, attn.meta_tokens):
            p.normal_(0, 0.5)
        attn.alpha_blend.fill_(0.7)
    swapped_logits = model(ids, use_cache=False).logits
    print(f'1. teacher-mode logits vs original model: rel err {rel_err(swapped_logits, teacher_logits):.1e}')

    # 2. Student mode: wrapper layer vs the reference LizardAttention class (verified in Level 1)
    toggle_attention(model, train=False)
    x = torch.randn(2, LENGTH, config.hidden_size)
    for i, layer in enumerate(model.model.layers):
        wrapper = layer.self_attn
        reference = LizardAttention(config, i, window=WINDOW, num_meta=4, feature_dim=16)
        missing, unexpected = reference.load_state_dict(wrapper.state_dict(), strict=False)
        assert not missing, missing
        err = rel_err(wrapper(x, use_cache=False)[0], reference(x)[0])
        print(f'2. layer {i}: wrapper vs reference class: rel err {err:.1e}')

    # 3a. Eval path: lm-eval calls the model with use_cache=True, which runs the recurrent prefill
    student_logits = model(ids, use_cache=False).logits
    cached_logits = model(ids, use_cache=True).logits
    print(f'3a. use_cache=True (eval path) vs use_cache=False: rel err {rel_err(cached_logits, student_logits):.1e}')

    # 3b. Generation: prefill 5 tokens, then decode one token at a time past the window
    cache = LizardAttentionCache()
    out = model(ids[:, :5], past_key_values=cache, use_cache=True)
    decoded = [out.logits]
    for t in range(5, LENGTH):
        out = model(ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
        decoded.append(out.logits)
    decoded = torch.cat(decoded, dim=1)
    print(f'3b. prefill + token-by-token decode vs full forward: rel err {rel_err(decoded, student_logits):.1e}')
```

**Results:**

| Check | What it covers | Result |
|---|---|---|
| 1. Teacher mode | Lizard replaces every attention layer in distillation mode. In this mode, the layer gives softmax attention from its own copied projections. The logits must not change. This checks the copy of q/k/v/o, the expansion of GQA heads, RoPE and the layer mapping. | Identical |
| 2. Student mode | The wrapper gives the same result as the reference class with the same weights. | Identical |
| 3a. Evaluation path | `use_cache=True` (recurrent prefill, which lm-eval uses) against the plain forward pass | Identical |
| 3b. Generation | Prefill, then decoding token by token beyond the window, against the full forward pass | Identical at the level of the logits |

### Why the results are "identical" and not only "near"

`LolcatsLlamaForCausalLM` casts its logits to float32 (`logits.float()` in `src/model/modeling_llama.py`). Thus comparisons at the model level are exact only to float32 precision (~1e-7). Checks 1, 2 and 3a run the same operations on the two sides, so their results can be identical to the bit. Check 3b uses different arithmetic: recurrent state updates against dense matrices. Thus check 3b got more examination:

- **Inside the model**, the decode output of layer 0 is different from its dense output by 2.2e-14, on values of approximately 29.5. The relative difference is ~7.5e-16, which is float64 rounding. The difference is not visible in the float32 logits.
- **One layer alone** gives a difference of 7.8e-16 between decode and dense, on values of approximately 1.1.
- **The decode path really runs.** It ran 38 times (2 layers × 19 decoded tokens). The cache counted 24 tokens, and the window cache contained 8 keys.
- **Negative control.** Decoding with an empty cache for each token (no history) gives a relative error of 0.59. Thus the check finds a missing history.

## What these checks show, and what they do not show

- **They show** that the code calculates the equations of the paper. They also show that the conversion connects Lizard attention into the model as intended. The evaluation path and the generation path calculate the same function as training.
- **They do not show** the quality of the trained weights. They also do not show the behavior at full scale (bf16, 2048-token sequences), except for the overflow fix in [document 5](05-nan-crash.md).

In the same harness, the teacher gets the expected score ([document 7](07-results.md)). With this result, the checks exclude the attention code, the connection into the model and the harness as causes of the poor results. The remaining cause is the training.

## Earlier tests (PR #1)

When PR #1 added the layer, similar tests ran on a tiny model ([document 1](01-lizard-in-lolcats.md)). They tested these properties:

- the match with the jku-thesis layer and with `lizard_loop` in float64,
- the distillation target and the gradients,
- cached generation against a full recalculation,
- a bf16 pass with only finite values.

The checks above repeat the core of these tests on the merged code, after the NaN fix. They also add the swap in teacher mode and the evaluation path.
