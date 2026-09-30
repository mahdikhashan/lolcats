# 8. Verifying the attention layer

**Question:** is the poor evaluation result caused by a bug in the attention layer, or by something
else? This document checks the code that trained and evaluated the model: `main` at the time,
including the NaN fix from PR #5.

All checks ran on CPU with PyTorch 2.0.1 and transformers 4.43.1 (the version lolcats pins),
Python 3.11.

## Level 1: the math matches the reference formulas

jku-thesis has a test suite (`test_lizard.py`) that compares its dense implementation with slow
loop and recurrent versions of every formula (`reference.py`) in float64, requiring a relative error
below 1e-10. These tests were pointed at the **lolcats** functions through a small shim module named
`lizard_attention.py`:

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

**Result: 26 passed, 2 deselected.** Covered:

- Hedgehog feature maps against their formula; both halves positive and summing to 1.
- Gated linear attention against the loop formula and the recurrent form; a zero gate keeps only the
  current token; constant values stay constant.
- Window attention with sinks against the loop formula; without sinks and with an unbounded window it
  equals causal softmax; several sinks equal one sink at their logsumexp.
- No gradient from future tokens in either branch; window gradients only inside the window.
- Gradients against finite differences; the full layer against the loop formula; causality and batch
  independence; every Lizard parameter receives a gradient.

The two deselected tests need a newer transformers than 4.43 (`LlamaRotaryEmbedding(config=...)`);
the same failure occurs on the thesis's own code in this environment. Level 2 covers what they check.

## Level 2: the LoLCATs wrapper and the model wiring

The Level 1 tests don't cover `LolcatsLizardAttention`, the conversion inside a real model, or the
cached paths used by evaluation and generation. This script checks them inside
`LolcatsLlamaForCausalLM` on a tiny random Llama in float64, with grouped-query attention (4 query
heads, 2 key/value heads, as in Llama-3.2), a window of 8 and sequences of 24 tokens:

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
| 1. Teacher mode | Replacing every attention with Lizard in distillation mode (which outputs softmax attention computed from the layer's own copied projections) must leave the logits unchanged: q/k/v/o copying, GQA head expansion, RoPE, layer mapping | identical |
| 2. Student mode | The wrapper computes the same as the reference class with the same weights | identical |
| 3a. Eval path | `use_cache=True` (recurrent prefill, used by lm-eval) vs the plain forward | identical |
| 3b. Generation | Prefill, then decoding token by token past the window, vs the full forward | identical at logit level |

### Why "identical" and not just "close"

`LolcatsLlamaForCausalLM` casts its logits to float32 (`logits.float()` in
`src/model/modeling_llama.py`), so model-level comparisons are exact only to float32 precision
(~1e-7). Checks 1, 2 and 3a run the same operations on both sides and can be bit-identical. Check 3b
uses different arithmetic (recurrent state updates vs dense matrices), so it was examined further:

- **Inside the model**, layer 0's decode output differs from its dense output by 2.2e-14 on values
  of about 29.5 (relative ~7.5e-16, float64 rounding). The difference disappears in the float32 logits.
- **A single layer on its own** gives a decode vs dense difference of 7.8e-16 on values of about 1.1.
- **The decode path really runs:** it was called 38 times (2 layers × 19 decoded tokens), the cache
  counted 24 tokens, and the window cache held 8 keys.
- **Negative control:** decoding each token with an empty cache (no history) gives a relative error
  of 0.59, so the check does detect a missing history.

## What these checks do and don't show

- **They show** that the code computes the paper's equations, that the conversion into the model is
  wired correctly, and that the evaluation and generation paths compute the same function as training.
- **They don't show** anything about the quality of the trained weights, or behavior at full scale
  (bf16, 2048-token sequences), beyond the overflow fix in [document 5](05-nan-crash.md).

Together with the teacher scoring as expected in the same harness ([document 7](07-results.md)),
this rules out the attention code, the wiring and the harness as causes of the poor results, and
leaves the training.

## Earlier tests (PR #1)

When the layer was first added, similar tests ran on a tiny model: equivalence to the jku-thesis
layer and `lizard_loop` in float64, the distillation target and gradients, cached generation vs
recomputation, and a finite bf16 pass ([document 1](01-lizard-in-lolcats.md)). The checks above
repeat the core of these on the merged code, after the NaN fix, and add the teacher-mode swap and
the eval path.
