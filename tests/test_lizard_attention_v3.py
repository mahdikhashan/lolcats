"""
Checks of Lizard attention v3 (src/model/linear_attention/lizard_attention_v3.py), docs/16-lizard-attention-v3.md

CPU, float64, tiny random layers. Run from the repo root: python -m pytest -q tests/test_lizard_attention_v3.py
The tests use v3 as it is. They reuse the loop reference of the v2 tests, which is written from
docs/math-formula.md and reads only the projections and the Lizard parameters of the layer.
"""
import pytest
import torch
from omegaconf import OmegaConf

from src.model.linear_attention.lizard_attention import LolcatsLizardAttention, LizardAttentionCache
from src.model.linear_attention.lizard_attention_v3 import LolcatsLizardAttentionV3
from test_lizard_attention_v2 import (FEATURES, LENGTH, OPTIONS, WINDOW, lizard_out, make_base, reference,
                                      rel_err, tiny_config)

torch.set_default_dtype(torch.float64)

# The option sets of v2 (convex is not a v3 option), plus the two v3 additions
OPTIONS_V3 = {**OPTIONS,
              'alpha_trained': {'train_alpha': True, 'alpha_init': 0.7},
              'reparam_exp': {'feature_activation': 'exp', 'gla_impl': 'reparam'}}


def make_layer(seed=0, base=None, **options):
    """v3 layer from a random LlamaAttention, with random (not initial) Lizard parameters"""
    base = make_base(seed) if base is None else base
    layer = LolcatsLizardAttentionV3(base, layer_idx=0, window_size=WINDOW, num_meta=4,
                                     feature_dim=FEATURES, **options)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name.startswith(('phi_', 'W_gamma', 'meta_tokens')):
                p.normal_(0, 0.5)
        if layer.alpha_blend.dim():  # different alpha for each head
            layer.alpha_blend.copy_(torch.rand(layer.heads) + 0.2)
    return layer.eval()


# 1. The v3 defaults give the outputs of v1 with alpha = 1, and alpha is not a trainable parameter
def test_defaults_match_v1():
    base = make_base(seed=1)
    v3 = make_layer(base=base)
    assert 'alpha_blend' not in dict(v3.named_parameters()) and v3.alpha_blend.item() == 1.0
    v1 = LolcatsLizardAttention(base, layer_idx=0, window_size=WINDOW, num_meta=4, feature_dim=FEATURES)
    v1.load_state_dict(v3.state_dict())  # the same names and shapes; alpha_blend is a buffer in v3
    x = torch.randn(2, LENGTH, 32)
    with torch.no_grad():
        assert rel_err(v3(x)[0], v1(x)[0]) < 1e-12


# 2. Parallel form against the loop form of the math document
@pytest.mark.parametrize('name', OPTIONS_V3)
def test_parallel_matches_loop_reference(name):
    layer = make_layer(seed=2, **OPTIONS_V3[name])
    x = torch.randn(2, LENGTH, 32)
    with torch.no_grad():
        y, _ = lizard_out(layer, x)
        assert rel_err(y, reference(layer, x)) < 1e-10


# 3. Decode (recurrent form, LizardAttentionCache) against the parallel form
@pytest.mark.parametrize('name', OPTIONS_V3)
def test_decode_matches_parallel(name):
    layer = make_layer(seed=3, **OPTIONS_V3[name])
    x = torch.randn(2, LENGTH, 32)
    with torch.no_grad():
        full = layer(x, position_ids=torch.arange(LENGTH)[None])[0]
        cache = LizardAttentionCache()
        chunks = [(0, 3), (3, 5)] + [(t, t + 1) for t in range(5, LENGTH)]  # one decode call has 2 tokens
        ys = [layer(x[:, s:e], position_ids=torch.arange(s, e)[None], past_key_value=cache)[0]
              for s, e in chunks]
        assert rel_err(torch.cat(ys, dim=1), full) < 1e-10


# 4. Stage 1: the trainable Lizard parameters of the defaults get a gradient; alpha stays at 1
def test_stage1_parameters_and_gradients():
    layer = make_layer(seed=4)
    layer.train_attention = True
    x = torch.randn(2, LENGTH, 32)
    _, ((_, _), (y_pred, y_true)), _ = layer(x, position_ids=torch.arange(LENGTH)[None])
    ((y_pred - y_true) ** 2).mean().backward()
    trainable = {n: p for n, p in layer.named_parameters() if not n.endswith('_proj.weight')}
    assert set(trainable) == {'phi_q.weight', 'phi_k.weight', 'W_gamma.weight', 'meta_tokens'}
    for n, p in trainable.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n


# 5. The attention type lolcats_llama_lizard_v3 converts a model, and the cached decode gives the same logits
def test_model_conversion_and_cached_decoding():
    from src.model.modeling_llama import LolcatsLlamaForCausalLM
    from src.model.convert_model import convert_attention, toggle_attention
    torch.manual_seed(7)
    model = LolcatsLlamaForCausalLM(tiny_config(initializer_range=0.3)).eval()
    ids = torch.randint(0, 97, (2, 24))
    attention_config = OmegaConf.load('configs/model/distill_llama3_2_1b_lizard_v3_w128_fd32_m4_fp32.yaml').attention
    attention_config.window_size, attention_config.feature_dim = 8, 16  # tiny sizes, v3 defaults otherwise
    with torch.no_grad():
        teacher = model(ids, use_cache=False).logits
        model = convert_attention(model, attention_config, train_attention=True, remove_base_attn=True)
        assert all(isinstance(l.self_attn, LolcatsLizardAttentionV3) for l in model.model.layers)
        out = model(ids, use_cache=False, output_attentions=True)
        assert rel_err(out.logits, teacher) < 1e-6  # teacher mode; the model casts logits to float32
        for layer in model.model.layers:
            for name, p in layer.self_attn.named_parameters():
                if name.startswith(('phi_', 'W_gamma', 'meta_tokens')):
                    p.normal_(0, 0.5)
        toggle_attention(model, train=False)
        student = model(ids, use_cache=False).logits
        assert rel_err(model(ids, use_cache=True).logits, student) < 1e-6  # eval path (lm-eval)
        cache = LizardAttentionCache()
        out = model(ids[:, :5], past_key_values=cache, use_cache=True)
        decoded = [out.logits]
        for t in range(5, 24):
            out = model(ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
            decoded.append(out.logits)
        assert rel_err(torch.cat(decoded, dim=1), student) < 1e-6
