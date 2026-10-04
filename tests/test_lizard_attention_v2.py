"""
Checks of Lizard attention v2 (src/model/linear_attention/lizard_attention_v2.py), docs/13-lizard-attention-v2.md

CPU, float64, tiny random layers. Run from the repo root: python -m pytest -q tests/test_lizard_attention_v2.py
The loop reference below is written from docs/math-formula.md (sections 1, 2, 3 and 7). It does not use
the code of v1 or v2, except for the q, k, v projections and the Lizard parameters it reads from the layer.
"""
import math

import pytest
import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb as hf_rope

from src.model.linear_attention.lizard_attention import LolcatsLizardAttention, LizardAttentionCache
from src.model.linear_attention.lizard_attention_v2 import LolcatsLizardAttentionV2

torch.set_default_dtype(torch.float64)
WINDOW, LENGTH, FEATURES = 4, 11, 6  # the sequence is longer than the window

# One option at a time (docs/12-gap-analysis-2.md, 5.2), then combinations
OPTIONS = {
    'defaults': {},
    'norm_none': {'gla_norm': 'none'},
    'norm_joint': {'gla_norm': 'joint'},
    'alpha_per_head': {'alpha_per_head': True},
    'alpha_frozen': {'train_alpha': False, 'alpha_init': 0.6},
    'map_per_head': {'feature_map_per_head': True},
    'exp_features': {'feature_activation': 'exp'},
    'gate_per_head': {'gate_per_head': True},
    'gate_bias': {'gate_bias_init': 3.0},
    'window_rope': {'window_rope': True},
    'all_joint': {'gla_norm': 'joint', 'alpha_per_head': True, 'feature_map_per_head': True,
                  'gate_per_head': True, 'gate_bias_init': 3.0, 'window_rope': True},
    'all_none_exp': {'gla_norm': 'none', 'alpha_per_head': True, 'feature_map_per_head': True,
                     'feature_activation': 'exp', 'gate_per_head': True, 'window_rope': True},
    'norm_hybrid': {'gla_norm': 'hybrid'},
    'all_hybrid': {'gla_norm': 'hybrid', 'alpha_per_head': True, 'alpha_init': 0.1, 'feature_map_per_head': True,
                   'gate_per_head': True, 'gate_bias_init': 3.0, 'window_rope': True},
}


def tiny_config(**kwargs):
    return LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2,  # GQA, as in Llama 3.2
                       max_position_embeddings=128, attn_implementation='eager', **kwargs)


def make_base(seed=0):
    torch.manual_seed(seed)
    base = LlamaAttention(tiny_config(), layer_idx=0)
    for p in base.parameters():
        p.data.normal_(0, 0.3)
    return base


def make_layer(seed=0, base=None, **options):
    """v2 layer from a random LlamaAttention, with random (not initial) Lizard parameters"""
    base = make_base(seed) if base is None else base
    layer = LolcatsLizardAttentionV2(base, layer_idx=0, window_size=WINDOW, num_meta=4,
                                     feature_dim=FEATURES, **options)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name.startswith(('phi_', 'W_gamma', 'meta_tokens')):
                p.normal_(0, 0.5)
        if layer.alpha_blend.dim():  # different alpha for each head
            layer.alpha_blend.copy_(torch.rand(layer.heads) + 0.2)
        elif isinstance(layer.alpha_blend, torch.nn.Parameter):
            layer.alpha_blend.fill_(0.7)
    return layer.eval()


def qkv(layer, x):
    q = layer.split(layer.q_proj(x), layer.heads)
    k = layer.split(layer.k_proj(x), layer.kv_heads)
    v = layer.split(layer.v_proj(x), layer.kv_heads)
    return q, k, v


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def reference(layer, x, position_ids=None):
    """
    Loop form of docs/math-formula.md for one layer, before o_proj, shape (b, h, l, d)
    - gated branch (section 2): sum_t prod_{l=t+1}^{i} gamma_l phi_q(q_i).phi_k(k_t) v_t, with
      the denominator of the parallel form ('row'), without it ('none'), or shared with the window ('joint').
      'hybrid' shares it too, with the window and sink weights divided by exp(max(window scores, t))
    - window branch (section 1): window of the last w positions, sinks exp(t_j) in the denominator (D5)
    - Hedgehog map (section 7): [act(W x) (+) act(-W x)], act = softmax or exp, W of the head or shared
    """
    q, k, v = qkv(layer, x)
    b, h, n, d = q.shape
    gamma = torch.sigmoid(layer.W_gamma(x))  # (b, l, 1 or h)
    alpha = layer.alpha_blend
    meta = layer.meta_tokens
    if layer.window_rope:
        positions = torch.arange(n)[None] if position_ids is None else position_ids
        cos, sin = layer.rotary_emb(v, positions)
        qw, kw = hf_rope(q, k, cos, sin)
    else:
        qw, kw = q, k

    def phi(weight, head, vec):
        w = weight[head] if weight.dim() == 3 else weight  # (f, d)
        z = w @ vec
        act = (lambda u: u.exp()) if layer.feature_activation == 'exp' else (lambda u: u.softmax(-1))
        return torch.cat([act(z), act(-z)])

    y = torch.zeros_like(q)
    for bi in range(b):
        for hi in range(h):
            g = gamma[bi, :, hi if gamma.shape[-1] > 1 else 0]
            a_h = alpha[hi] if alpha.dim() else alpha
            for i in range(n):
                fq = phi(layer.phi_q.weight, hi, q[bi, hi, i])
                num_gla, den_gla = torch.zeros(d), torch.zeros(())
                for t in range(i + 1):
                    weight = torch.prod(g[t + 1:i + 1]) * (fq @ phi(layer.phi_k.weight, hi, k[bi, hi, t]))
                    num_gla = num_gla + weight * v[bi, hi, t]
                    den_gla = den_gla + weight
                num_win, den_win, top = torch.zeros(d), meta.exp().sum(), meta.max()
                for t in range(max(0, i - WINDOW + 1), i + 1):
                    score = qw[bi, hi, i] @ kw[bi, hi, t] / math.sqrt(d)
                    e = torch.exp(score)
                    num_win = num_win + e * v[bi, hi, t]
                    den_win = den_win + e
                    top = torch.maximum(top, score)
                if layer.gla_norm == 'row':
                    y[bi, hi, i] = num_gla / den_gla + a_h * num_win / den_win
                elif layer.gla_norm == 'none':
                    y[bi, hi, i] = num_gla + a_h * num_win / den_win
                elif layer.gla_norm == 'joint':
                    y[bi, hi, i] = (num_gla + a_h * num_win) / (den_gla + a_h * den_win)
                else:  # hybrid
                    scale = abs(a_h) * torch.exp(-top)
                    y[bi, hi, i] = (num_gla + scale * num_win) / (den_gla + scale * den_win)
    return y


def lizard_out(layer, x, position_ids=None):
    """v2 output before o_proj, parallel form"""
    q, k, v = qkv(layer, x)
    gamma = torch.sigmoid(layer.W_gamma(x)).transpose(1, 2)
    positions = torch.arange(x.shape[1])[None] if position_ids is None else position_ids
    rope = layer.rotary_emb(v, positions) if layer.window_rope else None
    return layer.lizard(q, k, v, gamma, rope), (q, k, v, gamma, rope)


# 1. The default options give v1
def test_defaults_match_v1_exactly():
    base = make_base(seed=1)
    v2 = make_layer(base=base)
    v1 = LolcatsLizardAttention(base, layer_idx=0, window_size=WINDOW, num_meta=4, feature_dim=FEATURES)
    v1.load_state_dict(v2.state_dict())  # the same names and shapes as v1
    x = torch.randn(2, LENGTH, 32)
    pos = torch.arange(LENGTH)[None]
    with torch.no_grad():
        assert torch.equal(v1(x)[0], v2(x)[0])  # student forward
        for layer in (v1, v2):
            layer.train_attention = True
        (_, (p1, t1)), (_, (p2, t2)) = v1(x, position_ids=pos)[1], v2(x, position_ids=pos)[1]
        assert torch.equal(p1, p2) and torch.equal(t1, t2)
        for layer in (v1, v2):
            layer.train_attention = False
        outs = []
        for layer in (v1, v2):  # prefill 3 tokens, then decode one token at a time past the window
            cache = LizardAttentionCache()
            ys = [layer(x[:, :3], past_key_value=cache)[0]]
            ys += [layer(x[:, t:t + 1], past_key_value=cache)[0] for t in range(3, LENGTH)]
            outs.append(torch.cat(ys, dim=1))
        assert torch.equal(outs[0], outs[1])


# 2. Parallel form against the loop form of the math document
@pytest.mark.parametrize('name', OPTIONS)
def test_parallel_matches_loop_reference(name):
    layer = make_layer(seed=2, **OPTIONS[name])
    x = torch.randn(2, LENGTH, 32)
    with torch.no_grad():
        y, _ = lizard_out(layer, x)
        assert rel_err(y, reference(layer, x)) < 1e-10


# 3. Decode (recurrent form, LizardAttentionCache) against the parallel form
@pytest.mark.parametrize('name', OPTIONS)
def test_decode_matches_parallel(name):
    layer = make_layer(seed=3, **OPTIONS[name])
    x = torch.randn(2, LENGTH, 32)
    with torch.no_grad():
        full = layer(x, position_ids=torch.arange(LENGTH)[None])[0]
        cache = LizardAttentionCache()
        chunks = [(0, 3), (3, 5)] + [(t, t + 1) for t in range(5, LENGTH)]  # one decode call has 2 tokens
        ys = [layer(x[:, s:e], position_ids=torch.arange(s, e)[None], past_key_value=cache)[0]
              for s, e in chunks]
        assert rel_err(torch.cat(ys, dim=1), full) < 1e-10
        assert cache.k_cache[0].shape[2] == WINDOW


# 4. Causality: a change at position j does not change earlier outputs
@pytest.mark.parametrize('name', ['defaults', 'norm_joint', 'all_joint', 'all_none_exp', 'all_hybrid'])
def test_causal(name):
    layer = make_layer(seed=4, **OPTIONS[name])
    x = torch.randn(1, LENGTH, 32)
    x2 = x.clone()
    x2[:, 7] += torch.randn(32)
    pos = torch.arange(LENGTH)[None]
    with torch.no_grad():
        y, y2 = layer(x, position_ids=pos)[0], layer(x2, position_ids=pos)[0]
    assert torch.allclose(y[:, :7], y2[:, :7], rtol=0, atol=1e-13)
    assert not torch.allclose(y[:, 7:], y2[:, 7:])


# 5. XAI hook: the branch weights times v give the output
@pytest.mark.parametrize('name', ['defaults', 'norm_none', 'norm_joint', 'all_joint', 'all_none_exp',
                                  'norm_hybrid', 'all_hybrid'])
def test_branch_weights_reproduce_output(name):
    layer = make_layer(seed=5, **OPTIONS[name])
    x = torch.randn(2, LENGTH, 32)
    with torch.no_grad():
        y, (q, k, v, gamma, rope) = lizard_out(layer, x)
        a_gla, a_win = layer.branch_weights(q, k, gamma, rope)
        assert rel_err((a_gla + a_win) @ v, y) < 1e-12
        assert (a_gla >= 0).all() and (a_win >= 0).all()
        assert (a_gla.triu(1) == 0).all() and (a_win.triu(1) == 0).all()  # causal
        assert (a_win.tril(-WINDOW) == 0).all()  # window of WINDOW positions
        mass = (a_gla + a_win).sum(-1)
        if layer.gla_norm == 'row':  # the gated branch sums to 1 (D1)
            assert torch.allclose(a_gla.sum(-1), torch.ones(()), atol=1e-12)
        if layer.gla_norm in ('joint', 'hybrid'):  # a weighted mean of values: the sinks absorb the rest
            assert (mass < 1).all() and (mass > 0).all()


# 6. Shapes and parameter counts at the size of Llama-3.2-1B (fd32): gap analysis 2, table 5.2
def lizard_param_count(layer):
    return {n: p.numel() for n, p in layer.named_parameters() if not n.endswith(('q_proj.weight',
            'k_proj.weight', 'v_proj.weight', 'o_proj.weight'))}


def test_parameter_counts_1b_shapes():
    config = LlamaConfig(hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
                         attn_implementation='eager')
    base = LlamaAttention(config, layer_idx=0)
    make = lambda **o: LolcatsLizardAttentionV2(base, layer_idx=0, feature_dim=32, **o)
    v1 = lizard_param_count(make())
    assert v1 == {'phi_q.weight': 2048, 'phi_k.weight': 2048, 'W_gamma.weight': 2048,
                  'meta_tokens': 4, 'alpha_blend': 1}
    c1 = lizard_param_count(make(alpha_per_head=True))
    assert c1['alpha_blend'] - v1['alpha_blend'] == 31  # +496 for 16 layers
    c2 = lizard_param_count(make(feature_map_per_head=True))
    assert c2['phi_q.weight'] + c2['phi_k.weight'] == 131_072  # 4,096 -> 131,072 for each layer
    c4 = lizard_param_count(make(gate_per_head=True))
    assert c4['W_gamma.weight'] - v1['W_gamma.weight'] == 31 * 2048
    c6 = make(gate_bias_init=3.0)
    assert lizard_param_count(c6)['W_gamma.bias'] == 1  # +16 for 16 layers
    x = torch.randn(1, 5, 2048)
    assert torch.allclose(torch.sigmoid(c6.W_gamma(x)), torch.sigmoid(torch.tensor(3.0)))  # gamma ~ 0.953
    frozen = make(train_alpha=False, alpha_init=0.6)
    assert 'alpha_blend' not in dict(frozen.named_parameters())
    assert 'alpha_blend' in frozen.state_dict() and frozen.alpha_blend.item() == 0.6


# 7. The window-share ceiling alpha / (1 + alpha) (gap analysis 2, 3.1; docs/13, P2)
def test_window_share_ceiling():
    shares = {}
    for norm in ('row', 'joint'):
        layer = make_layer(seed=6, gla_norm=norm)
        x = torch.randn(1, LENGTH, 32)
        with torch.no_grad():
            q, k, v = qkv(layer, x)
            q, k = 4 * q, 4 * k  # large window scores
            gamma = torch.sigmoid(layer.W_gamma(x)).transpose(1, 2)
            a_gla, a_win = layer.branch_weights(q, k, gamma)
        shares[norm] = (a_win.sum(-1) / (a_gla.sum(-1) + a_win.sum(-1))).max().item()
    ceiling = 0.7 / 1.7
    assert shares['row'] <= ceiling + 1e-12  # the ceiling holds with the denominator of the gated branch
    assert shares['joint'] > ceiling  # one joint denominator removes it


# 8. Inside the model: conversion, the eval path (use_cache=True) and generation
def test_model_conversion_and_cached_decoding():
    from src.model.modeling_llama import LolcatsLlamaForCausalLM
    from src.model.convert_model import convert_attention, toggle_attention
    torch.manual_seed(7)
    model = LolcatsLlamaForCausalLM(tiny_config(initializer_range=0.3)).eval()
    ids = torch.randint(0, 97, (2, 24))
    attention_config = OmegaConf.create({
        'attention_type': 'lolcats_llama_lizard_v2', 'window_size': 8, 'num_meta': 4, 'feature_dim': 16,
        'softmax_attentions': [], **OPTIONS['all_joint']})
    with torch.no_grad():
        teacher = model(ids, use_cache=False).logits
        model = convert_attention(model, attention_config, train_attention=True, remove_base_attn=True)
        assert all(isinstance(l.self_attn, LolcatsLizardAttentionV2) for l in model.model.layers)
        # Teacher mode reproduces the original model; the distillation targets come out per layer
        out = model(ids, use_cache=False, output_attentions=True)
        assert rel_err(out.logits, teacher) < 1e-6  # the model casts logits to float32
        assert all(a[1][0].shape == a[1][1].shape for a in out.attentions)
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


# 9. Every trainable Lizard parameter gets a finite, nonzero gradient from the stage 1 loss
@pytest.mark.parametrize('name', OPTIONS)
def test_every_lizard_parameter_gets_a_gradient(name):
    layer = make_layer(seed=8, **OPTIONS[name])
    layer.train_attention = True
    x = torch.randn(2, LENGTH, 32)
    _, ((_, _), (y_pred, y_true)), _ = layer(x, position_ids=torch.arange(LENGTH)[None])
    ((y_pred - y_true) ** 2).mean().backward()
    # The q, k, v, o projections are frozen in training (load_and_convert_attns); o_proj is not in this loss
    trainable = {n: p for n, p in layer.named_parameters() if p.requires_grad and not n.endswith('_proj.weight')}
    assert set(trainable) >= {'phi_q.weight', 'phi_k.weight', 'W_gamma.weight', 'meta_tokens'}
    for n, p in trainable.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n


def gated_share(layer, length=128):
    """Mean share of the gated branch in the rows after the first window"""
    torch.manual_seed(9)
    x = torch.randn(1, length, 32)
    with torch.no_grad():
        q, k, v = qkv(layer, x)
        gamma = torch.sigmoid(layer.W_gamma(x)).transpose(1, 2)
        a_gla, a_win = layer.branch_weights(q, k, gamma)
    return (a_gla.sum(-1) / (a_gla.sum(-1) + a_win.sum(-1)))[..., layer.window:].mean().item()


# 10. At the start values, the gated branch must keep a share of the weight. With 'joint' it does not
#     (docs/13, run R1): its gated weights get a factor exp(-max score). A window of 32 tokens, because
#     with very few keys the largest window score is often negative, which hides the effect
def test_gated_share_at_start_values():
    base = make_base(seed=10)
    start = lambda **o: LolcatsLizardAttentionV2(base, layer_idx=0, window_size=32, num_meta=4,
                                                 feature_dim=FEATURES, **o).eval()
    shares = {'row': gated_share(start()), 'joint': gated_share(start(gla_norm='joint')),
              'hybrid': gated_share(start(gla_norm='hybrid', alpha_init=0.1))}
    assert shares['row'] > 0.3, shares
    assert shares['hybrid'] > 0.3, shares
    assert shares['joint'] < 0.01, shares


# 11. 'hybrid' uses |alpha|: the sign of alpha does not change the output
def test_hybrid_alpha_sign():
    layer = make_layer(seed=11, gla_norm='hybrid')
    x = torch.randn(1, LENGTH, 32)
    with torch.no_grad():
        y = layer(x)[0]
        layer.alpha_blend.neg_()
        assert torch.equal(layer(x)[0], y)
