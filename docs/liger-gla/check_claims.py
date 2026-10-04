"""
Float64 checks of the claims P7-P11 of docs/14-liger-gla.md, on tiny random layers (CPU, a few seconds)
-> Run from the repository root: python docs/liger-gla/check_claims.py
-> Each check prints its measured value and PASS or FAIL. The exit code is 1 if a check fails.
"""
import math
import sys

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

sys.path.insert(0, '.')
from src.model.linear_attention.lizard_attention_v2 import LolcatsLizardAttentionV2  # noqa: E402

torch.set_default_dtype(torch.float64)
FAILED = []


def check(name, value, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value}")
    if not ok:
        FAILED.append(name)


def make_layer(window=8, seed=0, **options):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         attn_implementation='eager')
    base = LlamaAttention(config, layer_idx=0)
    for p in base.parameters():
        p.data.normal_(0, 0.3)
    return LolcatsLizardAttentionV2(base, layer_idx=0, window_size=window, num_meta=4, feature_dim=6, **options)


def qkv_gamma(layer, x):
    q = layer.split(layer.q_proj(x), layer.heads)
    k = layer.split(layer.k_proj(x), layer.kv_heads)
    gamma = torch.sigmoid(layer.W_gamma(x)).transpose(1, 2)
    return q, k, gamma


def window_share(layer, x, position_ids=None):
    """Share of the window branch in the total weight of each row, mean over heads: (b, l)"""
    q, k, gamma = qkv_gamma(layer, x)
    rope = layer.rotary_emb(q, position_ids) if layer.window_rope else None
    a_gla, a_win = layer.branch_weights(q, k, gamma, rope)
    total = a_gla.sum(-1) + a_win.sum(-1)
    return (a_win.sum(-1) / total).mean(1), total


# P7: sink logits that start equal stay equal under Adam (they enter the loss symmetrically)
layer = make_layer(train_attention=True)
for p in layer.parameters():
    p.requires_grad_(False)
lizard = [layer.phi_q.weight, layer.phi_k.weight, layer.W_gamma.weight, layer.meta_tokens, layer.alpha_blend]
for p in lizard:
    p.requires_grad_(True)
with torch.no_grad():
    layer.phi_q.weight.normal_(0, 0.5)
    layer.phi_k.weight.normal_(0, 0.5)
    layer.W_gamma.weight.normal_(0, 0.5)
opt = torch.optim.Adam(lizard, lr=1e-2)
torch.manual_seed(1)
x = torch.randn(2, 24, 32)
pos = torch.arange(24)[None].expand(2, -1)
for _ in range(50):
    (_, (y_pred, y_true)), = [layer(x, position_ids=pos)[1]]
    loss = (y_pred - y_true).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
t = layer.meta_tokens.detach()
check('P7 sinks after 50 Adam steps, max |t_j - t_0|', f'{(t - t[0]).abs().max().item():.1e} (t_0 = {t[0].item():+.4f})',
      (t - t[0]).abs().max().item() == 0.0 and t[0].item() != 0.0)

# P8: row form total weight = 1 + alpha * rho; division by (1 + alpha) keeps the shares, total weight <= 1
layer = make_layer(alpha_init=0.7).eval()
with torch.no_grad():
    layer.phi_q.weight.normal_(0, 0.5)
    layer.phi_k.weight.normal_(0, 0.5)
    layer.W_gamma.weight.normal_(0, 0.5)
    layer.meta_tokens.normal_(0, 0.5)
torch.manual_seed(2)
x = torch.randn(1, 40, 32)
q, k, gamma = qkv_gamma(layer, x)
a_gla, a_win = layer.branch_weights(q, k, gamma)
alpha = layer.alpha_blend.item()
rho = a_win.sum(-1) / alpha
err = (a_gla.sum(-1) + a_win.sum(-1) - (1 + alpha * rho)).abs().max().item()
check('P8a row form: max |total weight - (1 + alpha rho)|', f'{err:.1e} (total weight up to '
      f'{(a_gla.sum(-1) + a_win.sum(-1)).max().item():.3f})', err < 1e-12)
c_gla, c_win = a_gla / (1 + alpha), a_win / (1 + alpha)
share_row = a_win.sum(-1) / (a_gla.sum(-1) + a_win.sum(-1))
share_cvx = c_win.sum(-1) / (c_gla.sum(-1) + c_win.sum(-1))
total_cvx = (c_gla.sum(-1) + c_win.sum(-1)).max().item()
check('P8b convex form: same shares as the row form, total weight <= 1',
      f'max share difference {(share_row - share_cvx).abs().max().item():.1e}, largest total weight {total_cvx:.4f}',
      (share_row - share_cvx).abs().max().item() < 1e-12 and total_cvx <= 1 + 1e-12)

# P8d: alpha = 1 and no sink weight (rho = 1): the convex form is 0.5 * GLA + 0.5 * window softmax
layer = make_layer(alpha_init=1.0).eval()
with torch.no_grad():
    layer.phi_q.weight.normal_(0, 0.5)
    layer.phi_k.weight.normal_(0, 0.5)
    layer.meta_tokens.fill_(-1e4)  # exp(t_j) = 0
v = layer.split(layer.v_proj(x), layer.kv_heads)
q, k, gamma = qkv_gamma(layer, x)
y_row = layer.lizard(q, k, v, gamma)
a_gla, a_win = layer.branch_weights(q, k, gamma)
y_half = 0.5 * (a_gla @ v) + 0.5 * (a_win @ v)  # split() repeats the key/value heads
err = ((y_row / 2 - y_half).norm() / y_half.norm()).item()
check('P8d alpha = 1, rho = 1: relative error of (row form) / 2 against 0.5 GLA + 0.5 window', f'{err:.1e}', err < 1e-12)

# P8c and P9: with gamma = 1 and a periodic input, the share is the same at i and i + period for row/convex,
# and decreases approximately as 1 / i for hybrid
period, window = 8, 8
torch.manual_seed(3)
block = torch.randn(1, period, 32)
x = block.repeat(1, 33, 1)  # 264 positions
shares = {}
for norm in ('row', 'hybrid'):
    layer = make_layer(window=window, gla_norm=norm, alpha_init=0.1 if norm == 'hybrid' else 1.0,
                       gate_bias_init=40.0).eval()  # W_gamma = 0 and bias 40: gamma = 1 - 4e-18
    with torch.no_grad():
        layer.phi_q.weight.normal_(0, 0.5)
        layer.phi_k.weight.normal_(0, 0.5)
    shares[norm] = window_share(layer, x)[0][0]
idx = [15, 31, 63, 127, 255]  # same position inside the period
row = [shares['row'][i].item() for i in idx]
hyb = [shares['hybrid'][i].item() for i in idx]
check('P8c row (and convex): share at i = 16, 32, 64, 128, 256', ', '.join(f'{s:.4f}' for s in row),
      max(row) - min(row) < 1e-12)
products = [s * (i + 1) for s, i in zip(hyb, idx)]
check('P9 hybrid, gamma = 1: share at i = 16, 32, 64, 128, 256', ', '.join(f'{s:.4f}' for s in hyb)
      + f'; share x i = ' + ', '.join(f'{p:.2f}' for p in products),
      all(a > b for a, b in zip(hyb, hyb[1:])) and products[-1] / products[-2] < 1.05)

# P10: without RoPE, a repeated token gets the same window weight at positions i-1 and i-2. With RoPE, it does not.
torch.manual_seed(4)
x = torch.randn(1, 12, 32)
x[0, 9] = x[0, 10]  # positions 9 and 10 hold the same token; the query is position 11
pos = torch.arange(12)[None]
for rope in (False, True):
    layer = make_layer(window=8, window_rope=rope).eval()
    q, k, gamma = qkv_gamma(layer, x)
    r = layer.rotary_emb(q, pos) if rope else None
    _, a_win = layer.branch_weights(q, k, gamma, r)
    gap = (a_win[0, :, 11, 10] - a_win[0, :, 11, 9]).abs().max().item()
    if rope:
        check('P10 window with RoPE: weights of positions 10 and 9 differ, max difference', f'{gap:.3f}', gap > 1e-3)
    else:
        check('P10 window without RoPE: weights of positions 10 and 9 are equal, max difference', f'{gap:.1e}',
              gap < 1e-15)

# P11: the GLA parallel form ((Q * B)(K / B)^T * M) V equals the recurrent form S_t = diag(a_t) S_{t-1} + k_t v_t^T
torch.manual_seed(5)
L, d, dv = 10, 5, 3
Q, K, V = torch.randn(L, d), torch.randn(L, d), torch.randn(L, dv)
A = torch.rand(L, d) * 0.5 + 0.5  # a gate for each position and each key dimension
B = A.cumprod(0)
M = torch.ones(L, L).tril()
O_par = (((Q * B) @ (K / B).T) * M) @ V
S, O_rec = torch.zeros(d, dv), []
for t in range(L):
    S = A[t][:, None] * S + K[t][:, None] * V[t][None]
    O_rec.append(Q[t] @ S)
O_rec = torch.stack(O_rec)
err = ((O_par - O_rec).norm() / O_rec.norm()).item()
check('P11 GLA parallel form against the recurrent form (gate for each dimension)', f'relative error {err:.1e}',
      err < 1e-12)
# With one scalar gate for each position, the same form is the gated branch of v2 without a denominator
layer = make_layer(gla_norm='none').eval()
with torch.no_grad():
    layer.phi_q.weight.normal_(0, 0.5)
    layer.phi_k.weight.normal_(0, 0.5)
    layer.W_gamma.weight.normal_(0, 0.5)
torch.manual_seed(6)
x = torch.randn(1, 10, 32)
q, k, gamma = qkv_gamma(layer, x)
fq, fk = layer.feature_maps(q, k)
a_gla, _ = layer.branch_weights(q, k, gamma)
C = gamma.cumprod(-1)[..., None]  # (b, 1, l, 1): scalar gate, the same for all heads
a_par = ((fq * C) @ (fk / C).transpose(-1, -2)) * torch.ones(10, 10).tril()
err = ((a_par - a_gla).norm() / a_gla.norm()).item()
check('P11 scalar gate: (phi(Q) * C)(phi(K) / C)^T * M against branch_weights of gla_norm none',
      f'relative error {err:.1e}', err < 1e-12)

# GLA gate start value and the gate logit that layer 15 of R1b would need with the exponent 1/16
s15 = 0.128 ** 16  # sigmoid(z)^(1/16) = 0.128  <=>  sigmoid(z) = 0.128^16
logit = math.log(s15 / (1 - s15))
check('GLA gate: sigmoid(0)^(1/16), and the logit for gamma = 0.128',
      f'{0.5 ** (1 / 16):.4f}, {logit:.2f} (without the exponent: {math.log(0.128 / (1 - 0.128)):.2f})', True)

print(f'\n{len(FAILED)} failed' if FAILED else '\nall checks pass')
sys.exit(1 if FAILED else 0)
