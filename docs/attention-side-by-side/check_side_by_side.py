"""
Checks of claims A and D' of docs/15-attention-math-side-by-side.md (CPU, a few seconds)
-> Run from the repository root: python docs/attention-side-by-side/check_side_by_side.py
-> A: LoLCATs Eq. 7, written out from the paper, against hybrid_attention_quadratic of the repo (float32,
   because that function casts to float32). The exit code is 1 if the relative error is 1e-5 or more.
-> D': the total weight of the gated branch of Liger (softmax features, gate sigmoid(k)^(1/16), no denominator)
   on random inputs, and the total weight of the mix 0.5 window + 0.5 gated branch
"""
import math
import sys

import torch

sys.path.insert(0, '.')
from src.model.linear_attention.linear_window_attention_sw import hybrid_attention_quadratic  # noqa: E402

torch.set_default_dtype(torch.float32)

# A: LoLCATs Eq. 7 against the code
torch.manual_seed(0)
b, h, L, d, f, w = 1, 2, 40, 8, 6, 8
q, k, v = torch.randn(b, h, L, d), torch.randn(b, h, L, d), torch.randn(b, h, L, d)
W = torch.randn(d, f) * 0.5
phi = lambda x: torch.cat([(x @ W).softmax(-1), (-x @ W).softmax(-1)], -1)  # noqa: E731
fq, fk = phi(q), phi(k)
a = torch.sigmoid(torch.tensor(-2.1972245773362196)) * torch.ones(1, h, 1, 1)  # window factor, start 0.1
y_code, _ = hybrid_attention_quadratic(q, k, fq, fk, v, a, 1, w)
y_eq = torch.zeros_like(v)
for n in range(L):
    lo = max(0, n - w + 1)
    s = (q[..., n, None, :] * k[..., lo:n + 1, :]).sum(-1) / math.sqrt(d)  # window scores
    e = a[..., 0] * (s - s.amax(-1, keepdim=True)).exp()
    num, den = (e[..., None] * v[..., lo:n + 1, :]).sum(-2), e.sum(-1)
    if lo > 0:  # linear terms only for the positions before the window
        kern = (fq[..., n, None, :] * fk[..., :lo, :]).sum(-1)
        num, den = num + (kern[..., None] * v[..., :lo, :]).sum(-2), den + kern.sum(-1)
    y_eq[..., n, :] = num / den[..., None]
err = ((y_eq - y_code).norm() / y_code.norm()).item()
print(f"{'PASS' if err < 1e-5 else 'FAIL'}  A: LoLCATs Eq. 7 against hybrid_attention_quadratic, "
      f"relative error {err:.1e}")

# D': total weight of the gated branch of Liger at random inputs
torch.manual_seed(1)
L2, positions = 512, (15, 63, 255, 511)
q2, k2 = torch.randn(1, 4, L2, 64), torch.randn(1, 4, L2, 64)
pq, pk = q2.softmax(-1), k2.softmax(-1)
log_b = (torch.nn.functional.logsigmoid(k2) / 16).cumsum(-2)  # log of the cumulative gates
mu = []
for i in positions:
    decay = (log_b[..., i, None, :] - log_b[..., :i + 1, :]).exp()  # b_i / b_j for each key dimension
    mu.append((pq[..., i, None, :] * decay * pk[..., :i + 1, :]).sum(-1).sum(-1).mean().item())
print("INFO  D': total weight of the gated branch at positions 16, 64, 256, 512: "
      + ', '.join(f'{m:.3f}' for m in mu))
print("INFO  D': total weight of 0.5 window + 0.5 gated branch: " + ', '.join(f'{0.5 + 0.5 * m:.3f}' for m in mu))
sys.exit(0 if err < 1e-5 else 1)
