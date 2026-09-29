"""
Lizard attention (https://arxiv.org/abs/2507.09025)

For each layer, we sum:
- Gated linear attention (GLA) with Hedgehog feature maps and a data-dependent decay
- Sliding window softmax attention with learnable sink ("meta") tokens (AWA)

The functions and `LizardAttention` are copied from jku-thesis/lizard_attention.py (dense
reference implementation that materializes L x L). `LolcatsLizardAttention` adapts it to the
LoLCATs attention swap and distillation interface, and `LizardAttentionCache` holds its
recurrent states for generation.
"""
from typing import List, Tuple, Optional
import math

import torch
import torch.nn as nn

from transformers.cache_utils import Cache

from src.model.rotary import apply_rotary_pos_emb
from .linear_attention import LinearAttentionState, softmax_attention


def hedgehog(x, weight):
    xw = x @ weight.T
    return torch.cat([xw.softmax(-1), (-xw).softmax(-1)], dim=-1)


def window_mask(length, window, device=None):
    i = torch.arange(length, device=device)[:, None]
    t = torch.arange(length, device=device)[None, :]
    return (t <= i) & (i - t < window)


def gate_products(gamma):
    causal = window_mask(gamma.shape[-1], gamma.shape[-1], gamma.device)
    g = torch.where(causal, gamma[..., None, :], 1.0)
    products = g.flip(-1).cumprod(-1).flip(-1)
    ones = torch.ones_like(products[..., :1])
    return torch.cat([products[..., 1:], ones], dim=-1) * causal


def gla(fq, fk, v, gamma):
    w = (fq @ fk.transpose(-1, -2)) * gate_products(gamma)[:, None]
    # Guard 0 / 0 when every weight in a row underflows
    return (w @ v) / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)


def sink_softmax(scores, meta):
    """
    exp(scores) / (sum(exp(meta)) + sum(exp(scores))) over the last dim, shifted
    by the row max over scores and meta so exp cannot overflow
    """
    m = torch.maximum(scores.amax(-1, keepdim=True), meta.max()).detach()
    e = (scores - m).exp()
    return e / ((meta - m).exp().sum(-1, keepdim=True) + e.sum(-1, keepdim=True))


def awa(q, k, v, meta, window):
    mask = window_mask(q.shape[-2], window, q.device)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return sink_softmax(scores.masked_fill(~mask, float("-inf")), meta) @ v


class LizardAttention(nn.Module):
    def __init__(self, config, layer_idx, window=128, num_meta=4,
                 feature_dim=128):
        super().__init__()
        hidden = config.hidden_size
        self.layer_idx = layer_idx
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = hidden // self.heads
        self.window = window
        kv_dim = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.phi_q = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.phi_k = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.W_gamma = nn.Linear(hidden, 1, bias=False)
        self.meta_tokens = nn.Parameter(torch.zeros(num_meta))
        self.alpha_blend = nn.Parameter(torch.ones(()))
        nn.init.normal_(self.phi_q.weight, std=0.02)
        nn.init.normal_(self.phi_k.weight, std=0.02)
        nn.init.zeros_(self.W_gamma.weight)

    def split(self, x, heads):
        b, n, _ = x.shape
        x = x.view(b, n, heads, self.head_dim).transpose(1, 2)
        return x.repeat_interleave(self.heads // heads, dim=1)

    def forward(self, hidden_states, **kwargs):
        x = hidden_states
        q = self.split(self.q_proj(x), self.heads)
        k = self.split(self.k_proj(x), self.kv_heads)
        v = self.split(self.v_proj(x), self.kv_heads)
        gamma = torch.sigmoid(self.W_gamma(x)).squeeze(-1)
        fq = hedgehog(q, self.phi_q.weight)
        fk = hedgehog(k, self.phi_k.weight)
        y = gla(fq, fk, v, gamma)
        y = y + self.alpha_blend * awa(q, k, v, self.meta_tokens, self.window)
        return self.o_proj(y.transpose(1, 2).flatten(2)), None


# ------------------------------
# LoLCATs attention swap classes
# ------------------------------
def upcast(x: torch.Tensor, *others: torch.Tensor) -> list[torch.Tensor]:
    """
    Cast tensors to x's dtype promoted to at least fp32 (e.g., bf16 -> fp32, fp64 stays)
    """
    dtype = torch.promote_types(x.dtype, torch.float32)
    return [t.to(dtype) for t in (x, *others)]


class LolcatsLizardAttention(LizardAttention):
    """
    Lizard attention initialized from a `LlamaAttention` object (base_attn),
    following the interface of `LolcatsLinearAttention`
    - Reuses base_attn's q, k, v, o projections; only the Lizard parameters are new
    - If train_attention, the layer outputs the original softmax attention and returns
      ((None, None), (y_pred, y_true)) as attention weights, so the distillation trainer
      supervises with MSE (Lizard attention weights are not returned; use xent_factor: 0)
    - As in the original, Lizard uses no RoPE and ignores padding masks
    """
    def __init__(self,
                 base_attn: nn.Module,  # like LlamaAttention
                 layer_idx: Optional[int] = None,
                 max_layer_idx: Optional[int] = None,
                 train_attention: Optional[bool] = False,
                 attention_type: Optional[str] = 'lolcats_llama_lizard',
                 window_size: int = 128,
                 num_meta: int = 4,
                 feature_dim: int = 128,
                 **kwargs: any) -> None:
        layer_idx = layer_idx if layer_idx is not None else base_attn.layer_idx
        super().__init__(base_attn.config, layer_idx, window=window_size,
                         num_meta=num_meta, feature_dim=feature_dim)
        self.max_layer_idx = max_layer_idx
        self.train_attention = train_attention
        self.attention_type = attention_type
        # Copy original model projection layers
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, getattr(base_attn, name))
        weight = self.q_proj.weight
        self.to(device=weight.device, dtype=weight.dtype)
        # Only for "ground-truth" softmax attention if position_embeddings are not passed in
        self.rotary_emb = getattr(base_attn, 'rotary_emb', None)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs,
               ) -> Tuple[torch.Tensor, Optional[tuple], Optional[Cache]]:
        """
        Forward pass consistent with HuggingFace Transformers LlamaAttention (v4.43)
        """
        x = hidden_states
        q = self.split(self.q_proj(x), self.heads)
        k = self.split(self.k_proj(x), self.kv_heads)
        v = self.split(self.v_proj(x), self.kv_heads)
        gamma = torch.sigmoid(*upcast(self.W_gamma(x))).squeeze(-1)

        attn_weights = None
        if self.train_attention:
            # 1. Compute "ground-truth" outputs of the original softmax attention (with RoPE)
            with torch.no_grad():
                cos, sin = (position_embeddings if position_embeddings is not None
                            else self.rotary_emb(v, position_ids))
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                y_true = softmax_attention(q_rope, k_rope, v)[0]
            # 2. Compute "predicted" Lizard outputs
            y_pred = self.lizard(q, k, v, gamma)
            attn_weights = ((None, None), (y_pred, y_true))
            y = y_true  # Next layer sees the original model's hidden states
        elif isinstance(past_key_value, LizardAttentionCache):  # Generating
            y = self.lizard_recurrent(q, k, v, gamma, past_key_value)
        else:  # Training, or generating without a cache (e.g., DynamicCache with use_cache=False)
            y = self.lizard(q, k, v, gamma)
        return self.o_proj(y.transpose(1, 2).flatten(2)), attn_weights, past_key_value

    def feature_maps(self, q: torch.Tensor, k: torch.Tensor):
        """Hedgehog feature maps in q's dtype"""
        return (hedgehog(q, self.phi_q.weight.to(q.dtype)),
                hedgehog(k, self.phi_k.weight.to(q.dtype)))

    def lizard(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               gamma: torch.Tensor) -> torch.Tensor:
        """
        Lizard attention over whole sequences, computed in (at least) fp32
        -> q, k, v are shape (b, h, l, head_dim); gamma is shape (b, l)
        """
        dtype = q.dtype
        q, k, v, alpha, meta = upcast(q, k, v, self.alpha_blend, self.meta_tokens)
        fq, fk = self.feature_maps(q, k)
        y = gla(fq, fk, v, gamma) + alpha * awa(q, k, v, meta, self.window)
        return y.to(dtype)

    def lizard_recurrent(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         gamma: torch.Tensor, cache: 'LizardAttentionCache') -> torch.Tensor:
        """
        Lizard attention when generating
        -> Prefill: attention over the prompt, then save the GLA states and last window of keys, values
        -> Decode: update the GLA states and window one token at a time (recurrent form)
        """
        dtype = q.dtype
        q, k, v, alpha, meta = upcast(q, k, v, self.alpha_blend, self.meta_tokens)
        fq, fk = self.feature_maps(q, k)
        if len(cache.kv_states) <= self.layer_idx:  # Prefill
            y = gla(fq, fk, v, gamma) + alpha * awa(q, k, v, meta, self.window)
            # decay[t] = prod_{s > t} gamma[s], i.e., the last row of gate_products(gamma)
            decay = torch.cat([gamma[:, 1:].flip(-1).cumprod(-1).flip(-1),
                               torch.ones_like(gamma[:, :1])], dim=-1)
            kv_state = torch.einsum('bl,bhlf,bhld->bhfd', decay, fk, v)
            k_state = torch.einsum('bl,bhlf->bhf', decay, fk)
            k_cache, v_cache = k[:, :, -self.window:], v[:, :, -self.window:]
        else:  # Decode
            kv_state = cache.kv_states[self.layer_idx]
            k_state = cache.k_states[self.layer_idx]
            k_cache = cache.k_cache[self.layer_idx]
            v_cache = cache.v_cache[self.layer_idx]
            y = []
            for i in range(q.shape[2]):
                # Gated linear attention
                g = gamma[:, i, None, None]  # b, 1, 1
                kv_state = (g[..., None] * kv_state
                            + torch.einsum('bhf,bhd->bhfd', fk[:, :, i], v[:, :, i]))
                k_state = g * k_state + fk[:, :, i]
                y_gla = (torch.einsum('bhf,bhfd->bhd', fq[:, :, i], kv_state) /
                         torch.einsum('bhf,bhf->bh', fq[:, :, i], k_state)[..., None]
                         .clamp_min(torch.finfo(q.dtype).tiny))
                # Sliding window softmax attention with sink tokens
                k_cache = torch.cat([k_cache, k[:, :, i:i + 1]], dim=2)[:, :, -self.window:]
                v_cache = torch.cat([v_cache, v[:, :, i:i + 1]], dim=2)[:, :, -self.window:]
                scores = (torch.einsum('bhd,bhnd->bhn', q[:, :, i], k_cache)
                          / math.sqrt(q.shape[-1]))
                y_awa = torch.einsum('bhn,bhnd->bhd', sink_softmax(scores, meta), v_cache)
                y.append(y_gla + alpha * y_awa)
            y = torch.stack(y, dim=2)
        cache.update(kv_state, k_state, k_cache, v_cache, self.layer_idx, q.shape[2])
        return y.to(dtype)


class LizardAttentionCache(LinearAttentionState):
    """
    Class for `past_key_values` when generating with Lizard attention
    -> GLA branch: gated "KV state" (b, h, f, head_dim) and "K state" (b, h, f), in (at least) fp32
    -> AWA branch: keys and values of the last `window` tokens
    """
    def __init__(self) -> None:
        super().__init__()
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []

    def update(self, kv_state: torch.Tensor, k_state: torch.Tensor,
               k_cache: torch.Tensor, v_cache: torch.Tensor,
               layer_idx: int, seq_len: int, **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Save a layer's states after it processes seq_len new tokens
        """
        if len(self.kv_states) <= layer_idx:  # Initializing states
            self.kv_states.append(kv_state)
            self.k_states.append(k_state)
            self.k_cache.append(k_cache)
            self.v_cache.append(v_cache)
        else:
            self.kv_states[layer_idx] = kv_state
            self.k_states[layer_idx] = k_state
            self.k_cache[layer_idx] = k_cache
            self.v_cache[layer_idx] = v_cache
        if layer_idx == 0:
            self._seen_tokens += seq_len
        while len(self._seen_tokens_by_layer) <= layer_idx:
            self._seen_tokens_by_layer.append(0)
        self._seen_tokens_by_layer[layer_idx] += seq_len
        return self.kv_states[layer_idx], self.k_states[layer_idx]
