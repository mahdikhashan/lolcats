"""
Lizard attention v2 (https://arxiv.org/abs/2507.09025, version 4)

The same layer as lizard_attention.py (v1), with options for the readings of the paper and for
the code changes C1-C6 of docs/12-gap-analysis-2.md. The default options give the outputs of v1,
so each experiment changes one option (docs/13-lizard-attention-v2.md):

- gla_norm ('row'): normalization of the gated branch (D1, C3)
    'row'   - one denominator for the gated branch: the parallel form of Section 3.1 (v1)
    'none'  - no denominator: the recurrent form of Section 3.1 and the matrix form of Section 4
    'joint' - one denominator for both branches and the sinks, as in the LoLCATs hybrid attention.
              Each row is a weighted mean of values, with the sinks absorbing weight (needs alpha >= 0)
- alpha_per_head (False): one window weight alpha for each head (C1, the LoLCATs default)
- train_alpha (True): train alpha. The paper lists only phi, W_gamma and t as learnable
- feature_map_per_head (False): one Hedgehog map for each head (D2, C2, the LoLCATs default)
- feature_activation ('softmax'): 'softmax' (Table 13, v1) or 'exp' (Section 4) (D3)
- gate_per_head (False): one gate for each head (C4). The paper uses one gate for all heads
- gate_bias_init (None): add a bias to W_gamma with this start value, e.g. 3.0 for gamma ~ 0.95 (C6)
- window_rope (False): RoPE on the queries and keys of the window branch only (C5)

As in v1: the sink logits t_j enter the window denominator as exp(t_j) (D5), the gated branch
has no RoPE, and the Lizard calculations run in (at least) fp32.
"""
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn

from transformers.cache_utils import Cache

from src.model.rotary import apply_rotary_pos_emb
from .linear_attention import softmax_attention
from .lizard_attention import gate_products, window_mask, sink_softmax, upcast, LizardAttentionCache


GLA_NORMS = ('row', 'none', 'joint')
FEATURE_ACTIVATIONS = ('softmax', 'exp')


class HeadwiseLinear(nn.Module):
    """
    One bias-free linear map for each head, so the weight keeps the name `weight`
    -> weight is shape (heads, out_features, in_features), like stacked nn.Linear weights
    """
    def __init__(self, heads: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(heads, out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x is (b, h, l, in_features)
        return torch.einsum('bhld,hfd->bhlf', x, self.weight.to(x.dtype))


def hedgehog_v2(x: torch.Tensor, weight: torch.Tensor, activation: str = 'softmax') -> torch.Tensor:
    """
    [act(xW) (+) act(-xW)] with act = softmax over the features (Table 13) or exp (Section 4)
    -> weight is (f, d) for one map, or (h, f, d) for one map for each head; x is (b, h, l, d)
    """
    weight = weight.to(x.dtype)
    xw = x @ weight.T if weight.dim() == 2 else torch.einsum('bhld,hfd->bhlf', x, weight)
    if activation == 'exp':
        return torch.cat([xw.exp(), (-xw).exp()], dim=-1)
    return torch.cat([xw.softmax(-1), (-xw).softmax(-1)], dim=-1)


def joint_shift(scores: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """
    max(0, max of the finite window scores, max sink logit) over the last dim, detached
    -> Dividing every weight of a row by exp(shift) keeps exp() of scores and sinks <= 1,
       and multiplies the gated weights by exp(-shift) <= 1
    """
    m = scores.masked_fill(~torch.isfinite(scores), 0).amax(-1, keepdim=True)
    return m.clamp_min(0).maximum(meta.max()).detach()


class LolcatsLizardAttentionV2(nn.Module):
    """
    Lizard attention v2, initialized from a `LlamaAttention` object (base_attn), with the interface
    of `LolcatsLizardAttention` (v1)
    - Reuses base_attn's q, k, v, o projections; only the Lizard parameters are new
    - If train_attention, the layer outputs the original softmax attention and returns
      ((None, None), (y_pred, y_true)) as attention weights for the distillation trainer
    - Generation uses `LizardAttentionCache`, as v1
    - The Lizard parameters keep the names of v1 (phi_q.weight, phi_k.weight, W_gamma.weight,
      meta_tokens, alpha_blend); W_gamma.bias exists only with gate_bias_init
    """
    def __init__(self,
                 base_attn: nn.Module,  # like LlamaAttention
                 layer_idx: Optional[int] = None,
                 max_layer_idx: Optional[int] = None,
                 train_attention: Optional[bool] = False,
                 attention_type: Optional[str] = 'lolcats_llama_lizard_v2',
                 window_size: int = 128,
                 num_meta: int = 4,
                 feature_dim: int = 128,
                 gla_norm: str = 'row',
                 alpha_per_head: bool = False,
                 alpha_init: float = 1.0,
                 train_alpha: bool = True,
                 feature_map_per_head: bool = False,
                 feature_activation: str = 'softmax',
                 gate_per_head: bool = False,
                 gate_bias_init: Optional[float] = None,
                 window_rope: bool = False,
                 **kwargs: any) -> None:
        super().__init__()
        assert gla_norm in GLA_NORMS, f'gla_norm must be one of {GLA_NORMS}, not {gla_norm}'
        assert feature_activation in FEATURE_ACTIVATIONS, \
            f'feature_activation must be one of {FEATURE_ACTIVATIONS}, not {feature_activation}'
        config = base_attn.config
        hidden = config.hidden_size
        self.layer_idx = layer_idx if layer_idx is not None else base_attn.layer_idx
        self.max_layer_idx = max_layer_idx
        self.train_attention = train_attention
        self.attention_type = attention_type
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = hidden // self.heads
        self.window = window_size
        self.gla_norm = gla_norm
        self.feature_activation = feature_activation
        self.window_rope = window_rope

        # Copy original model projection layers
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, getattr(base_attn, name))
        # For the "ground-truth" softmax attention and the window RoPE, if position_embeddings
        # are not passed in
        self.rotary_emb = getattr(base_attn, 'rotary_emb', None)

        # Lizard parameters (v1 start values: phi ~ N(0, 0.02), W_gamma = 0, t = 0, alpha = 1)
        if feature_map_per_head:  # One map for each query head; keys are repeated to all heads
            self.phi_q = HeadwiseLinear(self.heads, self.head_dim, feature_dim)
            self.phi_k = HeadwiseLinear(self.heads, self.head_dim, feature_dim)
        else:
            self.phi_q = nn.Linear(self.head_dim, feature_dim, bias=False)
            self.phi_k = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.W_gamma = nn.Linear(hidden, self.heads if gate_per_head else 1,
                                 bias=gate_bias_init is not None)
        self.meta_tokens = nn.Parameter(torch.zeros(num_meta))
        alpha = alpha_init * torch.ones(self.heads if alpha_per_head else ())
        if train_alpha:
            self.alpha_blend = nn.Parameter(alpha)
        else:  # A buffer: not trained and not in the checkpoint of trainable weights
            self.register_buffer('alpha_blend', alpha)
        nn.init.normal_(self.phi_q.weight, std=0.02)
        nn.init.normal_(self.phi_k.weight, std=0.02)
        nn.init.zeros_(self.W_gamma.weight)
        if gate_bias_init is not None:
            nn.init.constant_(self.W_gamma.bias, gate_bias_init)

        weight = self.q_proj.weight
        self.to(device=weight.device, dtype=weight.dtype)

    def split(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.view(b, n, heads, self.head_dim).transpose(1, 2)
        return x.repeat_interleave(self.heads // heads, dim=1)

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
        # One gate for all heads (b, 1, l), or one for each head (b, h, l)
        gamma = torch.sigmoid(*upcast(self.W_gamma(x))).transpose(1, 2)

        rope = None
        if self.train_attention or self.window_rope:
            rope = (position_embeddings if position_embeddings is not None
                    else self.rotary_emb(v, position_ids))

        attn_weights = None
        if self.train_attention:
            # 1. Compute "ground-truth" outputs of the original softmax attention (with RoPE)
            with torch.no_grad():
                q_rope, k_rope = apply_rotary_pos_emb(q, k, *rope)
                y_true = softmax_attention(q_rope, k_rope, v)[0]
            # 2. Compute "predicted" Lizard outputs
            y_pred = self.lizard(q, k, v, gamma, rope)
            attn_weights = ((None, None), (y_pred, y_true))
            y = y_true  # Next layer sees the original model's hidden states
        elif isinstance(past_key_value, LizardAttentionCache):  # Generating
            y = self.lizard_recurrent(q, k, v, gamma, past_key_value, rope)
        else:  # Training, or generating without a cache
            y = self.lizard(q, k, v, gamma, rope)
        return self.o_proj(y.transpose(1, 2).flatten(2)), attn_weights, past_key_value

    # -------------
    # Lizard parts
    # -------------
    def feature_maps(self, q: torch.Tensor, k: torch.Tensor):
        """Hedgehog feature maps in q's dtype"""
        return (hedgehog_v2(q, self.phi_q.weight, self.feature_activation),
                hedgehog_v2(k, self.phi_k.weight, self.feature_activation))

    def lizard_params(self, dtype: torch.dtype):
        """alpha (shape () or (1, h, 1, 1)) and the sink logits, in dtype"""
        alpha = self.alpha_blend.to(dtype)
        if alpha.dim() == 1:
            alpha = alpha[None, :, None, None]
        if self.gla_norm == 'joint':  # The joint denominator needs alpha >= 0 (docs/13, P2)
            alpha = alpha.clamp_min(0)
        return alpha, self.meta_tokens.to(dtype)

    def window_inputs(self, q: torch.Tensor, k: torch.Tensor, rope) -> tuple:
        """Queries and keys of the window branch: with RoPE if window_rope (C5)"""
        if not self.window_rope:
            return q, k
        cos, sin = rope
        return apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))

    def branch_weights(self, q: torch.Tensor, k: torch.Tensor, gamma: torch.Tensor, rope=None):
        """
        Dense weights of the two branches, in (at least) fp32
        -> a_gla, a_win are shape (b, h, l, l), and the output is (a_gla + a_win) @ v
        -> Also for XAI scripts: the weights of each branch, with alpha in a_win
        """
        q, k, gamma = upcast(q, k, gamma)
        alpha, meta = self.lizard_params(q.dtype)
        fq, fk = self.feature_maps(q, k)
        w = (fq @ fk.transpose(-1, -2)) * gate_products(gamma)  # gamma (b, 1 or h, l)
        qw, kw = self.window_inputs(q, k, rope)
        mask = window_mask(q.shape[-2], self.window, q.device)
        scores = (qw @ kw.transpose(-1, -2) / math.sqrt(q.shape[-1])).masked_fill(~mask, float('-inf'))
        if self.gla_norm == 'joint':
            m = joint_shift(scores, meta)
            den = ((-m).exp() * w.sum(-1, keepdim=True)
                   + alpha * ((scores - m).exp().sum(-1, keepdim=True)
                              + (meta - m).exp().sum(-1, keepdim=True)))
            return (-m).exp() * w / den, alpha * (scores - m).exp() / den
        if self.gla_norm == 'row':
            w = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
        return w, alpha * sink_softmax(scores, meta)

    def lizard(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               gamma: torch.Tensor, rope=None) -> torch.Tensor:
        """
        Lizard attention over whole sequences, computed in (at least) fp32
        -> q, k, v are shape (b, h, l, head_dim); gamma is shape (b, 1 or h, l)
        """
        dtype = q.dtype
        q, k, v, gamma = upcast(q, k, v, gamma)
        alpha, meta = self.lizard_params(q.dtype)
        fq, fk = self.feature_maps(q, k)
        w = (fq @ fk.transpose(-1, -2)) * gate_products(gamma)
        qw, kw = self.window_inputs(q, k, rope)
        mask = window_mask(q.shape[-2], self.window, q.device)
        scores = (qw @ kw.transpose(-1, -2) / math.sqrt(q.shape[-1])).masked_fill(~mask, float('-inf'))
        if self.gla_norm == 'joint':
            m = joint_shift(scores, meta)
            e = (scores - m).exp()
            num = (-m).exp() * (w @ v) + alpha * (e @ v)
            den = ((-m).exp() * w.sum(-1, keepdim=True)
                   + alpha * (e.sum(-1, keepdim=True) + (meta - m).exp().sum(-1, keepdim=True)))
            y = num / den
        else:
            y_gla = w @ v
            if self.gla_norm == 'row':  # Guard 0 / 0 when every weight in a row underflows (D8)
                y_gla = y_gla / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
            y = y_gla + alpha * (sink_softmax(scores, meta) @ v)
        return y.to(dtype)

    def lizard_recurrent(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         gamma: torch.Tensor, cache: LizardAttentionCache, rope=None) -> torch.Tensor:
        """
        Lizard attention when generating
        -> Prefill: attention over the prompt, then save the gated states and the last window of
           keys (with RoPE if window_rope) and values
        -> Decode: update the gated states and the window one token at a time (recurrent form)
        """
        dtype = q.dtype
        q, k, v, gamma = upcast(q, k, v, gamma)
        alpha, meta = self.lizard_params(q.dtype)
        fq, fk = self.feature_maps(q, k)
        qw, kw = self.window_inputs(q, k, rope)
        tiny = torch.finfo(q.dtype).tiny
        if len(cache.kv_states) <= self.layer_idx:  # Prefill
            y = self.lizard(q, k, v, gamma, rope)
            # decay[t] = prod_{s > t} gamma[s], i.e., the last row of gate_products(gamma)
            decay = torch.cat([gamma[..., 1:].flip(-1).cumprod(-1).flip(-1),
                               torch.ones_like(gamma[..., :1])], dim=-1)
            decay = decay.expand(-1, self.heads, -1)
            kv_state = torch.einsum('bhl,bhlf,bhld->bhfd', decay, fk, v)
            k_state = torch.einsum('bhl,bhlf->bhf', decay, fk)
            k_cache, v_cache = kw[:, :, -self.window:], v[:, :, -self.window:]
        else:  # Decode
            kv_state = cache.kv_states[self.layer_idx]
            k_state = cache.k_states[self.layer_idx]
            k_cache = cache.k_cache[self.layer_idx]
            v_cache = cache.v_cache[self.layer_idx]
            y = []
            for i in range(q.shape[2]):
                # Gated branch: S_i = gamma_i S_{i-1} + phi_k(k_i) v_i^T, and z_i for denominators
                g = gamma[:, :, i, None]  # b, 1 or h, 1
                kv_state = (g[..., None] * kv_state
                            + torch.einsum('bhf,bhd->bhfd', fk[:, :, i], v[:, :, i]))
                k_state = g * k_state + fk[:, :, i]
                num_gla = torch.einsum('bhf,bhfd->bhd', fq[:, :, i], kv_state)
                den_gla = torch.einsum('bhf,bhf->bh', fq[:, :, i], k_state)[..., None]
                # Window branch over the last `window` keys (rotated at their own positions)
                k_cache = torch.cat([k_cache, kw[:, :, i:i + 1]], dim=2)[:, :, -self.window:]
                v_cache = torch.cat([v_cache, v[:, :, i:i + 1]], dim=2)[:, :, -self.window:]
                scores = (torch.einsum('bhd,bhnd->bhn', qw[:, :, i], k_cache)
                          / math.sqrt(q.shape[-1]))
                a = alpha[..., 0] if alpha.dim() else alpha  # (1, h, 1) or ()
                if self.gla_norm == 'joint':
                    m = joint_shift(scores, meta)
                    e = (scores - m).exp()
                    num = (-m).exp() * num_gla + a * torch.einsum('bhn,bhnd->bhd', e, v_cache)
                    den = ((-m).exp() * den_gla
                           + a * (e.sum(-1, keepdim=True) + (meta - m).exp().sum(-1, keepdim=True)))
                    y.append(num / den)
                else:
                    y_gla = num_gla / den_gla.clamp_min(tiny) if self.gla_norm == 'row' else num_gla
                    y_awa = torch.einsum('bhn,bhnd->bhd', sink_softmax(scores, meta), v_cache)
                    y.append(y_gla + a * y_awa)
            y = torch.stack(y, dim=2)
        cache.update(kv_state, k_state, k_cache, v_cache, self.layer_idx, q.shape[2])
        return y.to(dtype)
