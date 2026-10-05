"""
Lizard attention v3: math-aligned implementation.

Canonical/default configuration is intentionally simple and matches the
normalized parallel equations used in the accompanying math:

  - GLA: row-normalized gated linear attention
  - AWA: causal sliding-window softmax attention + sink logits exp(t_j)
  - Hedgehog: softmax(xW) || softmax(-xW)
  - one gate shared by all heads
  - one scalar alpha
  - alpha is NOT trainable by default (paper's listed trainables)
  - no RoPE in the Lizard branches

Experimental options remain available, but are explicitly opt-in.
"""

from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn

from transformers.cache_utils import Cache

from src.model.rotary import apply_rotary_pos_emb
from .linear_attention import softmax_attention
from .lizard_attention import (
    gate_products,
    window_mask,
    sink_softmax,
    upcast,
    LizardAttentionCache,
)


GLA_NORMS = ("row", "none", "joint", "hybrid")
FEATURE_ACTIVATIONS = ("softmax", "exp")
GLA_IMPLS = ("direct", "reparam")


class HeadwiseLinear(nn.Module):
    """One bias-free linear map per attention head."""

    def __init__(self, heads: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(heads, out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, h, l, d)
        return torch.einsum("bhld,hfd->bhlf", x, self.weight.to(x.dtype))


def hedgehog_v3(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation: str = "softmax",
) -> torch.Tensor:
    """
    Hedgehog feature map.

    Mathematical convention:
        z = x W^T
        phi(x) = act(z) || act(-z)

    If x is (b,h,l,d), W is either (f,d) or (h,f,d).
    The output dimension is therefore 2*f.
    """
    if activation not in FEATURE_ACTIVATIONS:
        raise ValueError(f"activation must be one of {FEATURE_ACTIVATIONS}")

    weight = weight.to(x.dtype)
    if weight.dim() == 2:
        z = x @ weight.T
    elif weight.dim() == 3:
        z = torch.einsum("bhld,hfd->bhlf", x, weight)
    else:
        raise ValueError(f"Unexpected feature-map weight rank: {weight.dim()}")

    if activation == "exp":
        return torch.cat([z.exp(), (-z).exp()], dim=-1)

    return torch.cat([z.softmax(-1), (-z).softmax(-1)], dim=-1)


def _gate_from_linear(
    x: torch.Tensor,
    W_gamma: nn.Linear,
) -> torch.Tensor:
    """
    Gate with an explicit row-vector convention:

        gamma[b,h,i] = sigmoid(sum_d x[b,i,d] W_gamma[h,d] + bias[h])

    This is exactly nn.Linear(hidden, heads_or_1) followed by transpose.
    """
    return torch.sigmoid(W_gamma(x)).transpose(1, 2)


def _cumulative_gate_log(gamma: torch.Tensor) -> torch.Tensor:
    """
    log C_t where C_t = prod_{j=1}^t gamma_j.

    gamma is strictly in (0,1) because it comes from sigmoid. We still clamp
    before log for numerical robustness.
    """
    tiny = torch.finfo(gamma.dtype).tiny
    return torch.log(gamma.clamp_min(tiny)).cumsum(-1)


def reparameterized_gla_weights(
    fq: torch.Tensor,
    fk: torch.Tensor,
    gamma: torch.Tensor,
    causal: torch.Tensor,
) -> torch.Tensor:
    """
    Dense reparameterized GLA weights.

    For C_t = prod_{j<=t} gamma_j and causal t <= i:

        C_i / C_t = prod_{j=t+1}^i gamma_j.

    Thus
        (phi(q_i) C_i) (phi(k_t) / C_t)^T

    recovers the gated kernel without explicitly materializing gate products.

    A per-(batch,head) constant shift in log C is used so the two scaling
    factors are balanced while their product remains unchanged.
    """
    log_c = _cumulative_gate_log(gamma)  # (b, g, l)

    # Balance the query/key exponent ranges. Adding the same scalar shift to
    # all key logs and subtracting it from all query logs leaves products
    # exactly unchanged.
    shift = 0.5 * (log_c.amin(-1, keepdim=True) + log_c.amax(-1, keepdim=True))
    q_log_scale = log_c - shift
    k_log_scale = -log_c + shift

    q_tilde = fq * q_log_scale.unsqueeze(-1).exp()
    k_tilde = fk * k_log_scale.unsqueeze(-1).exp()

    w = q_tilde @ k_tilde.transpose(-1, -2)
    return w.masked_fill(~causal, 0.0)


def joint_shift(scores: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Stable common shift for joint GLA+AWA normalization."""
    finite_scores = scores.masked_fill(~torch.isfinite(scores), 0).amax(-1, keepdim=True)
    return finite_scores.clamp_min(0).maximum(meta.max()).detach()


def hybrid_shift(scores: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Stable shift for hybrid normalization."""
    return scores.amax(-1, keepdim=True).maximum(meta.max()).detach()


def sink_softmax_v3(scores: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """
    Softmax over window logits plus meta-memory logits.

    The meta parameters t_j are logits, so their contribution is exp(t_j).
    Meta tokens contribute no value vectors.
    """
    m = torch.maximum(scores.amax(-1, keepdim=True), meta.max()).detach()
    e = (scores - m).exp()
    denom = (meta - m).exp().sum(-1, keepdim=True) + e.sum(-1, keepdim=True)
    return e / denom


class LolcatsLizardAttentionV3(nn.Module):
    """
    Math-aligned Lizard attention v3.

    IMPORTANT DIMENSION CONVENTION
    --------------------------------
    feature_dim is the number of features in EACH Hedgehog half.
    Therefore the concatenated feature dimension is 2 * feature_dim.

    With the default feature_dim=128:
        dim(phi_q) = dim(phi_k) = 256.

    This removes the former ambiguity instead of silently changing the model.
    """

    def __init__(
        self,
        base_attn: nn.Module,
        layer_idx: Optional[int] = None,
        max_layer_idx: Optional[int] = None,
        train_attention: bool = False,
        attention_type: str = "lolcats_llama_lizard_v3",
        window_size: int = 128,
        num_meta: int = 4,
        feature_dim: int = 128,
        gla_norm: str = "row",
        alpha_per_head: bool = False,
        alpha_init: float = 1.0,
        train_alpha: bool = False,
        feature_map_per_head: bool = False,
        feature_activation: str = "softmax",
        gate_per_head: bool = False,
        gate_bias_init: Optional[float] = None,
        window_rope: bool = False,
        gla_impl: str = "direct",
        **kwargs: any,
    ) -> None:
        super().__init__()

        if gla_norm not in GLA_NORMS:
            raise ValueError(f"gla_norm must be one of {GLA_NORMS}, not {gla_norm}")
        if feature_activation not in FEATURE_ACTIVATIONS:
            raise ValueError(
                f"feature_activation must be one of {FEATURE_ACTIVATIONS}, not {feature_activation}"
            )
        if gla_impl not in GLA_IMPLS:
            raise ValueError(f"gla_impl must be one of {GLA_IMPLS}, not {gla_impl}")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

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
        self.gla_impl = gla_impl
        self.window_rope = window_rope

        # Explicit dimension semantics: feature_dim = per-sign dimension;
        # final Hedgehog feature dimension is 2 * feature_dim.
        self.feature_dim_per_sign = feature_dim
        self.feature_dim_out = 2 * feature_dim

        # Reuse the original Llama projections.
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(base_attn, name))

        self.rotary_emb = getattr(base_attn, "rotary_emb", None)

        if feature_map_per_head:
            self.phi_q = HeadwiseLinear(self.heads, self.head_dim, feature_dim)
            self.phi_k = HeadwiseLinear(self.heads, self.head_dim, feature_dim)
        else:
            self.phi_q = nn.Linear(self.head_dim, feature_dim, bias=False)
            self.phi_k = nn.Linear(self.head_dim, feature_dim, bias=False)

        # W_gamma is stored in nn.Linear's conventional shape:
        # (gate_outputs, hidden). The math is x @ W_gamma^T.
        gate_outputs = self.heads if gate_per_head else 1
        self.W_gamma = nn.Linear(
            hidden,
            gate_outputs,
            bias=gate_bias_init is not None,
        )

        self.meta_tokens = nn.Parameter(torch.zeros(num_meta))

        alpha = alpha_init * torch.ones(self.heads if alpha_per_head else ())
        if train_alpha:
            self.alpha_blend = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha_blend", alpha)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: any,
    ) -> Tuple[torch.Tensor, Optional[tuple], Optional[Cache]]:
        x = hidden_states
        q = self.split(self.q_proj(x), self.heads)
        k = self.split(self.k_proj(x), self.kv_heads)
        v = self.split(self.v_proj(x), self.kv_heads)

        # Explicit row-vector gate equation:
        # gamma[b,h,i] = sigmoid(x[b,i,:] @ W_gamma[h,:]^T + b[h]).
        gamma = _gate_from_linear(x, self.W_gamma)

        rope = None
        if self.train_attention or self.window_rope:
            rope = (
                position_embeddings
                if position_embeddings is not None
                else self.rotary_emb(v, position_ids)
            )

        attn_weights = None
        if self.train_attention:
            with torch.no_grad():
                q_rope, k_rope = apply_rotary_pos_emb(q, k, *rope)
                y_true = softmax_attention(q_rope, k_rope, v)[0]

            y_pred = self.lizard(q, k, v, gamma, rope)
            attn_weights = ((None, None), (y_pred, y_true))
            y = y_true
        elif isinstance(past_key_value, LizardAttentionCache):
            y = self.lizard_recurrent(q, k, v, gamma, past_key_value, rope)
        else:
            y = self.lizard(q, k, v, gamma, rope)

        return self.o_proj(y.transpose(1, 2).flatten(2)), attn_weights, past_key_value

    def feature_maps(self, q: torch.Tensor, k: torch.Tensor):
        """Return separate query/key Hedgehog maps."""
        return (
            hedgehog_v3(q, self.phi_q.weight, self.feature_activation),
            hedgehog_v3(k, self.phi_k.weight, self.feature_activation),
        )

    def lizard_params(self, dtype: torch.dtype):
        """Return alpha and sink logits in the requested dtype."""
        alpha = self.alpha_blend.to(dtype)
        if alpha.dim() == 1:
            alpha = alpha[None, :, None, None]

        if self.gla_norm == "joint":
            # Joint normalization is a weighted-mean construction.
            alpha = alpha.clamp_min(0)
        elif self.gla_norm == "hybrid":
            alpha = alpha.abs()

        return alpha, self.meta_tokens.to(dtype)

    def window_inputs(self, q: torch.Tensor, k: torch.Tensor, rope):
        """Optionally apply RoPE only to the AWA queries/keys."""
        if not self.window_rope:
            return q, k
        if rope is None:
            raise ValueError("rope is required when window_rope=True")
        cos, sin = rope
        return apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))

    def _direct_gla_weights(
        self,
        fq: torch.Tensor,
        fk: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        causal = window_mask(fq.shape[-2], fq.shape[-2], fq.device)
        w = fq @ fk.transpose(-1, -2)
        return w * gate_products(gamma) * causal

    def _reparam_gla_weights(
        self,
        fq: torch.Tensor,
        fk: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        if self.feature_activation != "exp":
            raise ValueError(
                "gla_impl='reparam' is reserved for feature_activation='exp' "
                "because that is the explicit Section-4 Hedgehog formulation."
            )
        causal = window_mask(fq.shape[-2], fq.shape[-2], fq.device)
        return reparameterized_gla_weights(fq, fk, gamma, causal)

    def branch_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        gamma: torch.Tensor,
        rope=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense GLA and AWA weights for XAI/inspection."""
        q, k, gamma = upcast(q, k, gamma)
        alpha, meta = self.lizard_params(q.dtype)
        fq, fk = self.feature_maps(q, k)

        if self.gla_impl == "direct":
            w = self._direct_gla_weights(fq, fk, gamma)
        else:
            w = self._reparam_gla_weights(fq, fk, gamma)

        qw, kw = self.window_inputs(q, k, rope)
        mask = window_mask(q.shape[-2], self.window, q.device)
        scores = (
            qw @ kw.transpose(-1, -2) / math.sqrt(q.shape[-1])
        ).masked_fill(~mask, float("-inf"))

        if self.gla_norm == "joint":
            m = joint_shift(scores, meta)
            den = (
                (-m).exp() * w.sum(-1, keepdim=True)
                + alpha
                * (
                    (scores - m).exp().sum(-1, keepdim=True)
                    + (meta - m).exp().sum(-1, keepdim=True)
                )
            )
            return (-m).exp() * w / den, alpha * (scores - m).exp() / den

        if self.gla_norm == "hybrid":
            m = hybrid_shift(scores, meta)
            den = (
                w.sum(-1, keepdim=True)
                + alpha
                * (
                    (scores - m).exp().sum(-1, keepdim=True)
                    + (meta - m).exp().sum(-1, keepdim=True)
                )
            )
            return w / den, alpha * (scores - m).exp() / den

        if self.gla_norm == "row":
            w = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)

        # `none` gives the exact unnormalized matrix/recurrent formulation.
        return w, alpha * sink_softmax_v3(scores, meta)

    def lizard(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gamma: torch.Tensor,
        rope=None,
    ) -> torch.Tensor:
        """
        Full-sequence Lizard attention.

        Default (`gla_norm='row'`) is:
            y = normalized_GLA + alpha * AWA
        """
        dtype = q.dtype
        q, k, v, gamma = upcast(q, k, v, gamma)
        alpha, meta = self.lizard_params(q.dtype)
        fq, fk = self.feature_maps(q, k)

        if self.gla_impl == "direct":
            w = self._direct_gla_weights(fq, fk, gamma)
        else:
            w = self._reparam_gla_weights(fq, fk, gamma)

        qw, kw = self.window_inputs(q, k, rope)
        mask = window_mask(q.shape[-2], self.window, q.device)
        scores = (
            qw @ kw.transpose(-1, -2) / math.sqrt(q.shape[-1])
        ).masked_fill(~mask, float("-inf"))

        if self.gla_norm == "joint":
            m = joint_shift(scores, meta)
            e = (scores - m).exp()
            num = (-m).exp() * (w @ v) + alpha * (e @ v)
            den = (
                (-m).exp() * w.sum(-1, keepdim=True)
                + alpha
                * (
                    e.sum(-1, keepdim=True)
                    + (meta - m).exp().sum(-1, keepdim=True)
                )
            )
            y = num / den

        elif self.gla_norm == "hybrid":
            m = hybrid_shift(scores, meta)
            e = (scores - m).exp()
            num = w @ v + alpha * (e @ v)
            den = (
                w.sum(-1, keepdim=True)
                + alpha
                * (
                    e.sum(-1, keepdim=True)
                    + (meta - m).exp().sum(-1, keepdim=True)
                )
            )
            y = num / den

        else:
            y_gla = w @ v
            if self.gla_norm == "row":
                y_gla = y_gla / w.sum(-1, keepdim=True).clamp_min(
                    torch.finfo(w.dtype).tiny
                )
            y = y_gla + alpha * (sink_softmax_v3(scores, meta) @ v)

        return y.to(dtype)

    def lizard_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gamma: torch.Tensor,
        cache: LizardAttentionCache,
        rope=None,
    ) -> torch.Tensor:
        """
        Cached generation path.

        The recurrent GLA equations are explicitly:

            S_i = gamma_i S_{i-1} + phi_k(k_i) v_i^T
            z_i = gamma_i z_{i-1} + phi_k(k_i)

            y_gla = phi_q(q_i)^T S_i / phi_q(q_i)^T z_i  (row norm)

        or

            y_gla = phi_q(q_i)^T S_i                       (none)

        Joint/hybrid keep their combined denominators.
        """
        dtype = q.dtype
        q, k, v, gamma = upcast(q, k, v, gamma)
        alpha, meta = self.lizard_params(q.dtype)
        fq, fk = self.feature_maps(q, k)
        qw, kw = self.window_inputs(q, k, rope)
        tiny = torch.finfo(q.dtype).tiny

        if len(cache.kv_states) <= self.layer_idx:
            # Prefill is computed by the exact same full-sequence equations.
            y = self.lizard(q, k, v, gamma, rope)

            decay = torch.cat(
                [
                    gamma[..., 1:].flip(-1).cumprod(-1).flip(-1),
                    torch.ones_like(gamma[..., :1]),
                ],
                dim=-1,
            )
            decay = decay.expand(-1, self.heads, -1)

            kv_state = torch.einsum(
                "bhl,bhlf,bhld->bhfd", decay, fk, v
            )
            k_state = torch.einsum(
                "bhl,bhlf->bhf", decay, fk
            )
            k_cache, v_cache = kw[:, :, -self.window:], v[:, :, -self.window:]

        else:
            kv_state = cache.kv_states[self.layer_idx]
            k_state = cache.k_states[self.layer_idx]
            k_cache = cache.k_cache[self.layer_idx]
            v_cache = cache.v_cache[self.layer_idx]
            y = []

            for i in range(q.shape[2]):
                # Recurrent GLA state equations.
                g = gamma[:, :, i, None]
                kv_state = (
                    g[..., None] * kv_state
                    + torch.einsum(
                        "bhf,bhd->bhfd", fk[:, :, i], v[:, :, i]
                    )
                )
                k_state = g * k_state + fk[:, :, i]

                num_gla = torch.einsum(
                    "bhf,bhfd->bhd", fq[:, :, i], kv_state
                )
                den_gla = torch.einsum(
                    "bhf,bhf->bh", fq[:, :, i], k_state
                )[..., None]

                # AWA cache.
                k_cache = torch.cat(
                    [k_cache, kw[:, :, i : i + 1]], dim=2
                )[:, :, -self.window :]
                v_cache = torch.cat(
                    [v_cache, v[:, :, i : i + 1]], dim=2
                )[:, :, -self.window :]

                scores = torch.einsum(
                    "bhd,bhnd->bhn", qw[:, :, i], k_cache
                ) / math.sqrt(q.shape[-1])

                a = alpha[..., 0] if alpha.dim() else alpha

                if self.gla_norm == "joint":
                    m = joint_shift(scores, meta)
                    e = (scores - m).exp()
                    num = (
                        (-m).exp() * num_gla
                        + a * torch.einsum("bhn,bhnd->bhd", e, v_cache)
                    )
                    den = (
                        (-m).exp() * den_gla
                        + a
                        * (
                            e.sum(-1, keepdim=True)
                            + (meta - m).exp().sum(-1, keepdim=True)
                        )
                    )
                    y.append(num / den)

                elif self.gla_norm == "hybrid":
                    m = hybrid_shift(scores, meta)
                    e = (scores - m).exp()
                    num = num_gla + a * torch.einsum(
                        "bhn,bhnd->bhd", e, v_cache
                    )
                    den = den_gla + a * (
                        e.sum(-1, keepdim=True)
                        + (meta - m).exp().sum(-1, keepdim=True)
                    )
                    y.append(num / den)

                else:
                    if self.gla_norm == "row":
                        y_gla = num_gla / den_gla.clamp_min(tiny)
                    else:
                        y_gla = num_gla

                    y_awa = torch.einsum(
                        "bhn,bhnd->bhd",
                        sink_softmax_v3(scores, meta),
                        v_cache,
                    )
                    y.append(y_gla + a * y_awa)

            y = torch.stack(y, dim=2)

        cache.update(
            kv_state,
            k_state,
            k_cache,
            v_cache,
            self.layer_idx,
            q.shape[2],
        )
        return y.to(dtype)
