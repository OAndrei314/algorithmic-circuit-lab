"""A minimal, hand-rolled 1-layer transformer, built so every intermediate
tensor needed for interpretability (per-head attention outputs, per-neuron
MLP activations) is directly addressable rather than hidden inside opaque
``nn.MultiheadAttention``/``nn.TransformerEncoderLayer`` modules.

Deliberately has no LayerNorm. That is not an oversight: LayerNorm makes the
"direct logit contribution" of a component depend on every other component
through the normalization statistics, which breaks the clean additive
decomposition used in ``circuit_lab.interp``. Dropping it (following Nanda
et al., "Progress measures for grokking via mechanistic interpretability")
costs a little optimization stability but keeps the circuit analysis exact.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    n_ctx: int = 3
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    seed: int = 0

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class OneLayerTransformer(nn.Module):
    """Embed -> 1 attention block -> 1 MLP block -> unembed, no LayerNorm."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        d_model, n_heads, d_head, d_mlp = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp

        def init(*shape):
            return nn.Parameter(torch.randn(*shape, generator=g) / shape[-1] ** 0.5)

        self.W_E = init(cfg.vocab_size, d_model)
        self.W_pos = init(cfg.n_ctx, d_model)

        self.W_Q = init(n_heads, d_model, d_head)
        self.W_K = init(n_heads, d_model, d_head)
        self.W_V = init(n_heads, d_model, d_head)
        self.W_O = init(n_heads, d_head, d_model)

        self.W_in = init(d_model, d_mlp)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = init(d_mlp, d_model)
        self.b_out = nn.Parameter(torch.zeros(d_model))

        self.W_U = init(d_model, cfg.vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        ablate_heads: Optional[list] = None,
        ablate_neurons: Optional[list] = None,
        mean_z: Optional[torch.Tensor] = None,
        mean_mlp_post: Optional[torch.Tensor] = None,
        return_cache: bool = False,
    ):
        """Run the model. ``tokens`` is (batch, n_ctx) int64.

        ``ablate_heads`` / ``ablate_neurons`` are lists of indices whose
        output is replaced before it is combined back into the residual
        stream -- with zeros, unless ``mean_z`` / ``mean_mlp_post`` (means
        computed over some reference distribution, typically the training
        set) are supplied, in which case mean-ablation is used instead.
        Mean-ablation is the more careful choice: zero-ablation also removes
        the component's *average* contribution, which can make an
        unimportant-but-nonzero-mean component look load-bearing.
        """
        cfg = self.cfg
        batch, n_ctx = tokens.shape
        cache = {}

        resid = self.W_E[tokens] + self.W_pos[:n_ctx]
        cache["resid_pre"] = resid

        q = torch.einsum("bpd,hde->bphe", resid, self.W_Q)
        k = torch.einsum("bpd,hde->bphe", resid, self.W_K)
        v = torch.einsum("bpd,hde->bphe", resid, self.W_V)

        attn_scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / (cfg.d_head**0.5)
        attn_pattern = attn_scores.softmax(dim=-1)
        z = torch.einsum("bhqk,bkhd->bqhd", attn_pattern, v)  # (batch, n_ctx, n_heads, d_head)
        cache["attn_pattern"] = attn_pattern
        cache["z"] = z

        z = z.clone()
        if ablate_heads:
            for h in ablate_heads:
                if mean_z is not None:
                    z[:, :, h, :] = mean_z[:, h, :]
                else:
                    z[:, :, h, :] = 0.0

        attn_out = torch.einsum("bqhd,hdm->bqm", z, self.W_O)
        cache["attn_out"] = attn_out
        resid_mid = resid + attn_out
        cache["resid_mid"] = resid_mid

        mlp_pre = torch.einsum("bpd,dm->bpm", resid_mid, self.W_in) + self.b_in
        mlp_post = nn.functional.gelu(mlp_pre)
        cache["mlp_pre"] = mlp_pre
        cache["mlp_post"] = mlp_post

        mlp_post = mlp_post.clone()
        if ablate_neurons:
            for i in ablate_neurons:
                if mean_mlp_post is not None:
                    mlp_post[:, :, i] = mean_mlp_post[:, i]
                else:
                    mlp_post[:, :, i] = 0.0

        mlp_out = torch.einsum("bpm,md->bpd", mlp_post, self.W_out) + self.b_out
        cache["mlp_out"] = mlp_out
        resid_final = resid_mid + mlp_out
        cache["resid_final"] = resid_final

        logits = torch.einsum("bpd,dv->bpv", resid_final, self.W_U)
        cache["logits"] = logits

        if return_cache:
            return logits, cache
        return logits
