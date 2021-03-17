#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    """Attention Based Decoder, compute log probabilities between `query` and `key`.

    Attributes:
        query_dim
        embed_dim
        num_heads
        bias
    """

    def __init__(
        self, query_dim: int, embed_dim: int, num_heads: int = 1, bias: bool = True, tanh_clipping: int = 10.0,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.tanh_clipping = tanh_clipping
        self._precompute = False
        self.query_proj = nn.Linear(self.query_dim, self.embed_dim, bias=self.bias)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        precomputed_k: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - log_prob: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        num_heads = self.num_heads
        tgt_len, bsz, query_dim = query.size()
        assert query_dim == self.query_dim
        assert precomputed_k is not None or key is not None, f"Keys need to be input or precompute"
        if key is None:
            src_len, _, embed_dim = precomputed_k.size()
        else:
            src_len, _, embed_dim = key.size()
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        assert embed_dim == self.embed_dim
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        q = self.query_proj(query)
        if precomputed_k is not None:
            k = precomputed_k
        else:
            k = self.key_proj(key)
        q = q * scaling

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, "Only bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, tgt_len, src_len]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    tgt_len,
                    src_len,
                ]:
                    if attn_mask.size(0) == bsz:
                        attn_mask = attn_mask.repeat_interleave(num_heads, 0)
                    else:
                        raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if self.tanh_clipping > 0:
            attn_output_weights = torch.tanh(attn_output_weights) * self.tanh_clipping

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))

        log_prob = F.log_softmax(attn_output_weights, dim=-1)
        log_prob = log_prob.view(bsz, num_heads, tgt_len, src_len).sum(dim=1) / num_heads

        return log_prob.squeeze(1)

    def precompute_keys(self, key: torch.Tensor) -> None:
        assert key.size(-1) == self.embed_dim
        precomputed_k = self.key_proj(key)
        return precomputed_k
