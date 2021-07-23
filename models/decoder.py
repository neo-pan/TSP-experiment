import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import wandb


class AttentionDecoder(nn.Module):
    """Attention Based Decoder, compute log probabilities between `query` and `key`.

    Attributes:
        query_dim
        embed_dim
        num_heads
        bias
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, bias: bool = True, tanh_clipping: int = 10.0,) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.tanh_clipping = tanh_clipping
        self.glimpse_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)

    def forward(
        self,
        query: torch.Tensor,
        glimpse_K: torch.Tensor,
        glimpse_V: torch.Tensor,
        logit_K: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - glimpse_K: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - glimpse_V: :math:`(S, N, E)`
            - logit_K: :math:`(N, S, E)`
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
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        src_len, _, embed_dim = glimpse_K.size()
        assert embed_dim == self.embed_dim
        assert list(glimpse_K.size()) == list(glimpse_V.size())
        assert list(logit_K.size()) == [bsz, src_len, embed_dim], f"{logit_K.size()} - {[bsz, src_len, embed_dim]}"
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, "Only bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            assert attn_mask.dim() == 3, "attn_mask's dimension {} is not supported".format(attn_mask.dim())
            if list(attn_mask.size()) == [
                bsz,
                tgt_len,
                src_len,
            ]:
                heads_mask = attn_mask.expand(num_heads, *attn_mask.size())
            else:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")

        # (n_heads, batch_size, target_len, head_dim)
        glimpse_Q = query.view(tgt_len, bsz, num_heads, head_dim).permute(2, 1, 0, 3)
        # (n_heads, batch_size, source_len, head_dim)
        glimpse_K = glimpse_K.view(src_len, bsz, num_heads, head_dim).permute(2, 1, 0, 3)
        glimpse_V = glimpse_V.view(src_len, bsz, num_heads, head_dim).permute(2, 1, 0, 3)

        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        assert list(compatibility.size()) == [num_heads, bsz, tgt_len, src_len]

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            compatibility.masked_fill_(heads_mask, torch.finfo(compatibility.dtype).min)

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        assert list(heads.size()) == [num_heads, bsz, tgt_len, head_dim]

        glimpse = self.glimpse_proj(heads.permute(1, 2, 0, 3).contiguous().view(bsz, tgt_len, embed_dim))

        final_Q = glimpse
        assert list(final_Q.size()) == [bsz, tgt_len, embed_dim]
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)) / math.sqrt(final_Q.size(-1))
        assert list(logits.size()) == [bsz, tgt_len, src_len]

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            logits.masked_fill_(attn_mask, torch.finfo(logits.dtype).min)

        log_prob = F.log_softmax(logits, dim=-1)
        if torch.isnan(log_prob).any():
            torch.save(
                {
                "glimpse_Q": glimpse_Q,
                "glimpse_K": glimpse_K,
                "glimpse_V": glimpse_V,
                "compatibility": compatibility,
                "heads": heads,
                "final_Q": final_Q,
                "logits": logits,
                }, "nan-tensor.pt")
            assert not torch.isnan(log_prob).any()

        return log_prob.squeeze(1)


class SimpleDecoder(nn.Module):
    """Attention Based Decoder, compute log probabilities between `query` and `key`.

    Attributes:
        query_dim
        embed_dim
        num_heads
        bias
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, bias: bool = True, tanh_clipping: int = 10.0,) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.tanh_clipping = tanh_clipping

    def forward(self, query: torch.Tensor, key: torch.Tensor, attn_mask: torch.Tensor = None,) -> torch.Tensor:
        r"""
        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - glimpse_K: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - glimpse_V: :math:`(S, N, E)`
            - logit_K: :math:`(N, S, E)`
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
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        src_len, _, embed_dim = key.size()
        assert embed_dim == self.embed_dim
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, "Only bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            assert attn_mask.dim() == 3, "attn_mask's dimension {} is not supported".format(attn_mask.dim())
            assert list(attn_mask.size()) == [bsz, tgt_len, src_len]

        # (n_heads, batch_size, target_len, head_dim)
        query = query.view(tgt_len, bsz, num_heads, head_dim).permute(2, 1, 0, 3)
        # (n_heads, batch_size, source_len, head_dim)
        key = key.view(src_len, bsz, num_heads, head_dim).permute(2, 1, 0, 3)

        compatibility = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        assert list(compatibility.size()) == [num_heads, bsz, tgt_len, src_len]

        compatibility = compatibility.permute(1, 2, 3, 0).contiguous()
        logits = compatibility.sum(-1)
        assert list(logits.size()) == [bsz, tgt_len, src_len]

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            logits.masked_fill_(attn_mask, -math.inf)

        log_prob = F.log_softmax(logits, dim=-1)
        assert not torch.isnan(log_prob).any()

        return log_prob.squeeze(1)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, tanh_clipping: int = 10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.clipping = tanh_clipping

        self.query_MLP = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            )
        self.query_norm = nn.BatchNorm1d(self.embed_dim, track_running_stats=False)

    
    def forward(self, query, key, value, mask=None):
        num_heads = self.num_heads
        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        _, src_len, embed_dim = key.size()
        assert embed_dim == self.embed_dim
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        query = query + self.query_MLP(myMHA(query, key, value, self.num_heads, mask, self.clipping)[0])
        query = self.query_norm(query.squeeze(1))
        query = query.unsqueeze(1)

        logits = myMHA(query, key, value, 1, mask, self.clipping)[1]
        assert list(logits.size()) == [bsz, tgt_len, src_len]

        log_prob = F.log_softmax(logits, dim=-1)
        assert not torch.isnan(log_prob).any()

        return log_prob.squeeze(1)

def myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    """
    Compute multi-head attention (MHA) given a query Q, key K, value V and attention mask :
      h = Concat_{k=1}^nb_heads softmax(Q_k^T.K_k).V_k 
    Note : We did not use nn.MultiheadAttention to avoid re-computing all linear transformations at each call.
    Inputs : Q of size (bsz, dim_emb, 1)                batch of queries
             K of size (bsz, dim_emb, nb_nodes+1)       batch of keys
             V of size (bsz, dim_emb, nb_nodes+1)       batch of values
             mask of size (bsz, nb_nodes+1)             batch of masks of visited cities
             clip_value is a scalar 
    Outputs : attn_output of size (bsz, 1, dim_emb)     batch of attention vectors
              attn_weights of size (bsz, 1, nb_nodes+1) batch of attention weights
    """
    bsz, nb_nodes, emd_dim = K.size() #  dim_emb must be divisable by nb_heads
    if nb_heads>1:
        # PyTorch view requires contiguous dimensions for correct reshaping
        assert list(Q.size()) == [bsz, 1, emd_dim], Q.size()
        assert list(V.size()) == [bsz, nb_nodes, emd_dim], V.size()
        Q = Q.transpose(1,2).contiguous() # size(Q)=(bsz, dim_emb, 1)
        Q = Q.view(bsz*nb_heads, emd_dim//nb_heads, 1) # size(Q)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        Q = Q.transpose(1,2).contiguous() # size(Q)=(bsz*nb_heads, 1, dim_emb//nb_heads)
        K = K.transpose(1,2).contiguous() # size(K)=(bsz, dim_emb, nb_nodes+1)
        K = K.view(bsz*nb_heads, emd_dim//nb_heads, nb_nodes) # size(K)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        K = K.transpose(1,2).contiguous() # size(K)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
        V = V.transpose(1,2).contiguous() # size(V)=(bsz, dim_emb, nb_nodes+1)
        V = V.view(bsz*nb_heads, emd_dim//nb_heads, nb_nodes) # size(V)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        V = V.transpose(1,2).contiguous() # size(V)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
    logits = torch.bmm(Q, K.transpose(1,2))/ Q.size(-1)**0.5 # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    if clip_value is not None:
        logits = clip_value * torch.tanh(logits)
    if mask is not None:
        assert list(mask.size()) == [bsz, 1, nb_nodes], mask.size()
        if nb_heads>1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0) # size(mask)=(bsz*nb_heads, 1, nb_nodes+1)
            assert list(mask.size()) == [bsz*nb_heads, 1, nb_nodes]
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    attn_weights = torch.softmax(logits, dim=-1) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    attn_output = torch.bmm(attn_weights, V) # size(attn_output)=(bsz*nb_heads, 1, dim_emb//nb_heads)
    if nb_heads>1:
        attn_output = attn_output.transpose(1,2).contiguous() # size(attn_output)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, emd_dim, 1) # size(attn_output)=(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1,2).contiguous() # size(attn_output)=(bsz, 1, dim_emb)
        logits = logits.view(bsz, nb_heads, 1, nb_nodes) # size(logits)=(bsz, nb_heads, 1, nb_nodes+1)
        logits = logits.mean(dim=1) # mean over the heads, size(logits)=(bsz, 1, nb_nodes+1)
    return attn_output, logits
