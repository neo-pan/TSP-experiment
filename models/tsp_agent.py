from environments.tsp import TSPState
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, NamedTuple, Tuple
from torch_geometric.data import Data, Batch
from .encoder import cal_size_list, MLP, GNNEncoder
from .decoder import AttentionDecoder


class AttentionInfoFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    graph_context_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


class TSPAgent(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.node_dim = args.node_dim
        self.edge_dim = args.edge_dim
        self.embed_dim = args.embed_dim
        self.num_gnn_layers = args.num_gnn_layers
        self.encoder_num_heads = args.encoder_num_heads
        self.decoder_num_heads = args.decoder_num_heads
        self.bias = args.bias
        self.tanh_clipping = args.tanh_clipping
        self.pooling_method = args.pooling_method
        self.normalization = args.normalization
        self.set_decode_type(args.decode_type)

        self.node_embedder = nn.Linear(self.node_dim, self.embed_dim)
        self.edge_embedder = nn.Linear(self.edge_dim, self.embed_dim)

        self.encoder = GNNEncoder(
            self.embed_dim,
            self.num_gnn_layers,
            self.encoder_num_heads,
            self.normalization,
            pooling_method=self.pooling_method,
        )

        self.graph_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.step_proj = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=False)
        self.project_node_embeddings = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        self.decoder = AttentionDecoder(
            self.embed_dim, self.decoder_num_heads, bias=False, tanh_clipping=self.tanh_clipping
        )

        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embed_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

    def set_decode_type(self, decode_type: str) -> None:
        assert decode_type in ["greedy", "sampling"]
        self.decode_type = decode_type

    def init_embed(self, data: Batch) -> Batch:
        assert data.pos.size(-1) == self.node_dim
        assert data.edge_attr.size(-1) == self.edge_dim

        x = self.node_embedder(data.pos)
        edge_attr = self.edge_embedder(data.edge_attr)

        d = data.clone()
        d.x = x
        d.edge_attr = edge_attr

        return d

    def precompute_fixed(self, node_embeddings: torch.Tensor, graph_feat: torch.Tensor) -> AttentionInfoFixed:
        graph_context = self.graph_proj(graph_feat)
        glimpse_K, glimpse_V, logit_K = self.project_node_embeddings(node_embeddings).chunk(3, dim=-1)
        glimpse_K = glimpse_K.permute(1, 0, 2).contiguous()  # (num_nodes, batch_size, embed_dim)
        glimpse_V = glimpse_V.permute(1, 0, 2).contiguous()  # (num_nodes, batch_size, embed_dim)
        logit_K = logit_K.contiguous()  # ((batch_size, num_nodes, embed_dim))

        return AttentionInfoFixed(
            node_embeddings=node_embeddings,
            graph_context_projected=graph_context,
            glimpse_key=glimpse_K,
            glimpse_val=glimpse_V,
            logit_key=logit_K,
        )

    def forward(self, state: TSPState, fixed: AttentionInfoFixed) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = fixed.node_embeddings.shape
        assert state.first_node is None or list(state.first_node.shape) == list(state.pre_node.shape) == [
            batch_size,
            1,
        ], f"{state.first_node.shape}-{state.pre_node.shape}-{batch_size}"

        assert list(state.avail_mask.shape) == [
            batch_size,
            num_nodes,
        ], f"{state.avail_mask.shape}-{[batch_size, num_nodes]}"

        # Transform node features for attention compute
        query = self._make_query(state, fixed.node_embeddings, fixed.graph_context_projected)

        mask = ~state.avail_mask.unsqueeze(1)  # (batch_size, 1, num_nodes)

        log_p = self.decoder(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
        selected = self._select_node(log_p, mask.squeeze())

        return selected, log_p

    def _make_query(self, state: TSPState, node_embeddings: torch.Tensor, graph_context: torch.Tensor) -> torch.Tensor:
        r"""
        query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
        """
        batch_size, _, embed_dim = node_embeddings.shape

        if state.first_node is not None:
            step_context = self.step_proj(
                node_embeddings.gather(
                    1, torch.cat((state.first_node, state.pre_node), 1)[:, :, None].expand(batch_size, 2, embed_dim),
                ).view(batch_size, -1)
            )
        else:
            step_context = self.step_proj(self.W_placeholder.expand(batch_size, -1))

        query = (graph_context + step_context).unsqueeze(0)
        assert list(query.shape) == [1, batch_size, embed_dim], query.shape

        return query.contiguous()

    def _select_node(self, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert log_p.shape == mask.shape, f"{log_p.shape}, {mask.shape}"
        probs = log_p.exp()
        assert not torch.isnan(probs).any(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            selected = selected.unsqueeze(-1)
            assert not mask.gather(1, selected).any(), "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1)
            while mask.gather(1, selected).any():
                print("Sampled bad values, resampling!")
                print(selected)
                print(mask)
                selected = probs.multinomial(1)
        else:
            assert False, "Unknown decode type"
        return selected
