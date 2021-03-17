from environments.tsp import TSPState
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Tuple
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils.to_dense_batch import to_dense_batch
from .encoder import cal_size_list, MLP, GNNEncoder
from .decoder import AttentionDecoder


class TSPAgent(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.embed_dim = args.embed_dim
        self.num_embed_layers = args.num_embed_layers
        self.num_gnn_layers = args.num_gnn_layers
        self.encoder_num_heads = args.encoder_num_heads
        self.decoder_num_heads = args.decoder_num_heads
        self.bias = args.bias
        self.tanh_clipping = args.tanh_clipping
        self.pooling_method = args.pooling_method
        self.normalization = args.normalization
        self.set_decode_type(args.decode_type)
        linear_size_list = cal_size_list(self.input_dim, self.embed_dim, self.num_embed_layers)

        self.linear_embedder = MLP(linear_size_list, bias=self.bias, last_activation=nn.Identity)
        self.encoder = GNNEncoder(
            self.embed_dim,
            self.num_gnn_layers,
            self.encoder_num_heads,
            self.normalization,
            pooling_method=self.pooling_method,
        )

        self.graph_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.step_proj = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=False)

        self.decoder = AttentionDecoder(
            self.embed_dim, self.embed_dim, self.decoder_num_heads, bias=False, tanh_clipping=self.tanh_clipping
        )

        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embed_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

    def set_decode_type(self, decode_type: str) -> None:
        assert decode_type in ["greedy", "sampling"]
        self.decode_type = decode_type

    def init_embed(self, data: Batch) -> Batch:
        assert data.x.size(-1) == self.input_dim
        x = self.linear_embedder(data.x)
        d = data.clone()
        d.x = x

        return d

    def forward(
        self, state: TSPState, dense_x: torch.Tensor, graph_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, embed_dim = dense_x.shape
        assert state.first_node is None or list(state.first_node.shape) == list(state.pre_node.shape) == [
            batch_size,
            1,
        ], f"{state.first_node.shape}-{state.pre_node.shape}-{batch_size}"

        assert list(state.avail_mask.shape) == [
            batch_size,
            num_nodes,
        ], f"{state.avail_mask.shape}-{[batch_size, num_nodes]}"

        # Transform node features for attention compute
        query = self._make_query(state, dense_x, graph_feat)
        mask = state.avail_mask
        key = dense_x.permute(1, 0, 2)

        log_p = self.decoder(query, key, attn_mask=~mask.unsqueeze(1))
        selected = self._select_node(log_p, mask)

        return selected, log_p

    def _make_query(self, state: TSPState, dense_x: torch.Tensor, graph_feat: torch.Tensor) -> torch.Tensor:
        r"""
        query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
        """
        batch_size, num_nodes, embed_dim = dense_x.shape

        graph_context = self.graph_proj(graph_feat)
        if state.first_node is not None:
            step_context = self.step_proj(
                dense_x.gather(
                    1, torch.cat((state.first_node, state.pre_node), 1)[:, :, None].expand(batch_size, 2, embed_dim),
                ).view(batch_size, -1)
            )
        else:
            step_context = self.step_proj(self.W_placeholder.expand(batch_size, -1))

        query = (graph_context + step_context).unsqueeze(0)
        assert list(query.shape) == [1, batch_size, embed_dim], query.shape

        return query

    def _select_node(self, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert log_p.shape == mask.shape, f"{log_p.shape}, {mask.shape}"
        probs = log_p.exp()
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            selected = selected.unsqueeze(-1)
            assert mask.gather(1, selected).all(), "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1)
            while not mask.gather(1, selected).all():
                print("Sampled bad values, resampling!")
                print(selected)
                print(mask)
                selected = probs.multinomial(1)
        else:
            assert False, "Unknown decode type"
        return selected
