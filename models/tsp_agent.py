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
        assert self.pooling_method in ["add", "max", "mean"]
        self.decode_type = args.decode_type
        assert self.decode_type in ["greedy", "sampling"]
        self.pooling_func = globals()[f"global_{self.pooling_method}_pool"]
        self.normalization = args.normalization
        assert self.normalization in ["batch", "instance"]
        linear_size_list = cal_size_list(
            self.input_dim, self.embed_dim, self.num_embed_layers
        )

        self.linear_embedder = MLP(linear_size_list, bias=self.bias)
        self.encoder = GNNEncoder(
            self.embed_dim, self.num_gnn_layers, self.encoder_num_heads, self.normalization
        )

        self.graph_proj = nn.Linear(self.embed_dim, self.embed_dim, self.bias)
        self.step_proj = nn.Linear(self.embed_dim * 2, self.embed_dim *2, self.bias)

        self.decoder = AttentionDecoder(
            self.embed_dim * 3, self.embed_dim, self.decoder_num_heads, self.bias, self.tanh_clipping
        )

    def set_decode_type(self, decode_type: str) -> None:
        assert decode_type in ["greedy", "sampling"]
        self.decode_type = decode_type

    def _embed(self, data: Batch) -> Batch:
        assert data.x.size(-1) == self.input_dim
        self._input_x = data.x
        x = self.linear_embedder(data.x)
        d = data.clone()
        d.x = x

        return d

    def encode(self, data: Batch) -> Batch:
        """
        Encode graph input features.
        Get node_embedding, graph_embedding and attention keys.
        Only invoke once in a full batch trajectory.
        """
        # Linear embedding graph node features
        data = self._embed(data)
        x = data.x
        edge_index = data.edge_index
        # GNN encoding graph node embeddings
        x = self.encoder(x, edge_index)
        self._dense_x, _dense_mask = to_dense_batch(x, data.batch)
        assert (
            _dense_mask.all()
        ), "For now only support a batch of graphs with the same number of nodes"
        # Pooling graph features
        self._graph_feat = self.pooling_func(x, data.batch)
        # Precompute attention decoder keys based on graph node features
        key = self._dense_x
        # Change its shape from [batch_size, max_num_nodes, embed_dim]
        # to [max_num_nodes, batch_size, embed_dim] for attention layer
        key = key.permute(1, 0, 2)  
        self._precomputed_k = self.decoder.precompute_keys(key)

        d = data.clone()
        d.x = x

        return d

    def forward(self, state: TSPState) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, embed_dim = self._dense_x.shape
        assert (
            list(state.first_node.shape)
            == list(state.pre_node.shape)
            == [batch_size, 1]
        )
        assert list(state.avail_mask.shape) == [batch_size, num_nodes]

        # Transform node features for attention compute
        query = self._make_query(state)
        mask = state.avail_mask

        log_p = self.decoder(query, precomputed_k=self._precomputed_k, attn_mask=~mask.unsqueeze(1))
        selected = self._select_node(log_p, mask)

        return selected, log_p

    def _make_query(self, state: TSPState) -> torch.Tensor:
        r"""
        query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
        """
        batch_size, num_nodes, embed_dim = self._dense_x.shape

        graph_context = self._graph_feat

        step_context = self._dense_x.gather(
            1,
            torch.cat((state.first_node, state.pre_node), 1)[:, :, None].expand(
                batch_size, 2, embed_dim
            ),
        ).view(batch_size, -1)

        query = torch.cat((graph_context, step_context), 1).unsqueeze(0)
        assert list(query.shape) == [1, batch_size, embed_dim * 3], query.shape

        return query

    def _select_node(self, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert log_p.shape == mask.shape, f"{log_p.shape}, {mask.shape}"
        probs = log_p.exp()
        assert (probs == probs).all(), "Probs should not contain any nans"
        
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            selected = selected.unsqueeze(-1)
            assert mask.gather(
                1, selected
            ).all(), "Decode greedy: infeasible action has maximum probability"
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
