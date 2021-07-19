from environments.tsp_2opt import TSP2OPTState
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, NamedTuple, Tuple
from torch_geometric.data import Data, Batch
from .encoder import TourEncoder, cal_size_list, MLP,GNNEncoder, TransformerEncoder
from .decoder import AttentionDecoder, TransformerDecoder


class TSP2OPTAgent(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.node_dim = args.node_dim
        self.edge_dim = args.edge_dim
        self.embed_dim = args.embed_dim
        self.num_gnn_layers = args.num_gnn_layers
        self.tour_gnn_layers = args.tour_gnn_layers
        self.encoder_num_heads = args.encoder_num_heads
        self.decoder_num_heads = args.decoder_num_heads
        self.bias = args.bias
        self.tanh_clipping = args.tanh_clipping
        self.pooling_method = args.pooling_method
        self.normalization = args.normalization
        self.set_decode_type(args.decode_type)

        self.node_embedder = nn.Linear(self.node_dim, self.embed_dim)
        self.edge_embedder = nn.Linear(self.edge_dim, self.embed_dim)

        self.encoder = TransformerEncoder(
            self.embed_dim,
            self.num_gnn_layers,
            self.encoder_num_heads,
            self.normalization,
            pooling_method=self.pooling_method,
        )

        self.curr_solution_encoder = TourEncoder(
            self.embed_dim,
            self.tour_gnn_layers,
            self.encoder_num_heads,
            self.normalization,
            pooling_method=args.tour_pooling_method,
        )

        self.best_solution_encoder = TourEncoder(
            self.embed_dim,
            self.tour_gnn_layers,
            self.encoder_num_heads,
            self.normalization,
            pooling_method=args.tour_pooling_method,
        )

        self.curr_solution_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.best_solution_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.step_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.k_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.two_opt_decoder = TransformerDecoder(
            self.embed_dim, self.decoder_num_heads, self.tanh_clipping
        )

        self.value_decoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim),
            nn.Linear(self.embed_dim, 1),
        )

        self.W_placeholder = nn.Parameter(torch.Tensor(self.embed_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

    def set_decode_type(self, decode_type: str) -> None:
        assert decode_type in ["greedy", "sampling"]
        self.decode_type = decode_type

    def init_embed(self, data: Batch) -> Batch:
        assert data.pos.size(-1) == self.node_dim
        assert data.edge_attr.size(-1) == self.edge_dim

        x = self.node_embedder(data.x)
        edge_attr = self.edge_embedder(data.edge_attr)

        d = data.clone()
        d.x = x
        d.edge_attr = edge_attr

        return d

    def forward(
        self, state: TSP2OPTState, node_embeddings: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert node_embeddings.dim() == 3
        batch_size, graph_size, embed_dim = node_embeddings.size()
        assert embed_dim == self.embed_dim
        device = node_embeddings.device

        best_solution_graph, _ = self.best_solution_encoder(
            node_embeddings, state.best_edge_list, batch
        )
        curr_solution_graph, curr_solution_node = self.curr_solution_encoder(
            node_embeddings, state.curr_edge_list, batch
        )

        curr_solution_node = curr_solution_node.gather(dim=1, index=state.curr_tour.unsqueeze(-1).expand_as(curr_solution_node))
        assert list(curr_solution_node.size()) == list(node_embeddings.size())

        values = self.value_decoder(torch.cat([best_solution_graph, curr_solution_graph], dim=-1))

        best_solution_graph = self.best_solution_proj(best_solution_graph)
        curr_solution_graph = self.curr_solution_proj(curr_solution_graph)

        K = self.k_linear(curr_solution_node)
        V = self.v_linear(curr_solution_node)

        mask = torch.zeros((batch_size, 1, graph_size), dtype=torch.bool, device=device)
        # select first edge to remove in 2-opt
        query1 = self._make_query(best_solution_graph, curr_solution_graph, curr_solution_node)
        log_p1 = self.two_opt_decoder(query1, K, V)
        selected1 = self._select_edge(log_p1, mask.squeeze())
        # update mask
        forbid = torch.stack([selected1, selected1 + 1, selected1 - 1], dim=2).detach() % graph_size
        mask.scatter_(dim=-1, index=forbid, value=1)
        # select second edge to remove in 2-opt
        query2 = self._make_query(best_solution_graph, curr_solution_graph, curr_solution_node, selected1)
        log_p2 = self.two_opt_decoder(query2, K, V, mask)
        selected2 = self._select_edge(log_p2, mask.squeeze())

        log_p = torch.stack([log_p1, log_p2], dim=1)  # shape=[batch_size, 2, graph_size]
        selected = torch.stack([selected1, selected2], dim=1)  # shape=[batch_size, 2, 1]

        return selected, log_p, values

    def _make_query(
        self,
        best_solution_graph: torch.Tensor,
        curr_solution_graph: torch.Tensor,
        curr_solution_node: torch.Tensor,
        node_selected: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, _, embed_dim = curr_solution_node.shape
        assert embed_dim == self.embed_dim
        if node_selected is not None:
            step_context = self.step_proj(
                curr_solution_node.gather(1, node_selected[..., None].expand(batch_size, 1, embed_dim),).view(
                    batch_size, embed_dim
                )
            )
        else:
            step_context = self.step_proj(self.W_placeholder.expand(batch_size, embed_dim))

        query = (best_solution_graph + curr_solution_graph + step_context).unsqueeze(1)
        assert list(query.shape) == [batch_size, 1, embed_dim], query.shape

        return query

    def _select_edge(self, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert log_p.shape == mask.shape, f"{log_p.shape}, {mask.shape}"
        probs = log_p.exp()
        assert not torch.isnan(probs).any(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            selected = probs.max(1).indices.unsqueeze(-1)
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
