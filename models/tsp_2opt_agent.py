from environments.tsp_2opt import TSP2OPTState
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, NamedTuple, Tuple
from torch_geometric.data import Data, Batch
from .encoder import TourEncoder, cal_size_list, MLP, GNNEncoder, EdgeFeatureExtractor
from .decoder import AttentionDecoder, SimpleDecoder
from .ActorCriticNetwork import Encoder, Decoder


class TSP2OPTAgent(nn.Module):
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

        self.solution_encoder = Encoder(
            input_dim=self.embed_dim,
            embedding_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            n_nodes=args.graph_size,
            n_rnn_layers=1,
        )

        self.best_encoder = Encoder(
            input_dim=self.embed_dim,
            embedding_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            n_nodes=args.graph_size,
            n_rnn_layers=1,
        )

        self.edge_extractor = EdgeFeatureExtractor(self.embed_dim, self.embed_dim)

        self.project_edge_embeddings = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.step_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.edge_decoder = AttentionDecoder(
            self.embed_dim, self.decoder_num_heads, bias=False, tanh_clipping=self.tanh_clipping
        )

        self.W_star = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.W_s = nn.Linear(self.embed_dim, self.embed_dim // 2)

        self.value_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
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

        best_node = node_embeddings.gather(dim=1, index=state.best_tour.unsqueeze(-1).expand_as(node_embeddings))
        s_out_star, s_hidden_star, _, _ = self.best_encoder(best_node)
        best_solution_graph = s_out_star.sum(dim=1)

        curr_node = node_embeddings.gather(dim=1, index=state.curr_tour.unsqueeze(-1).expand_as(node_embeddings))
        s_out, s_hidden, _, g_embedding = self.solution_encoder(curr_node)
        curr_solution_graph = s_out.sum(dim=1)

        edge_index_offset = torch.arange(batch_size, device=device) * graph_size
        edge_index = (
            (state.curr_edge_list + edge_index_offset[:, None, None]).flatten(0, 1).T
        )  # shape=[2, batch_size * graph_size]
        node_x = node_embeddings.flatten(0, 1)
        solution_x = torch.empty_like(s_out).scatter(
            dim=1, index=state.curr_tour.unsqueeze(-1).expand_as(node_embeddings), src=s_out
        ).flatten(0, 1)
        edge_embeddings = self.edge_extractor(node_x=node_x, solution_x=solution_x, edge_index=edge_index)
        assert edge_embeddings.dim() == 2
        assert edge_embeddings.size(0) == batch_size * graph_size
        dense_edge_embeddings = edge_embeddings.reshape(batch_size, graph_size, -1)
        curr_solution_edge = dense_edge_embeddings

        glimpse_K, glimpse_V, logit_K = self.project_edge_embeddings(curr_solution_edge).chunk(3, dim=-1)
        glimpse_K = glimpse_K.permute(1, 0, 2).contiguous()  # (num_edges, batch_size, embed_dim)
        glimpse_V = glimpse_V.permute(1, 0, 2).contiguous()  # (num_edges, batch_size, embed_dim)
        logit_K = logit_K.contiguous()  # ((batch_size, num_edges, embed_dim))

        mask = torch.zeros((batch_size, 1, graph_size), dtype=torch.bool, device=device)
        # select first edge to remove in 2-opt
        query1 = self._make_query(best_solution_graph, curr_solution_graph, curr_solution_edge)
        log_p1 = self.edge_decoder(query1, glimpse_K, glimpse_V, logit_K)
        selected1 = self._select_edge(log_p1, mask.squeeze())
        # update mask
        forbid = torch.stack([selected1, selected1 + 1, selected1 - 1], dim=2).detach() % graph_size
        mask.scatter_(dim=-1, index=forbid, value=1)
        # select second edge to remove in 2-opt
        query2 = self._make_query(best_solution_graph, curr_solution_graph, curr_solution_edge, selected1)
        log_p2 = self.edge_decoder(query2, glimpse_K, glimpse_V, logit_K, mask)
        selected2 = self._select_edge(log_p2, mask.squeeze())

        log_p = torch.stack([log_p1, log_p2], dim=1)  # shape=[batch_size, 2, graph_size]
        selected = torch.stack([selected1, selected2], dim=1)  # shape=[batch_size, 2, 1]

        enc_h = (s_hidden[0][-1], s_hidden[1][-1])
        enc_h_star = (s_hidden_star[0][-1], s_hidden_star[1][-1])

        v_g = torch.mean(g_embedding, dim=1).squeeze(1)
        h_v = torch.cat([self.W_star(enc_h_star[0]), self.W_s(enc_h[0])], dim=1)
        values = self.value_decoder(v_g + h_v)

        return selected, log_p, values

    def _make_query(
        self,
        best_solution_graph: torch.Tensor,
        curr_solution_graph: torch.Tensor,
        curr_solution_edge: torch.Tensor,
        edge_selected: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, _, embed_dim = curr_solution_edge.shape
        assert embed_dim == self.embed_dim
        if edge_selected is not None:
            step_context = self.step_proj(
                curr_solution_edge.gather(1, edge_selected[..., None].expand(batch_size, 1, embed_dim),).view(
                    batch_size, embed_dim
                )
            )
        else:
            step_context = self.step_proj(self.W_placeholder.expand(batch_size, embed_dim))

        query = (best_solution_graph + curr_solution_graph + step_context).unsqueeze(0)
        assert list(query.shape) == [1, batch_size, embed_dim], query.shape

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
