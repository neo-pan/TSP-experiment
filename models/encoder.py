from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import (
    Sequential,
    GATConv,
    GATv2Conv,
    GINConv,
    GatedGraphConv,
    TransformerConv,
    BatchNorm,
    InstanceNorm,
    GraphNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import contains_self_loops, is_undirected, to_dense_batch

from torch.utils.checkpoint import checkpoint


def get_pooling_func(pooling_method: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    pooling_funcs = {
        "add": global_add_pool,
        "max": global_max_pool,
        "mean": global_mean_pool,
    }
    assert pooling_method in pooling_funcs.keys(), f"Wrong pooling function: {pooling_method}"
    return pooling_funcs[pooling_method]


def get_normalization_class(normalization: str) -> nn.Module:
    norm_class = {"batch": BatchNorm, "instance": InstanceNorm}
    assert normalization in norm_class.keys(), f"Wrong normalization methon: {normalization}"
    return norm_class[normalization]


def get_gnn_layer_class(gnn_layer: str) -> nn.Module:
    gnn = {"gat": GATConv, "transformer": TransformerConv}
    assert gnn_layer in gnn.keys(), f"Wrong GNN layer class: {gnn_layer}"
    return gnn[gnn_layer]


def cal_size_list(in_dim: int, out_dim: int, layer_num: int) -> np.ndarray:
    return np.linspace(in_dim, out_dim, layer_num + 1, dtype="int")


class MLP(nn.Module):
    r"""Multilayer Perceptron
    
    Attributes:
        size_list: A numpy array of ints represents dimensions of each layer of MLP
        activation: Activation methods
        bias: A boolean indicates whether to use bias
    """

    def __init__(
        self,
        size_list: np.ndarray,
        activation: nn.Module = nn.LeakyReLU,
        bias: bool = True,
        last_activation: nn.Module = nn.LeakyReLU,
        last_bias: bool = True,
    ) -> None:
        assert size_list.ndim == 1
        assert size_list.dtype == np.int
        super().__init__()
        self.in_dim = size_list[0]
        self.out_dim = size_list[-1]

        self.mlp = nn.Sequential(
            *(
                nn.Sequential(nn.Linear(size_list[ln], size_list[ln + 1], bias=bias,), activation(),)
                for ln in range(len(size_list) - 2)
            ),
            nn.Sequential(nn.Linear(size_list[-2], size_list[-1], bias=last_bias), last_activation(),),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_dim, f"{x.shape} -- {self.in_dim}"
        out = self.mlp(x)

        return out


class GNNEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        heads: int = 8,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        pooling_method: str = "mean",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.heads = heads
        assert (self.embed_dim % self.heads) == 0
        self.pooling_func = get_pooling_func(pooling_method)
        self.norm_class = get_normalization_class(normalization)
        gnn_layer_list = []
        # norm_list = []
        ff_list = []
        for i in range(self.num_layers):
            gnn_layer = TransformerConv(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim // self.heads,
                heads=self.heads,
                edge_dim=self.embed_dim,
            )
            gnn_layer_list.append(gnn_layer)
            if i < self.num_layers-1:
                feed_forward = Sequential(
                    "x, batch",
                    [
                        (nn.GELU(), "x -> x"),
                        (GraphNorm(in_channels=self.embed_dim), "x, batch -> x"),
                    ],
                )
            else:
                feed_forward = Sequential(
                    "x, batch",
                    [
                        (nn.Linear(self.embed_dim, feed_forward_hidden), "x -> x"),
                        nn.GELU(),
                        (GraphNorm(in_channels=feed_forward_hidden), "x, batch -> x"),
                        nn.Linear(feed_forward_hidden, self.embed_dim),
                    ],
                )
            ff_list.append(feed_forward)

        self.gnn_layer_list = nn.ModuleList(gnn_layer_list)
        self.ff_list = nn.ModuleList(ff_list)

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode graph input features.
        
        Get node_embedding, graph_embedding.
        Only invoke once in a full batch trajectory.

        Args:
            data: a torch_geometric batch of graphs

        Returns:
            node_embeddings: encodded graph node features, shape->[batch_size, node_num, embed_dim]
            graph_feat: graph features produced by pooling function, shape->[batch_size, embed_dim]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        for gnn_layer, ff in zip(self.gnn_layer_list, self.ff_list):
            x = gnn_layer(x, edge_index, edge_attr)
            x = ff(x, batch)

        node_embeddings, dense_mask = to_dense_batch(x, data.batch)
        assert dense_mask.all(), "For now only support a batch of graphs with the same number of nodes"

        # Pooling graph features
        graph_feat = self.pooling_func(x, data.batch)

        return (node_embeddings, graph_feat)


class TourEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        heads: int = 8,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        pooling_method: str = "add",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.heads = heads
        assert (self.embed_dim % self.heads) == 0
        self.pooling_func = get_pooling_func(pooling_method)
        self.norm_class = get_normalization_class(normalization)
        self.norm_out = GraphNorm(self.embed_dim)
        self.gnn_layer = GatedGraphConv(out_channels=self.embed_dim, num_layers=1)
        self.reversed_gnn_layer = GatedGraphConv(out_channels=self.embed_dim, num_layers=1)

        self.edge_extractor = EdgeFeatureExtractor(in_channels=self.embed_dim, edge_dim=self.embed_dim)

    def forward(
        self, dense_x: torch.Tensor, dense_edge_index: torch.Tensor, batch: torch.Tensor, return_edge: bool = False
    ):
        assert dense_x.dim() == dense_edge_index.dim()
        assert dense_x.size(0) == dense_edge_index.size(0)  # batch_size
        assert dense_x.size(1) == dense_edge_index.size(1)  # graph_size
        assert dense_x.size(2) == self.embed_dim
        assert dense_edge_index.size(2) == 2, f"{dense_edge_index.size()}"

        batch_size = dense_x.size(0)
        graph_size = dense_x.size(1)
        device = dense_x.device

        node_x = dense_x.flatten(0, 1)  # flatten on first dimension(batch dim)
        x = node_x

        edge_index_offset = torch.arange(batch_size, device=device) * graph_size
        edge_index = (
            (dense_edge_index + edge_index_offset[:, None, None]).flatten(0, 1).T
        )  # shape=[2, batch_size * graph_size]
        reversed_edge_index = edge_index.flipud()

        x_dir_0 = x
        x_dir_1 = x
        for _ in range(self.num_layers):
            x_dir_0 = checkpoint(self.gnn_layer, x_dir_0, edge_index)
            x_dir_1 = checkpoint(self.reversed_gnn_layer, x_dir_1, reversed_edge_index)
        x = F.gelu(x_dir_0 + x_dir_1)
        x = self.norm_out(x, batch)
        tour_embeddings = self.pooling_func(x, batch)

        dense_edge_embeddings = None
        if return_edge:
            edge_embeddings = self.edge_extractor(node_x=node_x, solution_x=x, edge_index=edge_index, batch=batch)
            assert edge_embeddings.dim() == 2
            assert edge_embeddings.size(0) == batch_size * graph_size
            dense_edge_embeddings = edge_embeddings.reshape(batch_size, graph_size, -1)

        return tour_embeddings, dense_edge_embeddings


class EdgeFeatureExtractor(MessagePassing):

    _edge: Optional[torch.Tensor]

    def __init__(
        self, in_channels: int, edge_dim: int, bias: bool = True, **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(EdgeFeatureExtractor, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.bias = bias

        self.node_lin = nn.Linear(self.in_channels, self.in_channels, self.bias)
        self.solution_lin = nn.Linear(self.in_channels, self.in_channels, self.bias)
        # self.x_i_lin = nn.Linear(self.in_channels, self.edge_dim, self.bias)
        # self.x_j_lin = nn.Linear(self.in_channels, self.edge_dim, self.bias)
        self.edge_lin = nn.Linear(self.edge_dim, self.edge_dim, self.bias)

        self.norm_x = GraphNorm(self.in_channels)
        self.norm_edge = GraphNorm(self.edge_dim)

        self._edge = None
        self._batch = None

        self.reset_parameters()

    def reset_parameters(self):
        self.node_lin.reset_parameters()
        self.solution_lin.reset_parameters()
        # self.x_i_lin.reset_parameters()
        # self.x_j_lin.reset_parameters()
        self.edge_lin.reset_parameters()

    def forward(self, node_x: torch.Tensor, solution_x: torch.Tensor, edge_index: torch.Tensor, batch):

        assert node_x.size() == solution_x.size()
        num_nodes, _ = node_x.size()
        assert num_nodes == edge_index.size(1)
        assert not contains_self_loops(edge_index)

        node_x = self.node_lin(node_x)
        solution_x = self.solution_lin(solution_x)

        # x = self.norm_x(node_x + solution_x, batch)
        # x = F.relu(x)
        x = F.gelu(node_x+solution_x)
        x = self.norm_x(x, batch)

        self._batch = batch
        self.propagate(edge_index, x=x, size=None)

        edge_embedding = self._edge
        self._edge = None

        return edge_embedding

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:

        edge_embedding = x_i + x_j  # self.x_i_lin(x_i) + self.x_j_lin(x_j)

        edge_embedding = self.edge_lin(edge_embedding)
        edge_embedding = F.gelu(edge_embedding)
        edge_embedding = self.norm_edge(edge_embedding, self._batch)

        self._batch = None
        self._edge = edge_embedding

        return x_i
