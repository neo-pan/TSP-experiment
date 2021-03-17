#!/usr/bin/env python
# coding=utf-8

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GATConv as _GATConv
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import TransformerConv as _TransformerConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch


def get_pooling_func(pooling_method: str):
    pooling_funcs = {
        "add": global_add_pool,
        "max": global_max_pool,
        "mean": global_mean_pool,
    }
    assert pooling_method in pooling_funcs.keys(), f"Wrong pooling function: {pooling_method}"
    return pooling_funcs[pooling_method]


def get_normalization_class(normalization: str):
    norm_class = {"batch": BatchNorm, "instance": InstanceNorm}
    assert normalization in norm_class.keys(), f"Wrong normalization methon: {normalization}"
    return norm_class[normalization]


def get_gnn_layer_class(gnn_layer: str):
    gnn = {"gat": _GATConv, "transformer": _TransformerConv}
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
        norm_list = []
        ff_list = []
        for _ in range(self.num_layers):
            gnn_layer = _TransformerConv(
                in_channels=self.embed_dim, out_channels=self.embed_dim // self.heads, heads=self.heads,
            )
            gnn_layer_list.append(gnn_layer)
            norm = self.norm_class(self.embed_dim)
            norm_list.append(norm)
            feed_forward = nn.Sequential(
                nn.Linear(self.embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, self.embed_dim),
                self.norm_class(in_channels=self.embed_dim),
            )
            ff_list.append(feed_forward)

        self.gnn_layer_list = nn.ModuleList(gnn_layer_list)
        self.norm_list = nn.ModuleList(norm_list)
        self.ff_list = nn.ModuleList(ff_list)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:

        for gnn_layer, norm, ff in zip(self.gnn_layer_list, self.norm_list, self.ff_list):
            # identity = x
            x = gnn_layer(x, edge_index)
            x = norm(x)
            x = ff(x)
            # x += identity

        return x

    def encode(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode graph input features.
        
        Get node_embedding, graph_embedding.
        Only invoke once in a full batch trajectory.

        Args:
            data: a torch_geometric batch of graphs

        Returns:
            dense_x: encoded graph node features, shape->[batch_size, node_num, embed_dim]
            graph_feat: graph features produced by pooling function, shape->[batch_size, embed_dim]
        """
        x = data.x
        edge_index = data.edge_index

        x = self(x, edge_index)
        dense_x, dense_mask = to_dense_batch(x, data.batch)
        assert dense_mask.all(), "For now only support a batch of graphs with the same number of nodes"

        # Pooling graph features
        graph_feat = self.pooling_func(x, data.batch)

        return dense_x, graph_feat
