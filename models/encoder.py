#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv as _GATConv, TransformerConv as _TransformerConv


def cal_size_list(in_dim, out_dim, layer_num):
    return np.linspace(in_dim, out_dim, layer_num + 1, dtype="int")


class MLP(nn.Module):
    r"""
    Multilayer Perceptron
    
    Args:
    - size_list
    - activation
    - bias
    """
    def __init__(
        self,
        size_list: np.ndarray,
        activation: nn.Module = nn.LeakyReLU,
        bias: bool = True,
    ) -> None:
        assert size_list.ndim == 1
        assert size_list.dtype == np.int
        super().__init__()
        self.in_dim = size_list[0]
        self.out_dim = size_list[-1]

        self.mlp = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(size_list[ln], size_list[ln + 1], bias=bias,),
                    activation(),
                )
                for ln in range(len(size_list) - 1)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_dim, f"{x.shape} -- {self.in_dim}"
        out = self.mlp(x)

        return out


class GATEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.heads = heads
        assert (self.embed_dim % self.heads) == 0
        gnn_layer_list = []
        for _ in range(self.num_layers):
            gnn_layer = _TransformerConv(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim // self.heads,
                heads=self.heads,
            )
            gnn_layer_list.append(gnn_layer)

        self.gnn_layer_list = nn.ModuleList(gnn_layer_list)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        
        for gnn_layer in self.gnn_layer_list:
            x = gnn_layer(x, edge_index)

        return x
