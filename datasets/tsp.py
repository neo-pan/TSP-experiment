import os
import os.path as osp
from typing import Tuple, Union
from functools import partial

import networkx as nx
import numpy as np
from numpy.lib.arraysetops import isin
from numpy.random import default_rng
import torch
from torch_geometric import data
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import Distance
from torch_geometric.utils import from_networkx, to_undirected

_curdir = osp.dirname(osp.abspath(__file__))
_fake_dataset_root = osp.join(_curdir, "FAKEDataset")


def gen_graph_data(node_data: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(node_data, int):
        node_num = node_data
        pos = torch.empty(size=(node_num, 2)).uniform_(0, 1)
    elif isinstance(node_data, torch.Tensor):
        assert node_data.dim() == 2
        assert node_data.size(1) == 2
        node_num = node_data.size(0)
        pos = node_data

    index = torch.arange(node_num).unsqueeze(-1).expand([-1, node_num])
    rol = torch.reshape(index, [-1])
    col = torch.reshape(torch.t(index), [-1])
    edge_index = torch.stack([rol, col], dim=0)
    node_feat = pos

    return edge_index, pos, node_feat


def gen_knn_graph(node_data: Union[int, torch.Tensor], k=10, pos_feature: bool = True) -> Data:
    edge_index, pos, node_feat = gen_graph_data(node_data)

    edge_index = knn_graph(pos, k=k, loop=True)

    graph = Data(x=node_feat, edge_index=edge_index, pos=pos)
    graph = Distance(norm=False, cat=False)(graph)

    return graph


def gen_complete_graph(node_data: Union[int, torch.Tensor], pos_feature: bool = True) -> Data:
    edge_index, pos, node_feat = gen_graph_data(node_data)

    graph = Data(x=node_feat, edge_index=edge_index, pos=pos)
    graph = Distance(norm=False, cat=False)(graph)

    return graph


def get_graph_gen_func(graph_type: str, **kwargs):
    graph_gen_func = {
        "complete": gen_complete_graph,
        "knn": gen_knn_graph,
    }
    if graph_type == "knn":
        k = kwargs.get("graph_knn")
        assert k is not None
        return partial(gen_knn_graph, k=k)
    return graph_gen_func.get(graph_type)


class TSPDataset(Dataset):
    def __init__(
        self,
        size: int,
        graph_size: int,
        graph_type: str,
        train_flag=False,
        device=None,
        transform=None,
        pre_filter=None,
        load_path: str = None,
        **kwargs,
    ):
        self.size, self.train_flag = size, train_flag
        self.graph_size = graph_size
        self.graph_type = graph_type
        self.device = device
        self.load_path = load_path
        self.classification_flag = False
        self.data_list = None
        self.gen_graph = get_graph_gen_func(self.graph_type, **kwargs)
        assert self.gen_graph, f"Wrong graph type: {self.graph_type}"
        if self.load_path:
            self.data_list = []
            data = torch.load(self.load_path)
            graph_list = data["Points"]
            opt_list = data["OptDistance"]
            assert self.size <= len(graph_list)
            rng = default_rng()
            ids = rng.choice(range(len(graph_list)), self.size, replace=False)
            for i in ids:
                graph = self.gen_graph(node_data=graph_list[i])
                graph.opt = torch.tensor(opt_list[i])
                self.data_list.append(graph)

        super().__init__(
            _fake_dataset_root, transform=transform, pre_transform=None, pre_filter=pre_filter,
        )

    @property
    def raw_file_names(self):
        return ["raw_files.pt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def num_classes(self):
        return 1

    def download(self):
        pass

    def __getitem__(self, idx):
        assert isinstance(idx, slice) or isinstance(idx, int)
        if isinstance(idx, int):
            if idx < 0 or idx >= self.size:
                raise IndexError
        if isinstance(idx, slice) and self.data_list is None:
            raise IndexError
        if self.data_list is None:
            data = self.gen_graph(node_data=self.graph_size)
        else:
            data = self.data_list[idx]

        return data

    def __len__(self):
        return self.size
