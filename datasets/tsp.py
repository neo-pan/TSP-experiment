#!/usr/bin/env python
# coding=utf-8

import os
import os.path as osp

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import __repr__, files_exist
from torch_geometric.transforms import Distance
from torch_geometric.utils import from_networkx

_curdir = osp.dirname(osp.abspath(__file__))
_fake_dataset_root = osp.join(_curdir, "FAKEDataset")


def gen_random_graph(graph_size: int) -> Data:
    g = nx.complete_graph(graph_size)
    pos = torch.FloatTensor(size=(graph_size, 2)).uniform_(0, 1)
    graph = from_networkx(g)
    graph.pos = pos
    node_feat = torch.tensor(
        [[0, 1] if i == 0 else [1, 0] for i in range(graph_size)],
        dtype=torch.float,
    )
    graph.x = torch.cat([node_feat, pos], dim=-1)

    return graph


def gen_fully_connected_graph(
    node_num: int, device: torch.device = None, pos_feature: bool = True
) -> Data:
    index = torch.arange(node_num, device=device).unsqueeze(-1).expand([-1, node_num])
    rol = torch.reshape(index, [-1])
    col = torch.reshape(torch.t(index), [-1])
    edge_index = torch.stack([rol, col], dim=0)
    pos = torch.empty(size=(node_num, 2), device=device).uniform_(0, 1)
    node_feat = torch.tensor(
        [[0, 1] if i == 0 else [1, 0] for i in range(node_num)],
        dtype=torch.float,
        device=device,
    )

    graph = Data(x=torch.cat([node_feat, pos], dim=-1), edge_index=edge_index, pos=pos)

    return graph


class TSPDataset(InMemoryDataset):
    def __init__(
        self,
        size,
        device=None,
        train_flag=False,
        transform=None,
        pre_filter=None,
        **kwargs
    ):
        self.size, self.train_flag = size, train_flag
        self.device = device
        self.other_kwargs = kwargs
        self.classification_flag = False
        super(InMemoryDataset, self).__init__(
            _fake_dataset_root,
            transform=transform,
            pre_transform=None,
            pre_filter=pre_filter,
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

    def generate_data_list(self):
        data_list = [self._create(idx) for idx in range(self.size)]
        return data_list

    def process(self):
        data_list = self.generate_data_list()
        self.__data_list__ = data_list
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        self.data, self.slices = self.collate(data_list)
        self.data, self.slices = (
            self.data.to(self.device),
            {k: v.to(self.device) for k, v in self.slices.items()},
        )
        # torch.save((data, slices), self.processed_paths[0])

    def _process(self):
        if files_exist(self.processed_paths):
            for _path in self.processed_paths:
                os.remove(_path)
        super()._process()

    def _create(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError
        min_num_node, max_num_node = (
            self.other_kwargs["min_num_node"],
            self.other_kwargs["max_num_node"],
        )
        node_num = int(idx / self.size * (max_num_node - min_num_node) + min_num_node)
        data = gen_fully_connected_graph(node_num=node_num, device=self.device)

        return data
