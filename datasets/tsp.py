import os
import os.path as osp
from typing import Tuple

import networkx as nx
import torch
from torch_geometric import data
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import Distance
from torch_geometric.utils import from_networkx, to_undirected

_curdir = osp.dirname(osp.abspath(__file__))
_fake_dataset_root = osp.join(_curdir, "FAKEDataset")


def gen_graph_data(node_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.arange(node_num).unsqueeze(-1).expand([-1, node_num])
    rol = torch.reshape(index, [-1])
    col = torch.reshape(torch.t(index), [-1])
    edge_index = torch.stack([rol, col], dim=0)
    pos = torch.empty(size=(node_num, 2)).uniform_(0, 1)
    node_feat = torch.tensor([[0, 1] if i == 0 else [1, 0] for i in range(node_num)], dtype=torch.float,)

    return edge_index, pos, node_feat


def gen_knn_graph(node_num: int, k=10, pos_feature: bool = True) -> Data:
    edge_index, pos, node_feat = gen_graph_data(node_num)

    edge_index = knn_graph(pos, k=k, loop=True)

    graph = Data(x=torch.cat([node_feat, pos], dim=-1), edge_index=edge_index, pos=pos)
    graph = Distance(norm=False, cat=False)(graph)

    return graph


def gen_complete_graph(node_num: int, pos_feature: bool = True) -> Data:
    edge_index, pos, node_feat = gen_graph_data(node_num)

    graph = Data(x=torch.cat([node_feat, pos], dim=-1), edge_index=edge_index, pos=pos)
    graph = Distance(norm=False, cat=False)(graph)

    return graph


graph_gen_func = {
    "complete": gen_complete_graph,
    "knn": gen_knn_graph,
}


class TSPDataset(Dataset):
    def __init__(self, size, device=None, train_flag=False, transform=None, pre_filter=None, args=None, load_path=None):
        self.size, self.train_flag = size, train_flag
        self.device = device
        self.args = args
        self.load_path = load_path
        self.classification_flag = False
        self.gen_graph = graph_gen_func.get(args.graph_type, None)
        assert self.gen_graph, f"Wrong graph type: {args.graph_type}"
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
        if idx < 0 or idx >= self.size:
            raise IndexError
        data = self.gen_graph(node_num=self.args.graph_size)

        return data

    def __len__(self):
        return self.size
