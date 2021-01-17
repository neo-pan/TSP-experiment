from networkx.linalg.laplacianmatrix import directed_combinatorial_laplacian_matrix
from datasets.tsp import gen_fully_connected_graph, TSPDataset
from torch_geometric.transforms import Distance
from torch_geometric.utils import to_networkx
import networkx.algorithms.tree.mst as mst
import torch

def gen_mst_len_dataset(size: int, min_num_node: int, max_num_node: int)-> TSPDataset:
    dataset = TSPDataset(size=size, min_num_node=min_num_node, max_num_node=max_num_node, transform=Distance(cat=False))
    datalist = []
    for graph in dataset:
        ng = to_networkx(graph, edge_attrs=["edge_attr"], to_undirected=True)
        ng_mst = mst.minimum_spanning_tree(ng, "edge_attr")
        mst_len = sum([edge_value["edge_attr"] for edge_value in ng_mst.edges.values()])
        graph.mst_len = torch.tensor([mst_len], dtype=torch.float)
        datalist.append(graph)

    dataset.__data_list__ = datalist
    dataset.data, dataset.slices = dataset.collate(datalist)

    return dataset