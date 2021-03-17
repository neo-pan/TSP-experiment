import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from typing import Any, Tuple
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils.to_dense_batch import to_dense_batch
from .encoder import cal_size_list, MLP, GNNEncoder


class Baseline(object):
    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def epoch_callback(self, model, epoch):
        pass


class CriticBaseline(nn.Module, Baseline):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.embed_dim = args.embed_dim
        self.num_embed_layers = args.num_embed_layers
        self.num_gnn_layers = args.num_gnn_layers
        self.encoder_num_heads = args.encoder_num_heads
        self.bias = args.bias
        self.pooling_method = args.pooling_method
        assert self.pooling_method in ["add", "max", "mean"]
        self.pooling_func = globals()[f"global_{self.pooling_method}_pool"]
        linear_size_list = cal_size_list(self.input_dim, self.embed_dim, self.num_embed_layers)
        self.linear_embedder = MLP(linear_size_list, bias=self.bias)
        self.encoder = GNNEncoder(self.embed_dim, self.num_gnn_layers, self.encoder_num_heads)
        out_size_list = cal_size_list(self.embed_dim, 1, 1)
        self.out_proj = MLP(out_size_list, bias=self.bias)

    def forward(self, data: Batch) -> torch.Tensor:
        assert data.x.size(-1) == self.input_dim, f"{data.x.size()} -- {self.input_dim}"
        # Linear embed
        x = self.linear_embedder(data.x)
        edge_index = data.edge_index
        # GNN encoding graph node embeddings
        x = self.encoder(x, edge_index)
        # Pooling graph features
        graph_feat = self.pooling_func(x, data.batch)
        out = self.out_proj(graph_feat)

        return out

    def eval(self, data: Batch, target: torch.Tensor) -> torch.Tensor:

        out = self.forward(data)

        assert out.shape == target.shape, f"{out.shape}, {target.shape}"

        return out.detach(), F.mse_loss(out, target.detach())


class ExponentialBaseline(Baseline):
    def __init__(self, args) -> None:

        super().__init__()
        self.beta = args.exp_beta
        self.v = None

    def eval(self, data: Batch, target: torch.Tensor) -> torch.Tensor:

        if self.v is None:
            v = target.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * target.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, torch.tensor(0, dtype=torch.float, requires_grad=True)  # No loss

    def state_dict(self):
        return {"v": self.v}

    def load_state_dict(self, state_dict):
        self.v = state_dict["v"]

    def to(self, device):
        return self


class RolloutBaseline(nn.Module, Baseline):
    pass
