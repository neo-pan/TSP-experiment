#! /usr/bin/bash

pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-geometric

import torch
device = torch.device("cuda")
kwargs ={"min_num_node":20, "num_num_node":50}
from datasets.tsp import TSPDataset
a = TSPDataset(50, device=device, **kwargs)