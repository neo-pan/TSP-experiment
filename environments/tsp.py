#!/usr/bin/env python
# coding=utf-8

from typing import Tuple

import torch
import numpy as np
from typing import NamedTuple

from .base import _BaseEnv


class TSPState(NamedTuple):
    first_node: torch.Tensor
    pre_node: torch.Tensor
    avail_mask: torch.Tensor


class TSPEnv(_BaseEnv):
    def __init__(self, node_pos: torch.Tensor = None) -> None:
        super().__init__()
        self._need_reset = True
        if node_pos is not None:
            self.reset(node_pos)

    def step(self, action: torch.Tensor) -> Tuple[NamedTuple, torch.Tensor, bool, dict]:
        assert not self._need_reset
        assert list(action.shape) == [self.batch_size, 1]
        assert self._step_count <= self.num_nodes, f"TSPEnv is already terminated"
        assert self._avail_mask.gather(1, action).all(), f"Action not available, {self._avail_mask.gather(1, action)}"

        self._step_count += 1
        self._act_list.append(action)
        self._avail_mask[self._batch_idx, action.squeeze()] = False
        
        # If not first step, update tour_len
        if self._step_count > 1:
            last_act = self._act_list[-2]
            self.tour_len += self._distance_matrix[self._batch_idx, last_act.squeeze(), action.squeeze()]
        
        # If last step, complete tour_len, mask `done` as true
        if self._step_count == (self.num_nodes):
            done = True
            self.tour_len += self._distance_matrix[self._batch_idx, action.squeeze(), self._act_list[0].squeeze()]
        elif self._step_count > (self.num_nodes):
            raise ValueError
        else:
            done = False
        

        reward = -self.tour_len if done else torch.zeros(self.batch_size)

        return (
            TSPState(
                first_node=self._act_list[0],
                pre_node=action,
                avail_mask=self._avail_mask.clone(),
            ),
            reward,
            done,
            {},
        )

    @property
    def done(self) -> bool:
        assert not self._need_reset
        assert self._step_count <= self._node_num
        if self._step_count == (self._node_num):
            self._need_reset = True
            return True
        return False

    def random_action(self) -> torch.Tensor:
        assert not self._need_reset
        action = self._avail_mask.type(torch.float).multinomial(1)
        assert self._avail_mask.gather(1, action).all()
        return action

    def reset(self, node_pos: torch.Tensor) -> TSPState:
        if self._need_reset:
            self._need_reset = False
        self.batch_size, self.num_nodes, pos_dim = node_pos.shape
        self.device = node_pos.device
        assert pos_dim == 2
        self._node_pos = node_pos
        self._distance_matrix = (node_pos[:, :, None, :] - node_pos[:, None, :, :]).norm(p=2, dim=-1)
        self._step_count = 0
        self._avail_mask = torch.ones((self.batch_size, self.num_nodes), dtype=np.bool, device=self.device)
        self._batch_idx = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        self._act_list = []
        self.tour_len = torch.zeros((self.batch_size), dtype=torch.float, device=self.device)

        return TSPState(
            first_node=None,
            pre_node=None,
            avail_mask=self._avail_mask.clone(),
        )

    def __repr__(self) -> str:
        return f"TSP Environment, with {self.batch_size} graphs of size {self.num_nodes}"
