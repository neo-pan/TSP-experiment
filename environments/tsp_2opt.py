from typing import NamedTuple, Tuple

import numpy as np
import torch

from .base import _BaseEnv

FLOAT_SCALE = 10000


class TSP2OPTState(NamedTuple):
    curr_tour: torch.Tensor
    curr_edge_list: torch.Tensor
    curr_tour_len: torch.Tensor
    best_tour: torch.Tensor
    best_edge_list: torch.Tensor
    best_tour_len: torch.Tensor


class TSP2OPTEnv(_BaseEnv):
    def __init__(self, T: int = None, node_pos: torch.Tensor = None, init_tour: torch.Tensor = None) -> None:
        super().__init__()
        self._need_reset = True
        if T is not None and node_pos is not None:
            self.reset(T, node_pos, init_tour)

    def step(self, action: torch.Tensor) -> Tuple[NamedTuple, torch.Tensor, bool, dict]:
        assert not self._need_reset
        assert list(action.shape) == [self.batch_size, 2]
        assert self._step_count < self.T, f"TSP2OPTEnv is already terminated"

        assert action.max() < self.graph_size
        # assert (action[:, 0] != action[:, 1]).all(), action
        # assert ((action.max(1).values - action.min(1).values) != 1).all()

        # take 2-opt action
        self._step_count += 1
        action = action.detach()
        action = action.sort(dim=1).values
        self.apply_2opt(action)

        # get reward, shape=[batch_size, 1]
        reward = ((self.best_tour_len - self.curr_tour_len) / FLOAT_SCALE).float().clamp(min=0.0)

        # update best record
        new_best_mask = (self.curr_tour_len < self.best_tour_len).squeeze()  # shape=[batch_size]

        best_updated = False
        if new_best_mask.any():
            reward[new_best_mask] = reward[new_best_mask].clamp(min=1.0)
            self.best_tour[new_best_mask] = self.curr_tour[new_best_mask].clone()
            self.best_edge_list[new_best_mask] = self.curr_edge_list[new_best_mask].clone()
            self.best_tour_len[new_best_mask] = self.curr_tour_len[new_best_mask].clone()
            best_updated = True

        return (
            TSP2OPTState(
                curr_tour=self.curr_tour.clone(),
                curr_edge_list=self.curr_edge_list.clone(),
                curr_tour_len=self.curr_tour_len.float() / FLOAT_SCALE,
                best_tour=self.best_tour.clone(),
                best_edge_list=self.best_edge_list.clone(),
                best_tour_len=self.best_tour_len.float() / FLOAT_SCALE,
            ),
            reward,
            self.done,
            {"best_updated": best_updated},
        )

    def apply_2opt(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # gather removed edges from previous edge list
        edge_to_remove = self.curr_edge_list.gather(1, action[..., None].expand(self.batch_size, 2, 2))

        # get mask of partial tour that need flip
        low = action[:, 0][:, None].expand(self.batch_size, self.graph_size)
        high = action[:, 1][:, None].expand(self.batch_size, self.graph_size)
        idx = torch.arange(self.graph_size, device=self.device)[None, :].expand(self.batch_size, self.graph_size)
        mask = torch.logical_and(idx > low, idx <= high)

        # update tour and edge list
        flipped_part = self.curr_tour.fliplr().masked_select(mask.fliplr())
        self.curr_tour.masked_scatter_(mask, flipped_part)
        self.curr_edge_list = torch.stack([self.curr_tour, self.curr_tour.roll(-1, [1])], dim=2)

        # gather new edges from current edge list
        edge_to_add = self.curr_edge_list.gather(1, action[..., None].expand(self.batch_size, 2, 2))

        # compute new tour length
        self.curr_tour_len = (
            self.curr_tour_len - self._get_len_of_edge_list(edge_to_remove) + self._get_len_of_edge_list(edge_to_add)
        )

    @property
    def done(self) -> bool:
        assert not self._need_reset
        assert self._step_count < self.T
        if self._step_count == (self.T - 1):
            self._need_reset = True
            return True
        return False

    def random_action(self) -> torch.Tensor:
        assert not self._need_reset
        mask = torch.ones((self.batch_size, self.graph_size), device=self.device)
        action1 = mask.multinomial(1)

        forbid = torch.cat([action1, action1 + 1, action1 - 1], dim=1) % self.graph_size
        mask.scatter_(dim=-1, index=forbid, value=0)

        action2 = mask.multinomial(1)
        action = torch.cat([action1, action2], dim=1)

        return action

    def reset(self, T: int, node_pos: torch.Tensor, init_tour: torch.Tensor = None) -> TSP2OPTState:
        if self._need_reset:
            self._need_reset = False
        self.T = T
        self.batch_size, self.graph_size, pos_dim = node_pos.shape
        self.device = node_pos.device
        assert pos_dim == 2
        self._node_pos = node_pos.detach()
        self._distance_matrix = (torch.cdist(self._node_pos, self._node_pos, p=2.0) * FLOAT_SCALE).long()
        self._step_count = 0

        if init_tour is not None:
            assert init_tour.dim() == 2
            assert init_tour.size(0) == self.batch_size
            assert init_tour.size(1) == self.graph_size
            self.curr_tour = init_tour.detach().clone()
        else:
            self.curr_tour = torch.rand((self.batch_size, self.graph_size), device=self.device).argsort(dim=1)
        self.best_tour = self.curr_tour.clone()

        self.curr_edge_list = torch.stack(
            [self.curr_tour, self.curr_tour.roll(-1, [1])], dim=2
        )  # [batch_size, graph_size, 2]
        self.best_edge_list = self.curr_edge_list.clone()

        self.curr_tour_len = self._get_len_of_edge_list(self.curr_edge_list)
        self.best_tour_len = self.curr_tour_len.clone()

        return TSP2OPTState(
            curr_tour=self.curr_tour.clone(),
            curr_edge_list=self.curr_edge_list.clone(),
            curr_tour_len=self.curr_tour_len.float() / FLOAT_SCALE,
            best_tour=self.best_tour.clone(),
            best_edge_list=self.best_edge_list.clone(),
            best_tour_len=self.best_tour_len.float() / FLOAT_SCALE,
        )

    def _get_len_of_edge_list(self, edge_list: torch.Tensor) -> torch.Tensor:
        assert edge_list.dim() == 3
        assert edge_list.size(0) == self.batch_size
        assert edge_list.size(2) == 2
        num_edge = edge_list.size(1)

        return (
            self._distance_matrix.gather(
                1, edge_list[..., 0][..., None].expand((self.batch_size, num_edge, self.graph_size))
            )
            .gather(2, edge_list[..., 1][..., None])
            .sum(1)
        )  # Compute tour length according to distance matrix and edge list, shape=[batch_size, 1]

    def __repr__(self) -> str:
        return f"TSP 2-opt Environment, with {self.batch_size} graphs of size {self.graph_size}"
