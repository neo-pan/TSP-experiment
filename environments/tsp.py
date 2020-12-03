#!/usr/bin/env python
# coding=utf-8

from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .base import _BaseEnv


class TSPEnv(_BaseEnv):
    def __init__(self, node_pos: np.ndarray) -> None:
        assert node_pos.shape[1]==2
        self._node_pos = node_pos
        self._node_num = node_pos.shape[0]
        self._distance_matrix = cdist(node_pos, node_pos, metric="euclidean")
        self._avail_action = set(range(1,self._node_num))
        self._step_seq = 0
        self._visit_seq = np.zeros((self._node_num,1), dtype=np.int)
        self._visit_mask = np.zeros_like(self._visit_seq, dtype=np.bool)
        self._visit_mask[0] = True
        self._act_list = [0]
        self.tour_len = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert self._step_seq<self._node_num, f"TSPEnv is already terminated"
        assert action in self._avail_action, f"Action-{action} is not available"

        self._avail_action.remove(action)
        self._step_seq += 1
        self._visit_seq[action] = self._step_seq
        self._visit_mask[action] = True
        last_act = self._act_list[-1]
        self._act_list.append(action)
        self.tour_len += self._distance_matrix[last_act][action]

        if self._step_seq == (self._node_num-1):
            assert (len(self._avail_action)==0 and len(self._act_list)==self._node_num)
            done = True
            self.tour_len += self._distance_matrix[0][action]
        elif self._step_seq > (self._node_num-1):
            raise ValueError
        else:
            done = False

        reward = -self.tour_len if done else 0.0

        return self._visit_seq.copy(), reward, done, {"mask": self._visit_mask.copy()}

    @property
    def done(self):
        assert self._step_seq < self._node_num
        return self._step_seq == (self._node_num-1)

    def random_action(self) -> int:
        assert self._avail_action, "No available action"
        return np.random.choice(list(self._avail_action))

    def reset(self) -> None:
        self._avail_action = set(range(1,self._node_num))
        self._visit_seq.fill(0)
        self._visit_mask.fill(False)
        self._visit_mask[0] = True
        self._act_list = [0]
        self._step_seq = 0
        self.tour_len = 0
