#!/usr/bin/env python
# coding=utf-8


class _BaseEnv(object):

    reward_range = (-float("inf"), float("inf"))
    action_space = None
    observation_space = None

    def __init__(self):
        pass

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError
