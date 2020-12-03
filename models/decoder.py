#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn


class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1,) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads)
