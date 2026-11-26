"""Independent implementations of the published STID and STAEformer layouts.

Architecture references (accessed 2026-10-12):
https://github.com/zezhishao/STID/blob/master/stid/arch/stid_arch.py
https://github.com/zezhishao/STID/blob/master/stid/arch/mlp.py
https://github.com/XDZhelheim/STAEformer/blob/main/model/STAEformer.py

The external interface always uses [target_z, day_fraction, weekday_integer].
STID rescales weekday to [0,1) before its signal projection, preserving its
official convention. Features are captured immediately before output projection;
STAEformer averages the remaining time dimension as prescribed by ORBIT.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x):
        return x + self.layers(x)


class STID(nn.Module):
    def __init__(self, num_nodes: int, input_steps: int = 12, output_steps: int = 12,
                 slots_per_day: int = 288, identity_free: bool = False,
                 embedding_dim: int = 32, num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.input_steps, self.num_nodes = input_steps, num_nodes
        self.slots_per_day = slots_per_day
        self.signal = nn.Linear(input_steps * 3, embedding_dim)
        self.tod = nn.Embedding(slots_per_day, embedding_dim)
        self.dow = nn.Embedding(7, embedding_dim)
        nn.init.xavier_uniform_(self.tod.weight)
        nn.init.xavier_uniform_(self.dow.weight)
        self.node = None if identity_free else nn.Parameter(torch.empty(num_nodes, embedding_dim))
        if self.node is not None:
            nn.init.xavier_uniform_(self.node)
        self.feature_dim = embedding_dim * (3 if identity_free else 4)
        self.encoder = nn.Sequential(*[ResidualMLP(self.feature_dim, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(self.feature_dim, output_steps)

    def forward(self, x, return_features=False):
        if x.shape[1:] != (self.input_steps, self.num_nodes, 3):
            raise ValueError("STID input must have configured [time,nodes,3] dimensions")
        signal = x.clone()
        signal[..., 2] = signal[..., 2] / 7
        pieces = [self.signal(signal.transpose(1, 2).flatten(2))]
        if self.node is not None:
            pieces.append(self.node.unsqueeze(0).expand(x.shape[0], -1, -1))
        tod = (x[:, -1, :, 1] * self.slots_per_day).long().clamp(0, self.slots_per_day - 1)
        pieces.extend([self.tod(tod), self.dow(x[:, -1, :, 2].long())])
        features = self.encoder(torch.cat(pieces, dim=-1))
        prediction = self.head(features)
        return (prediction, features) if return_features else prediction


class ProjectedAttention(nn.Module):
    """Separate linear Q/K/V/output projections with the original initialization."""
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("attention width must divide evenly into heads")
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = [projection(x).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
                   for projection in (self.query, self.key, self.value)]
        weights = torch.softmax((q @ k.transpose(-1, -2)) / self.head_width ** 0.5, dim=-1)
        joined = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.output(joined)


class AxisAttention(nn.Module):
    """Q/K/V attention, residual post-norm, pointwise feed-forward post-norm."""
    def __init__(self, width: int, heads: int, feed_forward_dim: int, dropout: float):
        super().__init__()
        self.attention = ProjectedAttention(width, heads)
        self.ff = nn.Sequential(nn.Linear(width, feed_forward_dim), nn.ReLU(),
                                nn.Linear(feed_forward_dim, width))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, axis: int):
        arranged = x.transpose(axis, -2)
        shape = arranged.shape
        flat = arranged.reshape(-1, shape[-2], shape[-1])
        attention = self.attention(flat)
        hidden = self.norm1(flat + self.drop1(attention))
        hidden = self.norm2(hidden + self.drop2(self.ff(hidden)))
        return hidden.reshape(shape).transpose(axis, -2)


class STAEformer(nn.Module):
    def __init__(self, num_nodes: int, input_steps: int = 12, output_steps: int = 12,
                 slots_per_day: int = 288, identity_free: bool = False,
                 embedding_dim: int = 24, adaptive_dim: int = 80,
                 feed_forward_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_steps, self.num_nodes = input_steps, num_nodes
        self.slots_per_day = slots_per_day
        self.signal = nn.Linear(3, embedding_dim)
        self.tod, self.dow = nn.Embedding(slots_per_day, embedding_dim), nn.Embedding(7, embedding_dim)
        adaptive_dim = 0 if identity_free else adaptive_dim
        self.adaptive = None
        if adaptive_dim:
            self.adaptive = nn.Parameter(torch.empty(input_steps, num_nodes, adaptive_dim))
            nn.init.xavier_uniform_(self.adaptive)
        self.feature_dim = 3 * embedding_dim + adaptive_dim
        if self.feature_dim % num_heads:
            raise ValueError("feature width must be divisible by attention head count")
        # The original module initializes its mixed output projection first.
        self.head = nn.Linear(input_steps * self.feature_dim, output_steps)
        make_layer = lambda: AxisAttention(self.feature_dim, num_heads, feed_forward_dim, dropout)
        self.temporal = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.spatial = nn.ModuleList([make_layer() for _ in range(num_layers)])

    def forward(self, x, return_features=False):
        if x.shape[1:] != (self.input_steps, self.num_nodes, 3):
            raise ValueError("STAEformer input must have configured [time,nodes,3] dimensions")
        tod = (x[..., 1] * self.slots_per_day).long().clamp(0, self.slots_per_day - 1)
        pieces = [self.signal(x), self.tod(tod), self.dow(x[..., 2].long())]
        if self.adaptive is not None:
            pieces.append(self.adaptive.unsqueeze(0).expand(x.shape[0], -1, -1, -1))
        hidden = torch.cat(pieces, dim=-1)
        for layer in self.temporal:
            hidden = layer(hidden, axis=1)
        for layer in self.spatial:
            hidden = layer(hidden, axis=2)
        features = hidden.mean(dim=1)
        prediction = self.head(hidden.transpose(1, 2).flatten(2))
        return (prediction, features) if return_features else prediction


def make_backbone(name: str, **kwargs) -> nn.Module:
    constructors = {"stid": STID, "staeformer": STAEformer}
    if name.lower() not in constructors:
        raise ValueError(f"unsupported backbone {name!r}")
    return constructors[name.lower()](**kwargs)
