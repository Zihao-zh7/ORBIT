"""Exact sufficient histories; layouts are chosen for one compatible group."""
from __future__ import annotations
import numpy as np


class HistoryBank:
    def __init__(self, n_nodes, projections, gamma, layout='auto', epoch=0):
        self.projections = np.array(projections, dtype=np.float64, copy=True)
        if self.projections.ndim != 3 or not 0 <= gamma < 1:
            raise ValueError('Expected projections [archive,feature,lead] and 0<=gamma<1')
        self.projections.flags.writeable = False
        self.gamma, self.epoch = float(gamma), epoch
        j, p, d = self.projections.shape
        self.n_nodes = n_nodes
        self.layout = ('feature' if p <= j*d else 'projected') if layout == 'auto' else layout
        if self.layout not in ('feature', 'projected'):
            raise ValueError('Unknown trace layout')
        self.state = np.zeros((n_nodes, p) if self.layout == 'feature' else (j, n_nodes, d), dtype=np.float64)
        self.rounds = 0

    def update(self, features, epoch=0, input_gate=None):
        if epoch != self.epoch:
            raise ValueError('History epoch changed; reset or verified transport required')
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.n_nodes, self.projections.shape[1]) or not np.isfinite(features).all():
            raise ValueError('Invalid feature stream')
        gate = 1-self.gamma if input_gate is None else float(input_gate)
        if not 0 <= gate <= 1:
            raise ValueError('Input gate must be scalar in [0,1]')
        self.state *= self.gamma
        self.state += gate * (features if self.layout == 'feature' else np.einsum('np,jpd->jnd', features, self.projections))
        self.rounds += 1

    def projected(self):
        return np.einsum('np,jpd->jnd', self.state, self.projections) if self.layout == 'feature' else self.state.copy()

    @property
    def payload_bytes(self):
        return self.state.nbytes

    def reset(self, epoch=0):
        self.state.fill(0)
        self.epoch, self.rounds = epoch, 0
