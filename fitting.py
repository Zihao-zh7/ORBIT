"""Buffered FP64 normal equations for the masked, scaled residual objective."""
from __future__ import annotations

import numpy as np


class BlockRidgeAccumulator:
    """Accumulate archive and current designs in shared temporal blocks.

    ``p_archive`` and ``p_current`` are complete design widths (normally 4*p
    and p). An update copies one origin. ``finish`` flushes the partial block
    and returns two independent ``(gram, rhs, counts)`` snapshots, with shapes
    [lead,width,width], [lead,width], [lead]. Counts remain unnormalized so
    ``solve_ridge_statistics`` applies the paper's per-lead normalization once.

    Only the floating-point summation order changes from per-origin fitting;
    all rows, masks, per-node/lead scales and ridge objectives are preserved.
    A finite residual is required for a target to contribute, as in
    ``residual.ridge_statistics``. Repeated finish calls do not double count.
    """

    def __init__(self, p_archive, p_current, scale, block_size=32):
        widths = (p_archive, p_current)
        if any(not isinstance(p, (int, np.integer)) or p < 1 for p in widths):
            raise ValueError('Design widths must be positive integers')
        if not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError('Block size must be a positive integer')
        scale = np.asarray(scale, dtype=np.float64)
        if scale.ndim != 2 or min(scale.shape) < 1 or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Scale must be a positive finite [node,lead] matrix')
        self.n_nodes, self.n_leads = scale.shape
        self.widths, self.block_size = widths, int(block_size)
        self.inverse_scale = 1 / scale.copy()
        self.designs = [np.empty((block_size, self.n_nodes, width), dtype=np.float64)
                        for width in widths]
        self.residual = np.empty((block_size, self.n_nodes, self.n_leads), dtype=np.float64)
        self.valid = np.empty(self.residual.shape, dtype=bool)
        self.grams = [np.zeros((self.n_leads, width, width), dtype=np.float64) for width in widths]
        self.right = [np.zeros((self.n_leads, width), dtype=np.float64) for width in widths]
        self.counts = np.zeros(self.n_leads, dtype=np.int64)
        self.buffered = 0

    def update(self, archive_phi, current_phi, residual, valid):
        arrays = [np.asarray(archive_phi, dtype=np.float64), np.asarray(current_phi, dtype=np.float64)]
        residual, valid = np.asarray(residual, dtype=np.float64), np.asarray(valid, dtype=bool)
        if any(x.shape != (self.n_nodes, width) for x, width in zip(arrays, self.widths)):
            raise ValueError('Design shape differs from configured node count or width')
        if residual.shape != (self.n_nodes, self.n_leads) or valid.shape != residual.shape:
            raise ValueError('Residual and validity must match scale dimensions')
        if any(not np.isfinite(x).all() for x in arrays):
            raise ValueError('Designs must contain finite values')
        for target, source in zip(self.designs, arrays):
            target[self.buffered] = source
        self.residual[self.buffered] = residual
        self.valid[self.buffered] = valid & np.isfinite(residual)
        self.buffered += 1
        if self.buffered == self.block_size:
            self._flush()

    def _flush(self):
        if not self.buffered:
            return
        designs = [x[:self.buffered].reshape(-1, width) for x, width in zip(self.designs, self.widths)]
        residual = self.residual[:self.buffered].reshape(-1, self.n_leads)
        valid = self.valid[:self.buffered].reshape(-1, self.n_leads)
        # A flattened temporal block repeats the same node-dependent weights.
        inverse = np.broadcast_to(self.inverse_scale, (self.buffered, self.n_nodes, self.n_leads))
        inverse = inverse.reshape(-1, self.n_leads)
        for h in range(self.n_leads):
            mask = valid[:, h]
            count = int(mask.sum())
            if count == 0:
                continue
            weights = inverse[mask, h]
            y = residual[mask, h] * weights
            for index, phi in enumerate(designs):
                x = phi[mask]  # Advanced indexing creates a private, contiguous buffer.
                x *= weights[:, None]
                self.grams[index][h] += x.T @ x
                self.right[index][h] += x.T @ y
            self.counts[h] += count
        self.buffered = 0

    def finish(self):
        self._flush()
        return tuple((gram.copy(), rhs.copy(), self.counts.copy())
                     for gram, rhs in zip(self.grams, self.right))
