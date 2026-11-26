"""Equation (graph-construction): correlations use pairwise observed data only."""
from __future__ import annotations

from collections import deque
import numpy as np


def _from_moments(count, sums, squares, products, window_length, k):
    n = len(count)
    safe = np.maximum(count, 1)
    covariance = products - sums * sums.T / safe
    variance = np.maximum(squares - sums * sums / safe, 0)
    cancellation = 16*np.finfo(np.float64).eps*np.maximum(squares, sums*sums/safe)
    variance[variance <= cancellation] = 0
    denominator = np.sqrt(variance * variance.T)
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-20)
    correlation[count < max(16, (window_length + 1) // 2)] = 0
    correlation = np.clip(correlation, -1, 1)
    np.fill_diagonal(correlation, 0)
    selected = np.zeros((n, n), dtype=bool)
    # Stable order ensures the tie rule is observable channel order.
    for i in range(n):
        neighbors = np.argsort(-correlation[i], kind='stable')[:k]
        neighbors = neighbors[correlation[i, neighbors] > 0]
        selected[i, neighbors] = True
    adjacency = np.where(selected | selected.T, np.maximum(correlation, 0), 0)
    np.fill_diagonal(adjacency, 1)
    return adjacency / adjacency.sum(axis=1, keepdims=True)


def correlation_graph(values, valid=None, k=8):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or k < 0:
        raise ValueError('Expected a [time,node] window and nonnegative k')
    mask = np.isfinite(values) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(values)
    if mask.shape != values.shape:
        raise ValueError('Validity mask shape differs from values')
    # A fixed shift per column does not alter pairwise-centered covariance.
    # It avoids catastrophic cancellation for constant/large-offset channels.
    first = np.argmax(mask, axis=0) if len(values) else np.zeros(values.shape[1], dtype=int)
    offsets = np.where(mask.any(axis=0), values[first,np.arange(values.shape[1])],0) if len(values) else np.zeros(values.shape[1])
    observed = np.where(mask, values-offsets, 0)
    mask = mask.astype(np.float64)
    return _from_moments(mask.T @ mask, observed.T @ mask,
                         (observed * observed).T @ mask, observed.T @ observed, len(values), k)


class RollingGraph:
    """O(N²) update of exact pairwise sufficient moments for a trailing window."""
    def __init__(self, window, n_nodes, k=8):
        if window < 1:
            raise ValueError('window must be positive')
        self.window, self.k = int(window), int(k)
        self.queue = deque()
        self.count = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        self.sums = np.zeros_like(self.count)
        self.squares = np.zeros_like(self.count)
        self.products = np.zeros_like(self.count)
        self.steps = 0
        self.offsets = np.full(n_nodes,np.nan)

    def _accumulate(self, x, m, sign):
        self.count += sign * np.outer(m, m)
        self.sums += sign * np.outer(x, m)
        self.squares += sign * np.outer(x*x, m)
        self.products += sign * np.outer(x, x)

    def update(self, values, valid=None):
        values = np.asarray(values, dtype=np.float64)
        m = np.isfinite(values) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(values)
        new = m & np.isnan(self.offsets)
        self.offsets[new] = values[new]
        x = np.where(m, values-np.nan_to_num(self.offsets), 0)
        m = m.astype(np.float64)
        if len(self.queue) == self.window:
            old = self.queue.popleft()
            self._accumulate(*old, -1)
        self.queue.append((x.copy(), m.copy()))
        self._accumulate(x, m, 1)
        self.steps += 1
        # Bound rolling cancellation by periodically rebuilding moments.
        if self.steps % 4096 == 0:
            x, m = map(np.asarray, zip(*self.queue))
            self.count = m.T @ m
            self.sums = x.T @ m
            self.squares = (x*x).T @ m
            self.products = x.T @ x

    def operator(self):
        return _from_moments(self.count, self.sums, self.squares, self.products, len(self.queue), self.k)


def transport_operator(operator, q):
    operator = np.asarray(operator, dtype=np.float64)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in q):
        raise ValueError('Identity values must be integers')
    q = np.asarray(q, dtype=int)
    nonnull = q >= 0
    if len(set(q[nonnull])) != int(nonnull.sum()) or np.any(q < -1) or np.any(q >= len(operator)):
        raise ValueError('q must be a partial injection with -1 for null')
    result = np.zeros((len(q), len(q)), dtype=np.float64)
    result[np.ix_(nonnull, nonnull)] = operator[np.ix_(q[nonnull], q[nonnull])]
    return result


def archive_operators(values, valid, train_end, window, k=8, count=4, mode='default'):
    if train_end < count * window:
        raise ValueError('Training partition is too short for the paper archive windows')
    end = train_end if mode != 'stale' else count * window
    if mode not in ('default', 'stale', 'weak', 'all-null'):
        raise ValueError(f'Unknown archive mode: {mode}')
    operators = []
    for j in range(count):
        start = end - (count-j)*window
        op = correlation_graph(values[start:start+window], valid[start:start+window], k)
        if mode == 'weak':
            # Explicit reconstruction assumption: 90% identity, 10% archival graph.
            op = .9*np.eye(len(op)) + .1*op
        elif mode == 'all-null':
            op[:] = 0
        operators.append(op)
    return np.asarray(operators)
