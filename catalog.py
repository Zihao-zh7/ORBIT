"""Exact shared-catalog supports, evaluated numerically in float64.

The source operators B are already embedded in the canonical catalog. Null is
encoded as -1; its operator row and column are zero. Assignment combinatorics
are exact, but floating-point evaluations are not outward-rounded certificates.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class SupportResult:
    lower: np.ndarray
    upper: np.ndarray
    lower_coeff: np.ndarray
    upper_coeff: np.ndarray
    lower_witness: np.ndarray | None = None
    upper_witness: np.ndarray | None = None
    numerical_only: bool = True

    def __post_init__(self):
        if any(not np.all(np.isfinite(array)) for array in
               (self.lower, self.upper, self.lower_coeff, self.upper_coeff)):
            raise FloatingPointError("support overflowed or contains nonfinite coefficients")
        for array in (self.lower, self.upper, self.lower_coeff, self.upper_coeff,
                      self.lower_witness, self.upper_witness):
            if array is not None:
                array.flags.writeable = False

    @property
    def center(self):
        return self.lower * 0.5 + self.upper * 0.5

    @property
    def radius(self):
        return float(np.max(self.upper * 0.5 - self.lower * 0.5))


@dataclass(frozen=True)
class AssignmentBranch:
    """A feasible rooted assignment graph, with optional fixed-node removal."""

    output: int
    root: int
    rows: np.ndarray
    rights: tuple
    edge_rows: np.ndarray
    edge_cols: np.ndarray
    edge_labels: np.ndarray
    fixed: np.ndarray
    feasible_columns: np.ndarray


class CatalogProblem:
    """A finite nonempty family of domain-constrained partial injections.

    Parameters have shapes B[J,L,L], W[J,N,d], G[N,N], scale[N,d].
    Gate-independent arrays are copied; create a new problem after changing
    traces, operators, domains, or scale. A null may be repeated, a real ID may
    not. Domain constraints are the entire uncertain lineage model.

    cache_bytes bounds cached NumPy payload, split between complete supports
    and branch coefficients; Python object overhead is additional. Set it to
    zero to disable memoization. Entries are also capped at 16 supports and
    8192 branch coordinates. Cached/public support arrays are read-only.
    """

    def __init__(self, B, W, G, scale, domains, cache_bytes=32 * 1024**2):
        if not isinstance(cache_bytes, (int, np.integer)) or cache_bytes < 0:
            raise ValueError("cache_bytes must be a nonnegative integer")
        self._cache_limit = int(cache_bytes) // 2
        self._support_cache = OrderedDict()
        self._branch_cache = OrderedDict()
        self._support_cache_bytes = self._branch_cache_bytes = 0
        self._support_cache_hits = self._branch_cache_hits = 0
        self.B = np.array(B, dtype=np.float64, copy=True)
        self.W = np.array(W, dtype=np.float64, copy=True)
        self.G = np.array(G, dtype=np.float64, copy=True)
        self.scale = np.array(scale, dtype=np.float64, copy=True)
        if self.B.ndim != 3 or self.B.shape[1] != self.B.shape[2]:
            raise ValueError("B must have shape [J,L,L]")
        self.J, self.L, _ = self.B.shape
        if self.W.ndim != 3 or self.W.shape[0] != self.J:
            raise ValueError("W must have shape [J,N,d]")
        _, self.N, self.d = self.W.shape
        if min(self.J, self.N, self.d) < 1:
            raise ValueError("J, N and d must be positive")
        if self.G.shape != (self.N, self.N):
            raise ValueError("G must have shape [N,N]")
        if self.scale.shape != (self.N, self.d):
            raise ValueError("scale must have shape [N,d]")
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("scale must be positive and finite")
        if any(not np.all(np.isfinite(x)) for x in (self.B, self.W, self.G)):
            raise ValueError("B, W and G must be finite")
        if len(domains) != self.N:
            raise ValueError("one domain is required per current node")
        normalized = []
        for domain in domains:
            values = tuple(domain)
            if not values:
                raise ValueError("empty lineage domain")
            if any(not isinstance(a, (int, np.integer)) or isinstance(a, bool)
                   or a < -1 or a >= self.L for a in values):
                raise ValueError("domain IDs must be integers in [-1,L)")
            normalized.append(tuple(sorted(set(int(a) for a in values))))
        self.domains = tuple(normalized)
        self._effective_domains = self._eliminate_resolved(self.domains)
        self._anchor = np.einsum("il,jlk->jik", self.G, self.W, optimize=True)
        self._branches = {}
        roots = []
        for i, domain in enumerate(self._effective_domains):
            feasible = []
            for a in domain:
                branch = self._make_branch(i, a, True)
                if branch is not None:
                    self._branches[(i, a, True)] = branch
                    feasible.append(a)
            if not feasible:
                raise ValueError("infeasible domains: the lineage family is empty")
            roots.append(tuple(feasible))
        self.feasible_roots = tuple(roots)
        self.is_singleton = all(len(domain) == 1 for domain in self.feasible_roots)
        self._singleton_q = (tuple(domain[0] for domain in self.feasible_roots)
                             if self.is_singleton else None)
        self._singleton_coeff = (self.coefficients(self._singleton_q)
                                 if self.is_singleton else None)
        for array in (self.B, self.W, self.G, self.scale, self._anchor):
            array.flags.writeable = False

    @staticmethod
    def _eliminate_resolved(domains):
        reduced = [set(domain) for domain in domains]
        while True:
            fixed = [next(iter(domain)) for domain in reduced if len(domain) == 1]
            real = [a for a in fixed if a >= 0]
            if len(real) != len(set(real)):
                raise ValueError("infeasible singleton assignments violate injectivity")
            used = set(real)
            changed = False
            for domain in reduced:
                if len(domain) > 1:
                    smaller = domain - used
                    if not smaller:
                        raise ValueError("infeasible domains: the lineage family is empty")
                    if smaller != domain:
                        domain.intersection_update(smaller)
                        changed = True
            if not changed:
                return tuple(tuple(sorted(domain)) for domain in reduced)

    def _make_branch(self, i, a, eliminate_resolved):
        domains = self._effective_domains if eliminate_resolved else self.domains
        fixed = np.full(self.N, -2, dtype=np.int64)
        if eliminate_resolved:
            for node, domain in enumerate(domains):
                if len(domain) == 1:
                    fixed[node] = domain[0]
        fixed[i] = a
        occupied = fixed[fixed >= 0]
        if len(occupied) != len(set(occupied.tolist())):
            return None
        occupied = set(occupied.tolist())
        rows = np.flatnonzero(fixed == -2)
        labels_by_row = [tuple(b for b in domains[node] if b == -1 or b not in occupied)
                         for node in rows]
        if any(not labels for labels in labels_by_row):
            return None
        # A private null column cannot steal another node's null capacity.
        right_set = {b if b >= 0 else self.L + int(node)
                     for node, labels in zip(rows, labels_by_row) for b in labels}
        rights = tuple(sorted(right_set))
        right_index = {b: col for col, b in enumerate(rights)}
        edge_rows, edge_cols, edge_labels = [], [], []
        for row, (node, labels) in enumerate(zip(rows, labels_by_row)):
            for b in labels:
                edge_rows.append(row)
                edge_cols.append(right_index[b if b >= 0 else self.L + int(node)])
                edge_labels.append(b)
        edge_rows = np.array(edge_rows, dtype=np.int64)
        edge_cols = np.array(edge_cols, dtype=np.int64)
        edge_labels = np.array(edge_labels, dtype=np.int64)
        if len(rows):
            costs = np.full((len(rows), len(rights)), np.inf)
            costs[edge_rows, edge_cols] = 0.0
            try:
                selected_rows, selected_cols = linear_sum_assignment(costs)
            except ValueError:
                return None
            if len(selected_rows) != len(rows):
                return None
        else:
            selected_cols = np.empty(0, dtype=np.int64)
        for array in (rows, edge_rows, edge_cols, edge_labels, fixed, selected_cols):
            array.flags.writeable = False
        return AssignmentBranch(i, a, rows, rights, edge_rows, edge_cols,
                                edge_labels, fixed, selected_cols)

    def branches(self, eliminate_resolved=True):
        """Iterate feasible roots, exposing graph data for the independent LP."""
        for i, roots in enumerate(self.feasible_roots):
            for a in roots:
                key = (i, a, bool(eliminate_resolved))
                if key not in self._branches:
                    branch = self._make_branch(i, a, bool(eliminate_resolved))
                    if branch is None:
                        raise RuntimeError("feasible rooted graph unexpectedly vanished")
                    self._branches[key] = branch
                yield self._branches[key]

    def branch_coefficients(self, branch, k):
        """Return affine constant [J] and edge coefficients [edges,J]."""
        key = (id(branch), k)
        if key in self._branch_cache:
            self._branch_cache.move_to_end(key)
            self._branch_cache_hits += 1
            return self._branch_cache[key][0]
        i, a = branch.output, branch.root
        constant = -self._anchor[:, i, k].copy()
        edges = np.zeros((len(branch.edge_rows), self.J), dtype=np.float64)
        if a >= 0:
            nodes = np.flatnonzero(branch.fixed >= 0)
            identities = branch.fixed[nodes]
            constant += np.sum(self.B[:, a, identities] * self.W[:, nodes, k], axis=1)
            real = branch.edge_labels >= 0
            nodes = branch.rows[branch.edge_rows[real]]
            identities = branch.edge_labels[real]
            edges[real] = (self.B[:, a, identities] * self.W[:, nodes, k]).T
        constant = constant / self.scale[i, k]
        edges = edges / self.scale[i, k]
        constant.flags.writeable = edges.flags.writeable = False
        result = (constant, edges)
        size = constant.nbytes + edges.nbytes
        if size <= self._cache_limit:
            while self._branch_cache and (self._branch_cache_bytes + size > self._cache_limit
                                          or len(self._branch_cache) >= 8192):
                _, (_, old_size) = self._branch_cache.popitem(last=False)
                self._branch_cache_bytes -= old_size
            self._branch_cache[key] = (result, size)
            self._branch_cache_bytes += size
        return result

    def cache_info(self):
        """Return cache payload and hits for profiling; object overhead excluded."""
        return {"support_bytes": self._support_cache_bytes,
                "branch_bytes": self._branch_cache_bytes,
                "support_entries": len(self._support_cache),
                "branch_entries": len(self._branch_cache),
                "support_hits": self._support_cache_hits,
                "branch_hits": self._branch_cache_hits,
                "payload_limit_bytes": 2 * self._cache_limit}

    def _remember_support(self, key, result):
        size = sum(array.nbytes for array in (result.lower, result.upper,
                                             result.lower_coeff, result.upper_coeff))
        if size <= self._cache_limit:
            while self._support_cache and (self._support_cache_bytes + size > self._cache_limit
                                           or len(self._support_cache) >= 16):
                _, (_, old_size) = self._support_cache.popitem(last=False)
                self._support_cache_bytes -= old_size
            self._support_cache[key] = (result, size)
            self._support_cache_bytes += size
        return result

    def coefficients(self, q):
        """Compute every gate coefficient for one feasible canonical lineage."""
        q_array = np.asarray(q)
        if q_array.shape != (self.N,) or q_array.dtype.kind not in "iu":
            raise ValueError("q must contain N integer assignments")
        if any(int(a) not in domain for a, domain in zip(q_array, self.domains)):
            raise ValueError("q contains an assignment outside its domain")
        q_array = q_array.astype(np.int64, copy=False)
        real = q_array[q_array >= 0]
        if len(real) != len(set(real.tolist())):
            raise ValueError("q violates partial injectivity")
        residual = -self._anchor.copy()
        nodes = np.flatnonzero(q_array >= 0)
        ids = q_array[nodes]
        if len(nodes):
            pulled = self.B[:, ids[:, None], ids[None, :]]
            residual[:, nodes, :] += np.einsum("jil,jlk->jik", pulled,
                                               self.W[:, nodes, :], optimize=True)
        return np.moveaxis(residual / self.scale[None, :, :], 0, -1)

    def enumerate_lineages(self, max_lineages=100000):
        """Enumerate tiny families for QA; never used by production solvers.

        A bound on the Cartesian search size prevents accidental large runs.
        """
        search_size = 1
        for domain in self.domains:
            search_size *= len(domain)
        if search_size > max_lineages:
            raise ValueError("enumeration search exceeds max_lineages")
        for q in product(*self.domains):
            real = [a for a in q if a >= 0]
            if len(real) == len(set(real)):
                yield q

    def support(self, gates, return_witnesses=False):
        """Compute complete coordinate extrema, with attaining coefficients.

        Negative gates are supported. Witnesses are optional because their
        N-by-d-by-N storage can greatly exceed the support arrays themselves.
        Witness arrays are never retained in the per-problem cache. Repeated
        unit-gate queries share marginal supports across baseline controllers.
        """
        gates = np.asarray(gates, dtype=np.float64)
        if gates.shape != (self.J,) or not np.all(np.isfinite(gates)):
            raise ValueError("gates must be a finite vector of length J")
        key = gates.tobytes()
        if not return_witnesses and key in self._support_cache:
            self._support_cache.move_to_end(key)
            self._support_cache_hits += 1
            return self._support_cache[key][0]
        if self.is_singleton:
            values = self._singleton_coeff @ gates
            witness = (np.broadcast_to(self._singleton_q, (self.N, self.d, self.N)).copy()
                       if return_witnesses else None)
            result = SupportResult(values.copy(), values.copy(), self._singleton_coeff.copy(),
                                   self._singleton_coeff.copy(), witness, witness)
            return result if return_witnesses else self._remember_support(key, result)
        lower = np.full((self.N, self.d), np.inf)
        upper = np.full((self.N, self.d), -np.inf)
        low_coeff = np.empty((self.N, self.d, self.J))
        high_coeff = np.empty_like(low_coeff)
        low_q = np.empty((self.N, self.d, self.N), dtype=np.int64) if return_witnesses else None
        high_q = np.empty_like(low_q) if return_witnesses else None
        for branch in self.branches():
            i = branch.output
            nr, nc = len(branch.rows), len(branch.rights)
            edge_lookup = np.full((nr, nc), -1, dtype=np.int64)
            edge_lookup[branch.edge_rows, branch.edge_cols] = np.arange(len(branch.edge_rows))
            for k in range(self.d):
                constant, edge_coeff = self.branch_coefficients(branch, k)
                edge_values = edge_coeff @ gates
                for sign in (1.0, -1.0):
                    if nr:
                        costs = np.full((nr, nc), np.inf)
                        costs[branch.edge_rows, branch.edge_cols] = -sign * edge_values
                        selected_rows, selected_cols = linear_sum_assignment(costs)
                        selected_edges = edge_lookup[selected_rows, selected_cols]
                        coeff = constant + edge_coeff[selected_edges].sum(axis=0)
                    else:
                        selected_edges = np.empty(0, dtype=np.int64)
                        coeff = constant
                    value = float(coeff @ gates)
                    target = upper if sign > 0 else lower
                    if (sign > 0 and value > target[i, k]) or (sign < 0 and value < target[i, k]):
                        target[i, k] = value
                        (high_coeff if sign > 0 else low_coeff)[i, k] = coeff
                        if return_witnesses:
                            q = branch.fixed.copy()
                            q[branch.rows] = branch.edge_labels[selected_edges]
                            (high_q if sign > 0 else low_q)[i, k] = q
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise FloatingPointError("support overflowed; rescale the input coefficients")
        result = SupportResult(lower, upper, low_coeff, high_coeff, low_q, high_q)
        return result if return_witnesses else self._remember_support(key, result)
