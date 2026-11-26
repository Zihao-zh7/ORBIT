"""Exact finite general-lineage oracles, exponential in interaction width.

Table factors include every hard consistency/injectivity relation. Bucket
elimination uses a deterministic min-fill heuristic (not an optimal treewidth
algorithm), keeps argmax tables for witnesses, and refuses oversized buckets.
This module does not extend the polynomial shared-catalog guarantee. All
arithmetic is FP64 and numerical-only, just like the catalog implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import prod

import numpy as np

from .catalog import SupportResult


@dataclass(frozen=True)
class TableFactor:
    """Scalar factor indexed by domain positions; -inf marks a hard rejection."""
    scope: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class ResponseFactor:
    """Coefficient factor with table axes followed by [N,d,J]."""
    scope: tuple[int, ...]
    coefficients: np.ndarray


@dataclass(frozen=True)
class EliminationResult:
    value: float
    assignment: tuple
    order: tuple[int, ...]
    induced_width: int
    max_table_entries: int
    total_table_entries: int


def _domains(domains):
    normalized = tuple(tuple(d) for d in domains)
    if any(not d or len(set(d)) != len(d) for d in normalized):
        raise ValueError("Domains must be nonempty and contain distinct hashable values")
    return normalized


def _scope(scope, variables):
    result = tuple(scope)
    if (any(not isinstance(i, (int, np.integer)) or not 0 <= i < variables for i in result)
            or len(set(result)) != len(result)):
        raise ValueError("Factor scope must contain distinct valid variable indices")
    return result


def _min_fill(variables, scopes):
    neighbors = {v: set() for v in range(variables)}
    for scope in scopes:
        for a, b in combinations(scope, 2):
            neighbors[a].add(b)
            neighbors[b].add(a)
    remaining, order = set(neighbors), []
    while remaining:
        def cost(v):
            adjacent = neighbors[v] & remaining
            fill = sum(b not in neighbors[a] for a, b in combinations(adjacent, 2))
            return fill, len(adjacent), v
        chosen = min(remaining, key=cost)
        adjacent = neighbors[chosen] & remaining
        for a, b in combinations(adjacent, 2):
            neighbors[a].add(b)
            neighbors[b].add(a)
        remaining.remove(chosen)
        order.append(chosen)
    return tuple(order)


def max_sum(domains, factors, order=None, max_table_entries=1_000_000):
    """Compute a complete max-sum support and one feasible maximizing witness.

    The limit applies to each input/intermediate scalar table. It is a declared
    exponential-work guard, not a timeout or a safe approximate support bound.
    An oversized or infeasible problem raises ValueError; callers must dispatch
    elsewhere or use zero archival excess, never treat incomplete search as an
    enclosing support interval.
    """
    domains = _domains(domains)
    n = len(domains)
    if not isinstance(max_table_entries, (int, np.integer)) or max_table_entries < 1:
        raise ValueError("max_table_entries limit must be a positive integer")
    active = []
    for factor in factors:
        scope = _scope(factor.scope, n)
        values = np.asarray(factor.values, dtype=float)
        if values.shape != tuple(len(domains[i]) for i in scope):
            raise ValueError("Factor table shape does not match scope domains")
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("Factors allow finite values and negative infinity only")
        if values.size > max_table_entries:
            raise ValueError("Input factor exceeds max_table_entries limit")
        active.append(TableFactor(scope, values))
    if order is None:
        order = _min_fill(n, [f.scope for f in active])
    else:
        order = tuple(order)
        if set(order) != set(range(n)) or len(order) != n:
            raise ValueError("Elimination order must be a permutation of all variables")
    trace, width, largest, total = [], 0, 1, 0
    for variable in order:
        bucket = [f for f in active if variable in f.scope]
        active = [f for f in active if variable not in f.scope]
        # Include isolated variables so the witness covers every variable.
        union = tuple(sorted({variable} | {v for f in bucket for v in f.scope}))
        shape = tuple(len(domains[v]) for v in union)
        entries = prod(shape)
        if entries > max_table_entries:
            raise ValueError(f"Elimination bucket {union} exceeds max_table_entries limit ({entries})")
        width = max(width, len(union)-1)
        largest, total = max(largest, entries), total + entries
        accumulated = np.zeros(shape)
        for factor in bucket:
            # Reorder arbitrary factor scopes before inserting broadcast axes.
            sorted_scope = tuple(v for v in union if v in factor.scope)
            axes = tuple(factor.scope.index(v) for v in sorted_scope)
            values = np.transpose(factor.values, axes) if axes else factor.values
            broadcast = tuple(len(domains[v]) if v in factor.scope else 1 for v in union)
            accumulated += values.reshape(broadcast)
        axis = union.index(variable)
        remaining = tuple(v for v in union if v != variable)
        trace.append((variable, remaining, np.argmax(accumulated, axis=axis)))
        active.append(TableFactor(remaining, np.max(accumulated, axis=axis)))
    optimum = float(sum(float(f.values) for f in active))
    if not np.isfinite(optimum):
        raise ValueError("General lineage factor family is infeasible")
    indices = {}
    for variable, remaining, choices in reversed(trace):
        key = tuple(indices[v] for v in remaining)
        indices[variable] = int(choices[key])
    witness = tuple(domains[v][indices[v]] for v in range(n))
    return EliminationResult(optimum, witness, order, width, largest, total)


def all_different_factors(domains, scope, null=-1):
    """Pairwise injectivity factors; repeated private-null values remain legal."""
    domains = _domains(domains)
    scope = _scope(scope, len(domains))
    factors = []
    for a, b in combinations(scope, 2):
        table = np.zeros((len(domains[a]), len(domains[b])))
        for ia, va in enumerate(domains[a]):
            for ib, vb in enumerate(domains[b]):
                if va != null and vb != null and va == vb:
                    table[ia, ib] = -np.inf
        factors.append(TableFactor((a, b), table))
    return factors


class GenericFamilyProblem:
    """Explicit finite coefficient bank; storage/time are exponential in general.

    The first axis enumerates complete feasible joint lineages, not entrywise
    candidates independently. The configured bound prevents silent truncation.
    Supports satisfy the interface required by controllers.solve_joint.
    """
    def __init__(self, coefficients, lineages=None, max_lineages=4096):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 4 or min(coefficients.shape) < 1:
            raise ValueError("Expected nonempty coefficient table [Q,N,d,J]")
        if coefficients.shape[0] > max_lineages:
            raise ValueError("Explicit lineage count exceeds configured max_lineages limit")
        if not np.isfinite(coefficients).all():
            raise ValueError("Coefficients must be finite")
        self._coefficients = coefficients.copy()
        self.Q, self.N, self.d, self.J = coefficients.shape
        if lineages is None:
            lineages = [(i,) for i in range(self.Q)]
        self._lineages = tuple(tuple(q) for q in lineages)
        if len(self._lineages) != self.Q or len(set(self._lineages)) != self.Q:
            raise ValueError("One distinct lineage key is required for each coefficient row")
        if len({len(q) for q in self._lineages}) != 1:
            raise ValueError("Lineage keys must have a common length")
        self._indices = {q: i for i, q in enumerate(self._lineages)}
        self.is_singleton = self.Q == 1
        self.complexity = "explicit finite enumeration; exponential fallback"

    def enumerate_lineages(self):
        return iter(self._lineages)

    def coefficients(self, lineage):
        try:
            return self._coefficients[self._indices[tuple(lineage)]].copy()
        except KeyError as error:
            raise ValueError("Lineage is outside the feasible family") from error

    def support(self, gates):
        gates = np.asarray(gates, dtype=float)
        if gates.shape != (self.J,) or not np.isfinite(gates).all():
            raise ValueError("Gates must be finite and have shape [J]")
        values = self._coefficients @ gates
        lo, hi = values.argmin(axis=0), values.argmax(axis=0)
        nodes, leads = np.indices((self.N, self.d))
        lower_coeff = self._coefficients[lo, nodes, leads]
        upper_coeff = self._coefficients[hi, nodes, leads]
        witnesses = np.asarray(self._lineages, dtype=object)
        return SupportResult(values.min(axis=0), values.max(axis=0), lower_coeff,
                             upper_coeff, witnesses[lo], witnesses[hi])


class FactorFamilyProblem:
    """General additive scalar responses with arbitrary table hard factors.

    Every response factor contains its contribution to [N,d,J] coefficients.
    For one output/sign/gate vector, max-sum eliminates the complete hard and
    response interaction graph. No shared-catalog or unary-only assumption is
    made. Memory grows with the chosen elimination order's induced width.
    """
    def __init__(self, domains, n_nodes, output_width, n_archives, response_factors,
                 hard_factors=(), order=None, max_table_entries=1_000_000):
        self.domains = _domains(domains)
        self.N, self.d, self.J = int(n_nodes), int(output_width), int(n_archives)
        if min(self.N, self.d, self.J) < 1:
            raise ValueError("Positive N,d,J required")
        self.response_factors = []
        for factor in response_factors:
            scope = _scope(factor.scope, len(self.domains))
            values = np.asarray(factor.coefficients, dtype=float)
            expected = tuple(len(self.domains[v]) for v in scope) + (self.N, self.d, self.J)
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError("Response factor must have finite domain-table+[N,d,J] coefficients")
            self.response_factors.append(ResponseFactor(scope, values.copy()))
        self.hard_factors = tuple(TableFactor(tuple(f.scope), np.array(f.values, dtype=float, copy=True))
                                  for f in hard_factors)
        # Hard factors are 0/-inf; finite soft values would alter the support.
        if any(not np.isin(f.values, [0., -np.inf]).all() for f in self.hard_factors):
            raise ValueError("Hard-factor entries must be zero or negative infinity")
        self.max_table_entries = max_table_entries
        complete_scopes = [f.scope for f in self.hard_factors] + [f.scope for f in self.response_factors]
        self.order = _min_fill(len(self.domains), complete_scopes) if order is None else tuple(order)
        max_sum(self.domains, self.hard_factors, self.order, max_table_entries)
        self.is_singleton = all(len(d) == 1 for d in self.domains)
        self.last_support_stats = {}
        self.complexity = "table-factor bucket elimination; exponential in induced width"

    def coefficients(self, lineage):
        lineage = tuple(lineage)
        if len(lineage) != len(self.domains):
            raise ValueError("Wrong number of lineage variables")
        try:
            positions = [domain.index(value) for domain, value in zip(self.domains, lineage)]
        except ValueError as error:
            raise ValueError("Lineage value is outside its domain") from error
        for factor in self.hard_factors:
            if not np.isfinite(factor.values[tuple(positions[v] for v in factor.scope)]):
                raise ValueError("Lineage violates a hard relation")
        answer = np.zeros((self.N, self.d, self.J))
        for factor in self.response_factors:
            answer += factor.coefficients[tuple(positions[v] for v in factor.scope)]
        return answer

    def support(self, gates):
        gates = np.asarray(gates, dtype=float)
        if gates.shape != (self.J,) or not np.isfinite(gates).all():
            raise ValueError("Gates must be finite and have shape [J]")
        lower, upper = np.empty((self.N, self.d)), np.empty((self.N, self.d))
        lower_coeff = np.empty((self.N, self.d, self.J))
        upper_coeff = np.empty_like(lower_coeff)
        lower_witness = np.empty((self.N, self.d, len(self.domains)), dtype=object)
        upper_witness = np.empty_like(lower_witness)
        width, largest, total = 0, 1, 0
        for node, lead in np.ndindex(self.N, self.d):
            for sign in [-1., 1.]:
                factors = list(self.hard_factors) + [
                    TableFactor(f.scope, sign * (f.coefficients[..., node, lead, :] @ gates))
                    for f in self.response_factors]
                result = max_sum(self.domains, factors, self.order, self.max_table_entries)
                coefficients = self.coefficients(result.assignment)[node, lead]
                target, coefficient_target, witness_target = (
                    (lower, lower_coeff, lower_witness) if sign < 0 else (upper, upper_coeff, upper_witness))
                target[node, lead] = sign * result.value
                coefficient_target[node, lead] = coefficients
                witness_target[node, lead] = result.assignment
                width, largest = max(width, result.induced_width), max(largest, result.max_table_entries)
                total += result.total_table_entries
        self.last_support_stats = {"induced_width": width, "max_table_entries": largest,
                                   "total_eliminated_table_entries": total,
                                   "order": list(self.order), "scalar_optimizations": 2*self.N*self.d}
        return SupportResult(lower, upper, lower_coeff, upper_coeff, lower_witness, upper_witness)

    def enumerated(self, max_assignments=4096):
        candidate_count = prod(len(d) for d in self.domains)
        if candidate_count > max_assignments:
            raise ValueError("Cartesian assignment count exceeds enumeration limit")
        coefficients, lineages = [], []
        for q in product(*self.domains):
            try:
                response = self.coefficients(q)
            except ValueError:
                continue
            coefficients.append(response)
            lineages.append(q)
        return GenericFamilyProblem(np.stack(coefficients), lineages, max_lineages=max_assignments)
