"""Executed, deterministic scientific experiments for the ORBIT interface.

These generated problems test mathematical mechanisms. They are never presented
as METR-LA/PEMS-BAY/KnowAir forecasting results or as reproduced paper timings.
Independent enumeration uses Q.T @ B @ Q, not the assignment implementation.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import platform
import queue as queue_module
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .catalog import CatalogProblem
from .controllers import solve_baseline, solve_explicit, solve_joint
from .memory import HistoryBank
from .observation_dependency import observation_dependency_experiment
from .scale_comparison import scale_comparison_experiment


SEED = 20261012


def _rss_bytes():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _summary(result):
    return {"mass": float(result.mass), "radius": float(result.radius),
            "gates": result.gates.tolist(), "center": result.center.tolist(),
            "status": result.status, "numerical_only": True}


def _direct_coefficients(problem):
    """Independent finite-family reference; never uses problem.coefficients."""
    coefficients, lineages = [], []
    for q in itertools.product(*problem.domains):
        real = [a for a in q if a >= 0]
        if len(set(real)) != len(real):
            continue
        Q = np.zeros((problem.L, problem.N))
        for i, a in enumerate(q):
            if a >= 0:
                Q[a, i] = 1
        coefficients.append(np.stack([
            ((Q.T @ B @ Q - problem.G) @ W) / problem.scale
            for B, W in zip(problem.B, problem.W)], axis=-1))
        lineages.append(q)
    if not coefficients:
        raise ValueError("Enumeration reference requires a nonempty family")
    return np.stack(coefficients), lineages


def _finite_gate_lp(coefficients, caps, epsilon):
    rows = np.concatenate([(a - b).reshape(-1, len(caps))
                           for a in coefficients for b in coefficients])
    result = linprog(-np.ones(len(caps)), A_ub=rows,
                     b_ub=np.full(len(rows), 2 * epsilon),
                     bounds=[(0, float(c)) for c in caps], method="highs")
    if not result.success:
        raise RuntimeError("Independent finite robust LP failed: " + result.message)
    return float(-result.fun)


def _random_problem(seed, n=3, j=3, d=2, domains=None, kind="signed"):
    rng = np.random.default_rng(seed)
    labels = n + 1
    B = rng.normal(size=(j, labels, labels))
    if kind == "stochastic":
        B = np.abs(B)
        B /= B.sum(axis=2, keepdims=True)
    W = rng.normal(size=(j, n, d))
    G = rng.normal(size=(n, n))
    scale = rng.uniform(.5, 1.5, size=(n, d))
    if domains is None:
        domains = [(i, (i + 1) % n, -1) for i in range(n)]
    return CatalogProblem(B, W, G, scale, domains)


def _exactness_cases():
    cases = [
        ("dense_signed", _random_problem(SEED)),
        ("dense_all_different", _random_problem(SEED + 1, domains=[(0, 1, 2)] * 3)),
        ("private_nulls", _random_problem(SEED + 2, domains=[(-1, 0), (-1, 0), (-1, 1)])),
        ("infeasible_root", _random_problem(SEED + 3, domains=[(0, 1), (0,), (2, 3)])),
        ("single_output_node", _random_problem(SEED + 4, n=1, domains=[(-1, 0, 1)])),
        ("resolved_singleton", _random_problem(SEED + 5, domains=[(0,), (1,), (-1,)])),
        ("stochastic_operators", _random_problem(SEED + 6, kind="stochastic")),
    ]
    p = _random_problem(SEED + 7)
    B = p.B.copy()
    B[0, 2:, :] = B[0, :, 2:] = 0
    B[1, 3:, :] = B[1, :, 3:] = 0
    cases.append(("heterogeneous_partial_source_coverage", CatalogProblem(B, p.W, p.G, p.scale, p.domains)))
    p = _random_problem(SEED + 8)
    cases.append(("tied_optima", CatalogProblem(np.ones_like(p.B), np.ones_like(p.W),
                                               p.G, p.scale, [(0, 1, 2)] * 3)))
    cases.append(("zero_trace", CatalogProblem(p.B, np.zeros_like(p.W), p.G, p.scale, p.domains)))
    cases.append(("nonuniform_output_scales", CatalogProblem(p.B, p.W, p.G,
                                                             np.array([[.03, 20.], [2., .1], [10., .2]]), p.domains)))
    rng = np.random.default_rng(SEED + 9)
    stream = rng.normal(size=(13, 3, 5))
    projections = rng.normal(size=(3, 5, 2))
    histories = []
    for j, gamma in enumerate([0., .75, .98]):
        bank = HistoryBank(3, projections[j:j + 1], gamma)
        for H in stream:
            bank.update(H)
        histories.append(bank.projected()[0])
    cases.append(("incompatible_discounts_separate_histories", CatalogProblem(p.B, np.stack(histories),
                                                                            p.G, p.scale, p.domains)))
    return cases


def exactness_experiment():
    """Twelve reconstructed classes, each checked against an independent LP."""
    rows = []
    for name, problem in _exactness_cases():
        start = time.perf_counter()
        coefficients, lineages = _direct_coefficients(problem)
        caps = np.arange(1, problem.J + 1, dtype=float)
        caps /= caps.sum()
        error = 0.
        for gates in [caps, np.where(np.arange(problem.J) % 2, -caps, caps)]:
            support = problem.support(gates)
            values = coefficients @ gates
            error = max(error, float(np.max(np.abs(support.lower - values.min(axis=0)))),
                        float(np.max(np.abs(support.upper - values.max(axis=0)))))
        epsilon = .1
        optimum = _finite_gate_lp(coefficients, caps, epsilon)
        joint = solve_joint(problem, caps, epsilon)
        explicit = solve_explicit(problem, caps, epsilon)
        explicit_full = solve_explicit(problem, caps, epsilon, eliminate_resolved=False)
        comparisons = [solve_baseline(problem, caps, epsilon, method)
                       for method in ["legacy", "independent", "product"]]
        hierarchy = [r.mass for r in comparisons] + [joint.mass]
        rows.append({"case": name, "lineages": len(lineages), "support_error": error,
                     "finite_lp_mass": optimum,
                     "mass_error": max(abs(r.mass - optimum) for r in [joint, explicit, explicit_full]),
                     "joint": _summary(joint), "explicit": _summary(explicit),
                     "explicit_uneliminated": _summary(explicit_full),
                     "hierarchy_masses": hierarchy,
                     "hierarchy_valid": bool(np.all(np.diff(hierarchy) >= -1e-8)),
                     "elapsed_ms": (time.perf_counter() - start) * 1000})
    return rows


def cancellation_experiment():
    """The actual four-label, two-node construction in supplement lines 783–799."""
    first, second = np.zeros((4, 4)), np.zeros((4, 4))
    for a, b in [(0, 3), (1, 2)]:
        first[a, b] = first[b, a] = 1
    for a, b in [(0, 2), (1, 3)]:
        second[a, b] = second[b, a] = 1
    B = np.stack([first, second])
    W = np.array([[[2.], [-2.]], [[2.], [-2.]]])
    domains = [(0, 1), (2, 3)]
    problem = CatalogProblem(B, W, np.eye(2), np.ones((2, 1)), domains)
    result = {"joint": _summary(solve_joint(problem, [.5, .5], 0)),
              "explicit": _summary(solve_explicit(problem, [.5, .5], 0))}
    for method in ["legacy", "independent", "product"]:
        result[method] = _summary(solve_baseline(problem, [.5, .5], 0, method))
    zero_signal = CatalogProblem(B, W, np.array([[.25, .75], [.75, .25]]),
                                 np.ones((2, 1)), domains)
    result["zero_signal"] = _summary(solve_joint(zero_signal, [.5, .5], 0))
    # Parity: knowing either variable alone leaves exactly the same radius.
    masses = []
    for restricted in [domains, [(0,), (2, 3)], [(0, 1), (2,)], [(0,), (2,)]]:
        parity = CatalogProblem(first[None], np.array([[[0.], [1.]]]), np.eye(2),
                                 np.ones((2, 1)), restricted)
        masses.append(solve_joint(parity, [1.], 0).mass)
    result["complementarity_masses"] = masses
    aligned = CatalogProblem(np.repeat(B[:1], 2, axis=0), W, np.eye(2),
                              np.ones((2, 1)), domains)
    result["aligned_extrema_masses"] = [solve_baseline(aligned, [.5, .5], .1, method).mass
                                        for method in ["independent", "product"]]
    result["aligned_extrema_masses"].append(solve_joint(aligned, [.5, .5], .1).mass)
    return result


def _pullback(B, q):
    q = np.asarray(q)
    Q = np.zeros((B.shape[0], len(q)))
    for i, identity in enumerate(q):
        if identity >= 0:
            Q[identity, i] = 1
    return Q.T @ B @ Q


def boundary_experiment():
    rng = np.random.default_rng(SEED + 100)
    n, j, d = 5, 4, 3
    G = rng.uniform(size=(n, n))
    G /= G.sum(axis=1, keepdims=True)
    q = rng.permutation(n)
    Q = np.eye(n)[:, q]
    source = Q @ G @ Q.T
    B = np.repeat(source[None], j, axis=0)
    W = rng.normal(size=(j, n, d))
    base = rng.normal(size=(n, d))
    anchor = base + np.mean(np.einsum("il,jlk->jik", G, W), axis=0)
    resolved = CatalogProblem(B, W, G, np.ones((n, d)), [(int(a),) for a in q])
    reindex = solve_joint(resolved, np.full(j, 1/j), 0)
    absent = CatalogProblem(B, W, G, np.ones((n, d)), [(-1,)] * n)
    null = solve_joint(absent, np.full(j, 1/j), 0)
    return {"pure_reindex_max_difference": float(np.max(np.abs(reindex.center))),
            "all_null_frozen_max_difference": float(np.max(np.abs(anchor + null.center - base))),
            "all_null_mass": null.mass,
            "stale_archive_scope": "Zero lineage radius constrains correspondence, not predictive target error."}


def _replay(features, projections, operators, gamma):
    state = np.zeros((len(projections), operators.shape[1], projections.shape[2]))
    for H in features:
        increment = np.stack([A @ H @ V for A, V in zip(operators, projections)])
        state = gamma * state + (1 - gamma) * increment
    return state


def _timed(call, repetitions):
    durations = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = call()
        durations.append((time.perf_counter() - start) * 1000)
    return result, durations


def delayed_repair_experiment(quick=False):
    """Repair after generated delayed evidence; no fabricated forecasting MAE."""
    rows = []
    repetitions = 1 if quick else 5
    for index, (cadence, delays) in enumerate([
            ("METR-LA cadence", [0, 12, 72]), ("PEMS-BAY cadence", [0, 12, 72]),
            ("KnowAir cadence", [0, 4, 12])]):
        rng = np.random.default_rng(SEED + 200 + index)
        n, p, d, j, gamma = 12, 6, 3, 4, .95
        projections = rng.normal(size=(j, p, d))
        B = rng.uniform(size=(j, n, n))
        B /= B.sum(axis=2, keepdims=True)
        truth, wrong = rng.permutation(n), rng.permutation(n)
        correct_ops = np.stack([_pullback(M, truth) for M in B])
        wrong_ops = np.stack([_pullback(M, wrong) for M in B])
        for delay in delays:
            # Request is dispatched after round 8; zero delay is post-issuance.
            arrival = 8 + delay
            features = rng.normal(size=(arrival + 1, n, p))
            bank = HistoryBank(n, projections, gamma)
            for H in features:
                bank.update(H)
            reference = _replay(features, projections, correct_ops, gamma)
            stale = _replay(features, projections, wrong_ops, gamma)
            before_issuance = np.mean(stale, axis=0).copy()
            from .protocol import ForecastLedger
            ledger = ForecastLedger()
            ledger.seal(arrival, before_issuance, {"synthetic": True})
            # These timings are response repair only; inference and admission are excluded.
            operations = {
                "ORBIT repair": lambda: np.stack([A @ W for A, W in zip(correct_ops, bank.projected())]),
                "Full-prefix replay": lambda: _replay(features, projections, correct_ops, gamma),
                "Forward-only correction": lambda: stale.copy(),
                "Reset-on-answer": lambda: np.zeros_like(stale),
            }
            for method, operation in operations.items():
                repaired, durations = _timed(operation, repetitions)
                initial_error = float(np.max(np.abs(repaired - reference)))
                # Apply identical subsequent increments; only prior-state error differs.
                difference = repaired - reference
                recovery = 0
                while np.max(np.abs(difference)) > 1e-10 and recovery < 1000:
                    difference *= gamma
                    recovery += 1
                rows.append({"cadence": cadence, "delay_rounds": delay, "method": method,
                             "origin": "synthetic mechanism experiment", "n_nodes": n,
                             "feature_width": p, "output_width": d, "archives": j, "gamma": gamma,
                             "request_round": 8, "return_round": arrival,
                             "specified_return_phase": "post_issuance" if delay == 0 else "pre_issuance",
                             "specified_first_affected_forecast": arrival + int(delay == 0),
                             "causal_clock_simulated": False,
                             "max_replay_difference": initial_error,
                             "correction_mae_vs_replay": float(np.mean(np.abs(repaired - reference))),
                             "numerical_recovery_rounds": recovery,
                             "forecast_rewrites": int(not np.array_equal(ledger.predictions[arrival], before_issuance)),
                             "repair_p50_ms": float(np.quantile(durations, .5)),
                             "repair_p95_ms": float(np.quantile(durations, .95)),
                             "latency_samples_ms": durations, "deadline_ms": 1000.,
                             "deadline_miss_fraction": float(np.mean(np.array(durations) > 1000)),
                             "timing_scope": "response readout only; metadata validation and domain restriction excluded",
                             "target_forecasting_mae": None})
    return rows


def layout_experiment():
    rows = []
    rng = np.random.default_rng(SEED + 300)
    for name, n, d in [("METR-LA dimensions", 207, 12), ("PEMS-BAY dimensions", 325, 12),
                       ("KnowAir dimensions", 184, 8)]:
        projections = rng.normal(size=(4, 64, d))
        H = rng.normal(size=(n, 64))
        reference = None
        for layout in ["feature", "projected", "auto"]:
            bank = HistoryBank(n, projections, .9, layout=layout)
            start = time.perf_counter()
            for _ in range(3):
                bank.update(H)
            W = bank.projected()
            elapsed = (time.perf_counter() - start) * 1000
            if reference is None:
                reference = W
            rows.append({"case": name, "requested_layout": layout,
                         "selected_layout": bank.layout, "payload_bytes": bank.payload_bytes,
                         "minimum_formula_bytes": 8 * n * min(64, 4*d),
                         "max_layout_difference": float(np.max(np.abs(reference - W))),
                         "three_updates_and_projection_ms": elapsed,
                         "process_rss_after_bytes": _rss_bytes(),
                         "origin": "synthetic features at paper node dimensions"})
    return rows


def generic_relation_experiment():
    """Supplement MaxCut reduction with genuine cross-source hard equalities."""
    from .general import (TableFactor, ResponseFactor, FactorFamilyProblem,
                          all_different_factors)
    rows = []
    for name, vertices, edges in [("triangle", 3, [(0, 1), (1, 2), (2, 0)]),
                                   ("four_vertex_path", 4, [(0, 1), (1, 2), (2, 3)])]:
        start = time.perf_counter()
        j = len(edges)
        domains = [(0, 1), (2, 3)] * j
        hard, responses = [], []
        copies = {v: [] for v in range(vertices)}
        M = np.zeros((4, 4))
        M[0, 3] = M[3, 0] = M[1, 2] = M[2, 1] = 1
        for source, (u, v) in enumerate(edges):
            scope = (2*source, 2*source + 1)
            hard += all_different_factors(domains, scope)
            copies[u].append(scope[0])
            copies[v].append(scope[1])
            table = np.zeros((2, 2, 2, 1, j))
            for x, y in itertools.product(range(2), repeat=2):
                # Direct source pullback, not the already simplified cut formula.
                table[x, y, :, :, source] = (_pullback(M, (x, 2+y)) - np.eye(2)) @ np.array([[0.], [1.]])
            responses.append(ResponseFactor(scope, table))
        for same_vertex in copies.values():
            for a, b in zip(same_vertex[:-1], same_vertex[1:]):
                relation = np.array([[0. if va % 2 == vb % 2 else -np.inf
                                      for vb in domains[b]] for va in domains[a]])
                hard.append(TableFactor((a, b), relation))
        problem = FactorFamilyProblem(domains, 2, 1, j, responses, hard,
                                       max_table_entries=4096)
        finite = problem.enumerated(max_assignments=4096)
        gates = np.full(j, 1/j)
        support = problem.support(gates)
        stats = dict(problem.last_support_stats)
        reference = finite.support(gates)
        maximum_cut = max(sum(bits[u] != bits[v] for u, v in edges)
                          for bits in itertools.product([0, 1], repeat=vertices))
        joint = solve_joint(problem, gates, .1)
        coefficients = np.stack([finite.coefficients(q) for q in finite.enumerate_lineages()])
        optimum = _finite_gate_lp(coefficients, gates, .1)
        rows.append({"case": name, "dispatch": "generic_table_factor_elimination",
                     "source": "supplement MaxCut construction, main.tex lines 790-799",
                     "vertices": vertices, "archives_edges": edges,
                     "lineage_copy_variables": 2*j, "feasible_lineages": finite.Q,
                     "hard_relation_factors": len(hard), "maxcut": maximum_cut,
                     "actual_radius": support.radius, "maxcut_radius": maximum_cut/(2*j),
                     "support_error": float(max(np.max(np.abs(support.lower-reference.lower)),
                                                np.max(np.abs(support.upper-reference.upper)))),
                     "mass_error": abs(joint.mass-optimum), "controller": _summary(joint),
                     "elapsed_ms": 1000*(time.perf_counter()-start), **stats,
                     "complexity": "exponential in reported induced width; not catalog-polynomial"})
    return rows


def bank_representation_experiment(quick=False):
    """Measure five exact state strategies on an explicit generated family.

    Six cyclic permutations are the complete declared family for this experiment,
    not an implicit claim that six is the full all-different family. Timing is
    batched response retrieval for all six candidates; common source operators,
    projections, graph construction, and backbone work are excluded uniformly.
    """
    rng = np.random.default_rng(SEED + 350)
    n, p, d, j, gamma, candidates, interval = 6, 8, 3, 4, .95, 6, 8
    rounds, repetitions = (24, 1) if quick else (96, 5)
    features = rng.normal(size=(rounds, n, p))
    projections = rng.normal(size=(j, p, d))
    B = rng.uniform(size=(j, n, n))
    B /= B.sum(axis=2, keepdims=True)
    lineages = [tuple(np.roll(np.arange(n), shift)) for shift in range(candidates)]
    operators = np.stack([np.stack([_pullback(M, q) for M in B]) for q in lineages])
    z_reference = sum((1-gamma)*gamma**(rounds-1-t)*H for t, H in enumerate(features))
    reference = np.stack([np.stack([A @ z_reference @ V for A, V in zip(ops, projections)])
                           for ops in operators])
    checkpoint_rounds = ((rounds-1)//interval)*interval

    def from_projected(W):
        return np.einsum("cjil,jld->cjid", operators, W)

    rows = []
    for method in ["feature_history", "projected_history", "materialized_candidate_bank",
                   "full_prefix_replay", "checkpoint_suffix_replay"]:
        start = time.perf_counter()
        if method in ["feature_history", "projected_history"]:
            bank = HistoryBank(n, projections, gamma,
                               layout="feature" if method == "feature_history" else "projected")
            for H in features:
                bank.update(H)
            payload = bank.payload_bytes
            readout = lambda: from_projected(bank.projected())
        elif method == "materialized_candidate_bank":
            materialized = np.zeros((candidates, j, n, d))
            for H in features:
                increment = np.einsum("np,jpd->jnd", H, projections)
                materialized *= gamma
                materialized += (1-gamma)*from_projected(increment)
            payload = materialized.nbytes
            readout = lambda: materialized.copy()
        elif method == "full_prefix_replay":
            prefix = features.copy()
            payload = prefix.nbytes
            readout = lambda: np.stack([_replay(prefix, projections, ops, gamma) for ops in operators])
        else:
            checkpoint = np.zeros((n, p))
            for H in features[:checkpoint_rounds]:
                checkpoint *= gamma
                checkpoint += (1-gamma)*H
            suffix = features[checkpoint_rounds:].copy()
            payload = checkpoint.nbytes + suffix.nbytes

            def readout():
                z = checkpoint.copy()
                for H in suffix:
                    z *= gamma
                    z += (1-gamma)*H
                return from_projected(np.einsum("np,jpd->jnd", z, projections))
        build_ms = 1000*(time.perf_counter()-start)
        response, durations = _timed(readout, repetitions)
        rows.append({"method": method, "origin": "synthetic reconstruction",
                     "N": n, "p": p, "d": d, "J": j, "gamma": gamma,
                     "candidate_count": candidates, "candidate_family": "six cyclic permutations",
                     "prefix_rounds": rounds, "checkpoint_interval": interval,
                     "checkpoint_rounds_retained": checkpoint_rounds,
                     "suffix_rounds": rounds-checkpoint_rounds, "payload_bytes": int(payload),
                     "payload_scope": "persistent history only; common operators and projections excluded",
                     "stream_state_construction_ms": build_ms,
                     "p50_ms": float(np.quantile(durations, .5)),
                     "p95_ms": float(np.quantile(durations, .95)), "latency_samples_ms": durations,
                     "deadline_ms": 1000.,
                     "deadline_miss_fraction": float(np.mean(np.asarray(durations) > 1000)),
                     "timing_scope": "all-candidate response retrieval, not end-to-end forecasting",
                     "max_response_difference": float(np.max(np.abs(response-reference))),
                     "process_rss_after_bytes": _rss_bytes(),
                     "target_forecast_mae": None})
    return rows


def fault_experiment():
    """Actual rejection checks; semantic truth is kept outside structural checks."""
    from .protocol import MetadataSimulator, generate_manifest
    rows = {name: {"attempts": 0, "rejected": 0, "accepted": 0, "truth_excluded": 0}
            for name in ["expired_versions", "empty_families", "injectivity_conflicts",
                         "missing_source_epochs", "unsupported_constraints", "numerical_failures",
                         "missing_truth", "false_feasible_answers"]}
    for seed in range(100):
        manifest = generate_manifest(2, 3, unresolved=2, domain_size=3, seed=SEED + 400 + seed)
        for name in rows:
            rows[name]["attempts"] += 1
            accepted = False
            truth_excluded = False
            try:
                if name in ("expired_versions", "missing_source_epochs"):
                    sim = MetadataSimulator(manifest)
                    answer = {"epoch":manifest['epoch'],"node": 0, "identity": manifest["truth"][0]}
                    if name == "expired_versions":
                        answer["epoch"] = manifest["epoch"]
                        answer["source_versions"] = ["expired-source"]
                    accepted = sim.accept(answer)
                elif name == "unsupported_constraints":
                    sim = MetadataSimulator(manifest)
                    answer = {"epoch": manifest["epoch"], "source_versions":manifest['source_versions'],"node": 0,
                              "identity": manifest["truth"][0], "constraint_type": "pairwise_parity"}
                    accepted = sim.accept(answer)
                elif name == "false_feasible_answers":
                    sim = MetadataSimulator(manifest)
                    wrong = next(a for a in manifest["domains"][0] if a != manifest["truth"][0])
                    accepted = sim.accept({"epoch": manifest["epoch"], "source_versions":manifest['source_versions'],"node": 0, "identity": wrong})
                    truth_excluded = accepted and manifest["truth"][0] not in sim.domains[0]
                else:
                    domains = [(0, 1), (1, 2)]
                    B = np.ones((1, 3, 3))
                    if name == "empty_families":
                        domains = [(), (1, 2)]
                    elif name == "injectivity_conflicts":
                        domains = [(0,), (0,)]
                    elif name == "numerical_failures":
                        B[0, 0, 0] = np.nan
                    elif name == "missing_truth":
                        domains = [(2,), (1,)]  # Structural family valid; assumed truth (0,1) excluded.
                        truth_excluded = True
                    CatalogProblem(B, np.ones((1, 2, 1)), np.eye(2), np.ones((2, 1)), domains)
                    accepted = True
            except (ValueError, KeyError, TypeError):
                accepted = False
            rows[name]["accepted" if accepted else "rejected"] += 1
            rows[name]["truth_excluded"] += int(truth_excluded)
    return rows


def sweep_experiment(quick=False):
    rows = {"allowance": [], "discount": [], "dimension": [], "query_budget": []}
    problem = _random_problem(SEED + 500)
    caps = np.full(problem.J, 1 / problem.J)
    for epsilon in [0., .01, .1, .5, 2.]:
        start = time.perf_counter()
        result = solve_joint(problem, caps, epsilon)
        rows["allowance"].append({"epsilon": epsilon, **_summary(result),
                                   "elapsed_ms": (time.perf_counter() - start) * 1000})
    rng = np.random.default_rng(SEED + 501)
    stream = rng.normal(size=(20, 3, 5))
    projections = rng.normal(size=(3, 5, 2))
    for gamma in [0., .75, .9, .95, .98]:
        bank = HistoryBank(3, projections, gamma)
        for H in stream:
            bank.update(H)
        expected = sum((1-gamma)*gamma**(len(stream)-1-t)*H for t, H in enumerate(stream))
        rows["discount"].append({"gamma": gamma,
            "max_direct_sum_error": float(np.max(np.abs(bank.projected() - np.einsum("np,jpd->jnd", expected, projections)))),
            "payload_bytes": bank.payload_bytes})
    for p in ([1, 8, 64] if quick else [1, 8, 16, 32, 64, 128]):
        bank = HistoryBank(5, rng.normal(size=(4, p, 3)), .9)
        H = rng.normal(size=(5, p))
        _, durations = _timed(lambda: bank.update(H), 1 if quick else 5)
        rows["dimension"].append({"p": p, "n_nodes": 5, "d": 3, "J": 4,
                                   "selected_layout": bank.layout, "payload_bytes": bank.payload_bytes,
                                   "update_p95_ms": float(np.quantile(durations, .95))})
    from .protocol import MetadataSimulator, generate_manifest
    manifest = generate_manifest(12, 16, unresolved=8, domain_size=3, seed=SEED + 502)
    for refill in [0., 1/6, .5, 1.]:
        sim = MetadataSimulator(manifest, delay=4, refill=refill, capacity=4)
        for t in range(48):
            sim.before_round(t)
            sim.after_issuance(t)
        rows["query_budget"].append({"rounds": 48, "refill": refill, "capacity": 4,
                                     "dispatches": sim.dispatch_count,
                                     "resolved_nodes": sum(len(d) == 1 for d in sim.domains)})
    return rows


def _scale_case(n, unresolved, d=2):
    rng = np.random.default_rng(SEED + n * 10 + unresolved)
    labels, j = n + (35 if n >= 325 else 4), 4
    B = rng.uniform(size=(j, labels, labels))
    B /= B.sum(axis=2, keepdims=True)
    pool = list(range(unresolved)) + list(range(n, n + 4))
    domains = [(i,) for i in range(n)]
    # Guarantee all four otherwise-unused labels occur in the m=8 condition.
    for i in range(unresolved):
        candidates = [x for x in pool if x != i]
        distractors = list(rng.choice(candidates, size=2, replace=False))
        if i < 4:
            distractors[0] = n + i
            if distractors[0] == distractors[1]:
                distractors[1] = next(x for x in candidates if x != distractors[0])
        domains[i] = tuple([i] + [int(x) for x in distractors])
    start = time.perf_counter()
    problem = CatalogProblem(B, rng.normal(size=(j, n, d)), np.eye(n), np.ones((n, d)), domains)
    assembly_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    result = solve_joint(problem, np.full(j, .25), .01, max_iterations=10)
    return {"N": n, "m": unresolved, "L": labels, "J": j, "d": d, "K": 3,
            "assembly_ms": assembly_ms, "controller_ms": (time.perf_counter() - start) * 1000,
            "process_rss_after_bytes": _rss_bytes(), **_summary(result)}


def _scale_worker(queue, n, unresolved):
    try:
        queue.put({"result": _scale_case(n, unresolved, d=12)})
    except Exception as error:
        queue.put({"error": f"{type(error).__name__}: {error}"})


def scale_experiment(full_scale=False, deadline_seconds=30.):
    if not full_scale:
        rows = []
        for n, unresolved in [(8, 2), (8, 4), (16, 2), (16, 4)]:
            start = time.perf_counter()
            row = _scale_case(n, unresolved)
            wall = time.perf_counter() - start
            row.update({"wall_ms": wall*1000, "deadline_seconds": deadline_seconds,
                        "deadline_exceeded": wall > deadline_seconds,
                        "timeout_policy": "post-completion deadline accounting; small synthetic run"})
            rows.append(row)
        return rows
    # Isolated workers permit genuine wall-clock termination at large ambiguity.
    context = mp.get_context("spawn")
    rows = []
    for n in [325, 1000]:
        for unresolved in [8, 32, 128]:
            queue = context.Queue()
            process = context.Process(target=_scale_worker, args=(queue, n, unresolved))
            start = time.perf_counter()
            process.start()
            payload = None
            while time.perf_counter() - start < deadline_seconds:
                # Drain results before joining: a large center can fill the queue
                # pipe and otherwise prevent an already finished worker exiting.
                try:
                    payload = queue.get(timeout=min(.05, deadline_seconds))
                    break
                except queue_module.Empty:
                    if not process.is_alive():
                        break
            timed_out = payload is None and process.is_alive()
            if timed_out:
                process.terminate()
            process.join(timeout=2)
            wall = time.perf_counter() - start
            row = {"N": n, "m": unresolved, "deadline_seconds": deadline_seconds,
                   "timeout": timed_out, "wall_ms": wall*1000,
                   "timeout_policy": "hard child-process deadline including launch and assembly",
                   "latency_quantile_policy": "one measured run; no p95 claimed; timeouts retained"}
            if payload is not None:
                row.update(payload)
            elif not timed_out:
                row["error"] = f"worker exited without result (exitcode={process.exitcode})"
            queue.close()
            rows.append(row)
    return rows


def run_scientific(output: Path, quick=False, scale=False):
    """Write actual generated-case results. Large ambiguity is explicitly opt-in."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    report = {
        "provenance": {"source": "executed synthetic experiments", "seed": SEED,
                       "quick": bool(quick), "formal_interval_certification": False,
                       "python": platform.python_version(), "platform": platform.platform(),
                       "numpy": np.__version__, "paper_measurements_used_as_output": False,
                       "scope": "reconstructed mechanism experiments; not real-data forecast performance"},
        "exactness": exactness_experiment(),
        "cancellation": cancellation_experiment(),
        "boundaries": boundary_experiment(),
        "delayed_repair": delayed_repair_experiment(quick=quick),
        "layouts": layout_experiment(),
        "bank_representations": bank_representation_experiment(quick=quick),
        "general_relations": generic_relation_experiment(),
        "faults": fault_experiment(),
        "sweeps": sweep_experiment(quick=quick),
        "observation_dependency": observation_dependency_experiment(quick=quick),
        "scale": scale_experiment(full_scale=scale),
        "scale_comparison": scale_comparison_experiment(quick=not scale),
        "status": {"real_data_forecasting": "separate workflow required",
                   "external_online_baselines": "separate orbit external workflow; actual coverage in reproduction-status.json",
                   "cross_backbone_five_seed": "separate workflow required",
                   "author_epoch_manifests": "unavailable; reconstruction documented",
                   "author_twelve_exactness_classes": "not fully specified; twelve reconstructed classes executed",
                   "generic_relational_oracle": "table-factor elimination and bounded finite fallback implemented; tiny cross-source MaxCut executed",
                   "observation_derived_dependency_strength": "reconstructed past-observation Pearson graph sweep executed; author fixture unavailable",
                   "weak_stale_archive_forecasting": "separate real-data controls required",
                   "metadata_target_error_interventions": "separate orbit intervention reconstructed fixed-action/adaptive paired workflow",
                   "materialized_bank_checkpoint_benchmarks": "five synthetic representations executed with explicit candidate count and checkpoint interval; real-data author benchmark not reproduced",
                   "formal_interval_certification": "not implemented"},
    }
    report["elapsed_seconds"] = time.perf_counter() - start
    report = _jsonable(report)
    (output / "scientific.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
