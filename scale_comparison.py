"""Paired, process-isolated controller scale comparisons with explicit censoring.

This is a reconstructed synthetic experiment, not the authors' hardware benchmark.
No large trial runs at import time. Each actual solve lives in its own process.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import multiprocessing as mp
import platform
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from .catalog import CatalogProblem
from .controllers import solve_explicit, solve_joint, solver_thread_capability


METHODS = ("matching_cuts", "explicit_reduced", "explicit_unreduced")
DEFAULT_SEED = 20261012


def _positive_int(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def estimate_trial_size(n, unresolved, method, d=12, max_iterations=100):
    """Structural dimension upper bounds; deliberately generous memory heuristic.

    All roots are counted, even if infeasible. This fixture has m ambiguous
    rows, three labels per row, and at most m+4 available ambiguous real labels.
    Zero coefficients and occupied labels can only reduce the explicit sizes.
    The memory estimate includes Python-list construction and sparse copies;
    it is an early guard, not a guarantee about allocator/solver peak memory.
    """
    n, unresolved, d = (_positive_int(n, "N"), _positive_int(unresolved, "m"), _positive_int(d, "d"))
    if unresolved > n or method not in METHODS:
        raise ValueError("m must not exceed N and method must be supported")
    j, labels = 4, n + (35 if n >= 325 else 4)
    roots = n + 2 * unresolved
    if method == "explicit_unreduced":
        left, right, edges = n - 1, n + 3, n + 2 * unresolved - 1
    else:
        left, right, edges = unresolved, unresolved + 4, 3 * unresolved
    if method == "matching_cuts":
        variables = j
        constraints = n * d * _positive_int(max_iterations, "max_iterations")
        nonzeros = constraints * j
    else:
        variables = j + 2 * n * d + 2 * d * roots * (left + right)
        constraints = n * d + 2 * d * roots * (edges + 1)
        nonzeros = 2 * n * d + 2 * d * roots * (edges * (j + 2) + j + left + right + 1)
    dense = 8 * (2 * j * labels * labels + 2 * n * n + 2 * j * n * d + 2 * n * d)
    branches = roots * (8 * n + 8 * left + 24 * edges + 40 * right + 512)
    # 256 MiB runtime/import margin, 64 MiB for capped caches and overhead.
    # Sparse assembly holds three Python lists, numerical copies and HiGHS data.
    peak = dense + branches + 320 * 1024**2 + 160 * nonzeros + 100 * variables + 64 * constraints
    return {"lp_variables_upper_bound": int(variables), "lp_constraints_upper_bound": int(constraints),
            "lp_nonzeros_upper_bound": int(nonzeros), "estimated_peak_bytes": int(peak),
            "root_count_upper_bound": roots,
            "estimate_kind": "conservative structural upper bounds plus heuristic assembly/solver memory margin"}


def _fixture(n, unresolved, d, seed):
    # Same construction family as scientific._scale_case, including all four
    # unused candidate identities for m>=4, but an explicit repeat seed.
    rng = np.random.default_rng(seed)
    labels, j = n + (35 if n >= 325 else 4), 4
    B = rng.uniform(size=(j, labels, labels))
    B /= B.sum(axis=2, keepdims=True)
    pool = list(range(unresolved)) + list(range(n, n + 4))
    domains = [(i,) for i in range(n)]
    for i in range(unresolved):
        candidates = [value for value in pool if value != i]
        distractors = list(rng.choice(candidates, size=2, replace=False))
        if i < 4:
            distractors[0] = n + i
            if distractors[0] == distractors[1]:
                distractors[1] = next(value for value in candidates if value != distractors[0])
        domains[i] = tuple([i] + [int(value) for value in distractors])
    W, G, scale = rng.normal(size=(j, n, d)), np.eye(n), np.ones((n, d))
    digest = hashlib.sha256()
    for array in (B, W, G, scale):
        digest.update(array.tobytes())
    digest.update(json.dumps(domains, separators=(",", ":")).encode("ascii"))
    return (B, W, G, scale, domains), digest.hexdigest()


def _worker(connection, options):
    """Send short progress/results only; never transmit large center arrays."""
    try:
        with threadpool_limits(limits=8, user_api="blas"):
            started = time.perf_counter()
            arrays, digest = _fixture(options["N"], options["m"], options["d"], options["seed"])
            fixture_ms = 1000 * (time.perf_counter() - started)
            connection.send({"phase": "fixture_ready", "fixture_ms": fixture_ms, "fixture_sha256": digest})
            started = time.perf_counter()
            problem = CatalogProblem(*arrays)
            del arrays
            catalog_ms = 1000 * (time.perf_counter() - started)
            connection.send({"phase": "catalog_ready", "catalog_ms": catalog_ms,
                             "feasible_root_count": sum(map(len, problem.feasible_roots))})
            started = time.perf_counter()
            caps = np.full(problem.J, .25)
            if options["method"] == "matching_cuts":
                result = solve_joint(problem, caps, options["epsilon"], max_iterations=options["max_iterations"])
            else:
                result = solve_explicit(problem, caps, options["epsilon"],
                                        eliminate_resolved=options["method"] == "explicit_reduced")
            controller_ms = 1000 * (time.perf_counter() - started)
            if not all(math.isfinite(value) for value in (result.mass, result.radius, result.upper_mass)):
                raise FloatingPointError("controller returned nonfinite diagnostics")
            explicit = options["method"] != "matching_cuts"
            connection.send({"phase": "finished", "status": "completed", "controller_ms": controller_ms,
                             "mass": result.mass, "radius": result.radius, "upper_mass": result.upper_mass,
                             "mass_gap": max(0., result.upper_mass - result.mass),
                             "solver_status": result.status,
                             "optimal_numerical": result.status == "optimal_numerical",
                             "iterations": result.iterations, "retained_cuts": result.details.get("cuts"),
                             "lp_dimensions": ({"variables": result.details["lp_variables"],
                                                "constraints": result.details["lp_constraints"],
                                                "nonzeros": result.details.get("lp_nonzeros")} if explicit else None),
                             "child_rss_at_result_bytes": psutil.Process().memory_info().rss,
                             "lp_backend": solver_thread_capability()})
    except BaseException as error:
        try:
            connection.send({"phase": "failed", "status": "worker_error",
                             "error": f"{type(error).__name__}: {error}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def run_trial(n, unresolved, method, *, d=12, seed=DEFAULT_SEED, repeat=0,
              deadline_seconds=30., memory_budget_bytes=4 * 1024**3,
              poll_seconds=.01, preflight_guard=True, epsilon=.01, max_iterations=100):
    """Run one fresh child, or report an unexecuted memory-size guard.

    Deadline includes spawn/import, fixture generation, catalog assembly and
    solve/validation, measured by the parent until receipt of a complete result.
    Censored trials have no completed latency. Monitoring is sampled, so the
    RSS cap is a termination threshold and does not bound sub-poll transients.
    """
    estimate = estimate_trial_size(n, unresolved, method, d, max_iterations)
    memory_budget_bytes = _positive_int(memory_budget_bytes, "memory_budget_bytes")
    for value, name in ((deadline_seconds, "deadline_seconds"), (poll_seconds, "poll_seconds")):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    row = {"N": int(n), "m": int(unresolved), "L": n + (35 if n >= 325 else 4), "J": 4,
           "d": int(d), "K": 3, "method": method, "seed": int(seed), "repeat": int(repeat),
           "epsilon": float(epsilon), "max_iterations": int(max_iterations),
           "deadline_seconds": float(deadline_seconds), "memory_budget_bytes": memory_budget_bytes,
           "estimate": estimate, "executed": False, "child_pid": None, "child_exitcode": None,
           "status": None, "phase": "preflight", "peak_rss_sampled_bytes": None,
           "completed_total_ms": None, "termination_elapsed_ms": None,
           "fixture_ms": None, "catalog_ms": None, "controller_ms": None,
           "lp_dimensions": None, "fixture_sha256": None, "optimal_numerical": False}
    if preflight_guard and estimate["estimated_peak_bytes"] > memory_budget_bytes:
        row.update(status="guarded_not_executed", guard_reason="estimated assembly/solver memory exceeds configured budget")
        return row
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, row.copy()))
    started = time.perf_counter()
    peak = 0
    try:
        process.start()
        sender.close()
        row.update(executed=True, child_pid=process.pid, phase="spawned")
        monitor = psutil.Process(process.pid)
        while True:
            elapsed = time.perf_counter() - started
            try:
                peak = max(peak, monitor.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if peak > memory_budget_bytes:
                row["status"] = "memory_abort"
                break
            if elapsed >= deadline_seconds:
                row["status"] = "timeout"
                break
            if receiver.poll(min(poll_seconds, deadline_seconds - elapsed)):
                try:
                    message = receiver.recv()
                except EOFError:
                    row.update(status="worker_error", error="child closed the pipe without a final result")
                    break
                if message:
                    row.update(message)
                    if message.get("phase") == "finished":
                        measured = time.perf_counter() - started
                        peak = max(peak, row.get("child_rss_at_result_bytes", 0))
                        if peak > memory_budget_bytes:
                            row["status"] = "memory_abort"
                        elif measured > deadline_seconds:
                            row["status"] = "timeout"
                        else:
                            row["completed_total_ms"] = measured * 1000
                        break
                    if message.get("phase") == "failed":
                        break
            if not process.is_alive():
                # A normal child's result can remain readable after process exit.
                if receiver.poll():
                    continue
                row.update(status="worker_error", error="child exited without a final result")
                break
    except (OSError, RuntimeError) as error:
        row.update(status="worker_error", error=f"{type(error).__name__}: {error}")
    finally:
        row["termination_elapsed_ms"] = (time.perf_counter() - started) * 1000
        row["peak_rss_sampled_bytes"] = peak or None
        if process.pid is not None:
            if row["status"] == "worker_error":
                # Preserve a natural exit code when failure/EOF arrives just
                # before child teardown. Resource-limit terminations get no grace.
                process.join(timeout=.1)
            if row["status"] != "completed" and process.is_alive():
                process.terminate()
            process.join(timeout=.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            row["child_exitcode"] = process.exitcode
        receiver.close()
        sender.close()
    return row


def summarize_trials(rows):
    """Completed-sample quantiles with separate counts for every censor reason."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["N"], row["m"], row["d"], row["method"])].append(row)
    summaries = []
    for (n, m, d, method), group in sorted(groups.items()):
        complete = [row for row in group if row["status"] == "completed"]
        def quantiles(key):
            values = [row[key] for row in complete if row.get(key) is not None]
            return {"p50": float(np.percentile(values, 50)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None}
        summaries.append({"N": n, "m": m, "d": d, "method": method, "trials": len(group),
                          "completed_trials": len(complete),
                          "optimal_numerical_trials": sum(row.get("optimal_numerical", False) for row in complete),
                          "status_counts": dict(Counter(row["status"] for row in group)),
                          "completed_total_ms": quantiles("completed_total_ms"),
                          "controller_ms": quantiles("controller_ms"),
                          "max_sampled_rss_bytes": max((r.get("peak_rss_sampled_bytes") or 0 for r in group), default=0) or None,
                          "quantile_policy": "completed trials only; no timeout, memory-abort or guard imputation"})
    return summaries


def scale_comparison_experiment(quick=False, *, repeats=None, deadline_seconds=30.,
                                memory_budget_bytes=4 * 1024**3, seed=DEFAULT_SEED):
    """Compare three implementations on identical seeded fixture repeats.

    Default: N=325/1000, m=8/32/128, J=4, d=12, K=3, three repeats.
    Quick: N=6,m=2,d=2 and one repeat. Trials are serial child processes.
    """
    repeats = _positive_int((1 if quick else 3) if repeats is None else repeats, "repeats")
    settings, d = ([(6, 2)], 2) if quick else ([(n, m) for n in (325, 1000) for m in (8, 32, 128)], 12)
    rows = []
    for n, unresolved in settings:
        for repeat in range(repeats):
            fixture_seed = int(seed) + n * 1000 + unresolved * 10 + repeat
            # Rotate order to reduce systematic method-order effects. Processes
            # still share a host; hardware exclusivity is an operator condition.
            methods = METHODS[repeat % 3:] + METHODS[:repeat % 3]
            for method in methods:
                rows.append(run_trial(n, unresolved, method, d=d, seed=fixture_seed, repeat=repeat,
                                      deadline_seconds=deadline_seconds, memory_budget_bytes=memory_budget_bytes))
    return {"provenance": {"source": "executed paired synthetic scale trials", "quick": bool(quick),
                           "fixture": "scientific._scale_case construction family with explicit repeat seeds",
                           "base_seed": int(seed), "repeats": repeats, "python": platform.python_version(),
                           "platform": platform.platform(), "numpy": np.__version__,
                           "paper_comparison_valid": False, "formal_interval_certification": False,
                           "blas_threads": 8, "lp_backend": solver_thread_capability(),
                           "host_exclusivity": "not verified; run full experiments only on an idle host",
                           "rss_scope": "sampled child RSS, including imports/fixture/assembly/solve; may miss transients",
                           "deadline_scope": "hard parent wall deadline includes child launch/import and assembly",
                           "memory_guard": "preflight heuristic; guarded instances are not executed or timed",
                           "quantile_scope": "completed samples only; small repeat count has limited tail resolution"},
            "trials": rows, "summaries": summarize_trials(rows)}
