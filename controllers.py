"""Joint admission and independent comparison controllers.

All outputs are FP64 numerical results, not formally certified interval bounds.
The explicit formulation builds assignment-dual epigraphs, whereas the implicit
controller separates finite lineage-pair cuts through complete support calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import numpy as np
from scipy.optimize import linprog, OptimizeResult
from scipy.sparse import coo_matrix, csc_matrix

try:
    import highspy
except ImportError:
    highspy = None

from .catalog import CatalogProblem, SupportResult


@dataclass(frozen=True)
class ControllerResult:
    gates: np.ndarray
    center: np.ndarray
    radius: float
    mass: float
    status: str
    iterations: int
    upper_mass: float
    numerical_only: bool = True
    details: dict = field(default_factory=dict)


def _validate(problem, caps, epsilon):
    caps = np.asarray(caps, dtype=np.float64)
    if caps.shape != (problem.J,) or not np.all(np.isfinite(caps)) or np.any(caps < 0):
        raise ValueError("caps must be a finite nonnegative vector of length J")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    # The paper assumes sum(caps)=1. Supporting arbitrary nonnegative boxes is
    # useful for ablations; legacy normalization below preserves feasibility.
    return caps, epsilon


def solver_thread_capability():
    """Describe the configured LP backend, independently of BLAS threads.

    SciPy 1.10 ignores unknown HiGHS options, so its fallback makes no claim
    about a solver thread cap. Install the optional highspy dependency for
    reproducible one-thread solver profiling.
    """
    if highspy is None:
        import scipy
        return {"backend": "scipy_highs", "version": scipy.__version__,
                "thread_controlled": False, "threads": None, "parallel": None}
    version = ".".join(str(getattr(highspy, "HIGHS_VERSION_" + part))
                       for part in ("MAJOR", "MINOR", "PATCH"))
    return {"backend": "highspy", "version": version,
            "thread_controlled": True, "threads": 1, "parallel": "off"}


def _lp(objective, A_ub, b_ub, bounds, time_limit=None):
    options = {"primal_feasibility_tolerance": 1e-9,
               "dual_feasibility_tolerance": 1e-9}
    if time_limit is not None:
        options["time_limit"] = max(float(time_limit), 1e-6)
    if highspy is None:
        result = linprog(objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method="highs", options=options)
        result.update(solver_thread_capability())
        return result

    # Use the official API: old SciPy versions silently discard options such
    # as threads. The LP stays sparse and free dual variables keep both bounds.
    solver = highspy.Highs()
    options.update(output_flag=False, threads=1, parallel="off")
    for name, value in options.items():
        if solver.setOptionValue(name, value) != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS rejected option " + name)
    matrix = csc_matrix(A_ub, dtype=np.float64)
    model = highspy.HighsLp()
    model.num_col_ = len(objective)
    model.num_row_ = matrix.shape[0]
    model.col_cost_ = np.asarray(objective, dtype=np.float64)
    model.col_lower_ = np.asarray([-highspy.kHighsInf if lo is None else lo
                                  for lo, _ in bounds], dtype=np.float64)
    model.col_upper_ = np.asarray([highspy.kHighsInf if hi is None else hi
                                  for _, hi in bounds], dtype=np.float64)
    model.row_lower_ = np.full(matrix.shape[0], -highspy.kHighsInf)
    model.row_upper_ = np.asarray(b_ub, dtype=np.float64)
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_ = matrix.indptr
    model.a_matrix_.index_ = matrix.indices
    model.a_matrix_.value_ = matrix.data
    if solver.passModel(model) == highspy.HighsStatus.kError:
        raise ValueError("HiGHS rejected the LP model")
    run_status = solver.run()
    model_status = solver.getModelStatus()
    status_enum = highspy.HighsModelStatus
    if model_status == status_enum.kOptimal:
        status = 0
    elif model_status in (status_enum.kTimeLimit, status_enum.kIterationLimit,
                          status_enum.kObjectiveBound, status_enum.kObjectiveTarget):
        status = 1
    elif model_status == status_enum.kInfeasible:
        status = 2
    elif model_status == status_enum.kUnbounded:
        status = 3
    else:
        status = 4
    if run_status == highspy.HighsStatus.kError:
        status = 4
    solution, info = solver.getSolution(), solver.getInfo()
    metadata = solver_thread_capability()
    metadata["threads"] = solver.getOptionValue("threads")[1]
    metadata["parallel"] = solver.getOptionValue("parallel")[1]
    metadata["thread_controlled"] = (metadata["threads"] == 1 and metadata["parallel"] == "off")
    return OptimizeResult(success=status == 0, status=status,
                          message=solver.modelStatusToString(model_status),
                          x=np.asarray(solution.col_value, dtype=np.float64),
                          fun=float(info.objective_function_value),
                          nit=info.simplex_iteration_count + info.ipm_iteration_count,
                          **metadata)


def _result(gates, support, status, iterations, upper_mass, **details):
    details.setdefault("lp_backend", solver_thread_capability())
    return ControllerResult(np.asarray(gates).copy(), support.center, support.radius,
                            float(np.sum(gates)), status, iterations,
                            max(float(upper_mass), float(np.sum(gates))), details=details)


def _roundoff_tolerance(support, gates):
    """A scale-aware numerical tolerance, not a rigorous error enclosure."""
    magnitude = max(float(np.max(np.abs(support.lower_coeff) @ np.abs(gates))),
                    float(np.max(np.abs(support.upper_coeff) @ np.abs(gates))))
    return 64 * np.finfo(np.float64).eps * max(magnitude, np.finfo(np.float64).tiny)


def _safe_radial(problem, gates, epsilon, support):
    """Use only a complete support result to obtain a radial incumbent.

    At zero allowance residuals below a scale-aware roundoff tolerance are
    retained as numerical feasibility, not proof of exact equality. A support
    call evaluates any scaled gate vector, so reported centers are its midpoint.
    """
    radius = support.radius
    if radius <= epsilon or (epsilon == 0 and radius <= _roundoff_tolerance(support, gates)):
        return gates.copy(), support
    alpha = (epsilon / radius) * (1.0 - 64 * np.finfo(np.float64).eps)
    scaled = gates * alpha
    checked = problem.support(scaled)
    return scaled, checked


def solve_joint(problem: CatalogProblem, caps, epsilon, max_iterations=100, time_limit=None):
    """Maximize gate mass by assignment separation and finite LP cuts.

    A time limit is soft: already-started complete support calls finish before
    returning. Unfinished separation never supplies a purported radius bound.
    upper_mass is the numerical relaxation bound; status and numerical_only
    make clear that HiGHS tolerances are not outward-rounded certification.
    """
    caps, epsilon = _validate(problem, caps, epsilon)
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative")
    if problem.is_singleton or np.sum(caps) == 0:
        return _result(caps, problem.support(caps), "optimal_numerical", 0, np.sum(caps))
    started = monotonic()
    zero = np.zeros(problem.J)
    zeros = np.zeros((problem.N, problem.d))
    coeff_zeros = np.zeros((problem.N, problem.d, problem.J))
    best_support = SupportResult(zeros, zeros, coeff_zeros, coeff_zeros)
    best_gates = zero
    upper_mass = float(caps.sum())
    cuts = []
    cut_rhs = []
    cut_keys = set()
    status = "iteration_limit"
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        remaining = None if time_limit is None else time_limit - (monotonic() - started)
        if remaining is not None and remaining <= 0:
            status = "time_limit"
            break
        if cuts:
            relaxation = _lp(-np.ones(problem.J), np.array(cuts),
                             np.array(cut_rhs), list(zip(zero, caps)), remaining)
            if not relaxation.success:
                status = "time_limit" if relaxation.status == 1 else "solver_failure"
                break
            candidate = np.clip(relaxation.x, zero, caps)
            upper_mass = min(upper_mass, float(-relaxation.fun))
        else:
            candidate = caps.copy()
        complete_support = problem.support(candidate)
        iterations = iteration
        safe_gates, safe_support = _safe_radial(problem, candidate, epsilon, complete_support)
        if safe_gates.sum() > best_gates.sum():
            best_gates, best_support = safe_gates, safe_support
        numerical_tolerance = _roundoff_tolerance(complete_support, candidate)
        if (complete_support.radius <= epsilon + numerical_tolerance
                and upper_mass - safe_gates.sum() <= 1e-8 * max(1., caps.sum())):
            status = "optimal_numerical"
            # Usually safe_gates differs only by roundoff radial adjustment.
            best_gates, best_support = safe_gates, safe_support
            break
        widths = complete_support.upper - complete_support.lower
        violated = widths > 2 * epsilon + 2 * numerical_tolerance
        added = 0
        for cut in (complete_support.upper_coeff - complete_support.lower_coeff)[violated]:
            key = cut.tobytes()
            if key not in cut_keys:
                cut_keys.add(key)
                magnitude = np.max(np.abs(cut))
                cuts.append(cut / magnitude)
                cut_rhs.append(2 * epsilon / magnitude)
                added += 1
        if added == 0:
            status = "numerical_stall"
            break
    return _result(best_gates, best_support, status, iterations, upper_mass,
                   cuts=len(cuts), numerical_tolerance=_roundoff_tolerance(best_support, best_gates),
                   mass_gap=max(0.0, upper_mass - float(best_gates.sum())))


def solve_explicit(problem: CatalogProblem, caps, epsilon, eliminate_resolved=True):
    """Solve the sparse polynomial branch-dual LP, independently of cuts."""
    caps, epsilon = _validate(problem, caps, epsilon)
    if problem.is_singleton or caps.sum() == 0:
        return _result(caps, problem.support(caps), "optimal_numerical", 0, caps.sum(),
                       lp_variables=problem.J, lp_constraints=0)
    J, D = problem.J, problem.N * problem.d
    # Use a positive coordinate scaling to keep very small response units from
    # disappearing under the LP solver's absolute row-feasibility tolerance.
    # This bound is computed directly from input arrays, independently of the
    # support/cutting-plane path. Scaling changes neither widths nor gate sets.
    coordinate_scale = np.max(np.abs(problem._anchor), axis=0) / problem.scale
    trace_magnitudes = np.sum(np.abs(problem.W), axis=1)
    row_magnitudes = (np.max(np.abs(problem.B), axis=2) if problem.L
                      else np.zeros((J, 0)))
    for i, roots in enumerate(problem.feasible_roots):
        real_roots = [a for a in roots if a >= 0]
        if real_roots:
            source_bounds = np.max(row_magnitudes[:, real_roots], axis=1)
            readout_bound = np.max(source_bounds[:, None] * trace_magnitudes, axis=0)
            coordinate_scale[i] = np.maximum(coordinate_scale[i], readout_bound / problem.scale[i])
    coordinate_scale[coordinate_scale == 0] = 1
    bounds = list(zip(np.zeros(J), caps)) + [(None, None)] * (2 * D)
    rows, cols, data, rhs = [], [], [], []

    def add_constraint(indices, values, upper):
        row = len(rhs)
        for index, value in zip(indices, values):
            if value != 0:
                rows.append(row)
                cols.append(int(index))
                data.append(float(value))
        rhs.append(float(upper))

    for r in range(D):
        add_constraint([J + 2 * r, J + 2 * r + 1], [1, 1],
                       2 * epsilon / coordinate_scale.flat[r])
    gate_indices = list(range(J))
    for branch in problem.branches(eliminate_resolved=eliminate_resolved):
        nr, nc = len(branch.rows), len(branch.rights)
        for k in range(problem.d):
            r = branch.output * problem.d + k
            constant, edges = problem.branch_coefficients(branch, k)
            constant = constant / coordinate_scale[branch.output, k]
            edges = edges / coordinate_scale[branch.output, k]
            for sign_index, sign in enumerate((1.0, -1.0)):
                alpha_start = len(bounds)
                bounds.extend([(None, None)] * nr)
                beta_start = len(bounds)
                bounds.extend([(0.0, None)] * nc)
                for edge, (left, right) in enumerate(zip(branch.edge_rows, branch.edge_cols)):
                    add_constraint(gate_indices + [alpha_start + int(left), beta_start + int(right)],
                                   list(sign * edges[edge]) + [-1, -1], 0)
                # sign*c + sum(alpha) + sum(beta) <= t_r^sign.
                add_constraint(gate_indices + list(range(alpha_start, beta_start + nc))
                               + [J + 2 * r + sign_index],
                               list(sign * constant) + [1] * (nr + nc) + [-1], 0)
    nvar = len(bounds)
    objective = np.zeros(nvar)
    objective[:J] = -1
    matrix = coo_matrix((data, (rows, cols)), shape=(len(rhs), nvar)).tocsr()
    solution = _lp(objective, matrix, np.asarray(rhs), bounds)
    if not solution.success:
        raise RuntimeError(f"branch-dual LP failed: {solution.message}")
    candidate = np.clip(solution.x[:J], 0, caps)
    support = problem.support(candidate)
    gates, support = _safe_radial(problem, candidate, epsilon, support)
    return _result(gates, support, "optimal_numerical", 1, -solution.fun,
                   lp_variables=nvar, lp_constraints=len(rhs), lp_nonzeros=matrix.nnz,
                   eliminate_resolved=bool(eliminate_resolved))


def solve_baseline(problem: CatalogProblem, caps, epsilon, method):
    """Legacy, independent-envelope, and product-family admission controls.

    Each entry's marginal is the projection of the same feasible canonical
    family. Product admission drops cross-entry coupling only. The issued
    baseline center is the sum of independently computed entry midpoints; its
    guaranteed numerical envelope is correspondingly the product envelope.
    """
    caps, epsilon = _validate(problem, caps, epsilon)
    if method not in {"legacy", "independent", "product"}:
        raise ValueError("method must be legacy, independent, or product")
    marginal_lower = np.empty((problem.N, problem.d, problem.J))
    marginal_upper = np.empty_like(marginal_lower)
    for j in range(problem.J):
        unit = np.zeros(problem.J)
        unit[j] = 1
        support = problem.support(unit)
        marginal_lower[..., j] = support.lower
        marginal_upper[..., j] = support.upper
    widths = marginal_upper - marginal_lower
    budgets = widths.max(axis=(0, 1)) / 2
    if method == "legacy":
        rates = np.ones(problem.J)
        positive = budgets > 0
        # sum(caps)=1 recovers the paper exactly; larger boxes are normalized
        # solely for this legacy formula to maintain its envelope guarantee.
        rates[positive] = np.minimum(1.0, epsilon / (budgets[positive] * max(1., caps.sum())))
        gates = caps * rates
        upper_mass = float(gates.sum())
        iterations = 0
    else:
        A = budgets[None, :] if method == "independent" else widths.reshape(-1, problem.J) / 2
        row_scale = np.max(np.abs(A), axis=1)
        row_scale[row_scale == 0] = 1
        solution = _lp(-np.ones(problem.J), A / row_scale[:, None], epsilon / row_scale,
                       list(zip(np.zeros(problem.J), caps)))
        if not solution.success:
            raise RuntimeError(f"{method} LP failed: {solution.message}")
        gates = np.clip(solution.x, 0, caps)
        upper_mass = float(-solution.fun)
        iterations = 1
    lower, upper = marginal_lower @ gates, marginal_upper @ gates
    radius = float(np.max(upper - lower) / 2)
    product_support = SupportResult(lower, upper, marginal_lower, marginal_upper)
    tolerance = _roundoff_tolerance(product_support, gates)
    if radius > epsilon and not (epsilon == 0 and radius <= tolerance):
        gates *= (epsilon / radius) * (1 - 64 * np.finfo(np.float64).eps)
        lower, upper = marginal_lower @ gates, marginal_upper @ gates
        radius = float(np.max(upper - lower) / 2)
    joint_support = problem.support(gates)
    return ControllerResult(gates, (lower + upper) / 2, radius, float(gates.sum()),
                            "optimal_numerical" if method != "legacy" else "legacy_numerical",
                            iterations, max(upper_mass, float(gates.sum())),
                            details={"method": method, "joint_radius": joint_support.radius,
                                     "entry_budgets": budgets.tolist(),
                                     "lp_backend": solver_thread_capability()})
