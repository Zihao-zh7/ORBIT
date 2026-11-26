"""Observation-derived synthetic dependency experiments, with no author fixtures.

The shared latent loading controls the observation generator, not graph entries
or requested controller outcomes. Actual masked Pearson/top-k construction,
historically fitted linear responses, and full joint supports determine results.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .catalog import CatalogProblem
from .controllers import solve_baseline, solve_joint
from .graphs import correlation_graph


SEEDS = (20261012, 20261013, 20261014, 20261015)
STRENGTHS = (.1, .5, .9)


def generate_observations(strength, seed, length, n_catalog=6, missing_probability=.08):
    """Paired stationary AR(1) common/idiosyncratic factors, all loadings positive.

    x[t,i] = sqrt(s)*a[i]*z[t] + sqrt(1-s)*e[t,i].  Both factor
    families have AR coefficient .7 and unit stationary innovation variance.
    The same seed gives the same innovations and mask across requested strengths.
    Returned values retain latent missing coordinates solely to permit audits;
    downstream construction always uses the accompanying validity mask.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must lie in [0,1]')
    if length < 1 or n_catalog < 2 or not 0 <= missing_probability < 1:
        raise ValueError('invalid observation dimensions or missingness')
    rng = np.random.default_rng(seed)
    burn = 128
    innovations = rng.normal(size=(length + burn, n_catalog + 1))
    factors = np.zeros_like(innovations)
    for t in range(1, len(factors)):
        factors[t] = .7 * factors[t - 1] + np.sqrt(1 - .7 ** 2) * innovations[t]
    factors = factors[burn:]
    loadings = np.linspace(.8, 1.2, n_catalog)
    values = np.sqrt(strength) * factors[:, :1] * loadings + np.sqrt(1 - strength) * factors[:, 1:]
    valid = rng.uniform(size=values.shape) >= missing_probability
    return values, valid


def _pairwise_correlations(values, valid):
    result = []
    for first in range(values.shape[1]):
        for second in range(first + 1, values.shape[1]):
            mask = valid[:, first] & valid[:, second] & np.isfinite(values[:, first]) & np.isfinite(values[:, second])
            if mask.sum() < max(16, (len(values) + 1) // 2):
                continue
            a, b = values[mask, first], values[mask, second]
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                result.append(float(np.corrcoef(a, b)[0, 1]))
    return result


def build_observation_problem(values, valid, train_end, origin, window=64, gamma=.9,
                              ridge=.001, k=2):
    """Build one causal N=4,L=6,J=4,d=2 lineage admission problem.

    Archive windows occupy [0,4*window). Complete-target fitting follows those
    windows and ends before train_end. Evaluation histories reset at train_end
    and use observations through origin-1. The origin's future target is unused.
    Features are [1,last,previous,four-observation mean], standardized using only
    valid training observations; invalid feature inputs receive causal fill.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 6 or valid.shape != values.shape:
        raise ValueError('observations and masks must have shape [time,6]')
    if train_end > origin or origin > len(values):
        raise ValueError('training cutoff must be before the evaluation origin')
    if window < 32 or train_end <= 4 * window + 3 or origin < window or not 0 <= gamma < 1 or ridge <= 0:
        raise ValueError('insufficient historical observations or invalid fit parameters')
    valid = valid & np.isfinite(values)
    n, j, d, p = 4, 4, 2, 4
    archive_end = j * window
    B = np.stack([correlation_graph(values[a:a + window], valid[a:a + window], k)
                  for a in range(0, archive_end, window)])
    G = correlation_graph(values[origin - window:origin, :n], valid[origin - window:origin, :n], k)
    count = valid[:train_end, :n].sum(0)
    if np.any(count == 0):
        raise ValueError('each node needs observed training values')
    training = np.where(valid[:train_end, :n], values[:train_end, :n], 0)
    mean = training.sum(0) / count
    variance = np.where(valid[:train_end, :n], (values[:train_end, :n] - mean) ** 2, 0).sum(0) / count
    std = np.maximum(np.sqrt(variance), 1e-6)
    filled = np.empty((origin, n))
    last = mean.copy()
    for t in range(origin):
        last = np.where(valid[t, :n], values[t, :n], last)
        filled[t] = last
    standardized = (filled - mean) / std

    def feature(t):
        return np.stack([np.ones(n), standardized[t - 1], standardized[t - 2],
                         standardized[t - 4:t].mean(0)], axis=-1)

    caps = np.full(j, 1 / j)
    blocks, residuals, masks = [], [], []
    Z = np.zeros((n, p))
    fit_origins = range(archive_end, train_end - d + 1)
    for t in fit_origins:
        Z = gamma * Z + (1 - gamma) * feature(t)
        blocks.append(np.concatenate([caps[index] * B[index, :n, :n] @ Z for index in range(j)], axis=1))
        residuals.append(values[t:t + d, :n].T - filled[t - 1, :, None])
        masks.append(valid[t:t + d, :n].T)
    design, residual, mask = np.stack(blocks), np.stack(residuals), np.stack(masks)
    flattened = design.reshape(-1, j * p)
    projections = np.zeros((j, p, d))
    scale = np.zeros((n, d))
    for lead in range(d):
        observed = mask[..., lead].reshape(-1)
        if not observed.any():
            raise ValueError('each fitted lead needs at least one valid target')
        features = flattened[observed]
        labels = residual[..., lead].reshape(-1)[observed]
        normal = features.T @ features / len(features) + ridge * np.eye(j * p)
        projections[..., lead] = np.linalg.solve(normal, features.T @ labels / len(features)).reshape(j, p)
        pooled = float(np.sqrt(np.mean(labels ** 2)))
        for node in range(n):
            selected = mask[:, node, lead]
            scale[node, lead] = np.sqrt(np.mean(residual[selected, node, lead] ** 2)) if selected.any() else pooled
    scale = np.maximum(scale, 1e-6)
    Z = np.zeros((n, p))
    for t in range(train_end, origin + 1):
        Z = gamma * Z + (1 - gamma) * feature(t)
    W = np.einsum('np,jpd->jnd', Z, projections)
    # Truth is identity; all current nodes have their truth plus two distractors.
    # Real-label injectivity couples archives through one shared assignment.
    domains = [(0, 1, 4), (1, 2, 4), (2, 3, 5), (3, 0, 5)]
    problem = CatalogProblem(B, W, G, scale, domains)
    correlations = _pairwise_correlations(values[origin - window:origin, :n], valid[origin - window:origin, :n])
    archive_correlations = [_pairwise_correlations(values[a:a + window], valid[a:a + window])
                            for a in range(0, archive_end, window)]
    metadata = dict(N=n, L=6, J=j, d=d, p=p, k=k, gamma=gamma, ridge=ridge,
                    archive_window=window, archive_end_exclusive=archive_end,
                    train_end=train_end, origin=origin, last_observation_used=origin - 1,
                    last_fit_target_used=train_end - 1, fit_origins=len(blocks),
                    evaluation_trace_updates=origin - train_end + 1,
                    domains=[list(domain) for domain in domains], truth=[0, 1, 2, 3],
                    valid_observation_fraction=float(valid[:origin].mean()),
                    empirical_current_pairwise_correlations=correlations,
                    empirical_current_mean_correlation=float(np.mean(correlations)) if correlations else None,
                    empirical_archive_mean_correlations=[float(np.mean(row)) if row else None for row in archive_correlations],
                    history_payload_bytes=int(Z.nbytes), projection_norm=float(np.linalg.norm(projections)))
    return problem, metadata


def _geometry(problem, caps):
    lineages = [q for q in itertools.product(*problem.domains) if len(set(q)) == len(q)]
    coefficients = np.stack([problem.coefficients(q) for q in lineages])
    lower, upper = coefficients.min(0), coefficients.max(0)
    marginal_width = upper - lower
    joint = problem.support(caps)
    actual = coefficients @ caps
    support_error = max(float(np.max(np.abs(joint.lower - actual.min(0)))),
                        float(np.max(np.abs(joint.upper - actual.max(0)))))
    product_radius = float(np.max(marginal_width @ caps) / 2)
    independent_radius = float((marginal_width.max(axis=(0, 1)) / 2) @ caps)
    centered = coefficients - coefficients.mean(0, keepdims=True)
    pairs = []
    for first in range(problem.J):
        for second in range(first + 1, problem.J):
            a, b = centered[..., first].reshape(-1), centered[..., second].reshape(-1)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            pairs.append(float(a @ b / norm) if norm > 1e-15 else None)
    usable = [value for value in pairs if value is not None]
    return dict(enumerated_lineages=len(lineages), support_enumeration_max_error=support_error,
                geometry=dict(joint_full_cap_radius=joint.radius, product_full_cap_radius=product_radius,
                              independent_full_cap_radius=independent_radius,
                              joint_to_product_full_cap_radius_ratio=joint.radius / product_radius if product_radius else 0.,
                              uncertainty_pairwise_correlations=pairs,
                              uncertainty_mean_correlation=float(np.mean(usable)) if usable else None,
                              uncertainty_mean_absolute_correlation=float(np.mean(np.abs(usable))) if usable else None))


def observation_dependency_experiment(quick=False):
    """Three paired strengths x four seeds; quick uses one seed (three cases).

    Each case evaluates three predeclared allowances, retaining measured masses
    and radii even when dependence fails to improve them. No monotonic outcome is
    assumed, no paper table numbers are used, and no target loss is claimed.
    """
    started = time.perf_counter()
    rows = []
    for seed in SEEDS[:1] if quick else SEEDS:
        for strength in STRENGTHS:
            values, valid = generate_observations(strength, seed, length=464)
            assembly_start = time.perf_counter()
            problem, row = build_observation_problem(values, valid, train_end=384, origin=448)
            row.update(seed=seed, requested_shared_latent_strength=strength,
                       assembly_seconds=time.perf_counter() - assembly_start,
                       operator_row_sum_max_error=max(float(np.max(np.abs(problem.B.sum(-1) - 1))),
                                                      float(np.max(np.abs(problem.G.sum(-1) - 1)))))
            caps = np.full(problem.J, 1 / problem.J)
            row.update(_geometry(problem, caps))
            evaluations = []
            for epsilon in (0., .025, .1):
                methods = {}
                for name in ('legacy', 'independent', 'product', 'joint'):
                    t0 = time.perf_counter()
                    result = solve_joint(problem, caps, epsilon) if name == 'joint' else solve_baseline(problem, caps, epsilon, name)
                    duration = time.perf_counter() - t0
                    support = problem.support(result.gates)
                    # Baselines use product-envelope midpoints, so separately
                    # report discrepancy of their actually issued centers.
                    issued_radius = float(np.max(np.maximum(np.abs(result.center - support.lower),
                                                            np.abs(result.center - support.upper))))
                    methods[name] = dict(mass=result.mass, gates=result.gates.tolist(), status=result.status,
                                         controller_radius=result.radius, shared_family_radius=support.radius,
                                         issued_center_shared_family_radius=issued_radius,
                                         retained_correction_scaled_linf=float(np.max(np.abs(result.center))),
                                         controller_seconds=duration, iterations=result.iterations,
                                         upper_mass=result.upper_mass)
                masses = [methods[name]['mass'] for name in ('legacy', 'independent', 'product', 'joint')]
                evaluations.append(dict(epsilon=epsilon, methods=methods,
                                        hierarchy_valid=bool(np.all(np.diff(masses) >= -1e-7))))
            row['controllers'] = evaluations
            rows.append(row)
    return dict(provenance=dict(source='synthetic observations; reconstructed dependency-strength experiment',
                                author_fixture=False, paper_values_used_as_output=False,
                                numerical_only=True, formal_interval_certificate=False,
                                seeds=list(SEEDS[:1] if quick else SEEDS), strengths=list(STRENGTHS),
                                quick=bool(quick), observation_model='paired AR(1) common and idiosyncratic factors',
                                operator_builder='orbit.graphs.correlation_graph, common-valid Pearson, positive top-k union and row normalization',
                                forecast_accuracy_claimed=False,
                                monotonic_gate_mass_claimed=False),
                cases=rows, elapsed_seconds=time.perf_counter() - started)
