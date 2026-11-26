"""Compact paired accuracy evidence and measured, same-gates repair audits.

Bootstrap sampling is over contiguous temporal origin blocks. Nodes and leads
are never independently resampled. Repair measurements reuse the exact feature
stream already realized by evaluation and exclude backbone/graph/controller work.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np

from .catalog import CatalogProblem
from .protocol import MetadataSimulator


DAY_BLOCK_ORIGINS = {'metr-la': 288, 'pems-bay': 288, 'knowair': 8}


def _loss_row(prediction, target, valid, leads):
    absolute = np.zeros(len(leads), dtype=np.float64)
    square = np.zeros(len(leads), dtype=np.float64)
    count = np.zeros(len(leads), dtype=np.int64)
    for column, lead in enumerate(leads):
        mask = np.asarray(valid[..., lead-1], dtype=bool)
        # Cast before subtraction, including for float16-backed compact caches.
        difference = (np.asarray(prediction[..., lead-1], dtype=np.float64)
                      - np.asarray(target[..., lead-1], dtype=np.float64))[mask]
        if not np.isfinite(difference).all():
            raise FloatingPointError('Nonfinite prediction/target on a valid coordinate')
        absolute[column] = np.abs(difference).sum()
        square[column] = (difference*difference).sum()
        count[column] = len(difference)
    return absolute, square, count


class OriginLossAccumulator:
    """Keep only three numbers per origin/lead, rather than node forecasts."""
    def __init__(self, origins, leads):
        self.origins = np.asarray(origins).copy()
        self.leads = np.asarray(leads, dtype=int)
        shape = (len(self.origins), len(self.leads))
        self.absolute = np.empty(shape, dtype=np.float64)
        self.square = np.empty(shape, dtype=np.float64)
        self.count = np.empty(shape, dtype=np.int64)

    def update(self, index, prediction, target, valid):
        self.absolute[index], self.square[index], self.count[index] = _loss_row(prediction, target, valid, self.leads)

    def result(self):
        return {name: getattr(self, name) for name in ['absolute', 'square', 'count', 'origins', 'leads']}


def origin_loss_statistics(predictions, targets, valid, leads, origins=None):
    """Read arrays/memmaps one origin at a time; never copy a full forecast cube."""
    if predictions.shape != targets.shape or predictions.shape != valid.shape or len(predictions.shape) != 3:
        raise ValueError('Expected matching [origin,node,lead] prediction/target/mask shapes')
    if len(leads) == 0 or min(leads) < 1 or max(leads) > predictions.shape[-1]:
        raise ValueError('Selected leads are outside the forecast horizon')
    origins = np.arange(predictions.shape[0]) if origins is None else origins
    if len(origins) != predictions.shape[0]:
        raise ValueError('One origin identifier is required per prediction')
    accumulator = OriginLossAccumulator(origins, leads)
    for index in range(predictions.shape[0]):
        accumulator.update(index, predictions[index], targets[index], valid[index])
    return accumulator.result()


def save_origin_losses(path, statistics):
    """Write compact arrays for every method to one compressed NumPy archive."""
    payload = {}
    for method, values in statistics.items():
        values = values.result() if isinstance(values, OriginLossAccumulator) else values
        payload.update({f'{method}__{key}': value for key, value in values.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_origin_losses(path, method):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[f'{method}__{key}'] for key in ['absolute', 'square', 'count', 'origins', 'leads']}


def paired_block_bootstrap(a, b, dataset, leads, block_length=None, resamples=1000, seed=2026):
    """Percentile 95% paired circular-block intervals for a minus b, in units.

    Each draw concatenates independently sampled circular blocks and truncates
    to the original number of origins. The same origin indices select both
    methods and all leads. MAE pools sums/counts; RMSE takes the square root only
    after pooling squared errors. The all-leads result pools valid coordinates.
    This conditions on the trained checkpoint and selected hyperparameters.
    """
    leads = list(leads)
    if dataset not in DAY_BLOCK_ORIGINS:
        raise ValueError('Dataset must have a predeclared temporal block length')
    block_length = DAY_BLOCK_ORIGINS[dataset] if block_length is None else block_length
    if (not isinstance(block_length, (int, np.integer)) or block_length < 1
            or not isinstance(resamples, (int, np.integer)) or resamples < 1):
        raise ValueError('Require positive integer block length and resample count')
    arrays = [{key: np.asarray(values[key]) for key in ['absolute', 'square', 'count']} for values in [a, b]]
    shape = arrays[0]['count'].shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] != len(leads):
        raise ValueError('Statistics require nonempty [origin,selected lead] arrays')
    if any(value.shape != shape or not np.isfinite(value).all() or np.any(value < 0)
           for values in arrays for value in values.values()):
        raise ValueError('Loss statistics must be finite nonnegative arrays with matching shapes')
    if not np.array_equal(arrays[0]['count'], arrays[1]['count']):
        raise ValueError('Paired statistics must have matching valid-coordinate counts')
    if any(not np.all(values['count'] == np.floor(values['count'])) for values in arrays):
        raise ValueError('Counts must be integers')
    for key in ['origins', 'leads']:
        if (key in a) != (key in b) or (key in a and not np.array_equal(a[key], b[key])):
            raise ValueError(f'Paired statistics have different {key}')
    if 'leads' in a and not np.array_equal(a['leads'], leads):
        raise ValueError('Selected leads differ from stored loss-statistic leads')
    if 'origins' in a and (len(a['origins']) != shape[0] or np.any(np.diff(a['origins']) != 1)):
        raise ValueError('Temporal block evidence requires consecutive paired origins')
    if np.any(arrays[0]['count'].sum(axis=0) == 0):
        raise ValueError('Every selected lead needs at least one valid coordinate')

    # Append an all-selected-leads column, using exactly the same sampled blocks.
    packed = []
    for values in arrays:
        packed.append({key: np.concatenate([value, value.sum(axis=1, keepdims=True)], axis=1)
                       for key, value in values.items()})

    def estimates(values, chosen=None):
        counts = values['count'] if chosen is None else values['count'][chosen]
        denominator = counts.sum(axis=0)
        absolute = values['absolute'] if chosen is None else values['absolute'][chosen]
        square = values['square'] if chosen is None else values['square'][chosen]
        mae = np.divide(absolute.sum(axis=0), denominator, out=np.full(len(leads)+1, np.nan), where=denominator > 0)
        mse = np.divide(square.sum(axis=0), denominator, out=np.full(len(leads)+1, np.nan), where=denominator > 0)
        return np.stack([mae, np.sqrt(mse)])

    observed_a, observed_b = estimates(packed[0]), estimates(packed[1])
    rng = np.random.default_rng(seed)
    n_origins = shape[0]
    n_blocks = (n_origins+block_length-1)//block_length
    offsets = np.arange(min(block_length, n_origins))
    # With at most one block, circular sampling only permutes the entire
    # series. Reporting its point-valued interval would imply false precision.
    inferential_resamples = resamples if n_origins > block_length else 0
    draws = np.empty((inferential_resamples, 2, len(leads)+1), dtype=np.float64)
    for index in range(inferential_resamples):
        starts = rng.integers(0, n_origins, size=n_blocks)
        chosen = ((starts[:, None]+offsets)%n_origins).ravel()[:n_origins]
        draws[index] = estimates(packed[0], chosen)-estimates(packed[1], chosen)

    def report_column(column):
        report = {}
        for metric_index, metric in enumerate(['mae', 'rmse']):
            samples = draws[:, metric_index, column]
            samples = samples[np.isfinite(samples)]
            report[metric] = {'a': float(observed_a[metric_index, column]),
                              'b': float(observed_b[metric_index, column]),
                              'difference': float(observed_a[metric_index, column]-observed_b[metric_index, column]),
                              'ci95': np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
                              'finite_resamples': int(len(samples)),
                              'bootstrap_fraction_a_better': float(np.mean(samples < 0)) if len(samples) else None}
        return report

    return {'dataset': dataset, 'origins': n_origins, 'leads': leads,
            'block_length_origins': int(block_length), 'resamples': int(resamples), 'seed': int(seed),
            'default_block_basis': 'one day: 288 five-minute traffic origins or 8 three-hour KnowAir origins',
            'interval': 'paired circular temporal block bootstrap, percentile 95%',
            'inference_status': 'available' if inferential_resamples else 'insufficient_temporal_replication',
            'difference_direction': 'a minus b',
            'resampling_unit': 'temporal blocks of origins; all nodes and selected leads travel together',
            'scope': 'test-origin uncertainty conditional on the fitted checkpoint and validation-selected configuration; not across-seed uncertainty',
            'short_series_caveat': 'series has at most one block; inferential intervals withheld because circular resampling only permutes all origins' if n_origins <= block_length else None,
            'per_lead': {str(lead): report_column(column) for column, lead in enumerate(leads)},
            'all_selected_leads': report_column(len(leads))}


class TimedMetadataSimulator(MetadataSimulator):
    """Measure validation/domain restriction, excluding queue wait and dispatch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accept_timings = []

    def accept(self, answer):
        started = time.perf_counter()
        accepted = False
        try:
            accepted = super().accept(answer)
            return accepted
        finally:
            finished = time.perf_counter()
            self._accept_timings.append({'accepted': bool(accepted), 'phase': answer.get('phase', 'unspecified'),
                                         'node': answer.get('node'), 'duration_ms': 1000*(finished-started)})

    def drain_accept_timings(self):
        events, self._accept_timings = self._accept_timings, []
        return events


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class RepairAudit:
    """Paired repairs at real accepted-answer batches, with no synthetic timings."""
    def __init__(self, embedded, scale, projections, gamma, caps):
        self.embedded = embedded
        self.scale = scale
        self.projections = projections
        self.gamma = gamma
        self.caps = caps
        self.events = []
        self.skipped_uninitialized = 0
        self.rejected_answers = 0

    def _readout(self, w, previous, domains):
        anchor = previous['base']+np.einsum('nv,jvd,j->nd', previous['G'], w, self.caps)
        center = CatalogProblem(self.embedded, w, previous['G'], self.scale, domains).support(previous['gates']).center
        return anchor+self.scale*center

    def record(self, round_index, simulator, previous, bank, history):
        timings = simulator.drain_accept_timings()
        accepted = [event for event in timings if event['accepted']]
        self.rejected_answers += sum(not event['accepted'] for event in timings)
        if not accepted:
            return
        if previous is None:
            self.skipped_uninitialized += 1
            return
        if not history or len(history) != bank.rounds:
            raise ValueError('Repair prefix and retained sufficient history must cover the same origins')
        issued_digest = _digest(previous['issued'])
        durations, readouts = {}, {}
        # Alternate which operation runs first, reducing consistent cache/order bias.
        order = ['retained', 'replay'] if len(self.events)%2 == 0 else ['replay', 'retained']
        reconstructed = 0
        for method in order:
            started = time.perf_counter()
            if method == 'retained':
                w = bank.projected()
            else:
                z = np.zeros_like(history[0])
                for features in history:
                    z = self.gamma*z+(1-self.gamma)*features
                    reconstructed += 1
                w = np.einsum('np,jpd->jnd', z, self.projections)
            readouts[method] = self._readout(w, previous, simulator.domains)
            durations[method] = 1000*(time.perf_counter()-started)
        if any(not np.isfinite(value).all() for value in readouts.values()):
            raise FloatingPointError('Nonfinite repaired readout')
        if _digest(previous['issued']) != issued_digest:
            raise RuntimeError('Repair audit mutated an issued prediction')
        validation_ms = sum(event['duration_ms'] for event in timings)
        self.events.append({'round': int(round_index), 'phase': accepted[0]['phase'],
                            'previous_round': int(previous['round']), 'previous_origin': int(previous['origin']),
                            'accepted_answers': len(accepted), 'prefix_origins': len(history),
                            'replay_reconstructed_origins': reconstructed,
                            'validation_and_restriction_ms': validation_ms,
                            'retained_readout_ms': durations['retained'], 'replay_readout_ms': durations['replay'],
                            'retained_repair_only_ms': validation_ms+durations['retained'],
                            'replay_repair_only_ms': validation_ms+durations['replay'],
                            'measurement_order': order, 'trace_layout': bank.layout,
                            'trace_payload_bytes': bank.payload_bytes,
                            'shared_prefix_payload_bytes': sum(features.nbytes for features in history),
                            'max_abs_readout_difference': float(np.max(np.abs(readouts['retained']-readouts['replay']))),
                            'issued_sha256': issued_digest, 'issued_unchanged': True})

    def result(self):
        def timing(key):
            values = [event[key] for event in self.events]
            return {'samples': len(values), **{f'p{q}_ms': float(np.percentile(values, q)) if values else None for q in [50, 95, 99]},
                    'total_ms': float(sum(values))}
        retained = timing('retained_repair_only_ms')
        replay = timing('replay_repair_only_ms')
        return {'samples': len(self.events), 'events': self.events,
                'retained_repair': retained, 'replay_repair': replay,
                'ratio_total_replay_to_retained': replay['total_ms']/retained['total_ms'] if retained['total_ms'] > 0 else None,
                'max_abs_readout_difference': max((event['max_abs_readout_difference'] for event in self.events), default=None),
                'forecast_rewrites': 0, 'skipped_uninitialized_batches': self.skipped_uninitialized,
                'rejected_answers': self.rejected_answers,
                'timing_scope': 'paired actual accepted-answer batches: evidence validation/domain restriction plus same-gates corrected readout at the previously issued graph; retained projection versus explicit full-prefix history reconstruction and projection',
                'shared_work': 'one realized transformed feature stream, frozen sources/projections, previous base forecast/graph/gates, and one measured validation/restriction event reused in both totals',
                'excluded': ['backbone inference', 'feature transformation', 'graph construction', 'controller admission',
                             'feature-prefix storage', 'exogenous answer waiting and queue dispatch', 'artifact serialization'],
                'host_exclusivity': 'shared-host wall-clock measurement; exclusivity is not assumed or independently verified',
                'interpretation': 'repair-only microbenchmark, not cached-pipeline or end-to-end latency; timing order alternates; no padding, extrapolated or synthetic samples'}
