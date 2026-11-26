"""Controlled manifests and distinct observation/metadata clocks (supplement IV)."""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment


def feasible_domains(domains):
    if not domains or any(not d for d in domains):
        return False
    if any(not isinstance(q, (int, np.integer)) or isinstance(q, bool) or q < -1 for d in domains for q in d):
        return False
    labels = sorted({int(q) for d in domains for q in d if q >= 0})
    index = {q: i for i, q in enumerate(labels)}
    costs = np.full((len(domains), len(labels)+len(domains)), np.inf)
    for i, domain in enumerate(domains):
        for q in domain:
            if q < -1:
                return False
            costs[i, len(labels)+i if q == -1 else index[q]] = 0
    try:
        row, col = linear_sum_assignment(costs)
        return len(row) == len(domains) and bool(np.isfinite(costs[row, col]).all())
    except ValueError:
        return False


def generate_manifest(n_nodes, catalog_size=None, unresolved=8, domain_size=3,
                      seed=2026, epoch=0, allow_null=False, require_unused_coverage=False):
    """Reconstruction default: observable order is the true catalog order.

    Training/source coordinates are not secretly permuted. Ambiguity is generated
    around this truth; the controller only receives the domains and known sources.
    """
    catalog_size = n_nodes+4 if catalog_size is None else int(catalog_size)
    if catalog_size < n_nodes or not 0 <= unresolved <= n_nodes or domain_size < 1:
        raise ValueError('Invalid manifest dimensions')
    rng = np.random.default_rng(seed)
    truth = np.arange(n_nodes)
    unresolved_nodes = np.sort(rng.choice(n_nodes, unresolved, replace=False))
    pool = np.concatenate((truth[unresolved_nodes], np.arange(n_nodes, catalog_size)))
    if domain_size > len(pool)+int(allow_null) and unresolved:
        raise ValueError('Insufficient distractors for requested domain size')
    if allow_null and domain_size < 2 and unresolved:
        raise ValueError('Truth plus optional null requires K>=2')
    for _ in range(10000):
        domains = [[int(q)] for q in truth]
        for i in unresolved_nodes:
            base = [int(truth[i])] + ([-1] if allow_null else [])
            candidates = pool[pool != truth[i]]
            domains[i] = sorted(base + list(map(int, rng.choice(candidates, domain_size-len(base), replace=False))))
        used = {q for i in unresolved_nodes for q in domains[i]}
        if not require_unused_coverage or set(range(n_nodes, catalog_size)).issubset(used):
            break
    else:
        raise ValueError('Cannot realize unused-label coverage')
    result = {'epoch': epoch, 'source_versions': ['source-0','source-1','source-2','source-3'],
              'catalog_size': catalog_size, 'truth': truth.tolist(), 'domains': domains,
              'unresolved_nodes': unresolved_nodes.tolist(), 'generator_seed': int(seed),
              'assumption': 'reconstructed manifest; true identity follows observable dataset order'}
    validate_manifest(result)
    return result


def validate_manifest(manifest):
    if 'epoch' not in manifest or 'source_versions' not in manifest:
        raise ValueError('Epoch and source versions are required')
    domains = manifest['domains']
    if any(q >= manifest['catalog_size'] or q < -1 for d in domains for q in d) or not feasible_domains(domains):
        raise ValueError('Invalid or empty injective family')
    if 'truth' in manifest:
        truth = manifest['truth']
        real = [q for q in truth if q >= 0]
        if len(truth) != len(domains) or len(set(real)) != len(real) or any(q not in d for q, d in zip(truth, domains)):
            raise ValueError('Generating truth is not contained')


class MetadataSimulator:
    def __init__(self, manifest, delay=12, refill=1/6, capacity=4):
        validate_manifest(manifest)
        if delay < 0 or refill < 0 or capacity < 0:
            raise ValueError('Invalid metadata clock/budget')
        self.manifest = copy.deepcopy(manifest)
        self._domains = [tuple(d) for d in manifest['domains']]
        self.delay, self.refill, self.capacity = int(delay), float(refill), float(capacity)
        self.balance, self.cursor = 0., 0
        self.pending = {}
        self.events = []
        self.dispatch_count = 0
        self.rejected = 0
        self.current_round = -1
        self.immediate_returns = []

    @property
    def domains(self):
        return list(self._domains)

    def accept(self, answer):
        required = {'epoch', 'source_versions', 'node', 'identity'}
        allowed = required | {'round', 'phase', 'request_round'}
        valid = required.issubset(answer) and not (set(answer)-allowed)
        valid = valid and answer.get('epoch') == self.manifest['epoch']
        valid = valid and answer.get('source_versions') == self.manifest['source_versions']
        node, identity = answer.get('node'), answer.get('identity')
        valid = valid and isinstance(node, (int, np.integer)) and not isinstance(node,bool) and 0 <= node < len(self._domains)
        valid = valid and isinstance(identity, (int, np.integer)) and not isinstance(identity,bool)
        if valid:
            valid = identity in self._domains[node]
        if valid:
            restricted = list(self._domains)
            restricted[node] = (int(identity),)
            valid = feasible_domains(restricted)
        if not valid:
            self.rejected += 1
            self.events.append({'type': 'rejection', 'round': self.current_round, 'answer': copy.deepcopy(answer)})
            return False
        self._domains = restricted
        self.events.append({'type': 'return', 'round': self.current_round, 'answer': copy.deepcopy(answer)})
        return True

    def before_round(self, t):
        if t <= self.current_round:
            raise ValueError('Rounds must strictly increase')
        self.current_round = int(t)
        returned = []
        for node, request in list(self.pending.items()):
            if request['effective_round'] <= t:
                answer = {'epoch': self.manifest['epoch'], 'source_versions': self.manifest['source_versions'],
                          'node': node, 'identity': self.manifest['truth'][node],
                          'round': request['return_round'], 'phase': request['phase'],
                          'request_round': request['request_round']}
                if self.accept(answer):
                    returned.append(answer)
                del self.pending[node]
        return returned

    def after_issuance(self, t):
        if t != self.current_round:
            raise ValueError('Call before_round before sealing/dispatching')
        self.balance = min(self.capacity, self.balance+self.refill)
        self.immediate_returns = []
        dispatched = []
        for _ in range(len(self._domains)):
            if self.balance+1e-12 < 1:
                break
            node = self.cursor
            self.cursor = (self.cursor+1) % len(self._domains)
            if len(self._domains[node]) <= 1 or node in self.pending:
                continue
            request = {'node': node, 'request_round': int(t), 'return_round': int(t+self.delay),
                       'effective_round': int(t+max(1, self.delay)),
                       'phase': 'post-issuance' if self.delay == 0 else 'pre-issuance'}
            self.pending[node] = request
            self.balance = max(0., self.balance-1)
            self.dispatch_count += 1
            self.events.append({'type': 'dispatch', **request})
            dispatched.append(request)
            if self.delay == 0:
                answer = {'epoch': self.manifest['epoch'], 'source_versions': self.manifest['source_versions'],
                          'node': node, 'identity': self.manifest['truth'][node], 'round': int(t),
                          'phase': 'post-issuance', 'request_round': int(t)}
                if self.accept(answer):
                    self.immediate_returns.append(answer)
                del self.pending[node]
        return dispatched


def eligible_origins(current_round, output_steps, already_used=()):
    used = set(already_used)
    return [s for s in range(max(0, current_round-output_steps+1)) if s not in used]


@dataclass
class ForecastLedger:
    predictions: dict = field(default_factory=dict)
    records: list = field(default_factory=list)

    def seal(self, t, prediction, metadata):
        if t in self.predictions:
            raise ValueError('Issued forecasts are immutable')
        prediction = np.array(prediction, dtype=np.float64, copy=True)
        prediction.flags.writeable = False
        self.predictions[t] = prediction
        self.records.append({'round': int(t), **copy.deepcopy(metadata)})
