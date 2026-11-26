"""Real-data experiments. Published numbers are comparison references only."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import itertools
import json
import platform
import shutil
import time
import numpy as np
import psutil
from .catalog import CatalogProblem
from .controllers import solve_joint, solve_baseline, ControllerResult
from .graphs import RollingGraph, archive_operators
from .memory import HistoryBank
from .protocol import MetadataSimulator, generate_manifest
from .residual import FeatureTransform, residual_scale, solve_ridge_statistics


PAPER_REFERENCE = {
    'metr-la': {'frozen': {'mae': 3.588, 'rmse': 7.145}, 'tuned': {'mae': 3.473, 'rmse': 6.905}, 'orbit': {'mae': 3.327, 'rmse': 6.603}},
    'pems-bay': {'frozen': {'mae': 2.056, 'rmse': 4.772}, 'tuned': {'mae': 1.936, 'rmse': 4.560}, 'orbit': {'mae': 1.865, 'rmse': 4.376}},
    'knowair': {'frozen': {'mae': 22.555, 'rmse': 34.971}, 'tuned': {'mae': 21.379, 'rmse': 33.811}, 'orbit': {'mae': 20.297, 'rmse': 31.712}},
}


def write_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=encode, allow_nan=False), encoding='utf-8')


class MetricAccumulator:
    def __init__(self, leads):
        self.leads = list(leads)
        self.absolute = np.zeros(len(leads), dtype=np.float64)
        self.square = np.zeros(len(leads), dtype=np.float64)
        self.count = np.zeros(len(leads), dtype=np.int64)

    def update(self, prediction, target, valid):
        for i, h in enumerate(self.leads):
            mask = np.asarray(valid[..., h-1], dtype=bool)
            delta = np.asarray(prediction[..., h-1]-target[..., h-1], dtype=np.float64)[mask]
            if not np.isfinite(delta).all():
                raise FloatingPointError('Nonfinite forecast on a valid target')
            self.absolute[i] += np.abs(delta).sum()
            self.square[i] += (delta*delta).sum()
            self.count[i] += delta.size

    def result(self):
        return {str(h): {'mae': float(self.absolute[i]/self.count[i]) if self.count[i] else None,
                         'rmse': float(np.sqrt(self.square[i]/self.count[i])) if self.count[i] else None,
                         'count': int(self.count[i])} for i, h in enumerate(self.leads)}


def scored_mae(metrics, leads):
    """Unweighted MAE over all valid coordinates and selected leads."""
    selected = [metrics[str(h)] for h in leads]
    count = sum(row['count'] for row in selected)
    if count == 0:
        raise ValueError('No valid evaluation coordinates')
    return sum(row['mae']*row['count'] for row in selected if row['count'])/count


@dataclass
class OrbitConfig:
    k: int = 8
    gamma: float = .9
    beta: float = .001
    feature_width: int = 64
    epsilon: float = .1
    unresolved: int = 8
    domain_size: int = 3
    generator_seed: int = 2026
    epoch_rounds: int = 288
    delay: int | None = None
    refill: float | None = None
    capacity: float = 4
    time_limit: float | None = None
    max_iterations: int = 100
    archive_mode: str = 'default'
    reset_on_manifest_epoch: bool = True


def experiment_matrix():
    matrix = []
    for dataset in PAPER_REFERENCE:
        for kind in ['main_stid', 'identity_free', 'delayed_repair', 'archive_signal', 'metadata_intervention', 'resources', 'sensitivity']:
            matrix.append({'dataset': dataset, 'kind': kind, 'status': 'not_run'})
    for dataset in ['metr-la', 'knowair']:
        for seed in range(5):
            matrix.append({'dataset': dataset, 'kind': 'staeformer', 'seed': seed, 'status': 'not_run'})
    for method in ['FSNet', 'OneNet', 'PROCEED', 'DSOF', 'ODGNet', 'ADAPT-Z', 'PADRE']:
        matrix.append({'method': method, 'kind': 'external_baseline', 'status': 'implemented_full_benchmark_pending',
                       'reason': 'Native source mechanisms ported for six methods; PADRE independently implemented from paper. ORBIT adaptation settings are reconstructed; measured completion is tracked separately.'})
    matrix += [{'kind': kind, 'status': 'not_run'} for kind in ['exactness_12', 'joint_geometry', 'synthetic_replay', 'faults', 'solver_scale']]
    return matrix


def load_cache(path):
    path = Path(path)
    metadata = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('status')=='in_progress' or metadata.get('completed') is False:
        raise ValueError('Feature cache is incomplete; extract it again before evaluation')
    caches = {split: {name: np.load(path/split/(name+'.npy'), mmap_mode='r') for name in
                      ['origins','predictions','targets','valid','features']} for split in ['train','val','test']}
    return caches, metadata


def _graph_stream(data, origins, k):
    window = 240 if data.name == 'knowair' else 2016
    rolling = RollingGraph(window, data.num_nodes, k)
    cursor = max(0, int(origins[0])-window)
    for origin in origins:
        while cursor < origin:
            rolling.update(data.values[cursor], data.valid[cursor])
            cursor += 1
        yield rolling.operator()


def _sources(data, config):
    return archive_operators(data.values, data.valid, data.splits['train'][1],
                             720 if data.name == 'knowair' else 4032, config.k, mode=config.archive_mode)


def fit_interface(data, train, config, transform=None, scale=None):
    """Stream normal equations; no future validation/test labels participate.

    All four current-only blocks are identical because histories agree, so the
    tuned fit reduces algebraically to width p and ridge 4*beta. The archive fit
    retains all four heterogeneous operator blocks (width 4*p).
    """
    started = time.perf_counter()
    transform = FeatureTransform.fit(train['features'], config.feature_width) if transform is None else transform
    if scale is None:
        from .residual import residual_scale_cached
        scale = residual_scale_cached(train['targets'],train['predictions'],train['valid'])
    p, d, n = 1+transform.directions.shape[1], data.output_steps, data.num_nodes
    sources = _sources(data, config)
    from .fitting import BlockRidgeAccumulator
    accumulator=BlockRidgeAccumulator(4*p,p,scale,block_size=32)
    z = np.zeros((n, p))
    for t, graph in enumerate(_graph_stream(data, train['origins'], config.k)):
        z = config.gamma*z+(1-config.gamma)*transform.transform(train['features'][t])
        archived = np.einsum('jnv,vp->njp', sources, z).reshape(n, 4*p)/4
        current = graph@z
        residual=np.asarray(train['targets'][t],dtype=np.float64)-train['predictions'][t]
        accumulator.update(archived,current,residual,train['valid'][t])
        if (t+1) % 1000 == 0:
            print(json.dumps({'stage':'ridge_statistics','dataset':data.name,'origins':t+1,'total':len(train['origins'])}), flush=True)
    archive_statistics,current_statistics=accumulator.finish()
    return {'transform': transform, 'scale': scale, 'sources': sources,
            'archive_statistics': archive_statistics, 'current_statistics': current_statistics,
            'feature_width': p, 'seconds': time.perf_counter()-started}


def projections_for_beta(interface, beta):
    p = interface['feature_width']
    archive = solve_ridge_statistics(interface['archive_statistics'], beta).reshape(4, p, -1)
    current = solve_ridge_statistics(interface['current_statistics'], 4*beta)
    return archive, np.repeat(current[None, ...], 4, axis=0)


def save_interface(path, interface, archive_v, current_v, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = interface['transform']
    np.savez(path, mean=transform.mean, std=transform.std, directions=transform.directions,
             scale=interface['scale'], sources=interface['sources'], archive_v=archive_v, current_v=current_v,
             config=json.dumps(asdict(config)), provenance=json.dumps(interface.get('provenance',{})))


def read_interface(path):
    values = np.load(path, allow_pickle=False)
    interface = {'transform': FeatureTransform(values['mean'], values['std'], values['directions']),
                 'scale': values['scale'], 'sources': values['sources'],
                 'provenance':json.loads(str(values['provenance'])) if 'provenance' in values else {}}
    return interface, values['archive_v'], values['current_v'], OrbitConfig(**json.loads(str(values['config'])))


def _checked_problem(*args):
    try:
        return CatalogProblem(*args),None
    except (ValueError,FloatingPointError,RuntimeError) as error:
        return None,f'{type(error).__name__}: {error}'


def _admit(problem,solve,shape,archives,error=None):
    try:
        if problem is None:
            raise ValueError(error or 'Unavailable feasible lineage family')
        result=solve()
        if not np.isfinite(result.center).all() or not np.isfinite(result.gates).all() or not np.isfinite(result.radius):
            raise FloatingPointError('Nonfinite admission output')
        return result
    except (ValueError,FloatingPointError,RuntimeError) as failure:
        return ControllerResult(np.zeros(archives),np.zeros(shape),0.,0.,'fallback_current_anchor',0,1.,
                                details={'reason':f'{type(failure).__name__}: {failure}'})


def evaluate_stream(data, cache, interface, archive_v, current_v, config, methods=None, output=None,
                    injected_answers=None, fixed_gates=None, save_predictions=True,
                    audit_repair=False, capture_origin_losses=False):
    methods = methods or ['frozen','strict','tuned','legacy','independent','product','orbit','forward','reset']
    allowed = {'frozen','strict','tuned','legacy','independent','product','orbit','forward','reset','replay'}
    if set(methods)-allowed:
        raise ValueError('Unknown method')
    if config.epoch_rounds < 1:
        raise ValueError('epoch_rounds must be positive')
    if audit_repair and 'orbit' not in methods:
        raise ValueError('Repair audit requires the orbit method')
    need_joint = bool(set(methods)&{'orbit','forward','replay'})
    metrics = {m: MetricAccumulator(data.leads) for m in methods}
    losses = {}
    if capture_origin_losses:
        from .evidence import OriginLossAccumulator
        losses = {m: OriginLossAccumulator(cache['origins'], data.leads) for m in methods}
    caps = np.full(4, .25)
    bank = HistoryBank(data.num_nodes, archive_v, config.gamma)
    tuned = HistoryBank(data.num_nodes, current_v, config.gamma)
    reset = HistoryBank(data.num_nodes, archive_v, config.gamma)
    sources = interface['sources']
    catalog_size = data.num_nodes+4
    embedded = np.zeros((4, catalog_size, catalog_size))
    embedded[:, :data.num_nodes, :data.num_nodes] = sources
    audit = None
    simulator_class = MetadataSimulator
    if audit_repair:
        from .evidence import RepairAudit, TimedMetadataSimulator
        audit = RepairAudit(embedded, interface['scale'], archive_v, config.gamma, caps)
        simulator_class = TimedMetadataSimulator
    delay = (4 if data.name == 'knowair' else 12) if config.delay is None else config.delay
    refill = (.5 if data.name == 'knowair' else 1/6) if config.refill is None else config.refill
    rows, manifests, latency, replay_history = [], [], [], []
    simulator, previous, forward_bias = None, None, np.zeros_like(interface['scale'])
    orbit_replay_max_abs_difference = None
    pred_maps = {}
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if save_predictions:
            pred_maps = {m: np.lib.format.open_memmap(output/(m+'.npy'), mode='w+', dtype=np.float64,
                         shape=cache['targets'].shape) for m in methods}
    for t, graph in enumerate(_graph_stream(data, cache['origins'], config.k)):
        started = time.perf_counter()
        if simulator is None or t % config.epoch_rounds == 0:
            manifest = generate_manifest(data.num_nodes, catalog_size, min(config.unresolved,data.num_nodes),
                                         config.domain_size, config.generator_seed+t//config.epoch_rounds, epoch=t//config.epoch_rounds)
            manifests.append(manifest)
            previous_balance = simulator.balance if simulator is not None else 0.
            previous_cursor = simulator.cursor if simulator is not None else 0
            simulator = simulator_class(manifest, delay, refill, config.capacity)
            simulator.balance, simulator.cursor = previous_balance, previous_cursor
            # A fresh ambiguity manifest does not change the feature transform,
            # source operators or coordinate system. Retain sufficient history
            # when that reconstruction is selected; keep legacy resets opt-in.
            if config.reset_on_manifest_epoch:
                for memory in [bank, tuned, reset]:
                    memory.reset()
                replay_history, previous = [], None
            forward_bias.fill(0)
        old_domains = simulator.domains
        repair_error=None
        returned = simulator.before_round(t)
        # Controlled intervention: simulator supplies truth, controller sees only
        # structurally checked evidence at its scheduled pre-issuance arrival.
        if injected_answers and t in injected_answers:
            for node in injected_answers[t]:
                answer={'epoch':simulator.manifest['epoch'],'source_versions':simulator.manifest['source_versions'],
                        'node':int(node),'identity':int(simulator.manifest['truth'][node]),'round':t,'phase':'pre-issuance'}
                if simulator.accept(answer):
                    returned.append(answer)
        if audit is not None:
            audit.record(t, simulator, previous, bank, replay_history)
        if returned:
            reset.reset()
            if previous is not None and 'forward' in methods:
                try:
                    old = CatalogProblem(embedded, previous['W'], graph, interface['scale'], old_domains).support(previous['gates'])
                    new = CatalogProblem(embedded, previous['W'], graph, interface['scale'], simulator.domains).support(previous['gates'])
                    forward_bias += interface['scale']*(old.center-new.center)
                except (ValueError,FloatingPointError,RuntimeError) as error:
                    forward_bias.fill(0)
                    repair_error=f'{type(error).__name__}: {error}'
        features = interface['transform'].transform(cache['features'][t])
        if 'replay' in methods or audit is not None:
            replay_history.append(features.copy())
        for memory in [bank, tuned, reset]:
            memory.update(features)
        w, wt = bank.projected(), tuned.projected()
        base = np.asarray(cache['predictions'][t], dtype=np.float64)
        anchor = base + np.einsum('nv,jvd,j->nd', graph, w, caps)
        if not np.isfinite(anchor).all():
            raise FloatingPointError('Current anchor is nonfinite; no valid fallback forecast exists')
        predictions = {'frozen': base, 'strict': anchor,
                       'tuned': base+np.einsum('nv,jvd,j->nd', graph, wt, caps)}
        problem,problem_error = None,None
        results = {}
        if set(methods)&{'legacy','independent','product','orbit','forward','replay'}:
            problem,problem_error = _checked_problem(embedded, w, graph, interface['scale'], simulator.domains)
        for method in ['legacy','independent','product']:
            if method in methods:
                results[method] = _admit(problem,lambda:solve_baseline(problem,caps,config.epsilon,method),base.shape,4,problem_error)
                predictions[method] = anchor.copy() if results[method].status=='fallback_current_anchor' else anchor+interface['scale']*results[method].center
        if need_joint:
            if fixed_gates is None:
                result = _admit(problem,lambda:solve_joint(problem,caps,config.epsilon,config.max_iterations,config.time_limit),base.shape,4,problem_error)
            else:
                gates=np.asarray(fixed_gates[t],dtype=float)
                def fixed_result():
                    support=problem.support(gates)
                    return ControllerResult(gates,support.center,support.radius,float(gates.sum()),'fixed_future_actions',0,float(gates.sum()))
                result=_admit(problem,fixed_result,base.shape,4,problem_error)
            results['orbit'] = result
            predictions['orbit'] = anchor.copy() if result.status=='fallback_current_anchor' else anchor+interface['scale']*result.center
            forward_bias *= config.gamma
            predictions['forward'] = predictions['orbit']+(forward_bias if result.status!='fallback_current_anchor' else 0)
            if repair_error:
                predictions['forward']=anchor.copy()
            previous = {'W':w.copy(), 'gates':result.gates.copy()}
            if audit is not None:
                issued = predictions['orbit']
                issued.flags.writeable = False
                previous.update({'G':graph.copy(), 'base':base.copy(), 'issued':issued,
                                 'round':t, 'origin':int(cache['origins'][t])})
        if 'reset' in methods:
            wr = reset.projected()
            reset_problem,reset_error = _checked_problem(embedded, wr, graph, interface['scale'], simulator.domains)
            results['reset'] = _admit(reset_problem,lambda:solve_joint(reset_problem,caps,config.epsilon,config.max_iterations,config.time_limit),base.shape,4,reset_error)
            reset_anchor=base+np.einsum('nv,jvd,j->nd',graph,wr,caps)
            predictions['reset'] = reset_anchor if results['reset'].status=='fallback_current_anchor' else reset_anchor+interface['scale']*results['reset'].center
        if 'replay' in methods:
            z_replay = np.zeros_like(features)
            for h in replay_history:
                z_replay = config.gamma*z_replay+(1-config.gamma)*h
            replay_w = np.einsum('np,jpd->jnd', z_replay, archive_v)
            replay_problem,replay_error = _checked_problem(embedded, replay_w, graph, interface['scale'], simulator.domains)
            def replay_result():
                support=replay_problem.support(results['orbit'].gates)
                return ControllerResult(results['orbit'].gates,support.center,support.radius,results['orbit'].mass,'replay',0,1.)
            replay_support=_admit(replay_problem,replay_result,base.shape,4,replay_error)
            results['replay']=replay_support
            replay_anchor=base+np.einsum('nv,jvd,j->nd',graph,replay_w,caps)
            predictions['replay'] = replay_anchor if replay_support.status=='fallback_current_anchor' else replay_anchor+interface['scale']*replay_support.center
            if 'orbit' in methods:
                difference = float(np.max(np.abs(predictions['orbit']-predictions['replay'])))
                orbit_replay_max_abs_difference = max(orbit_replay_max_abs_difference or 0., difference)
        for method in methods:
            if not np.isfinite(predictions[method]).all():
                raise FloatingPointError(f'Cannot issue nonfinite {method} forecast')
            metrics[method].update(predictions[method], cache['targets'][t], cache['valid'][t])
            if capture_origin_losses:
                losses[method].update(t, predictions[method], cache['targets'][t], cache['valid'][t])
            if method in pred_maps:
                pred_maps[method][t] = predictions[method]
        # Simulator truth is consulted only after predictions are fixed, for auditing.
        row = {'round':t,'origin':int(cache['origins'][t]),'epoch':simulator.manifest['epoch'],
               'returns':len(returned),'unresolved':sum(len(d)>1 for d in simulator.domains),'repair_error':repair_error,'methods':{}}
        for method, result in results.items():
            discrepancy = None
            if not np.any(result.gates):
                discrepancy=0.
            elif method != 'reset' and problem is not None:
                truth_z = problem.coefficients(simulator.manifest['truth'])@result.gates
                discrepancy = float(np.max(np.abs(result.center-truth_z)))
            row['methods'][method] = {'mass':result.mass,'gates':result.gates.tolist(),'radius':result.radius,
                                      'discrepancy':discrepancy,'status':result.status,'upper_mass':result.upper_mass,
                                      'fallback_reason':result.details.get('reason')}
        row['requests'] = simulator.after_issuance(t)
        if audit is not None:
            audit.record(t, simulator, previous, bank, replay_history)
        if simulator.immediate_returns:
            reset.reset()
            if 'forward' in methods and need_joint and problem is not None:
                try:
                    old = problem.support(results['orbit'].gates)
                    repaired = CatalogProblem(embedded,w,graph,interface['scale'],simulator.domains).support(results['orbit'].gates)
                    forward_bias += interface['scale']*(old.center-repaired.center)
                except (ValueError,FloatingPointError,RuntimeError) as error:
                    forward_bias.fill(0)
                    row['post_issuance_repair_error']=f'{type(error).__name__}: {error}'
            row['post_issuance_returns'] = simulator.immediate_returns
        row['cached_pipeline_ms'] = 1000*(time.perf_counter()-started)
        rows.append(row)
        latency.append(row['cached_pipeline_ms'])
        if (t+1)%100 == 0:
            print(json.dumps({'stage':'evaluate','dataset':data.name,'round':t+1,'total':len(cache['origins'])}),flush=True)
    summary = {'dataset':data.name,'origins':len(rows),'config':asdict(config),
               'metrics':{m:v.result() for m,v in metrics.items()},
               'trace_payload_bytes':bank.payload_bytes,'rss_at_end_bytes':psutil.Process().memory_info().rss,
               'cached_pipeline_p50_ms':float(np.percentile(latency,50)), 'cached_pipeline_p95_ms':float(np.percentile(latency,95)),
               'timing_scope':'all requested controllers together, using cached backbone features; graph iterator/backbone excluded; NOT end-to-end paper latency',
               'predictions_saved':bool(output and save_predictions),
               'orbit_replay_max_abs_difference':orbit_replay_max_abs_difference,
               'numerical_only':True,'formal_interval_certificate':False,
               'assumptions':[(
                                  'reconstructed identity manifests with periodic ambiguity epochs and history resets'
                                  if config.reset_on_manifest_epoch else
                                  'reconstructed identity manifests with periodic ambiguity epochs; histories retained across domain refresh because sources, feature transform and coordinates remain fixed'),
                              'forward proxy midpoint offset resets at each ambiguity manifest epoch',
                              'forward is a reconstructed proxy: preserves pre-answer aggregate contribution via decaying midpoint offset; not author implementation',
                              'KnowAir weather excluded from backbone input; target and calendar retained'],
               'paper_reference_longest_lead':PAPER_REFERENCE[data.name]}
    if audit is not None:
        summary['repair_audit'] = audit.result()
        summary['timing_scope'] += '; includes additional opt-in paired repair auditing work'
    if capture_origin_losses:
        summary['origin_loss_statistics'] = {
            'storage':'per-origin per-selected-lead absolute-error sum, squared-error sum and valid-coordinate count',
            'resampling_unit':'temporal origins, pooling nodes', 'leads':list(data.leads),
            'path':str(output/'loss_statistics.npz') if output else None}
    if output:
        for arr in pred_maps.values():
            arr.flush()
        if capture_origin_losses:
            from .evidence import save_origin_losses
            save_origin_losses(output/'loss_statistics.npz', losses)
        write_json(output/'summary.json',summary)
        write_json(output/'manifests.json',manifests)
        with (output/'ledger.jsonl').open('w',encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    return summary


def run_forecasting(data, cache_path, output, config=None, grid=False, methods=None, resume=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config = config or OrbitConfig(epoch_rounds=56 if data.name=='knowair' else 288)
    caches, metadata = load_cache(cache_path)
    if metadata['dataset'] != data.name or metadata['dataset_sha256'] != data.metadata.get('sha256'):
        raise ValueError('Cache provenance does not match dataset')
    from .data import file_sha256
    signature={'config':asdict(config),'grid':bool(grid),'cache':{key:metadata[key] for key in
               ['dataset','dataset_sha256','checkpoint_sha256','backbone','feature_dim','counts','limited']},
               'implementation':{path.name:file_sha256(path) for path in Path(__file__).parent.glob('*.py')}}
    progress_path=output/'search_progress.json'
    validations=[]
    if resume and (output/'validation.json').exists():
        if not progress_path.exists() or json.loads(progress_path.read_text(encoding='utf-8'))['signature']!=signature:
            raise ValueError('Resume search provenance/configuration differs; choose a fresh output directory')
        validations=json.loads((output/'validation.json').read_text(encoding='utf-8'))
    write_json(progress_path,{'signature':signature,'completed':False})
    transform = FeatureTransform.fit(caches['train']['features'],config.feature_width)
    from .residual import residual_scale_cached
    scale = residual_scale_cached(caches['train']['targets'],caches['train']['predictions'],caches['train']['valid'])
    choices = itertools.product([4,8,16],[0,.75,.9,.95,.98]) if grid else [(config.k,config.gamma)]
    best={'orbit':None,'tuned':None}
    completed={(row['config']['k'],row['config']['gamma'],row['config']['beta']) for row in validations}
    for method in best:
        if validations:
            row=min(validations,key=lambda row:(row['validation_mae'][method],row['config']['k'],row['config']['gamma'],-row['config']['beta']))
            selected_config=OrbitConfig(**row['config'])
            interface,va,vc,saved_config=read_interface(output/(method+'_interface.npz'))
            # A crash between saving a new best and its validation record can
            # leave an ahead-of-log interface. Refit only the recorded winner.
            if asdict(saved_config)!=asdict(selected_config):
                interface=fit_interface(data,caches['train'],selected_config,transform,scale)
                interface['provenance']={key:metadata[key] for key in ['dataset','dataset_sha256','checkpoint_sha256','backbone','feature_dim','counts']}
                va,vc=projections_for_beta(interface,selected_config.beta)
                save_interface(output/(method+'_interface.npz'),interface,va,vc,selected_config)
            best[method]={'key':(row['validation_mae'][method],selected_config.k,selected_config.gamma,-selected_config.beta),
                          'config':selected_config,'interface':interface,'va':va,'vc':vc}
    for k,gamma in choices:
        betas=[1e-5,1e-4,1e-3,1e-2,.1] if grid else [config.beta]
        if all((k,gamma,beta) in completed for beta in betas):
            continue
        candidate=replace(config,k=k,gamma=gamma)
        interface=fit_interface(data,caches['train'],candidate,transform,scale)
        interface['provenance']={key:metadata[key] for key in ['dataset','dataset_sha256','checkpoint_sha256','backbone','feature_dim','counts']}
        for beta in betas:
            if (k,gamma,beta) in completed:
                continue
            candidate=replace(candidate,beta=beta)
            va,vc=projections_for_beta(interface,beta)
            result=evaluate_stream(data,caches['val'],interface,va,vc,candidate,methods=['orbit','tuned'])
            scores={m:float(scored_mae(result['metrics'][m],data.leads)) for m in best}
            record={'config':asdict(candidate),'validation_mae':scores}
            validations.append(record)
            for method in best:
                key=(scores[method],k,gamma,-beta)
                if best[method] is None or key<best[method]['key']:
                    best[method]={'key':key,'config':candidate,'interface':interface,'va':va,'vc':vc}
                    save_interface(output/(method+'_interface.npz'),interface,va,vc,candidate)
            write_json(output/'validation.json',validations)
            print(json.dumps({'stage':'validation',**record}),flush=True)
    selected=best['orbit']
    result=evaluate_stream(data,caches['test'],selected['interface'],selected['va'],selected['vc'],selected['config'],methods,output/'test')
    # Tuned-current must use its independently selected k/gamma/beta.
    tuned_best=best['tuned']
    tuned_result=evaluate_stream(data,caches['test'],tuned_best['interface'],tuned_best['va'],tuned_best['vc'],tuned_best['config'],['tuned'],output/'tuned_test')
    result['metrics']['tuned']=tuned_result['metrics']['tuned']
    if (output/'test'/'tuned.npy').exists():
        shutil.copyfile(output/'tuned_test'/'tuned.npy',output/'test'/'tuned.npy')
    result['prediction_artifacts']={method:str((output/('tuned_test' if method=='tuned' else 'test')/(method+'.npy')).resolve()) for method in result['metrics']}
    result['tuned_selected_config']=asdict(tuned_best['config'])
    result['cache_provenance']=metadata
    result['validation_grid_complete']=bool(grid)
    result['run_class']='smoke' if metadata['limited'] else ('reconstructed_full_protocol' if grid else 'fixed_config_experiment')
    result['paper_comparison_valid']=False
    result['comparison_qualification']='Reconstructed configs/manifests; full-origin grid runs may be compared descriptively, not claimed identical to author protocol'
    write_json(output/'result.json',result)
    write_json(output/'test'/'summary.json',result)
    write_json(progress_path,{'signature':signature,'completed':True,'validation_candidates':len(validations)})
    return result


def run_sweeps(data, cache_path, fitted_path, output, kind):
    caches, metadata=load_cache(cache_path)
    interface,va,vc,base=read_interface(fitted_path)
    if metadata['dataset']!=data.name or metadata['dataset_sha256']!=data.metadata['sha256']:
        raise ValueError('Sweep cache does not belong to dataset')
    if not interface['provenance'] or any(interface['provenance'].get(key)!=metadata[key] for key in ['dataset','dataset_sha256','checkpoint_sha256','feature_dim','counts']):
        raise ValueError('Fitted interface provenance differs from cache; refit with this checkpoint and feature stream')
    output=Path(output)
    results=[]
    tuned_path=Path(fitted_path).with_name('tuned_interface.npz')
    fixed_tuned=None
    if kind=='signal' and tuned_path.exists():
        tuned_interface,tuned_arc,tuned_current,tuned_config=read_interface(tuned_path)
        if tuned_interface['provenance']!=interface['provenance']:
            raise ValueError('Separately tuned reference provenance differs')
        fixed_tuned=evaluate_stream(data,caches['test'],tuned_interface,tuned_arc,tuned_current,tuned_config,['tuned'],output/'tuned_reference')
    if kind=='delay':
        settings=[('delay',d) for d in ([0,4,12] if data.name=='knowair' else [0,12,72])]
    elif kind=='allowance':
        settings=[('epsilon',x) for x in [0,.025,.05,.1,.2,.5]]
    elif kind=='budget':
        settings=[('refill',x) for x in [0,1/12,1/6,.5,1.]]
    elif kind=='signal':
        settings=[('archive_mode',x) for x in ['all-null','weak','default','stale']]
    elif kind=='discount':
        settings=[('gamma',x) for x in [0,.75,.9,.95,.98]]
    elif kind=='dimension':
        settings=[('feature_width',x) for x in [8,16,32,64]]
    else:
        raise ValueError('kind must be delay/allowance/budget/signal/discount/dimension')
    for field,value in settings:
        config=replace(base,**{field:value})
        if field=='archive_mode' and value=='stale':
            config=replace(config,unresolved=0)
        active,arc,cur=interface,va,vc
        if field in ['gamma','feature_width'] or (field=='archive_mode' and value not in ['all-null','default']):
            active=fit_interface(data,caches['train'],config)
            arc,cur=projections_for_beta(active,config.beta)
        elif field=='archive_mode' and value=='all-null':
            active=dict(interface)
            active['sources']=np.zeros_like(interface['sources'])
        methods=['frozen','tuned','orbit','forward','reset','replay'] if kind=='delay' else ['frozen','tuned','orbit']
        result=evaluate_stream(data,caches['test'],active,arc,cur,config,methods,output/(field+'_'+str(value)))
        if fixed_tuned is not None:
            result['metrics']['tuned']=fixed_tuned['metrics']['tuned']
            shutil.copyfile(output/'tuned_reference'/'tuned.npy',output/(field+'_'+str(value))/'tuned.npy')
            result['current_comparator']='independently validation-selected, fixed across archival conditions'
        else:
            result['current_comparator']='matched current refit at the supplied/sweep configuration; not separately validation-selected'
        result['run_class']='smoke' if metadata['limited'] else 'reconstructed_sensitivity'
        write_json(output/(field+'_'+str(value))/'summary.json',result)
        results.append(result)
        write_json(output/'sweep.json',results)
    return results


def run_metadata_intervention(data,cache_path,fitted_path,output,arrival=12,batch_size=4):
    """Reconstructed paired target-error intervention, with a fixed query stream.

    Both arms have zero autonomous queries and one long epoch. The treatment
    receives a truthful batch before `arrival`; fixed-action gates are replayed
    from control, while policy adaptation recomputes gates. This supplies a
    disclosed experiment, not the absent authors' intervention manifests.
    """
    caches,metadata=load_cache(cache_path)
    interface,va,vc,base=read_interface(fitted_path)
    if metadata['dataset']!=data.name or metadata['dataset_sha256']!=data.metadata['sha256']:
        raise ValueError('Intervention cache does not belong to supplied dataset')
    if any(interface['provenance'].get(key)!=metadata[key] for key in ['dataset_sha256','checkpoint_sha256','counts']):
        raise ValueError('Interface/cache provenance mismatch')
    length=len(caches['test']['origins'])
    if not 0<arrival<length:
        raise ValueError('Arrival must leave both pre- and post-arrival forecasts')
    if not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('Batch size must be a positive integer')
    config=replace(base,refill=0,epoch_rounds=length+1)
    output=Path(output)
    control=evaluate_stream(data,caches['test'],interface,va,vc,config,['orbit'],output/'control')
    manifest=json.loads((output/'control'/'manifests.json').read_text(encoding='utf-8'))[0]
    nodes=manifest['unresolved_nodes'][:batch_size]
    ledger=[json.loads(row) for row in (output/'control'/'ledger.jsonl').read_text().splitlines()]
    gates=[row['methods']['orbit']['gates'] for row in ledger]
    fixed=evaluate_stream(data,caches['test'],interface,va,vc,config,['orbit'],output/'fixed_actions',
                          injected_answers={arrival:nodes},fixed_gates=gates)
    adaptive=evaluate_stream(data,caches['test'],interface,va,vc,config,['orbit'],output/'adaptive',
                             injected_answers={arrival:nodes})
    predictions={arm:np.load(output/arm/'orbit.npy',mmap_mode='r') for arm in ['control','fixed_actions','adaptive']}
    rows={}
    for arm in ['fixed_actions','adaptive']:
        baseline=MetricAccumulator(data.leads)
        treatment=MetricAccumulator(data.leads)
        negative=0
        for t in range(arrival,length):
            baseline.update(predictions['control'][t],caches['test']['targets'][t],caches['test']['valid'][t])
            treatment.update(predictions[arm][t],caches['test']['targets'][t],caches['test']['valid'][t])
            leads=np.asarray(data.leads)-1
            mask=caches['test']['valid'][t][:,leads]
            if mask.any():
                old=np.abs(predictions['control'][t][:,leads]-caches['test']['targets'][t][:,leads])[mask].mean()
                new=np.abs(predictions[arm][t][:,leads]-caches['test']['targets'][t][:,leads])[mask].mean()
                negative+=int(new>old)
        rows[arm]={'post_arrival_mae_gain':scored_mae(baseline.result(),data.leads)-scored_mae(treatment.result(),data.leads),
                   'negative_gain_rounds':negative,'pre_arrival_max_difference':float(np.max(np.abs(predictions[arm][:arrival]-predictions['control'][:arrival])))}
    report={'dataset':data.name,'arrival':arrival,'batch_size':len(nodes),'requested_batch_size':batch_size,'rounds':length,'comparisons':rows,
            'assumptions':'reconstructed fixed zero-query stream; one truthful batch; one epoch; only gates adapt',
            'run_class':'smoke' if metadata['limited'] else 'reconstructed_intervention'}
    write_json(output/'intervention.json',report)
    return report
