"""Storage-bounded, resumable execution of the three agreed server experiments."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import shutil
import time
from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np

from .data import file_sha256, load_dataset
from .experiments import (OrbitConfig, PAPER_REFERENCE, evaluate_stream, fit_interface,
                          projections_for_beta, read_interface, save_interface,
                          scored_mae, write_json)
from .residual import FeatureTransform, residual_scale_cached
from .storage import ensure_storage_budget, estimate_cache_bytes
from .training import extract_cache, load_checkpoint


def candidate_grid():
    return list(itertools.product([4, 8, 16], [0, .75, .9, .95, .98],
                                  [1e-5, 1e-4, 1e-3, 1e-2, .1]))


def select_validation_winner(rows, method):
    return min(rows, key=lambda row: (row['validation_mae'][method], row['config']['k'],
                                     row['config']['gamma'], -row['config']['beta']))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def remove_owned_cache(cache, root):
    cache, root = Path(cache).resolve(), Path(root).resolve()
    if cache == root or root not in cache.parents:
        raise ValueError('Disposable cache must be strictly within its run root')
    if not cache.exists():
        return
    marker = cache/'.orbit-disposable-cache'
    if not marker.is_file() or marker.read_text(encoding='utf-8') != str(root):
        raise ValueError('Refusing cleanup without matching cache owner marker')
    shutil.rmtree(cache)


def read_matching_record(path, signature):
    path = Path(path)
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('signature') != signature:
        raise ValueError(f'Resume provenance mismatch: {path}')
    for name, expected in result.get('artifacts', {}).items():
        artifact = path.parent/name
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise ValueError(f'Artifact digest mismatch: {artifact}')
    return result


def validate_training(training, payload, dataset_hash):
    config = payload['run_config']
    expected = dict(backbone='stid',seed=0,epochs_requested=100,batch_size=64,
                    patience=100,limited=False)
    if (training.get('completed') is not True or training.get('epochs_completed') != 100
            or training.get('config') != config or payload.get('backbone') != 'stid'
            or payload.get('dataset_sha256') != dataset_hash
            or any(config.get(k) != v for k,v in expected.items())
            or config.get('model_config',{}).get('identity_free') is not False):
        raise ValueError('Expected complete matching STID seed-0, batch-64, 100-epoch training')


def validate_phase_arrays(data, arrays, metadata):
    for split, cache in arrays.items():
        origins = data.origins(split)
        count = len(origins)
        output_shape = (count,data.num_nodes,data.output_steps)
        expected = dict(origins=((count,),np.dtype('int64')),
                        features=((count,data.num_nodes,metadata['feature_dim']),np.dtype('float32')),
                        predictions=(output_shape,np.dtype('float32')),
                        targets=(output_shape,np.dtype('float32')),
                        valid=(output_shape,np.dtype('bool')))
        if metadata['counts'].get(split) != count or not np.array_equal(cache['origins'],origins):
            raise ValueError('Cache origins do not cover the complete declared split')
        if any(cache[key].shape != shape or cache[key].dtype != dtype
               for key,(shape,dtype) in expected.items()):
            raise ValueError('Cache array shape/dtype differs from the declared feature stream')


def _close_cache(caches):
    for cache in caches.values():
        for value in cache.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
    caches.clear()
    gc.collect()


def _load_phase(path, splits):
    path = Path(path)
    metadata = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if not metadata.get('completed') or metadata.get('requested_splits') != list(splits):
        raise ValueError('Incomplete or mismatched phase cache')
    arrays = {s: {key: np.load(path/s/(key+'.npy'), mmap_mode='r')
                   for key in ['origins','features','predictions','targets','valid']}
              for s in splits}
    return arrays, metadata


def _phase_cache(data, checkpoint, path, splits, run_root, budget_root, device):
    path = Path(path)
    if (path/'metadata.json').exists():
        arrays = {}
        try:
            arrays, metadata = _load_phase(path, splits)
            if (metadata['checkpoint_sha256'] != file_sha256(checkpoint)
                    or metadata['dataset_sha256'] != data.metadata['sha256']
                    or metadata['feature_dtype'] != 'float32' or metadata['limited']):
                raise ValueError('Cache provenance mismatch')
            validate_phase_arrays(data,arrays,metadata)
            return arrays, metadata
        except (ValueError, KeyError, OSError):
            _close_cache(arrays)
            remove_owned_cache(path, run_root)
    elif path.exists():
        remove_owned_cache(path, run_root)
    model, _, _ = load_checkpoint(checkpoint, 'cpu')
    needed = estimate_cache_bytes(data, model.feature_dim, splits=splits, feature_dtype='float32')
    del model
    # Reserve source-statistics, manifests, losses and logs in addition to cache.
    plan = ensure_storage_budget(budget_root, needed+250_000_000, budget_bytes=9_950_000_000)
    print(json.dumps({'stage':'storage_preflight', **plan}), flush=True)
    path.mkdir(parents=True)
    (path/'.orbit-disposable-cache').write_text(str(Path(run_root).resolve()), encoding='utf-8')
    arrays = extract_cache(data, checkpoint, path, 128, device,
                           splits=splits, feature_dtype='float32')
    metadata = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    return arrays, metadata


def _save_statistics(path, interface):
    transform = interface['transform']
    np.savez(path, mean=transform.mean, std=transform.std, directions=transform.directions,
             scale=interface['scale'], sources=interface['sources'],
             archive_gram=interface['archive_statistics'][0],
             archive_rhs=interface['archive_statistics'][1],
             archive_count=interface['archive_statistics'][2],
             current_gram=interface['current_statistics'][0],
             current_rhs=interface['current_statistics'][1],
             current_count=interface['current_statistics'][2],
             feature_width=interface['feature_width'],
             provenance=json.dumps(interface['provenance']))


def _read_statistics(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(transform=FeatureTransform(values['mean'], values['std'], values['directions']),
                    scale=values['scale'], sources=values['sources'],
                    archive_statistics=tuple(values['archive_'+k] for k in ['gram','rhs','count']),
                    current_statistics=tuple(values['current_'+k] for k in ['gram','rhs','count']),
                    feature_width=int(values['feature_width']),
                    provenance=json.loads(str(values['provenance'])))


def _statistics_name(k, gamma):
    return f'k{k}_gamma{gamma:g}.npz'


def prune_statistics(out, signature):
    """Remove verified non-winning statistics after full selection is durable.

    Unknown files and archives without a matching record remain untouched. All
    planned pairs are verified before the first deletion; no directory traversal
    or recursive deletion is used. Winner statistics and fitted interfaces stay.
    """
    out = Path(out).resolve()
    selection = read_matching_record(out/'selection.json', signature)
    if (not selection or selection.get('completed') is not True
            or selection.get('full_paper_grid') is not True
            or selection.get('candidates') != 75
            or set(selection.get('winners', {})) != {'orbit', 'tuned'}):
        raise ValueError('Statistics pruning requires completed full-grid selection')
    pairs = sorted({(k, gamma) for k, gamma, _ in candidate_grid()})
    winner_pairs = {(row['config']['k'], row['config']['gamma'])
                    for row in selection['winners'].values()}
    if not winner_pairs.issubset(set(pairs)):
        raise ValueError('Selected statistics are outside the declared grid')
    statistics = out/'statistics'
    if statistics.resolve() != statistics:
        raise ValueError('Refusing to prune a linked statistics directory')
    pending = []
    for k, gamma in pairs:
        if (k, gamma) in winner_pairs:
            continue
        archive = statistics/_statistics_name(k, gamma)
        record = archive.with_suffix('.json')
        if not record.exists():
            continue
        if (record.is_symlink() or archive.is_symlink()
                or record.resolve().parent != statistics or archive.resolve().parent != statistics):
            raise ValueError('Refusing to prune linked statistics files')
        saved = read_matching_record(record, signature)
        if saved.get('completed') is not True or set(saved.get('artifacts', {})) != {archive.name}:
            raise ValueError('Statistics pair lacks a complete matching ownership record')
        pending.append((record, archive, record.stat().st_size+archive.stat().st_size))
    for record, archive, _ in pending:
        # A crash between these unlinks leaves an unrecorded archive, which a
        # future pruning pass conservatively preserves instead of trusting it.
        record.unlink()
        archive.unlink()
    return {'removed_pairs': len(pending), 'removed_bytes': sum(size for _, _, size in pending),
            'retained_winner_statistics': [_statistics_name(k, gamma) for k, gamma in sorted(winner_pairs)]}


_WORKER_DATA = _WORKER_CACHE = None


def _validation_init(data_root, name, cache_path):
    global _WORKER_DATA, _WORKER_CACHE
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    # Keep the limiter object alive for the worker lifetime.
    global _WORKER_LIMITER
    _WORKER_LIMITER = threadpool_limits(limits=1)
    _WORKER_DATA = load_dataset(data_root, name)
    _WORKER_CACHE = _load_phase(cache_path, ('train','val'))[0]['val']


def _validation_candidate(stats_path, config_dict):
    config = OrbitConfig(**config_dict)
    interface = _read_statistics(stats_path)
    va, vc = projections_for_beta(interface, config.beta)
    started = time.perf_counter()
    result = evaluate_stream(_WORKER_DATA, _WORKER_CACHE, interface, va, vc, config,
                             ['orbit','tuned'])
    return {'config':config_dict,
            'validation_mae':{m:scored_mae(result['metrics'][m], _WORKER_DATA.leads)
                              for m in ['orbit','tuned']},
            'seconds':time.perf_counter()-started,
            'metrics':result['metrics']}


def _search(data, checkpoint, out, run_root, budget_root, data_root, base, signature,
            device, workers, update):
    selection = out/'selection.json'
    existing = read_matching_record(selection, signature)
    cache_path = out/'scratch'/'train-val'
    if existing and existing.get('completed'):
        if cache_path.exists():
            remove_owned_cache(cache_path, run_root)
        return existing
    progress_path = out/'validation.json'
    previous = read_matching_record(progress_path, signature)
    rows = previous.get('rows', []) if previous else []
    completed = {(r['config']['k'],r['config']['gamma'],r['config']['beta']) for r in rows}
    if len(completed) != len(rows) or not completed.issubset(set(candidate_grid())):
        raise ValueError('Validation progress contains duplicate or undeclared candidates')
    update('extract_train_validation', data.name, validation_candidates=len(rows))
    caches, metadata = _phase_cache(data, checkpoint, cache_path, ('train','val'),
                                     run_root, budget_root, device)
    write_json(out/'train-validation-cache-manifest.json', metadata)
    provenance = {k:metadata[k] for k in ['dataset','dataset_sha256','checkpoint_sha256','backbone','feature_dim']}
    provenance['counts'] = {s:len(data.origins(s)) for s in ['train','val','test']}
    transform = FeatureTransform.fit(caches['train']['features'], base.feature_width)
    scale = residual_scale_cached(caches['train']['targets'], caches['train']['predictions'],
                                  caches['train']['valid'])
    stats_dir = out/'statistics'
    stats_dir.mkdir(exist_ok=True)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_validation_init, initargs=(str(data_root),data.name,str(cache_path))) as pool:
            for k,gamma in itertools.product([4,8,16],[0,.75,.9,.95,.98]):
                stats = stats_dir/_statistics_name(k,gamma)
                stat_record = stats.with_suffix('.json')
                candidate = replace(base, k=k, gamma=gamma)
                if not read_matching_record(stat_record, signature):
                    update('fit_residual', data.name, k=k, gamma=gamma,
                           validation_candidates=len(rows))
                    interface = fit_interface(data, caches['train'], candidate, transform, scale)
                    interface['provenance'] = provenance
                    _save_statistics(stats, interface)
                    atomic_json(stat_record, {'signature':signature, 'completed':True,
                                              'artifacts':{stats.name:file_sha256(stats)}})
                pending = [replace(candidate,beta=beta) for beta in [1e-5,1e-4,1e-3,1e-2,.1]
                           if (k,gamma,beta) not in completed]
                update('validation', data.name, k=k, gamma=gamma, validation_candidates=len(rows))
                futures = [pool.submit(_validation_candidate, str(stats), asdict(cfg)) for cfg in pending]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    completed.add((row['config']['k'],row['config']['gamma'],row['config']['beta']))
                    atomic_json(progress_path, {'signature':signature,'completed':False,'rows':rows})
                    print(json.dumps({'stage':'candidate_complete', 'dataset':data.name,
                                      'completed':len(rows),**row}), flush=True)
                    update('validation', data.name, validation_candidates=len(rows),
                           latest_config=row['config'], latest_validation_mae=row['validation_mae'])
        winners = {}
        for method in ['orbit','tuned']:
            row = select_validation_winner(rows, method)
            config = OrbitConfig(**row['config'])
            interface = _read_statistics(stats_dir/_statistics_name(config.k,config.gamma))
            va, vc = projections_for_beta(interface,config.beta)
            save_interface(out/(method+'_interface.npz'),interface,va,vc,config)
            winners[method] = row
        atomic_json(progress_path, {'signature':signature,'completed':True,'rows':rows})
        result = {'signature':signature,'completed':True,'candidates':len(rows),
                  'full_paper_grid':len(rows)==75,'winners':winners,
                  'artifacts':{name:file_sha256(out/name) for name in
                               ['orbit_interface.npz','tuned_interface.npz','validation.json']}}
        atomic_json(selection,result)
    finally:
        _close_cache(caches)
    remove_owned_cache(cache_path,run_root)
    return result


def _summarize_ledger(path, epsilon):
    sums = {}
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            for method, measured in row['methods'].items():
                summary = sums.setdefault(method,dict(rounds=0,mass_sum=0.,max_radius=0.,
                   max_discrepancy=0.,radius_violations=0,discrepancy_violations=0,fallbacks=0))
                summary['rounds'] += 1
                summary['mass_sum'] += measured['mass']
                summary['max_radius'] = max(summary['max_radius'],measured['radius'])
                discrepancy = measured['discrepancy']
                if discrepancy is not None:
                    summary['max_discrepancy'] = max(summary['max_discrepancy'],discrepancy)
                    summary['discrepancy_violations'] += int(discrepancy > epsilon+1e-7)
                summary['radius_violations'] += int(measured['radius'] > epsilon+1e-7)
                summary['fallbacks'] += int(measured['status']=='fallback_current_anchor')
    for summary in sums.values():
        summary['mean_mass'] = summary.pop('mass_sum')/summary['rounds']
    return sums


def _evaluate_main(data, caches, out, selection, signature):
    from .evidence import load_origin_losses, paired_block_bootstrap
    result_path = out/'main_result.json'
    old = read_matching_record(result_path,signature)
    if old and old.get('completed'):
        return old
    interface,va,vc,config = read_interface(out/'orbit_interface.npz')
    main = evaluate_stream(data,caches['test'],interface,va,vc,config,
                ['frozen','strict','orbit','independent','product'],out/'main',
                save_predictions=False,capture_origin_losses=True)
    ti,ta,tc,tconf = read_interface(out/'tuned_interface.npz')
    tuned = evaluate_stream(data,caches['test'],ti,ta,tc,tconf,['tuned'],out/'current',
                           save_predictions=False,capture_origin_losses=True)
    main['metrics']['tuned'] = tuned['metrics']['tuned']
    base = load_origin_losses(out/'current'/'loss_statistics.npz','tuned')
    uncertainty, sensitivity = {}, {}
    for method in ['frozen','strict','orbit','independent','product']:
        losses = load_origin_losses(out/'main'/'loss_statistics.npz',method)
        uncertainty[method+'_minus_tuned'] = paired_block_bootstrap(losses,base,data.name,data.leads)
        sensitivity[method+'_minus_tuned'] = paired_block_bootstrap(losses,base,data.name,data.leads,
                                  block_length=56 if data.name=='knowair' else 2016)
    mechanism = _summarize_ledger(out/'main'/'ledger.jsonl',config.epsilon)
    result = dict(signature=signature, completed=True, run_class='single_seed_reconstructed_full_grid',
                  dataset=data.name, metrics=main['metrics'], selected_configs={m:v['config'] for m,v in selection['winners'].items()},
                  validation_candidates=selection['candidates'], bootstrap=uncertainty,
                  bootstrap_seven_day_sensitivity=sensitivity,
                  controller_ablation=mechanism, origins=main['origins'],
                  paper_reference_longest_lead=PAPER_REFERENCE[data.name],
                  paper_comparison_valid=False,
                  qualifications=['One STID training seed; original author configuration and manifests unavailable',
                                  'All 75 candidates use complete validation; test does not select configurations',
                                  'Shared fixed-source histories retained across domain refreshes; identity state transport',
                                  'Independent and product controllers share ORBIT-selected parameters for controlled ablation; they are not separately tuned',
                                  'Bootstrap uses temporal blocks with all nodes kept together; not multi-seed variability',
                                  'Increased admitted mass does not guarantee improved forecasting'],
                  artifacts={str(p.relative_to(out)):file_sha256(p) for p in
                    [out/'main'/'loss_statistics.npz',out/'current'/'loss_statistics.npz',
                     out/'main'/'ledger.jsonl',out/'main'/'manifests.json',
                     out/'current'/'ledger.jsonl',out/'current'/'manifests.json']})
    atomic_json(result_path,result)
    return result


def _repair_windows(data,caches,out,signature):
    path = out/'repair_result.json'
    old = read_matching_record(path,signature)
    if old and old.get('completed'):
        return old
    interface,va,vc,config = read_interface(out/'orbit_interface.npz')
    length = min(256,len(caches['test']['origins']))
    total = len(caches['test']['origins'])
    starts = sorted(set([0,(total-length)//2,total-length]))
    results, artifacts = [], {}
    for delay in [0,4,12]:
        for start in starts:
            directory = out/'repair'/f'delay{delay}_start{start}'
            window = {k:v[start:start+length] for k,v in caches['test'].items()}
            result = evaluate_stream(data,window,interface,va,vc,replace(config,delay=delay),
                         ['orbit','replay','forward','reset'],directory,
                         save_predictions=False,audit_repair=True,capture_origin_losses=True)
            result['window_start_index'],result['window_length'] = start,length
            audit = result['repair_audit']
            result['numerical_equivalence_passed'] = bool(
                result['orbit_replay_max_abs_difference'] <= 1e-9 and audit['samples'] > 0
                and audit['max_abs_readout_difference'] <= 1e-9 and audit['forecast_rewrites'] == 0)
            results.append(result)
            artifacts.update({str(p.relative_to(out)):file_sha256(p) for p in
                              [directory/name for name in ['summary.json','ledger.jsonl',
                                                           'manifests.json','loss_statistics.npz']]})
    result = dict(signature=signature,completed=True,dataset=data.name,results=results,
                  artifacts=artifacts,
                  all_numerical_equivalence_passed=all(r['numerical_equivalence_passed'] for r in results),
                  scope='Predeclared early/middle/late 256-origin windows; each window is a separate zero-initialized stream',
                  timing_qualification='Actual paired repair-only measurements on a shared server; no exclusive-host speed claim',
                  forward_qualification='Forward-only is a reconstructed proxy, not an author-supplied implementation')
    atomic_json(path,result)
    return result


def write_report(root):
    root = Path(root)
    rows,results = [],{}
    for name in ['metr-la','pems-bay','knowair']:
        path = root/name/'main_result.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding='utf-8'))
        results[name] = result
        for lead in result['metrics']['orbit']:
            for method in ['frozen','strict','tuned','independent','product','orbit']:
                metric=result['metrics'][method][lead]
                rows.append(f"| {name} | {lead} | {method} | {metric['mae']:.6f} | {metric['rmse']:.6f} |")
    text = ['# ORBIT server experiments','',
       'Only completed measurements appear below. Single STID seed; reconstructed training and metadata; full validation grid. No claim of identical author protocol.',
       '', '| Dataset | Lead | Method | MAE | RMSE |', '|---|---:|---|---:|---:|',*rows,'',
       'Detailed paired temporal-block intervals and controller constraint audits are in each dataset/main_result.json.',
       'KnowAir repair_result.json contains the predeclared delayed-repair windows once completed.']
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    atomic_json(root/'results.json',results)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data')
    parser.add_argument('--out',default='runs/server-three-20261013')
    parser.add_argument('--datasets',nargs='+',default=['knowair','metr-la','pems-bay'],
                        choices=['knowair','metr-la','pems-bay'])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    if args.workers<1 or args.threads<1:
        parser.error('positive workers and threads required')
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    root=Path(args.out).resolve()
    project=Path(__file__).resolve().parent.parent
    if project not in root.parents:
        parser.error('--out must be strictly within the budgeted project directory')
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'experiment-worker.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        try:
            process=psutil.Process(old['pid'])
            if abs(process.create_time()-old['birth'])<.01 and process.is_running():
                raise RuntimeError('An experiment worker is already running')
        except psutil.NoSuchProcess:
            pass
    atomic_json(lock,dict(pid=os.getpid(),birth=psutil.Process().create_time(),started=time.time()))
    torch.set_num_threads(args.threads)
    implementation={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}
    started=time.time()

    def update(phase,dataset=None,**more):
        if (root/'STOP').exists():
            raise KeyboardInterrupt('STOP requested')
        if any(file_sha256(Path(__file__).parent/name)!=digest for name,digest in implementation.items()):
            raise RuntimeError('Implementation changed while experiment was running')
        usage=ensure_storage_budget(project,0,budget_bytes=9_950_000_000)
        atomic_json(root/'status.json',dict(phase=phase,dataset=dataset,completed=False,
                   elapsed_seconds=time.time()-started,updated=time.time(),storage=usage,**more))

    try:
        with threadpool_limits(limits=args.threads):
            for name in args.datasets:
                data=load_dataset(args.data,name)
                out=root/name
                checkpoint=out/'backbone'/'checkpoint.pt'
                training_path=out/'backbone'/'training.json'
                if not training_path.exists() or not json.loads(training_path.read_text()).get('completed'):
                    raise RuntimeError(f'Training must complete before experiments: {name}')
                model,_,payload=load_checkpoint(checkpoint,'cpu')
                del model
                validate_training(json.loads(training_path.read_text()),payload,data.metadata['sha256'])
                base=OrbitConfig(epoch_rounds=56 if name=='knowair' else 288,
                                 reset_on_manifest_epoch=False)
                signature=dict(dataset_sha256=data.metadata['sha256'],checkpoint_sha256=file_sha256(checkpoint),
                               code=implementation,base_config=asdict(base),grid=candidate_grid(),
                               protocol='fixed-source-identity-transport-v1',cache_dtype='float32')
                # JSON canonicalization keeps tuple/list round trips comparable.
                signature=json.loads(json.dumps(signature))
                previous_selection=read_matching_record(out/'selection.json',signature)
                previous_main=read_matching_record(out/'main_result.json',signature)
                previous_repair=read_matching_record(out/'repair_result.json',signature) if name=='knowair' else {'completed':True}
                if all(item and item.get('completed') for item in
                       [previous_selection,previous_main,previous_repair]):
                    for phase in ['train-val','test']:
                        if (out/'scratch'/phase).exists():
                            remove_owned_cache(out/'scratch'/phase,root)
                    write_report(root)
                    update('dataset_already_completed',name,validation_candidates=75)
                    continue
                selection=_search(data,checkpoint,out,root,project,Path(args.data).resolve(),base,
                                  signature,args.device,args.workers,update)
                pruned=prune_statistics(out,signature)
                print(json.dumps({'stage':'prune_statistics','dataset':name,**pruned}),flush=True)
                update('extract_test',name,validation_candidates=75)
                cache_path=out/'scratch'/'test'
                caches,metadata=_phase_cache(data,checkpoint,cache_path,('test',),root,project,args.device)
                write_json(out/'test-cache-manifest.json',metadata)
                try:
                    update('main_and_controller_ablation',name,validation_candidates=75)
                    _evaluate_main(data,caches,out,selection,signature)
                    write_report(root)
                    if name=='knowair':
                        update('delayed_repair',name,validation_candidates=75)
                        _repair_windows(data,caches,out,signature)
                finally:
                    _close_cache(caches)
                remove_owned_cache(cache_path,root)
                write_report(root)
                update('dataset_completed',name,validation_candidates=75)
        usage=ensure_storage_budget(project,0)
        atomic_json(root/'status.json',dict(phase='completed',completed=True,
                    elapsed_seconds=time.time()-started,updated=time.time(),storage=usage))
    except BaseException as error:
        status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        status.update(phase='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',completed=False,
                      error=f'{type(error).__name__}: {error}',updated=time.time())
        atomic_json(root/'status.json',status)
        raise


if __name__=='__main__':
    main()
