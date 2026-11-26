"""Command line entry points; every measured experiment writes its provenance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='ORBIT paper reconstruction and scientific experiments')
    parser.add_argument('--threads', type=int, default=8)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit', help='Inspect original datasets without modifying them')
    audit.add_argument('--data', default='data')
    audit.add_argument('--out', default='runs/data-audit.json')
    sci = sub.add_parser('scientific', help='Exactness, geometry, replay, faults and scale checks')
    sci.add_argument('--out',default='runs/scientific')
    sci.add_argument('--quick',action='store_true')
    sci.add_argument('--scale',action='store_true',help='Also execute full graph sizes in isolated deadline-limited processes')
    matrix = sub.add_parser('matrix', help='Write the full experiment coverage inventory')
    matrix.add_argument('--out',default='runs/experiment-matrix.json')
    train = sub.add_parser('train',help='Train STID or STAEformer using train/validation only')
    cache = sub.add_parser('cache',help='Freeze checkpoint and materialize nodewise feature stream')
    forecast = sub.add_parser('forecast',help='Fit residuals, validate parameters and evaluate controllers')
    sweep = sub.add_parser('sweep',help='Run paper delay/signal/sensitivity experiment families')
    intervention = sub.add_parser('intervention',help='Paired fixed-action/adaptive metadata experiment')
    profile = sub.add_parser('profile',help='Isolated batch-one live-backbone resource measurement')
    external = sub.add_parser('external',help='Native external baselines with causal delayed labels')
    full = sub.add_parser('reproduce',help='Train, cache and execute reconstructed protocol for selected datasets/backbones/seeds')
    for p in [train,cache,forecast,sweep,intervention,profile,external]:
        p.add_argument('--data',default='data')
        p.add_argument('--dataset',choices=['metr-la','pems-bay','knowair'],required=True)
        p.add_argument('--out',required=True)
    for p in [train,full]:
        p.add_argument('--backbone',choices=['stid','staeformer'],default='stid')
        p.add_argument('--epochs',type=int,default=100)
        p.add_argument('--batch-size',type=int,default=32)
        p.add_argument('--device',default='auto')
        p.add_argument('--identity-free',action='store_true')
        p.add_argument('--patience',type=int,default=20)
    train.add_argument('--seed',type=int,default=0)
    train.add_argument('--max-train-origins',type=int)
    train.add_argument('--max-val-origins',type=int)
    cache.add_argument('--checkpoint',required=True)
    cache.add_argument('--batch-size',type=int,default=32)
    cache.add_argument('--device',default='auto')
    cache.add_argument('--max-origins',type=int)
    forecast.add_argument('--cache',required=True)
    forecast.add_argument('--grid',action='store_true',help='Full paper 75-combination validation; can take days on a laptop')
    forecast.add_argument('--config',help='JSON OrbitConfig overrides; unknown keys are rejected')
    forecast.add_argument('--methods',nargs='+',default=None)
    forecast.add_argument('--resume',action='store_true',help='Reuse completed validation candidates with identical code/config/cache provenance')
    sweep.add_argument('--cache',required=True)
    sweep.add_argument('--interface',required=True)
    sweep.add_argument('--kind',required=True,choices=['delay','signal','allowance','budget','discount','dimension'])
    intervention.add_argument('--cache',required=True)
    intervention.add_argument('--interface',required=True)
    intervention.add_argument('--arrival',type=int,default=12)
    intervention.add_argument('--batch-size',type=int,default=4)
    profile.add_argument('--checkpoint',required=True)
    profile.add_argument('--interface',required=True)
    profile.add_argument('--method',default='orbit',choices=['orbit','legacy','independent','product','strict','tuned','frozen'])
    profile.add_argument('--rounds',type=int,default=100)
    profile.add_argument('--warmup',type=int,default=10)
    profile.add_argument('--deadline-ms',type=float,default=1000)
    profile.add_argument('--device',default='cpu')
    external.add_argument('--method',required=True,choices=['fsnet','onenet','proceed','dsof','odgnet','adapt-z','padre'])
    external.add_argument('--checkpoint',help='Historically pretrained STID for PROCEED/ADAPT-Z/PADRE')
    external.add_argument('--device',default='auto')
    external.add_argument('--seed',type=int,default=0)
    external.add_argument('--epochs',type=int,default=20)
    external.add_argument('--batch-size',type=int,default=32)
    external.add_argument('--patience',type=int,default=5)
    external.add_argument('--max-train-origins',type=int)
    external.add_argument('--max-val-origins',type=int)
    external.add_argument('--max-test-origins',type=int)
    external.add_argument('--configs',help='JSON list of constructor configurations; selected on causal validation only')
    full.add_argument('--data',default='data')
    full.add_argument('--datasets',nargs='+',default=['metr-la','pems-bay','knowair'],choices=['metr-la','pems-bay','knowair'])
    full.add_argument('--seeds',type=int,nargs='+',default=[0])
    full.add_argument('--out',default='runs/full')
    full.add_argument('--smoke',action='store_true',help='1 epoch and limited origins, never a full reproduction claim')
    full.add_argument('--resume',action='store_true',help='Reuse complete locally generated checkpoints/cache with matching provenance')
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    from threadpoolctl import threadpool_limits
    import torch
    from .experiments import write_json, experiment_matrix, OrbitConfig, run_forecasting, run_sweeps
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        if args.command=='scientific':
            from .scientific import run_scientific
            result=run_scientific(args.out,args.quick,args.scale)
            print(json.dumps({'artifact':str(Path(args.out).resolve()),'status':'scientific_suite_finished'}))
        elif args.command=='matrix':
            write_json(args.out,experiment_matrix())
            print(args.out)
        elif args.command=='audit':
            from .data import load_dataset
            result={name:load_dataset(args.data,name).metadata for name in ['metr-la','pems-bay','knowair']}
            write_json(args.out,result)
            print(json.dumps({k:{'shape':v['shape'],'splits':v['splits'],'invalid_count':v['invalid_count']} for k,v in result.items()}))
        elif args.command=='reproduce':
            _reproduce(args)
        else:
            from .data import load_dataset
            data=load_dataset(args.data,args.dataset)
            if args.command=='train':
                from .training import train_backbone
                path=train_backbone(data,args.out,args.backbone,args.seed,args.epochs,args.batch_size,args.device,
                                    args.max_train_origins,args.max_val_origins,args.identity_free,patience=args.patience)
                print(str(path))
            elif args.command=='cache':
                from .training import extract_cache
                extract_cache(data,args.checkpoint,args.out,args.batch_size,args.device,args.max_origins)
                print(str(Path(args.out).resolve()))
            elif args.command=='forecast':
                overrides=json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
                overrides.setdefault('epoch_rounds',56 if data.name=='knowair' else 288)
                result=run_forecasting(data,args.cache,args.out,OrbitConfig(**overrides),args.grid,args.methods,resume=args.resume)
                print(json.dumps({'result':str(Path(args.out)/'result.json'),'run_class':result['run_class'],'metrics':result['metrics']}))
            elif args.command=='sweep':
                run_sweeps(data,args.cache,args.interface,args.out,args.kind)
                print(str(Path(args.out)/'sweep.json'))
            elif args.command=='intervention':
                from .experiments import run_metadata_intervention
                run_metadata_intervention(data,args.cache,args.interface,args.out,args.arrival,args.batch_size)
                print(str(Path(args.out)/'intervention.json'))
            elif args.command=='profile':
                from .profiling import profile_stream
                interface=Path(args.interface)
                if args.method=='tuned' and interface.name=='orbit_interface.npz':
                    interface=interface.with_name('tuned_interface.npz')
                profile_stream(data,args.checkpoint,interface,args.out,args.method,args.rounds,args.warmup,args.deadline_ms,args.device)
                print(str(Path(args.out)/'profile.json'))
            elif args.command=='external':
                from .external.runner import run_external
                configs=json.loads(Path(args.configs).read_text(encoding='utf-8')) if args.configs else None
                result=run_external(data,args.out,args.method,args.checkpoint,args.device,args.seed,args.epochs,
                                    args.batch_size,args.patience,args.max_train_origins,args.max_val_origins,
                                    args.max_test_origins,configs=configs)
                print(json.dumps({'result':str(Path(args.out)/'result.json'),'run_class':result['run_class'],'metrics':result['metrics']}))


def _reproduce(args):
    from .data import load_dataset
    from .training import train_backbone, extract_cache
    from .experiments import OrbitConfig,run_forecasting,write_json
    root=Path(args.out)
    status=[]
    for name in args.datasets:
        data=load_dataset(args.data,name)
        for seed in args.seeds:
            out=root/name/(args.backbone+('_identity_free' if args.identity_free else ''))/('seed_'+str(seed))
            out.mkdir(parents=True,exist_ok=True)
            checkpoint=out/'checkpoint.pt'
            if not args.resume or not checkpoint.exists():
                checkpoint=train_backbone(data,out,args.backbone,seed,1 if args.smoke else args.epochs,args.batch_size,args.device,
                                          max_train_origins=512 if args.smoke else None,max_val_origins=128 if args.smoke else None,
                                          identity_free=args.identity_free,patience=args.patience)
            else:
                from .training import load_checkpoint
                _,_,existing=load_checkpoint(checkpoint,'cpu')
                run=existing['run_config']
                training_record=json.loads((checkpoint.parent/'training.json').read_text(encoding='utf-8'))
                if not training_record.get('completed',False):
                    raise ValueError('Resume requires a completed training record, not merely an intermediate checkpoint')
                if (existing['dataset_sha256']!=data.metadata['sha256'] or run['seed']!=seed or
                    run['backbone']!=args.backbone or run['limited']!=args.smoke or
                    run['batch_size']!=args.batch_size or run['patience']!=args.patience or
                    existing['model_config']['identity_free']!=args.identity_free or
                    run['epochs_requested']!=(1 if args.smoke else args.epochs)):
                    raise ValueError('Resume checkpoint configuration differs; choose a fresh output directory')
            if not args.resume or not (out/'cache'/'metadata.json').exists():
                extract_cache(data,checkpoint,out/'cache',args.batch_size,args.device,
                              {'train':128,'val':16,'test':24} if args.smoke else None)
            else:
                from .data import file_sha256
                cache_record=json.loads((out/'cache'/'metadata.json').read_text(encoding='utf-8'))
                if cache_record.get('completed') is False or cache_record.get('status')=='in_progress':
                    raise ValueError('Resume found an incomplete feature cache; regenerate it')
                if cache_record['checkpoint_sha256']!=file_sha256(checkpoint) or cache_record['dataset_sha256']!=data.metadata['sha256']:
                    raise ValueError('Resume cache belongs to a different checkpoint or dataset; regenerate it')
            config=OrbitConfig(epoch_rounds=56 if name=='knowair' else 288)
            result=run_forecasting(data,out/'cache',out/'forecast',config,grid=not args.smoke,resume=args.resume)
            status.append({'dataset':name,'backbone':args.backbone,'seed':seed,'result':str(out/'forecast'/'result.json'),
                           'run_class':result['run_class'],'metrics':result['metrics']})
            write_json(root/'completed_runs.json',status)
    if len(args.seeds)>1:
        import numpy as np
        aggregate={}
        for name in args.datasets:
            runs=[r for r in status if r['dataset']==name]
            aggregate[name]={}
            for method in runs[0]['metrics']:
                aggregate[name][method]={}
                for h in runs[0]['metrics'][method]:
                    aggregate[name][method][h]={metric:{'mean':float(np.mean([r['metrics'][method][h][metric] for r in runs])),
                                                    'sample_std':float(np.std([r['metrics'][method][h][metric] for r in runs],ddof=1))}
                                               for metric in ['mae','rmse']}
        write_json(root/'seed_summary.json',{'seeds':args.seeds,'results':aggregate,'limited':args.smoke})
