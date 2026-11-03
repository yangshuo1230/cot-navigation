"""Reusable follow-up: exhaustive route blocking, error and set probes.

Run on physical GPU0. No navigation weights, probe checkpoints, W&B runs,
activation dumps or large rollout caches are written. The earlier study is
read-only. Each model contributes one resumable JSON and a few decoded cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=Path('studies/mechanistic/results'))
    parser.add_argument('--output',type=Path,default=Path('studies/mechanistic_followup/results'))
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--models',nargs='*')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stages',nargs='+',choices=['sets','errors','blocking'],default=['sets','errors','blocking'])
    args=parser.parse_args()
    if args.gpu!='0': parser.error('Use physical GPU0, as authorized')
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
    import numpy as np
    import torch
    from .common import AnalysisModel,SEED,write_json,sha256
    from .probes import extract,fit,evaluate,shuffled_labels,subset
    from .patching import free_rollouts
    from .set_metrics import evaluate_sets,evaluate_errors,scores
    from .error_detection import events,extract_events
    from .route_blocking import run_blocking
    torch.set_num_threads(4)
    old_settings=json.loads((args.previous/'settings.json').read_text())
    selection=json.loads((args.previous/'selection.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'seed':SEED,'gpu':0,'previous':str(args.previous),'probe_steps':250,
              'set_queries':{k:old_settings[k+'_queries'] for k in ['train','val','test']},
              'natural_error_train_queries':128,'natural_error_val_queries':64,
              'natural_error_training_generation_cap':512,'test_generation_cap':'native context',
              'synthetic_error_train_queries':256,'synthetic_error_val_queries':128,
              'error_readouts':['wrong_token','same_next_token'], 'patch_cases':40,
              'error_oracle':'semantic first violation; order-invariant sets/legal PUSH/path; BFS distance tiers',
              'error_next_target':'whether designated PREVIOUS token was erroneous, identical next token in both arms'}
    source_names=['followup.py','set_metrics.py','error_detection.py','route_blocking.py']
    settings['source_sha256']={name:sha256(Path(__file__).parent/name) for name in source_names}
    settings_path=args.output/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Settings/source changed: choose a fresh output directory')
    write_json(settings_path,settings)
    entries=[e for e in selection['models'] if e['id']!='context_filler_50k_control']
    write_json(args.output/'selection.json',{'models':entries,'selection_reused_from_previous':True})
    def joined(items):
        items=[x for x in items if x is not None]
        return {k:np.concatenate([x[k] for x in items]) for k in items[0]} if items else None
    def randomized_errors(data,seed):
        # Unlike answer probes, each query has BOTH wrong/correct labels.
        # The answer shuffler assumes one label/query and cannot be used here.
        return {**data,'y':np.random.default_rng(seed).permutation(data['y'])}
    for entry in entries:
        if args.models and entry['id'] not in args.models: continue
        name=entry['id'];path=args.output/'models'/f'{name}.json'
        old=json.loads((args.previous/'models'/f'{name}.json').read_text())
        ckpt=Path.cwd()/entry['run']/'checkpoints/best.pt'
        if sha256(ckpt)!=old['checkpoint']['sha256']: raise ValueError('Checkpoint changed: '+name)
        result=json.loads(path.read_text()) if args.resume and path.exists() else {'entry':entry,'checkpoint':old['checkpoint']}
        if all(stage in result for stage in args.stages): print('SKIP',name,flush=True);continue
        print('MODEL',name,flush=True)
        am=AnalysisModel(entry,Path.cwd())
        train=am.examples('train',old_settings['train_queries']);val=am.examples('val',old_settings['val_queries'])
        test=am.examples('test',old_settings['test_queries'])
        free,rollouts=free_rollouts(am,test,max_new=0)
        original={r['query']:r['pred'] for r in old['rollout']['by_query']}
        result['rollout']={'n':len(test),'accuracy':float(np.mean([r['correct'] for r in rollouts])),
            'limits':sum(r['limit'] for r in rollouts),
            'agreement_previous':float(np.mean([r['pred']==original[r['query']] for r in rollouts]))}
        result['query_ids']={k:[e.index for e in ex] for k,ex in [('train',train),('val',val),('test',test)]}
        assert not set(result['query_ids']['train'])&set(result['query_ids']['test'])
        assert not set(result['query_ids']['train'])&set(result['query_ids']['val'])
        assert not set(result['query_ids']['val'])&set(result['query_ids']['test'])
        if 'sets' in args.stages and 'sets' not in result:
            print(' SET PROBABILITIES',flush=True)
            tr,va,te,ge=extract(am,train),extract(am,val),extract(am,test),extract(am,test,free)
            out={}
            for target in ['successors','visited']:
                if target not in tr or target not in va or target not in te: continue
                fitted=fit(tr[target],va[target],device=am.device,seed=SEED)
                control=fit(tr[target],va[target],device=am.device,seed=SEED+17,shuffled=True)
                shuffled={**te[target],'y':shuffled_labels(te[target],SEED+17+202)}
                record={'layers':[4,5,6,'input_control'],
                        'reference':evaluate_sets(fitted,te[target]),'shuffled_reference':evaluate_sets(control,shuffled),
                        'generated':evaluate_sets(fitted,ge[target]) if target in ge else None,
                        'selected_steps':fitted['steps'], 'n_anchors':{k:len(d[target]['y']) for k,d in [('train',tr),('val',va),('test',te)]}}
                previous=old['probes']['targets'].get(target,{})
                if target in ge and 'generated_test' in previous:
                    current=evaluate(fitted,ge[target])
                    record['refit_f1_max_difference']=float(max(abs(a['f1']-b['f1']) for a,b in zip(current,previous['generated_test'])))
                out[target]=record
                del fitted,control
            result['sets']=out;write_json(path,result)
            del tr,va,te,ge
        if 'blocking' in args.stages and 'blocking' not in result:
            if am.protocol in {'dfs_state','bfs_state'}:
                print(' ALL-ROUTE BLOCKING',flush=True)
                result['blocking']={'including_goal':run_blocking(am,test,free),
                                    'excluding_goal':run_blocking(am,test,free,exclude_goal=True)}
            else: result['blocking']={'status':'not applicable: no explicit visited field'}
            write_json(path,result)
        if 'errors' in args.stages and 'errors' not in result:
            if am.protocol in {'direct','filler'}:
                result['errors']={'status':'not applicable: no freely generated CoT tokens'}
            else:
                print(' ERROR EVENTS',flush=True)
                natural={};synthetic={};splits={'train':train,'val':val,'test':test}
                error_rollouts={}
                for split,examples in splits.items():
                    if split=='test': seqs=free;rr=rollouts;used=examples
                    else:
                        count=128 if split=='train' else 64;used=examples[:count]
                        seqs,rr=free_rollouts(am,used,max_new=512)
                    natural[split]=[event for ex in used if (event:=events(ex,am.protocol,am.tok,seqs[ex.index]))]
                    error_rollouts[split]={'queries':len(used),'truncated':sum(r['limit'] for r in rr),
                        'first_errors':len(natural[split]),'fields':dict(Counter(e['field'] for e in natural[split])),
                        'kinds':dict(Counter(e['kind'] for e in natural[split]))}
                    limit=256 if split=='train' else 128 if split=='val' else len(examples)
                    synthetic[split]=[event for ex in examples[:limit]
                        if (event:=events(ex,am.protocol,am.tok,ex.row.tokens,synthetic=True,seed=SEED))]
                error_result={'status':'complete','natural_inventory':error_rollouts,
                    'synthetic_counts':{k:len(v) for k,v in synthetic.items()},'readouts':{}}
                for offset,readout in [(0,'wrong_token'),(1,'next_token')]:
                    print('  FIT',readout,flush=True)
                    data={}
                    for split,examples in splits.items():
                        for source,event_dict in [('natural',natural),('synthetic',synthetic)]:
                            data[(split,source)]=extract_events(am,examples,event_dict[split],offset)
                    tr=joined([data[('train','natural')],data[('train','synthetic')]])
                    va=joined([data[('val','natural')],data[('val','synthetic')]])
                    if tr is None or va is None:
                        error_result['readouts'][readout]={'status':'insufficient train/val pairs'};continue
                    fitted=fit(tr,va,device=am.device,seed=SEED)
                    control=fit(randomized_errors(tr,SEED+17),randomized_errors(va,SEED+118),
                                device=am.device,seed=SEED+17)
                    record={'layers':[4,5,6,'input_control'],'selected_steps':fitted['steps'],
                            'train_rows':len(tr['y']),'val_rows':len(va['y']),
                            'train':evaluate_errors(fitted,tr),'tests':{}}
                    for source in ['natural','synthetic']:
                        te=data[('test',source)]
                        if te is None: record['tests'][source]={'status':'no eligible errors'};continue
                        td=randomized_errors(te,SEED+219)
                        summary={'all':evaluate_errors(fitted,te),'shuffled':evaluate_errors(control,td),
                                 'n_pairs':len(te['y'])//2,'by_kind':{},'by_field':{}}
                        if offset:
                            valid=te['repair_next_valid']
                            summary['strict_repaired_next_legal_pairs']=int(valid.sum()//2)
                            if valid.any():
                                summary['strict_repaired_next_legal']=evaluate_errors(fitted,subset(te,valid))
                        for key in ['kind','field']:
                            for value in np.unique(te[key]):
                                sub=subset(te,te[key]==value)
                                summary['by_'+key][str(value)]=evaluate_errors(fitted,sub)
                        record['tests'][source]=summary
                    error_result['readouts'][readout]=record
                    del data,tr,va,fitted,control
                # Keep a handful of paired examples, never complete activation caches.
                error_result['examples']=[]
                for source,event_dict in [('natural',natural),('synthetic',synthetic)]:
                    for e in event_dict['test'][:3]:
                        error_result['examples'].append({k:v for k,v in e.items() if k not in {'wrong','correct'}}|
                            {'wrong_prefix_tail':am.tok.decode(e['wrong'][-80:]),
                             'correct_prefix_tail':am.tok.decode(e['correct'][-80:]),
                             'wrong_token_name':am.tok.name(e['wrong_token']),
                             'correct_token_name':am.tok.name(e['correct_token']),
                             'next_token_name':am.tok.name(e['next_token']) if e['next_token'] is not None else None})
                result['errors']=error_result
            write_json(path,result)
        result['checkpoint_unchanged_after']=sha256(ckpt)==old['checkpoint']['sha256']
        write_json(path,result)
        print(' DONE',name,flush=True)
        del am,free
        torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


if __name__=='__main__': main()
