"""Matched frozen-checkpoint OOD analysis, configurable datasets and limits.

Reuse the existing distances/attention/lesion readers and V/visited editors.
No training and no autoregressive final-answer evaluation is launched.
Hard DAG references reuse their existing 32K prepared cache; only valid
prefixes inside the evaluated models' 4K context are selected, explicitly
counting coverage and censored states. Never relabel these as hard-trained
checkpoints or extrapolate local next-token accuracy to final answers.
"""
import argparse,json,math,runpy,time
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
M=runpy.run_path(str(P/'run.py'));D=M['D']
G=runpy.run_path(str(P/'graph_values_visited.py'))
VALUE_CONDITIONS=['identity','all_graph_contextfree_V','marker_contextfree_V','endpoint_contextfree_V',
                  'identity_L2','visited_all_L2','visited_current_L2','visited_other_L2',
                  'identity_L4','visited_all_L4','visited_current_L4','visited_other_L4']
G['evaluate'].__globals__['CONDITIONS']=VALUE_CONDITIONS
HARD=D['ROOT']/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state'


def finite_json(value):
    """Unobserved role metrics are null, never non-standard JSON NaN."""
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write(path,value):M['write'](path,finite_json(value))


def bounded_cases(source,cache,tok,ids,max_len,hard=False):
    cases=M['build_cases'](source,cache,tok,ids)
    coverage=dict(queries=len(cases),all_candidate_states=0,inside_context_states=0,
                  sampled_states=0,queries_without_usable_state=0,reference_truncated_queries=0,
                  max_selected_prefix=0,selection='uniform up to four non-FOUND states per query')
    output=[]
    for c in cases:
        row=cache.row(c['query_id'],0);recs=M['records'](row,tok)
        nonfound=[r for r in recs if r['action']!=[tok.id('<FOUND>')]]
        fitting=[r for r in nonfound if r['spans']['cands'][0]<=max_len]
        coverage['all_candidate_states']+=len(nonfound)
        coverage['inside_context_states']+=len(fitting)
        coverage['reference_truncated_queries']+=int(row.meta.get('truncated',False))
        if hard:
            # Hard references may exceed 4K. Sample only the in-context pool,
            # while preserving original query identities and the real states.
            rng=np.random.default_rng(131071+c['query_id'])
            chosen=sorted(rng.choice(len(fitting),min(4,len(fitting)),replace=False).tolist())
            ps=[]
            for ri in chosen:
                rec=fitting[ri];clo,_=rec['spans']['cands']
                ps.append(dict(role=M['ROLES'].index('candidate_nonempty' if rec['cands'] else 'candidate_empty'),
                    q=clo-1,start=rec['start'],frontier_end=rec['start']+len(rec['stack']),
                    vis_span=rec['spans']['visited'],cur=rec['stack'][-1],eligible=rec['cands'],record=rec))
            c['probes']=ps
        else:
            c['probes']=[p for p in c['probes'] if M['ROLES'][p['role']] in ('candidate_nonempty','candidate_empty') and p['q']<max_len]
        if not c['probes']:
            coverage['queries_without_usable_state']+=1;continue
        end=max(p['q'] for p in c['probes'])+1
        c['tokens']=c['tokens'][:end]
        assert end<=max_len and all(c['tokens'][p['q']]==tok.id('|') for p in c['probes'])
        coverage['max_selected_prefix']=max(coverage['max_selected_prefix'],end)
        coverage['sampled_states']+=len(c['probes']);output.append(c)
    return output,coverage


def predictions(folder,case):
    path=folder/'eval'/('predictions_test_last'+('' if case=='main' else '-'+case)+'.jsonl')
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',default=['deep','wide','larger','hard_dag'])
    parser.add_argument('--queries',type=int,default=256)
    parser.add_argument('--validation-queries',type=int,default=256)
    parser.add_argument('--out',type=Path,default=P/'harder')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    settings=dict(cases=args.cases,test_queries=args.queries,validation_queries=args.validation_queries,
        value_conditions=VALUE_CONDITIONS,scope='same simple-graph-trained models, OOD datasets; local correct-prefix analyses',
        roles=M['ROLES'],features=M['FEATURES'],max_seq_len=4096)
    write(args.out/'settings.json',settings)
    for case in args.cases:
        dest=args.out/(case+'.json')
        if dest.exists():continue
        start=time.time();cfg,tok,model=M['load_model']('clean')
        extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
        gains=set();answer=None
        if case=='hard_dag':
            data_cfg=D['load_config']([HARD/'resolved_config.yaml'])
            source=D['ContextStore'](data_cfg.data.root)
            cache=D['ScalePreparedRows'](source,data_cfg,tok)
            old=predictions(HARD,'main');ids=[r['query_id'] for r in old][:args.queries]
        else:
            source=D['ContextStore'](extra[case]);cache=D['ScalePreparedRows'](source,cfg,tok)
            rows=[predictions(D['FOLDERS']['dfs_'+a],case) for a in ('clean','error')]
            assert [r['query_id'] for r in rows[0]]==[r['query_id'] for r in rows[1]]
            ids=[r['query_id'] for r in rows[0]][:args.queries]
            def correct(r):
                if 'correct' in r:return bool(r['correct'])
                return r['pred']==r['label']
            answer=dict(n=len(ids),clean_correct=sum(correct(r) for r in rows[0][:len(ids)]),
                        error_correct=sum(correct(r) for r in rows[1][:len(ids)]),source='saved original batch16 free-generation predictions')
            gains={a['query_id'] for a,b in zip(*rows) if not correct(a) and correct(b)}
        a=source.arrays
        valids=M['stratified_ids'](source.indices('val'),a['shape_id'],a['distance'],a['label'],args.validation_queries,cfg.evaluation.seed)
        assert not set(ids)&set(map(int,valids))
        test,coverage=bounded_cases(source,cache,tok,ids,model.max_seq_len,case=='hard_dag')
        validation,valcoverage=bounded_cases(source,cache,tok,valids,model.max_seq_len,case=='hard_dag')
        ids=[c['query_id'] for c in test]
        result=dict(case=case,query_ids=ids,coverage=coverage,validation_coverage=valcoverage,
                    navigation=answer,value={},checks=[],calibration=[])
        attention=[];lesion=[]
        for arm in ('clean','error'):
            if arm=='error':
                del model;torch.cuda.empty_cache();_,other_tok,model=M['load_model']('error')
                assert other_tok.itos[:len(tok)]==tok.itos
            print(case,arm,'calibrate',flush=True)
            mu,count=M['calibrate'](model,validation)
            print(case,arm,'observe',flush=True)
            attention.append(M['observe'](model,test,mu,count))
            result['calibration'].append(dict(probes_per_role=count[:,0].tolist(),
                max_supported_distance=[int(np.flatnonzero(c)[-1]) if c.any() else -1 for c in count]))
            print(case,arm,'lesions',flush=True)
            v,ch=M['lesions'](model,tok,test);lesion.append(v);result['checks'].append(ch)
            print(case,arm,'graph values and visited',flush=True)
            result['value'][arm]=G['evaluate'](model,tok,test)
            write(dest.with_suffix('.partial.json'),result)
        result['attention']=M['paired_summary'](*attention,ids,gains)
        result['lesion']=M['paired_summary'](*lesion,ids,gains)
        result['seconds']=time.time()-start
        result['finished']=True;write(dest,result)
        dest.with_suffix('.partial.json').unlink(missing_ok=True)
        print('FINISHED',case,round(result['seconds'],1),flush=True)
        del model;torch.cuda.empty_cache()
    write(args.out/'finished.json',dict(finished=True))


if __name__=='__main__':main()
