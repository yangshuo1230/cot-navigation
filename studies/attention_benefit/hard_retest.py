"""Actual hard DAG test: full free navigation and hard-trained reference.

The exact same 256 balanced N100/M200 multiscale-DAG queries are evaluated
from graph+query ONLY. Each checkpoint retains its native context limit.
No training, checkpoint writes, W&B runs or full activation/trace dumps.
"""
import argparse,hashlib,json,runpy,time
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
H=runpy.run_path(str(P/'harder.py'));M=H['M'];D=M['D'];G=H['G']
OUT=P/'hard_retest'
FOLDERS={'hard_clean':H['HARD'],'simple_clean':D['FOLDERS']['dfs_clean'],
         'simple_error':D['FOLDERS']['dfs_error']}


def load(name):
    folder=FOLDERS[name];cfg=D['load_config']([folder/'resolved_config.yaml'])
    tok=D['Tokenizer'].from_config(cfg)
    model=D['build_model'](cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    ckpt=folder/'checkpoints/latest.pt';blob=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert blob['step']==200000;model.load_state_dict(blob['model'],strict=True);del blob
    meta=dict(checkpoint=str(ckpt),step=200000,sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
              max_seq_len=model.max_seq_len,lambda_trace=cfg.train.trace_loss_lambda,
              training_graph_family=cfg.data.graph_family,training_data_root=cfg.data.root,
              protocol=cfg.protocol.name,enable_mistake=getattr(cfg.protocol,'enable_mistake',False))
    return cfg,tok,model,meta


def test_source():
    cfg=D['load_config']([H['HARD']/'resolved_config.yaml']);tok=D['Tokenizer'].from_config(cfg)
    source=D['ContextStore'](cfg.data.root);cache=D['ScalePreparedRows'](source,cfg,tok)
    old=H['predictions'](H['HARD'],'main')
    assert len(old)==len({r['query_id'] for r in old})==256
    assert sum(r['label'] for r in old)==128
    assert cfg.data.graph_family=='multiscale_dag_v1'
    assert cfg.data.n_nodes==100 and cfg.data.n_edges==200
    assert set(r['query_id'] for r in old)<=set(map(int,source.indices('test')))
    return cfg,tok,source,cache,old


def navigation(names):
    source_cfg,source_tok,source,cache,old=test_source()
    for name in names:
        dest=OUT/(name+'_navigation.json')
        if dest.exists() and json.loads(dest.read_text()).get('finished'):continue
        cfg,tok,model,meta=load(name)
        assert tok.itos[:len(source_tok)]==source_tok.itos
        start=time.time();rows=[];examples=[]
        for offset in range(0,len(old),16):
            batch=old[offset:offset+16];refs=[cache.row(r['query_id'],0) for r in batch]
            progress=dict(steps=0,last=time.time())
            def progress_hook(module,args,out):
                if args[0].shape[1]==1:progress['steps']+=1
                if time.time()-progress['last']>=25:
                    print('DECODE',name,'queries_done',offset,'batch_decode_steps',progress['steps'],
                          'elapsed',round(time.time()-start,1),flush=True)
                    progress['last']=time.time()
            handle=model.register_forward_hook(progress_hook)
            try:
                with torch.no_grad():
                    generated=D['generate'](model,[r.tokens[:r.prompt_len] for r in refs],cfg.protocol.name,tok,
                        cfg.train.precision,max_new_tokens=cfg.evaluation.max_new_tokens,
                        cot_margin=cfg.protocol.cot_margin,crop_cache=True,compact_finished=True)
            finally:handle.remove()
            for previous,ref,value in zip(batch,refs,generated):
                idx=int(previous['query_id']);nodes=source.node_vocab_ids(idx,tok)
                adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(source.adj(idx))}
                s=nodes[int(source.arrays['s_topo'][idx])];t=nodes[int(source.arrays['t_topo'][idx])]
                seen={s};todo=[s]
                while todo:
                    for v in adj[todo.pop()]:
                        if v not in seen:seen.add(v);todo.append(v)
                assert int(t in seen)==ref.label==previous['label']
                audit=D['audit'](value['trace_tokens'],value['pred'],adj,s,t,tok,False)
                correct=not value['format_fail'] and value['pred']==ref.label
                row=dict(query_id=idx,label=ref.label,pred=value['pred'],correct=correct,
                         format_fail=value['format_fail'],interrupted=value['cot_interrupted'],
                         forced_stop=value['forced_stop'],trace_length=len(value['trace_tokens']),
                         raw_mistake_tokens=value['trace_tokens'].count(tok.stoi.get('<MISTAKE>',-1)),
                         trace_sha256=hashlib.sha256(str(value['trace_tokens']).encode()).hexdigest(),**audit)
                if name=='hard_clean':row['matches_saved']=value['pred']==previous['pred'] and correct==previous['correct']
                rows.append(row)
                if len(examples)<2 and not audit['effective_complete']:
                    examples.append(dict(query_id=idx,correct=correct,first_error=audit['first_error'],
                        trace_begin=tok.decode(value['trace_tokens'][:60]),trace_end=tok.decode(value['trace_tokens'][-60:])))
            counts=dict(n=len(rows),correct=sum(r['correct'] for r in rows),
                yes_n=sum(r['label']==1 for r in rows),no_n=sum(r['label']==0 for r in rows),
                yes_correct=sum(r['label']==1 and r['correct'] for r in rows),
                no_correct=sum(r['label']==0 and r['correct'] for r in rows),
                effective_complete=sum(r['effective_complete'] for r in rows),
                interrupted=sum(r['interrupted'] for r in rows),forced_stop=sum(r['forced_stop'] for r in rows),
                candidate_bad_queries=sum(r['candidate_bad']>0 for r in rows),
                marker_queries=sum(r['raw_mistake_tokens']>0 for r in rows))
            if name=='hard_clean':counts['matches_saved']=sum(r['matches_saved'] for r in rows)
            result=dict(model=name,source=str(source.root),metadata=meta,batch_size=16,
                prompt='edge list + query only; no gold trajectory',native_context=True,
                counts=counts,rows=rows,examples=examples,seconds=time.time()-start,finished=len(rows)==256)
            H['write'](dest,result)
            print('NAV',name,len(rows),counts,'seconds',round(time.time()-start,1),flush=True)
        del model;torch.cuda.empty_cache()


def anchor_attention():
    dest=OUT/'hard_clean_attention.json'
    if dest.exists() and json.loads(dest.read_text()).get('finished'):return
    cfg,tok,model,meta=load('hard_clean')
    _,_,source,cache,old=test_source();ids=[r['query_id'] for r in old]
    a=source.arrays
    valids=M['stratified_ids'](source.indices('val'),a['shape_id'],a['distance'],a['label'],256,cfg.evaluation.seed)
    assert not set(ids)&set(map(int,valids))
    # Exact same states as the two simple-trained models' completed hard-DAG
    # local analysis. Preserve the hard model's native 32K configuration.
    test,coverage=H['bounded_cases'](source,cache,tok,ids,4096,hard=True)
    val,valcoverage=H['bounded_cases'](source,cache,tok,valids,4096,hard=True)
    start=time.time();print('ANCHOR calibrate',flush=True)
    mu,count=M['calibrate'](model,val)
    print('ANCHOR observe',flush=True);obs=M['observe'](model,test,mu,count)
    print('ANCHOR lesions',flush=True);lesion,checks=M['lesions'](model,tok,test)
    print('ANCHOR V/visited',flush=True);value=G['evaluate'](model,tok,test)
    H['write'](dest,dict(finished=True,metadata=meta,source=str(source.root),coverage=coverage,
        validation_coverage=valcoverage,common_analysis_prefix_limit=4096,
        attention=M['paired_summary'](obs,obs,ids,set())['clean'],
        lesion=M['paired_summary'](lesion,lesion,ids,set())['clean'],lesion_checks=checks,
        value=value,seconds=time.time()-start,query_ids=ids))
    del model;torch.cuda.empty_cache()


def anchor_graph6():
    dest=OUT/'hard_clean_graph6.json'
    if dest.exists() and json.loads(dest.read_text()).get('finished'):return
    cfg,tok,model,meta=load('hard_clean')
    _,_,source,cache,old=test_source()
    test,coverage=H['bounded_cases'](source,cache,tok,[r['query_id'] for r in old],4096,hard=True)
    G['evaluate'].__globals__['CONDITIONS']=['identity','all_graph_contextfree_V','marker_contextfree_V','endpoint_contextfree_V']
    value=G['evaluate'](model,tok,test,graph_layer=6)
    H['write'](dest,dict(finished=True,metadata=meta,coverage=coverage,value=value))
    del model;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['navigation','attention','graph6','all'],default='all')
    parser.add_argument('--models',nargs='+',choices=list(FOLDERS),default=list(FOLDERS))
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    if args.phase in ('attention','all'):anchor_attention()
    if args.phase=='graph6':anchor_graph6()
    if args.phase in ('navigation','all'):navigation(args.models)
    H['write'](OUT/('finished_'+args.phase+'.json'),dict(finished=True,models=args.models))


if __name__=='__main__':main()
