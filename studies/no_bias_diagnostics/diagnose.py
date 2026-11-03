"""Paired frozen oracle-answer and free-trace diagnostics; no training."""
from pathlib import Path
from types import SimpleNamespace
from collections import deque
import argparse,gzip,hashlib,json,os,sys,time
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
os.chdir(ROOT)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies/curriculum_trace_errors')]
import numpy as np
import torch
import audit
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore,ParametricWorld
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import context_prompt,parametric_prompt,render_task,truncate_example
from cot_navigation.training.augmentation import random_path,with_body
from cot_navigation.evaluation.rollout import generate,pad_rows

OUT=ROOT/'studies/no_bias_diagnostics/results'

def models():
    entries=json.loads((ROOT/'studies/scale_mechanistic/results/selection.json').read_text())['models']
    jobs=[{**e,'tag':'best'} for e in entries if e['scale']=='curriculum']
    base=ROOT/'outputs/curriculum_loss_swa_v1/launches/curriculum_dfs_loss_swa_s0/context/dfs_state'
    for arm in ('full_l1','full_l4','full_l16','swa1_l1','swa2_l1'):
        jobs.append(dict(id='context_loss_'+arm,run=str(base/arm),tag='last',kind='context',protocol='dfs_state',scale='curriculum'))
    return jobs

def oracle_trace(ex,prompt,protocol,tok,cfg,idx):
    row=render_task(tok,protocol,ex.label,prompt,ex.adj,ex.source,ex.target,ex.nodes)
    if protocol=='path' and ex.label==0 and cfg.protocol.negative_random_path:
        path=random_path(ex.adj,ex.source,np.random.default_rng(2026102200+idx))
        row=with_body(row,[ex.nodes[v] for v in path])
    full=row.tokens[row.prompt_len:row.answer_index]
    truncated=truncate_example(row,tok,cfg.model.max_seq_len,cfg.protocol.cot_margin)
    return truncated.tokens[:truncated.answer_index],full,bool(truncated.meta.get('truncated',False))

@torch.inference_mode()
def answer_logits(model,prefixes,tok,precision):
    results=[None]*len(prefixes)
    order=sorted(range(len(prefixes)),key=lambda i:len(prefixes[i]))
    # Small prefill batches keep 4096-token masked attention bounded.
    for at in range(0,len(order),4):
        ids=order[at:at+4];rows=[prefixes[i] for i in ids]
        x,lengths=pad_rows(rows,tok.pad,'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            logits=model(x)[torch.arange(len(ids),device='cuda'),lengths-1].float()
        probs=logits[:,[tok.label_id(0),tok.label_id(1)]].softmax(-1).cpu().tolist()
        picks=logits.argmax(-1).cpu().tolist()
        for i,pick,prob in zip(ids,picks,probs):
            pred=1 if pick==tok.label_id(1) else 0 if pick==tok.label_id(0) else -1
            results[i]={'pred':pred,'p_yes':prob[1],'label_only_pred':int(prob[1]>prob[0]),'argmax_token':pick}
    return results

def replay(ex,body,protocol,tok):
    adj={ex.nodes[u]:set(ex.nodes[v] for v in vs) for u,vs in enumerate(ex.adj)}
    if protocol.endswith('_state'):
        return audit.full_state_audit(ex,body,tok,protocol=='bfs_state')
    if protocol=='direct':
        return {'complete_valid':None,'prefix_valid':None,'events':[],'first_error':None}
    return audit.sequence_audit(body,adj,ex.nodes[ex.source],ex.nodes[ex.target],ex.label,protocol)

def graph_reached(ex):
    reached={ex.source};stack=[ex.source]
    while stack:
        u=stack.pop()
        for v in ex.adj[u]:
            if v not in reached:reached.add(v);stack.append(v)
    return reached

def target_is_current(body,protocol,tok,target):
    if not protocol.endswith('_state'):return target in body
    start=0
    for end,v in enumerate(body):
        if v!=tok.id(';'):continue
        record=body[start:end];start=end+1
        if tok.id('|') not in record:continue
        frontier=record[:record.index(tok.id('|'))]
        if frontier and frontier[0 if protocol=='bfs_state' else -1]==target:return True
    return False

def run(entry,per_cell,batch):
    dest=OUT/entry['id'];dest.mkdir(parents=True,exist_ok=True)
    cfg=load_config([ROOT/entry['run']/'resolved_config.yaml'])
    tok=Tokenizer.from_config(cfg)
    checkpoint=ROOT/entry['run']/'checkpoints'/('latest.pt' if entry['tag']=='last' else 'best.pt')
    before=audit.sha(checkpoint)
    if entry.get('checkpoint_sha256'):assert before==entry['checkpoint_sha256']
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    # Same verified record mask; avoid new flex autotuning during frozen prefill.
    if model.swa_records:model.swa_backend='sdpa'
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(blob['model'],strict=True)
    step=int(blob['step']);del blob
    context=entry['kind']=='context';roots=cfg.train.scale_roots if context else cfg.train.world_roots
    if context:
        cases=[('n100',cfg.train.scale_roots[-1])]+[tuple(x.split('=',1)) for x in cfg.evaluation.extra_sets if x.split('=',1)[0] in ('deep','larger')]
    else:
        # Only a world that actually participated in task training.
        trained={Path(x).resolve() for x in cfg.train.scale_roots}
        cases=[('n100',p) for p in roots if Path(p).parent.resolve() in trained and 'n100_' in str(p) and 'l10w10' in str(p)]
        assert len(cases)==1,(entry['id'],cases)
    for case,root in cases:
        receipt=dest/(case+'.json')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if not entry['protocol'].endswith('_state') or old.get('audit_version')==2:continue
        started=time.time();source=ContextStore(root) if context else ParametricWorld(root)
        a=source.arrays if context else source.queries
        chosen,cells=audit.select_ids(source,context,per_cell,20261022)
        world=[] if context else [tok.id(f'<WORLD_{[Path(x).resolve() for x in roots].index(Path(root).resolve())}>')]
        rows=[]
        for at in range(0,len(chosen),batch):
            prepared=[]
            for idx in chosen[at:at+batch]:
                adj=source.adj(idx) if context else source.adj()
                nodes=source.node_vocab_ids(idx,tok) if context else source.node_vocab_ids(tok)
                s,t=int(a['s_topo'][idx]),int(a['t_topo'][idx]);label=int(a['label'][idx])
                ex=SimpleNamespace(nodes=nodes,adj=adj,source=s,target=t,label=label)
                reached=graph_reached(ex);assert int(t in reached)==label
                if context:
                    edges=list(zip(a['edge_src'][idx].astype(int).tolist(),a['edge_dst'][idx].astype(int).tolist()))
                    prompt=context_prompt(tok,nodes,edges,s,t)
                else:prompt=world+parametric_prompt(tok,nodes,s,t)
                prefix,full,cut=oracle_trace(ex,prompt,entry['protocol'],tok,cfg,int(idx))
                if entry['protocol']!='direct':
                    valid=replay(ex,full,entry['protocol'],tok)
                    assert valid['complete_valid'],(entry['id'],case,idx,valid)
                prepared.append((int(idx),ex,prompt,prefix,full,cut,reached))
            oracle=answer_logits(model,[x[3] for x in prepared],tok,cfg.train.precision)
            outputs=generate(model,[x[2] for x in prepared],entry['protocol'],tok,cfg.train.precision,
                max_new_tokens=cfg.evaluation.max_new_tokens,cot_margin=cfg.protocol.cot_margin)
            for (idx,ex,prompt,prefix,full,cut,reached),o,out in zip(prepared,oracle,outputs):
                raw=list(out['trace_tokens']);stop=tok.id('<STOP>') in raw
                body=raw[:raw.index(tok.id('<STOP>'))] if stop else raw
                r=replay(ex,body,entry['protocol'],tok)
                complete=bool(r['complete_valid']) and not stop if entry['protocol']!='direct' else None
                visited_tokens=set(body)&set(ex.nodes)
                # For PATH, this coverage is descriptive; a random path is not an exhaustive search.
                reached_tokens={ex.nodes[v] for v in reached}
                errors=sorted({k for e in r.get('events',[]) for k in e.get('kinds',[e.get('kind')]) if k})
                item=dict(query=idx,label=ex.label,gap=int(a['distance'][idx]),free_pred=out['pred'],
                    free_correct=out['pred']==ex.label,format_fail=bool(out['format_fail']),
                    stop=stop,forced_stop=bool(out['forced_stop']),trace_tokens=len(body),
                    oracle=o,oracle_correct=o['pred']==ex.label,oracle_truncated=cut,
                    oracle_tokens=len(prefix)-len(prompt),oracle_full_tokens=len(full),
                    execution_valid=(bool(r.get('action_valid')) and not stop) if entry['protocol']!='direct' else None,
                    trace_body=body,complete_valid=complete,prefix_valid=r['prefix_valid'],first_error=r['first_error'],
                    errors=errors,all_reachable_nodes_mentioned=reached_tokens<=visited_tokens,
                    reachable_node_mention_coverage=len(reached_tokens&visited_tokens)/len(reached_tokens),
                    target_printed=ex.nodes[ex.target] in visited_tokens,
                    target_current=target_is_current(body,entry['protocol'],tok,ex.nodes[ex.target]),
                    has_found=tok.id('<FOUND>') in body,has_exhaust=tok.id('<EXHAUST>') in body,
                    negative_certificate_supported=entry['protocol'] in ('dfs_state','bfs_state','dfs_nodes','bfs_nodes'))
                if len(rows)<4 or (item['free_pred']==0 and item['label']==1):
                    item['output_tail']=tok.decode(raw[-100:])
                    item['oracle_tail']=tok.decode(full[-70:])
                rows.append(item)
            print(entry['id'],case,min(at+batch,len(chosen)),'/',len(chosen),flush=True)
        with gzip.open(dest/(case+'.jsonl.gz'),'wt',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        assert audit.sha(checkpoint)==before
        audit.write(receipt,{'model':entry,'case':case,'root':str(root),'checkpoint_sha256':before,
            'audit_version':2,'step':step,'n':len(rows),'per_cell':per_cell,'batch':batch,'sampling_cells':cells,
            'query_ids':chosen,'elapsed_seconds':time.time()-started,
            'oracle_note':'No answer/EOS supplied. Explicit terminal actions in full reference expose an answer-related cue; not proof of graph reasoning.',
            'path_note':'A valid negative random path does not certify nonreachability.',
            'direct_note':'Direct has no trained CoT; oracle condition equals original prompt.'})
        del source
    del model;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--per-cell',type=int,default=4);p.add_argument('--batch',type=int,default=16)
    args=p.parse_args();torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    audit.write(OUT/f'worker{args.worker}_plan.json',{'models':models()[args.worker::args.workers], 'args':vars(args)})
    for entry in models()[args.worker::args.workers]:run(entry,args.per_cell,args.batch)
    audit.write(OUT/f'worker{args.worker}_done.json',{'done':True})
