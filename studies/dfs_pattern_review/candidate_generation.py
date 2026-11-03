"""Candidate-field generation and a position-preserving history ablation.

Freeze weights. Keep QUERY/WORLD/EDGE and current stack+visited unchanged.
Block old CoT keys only for queries in the current record, at every layer,
including while decoding the candidate list. No oracle candidate tokens given.
"""
from pathlib import Path
from types import MethodType
from collections import defaultdict
import json,hashlib
import numpy as np
import torch
import measure as m
from cot_navigation.evaluation.rollout import pad_rows,_gather_row,_decode_step

NAMES=['context_n16','context_n100','fixed_simple_n100','fixed_hard_n100']
OUT=m.OUT/'candidate_generation.json'

class HistoryEditor:
    def __init__(self,model,states,mode):
        self.model=model;self.mode=mode;self.saved=[];self.handles=[];self.pos={}
        self.starts=torch.tensor([s['start'] for s in states],device='cuda')
        self.prompts=torch.tensor([s['prompt_len'] for s in states],device='cuda')
        for li,block in enumerate(model.blocks):
            attn=block.attn;old=attn._attend;self.saved.append((attn,old))
            def pre(module,args,kwargs,li=li):
                self.pos[li]=kwargs.get('positions',args[1] if len(args)>1 else None)
            self.handles.append(attn.register_forward_pre_hook(pre,with_kwargs=True))
            def attend(this,q,k,v,keep,force_math,causal=False,li=li,old=old):
                positions=self.pos[li]
                keys=torch.arange(k.shape[-2],device=q.device)
                allowed=keys[None,None,:]<=positions[:,:,None]
                if self.mode=='no_old_cot':
                    target=positions>=self.starts[:,None]
                    oldcot=(keys[None,:]>=self.prompts[:,None])&(keys[None,:]<self.starts[:,None])
                    allowed &= ~(target[:,:,None]&oldcot[:,None,:])
                allowed=allowed[:,None]
                if keep is not None:allowed=allowed&keep
                return old(q,k,v,allowed,force_math=force_math,causal=False)
            attn._attend=MethodType(attend,attn)
    def close(self):
        for attn,method in self.saved:attn._attend=method
        for h in self.handles:h.remove()

@torch.inference_mode()
def decode(model,tok,states,mode):
    tokens,lengths=pad_rows([s['prefix'] for s in states],tok.pad,'cuda')
    editor=HistoryEditor(model,states,mode)
    try:
        # Identity comparison uses the same padded batch and selected rows.
        if mode=='full':
            editor.close();native=_gather_row(model(tokens),lengths-1).float()
            editor=HistoryEditor(model,states,mode)
            modified=_gather_row(model(tokens),lengths-1).float()
            identity=dict(argmax_drift=int((native.argmax(-1)!=modified.argmax(-1)).sum()),
                          max_logit_error=float((native-modified).abs().max()))
        else:identity=None
        cache=model.new_cache(len(states),torch.device('cuda'),torch.float32)
        positions=torch.arange(tokens.shape[1],device='cuda')[None].expand(len(states),-1)
        logits=model(tokens,positions=positions,cache=cache,cache_mode='prefill')
        cache.lengths=lengths.clone();cache.valid_length=tokens.shape[1]
        next_logits=_gather_row(logits,lengths-1)
        live=[True]*len(states);body=[[] for _ in states];finished=[False]*len(states);bad=[None]*len(states)
        for step in range(101):
            picks=next_logits.argmax(-1).tolist();feed=[tok.pad]*len(states)
            for i in range(len(states)):
                if not live[i]:continue
                p=picks[i]
                if p==tok.id('|'):live[i]=False;finished[i]=True
                elif p not in states[i]['node_set']:
                    live[i]=False
                    if tok.node_lo<=p<tok.node_lo+tok.n_node_tokens:
                        body[i].append(p);bad[i]='outside_graph_node:'+tok.name(p)
                    else:bad[i]='non_node:'+tok.name(p)
                elif int(cache.lengths[i])>=model.max_seq_len-1:live[i]=False;bad[i]='context_limit'
                else:body[i].append(p);feed[i]=p
            if not any(live):break
            next_logits=_decode_step(model,cache,torch.tensor(feed,device='cuda')[:,None],'fp32',torch.tensor(live,device='cuda'))
        return [dict(tokens=xs,closed=closed,error=error) for xs,closed,error in zip(body,finished,bad)],identity
    finally:editor.close()

def summarize(rows):
    if not rows:return dict(n_states=0)
    groups=defaultdict(list)
    for r in rows:groups[r['query_id']].append(r)
    keys=['set_correct','closed','duplicate','nonedge','visited_output','missing','extra','first_legal']
    return dict(n_states=len(rows),n_queries=len(groups),
        **{key:dict(state_mean=float(np.mean([r[key] for r in rows])),
                    query_mean=float(np.mean([np.mean([r[key] for r in rs]) for rs in groups.values()]))) for key in keys})

def main(name):
    folder,data=m.SETTINGS[name];cfg,tok,model,step=m.base.load_model(folder,torch.device('cuda'))
    model.requires_grad_(False);context=name.startswith('context')
    source=m.ContextStore(data) if context else m.ParametricWorld(data)
    info=json.loads((m.OUT/f'{name}.json').read_text());states=[]
    for i in info['ids']['test']:
        if context:
            item=m.base.prepare_context(source,tok,i);adj=source.adj(i);nodes=source.node_vocab_ids(i,tok)
        else:
            row=source.render_query(i,'dfs_state',tok);nodes=source.node_vocab_ids(tok);adj=source.adj()
            header=[m.base.world_token(cfg,tok,data)] if cfg.train.world_roots else []
            item=dict(tokens=header+row.tokens,prompt_len=row.prompt_len+len(header))
        tokenadj={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(adj)}
        seen=set();kept=set()
        for ri,r in enumerate(m.records(item['tokens'],item['prompt_len'],tok)):
            a,b,c=r['separators'];cur=r['frontier'][-1];repeat=cur in seen;seen.add(cur)
            if r['action'][0]==tok.id('<FOUND>') or b>=4096:continue
            expected=tokenadj[cur]-set(r['visited'])
            assert expected==set(r['candidates'])
            bucket=('repeat' if repeat else 'fresh')+('_nonempty' if expected else '_empty')
            if bucket in kept:continue
            kept.add(bucket)
            states.append(dict(query_id=i,record=ri+1,start=r['start'],prompt_len=item['prompt_len'],
                prefix=item['tokens'][:b+1],node_set=set(nodes),expected=expected,
                expected_ordered=r['candidates'],node_names={v:f'V{j}' for j,v in enumerate(nodes)},
                current=cur,visited=set(r['visited']),adjacency=tokenadj[cur],bucket=bucket))
    results={};verification=[]
    for mode in ('full','no_old_cot'):
        rows=[]
        for lo in range(0,len(states),8):
            batch=states[lo:lo+8];generated,identity=decode(model,tok,batch,mode)
            if identity:verification.append(identity)
            for s,g in zip(batch,generated):
                ys=g['tokens'];actual=set(ys);expected=s['expected']
                rows.append(dict(query_id=s['query_id'],record=s['record'],bucket=s['bucket'],
                    set_correct=bool(g['closed'] and len(ys)==len(actual) and actual==expected),closed=g['closed'],
                    duplicate=len(ys)!=len(actual),nonedge=bool(actual-s['adjacency']),visited_output=bool(actual&s['visited']),
                    missing=bool(expected-actual),extra=bool(actual-expected),
                    first_legal=(ys[0] in expected) if ys else bool(g['closed'] and not expected),
                    current=s['node_names'][s['current']],
                    expected_names=[s['node_names'][v] for v in s['expected_ordered']],
                    generated_names=[s['node_names'].get(v,tok.name(v)+'[outside graph]') for v in ys],
                    expected=sorted(expected),generated=ys,error=g['error']))
        results[mode]=dict(overall=summarize(rows),
            by_bucket={bucket:summarize([r for r in rows if r['bucket']==bucket]) for bucket in sorted({s['bucket'] for s in states})},rows=rows)
        print(name,mode,results[mode]['overall'],flush=True)
    # Root-state identity: there are no old CoT keys to remove.
    for s,a,b in zip(states,results['full']['rows'],results['no_old_cot']['rows']):
        if s['start']==s['prompt_len']:assert a['generated']==b['generated'] and a['closed']==b['closed']
    assert all(x['argmax_drift']==0 for x in verification)
    result=dict(name=name,run=str(folder),checkpoint_step=step,
        checkpoint_sha256=hashlib.sha256((folder/'checkpoints/best.pt').read_bytes()).hexdigest(),
        precision='fp32',selection='Same20 heldout queries; first state per fresh/repeat x empty/nonempty bucket, excluding FOUND; candidate prefix <=4096.',
        definition='Generate the complete candidate field until its own |; never supply gold candidate tokens. Input current stack+visited is gold.',
        intervention='Preserve tokens and absolute positions; every current-record attention row in every layer cannot read old CoT keys; original graph/query/world and current record remain visible.',
        identity_checks=verification,results=results)
    del model;torch.cuda.empty_cache();return result

if __name__=='__main__':
    torch.set_num_threads(2);all_results={}
    for name in NAMES:
        all_results[name]=main(name)
        OUT.write_text(json.dumps(all_results,indent=2)+'\n')
