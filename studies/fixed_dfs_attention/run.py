"""Read-only selected attention rows for fixed/context DFS state, GPU0.

Distance baselines use validation only. Reference and native legal prefixes
are reported separately. No activation/attention matrices or weights saved.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import os,sys,json,hashlib,importlib.util
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));OUT=ROOT/'studies/fixed_dfs_attention'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies/fixed100_legality')]
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ContextStore,ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.data.mistakes import records
from cot_navigation.evaluation.sampling import stratified_ids
from replay import StateReplay
from stream import generate_checked
from cache_prefix import enable
spec=importlib.util.spec_from_file_location('attention_reader',ROOT/'studies/attention_benefit/run.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
FOLDERS={
 'simple_fixed':ROOT/'outputs/fixed_short3_v1/launches/short3_n100_s0/all_hops/dfs_state',
 'hard_fixed':ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/fixed/n100/stage_b/dfs_state',
 'simple_context':ROOT/'outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state',
 'hard_context':ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state'}
GROUPS=['input_graph','input_query','current_frontier','current_visited','current_candidates','current_other','previous_record','older_cot']
EXTRA=['visited_current','visited_other','frontier_current','current_node_any_cot','query_target','graph_related_dest','graph_related_successor_marker']

def make_case(row,tok,query_id,graph_end,valid_count=None):
    recs=records(row,tok)
    if valid_count is not None:recs=recs[:valid_count]
    eligible=[i for i,r in enumerate(recs) if r['action']!=[tok.id('<FOUND>')] and r['action_start']<=4096]
    rng=np.random.default_rng(131071+query_id)
    selected=sorted(rng.choice(eligible,min(4,len(eligible)),replace=False).tolist())
    probes=[]
    for ri in selected:
        rec=recs[ri];vlo,_=rec['spans']['visited'];clo,_=rec['spans']['cands']
        roles=[('frontier_first',rec['start']),('visited_first',vlo),('action',rec['action_start'])]
        if rec['cands']:roles.append(('candidate_nonempty',clo))
        for role,pos in roles:
            probes.append(dict(role=M.ROLES.index(role),q=pos-1,record=rec,cur=rec['stack'][-1],
                               previous=recs[ri-1] if ri else None))
    if not probes:return None
    end=max(p['q'] for p in probes)+1
    return dict(query_id=int(query_id),tokens=row.tokens[:end],prompt_len=row.prompt_len,
                graph_end=graph_end,probes=probes)

def masks(case,p):
    q=p['q'];keys=np.arange(q+1);tokens=np.asarray(case['tokens'][:q+1]);rec=p['record'];prev=p['previous']
    flo=rec['start'];fhi=flo+len(rec['stack']);vlo,vhi=rec['spans']['visited'];clo,chi=rec['spans']['cands']
    graph=keys<case['graph_end'];query=(keys>=case['graph_end'])&(keys<case['prompt_len'])
    frontier=(keys>=flo)&(keys<fhi);visited=(keys>=vlo)&(keys<vhi);cands=(keys>=clo)&(keys<chi)
    current=(keys>=flo);other=current&~(frontier|visited|cands)
    previous=(keys>=prev['start'])&(keys<prev['end']) if prev else np.zeros_like(keys,dtype=bool)
    older=(keys>=case['prompt_len'])&(keys<flo)&~previous
    groups=[graph,query,frontier,visited,cands,other,previous,older]
    assert np.all(np.stack(groups).sum(0)==1)
    related=[];markers=[]
    if case['graph_end']:
        for start in range(0,case['graph_end'],3):
            if case['tokens'][start+1]==p['cur']:
                related.append(start+2);markers.append(start+3)
    extra=[visited&(tokens==p['cur']),visited&(tokens!=p['cur']),frontier&(tokens==p['cur']),
           (keys>=case['prompt_len'])&(tokens==p['cur']),keys==case['prompt_len']-2,
           np.isin(keys,related),np.isin(keys,markers)]
    return groups,extra

def analyze(model,cases,mu,count):
    sums={};ns={};reader=M.SelectedRows(model)
    try:
        for lo in range(0,len(cases),4):
            batch=cases[lo:lo+4];weights=reader.forward(batch)
            for bi,c in enumerate(batch):
                local={}
                for pi,p in enumerate(c['probes']):
                    role=M.ROLES[p['role']];q=p['q'];distance=q-np.arange(q+1)
                    groups,extras=masks(c,p);allmasks=groups+extras
                    values=np.full((6,len(allmasks),3),np.nan)
                    for li,w in weights.items():
                        a=w[bi,0,pi,:q+1];covered=count[p['role'],distance]>0
                        residual=a-mu[li,0,p['role'],distance]
                        for k,mask in enumerate(allmasks):
                            values[li,k,0]=a[mask].sum()
                            supported=mask&covered
                            if supported.any():values[li,k,1]=residual[supported].mean()
                            values[li,k,2]=supported.sum()/mask.sum() if mask.any() else np.nan
                    local.setdefault(role,[]).append(values)
                for role,values in local.items():
                    with np.errstate(invalid='ignore'):
                        v=np.nanmean(values,axis=0)
                    if role not in sums:sums[role]=np.zeros_like(v);ns[role]=np.zeros_like(v)
                    valid=np.isfinite(v);sums[role][valid]+=v[valid];ns[role][valid]+=1
    finally:reader.close()
    result={}
    for role in sums:
        a=np.divide(sums[role],ns[role],out=np.zeros_like(sums[role]),where=ns[role]>0)
        result[role]=dict(raw_mass=a[:,:,0].tolist(),distance_centered_density=a[:,:,1].tolist(),
                          baseline_support_fraction=a[:,:,2].tolist(),query_counts=ns[role][:,:,0].tolist())
    return dict(n_queries=len(cases),n_probes=sum(len(c['probes']) for c in cases),roles=result,
                readonly_identity_equal=reader.checked)

def main(name,folder):
    cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    context=cfg.data.knowledge=='context_graph'
    source=ContextStore(cfg.train.scale_roots[-1] if cfg.train.scale_roots else cfg.data.root) if context else ParametricWorld(Path(cfg.data.root)/f'{cfg.data.graph_shape}-seed{cfg.data.graph_seed}')
    arrays=source.arrays if context else source.queries;cache=ScalePreparedRows(source,cfg,tok)
    header=[] if context or not cfg.train.world_roots else [tok.id('<WORLD_0>')]
    def get_row(i):
        row=cache.row(int(i),0)
        if header:row=replace(row,tokens=header+row.tokens,prompt_len=row.prompt_len+1,answer_index=row.answer_index+1)
        return row
    cases={};ids_by_split={}
    for split in ('val','test'):
        ids=source.indices(split)
        if not context:ids=np.asarray([i for i in ids if int(arrays['t_topo'][i])//10>int(arrays['s_topo'][i])//10])
        shapes=arrays['shape_id'] if context else np.zeros(len(arrays['label']),np.int8)
        ids=stratified_ids(ids,shapes,arrays['distance'],arrays['label'],64,20261026)
        ids_by_split[split]=list(map(int,ids));pool=[]
        for i in ids:
            row=get_row(i);c=make_case(row,tok,int(i),row.prompt_len-4 if context else 0)
            if c:pool.append(c)
        cases[split]=pool
    assert not set(ids_by_split['val'])&set(ids_by_split['test'])
    checkpoint=folder/'checkpoints/best.pt';blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    model.load_state_dict(blob['model']);step=int(blob['step']);del blob
    mu,count=M.calibrate(model,cases['val'],batch_size=4)
    result=dict(name=name,run=str(folder),step=step,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                data_root=str(source.root),groups=GROUPS+EXTRA,calibration_queries=len(cases['val']),
                reference=analyze(model,cases['test'],mu,count))
    # Independent native-prefix diagnostic: first two queries per gap/label,
    # chosen before observing output; stop at first proven semantic error.
    chosen=[];buckets={}
    for i in ids_by_split['test']:
        key=(int(arrays['distance'][i]),int(arrays['label'][i]));buckets.setdefault(key,[]).append(i)
    for key in sorted(buckets):chosen.extend(buckets[key][:1])
    prompts=[];replays=[]
    for i in chosen:
        row=get_row(i);prompts.append(row.tokens[:row.prompt_len])
        adj=source.adj(i) if context else source.adj();nodes=source.node_vocab_ids(i,tok) if context else source.node_vocab_ids(tok)
        tokenadj={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(adj)}
        s,t=int(arrays['s_topo'][i]),int(arrays['t_topo'][i]);replays.append(StateReplay(tokenadj,nodes[s],nodes[t],tok))
    enable(model);outputs=generate_checked(model,prompts,tok,cfg.train.precision,replays,cfg.protocol.cot_margin)
    native=[]
    for i,prompt,out in zip(chosen,prompts,outputs):
        valid=out['audit']['prefix_valid_records'];body=out['trace']
        # records() expects complete records only: omit first invalid/partial
        # suffix, using record boundaries before parsing.
        boundaries=[j+1 for j,v in enumerate(body) if v==tok.id(';')]
        end=boundaries[valid-1] if valid else 0
        row=SimpleNamespace(tokens=prompt+body[:end],prompt_len=len(prompt),answer_index=len(prompt)+end)
        c=make_case(row,tok,i,len(prompt)-4 if context else 0)
        if c:native.append(c)
    result['native_legal_prefix']=analyze(model,native,mu,count)
    result['native_panel']=dict(n=len(chosen),semantic_valid=sum(o['audit']['semantic_valid'] for o in outputs),
                               censored=sum(o['early_censored'] for o in outputs),ids=chosen)
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(name,'done',step,'reference',result['reference']['n_queries'],'native',len(native),flush=True)
    del model,cache;torch.cuda.empty_cache()

if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    for name,folder in FOLDERS.items():
        if not (OUT/f'{name}.json').exists():main(name,folder)
