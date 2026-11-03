"""N100 simple/hard fixed graphs: answers, progress, layers, structural readouts.

Frozen best checkpoints. All query phases stay in their original split.
Structural probes additionally hold out BOTH endpoint identities. No future
answer tokens enter model activations. No activations or probe weights saved.
"""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch

ROOT=Path.cwd();HERE=ROOT/'studies/fixed100_layer_probes'
sys.path[:0]=[str(ROOT/'src'),str(HERE)]
import linear
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.hard_graph import shortest_distances
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

PHASES=('prompt','k1','k2','k4','k8','p25','p50','p75','end')
TARGETS=('rank_band','topological_rank','depth_from_sources','height_to_sinks',
         'in_degree','out_degree','ancestor_count','descendant_count')
RUNS={
 'simple':ROOT/'outputs/fixed_short3_v1/launches/short3_n100_s0/all_hops',
 'hard':ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/fixed/n100/stage_b'}


def write(p,obj):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(p)


def attrs(adj,width):
    n=len(adj);ind=np.zeros(n);depth=np.zeros(n);height=np.zeros(n)
    ancestors=[set() for _ in adj];descendants=[set() for _ in adj]
    for u in range(n):
        for v in adj[u]:
            ind[v]+=1;depth[v]=max(depth[v],depth[u]+1)
            ancestors[v]|={u}|ancestors[u]
    for u in reversed(range(n)):
        for v in adj[u]:
            height[u]=max(height[u],height[v]+1);descendants[u]|={v}|descendants[v]
    return np.column_stack((np.arange(n)//width,np.arange(n),depth,height,ind,
        [len(x) for x in adj],list(map(len,ancestors)),list(map(len,descendants)))).astype(np.float32)


def phase_positions(length):
    out={'prompt':0}
    for k in (1,2,4,8):
        if length>=k:out[f'k{k}']=k
    if length:
        for p in (25,50,75):out[f'p{p}']=max(1,int(np.ceil(length*p/100)))
        out['end']=length
    return out


def summarize_predictions(labels,scores,threshold,meta,ids,goal_seen=None):
    result={'all':linear.classification(labels[ids],scores[ids],threshold)}
    if goal_seen is not None:
        subset=ids[~goal_seen[ids]]
        result['goal_not_seen']=linear.classification(labels[subset],scores[subset],threshold)
    for lo,hi,name in ((1,1,'h1'),(2,2,'h2'),(3,4,'h3_4'),(5,6,'h5_6'),(7,999,'h7plus')):
        sel=ids[(meta['hops'][ids]>=lo)&(meta['hops'][ids]<=hi)&(labels[ids]==1)]
        result[name]=linear.classification(labels[sel],scores[sel],threshold)
    return result


@torch.inference_mode()
def collect(setting,protocol):
    folder=RUNS[setting]/protocol;cfg=load_config([folder/'resolved_config.yaml'])
    tok=Tokenizer.from_config(cfg);world=ParametricWorld(Path(cfg.data.root)/f'{cfg.data.graph_shape}-seed{cfg.data.graph_seed}')
    adj=world.adj();q=world.queries;nodes=world.node_vocab_ids(tok)
    width=next(iter(cfg.data.shapes.values())).width
    selected=np.flatnonzero(q['t_topo']//width>q['s_topo']//width)
    src=q['s_topo'][selected].astype(int);tgt=q['t_topo'][selected].astype(int)
    meta=dict(query_ids=selected,source=src,target=tgt,label=q['label'][selected].astype(int),
              split=q['split'][selected].astype(int),hops=np.asarray([shortest_distances(adj,int(s))[int(t)] for s,t in zip(src,tgt)]))
    blob=torch.load(folder/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    model.load_state_dict(blob['model'],strict=True)
    header=[]
    if cfg.train.world_roots:
        worlds=[Path(p).resolve() for p in cfg.train.world_roots]
        header=[tok.id(f'<WORLD_{worlds.index(world.root.resolve())}>')]
    prompts=[header+[tok.id('<QUERY>'),nodes[s],nodes[t],tok.id('|')] for s,t in zip(src,tgt)]
    plen=len(prompts[0]);seqs=[];gens=[];inverse={v:u for u,v in enumerate(nodes)}
    for start in range(0,len(prompts),32):
        results=generate(model,prompts[start:start+32],protocol,tok,cfg.train.precision,
                         max_new_tokens=0,cot_margin=cfg.protocol.cot_margin)
        for j,g in enumerate(results):
            i=start+j;trace=list(g['trace_tokens'])
            seqs.append(prompts[i]+trace)
            gens.append(dict(pred=-1 if g['pred'] is None else int(g['pred']),trace=trace,
                             interrupted=bool(g.get('cot_interrupted')),limit=bool(g['generation_limit_reached'])))
        if start%512==0:print(setting,protocol,'generate',start,'/',len(prompts),flush=True)
    h=cfg.model.layers+1;d=cfg.model.d_model;count=len(prompts)
    structure=np.empty((count,3,h,d),np.float32)
    phase_x=np.full((count,len(PHASES),h,d),np.nan,np.float32)
    phase_mask=np.zeros((count,len(PHASES)),bool)
    goal_seen=np.zeros_like(phase_mask);observable=np.zeros((count,len(PHASES),4),np.float32)
    current=np.zeros((count,len(PHASES)),int);lens=np.full((count,len(PHASES),h),-1,int)
    embed=model.embed.weight.detach().float().cpu().numpy();layer_names=['E']+[f'L{i}' for i in range(1,h)]
    for start in range(0,count,64):
        rows=seqs[start:start+64];maximum=max(map(len,rows))
        tensor=torch.full((len(rows),maximum),tok.pad,device='cuda',dtype=torch.long)
        for b,row in enumerate(rows):tensor[b,:len(row)]=torch.tensor(row,device='cuda')
        capture={}
        with torch.autocast('cuda',dtype=torch.bfloat16):model(tensor,capture=capture)
        hidden=[model.embed(tensor)]+capture['resid']
        stacked=torch.stack([x.float() for x in hidden])
        all_logits=model.lm_head(model.final_norm(stacked))
        lens_cpu=(all_logits[...,tok.yes]>all_logits[...,tok.no]).cpu().numpy().astype(int)
        hidden_cpu=stacked.cpu().numpy()
        for b,row in enumerate(rows):
            i=start+b;positions=[plen-3,plen-2,plen-1]
            structure[i]=hidden_cpu[:,b,positions,:].transpose(1,0,2)
            anchors=phase_positions(len(row)-plen) if protocol=='path' else {'prompt':0}
            for phase,k in anchors.items():
                ip=PHASES.index(phase);pos=plen-1+k
                phase_x[i,ip]=hidden_cpu[:,b,pos,:]
                phase_mask[i,ip]=True;current[i,ip]=row[pos]
                seen=nodes[tgt[i]] in row[plen:pos+1];goal_seen[i,ip]=seen
                observable[i,ip]=[k/100,pos/100,float(seen),float(row[pos]==nodes[tgt[i]])]
                lens[i,ip]=lens_cpu[:,b,pos]
        del hidden,capture,tensor,stacked,all_logits
    node_embed=embed[nodes];attributes=attrs(adj,width)
    manifest=dict(setting=setting,protocol=protocol,run=str(folder),checkpoint_step=int(blob['step']),
        checkpoint_sha256=hashlib.sha256((folder/'checkpoints/best.pt').read_bytes()).hexdigest(),
        n_nodes=world.n_nodes,n_edges=cfg.data.n_edges,max_steps=cfg.train.max_steps,
        max_seq_len=cfg.model.max_seq_len,world_prefix_tokens=len(header),
        model=dict(layers=cfg.model.layers,heads=cfg.model.heads,d_model=d,d_ff=cfg.model.d_ff),
        query_ratios=list(cfg.data.query_ratios),walks=world.n_walks(),
        reverse_queries_excluded=int(len(q['label'])-count),
        split_counts={str(s):int((meta['split']==s).sum()) for s in (0,1,2)},
        phase_convention='k counts generated tokens; percentages are retrospective trajectory anchors, not causal stopping rules',
        rank_band_convention='true supplied layer for simple; hidden generator rank band for hard, not an intrinsic DAG layer')
    p=HERE/f'{setting}_{protocol}_queries.jsonl.gz'
    with gzip.open(p,'wt') as f:
        for i,g in enumerate(gens):
            f.write(json.dumps({k:int(v[i]) for k,v in meta.items()}|g)+'\n')
    del model,blob;torch.cuda.empty_cache()
    return dict(meta=meta,structure=structure,phase_x=phase_x,phase_mask=phase_mask,
                goal_seen=goal_seen,observable=observable,current=current,embed=embed,
                node_embed=node_embed,attributes=attributes,layers=layer_names,lens=lens,
                manifest=manifest,gens=gens)


def answer_analysis(pack):
    m=pack['meta'];y=m['label'];result={}
    native=np.asarray([g['pred'] for g in pack['gens']])
    test=np.flatnonzero(m['split']==2)
    native_summary={'all':dict(n=len(test),accuracy=float((native[test]==y[test]).mean()))}
    for hop in sorted(set(m['hops'][test])-{ -1 }):
        ids=test[(m['hops'][test]==hop)&(y[test]==1)]
        native_summary[f'h{hop}']=dict(n=len(ids),accuracy=float((native[ids]==1).mean()))
    negative=test[y[test]==0];native_summary['negative']=dict(n=len(negative),accuracy=float((native[negative]==0).mean()))
    pair=np.concatenate((pack['node_embed'][m['source']],pack['node_embed'][m['target']]),axis=1)
    for ip,phase in enumerate(PHASES):
        valid=pack['phase_mask'][:,ip]
        if not valid.any():continue
        splits=[np.flatnonzero(valid&(m['split']==s)) for s in (0,1,2)]
        if any(len(x)<4 or len(np.unique(y[x]))<2 for x in splits):continue
        x=pack['phase_x'][valid,ip];indices=np.flatnonzero(valid);inverse={v:i for i,v in enumerate(indices)}
        local_split=[np.asarray([inverse[i] for i in p]) for p in splits]
        local_meta={k:v[valid] for k,v in m.items()};local_y=y[valid]
        features={'residual':x,'input_pair':pair[valid],
                  'current_embedding':pack['embed'][pack['current'][valid,ip]],
                  'position_goal_control':pack['observable'][valid,ip]}
        entry={}
        for name,features_x in features.items():
            pred,fit=linear.fit(features_x,local_y,local_split,binary=True)
            entry[name]=dict(fit=fit,layers=[])
            for layer in range(pred.shape[1]):
                entry[name]['layers'].append(summarize_predictions(local_y,pred[:,layer,0],fit['threshold'][layer][0],local_meta,local_split[2],pack['goal_seen'][valid,ip]))
        # Query-level shuffled labels preserve all graph/split/phase boundaries.
        random=np.random.default_rng(181+ip);shuffled=local_y.copy()
        for ids in local_split:shuffled[ids]=random.permutation(shuffled[ids])
        pred,fit=linear.fit(x,shuffled,local_split,binary=True)
        entry['shuffled_labels']=[linear.classification(shuffled[local_split[2]],pred[local_split[2],l,0],fit['threshold'][l][0]) for l in range(pred.shape[1])]
        entry['uncalibrated_logit_lens']=[linear.classification(y[splits[2]],pack['lens'][splits[2],ip,l],.5) for l in range(len(pack['layers']))]
        entry['test_goal_seen']=int(pack['goal_seen'][splits[2],ip].sum())
        result[phase]=entry
        print(pack['manifest']['setting'],pack['manifest']['protocol'],'answer probes',phase,flush=True)
    return dict(native_final=native_summary,linear_readout=result)


def regression_report(features,labels,split):
    p,fit=linear.fit(features,labels,split)
    return dict(fit=fit,layers=[{target:linear.regression(labels[split[2],k],p[split[2],l,k]) for k,target in enumerate(TARGETS)} for l in range(p.shape[1])])


def structural_analysis(pack):
    m=pack['meta'];attributes=pack['attributes'];query_split=[np.flatnonzero(m['split']==s) for s in (0,1,2)]
    roles={'source_token':(0,m['source']),'target_token':(1,m['target']),
           'bar_source':(2,m['source']),'bar_target':(2,m['target'])}
    result={'query_disjoint':{},'node_disjoint':[]}
    for name,(pos,nodes) in roles.items():
        result['query_disjoint'][name]=regression_report(pack['structure'][:,pos],attributes[nodes],query_split)
    rng=np.random.default_rng(2026);random_embed=rng.normal(size=pack['node_embed'].shape).astype(np.float32)
    random_embed*=pack['node_embed'].std()/random_embed.std()
    for seed in (17,31,47):
        perm=np.random.default_rng(seed).permutation(100);node_split=[perm[:60],perm[60:80],perm[80:]]
        split=[np.flatnonzero(np.isin(m['source'],ids)&np.isin(m['target'],ids)) for ids in node_split]
        assert not set(node_split[0])&set(node_split[2])
        fold=dict(seed=seed,node_ids=[x.tolist() for x in node_split],query_counts=list(map(len,split)),
            embedding=regression_report(pack['node_embed'],attributes,node_split),
            random_embedding=regression_report(random_embed,attributes,node_split),roles={})
        shuffled=attributes[np.random.default_rng(seed+88).permutation(100)]
        fold['shuffled_node_attributes']=regression_report(pack['node_embed'],shuffled,node_split)
        for name,(pos,nodes) in roles.items():
            fold['roles'][name]=regression_report(pack['structure'][:,pos],attributes[nodes],split)
            # Node lookup gets perfect answers when identities overlap. Held-out
            # nodes have no table entry and can only get the training-node mean.
            fold['roles'][name]['unseen_identity_mean_baseline']={t:linear.regression(attributes[nodes[split[2]],k],
                np.full(len(split[2]),attributes[node_split[0],k].mean())) for k,t in enumerate(TARGETS)}
        result['node_disjoint'].append(fold)
        print(pack['manifest']['setting'],pack['manifest']['protocol'],'structure fold',seed,flush=True)
    return result


def main():
    HERE.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    write(HERE/'status.json',dict(phase='running',pid=os.getpid(),started=time.time()))
    for setting in ('simple','hard'):
        for protocol in ('direct','path'):
            pack=collect(setting,protocol)
            result=dict(manifest=pack['manifest'],layers=pack['layers'],targets=list(TARGETS),
                        answers=answer_analysis(pack),structure=structural_analysis(pack))
            write(HERE/f'{setting}_{protocol}.json',result)
            del pack,result;torch.cuda.empty_cache()
    write(HERE/'status.json',dict(phase='complete',pid=os.getpid(),finished=time.time()))


if __name__=='__main__':main()
