"""Before-emission layerwise node tendencies and causal within-token correction.

Frozen checkpoints and unedited natural prefixes. No already-emitted bad token
is supplied at the investigated position. Current-token legality is the endpoint.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'2')
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
OUT=P/'current'
R=runpy.run_path(str(P.parent/'natural_history_probes/run.py'))
import numpy as np
import torch
from torch.nn import functional as F
from cot_navigation.analysis.within_token import (candidate_slots,cached_tail,lens_logits,fit_lens,
    fit_binary_heads,predict_binary,binary_accuracy,protected_pair)
from cot_navigation.analysis.residual_control import steer,orthogonal_axis
from cot_navigation.evaluation.rollout import pad_rows
from cot_navigation.experiments.artifacts import atomic_json,sha_file

KINDS=['raw','calibrated'];LAYERS=[2,3,4,5]


def prepare(am,cases):
    """4 node-required + 1 closure slots/query, selected without receiver logits."""
    rows={};graphs={};rng=np.random.default_rng(20261101)
    for split in ['train','val','test']:
        pool=defaultdict(dict);onsets=defaultdict(list)
        for c in cases[split]:
            q=c['query']
            if q not in graphs:graphs[q]=R['PREP']['graph_tokens'](am,q)[0]
            for slot in candidate_slots(c,graphs[q],am.tok,include_closure=True):
                i=slot['pos'];key=hashlib.sha256(np.asarray(c['trace'][:i],np.int32).tobytes()).hexdigest()
                if key not in pool[q] or (slot['natural_onset'] and not pool[q][key]['natural_onset']):
                    pool[q][key]=dict(slot,query=q,sid=c['sid'],prefix_hash=key,
                        next_from_source=c['trace'][i],source=c['source'],split=split,
                        native_error_kind=c['first']['kind'] if slot['natural_onset'] else None)
                if slot['natural_onset']:onsets[q].append(key)
        selected=[]
        for q in sorted(pool):
            values=list(pool[q].values())
            chosen=[]
            for node_needed,count in [(True,4),(False,1)]:
                choices=[r for r in values if r['requires_node']==node_needed]
                order=rng.permutation(len(choices))[:count]
                chosen.extend(choices[j] for j in order)
            by_key={r['prefix_hash']:dict(r,populations=['main']) for r in chosen}
            if onsets[q]:
                key=sorted(set(onsets[q]))[int(rng.integers(len(set(onsets[q]))))]
                if key in by_key:by_key[key]['populations'].append('natural_onset')
                else:by_key[key]=dict(pool[q][key],populations=['natural_onset'])
            selected.extend(by_key.values())
        rows[split]=selected
    tq=sorted({r['query'] for r in rows['train']});vq=sorted({r['query'] for r in rows['val']})
    rng.shuffle(tq);rng.shuffle(vq);lens_train=set(tq[:512]);lens_val=set(vq[:128])
    parts={name:[] for name in ['lens_train','probe_train','lens_val','probe_val','test']}
    for split,rr in rows.items():
        for r in rr:
            part=('lens_train' if r['query'] in lens_train else 'probe_train') if split=='train' else (
                ('lens_val' if r['query'] in lens_val else 'probe_val') if split=='val' else 'test')
            parts[part].append(r)
    sets={k:{r['query'] for r in rr} for k,rr in parts.items()}
    assert all(not sets[a]&sets[b] for i,a in enumerate(sets) for b in list(sets)[i+1:])
    return parts


def sequences(am,cases,rows):
    seq=[]
    for r in rows:
        c=cases[r['split']][r['sid']]
        # Exclusive boundary: native current token is not fed to the model.
        prefix=am.example(r['query'],r['split']).prompt+c['trace'][:r['pos']]
        assert r['pos']>0 and len(prefix)<am.model.max_seq_len
        assert c['tau'] is None or r['pos']<=c['tau']
        seq.append(prefix)
    return seq


@torch.no_grad()
def extract(am,cases,rows,batch):
    seq=sequences(am,cases,rows);n=len(rows);vocab=len(am.tok);dim=am.model.d_model
    x=np.zeros((n,6,dim),np.float32);logits=np.zeros((n,vocab),np.float32)
    control=np.zeros((n,dim+4),np.float32)
    for start in range(0,n,batch):
        chunk=seq[start:start+batch];ids,lengths=pad_rows(chunk,am.tok.pad,am.device);cap={}
        z=am.model(ids,capture=cap);b=torch.arange(len(chunk),device=am.device);pos=lengths-1
        x[start:start+len(chunk)]=torch.stack([h[b,pos] for h in cap['resid']],1).cpu().numpy()
        logits[start:start+len(chunk)]=z[b,pos].cpu().numpy()
        control[start:start+len(chunk),:dim]=am.model.embed(ids[b,pos]).cpu().numpy()
        for j,r in enumerate(rows[start:start+len(chunk)]):
            control[start+j,dim:]=[len(chunk[j])/4096,r['depth']/100,r['visited']/100,r['printed']/100]
        if start%(batch*32)==0:print('FEATURES',start,'/',n,flush=True)
    allowed=np.zeros((n,vocab),bool)
    for i,r in enumerate(rows):allowed[i,r['allowed']]=True
    return dict(h=x,logits=logits,control=control,allowed=allowed,
        query=np.array([r['query'] for r in rows]),requires_node=np.array([r['requires_node'] for r in rows]))


@torch.no_grad()
def decoded(am,lens,data):
    predictions={};labels=[];masks=[]
    for kind in KINDS:
        tokens=[];nodes=[]
        for layer in range(6):
            z=lens_logits(am.model,lens,torch.as_tensor(data['h'][:,layer],device=am.device),layer,kind)
            tokens.append(z.argmax(-1).cpu().numpy())
            nodes.append((z[:,am.tok.node_lo:am.tok.node_lo+am.tok.n_node_tokens].argmax(-1)+am.tok.node_lo).cpu().numpy())
        token=np.stack(tokens,1);node=np.stack(nodes,1)
        valid=data['allowed'][np.arange(len(token))[:,None],token]
        node_valid=data['allowed'][np.arange(len(node))[:,None],node]
        predictions[kind]=dict(token=token,node=node,valid=valid,node_valid=node_valid)
        for layer in LAYERS:
            # Primary detector asks whether a genuine provisional NODE choice
            # is illegal while another node is required, not a grammar cue.
            mask=data['requires_node']&(token[:,layer]>=am.tok.node_lo)&(token[:,layer]<am.tok.node_lo+am.tok.n_node_tokens)
            labels.append(~valid[:,layer]);masks.append(mask)
    # Primary prospective target: the CURRENT not-yet-emitted token's actual
    # final-layer legality, shared across layers. No bad token enters the input.
    actual=data['logits'].argmax(-1)
    future_error=~data['allowed'][np.arange(len(actual)),actual]
    for layer in LAYERS:
        labels.append(future_error);masks.append(data['requires_node'])
    return predictions,np.stack(labels,1),np.stack(masks,1)


def binary_data(data,y,mask):
    n=len(y);heads=y.shape[1];dim=data['h'].shape[-1]
    x=np.zeros((n,heads*2,dim+4),np.float32)
    for head in range(heads):x[:,head,:dim]=data['h'][:,LAYERS[head%len(LAYERS)]]
    x[:,heads:]=data['control'][:,None]
    return dict(x=x,y=np.tile(y,(1,2)),mask=np.tile(mask,(1,2)),query=data['query'])


def probe_reports(fitted,bd,rows,architecture):
    z=predict_binary(fitted,bd['x']);out=[]
    for head in range(12):
        for control in [False,True]:
            j=head+12*control
            if not fitted['valid'][j]:continue
            for pop in ['main','natural_onset']:
                keep=np.array([pop in r['populations'] for r in rows])&bd['mask'][:,j]
                m=binary_accuracy(bd['y'][keep,j],z[keep,j],bd['query'][keep])
                if m:out.append(dict(architecture=architecture,lens=KINDS[head//4] if head<8 else 'shared',
                    target='provisional' if head<8 else 'prospective',layer=LAYERS[head%4]+1,
                    control=control,population=pop,**m))
    return out


def axis_bank(fitted,bd):
    axes={};dim=bd['x'].shape[-1]-4
    for head in range(12):
        if not fitted['valid'][head]:continue
        u=(fitted['net'].w[head,:dim,0]/fitted['scale'][head,:dim]).detach().cpu().numpy()
        u=u/np.linalg.norm(u);use=bd['mask'][:,head];y=bd['y'][:,head];h=bd['x'][:,head,:dim]
        means=[float((h[use&(y==c)]@u).mean()) for c in [0,1]]
        axes[head]=dict(u=u,means=means)
    return axes


@torch.no_grad()
def one_probe(fitted,head,h):
    padded=torch.cat([h,torch.zeros((len(h),4),device=h.device)],-1)
    x=(padded-fitted['mean'][head])/fitted['scale'][head]
    p=fitted['net'];z=x@p.w[head]+p.b[head]
    return ((F.gelu(z)*p.v[head]).sum(-1)+p.c[head] if p.hidden else z[:,0]).sigmoid()


@torch.no_grad()
def interventions(am,cases,rows,lens,linear,mlp,axes,batch,seed):
    rows=[r for r in rows if r['requires_node']]
    seq=sequences(am,cases,rows);out=[];max_sham_diff=0.;sham_mismatches=0
    for start in range(0,len(rows),batch):
        rr=rows[start:start+batch];ids,lengths=pad_rows(seq[start:start+batch],am.tok.pad,am.device)
        cache=am.model.new_cache(len(rr),am.device,torch.float32);cache.valid_length=ids.shape[1]
        capture={};z=am.model(ids,cache=cache,cache_mode='prefill',capture=capture)
        b=torch.arange(len(rr),device=am.device);pos=(lengths-1)[:,None]
        original=z[b,lengths-1];allowed=torch.zeros_like(original,dtype=torch.bool)
        for i,r in enumerate(rr):allowed[i,r['allowed']]=True
        for kind in KINDS:
            for layer in [3,4,5]:
                for probe_target,head in [('provisional',KINDS.index(kind)*4+LAYERS.index(layer)),('prospective',8+LAYERS.index(layer))]:
                    if head not in axes:continue
                    h=capture['resid'][layer][b,lengths-1].float()
                    initial=lens_logits(am.model,lens,h,layer,kind)
                    tentative=initial.argmax(-1);tnode=initial[:,am.tok.node_lo:am.tok.node_lo+am.tok.n_node_tokens].argmax(-1)+am.tok.node_lo
                    sham=cached_tail(am.model,cache,h,pos,layer)
                    max_sham_diff=max(max_sham_diff,float((sham-original).abs().max()))
                    sham_mismatches+=int((sham.argmax(-1)!=original.argmax(-1)).sum())
                    u=torch.as_tensor(axes[head]['u'],device=am.device)
                    v=torch.as_tensor(orthogonal_axis(axes[head]['u'],73001+100*seed+head),device=am.device)
                    before=one_probe(mlp,head,h);lp0=one_probe(linear,head,h)
                    base_pred=sham.argmax(-1);base_p=sham.softmax(-1)
                    for condition in ['legal','error']:
                        target=axes[head]['means'][condition=='error']
                        moved=steer(h,u,target);sd=moved-h
                        amount=(sd*u).sum(-1);nd=amount[:,None]*v
                        for mode in ['unconstrained','preserve_tendency']:
                            if mode=='preserve_tendency':
                                edited,random,factor=protected_pair(am.model,lens,h,layer,kind,sd,nd,am.tok.node_lo,am.tok.n_node_tokens)
                            else:edited=h+sd;random=h+nd;factor=torch.ones(len(h),device=am.device)
                            for name,state in [(condition,edited),('random_'+condition,random)]:
                                logits=cached_tail(am.model,cache,state,pos,layer)
                                pred=logits.argmax(-1);p=logits.softmax(-1)
                                interim=lens_logits(am.model,lens,state,layer,kind)
                                ip=interim.argmax(-1);inode=interim[:,am.tok.node_lo:am.tok.node_lo+am.tok.n_node_tokens].argmax(-1)+am.tok.node_lo
                                if mode=='preserve_tendency':
                                    assert torch.equal(ip,tentative) and torch.equal(inode,tnode)
                                    if layer==5:assert torch.equal(pred,sham.argmax(-1))
                                after=one_probe(mlp,head,state);lp=one_probe(linear,head,state)
                                token_values=torch.stack([tnode,tentative,base_pred,pred,ip,inode],1).cpu().numpy()
                                floats=torch.stack([allowed[b,tentative].float(),allowed[b,base_pred].float(),allowed[b,pred].float(),
                                    (base_p*allowed).sum(-1),(p*allowed).sum(-1),before,after,lp0,lp,factor,
                                    (state-h).norm(dim=-1)/h.norm(dim=-1).clamp_min(1e-8),allowed[b,tnode].float()],1).cpu().numpy()
                                for i,r in enumerate(rr):
                                    t=token_values[i];v0=floats[i]
                                    item=dict(query=r['query'],prefix_hash=r['prefix_hash'],populations=r['populations'],
                                        lens=kind,layer=layer+1,probe_target=probe_target,condition=name,mode=mode,
                                        tentative_node=int(t[0]),tentative_token=int(t[1]),
                                        tentative_is_node=am.tok.node_lo<=int(t[1])<am.tok.node_lo+am.tok.n_node_tokens,
                                        tentative_legal=bool(v0[0]),tentative_node_legal=bool(v0[11]),tentative_preserved=int(t[4])==int(t[1]) and int(t[5])==int(t[0]),
                                        pred_before=int(t[2]),pred_after=int(t[3]),correct_before=bool(v0[1]),correct_after=bool(v0[2]),
                                        legal_mass_before=float(v0[3]),legal_mass_after=float(v0[4]),
                                        mlp_before=float(v0[5]),mlp_after=float(v0[6]),linear_before=float(v0[7]),linear_after=float(v0[8]),
                                        backtrack=float(v0[9]),relative_norm=float(v0[10]))
                                    out.append(item)
        if start%(batch*8)==0:print('INTERVENTIONS',start,'/',len(rows),flush=True)
        del cache,capture,z,ids
    assert sham_mismatches==0,'Cached counterfactual sham changed actual next-token argmax'
    return out,dict(cached_sham_max_logit_difference=max_sham_diff,cached_sham_argmax_mismatches=sham_mismatches,
        no_current_target_in_input=True,protected_tendency_and_final_layer_zero=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lens-steps',type=int,default=2000)
    parser.add_argument('--probe-steps',type=int,default=5000)
    parser.add_argument('--batch',type=int,default=16)
    args=parser.parse_args();os.chdir(ROOT);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.32)
    (OUT/'results').mkdir(parents=True,exist_ok=True)
    identity=dict(vars(args),gpu=0,precision='fp32',threshold=.5,max_relative_norm=.2,
        sources={str(p.relative_to(ROOT)):sha_file(p) for p in [Path(__file__),ROOT/'src/cot_navigation/analysis/within_token.py',ROOT/'src/cot_navigation/analysis/residual_control.py']})
    atomic_json(OUT/'plan.json',identity);cases=parts=None
    for seed in range(4):
        for arm in ['matched_clean','correction']:
            dest=OUT/'results'/f'{arm}_s{seed}.json.gz'
            if dest.exists():
                old=R['PREP']['read_cache'](dest)
                if old.get('complete'):
                    assert old['identity']==identity;continue
            started=time.time();am=R['Model'](arm,seed)
            if cases is None:
                cases,_,_=R['load_data'](am.tok);parts=prepare(am,cases)
                atomic_json(OUT/'selection.json',dict(parts=parts,counts={k:dict(rows=len(rr),queries=len({r['query'] for r in rr})) for k,rr in parts.items()}))
                print('COUNTS',{k:len(v) for k,v in parts.items()},flush=True)
            features={k:extract(am,cases,rr,args.batch) for k,rr in parts.items()}
            lens=fit_lens(am.model,features['lens_train']['h'],features['lens_train']['logits'],
                features['lens_val']['h'],features['lens_val']['logits'],args.lens_steps)
            print('LENSES FIT',arm,seed,flush=True)
            dec={};bd={}
            for name in ['probe_train','probe_val','test']:
                dec[name],y,mask=decoded(am,lens,features[name]);bd[name]=binary_data(features[name],y,mask)
            linear=fit_binary_heads(bd['probe_train'],bd['probe_val'],steps=args.probe_steps,seed=0)
            mlp=fit_binary_heads(bd['probe_train'],bd['probe_val'],hidden=64,steps=args.probe_steps,seed=1)
            axes=axis_bank(linear,bd['probe_train'])
            result=dict(complete=False,arm=arm,model_seed=seed,identity=identity,checkpoint=am.checkpoint,
                calibration={k:v for k,v in lens.items() if k not in ['a','b']},
                detector_support=linear['supports'],detector_valid=linear['valid'],probes=[],transitions=[],interventions=[])
            for arch,f in [('linear',linear),('mlp64',mlp)]:result['probes']+=probe_reports(f,bd['test'],parts['test'],arch)
            f=features['test'];actual=f['logits'].argmax(-1)
            for kind in KINDS:
                d=dec['test'][kind]
                assert np.array_equal(d['token'][:,-1],actual)
                for i,r in enumerate(parts['test']):
                    result['transitions'].append(dict(query=r['query'],prefix_hash=r['prefix_hash'],populations=r['populations'],
                        requires_node=r['requires_node'],allowed=r['allowed'],native_next=r['next_from_source'],
                        lens=kind,provisional=d['token'][i].tolist(),node_provisional=d['node'][i].tolist(),
                        legal=d['valid'][i].tolist(),node_legal=d['node_valid'][i].tolist(),
                        provisional_is_node=((d['token'][i]>=am.tok.node_lo)&(d['token'][i]<am.tok.node_lo+am.tok.n_node_tokens)).tolist(),
                        actual=int(actual[i]),actual_legal=bool(f['allowed'][i,actual[i]])))
            print('PROBES FIT',arm,seed,flush=True);R['PREP']['write_cache'](dest,result)
            result['interventions'],result['verification']=interventions(am,cases,parts['test'],lens,linear,mlp,axes,args.batch,seed)
            shuffled={}
            for j,name in enumerate(['probe_train','probe_val','test']):
                d=bd[name];rng=np.random.default_rng(9911+j);flips={q:rng.integers(2) for q in np.unique(d['query'])}
                yy=np.logical_xor(d['y'],np.array([flips[q] for q in d['query']])[:,None])
                shuffled[name]={**d,'y':yy}
            null=fit_binary_heads(shuffled['probe_train'],shuffled['probe_val'],hidden=64,steps=args.probe_steps,seed=2)
            result['shuffle_probes']=probe_reports(null,shuffled['test'],parts['test'],'mlp64_shuffle')
            result.update(complete=True,seconds=time.time()-started,
                checkpoint_unchanged=sha_file(Path(am.checkpoint['path']))==am.checkpoint['sha256'])
            assert result['checkpoint_unchanged'];R['PREP']['write_cache'](dest,result)
            print('MODEL FINISHED',arm,seed,round(result['seconds']),flush=True)
            del am,features,bd,dec,lens,linear,mlp,axes,null,shuffled;gc.collect();torch.cuda.empty_cache()
    print('WITHIN TOKEN COMPLETE',flush=True)


if __name__=='__main__':main()
