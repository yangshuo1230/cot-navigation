"""Natural-error residual interventions on frozen matched/correction checkpoints.

Train/validation only define linear axes and class-mean targets. Test queries
are never used to pick directions, amplitudes, layers or successful examples.
Local fixed-text decision readouts and native-budget free continuations are
reported separately. All navigation queries are reachable positive examples.
"""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import runpy
import time

os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:
    os.environ.setdefault(k,'2')
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
R=runpy.run_path(str(P.parent/'natural_history_probes/run.py'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cot_navigation.analysis.history_probe import fit_positions
from cot_navigation.analysis.natural_history import match_readouts
from cot_navigation.analysis.residual_control import probe_axis,orthogonal_axis,residual_edits,steer
from cot_navigation.evaluation.rollout import generate,pad_rows
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.experiments.artifacts import atomic_json,sha_file

LAYERS=[3,4,5]
SITES=['onset','decision']
LOCAL=[('clean',.5),('clean',1.),('error',.5),('error',1.),('random_clean',1.),('random_error',1.)]
FREE=[('clean',1.),('error',1.),('random_clean',1.),('random_error',1.)]


def boundary(c,pos,tok):
    ri=next((i for i,(a,b) in enumerate(c['spans']) if a<=pos<b),None)
    if ri is None:return None
    a,b=c['spans'][ri];bars=[p for p in range(a,b) if c['trace'][p]==tok.id('|')]
    return (ri,a,bars[2],bars) if len(bars)>=3 else None


def paired(rows):
    pairs=defaultdict(list)
    for r in rows:pairs[r['event']].append(r)
    return {e:sorted(rr,key=lambda r:-r['y']) for e,rr in pairs.items() if len(rr)==2}


def plans_for(cases,tok):
    plans={};sampling={};separate={}
    for j,split in enumerate(['train','val','test']):
        separate[split]={};sampling[split]={}
        for site in SITES:
            rr,info=match_readouts(cases[split],tok,20261030+j,error_offsets=[0],
                anchor='first_error' if site=='onset' else 'action_boundary')
            pairs=paired(rr)
            if site=='decision':
                pairs={e:pair for e,pair in pairs.items() if all(
                    boundary(cases[split][r['sid']],r['pos'],tok) is not None
                    and boundary(cases[split][r['sid']],r['pos'],tok)[2]==r['pos'] for r in pair)}
            rr=[{**r,'cell':SITES.index(site)*10000,'phase':SITES.index(site)} for pair in pairs.values() for r in pair]
            for r in rr:
                c=cases[split][r['sid']]
                assert r['y']==int(c['tau'] is not None and r['pos']>=c['tau'])
            separate[split][site]=rr
            sampling[split][site]=dict(positive_queries=len(pairs),pairs=len(rr)//2)
        plans[split]=sum(separate[split].values(),[])
    return plans,separate,sampling


def select_events(cases,separate,tok,local_pairs,free_pairs):
    onset=paired(separate['onset']);decision=paired(separate['decision']);eligible=[]
    for event in sorted(set(onset)&set(decision)):
        pair=onset[event];dpair=decision[event]
        if not pair[0]['kind'].startswith('candidate_'):continue
        if pair[0]['sid']!=dpair[0]['sid']:continue
        ok=True
        for r in pair+dpair:
            c=cases[r['sid']];b=boundary(c,r['pos'],tok)
            if b is None or r['pos']>b[2]:ok=False;break
            # A fixed-text bridge must not introduce a new natural error into
            # a negative readout between its onset-matched site and action.
            if not r['y'] and c['tau'] is not None and c['tau']<=b[2]:ok=False;break
        if ok:eligible.append(event)
    rng=np.random.default_rng(20261031);rng.shuffle(eligible)
    selected=eligible[:local_pairs];free=set(selected[:free_pairs])
    out=[]
    for event in selected:
        for site,pairs in [('onset',onset),('decision',decision)]:
            for row in pairs[event]:
                c=cases[row['sid']];ri,a,end,bars=boundary(c,row['pos'],tok)
                out.append(dict(row,event=event,site=site,record=ri,decision_pos=end,
                    free=event in free and site=='onset'))
    assert len(selected)>=16,'Insufficient natural events; do not replace with artificial edits'
    return out,dict(eligible_pairs=len(eligible),local_pairs=len(selected),free_pairs=len(free),
        selection_seed=20261031,first_types={k:sum(r['site']=='onset' and r['y']==1 and r['kind']==k for r in out) for k in sorted({r['kind'] for r in out})})


def probe_logit(fitted,cell,head,h):
    g=fitted['mapping'][cell];idx=g*fitted['heads']+head
    x=(h-fitted['mean'][g,head])/fitted['scale'][g,head]
    p=fitted['probe'];z=x@p.w[idx]+p.b[idx]
    return (F.gelu(z)*p.v[idx]).sum(-1)+p.c[idx] if fitted['hidden'] else z[...,0]


def make_axes(fitted,train,seed):
    axes={}
    for site in SITES:
        for hi,layer in enumerate(LAYERS):
            a=probe_axis(fitted,train,SITES.index(site)*10000,hi)
            a['random']=orthogonal_axis(a['direction'],9173+100*seed+10*SITES.index(site)+layer)
            axes[site,layer]=a
    return axes


def edit_for(job,row,axes,absolute_position):
    a=axes[row['site'],job['layer']];target=a['means'][0 if job['condition'].endswith('clean') else 1]
    return (absolute_position,a['direction'],target,job['dose'],a['random'] if job['condition'].startswith('random_') else None)


def graph_and_wanted(am,c,row):
    graph,src,dst=R['PREP']['graph_tokens'](am,row['query'])
    _,a,_,bars=boundary(c,row['pos'],am.tok)
    stack=c['trace'][a:bars[0]];visited=c['trace'][bars[0]+1:bars[1]]
    cur=stack[-1];eligible=set(graph.get(cur,[]))-set(visited)
    op='<FOUND>' if cur==dst else '<PUSH>' if eligible else '<POP>' if len(stack)>1 else '<EXHAUST>'
    return graph,src,dst,am.tok.id(op)


def local_readout(am,cases,rows,axes,fitted,mlp,batch):
    jobs=[];probe_quality=defaultdict(list);ops=[am.tok.id(s) for s in ['<PUSH>','<POP>','<FOUND>','<EXHAUST>','<MISTAKE>']]
    mistake=ops[-1];prefixes={};graphs={}
    for i,row in enumerate(rows):
        c=cases[row['sid']];ex=am.example(row['query'],'test')
        prefix=ex.prompt+c['trace'][:row['decision_pos']+1]
        if len(prefix)>am.model.max_seq_len-256:continue
        prefixes[i]=prefix;graphs[i]=graph_and_wanted(am,c,row)
        jobs.append(dict(row=i,layer=-1,condition='sham',dose=0.))
        jobs.extend(dict(row=i,layer=l,condition=cond,dose=dose) for l in LAYERS for cond,dose in LOCAL)
    distributions={};output=[]
    for begin in range(0,len(jobs),batch):
        chunk=jobs[begin:begin+batch];seqs=[prefixes[j['row']] for j in chunk]
        ids,lengths=pad_rows(seqs,am.tok.pad,am.device);edits=defaultdict(list);capture={}
        for b,j in enumerate(chunk):
            r=rows[j['row']]
            if j['layer']>=0:
                pos=len(seqs[b])-r['decision_pos']-1+r['pos']
                edits[j['layer']].append((b,*edit_for(j,r,axes,pos)))
        with torch.no_grad(),residual_edits(am.model,edits):logits=am.model(ids,capture=capture)
        selected=logits[torch.arange(len(chunk),device=am.device),lengths-1].float()
        prob=selected.softmax(-1);op_pred=selected.argmax(-1)
        for b,j in enumerate(chunk):
            i=j['row'];r=rows[i];cell=SITES.index(r['site'])*10000
            desired=mistake if r['y'] else graphs[i][3]
            item=dict(**{k:r[k] for k in ['event','query','positive_query','y','site','kind','exact_token']},
                condition=j['condition'],layer=j['layer']+1,dose=j['dose'],
                onset_equals_decision=r['pos']==r['decision_pos'],
                mistake_p=float(prob[b,mistake]),desired_p=float(prob[b,desired]),
                decision_correct=int(op_pred[b])==desired,decision_mistake=int(op_pred[b])==mistake,
                valid_operator=int(op_pred[b]) in ops)
            distributions[len(output)]=prob[b].cpu().numpy()
            pos=len(seqs[b])-r['decision_pos']-1+r['pos']
            if j['layer']<0:
                for hi,l in enumerate(LAYERS):
                    h=capture['resid'][l][b,pos].float()
                    for name,f in [('linear',fitted),('mlp64',mlp)]:
                        z=float(probe_logit(f,cell,hi,h))
                        probe_quality[r['site'],l+1,name].append((r['y'],z))
            else:
                hi=LAYERS.index(j['layer']);h=capture['resid'][j['layer']][b,pos].float()
                _,u,target,dose,null=edit_for(j,r,axes,pos)
                edited=steer(h,torch.as_tensor(u,device=h.device),target,dose=dose,
                    random_direction=None if null is None else torch.as_tensor(null,device=h.device))
                item['relative_norm']=float((edited-h).norm()/h.norm().clamp_min(1e-8))
                for name,f in [('linear',fitted),('mlp64',mlp)]:
                    item[name+'_before']=float(probe_logit(f,cell,hi,h).sigmoid())
                    item[name+'_after']=float(probe_logit(f,cell,hi,edited).sigmoid())
            output.append(item)
        del capture,ids,logits,prob
        if begin%(batch*16)==0:print('LOCAL',begin,'/',len(jobs),flush=True)
    baselines={j['row']:index for index,j in enumerate(jobs) if j['condition']=='sham'}
    for k,(j,item) in enumerate(zip(jobs,output)):
        p=distributions[k];q=distributions[baselines[j['row']]]
        item['kl_from_sham']=float(np.sum(q*np.log((q+1e-30)/(p+1e-30))))
        if item['site']=='onset' and item['layer']==6 and not item['onset_equals_decision']:
            assert np.max(np.abs(p-q))<2e-6,'Past final-layer intervention must not reach later fixed-text logits'
    quality=[dict(site=k[0],layer=k[1],architecture=k[2],n=len(v),
        accuracy=float(np.mean([(z>=0)==bool(y) for y,z in v])),threshold=.5) for k,v in probe_quality.items()]
    return output,quality


class PatchedPrefill(nn.Module):
    """Eager edited prefill; original compiled cached decoder, hooks removed."""
    def __init__(self,raw,compiled,edits):
        super().__init__();self.raw=raw;self.compiled=compiled;self.edits=edits;self.max_seq_len=raw.max_seq_len
    def new_cache(self,*a,**kw):return self.raw.new_cache(*a,**kw)
    def forward(self,tokens,positions=None,cache=None,cache_mode=None):
        if cache_mode=='prefill':
            with residual_edits(self.raw,self.edits):
                return self.raw(tokens,positions=positions,cache=cache,cache_mode=cache_mode)
        return self.compiled(tokens,positions=positions,cache=cache,cache_mode=cache_mode)


def free_continuations(am,cases,rows,axes,batch,result,dest):
    selected=[r for r in rows if r['free']];jobs=[];prefixes={};graphs={}
    for i,r in enumerate(selected):
        c=cases[r['sid']];ex=am.example(r['query'],'test')
        prefix=ex.prompt+c['trace'][:r['pos']+1]
        if len(prefix)>am.model.max_seq_len-256:continue
        prefixes[i]=prefix;graphs[i]=graph_and_wanted(am,c,r)
        jobs.append(dict(row=i,layer=-1,condition='sham',dose=0.))
        jobs.extend(dict(row=i,layer=l,condition=cond,dose=dose) for l in LAYERS for cond,dose in FREE)
    compiled=CompiledInference(am.model).eval();output=[];examples=[]
    for begin in range(0,len(jobs),batch):
        chunk=jobs[begin:begin+batch];edits=defaultdict(list)
        for b,j in enumerate(chunk):
            if j['layer']>=0:
                r=selected[j['row']]
                edits[j['layer']].append((b,*edit_for(j,r,axes,len(prefixes[j['row']])-1)))
        wrapper=PatchedPrefill(am.model,compiled,edits).eval()
        with torch.no_grad():outs=generate(wrapper,[prefixes[j['row']] for j in chunk],
            'dfs_state',am.tok,'fp32',cot_margin=am.cfg.protocol.cot_margin,max_new_tokens=0)
        for j,g in zip(chunk,outs):
            r=selected[j['row']];c=cases[r['sid']];graph,src,dst,wanted=graphs[j['row']]
            trace=c['trace'][:r['pos']+1]+g['trace_tokens']
            audit=replay(trace,g['pred'],graph,src,dst,am.tok,label=1,interrupted=g['cot_interrupted'])
            ends=[p for p,t in enumerate(trace) if t==am.tok.id(';')]
            focus_end=ends[r['record']] if len(ends)>r['record'] else None
            focus_mark=False;focus_false=False
            if focus_end is not None:
                start=ends[r['record']-1]+1 if r['record'] else 0
                fields=[];a=start
                for p in range(start,focus_end):
                    if trace[p]==am.tok.id('|'):fields.append(trace[a:p]);a=p+1
                action=trace[a:focus_end]
                focus_mark=action==[am.tok.id('<MISTAKE>')] and len(fields)==3
                if focus_mark:
                    fa=replay(trace[:focus_end+1],None,graph,src,dst,am.tok,label=1)
                    focus_false=fa['false_marks']>0
            item=dict(**{k:r[k] for k in ['event','query','positive_query','y','kind','exact_token']},
                condition=j['condition'],layer=j['layer']+1,dose=j['dose'],
                focus_mark=focus_mark,focus_false_mark=focus_false,
                first_detected=audit['first_detected'],first_repaired=audit['first_repaired'],
                correct=audit['correct'],trace_valid=audit['trace_valid'],answer_and_trace_valid=audit['answer_and_trace_valid'],
                true_marks=audit['true_marks'],false_marks=audit['false_marks'],repairs=audit['repairs'],
                generated_tokens=len(g['generated']),limit=g['generation_limit_reached'],
                forced_stop=g['forced_stop'],pred=g['pred'],outcome=audit['outcome'])
            if r['y']:assert audit['first']['token']==c['first']['token']
            if len(examples)<8 and (focus_mark or audit['first_repaired']):
                examples.append(dict(item,excerpt=am.tok.decode(trace[max(0,r['pos']-8):r['pos']+90])))
            output.append(item)
        result['rollouts']=output;result['examples']=examples;atomic_json(dest,result)
        print('FREE',begin+len(chunk),'/',len(jobs),flush=True)
    result['compile_fallbacks']=sorted(compiled.compile_fallbacks)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--local-pairs',type=int,default=128)
    parser.add_argument('--rollout-pairs',type=int,default=32)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--model-seeds',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args();os.chdir(ROOT);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.32)
    identity=dict(vars(args),source_sha256={str(p.relative_to(ROOT)):sha_file(p) for p in [Path(__file__),
        ROOT/'src/cot_navigation/analysis/residual_control.py',ROOT/'src/cot_navigation/analysis/natural_history.py',
        ROOT/'src/cot_navigation/analysis/history_probe.py']},gpu=0,
        precision='fp32 throughout paired intervention comparisons',max_relative_norm=.2,
        target='training class-mean coordinate; no test tuning; independent within-model axes')
    (P/'results').mkdir(parents=True,exist_ok=True);atomic_json(P/'plan.json',identity)
    cases=plans=separate=sampling=selection=selected=None
    for seed in args.model_seeds:
        for arm in ['matched_clean','correction']:
            dest=P/'results'/f'{arm}_s{seed}.json'
            existing=json.loads(dest.read_text()) if dest.exists() else None
            if existing and existing.get('complete'):
                assert existing['identity']==identity
                assert sha_file(Path(existing['checkpoint']['path']))==existing['checkpoint']['sha256'];continue
            start=time.time();am=R['Model'](arm,seed)
            if cases is None:
                cases,_,_=R['load_data'](am.tok)
                plans,separate,sampling=plans_for(cases,am.tok)
                selected,selection=select_events(cases['test'],separate['test'],am.tok,args.local_pairs,args.rollout_pairs)
                atomic_json(P/'selection.json',dict(sampling=sampling,selection=selection,
                    selected_readouts=selected,split_audit='official graph/query disjointness checked by load_data'))
                print('SELECTION',selection,flush=True)
            data={s:R['extract'](am,cases[s],plans[s]) for s in ['train','val']}
            for d in data.values():d['x']=np.ascontiguousarray(d['x'][:,LAYERS,:am.model.d_model])
            fitted=fit_positions(data['train'],data['val'],args.steps,0,hidden=0)
            mlp=fit_positions(data['train'],data['val'],args.steps,1,hidden=64)
            assert fitted is not None and mlp is not None and set(fitted['mapping'])=={0,10000}
            axes=make_axes(fitted,data['train'],seed)
            result=dict(complete=False,identity=identity,arm=arm,model_seed=seed,checkpoint=am.checkpoint,
                sampling=sampling,selection=selection,
                axes={f'{site}_L{l+1}':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in a.items()}
                      for (site,l),a in axes.items()},probe_steps=fitted['selected_steps'],
                probe_validation_bce=fitted['validation_bce'],local=[],rollouts=[])
            result['local'],result['probe_test_accuracy']=local_readout(am,cases['test'],selected,axes,fitted,mlp,args.batch)
            atomic_json(dest,result)
            del data,fitted,mlp;gc.collect();torch.cuda.empty_cache()
            free_continuations(am,cases['test'],selected,axes,args.batch,result,dest)
            result.update(complete=True,seconds=time.time()-start,
                checkpoint_unchanged=sha_file(Path(am.checkpoint['path']))==am.checkpoint['sha256'])
            assert result['checkpoint_unchanged'];atomic_json(dest,result)
            print('MODEL FINISHED',arm,seed,round(result['seconds']),flush=True)
            del am,axes;gc.collect();torch.cuda.empty_cache()
    print('CAUSAL INTERVENTIONS COMPLETE',flush=True)


if __name__=='__main__':main()
