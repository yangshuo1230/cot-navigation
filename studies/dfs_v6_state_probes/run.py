"""Repeat query-held-out search-state probes on all four final v6 DFS nodes.

    Reuse the previous feature extractor, fitting objective and pairing rules.
    Evaluate both frozen legal variants and natural legal generated prefixes.
    No model/probe weights or activation caches are written.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['WANDB_MODE']='disabled'
import sys,json,hashlib,random,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
V6=ROOT/'studies/fixed_navigation_protocols_v6'
sys.path[:0]=[str(V6),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('previous_filter_audit',ROOT/'studies/search_filter_audit/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
os.environ['CUDA_VISIBLE_DEVICES']='0'
from packed_bank import PackedBank
from semantics import replay_nodes
from cot_navigation.analysis.common import AnalysisModel
from cot_navigation.analysis.probes import fit,shuffled_labels
from cot_navigation.analysis.set_metrics import scores
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.dfs import Step,dfs_search
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

OUT=Path(__file__).resolve().parent
PANELS=['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']
SEED=20261026


def quality(z,y,mask):
    """Vectorized tie-aware counterpart of the historical within-state metrics.

    Return one dict/head. Empty/full sets are excluded from rank statistics.
    Full-set accuracy below uses oracle k, just as the earlier experiments.
    """
    valid_columns=np.flatnonzero(np.asarray(mask,bool).any(0))
    yy=np.asarray(y[:,valid_columns],bool);mm=np.asarray(mask[:,valid_columns],bool)
    pos=yy&mm;neg=(~yy)&mm;np_=pos.sum(1);nn=neg.sum(1);mixed=(np_>0)&(nn>0)
    result=[];n,k=yy.shape;rows=np.arange(n)[:,None];ar=np.arange(k)[None,:]
    for h in range(z.shape[1]):
        zz=z[:,h,valid_columns];masked=np.where(mm,zz,-np.inf)
        order=np.argsort(masked,axis=1,kind='stable');sz=np.take_along_axis(masked,order,1)
        sp=np.take_along_axis(pos,order,1);sn=np.take_along_axis(neg,order,1)
        cumulative=sn.cumsum(1)
        start=np.maximum.accumulate(np.where(np.c_[np.ones(n,bool),sz[:,1:]!=sz[:,:-1]],ar,0),axis=1)
        end=np.minimum.accumulate(np.where(np.c_[sz[:,:-1]!=sz[:,1:],np.ones(n,bool)],ar,k)[:,::-1],axis=1)[:,::-1]
        below=np.where(start>0,np.take_along_axis(cumulative,np.maximum(start-1,0),1),0)
        ties=np.take_along_axis(cumulative,end,1)-below
        auc=((below+.5*ties)*sp).sum(1)/np.maximum(1,np_*nn)
        descending=np.argsort(-masked,axis=1,kind='stable')
        sorted_true=np.take_along_axis(pos,descending,1)
        hit=(sorted_true&(ar<np_[:,None])).sum(1)
        strict=np.where(pos,zz,np.inf).min(1)>np.where(neg,zz,-np.inf).max(1)
        pred=(zz>=0)&mm
        tp=int((pred&pos).sum());fp=int((pred&neg).sum());fn=int(((~pred)&pos).sum())
        tn=int(((~pred)&neg).sum())
        item=dict(n_rows=n,nontrivial_rows=int(mixed.sum()),empty_target_rows=int((np_==0).sum()),
                  full_target_rows=int(((nn==0)&(np_>0)).sum()),n_positive=int(np_.sum()),n_negative=int(nn.sum()),
                  f1=2*tp/max(1,2*tp+fp+fn),
                  balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)),
                  threshold_exact_set_accuracy=float(((pred==pos)|~mm).all(1).mean()))
        if mixed.any():
            item.update(pairwise_auc=float(auc[mixed].mean()),topk_exact=float((hit[mixed]==np_[mixed]).mean()),
                        topk_recall=float((hit[mixed]/np_[mixed]).mean()),strict_separation=float(strict[mixed].mean()))
        result.append(item)
    return result


def sample(world,panel,count,seed):
    """Deterministic gap/label round robin, without test-informed selection."""
    cells={};rng=random.Random(seed)
    for q in world['panels'][panel]:
        r=world['records'][q];cells.setdefault((r['gap'],r['y']),[]).append(q)
    for ids in cells.values():rng.shuffle(ids)
    selected=[]
    while len(selected)<count:
        before=len(selected)
        for key in sorted(cells):
            if cells[key] and len(selected)<count:selected.append(cells[key].pop())
        if len(selected)==before:break
    return selected


def examples(world,tokens,ids):
    out=[]
    for q in ids:
        r=world['records'][q]
        prompt=[TOK.id('<QUERY>'),tokens[r['s']],tokens[r['t']],TOK.id('|')]
        out.append(SimpleNamespace(index=q,adj=world['adj'],nodes=tokens,source=r['s'],target=r['t'],
                                   prompt=prompt,row=SimpleNamespace(prompt_len=4)))
    return out


def generated_steps(world,inverse,out,qid):
    r=world['records'][qid];seq=[];steps=[]
    for token in out['trace_tokens']:
        if token not in inverse:break
        v=inverse[token]
        if not seq:
            if v!=r['s']:break
            stack=[v]
        else:
            prior=old.gen.decision_state(world['adj'],r['s'],r['t'],seq,False)
            if prior is None or v not in prior['legal_next']:break
            stack=prior['frontier']+[v]
        seq.append(v);visited=set(seq);cands=tuple(w for w in world['adj'][v] if w not in visited)
        action='FOUND' if v==r['t'] else 'PUSH' if cands else 'POP' if len(stack)>1 else 'EXHAUST'
        steps.append(Step(v,cands,tuple(stack),tuple(seq),action,cands[0] if action=='PUSH' else None))
    return steps


def subset_bundle(bundle,select):
    return {k:v[select] for k,v in bundle.items()}


def evaluate(fitted,bundle,meta,task):
    z=scores(fitted,bundle)
    item={'scores':quality(z,bundle['y'],bundle['mask'])}
    if task in ['allowed','filter']:
        neighbor=np.stack([old.encode(m['nodes'],m['succ'],TOK.n_node_tokens,TOK.node_lo) for m in meta])
        item['neighbor_only']=quality(z,bundle['y'],neighbor)
        item['matched']=old.pair_metrics(z,old.matched_pairs(meta),meta,TOK.node_lo)
        item['matched_exact_stack']=old.pair_metrics(z,old.matched_pairs(meta,True),meta,TOK.node_lo)
    return item


def main():
    global TOK
    torch.set_num_threads(4);TOK=Tokenizer(256)
    manifest=json.loads((V6/'manifest.json').read_text());worldmeta=next(m for m in manifest['worlds'] if m['name']=='n80_g0')
    bank=PackedBank(worldmeta,'dfs_nodes',torch.device('cpu'));world=bank.world
    nodes=[TOK.node_id(v) for v in world['node_names']];inverse={t:i for i,t in enumerate(nodes)}
    train=sample(world,'short_train',192,SEED)+sample(world,'long_train',64,SEED+1)
    val=sample(world,'short_val',72,SEED+2)+sample(world,'long_AA_val',24,SEED+3)
    tests={name:world['panels'][name] for name in PANELS}
    assert not(set(train)&set(val))
    for ids in tests.values():assert not(set(ids)&(set(train)|set(val)))
    def frozen_steps(adj,s,t,protocol,qid,variant=0):
        rng=random.Random(71000000+world['width']*100000+world['seed']*10000+qid*37+variant)
        shuffled=[list(vs) for vs in adj]
        for vs in shuffled:rng.shuffle(vs)
        steps=dfs_search(shuffled,s,t)
        row=int(bank.qptr[qid])+variant
        assert int(bank.cpu['variants'][row])==variant
        a=int(bank.cpu['offsets'][row])+4;b=int(bank.cpu['offsets'][row])+int(bank.cpu['answers'][row])
        assert bank.cpu['tokens'][a:b].tolist()==[nodes[v] for v in steps[-1].visited]
        return steps
    doc={'heads':old.HEADS,'train_ids':train,'val_ids':val,'test_ids':tests,'variants':8,'steps':250,
         'protocol':'query-disjoint, short+AA probe fitting only, no new node holdout; all4 final50k',
         'free_note':'Only valid generated expansion prefixes, excludes source and current goal; failed whole traces retain prior valid anchors.',
         'models':{}}
    path=OUT/'results.json'
    if path.exists():doc=json.loads(path.read_text())
    for seed in range(4):
        name=f'seed{seed}'
        if name in doc['models']:continue
        run=ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/dfs_nodes'
        blob=torch.load(run/'latest.pt',map_location='cpu',weights_only=False);assert blob['step']==50000
        model=TransformerLM(len(TOK),**blob['job']['model'],dropout=0.).to('cuda:0');model.load_state_dict(blob['model']);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        class Adapter:
            forward=torch.inference_mode()(AnalysisModel.forward)
        am=Adapter();am.model=model;am.tok=TOK;am.device=torch.device('cuda:0');am.protocol='dfs_nodes'
        old.ref.shuffled_steps=frozen_steps
        features={};metas={}
        for panel,ids in [('train',train),('val',val)]+list(tests.items()):
            print('COLLECT',seed,panel,len(ids),flush=True)
            features[panel],metas[panel]=old.collect(am,examples(world,nodes,ids),8)
        free={};free_meta={};completion={};gen_checks={}
        compiled=torch.compile(model,dynamic=True)
        for panel,ids in tests.items():
            print('FREE',seed,panel,flush=True)
            exs=examples(world,nodes,ids);outputs=[]
            for at in range(0,len(exs),128):
                outputs+=generate(compiled,[e.prompt for e in exs[at:at+128]],'dfs_nodes',TOK,'bf16',max_new_tokens=bank.info['max_new_tokens'])
            mapping={qid:generated_steps(world,inverse,out,qid) for qid,out in zip(ids,outputs)}
            completion[panel]={qid:bool(replay_nodes(world['adj'],world['records'][qid]['s'],world['records'][qid]['t'],out['trace_tokens'],inverse)['valid']) and not out['generation_limit_reached'] for qid,out in zip(ids,outputs)}
            gen_checks[panel]={'queries':len(ids),'complete_legal':sum(completion[panel].values()),
                               'queries_with_valid_anchors':sum(len(mapping[q])>1 for q in ids)}
            old.ref.shuffled_steps=lambda adj,s,t,protocol,qid,variant=0:mapping[qid]
            usable=[e for e in exs if mapping[e.index]]
            free[panel],free_meta[panel]=old.collect(am,usable,1)
        old.ref.shuffled_steps=frozen_steps
        result={'checkpoint_sha256':hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest(),
                'rows':{k:len(v) for k,v in metas.items()},'generation':gen_checks,'tasks':{}}
        for task in ['succ','allowed','filter','visited','frontier']:
            print('FIT',seed,task,flush=True)
            base='allowed' if task=='filter' else task
            def bundle(data):
                b=dict(data[base])
                if task=='filter':b['mask']=data['succ']['y']
                return b
            tr=bundle(features['train']);va=bundle(features['val'])
            fitted=fit(tr,va,device='cuda:0',seed=SEED,steps=250)
            item={'selected_steps':fitted['steps'],'reference':{},'free':{},'free_by_completion':{}}
            for panel in tests:
                item['reference'][panel]=evaluate(fitted,bundle(features[panel]),metas[panel],task)
                b=bundle(free[panel]);mm=free_meta[panel]
                item['free'][panel]=evaluate(fitted,b,mm,task)
                # Same fitted decoder; do not refit separately on successful or failed outputs.
                item['free_by_completion'][panel]={}
                for label in [True,False]:
                    choose=np.asarray([completion[panel][m['query']]==label for m in mm])
                    if choose.any():
                        zz=scores(fitted,subset_bundle(b,choose))
                        item['free_by_completion'][panel][str(label)]={'queries':len(set(b['query'][choose].tolist())),
                            'scores':quality(zz,b['y'][choose],b['mask'][choose])}
            result['tasks'][task]=item
            print('L6 TT',seed,task,item['reference']['long_TT_test']['scores'][5].get('pairwise_auc'),
                  item['free']['long_TT_test']['scores'][5].get('pairwise_auc'),flush=True)
            del fitted
        tr=dict(features['train']['allowed'],mask=features['train']['succ']['y'])
        va=dict(features['val']['allowed'],mask=features['val']['succ']['y'])
        fitted=fit(tr,va,device='cuda:0',seed=SEED+17,steps=250,shuffled=True)
        result['shuffled_filter']={}
        for panel in tests:
            b=dict(features[panel]['allowed'],mask=features[panel]['succ']['y'])
            result['shuffled_filter'][panel]=quality(scores(fitted,b),shuffled_labels(b,SEED+219),b['mask'])
        doc['models'][name]=result
        path.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        del fitted,compiled,model,am,blob,features,metas,free,free_meta
        torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':main()
