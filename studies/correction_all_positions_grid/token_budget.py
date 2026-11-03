"""CPU-only expected nonpadding token budgets; never launches training."""
import json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.scale_prepared import identity,cache_path
from cot_navigation.data.natural_mistakes import pool_path
from cot_navigation.protocols.tokenizer import Tokenizer


def main():
    cfg=load_config([ROOT/'configs/generated/correction_all_positions_v2/dfs_all_positions_l6l9_p0p30p60_20261102/l6_p60_s0.yaml'])
    tok=Tokenizer.from_config(cfg);scales=[]
    for root in cfg.train.scale_roots:
        source=ContextStore(root);qs=source.indices('train')
        with np.load(cache_path(source,identity(source,cfg,tok)),allow_pickle=False) as a:
            offsets=a['offsets'];ptr=a['query_offsets'];fields=a['fields']
            clean=[];cot=[]
            for q in qs:
                lo,hi=ptr[q:q+2];clean.append(float(np.diff(offsets[lo:hi+1]).mean()))
                cot.append(float((fields[lo:hi,1]-fields[lo:hi,0]).mean()))
            clean_mean=float(np.mean(clean));clean_cot=float(np.mean(cot))
        with np.load(pool_path(source,cfg,tok),allow_pickle=False) as a:
            train=np.flatnonzero(a['split']==0);length=np.diff(a['offsets']);fields=a['fields'];meta=a['metadata']
            complex_=np.array([json.loads(str(v))['natural_complex'] for v in meta],bool)
            err=errcot=0.
            for label in [0,1]:
                prob=float(np.mean(source.arrays['label'][qs]==label))
                all_=train[fields[train,2]==label];hard=all_[complex_[all_]]
                if not len(hard):hard=all_
                err+=prob*.5*(float(length[all_].mean())+float(length[hard].mean()))
                cc=fields[:,1]-fields[:,0];errcot+=prob*.5*(float(cc[all_].mean())+float(cc[hard].mean()))
        scales.append(dict(root=root,clean_tokens=clean_mean,error_tokens=err,clean_cot_tokens=clean_cot,error_cot_tokens=errcot))
        print(root,round(clean_mean,2),round(err,2),flush=True);del source
    weights=np.array(cfg.train.scale_stage_weights).reshape(4,6)
    schedule={}
    for percent in [0,30,60]:
        p=percent/100;expected=weights@np.array([(1-p)*s['clean_tokens']+p*s['error_tokens'] for s in scales])
        reference=weights@np.array([.4*s['clean_tokens']+.6*s['error_tokens'] for s in scales])
        steps=np.rint(50000*reference/expected).astype(int)
        schedule[str(percent)]=dict(stage_steps=steps.tolist(),stage_ends=np.cumsum(steps).tolist(),max_steps=int(steps.sum()),
            expected_tokens_per_query=expected.tolist(),expected_training_tokens=float((steps*64*expected).sum()),
            stage_expected_tokens=(steps*64*expected).tolist())
    result=dict(reference='60% = 200k steps, 50k per curriculum stage',
        token_definition='nonpadding full sequence: graph+query+CoT+answer+EOS; excludes validation and padding',
        method='exact cache-length expectation, uniform training query/variant; label-conditioned complex-share0.5 error sampling',
        batch=64,scales=scales,schedule=schedule,training_launched=False,
        note='Token-matched in expectation per stage; realized counts have sampling variation. Applies equally to l6/l9 and both initialization seeds.')
    out=ROOT/'outputs/correction_all_positions_v2/token_matched_plan_20261102.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps(schedule,indent=2))


if __name__=='__main__':main()
