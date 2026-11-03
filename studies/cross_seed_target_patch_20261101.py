"""Matched target counterfactuals and residual interchange, same graph/history.

No navigation training. Freeze source checkpoint. Pair choices do not inspect
model scores. Counterfactual target is unvisited and preferably same graph band.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ContextStore
from cot_navigation.models.transformer import build_model

OUT=ROOT/'studies/cross_seed_attention_20261101'

def margin(logits,tok):
    v=logits[0,-1].float();others=torch.cat((v[:tok.yes],v[tok.yes+1:]))
    return float(v[tok.yes]-torch.logsumexp(others,0))

def main():
    torch.set_num_threads(2);start=time.time();spec=json.loads((OUT/'summary.json').read_text())
    pairs=[];stores={}
    for n,chosen in spec['selected'].items():
        n=int(n);cfg=load_config([ROOT/f'configs/generated/fine_node_curriculum_20261028/data_simple_n{n}.yaml'])
        store=ContextStore(ROOT/cfg.data.root);stores[n]=store
        # Exactly the predeclared 48 positive reference queries, no accuracy filter.
        for ix in chosen:
            if int(store.arrays['label'][ix])!=1:continue
            nt=store.node_vocab_ids(ix,Tokenizer.from_config(cfg));s=int(store.arrays['s_topo'][ix]);t=int(store.arrays['t_topo'][ix])
            row=store.render(ix,'dfs_nodes',Tokenizer.from_config(cfg));prefix=row.tokens[:row.answer_index]
            visited={nt.index(v)for v in prefix[row.prompt_len:]};width=int(cfg.data.shapes['main'].width)
            options=[v for v in range(n)if v not in visited and v//width==t//width]
            same_band=bool(options)
            if not options:options=[v for v in range(n)if v not in visited]
            if not options:continue
            pairs.append(dict(n_nodes=n,index=ix,target=t,alternative=min(options),same_band=same_band))
    results=[]
    with torch.inference_mode():
        for seed in [0,1,2]:
            run=ROOT/f'outputs/fine_node_curriculum_20261028/s{seed}/dfs_nodes/simple';cfg=load_config([run/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
            model=build_model(cfg.model,len(tok),tok.pad).eval();state=torch.load(run/'checkpoints/latest.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.requires_grad_(False)
            for pair in pairs:
                store=stores[pair['n_nodes']];ix=pair['index'];row=store.render(ix,'dfs_nodes',tok);nt=store.node_vocab_ids(ix,tok)
                base=row.tokens[:row.answer_index];donor=base[:];donor[row.prompt_len-2]=nt[pair['alternative']]
                # The entire consumed prefix remains legal: the alternative
                # target has not been visited, and all earlier expansions agree.
                assert donor[row.prompt_len:]==base[row.prompt_len:] and len(set(base[row.prompt_len:]))==len(base[row.prompt_len:])
                bx=torch.tensor(base)[None];dx=torch.tensor(donor)[None];capture={}
                blog=model(bx);dlog=model(dx,capture=capture);bm=margin(blog,tok);dm=margin(dlog,tok)
                r=dict(seed=seed,**pair,base_margin=bm,donor_margin=dm,base_yes=int(blog[0,-1].argmax()==tok.yes),donor_yes=int(dlog[0,-1].argmax()==tok.yes),patches={})
                for site,pos in [('target',row.prompt_len-2),('bar',row.prompt_len-1),('current',len(base)-1)]:
                    vals=[]
                    for l in range(6):
                        donor_value=capture['resid'][l][0,pos].clone()
                        def hook(module,args,output,pos=pos,value=donor_value):
                            x,weights=output;x=x.clone();x[0,pos]=value;return x,weights
                        handle=model.blocks[l].register_forward_hook(hook)
                        try:plog=model(bx)
                        finally:handle.remove()
                        vals.append(dict(margin=margin(plog,tok),yes=int(plog[0,-1].argmax()==tok.yes),max_donor_logit_diff=float((plog-dlog).abs().max())))
                    r['patches'][site]=vals
                # Controls: identical-activation swap is noop; last-layer
                # non-current swaps cannot affect the current row; final-current
                # swap exactly reproduces the donor's output at current position.
                assert abs(r['patches']['target'][-1]['margin']-bm)<1e-5
                assert abs(r['patches']['bar'][-1]['margin']-bm)<1e-5
                assert abs(r['patches']['current'][-1]['margin']-dm)<1e-5
                self_capture={};model(bx,capture=self_capture);value=self_capture['resid'][2][0,row.prompt_len-1].clone()
                def sham(module,args,output):
                    x,w=output;x=x.clone();x[0,row.prompt_len-1]=value;return x,w
                h=model.blocks[2].register_forward_hook(sham)
                try:assert torch.equal(model(bx),blog)
                finally:h.remove()
                results.append(r)
            print(f'target patches seed{seed}: {len(pairs)} pairs done, {time.time()-start:.1f}s',flush=True)
    result=dict(design='Same preselected positive reference queries; identical graph and consumed DFS history; change only QUERY target to unvisited node (same band preferred). No score-based selection. Swap one donor residual at one position after one complete Transformer block. L=1..6. No-op and final-current controls checked for every pair.',
        pairs=pairs,rows=results,elapsed=time.time()-start,controls_passed=len(results))
    (OUT/'target_patch.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(pairs_per_seed=len(pairs),controls_passed=len(results),elapsed=result['elapsed'])),flush=True)

if __name__=='__main__':main()
