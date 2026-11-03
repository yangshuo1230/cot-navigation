"""Frozen v6 Direct separator readouts; no saved activations or probe weights."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path.cwd()
OUT = ROOT / 'studies/direct_v6_layers'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'studies/fixed100_layer_probes')]
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
import linear

def evaluate(y, score, ix, threshold=0):
    return [linear.classification(y[ix], score[ix,h], float(np.asarray(threshold).reshape(-1)[h])
            if np.ndim(threshold) else threshold) for h in range(score.shape[1])]

def main():
    OUT.mkdir(exist_ok=True)
    world=json.loads((ROOT/'studies/fixed_navigation_v5/data/n80_g0.json').read_text())
    panels=world['panels']; rows=world['records']; tok=Tokenizer(256)
    nodes=[tok.node_id(v) for v in world['node_names']]
    ids=torch.tensor([[tok.id('<QUERY>'),nodes[r['s']],nodes[r['t']],tok.id('|')] for r in rows])
    y=np.array([r['y'] for r in rows]);names=['E']+[f'L{i}{s}' for i in range(1,7) for s in ('A','F')]
    # Independently confirm every label from the graph.
    reach=np.eye(80,dtype=bool)
    for u in reversed(range(80)):
        for v in world['adj'][u]: reach[u]|=reach[v]
    assert all(reach[r['s'],r['t']]==bool(r['y']) for r in rows)
    for seed in range(4):
        folder=ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/direct'
        path=folder/'latest.pt';blob=torch.load(path,map_location='cpu',weights_only=False)
        assert blob['step']==50000
        job=blob['job'];model=TransformerLM(len(tok),**job['model'],dropout=0).eval().requires_grad_(False)
        model.load_state_dict(blob['model']);del blob
        model.to('cuda:0')
        observed={};handles=[]
        def before(i):
            def hook(m,a): observed[i,'input']=a[0]
            return hook
        def after(i):
            def hook(m,a,o): observed[i,'A']=observed[i,'input']+o[0]
            return hook
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): reference=model(ids[:128].to('cuda:0'))
        for i,b in enumerate(model.blocks):
            handles += [b.register_forward_pre_hook(before(i)), b.attn.register_forward_hook(after(i))]
        xs=[];scores=[];native=[];error=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(ids),128):
                tokens=ids[start:start+128].to('cuda:0');cap={};logits=model(tokens,capture=cap)
                if start==0:
                    error=float((reference-logits).abs().max()); assert error==0
                states=[model.embed(tokens)[:,-1]]
                for i in range(6): states.extend([observed[i,'A'][:,-1],cap['resid'][i][:,-1]])
                x=torch.stack(states,1); lens=model.lm_head(model.final_norm(x))
                xs.append(x.float().cpu().numpy());scores.append((lens[:,:,tok.yes]-lens[:,:,tok.no]).float().cpu().numpy())
                native.extend(logits[:,-1].argmax(-1).tolist())
        for h in handles:h.remove()
        x=np.concatenate(xs);score=np.concatenate(scores);native=np.array(native)
        tests={k:np.array(v) for k,v in panels.items() if k.endswith('_test')}
        result=dict(seed=seed,step=50000,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            stages=names,hook_max_error=error,native={},native_head={},probes={})
        official=json.loads((folder/'final.json').read_text())
        for name,ix in tests.items():
            target=np.where(y[ix],tok.yes,tok.no);yes=y[ix]==1; correct=native[ix]==target
            ba=(correct[yes].mean()+correct[~yes].mean())/2
            result['native'][name]=dict(n=len(ix),yes_recall=float(correct[yes].mean()),no_recall=float(correct[~yes].mean()),ba=float(ba))
            result['native'][name]['official_ba']=official['panels'][name]['metrics']['all']['ba']
            result['native_head'][name]=evaluate(y,score,ix)
        train=np.array(panels['short_train']+panels['long_train']);val=np.array(panels['short_val']+panels['long_AA_val'])
        protocols={'query_holdout':(train,val,np.array(panels['short_test']+panels['long_AA_test']))}
        # Probe fitting also excludes V/T nodes, including all short queries.
        A=set(world['node_groups']['A']);V=set(world['node_groups']['V']);T=set(world['node_groups']['T'])
        strict_train=np.array([i for i in train if rows[i]['s'] in A and rows[i]['t'] in A])
        strict_val=np.array([i for i,r in enumerate(rows) if r['s'] in V and r['t'] in V])
        strict_test=np.array([i for i,r in enumerate(rows) if r['s'] in T and r['t'] in T])
        assert not (A&V or A&T or V&T)
        protocols['both_endpoint_node_holdout']=(strict_train,strict_val,strict_test)
        for protocol,split in protocols.items():
            pred,fit=linear.fit(x,y,split,binary=True)
            entry=dict(fit=fit,panels={name:evaluate(y,pred[:,:,0],ix,fit['threshold']) for name,ix in tests.items()},
                       heldout=evaluate(y,pred[:,:,0],split[2],fit['threshold']))
            shuffled=y.copy();rng=np.random.default_rng(181)
            for ix in split: shuffled[ix]=rng.permutation(shuffled[ix])
            control,cf=linear.fit(x,shuffled,split,binary=True)
            entry['shuffled_heldout']=evaluate(shuffled,control[:,:,0],split[2],cf['threshold'])
            result['probes'][protocol]=entry
        (OUT/f'seed{seed}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'seed':seed,'native':result['native'],'strict_TT': [round(v['balanced_accuracy'],4) for v in result['probes']['both_endpoint_node_holdout']['panels']['long_TT_test']]}),flush=True)

if __name__=='__main__':main()
