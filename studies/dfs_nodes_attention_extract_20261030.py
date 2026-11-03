"""CPU-only measured attention from a held-out, freely generated DFS Nodes example."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT = Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0, str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ContextStore
from cot_navigation.evaluation.rollout import generate
from cot_navigation.protocols.search_replay import replay_nodes

def main():
    torch.set_num_threads(2)
    outdir = ROOT/'studies/dfs_nodes_attention_20261030'
    outdir.mkdir(exist_ok=True)
    run = ROOT/'outputs/fine_node_curriculum_20261028/s0/dfs_nodes/simple'
    cfg = load_config([run/'resolved_config.yaml'])
    tok = Tokenizer.from_config(cfg)
    cp = run/'checkpoints/latest.pt'
    state = torch.load(cp,map_location='cpu',weights_only=False)
    model = build_model(cfg.model,len(tok),tok.pad).eval()
    model.load_state_dict(state['model']); model.requires_grad_(False)
    data_cfg = load_config([ROOT/'configs/generated/fine_node_curriculum_20261028/data_simple_n12.yaml'])
    store = ContextStore(ROOT/data_cfg.data.root)
    attempts=[]
    for ix in store.indices('test').tolist():
        row = store.render(ix,'dfs_nodes',tok)
        if row.label != 1 or not 5 <= len(row.trace_indices)-1 <= 9: continue
        nt = store.node_vocab_ids(ix,tok); inv={v:i for i,v in enumerate(nt)}
        s=int(store.arrays['s_topo'][ix]); t=int(store.arrays['t_topo'][ix]); adj=store.adj(ix)
        with torch.inference_mode():
            out=generate(model,[row.tokens[:row.prompt_len]],'dfs_nodes',tok,'fp32',max_new_tokens=64)[0]
        rep=replay_nodes(adj,s,t,out['trace_tokens'],inv)
        order=[inv[v] for v in out['trace_tokens'] if v in inv]
        back=sum(v not in adj[u] for u,v in zip(order,order[1:]))
        good=out['pred']==1 and rep['valid'] and back>0 and not out['generation_limit_reached']
        attempts.append(dict(index=ix,valid=rep['valid'],correct=out['pred']==1,backtracks=back))
        if good: break
        if len(attempts)>=80: raise RuntimeError('No suitable rollout')
    else: raise RuntimeError('No held-out example')
    tokens=out['tokens']; p=row.prompt_len
    with torch.inference_mode():
        normal=model(torch.tensor(tokens)[None])
        logits,weights=model(torch.tensor(tokens)[None],return_attn=True)
    attention=np.stack([w[0,0].numpy() for w in weights])
    assert normal[0,p-1:-1].argmax(-1).tolist()==tokens[p:]
    assert np.max(np.triu(attention,1))==0
    assert np.allclose(attention.sum(-1),1,atol=2e-6)
    names={v:chr(65+i) for i,v in enumerate(nt)}
    labels=[names.get(v,tok.name(v).strip('<>')) for v in tokens]
    edges=list(zip(store.arrays['edge_src'][ix].astype(int).tolist(),store.arrays['edge_dst'][ix].astype(int).tolist()))
    d=dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),step=int(state['step']),
      dataset=str(ROOT/data_cfg.data.root),index=ix,sample_id=int(store.arrays['sample_id'][ix]),split='test',
      source=chr(65+s),target=chr(65+t),edges=edges,node_tokens=nt,labels=labels,tokens=tokens,
      prompt_len=p,attention=np.round(attention,8).tolist(),generated_order=[chr(65+v) for v in order],
      strict_replay=rep,attempts=attempts,precision='fp32',device='cpu',
      full_forward_greedy_match=True,max_math_sdpa_logit_difference=float((normal-logits).abs().max()))
    (outdir/'data.json').write_text(json.dumps(d,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({k:v for k,v in d.items() if k not in ['attention','tokens','node_tokens','labels']},indent=2),flush=True)

if __name__=='__main__': main()
