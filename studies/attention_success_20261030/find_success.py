"""Select a short, genuinely successful held-out in-context DFS rollout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ContextStore
from cot_navigation.protocols.dfs import dfs_summary
from cot_navigation.protocols.search_replay import replay_state
from cot_navigation.protocols.validate_trace import parse_completed_steps
from cot_navigation.evaluation.rollout import generate

OUT=ROOT/'studies/attention_success_20261030'
RUN=ROOT/'outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state'
DATA=ROOT/'data/scale_v1/context_l4w4_n16_m24_train50000_s0'

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    cfg=load_config([RUN/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    model=build_model(cfg.model,len(tok),tok.pad).eval();cp=RUN/'checkpoints/best.pt'
    state=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.requires_grad_(False)
    store=ContextStore(DATA);candidates=[]
    for ix in store.indices('test').tolist():
        row=store.render(ix,'dfs_state',tok);s=dfs_summary(row.meta['reference_steps'])
        if row.label==1 and s['pops']>=1 and s['branch_records']>=1 and s['steps']<=10:
            candidates.append((len(row.tokens),ix))
    candidates.sort();attempts=[];chosen=None
    print('eligible',len(candidates),'shortest',candidates[:8],flush=True)
    # Deterministic short-first scan. Stop on the first answer-correct, strictly
    # legal free rollout with at least one branch and POP; no oracle prefix.
    for length,ix in candidates[:48]:
        row=store.render(ix,'dfs_state',tok);prompt=row.tokens[:row.prompt_len]
        with torch.inference_mode():out=generate(model,[prompt],'dfs_state',tok,'fp32',max_new_tokens=512)[0]
        nt=store.node_vocab_ids(ix,tok);inv={v:i for i,v in enumerate(nt)}
        src=int(store.arrays['s_topo'][ix]);dst=int(store.arrays['t_topo'][ix])
        replay=replay_state(store.adj(ix),src,dst,out['trace_tokens'],tok,inv)
        parsed=parse_completed_steps(out['trace_tokens'],tok)
        pops=sum(r['action']=='POP' for r in parsed);branches=sum(len(r['cands'])>1 for r in parsed)
        good=bool(out['pred']==1 and replay.get('valid') and pops>=1 and branches>=1 and not out['generation_limit_reached'])
        attempt=dict(index=ix,sample_id=int(store.arrays['sample_id'][ix]),reference_length=length,
                     generated_length=len(out['tokens']),pred=out['pred'],replay=replay,pops=pops,branches=branches,accepted=good)
        attempts.append(attempt);print(json.dumps(attempt),flush=True)
        if good:chosen=(ix,row,out,parsed,nt,replay);break
    (OUT/'selection.json').write_text(json.dumps(dict(eligible=len(candidates),criteria='Positive test query; reference branch and POP; shortest reference first, sample index tie-break; actual free generation correct with strict replay and branch/POP.',attempts=attempts),indent=2))
    if chosen is None:raise RuntimeError('No successful branch/POP rollout in bounded scan')
    ix,row,out,parsed,nt,replay=chosen;tokens=out['tokens']
    with torch.inference_mode():
        ids=torch.tensor(tokens)[None];logits,weights=model(ids,return_attn=True);normal=model(ids)
    attention=np.stack([w[0,0].numpy() for w in weights]);assert np.allclose(attention.sum(-1),1,atol=2e-6);assert np.max(np.triu(attention,1))==0
    # Reconfirm every generated next-token choice using a full causal pass.
    choices=normal[0,row.prompt_len-1:len(tokens)-1].argmax(-1).tolist()
    assert choices==tokens[row.prompt_len:],'Full forward disagrees with greedy free generation'
    np.savez_compressed(OUT/'attention.npz',attention=attention,tokens=np.array(tokens),prompt_len=row.prompt_len)
    aliases={v:chr(65+i) for i,v in enumerate(nt)}
    names=[aliases.get(t,tok.name(t).strip('<>')) for t in tokens]
    records=[];cursor=row.prompt_len
    for i,r in enumerate(parsed):
        end=tokens.index(tok.id(';'),cursor)+1
        records.append(dict(record=i+1,start=cursor,end=end,stack=[aliases[t] for t in r['stack']],
            visited=[aliases[t] for t in r['visited']],candidates=[aliases[t] for t in r['cands']],
            action=r['action']+(' '+aliases[r['arg']] if r['arg'] is not None else '')));cursor=end
    edges=list(zip(store.arrays['edge_src'][ix].astype(int).tolist(),store.arrays['edge_dst'][ix].astype(int).tolist()))
    metadata=dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),step=int(state['step']),
      dataset=str(DATA),index=ix,sample_id=int(store.arrays['sample_id'][ix]),split='test',n_nodes=16,n_edges=len(edges),
      node_mapping={chr(65+i):t for i,t in enumerate(nt)},raw_node_ids=store.arrays['node_token'][ix].astype(int).tolist(),
      source=chr(65+int(store.arrays['s_topo'][ix])),target=chr(65+int(store.arrays['t_topo'][ix])),
      edges=edges,records=records,token_labels=names,prompt_len=row.prompt_len,free_result=out,
      strict_replay=replay,full_forward_greedy_match=True,
      max_math_sdpa_logit_difference=float((logits-normal).abs().max()),
      mode='Actual greedy free generation; final answer correct, complete state trace strictly legal.',
      cpu_threads=2,precision='fp32',reference_equal=out['tokens']==row.tokens[:-1])
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (OUT/'trace.txt').write_text('\n'.join([f'Query {metadata["source"]}->{metadata["target"]}',metadata['mode'],'',
       *[f'R{r["record"]}: {" ".join(r["stack"])} | {" ".join(r["visited"])} | {" ".join(r["candidates"])} | {r["action"]} ;' for r in records],'YES'])+'\n')
    print('CHOSEN',ix,metadata['source'],metadata['target'],len(tokens),flush=True)

if __name__=='__main__':main()
