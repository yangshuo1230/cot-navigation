"""Export both fully legal long native rollouts, without rerunning generation."""
import sys,json,gzip,hashlib,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'))
import analyze as a
OUT=ROOT/'studies/attention_long_20261031';OUT.mkdir(exist_ok=True)

def main():
    torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
    model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False)
    model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
    store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100'])
    cases=[x for x in json.loads((a.OUT/'free_rollouts.json').read_text()) if x['full_valid']]
    assert {x['index'] for x in cases}=={50708,50951}
    for case in cases:
        ix=case['index'];name='yes' if case['label'] else 'no';row=store.render(ix,'dfs_state',tok);P=row.prompt_len
        tokens=case['tokens'];ids=store.node_vocab_ids(ix,tok);mapping={int(v):f'N{i+1:02d}' for i,v in enumerate(ids)};inv={int(v):i for i,v in enumerate(ids)}
        recs=a.parse(tokens,P,tok)
        audit=a.replay_state(store.adj(ix),int(store.arrays['s_topo'][ix]),int(store.arrays['t_topo'][ix]),tokens[P:recs[-1]['end']],tok,inv)
        assert audit['valid'] and case['correct']
        labels=[mapping.get(v,tok.name(v).strip('<>')) for v in tokens]
        info=[dict(role=['graph_marker','graph_source','graph_target'][p%3],record=-1,description=f'边{p//3+1} / '+['EDGE','起点','终点'][p%3]) for p in range(P-4)]
        info.extend([dict(role='syntax',record=-1,description='查询标记 QUERY'),dict(role='source',record=-1,description='查询起点'),dict(role='target',record=-1,description='查询目标'),dict(role='syntax',record=-1,description='查询结束 |')])
        info.extend(dict(role='syntax',record=-1,description='格式符号') for _ in range(len(tokens)-P));records=[]
        for ri,r in enumerate(recs):
            for lo,hi,role in [(r['start'],r['a'],'stack'),(r['a']+1,r['b'],'visited'),(r['b']+1,r['c'],'candidates'),(r['c']+1,r['end']-1,'action')]:
                for p in range(lo,hi):
                    actual='argument' if role=='action' and tokens[p] in mapping else role
                    info[p]=dict(role=actual,record=ri,description=f'R{ri+1} '+{'stack':'stack','visited':'visited','candidates':'候选','action':'动作','argument':'动作参数'}[actual])
            for p,desc in [(r['a'],'stack结束 |'),(r['b'],'visited结束 |'),(r['c'],'候选结束 |'),(r['end']-1,'记录结束 ;')]:info[p]=dict(role='syntax',record=ri,description=f'R{ri+1} {desc}')
            records.append(dict(record=ri+1,start=r['start'],end=r['end'],a=r['a'],b=r['b'],c=r['c'],stack=[mapping[v] for v in r['stack']],visited=[mapping[v] for v in r['visited']],candidates=[mapping[v] for v in r['candidates']],action=r['action'],argument=mapping.get(r['argument'])))
        info[-1]=dict(role='answer',record=len(recs)-1,description='最终答案')
        with torch.inference_mode():logits,weights=model(torch.tensor(tokens)[None],return_attn=True)
        argmax=logits[0,P-1:-1].argmax(-1).numpy();assert np.array_equal(argmax,np.array(tokens[P:]))
        att=np.stack([w[0,0].numpy() for w in weights]).astype('<f4');assert np.allclose(att.sum(-1),1,atol=3e-6)
        raw=att.tobytes();(OUT/f'{name}.attention.f32.gz').write_bytes(gzip.compress(raw,compresslevel=6))
        edges=[[inv[tokens[p+1]],inv[tokens[p+2]]] for p in range(0,P-4,3)]
        summary=dict(index=ix,label=case['label'],records=len(records),tokens=len(tokens),output_tokens=len(tokens)-P,
            max_depth=max(len(r['stack']) for r in records),visited=max(len(r['visited']) for r in records),
            pushes=sum(r['action']=='PUSH' for r in records),pops=sum(r['action']=='POP' for r in records),
            branching_records=sum(len(r['candidates'])>1 for r in records),revisits=len(records)-len(set(r['stack'][-1] for r in records)),full_valid=True,all_free_argmax_match=True)
        data=dict(labels=labels,raw_ids=tokens,info=info,prompt_len=P,records=records,graph=dict(labels=list(mapping.values()),edges=edges),
            source=labels[P-3],target=labels[P-2],default_position=recs[len(recs)//2]['c'],summary=summary,
            attention_shape=list(att.shape),attention_sha256=hashlib.sha256(raw).hexdigest(),checkpoint=str(a.RUN/'checkpoints/best.pt'))
        (OUT/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
