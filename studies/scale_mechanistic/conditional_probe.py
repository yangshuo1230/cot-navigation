"""Probe VIS filtering only among actual neighbors, excluding adjacency shortcuts.

No free generation is needed. Reference test states use the full original probe
query split. Saved natural states are a small, explicitly selected case subset.
"""
from pathlib import Path
import argparse
import json
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('studies/scale_mechanistic/results'))
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
    import numpy as np
    import torch
    from cot_navigation.analysis.common import SEED,sha256,write_json
    from cot_navigation.analysis.scale_followup import make_adapter,extract
    from cot_navigation.analysis.probes import fit,evaluate,shuffled_labels,subset
    from cot_navigation.analysis.set_metrics import evaluate_sets
    torch.set_num_threads(3)
    for f in sorted((args.output/'models').glob('*.json')):
        d=json.loads(f.read_text());entry=d['entry']
        if 'elapsed_seconds' not in d or not entry['protocol'].endswith('_state') or 'conditional_probe' in d:continue
        print('CONDITIONAL',entry['id'],flush=True)
        am=make_adapter(entry,Path.cwd())
        assert am.checkpoint['sha256']==entry['checkpoint_sha256']
        splits={s:[am.example(q,s) for q in d['query_ids'][s]] for s in ['train','val','test']}
        data={}
        for split,examples in splits.items():
            features=extract(am,examples)
            allowed=features.get('allowed_successors');successors=features.get('successors')
            if allowed is None or successors is None:continue
            def key(item,i):return (int(item['query'][i]),int(round(float(item['x'][i,-1,-4])*am.model.max_seq_len)))
            neighbors={key(successors,i):successors['y'][i] for i in range(len(successors['y']))}
            mask=np.stack([neighbors[key(allowed,i)] for i in range(len(allowed['y']))])
            assert not np.any(allowed['y']>mask)
            data[split]=subset({**allowed,'mask':mask},mask.any(-1))
            del features
        if any(s not in data or len(data[s]['y'])==0 for s in ['train','val','test']):
            d['conditional_probe']={'status':'insufficient reference states'};write_json(f,d);continue
        head=fit(data['train'],data['val'],device=am.device,seed=SEED,steps=250)
        ctrl=fit(data['train'],data['val'],device=am.device,seed=SEED+17,steps=250,shuffled=True)
        result={'source_sha256':sha256(Path(__file__)),'probe_gpu':0,
                'label':'unvisited among real outgoing neighbors; all non-neighbors masked out',
                'reference_queries':{s:len(ex) for s,ex in splits.items()},
                'selected_steps':head['steps'],'reference':evaluate(head,data['test']),
                'reference_sets':evaluate_sets(head,data['test']),
                'train':evaluate(head,data['train']),
                'reference_shuffled_sets':evaluate_sets(ctrl,{**data['test'],'y':shuffled_labels(data['test'],SEED+219)})}
        cases={}
        for variant in ['including_goal','excluding_goal']:
            for c in d.get('blocking',{}).get(variant,{}).get('interesting_examples',[]):
                prefix=c['prefix_tokens']['clean'];cases[(c['query'],len(prefix))]=c
        examples={e.index:e for e in splits['test']};items=[]
        for (query,_),case in cases.items():
            ex=examples[query];prefix=case['prefix_tokens']['clean'];current=case['detail']['frontier'][0 if am.protocol=='bfs_state' else -1]
            inv={v:i for i,v in enumerate(ex.nodes)}
            if current not in inv:continue
            end=len(prefix)-1;start=end
            while start>0 and prefix[start-1]!=am.tok.id('|'):start-=1
            vis=set(prefix[start:end]);neighbor={ex.nodes[v] for v in ex.adj[inv[current]]}
            if not neighbor:continue
            _,resid,_=am.forward([prefix]);pos=len(prefix)-1
            h=torch.nn.functional.pad(torch.stack([resid[l][0,pos].float() for l in am.layers]),(0,4))
            emb=am.model.embed.weight[current].float();seen=ex.nodes[ex.target] in prefix[ex.row.prompt_len:]
            obs=emb.new_tensor([pos/am.model.max_seq_len,(pos-ex.row.prompt_len)/am.model.max_seq_len,float(seen),float(current==ex.nodes[ex.target])])
            x=torch.cat([h,torch.cat([emb,obs])[None]],0).cpu().numpy()
            mask=np.zeros(am.tok.n_node_tokens,np.float32);y=np.zeros_like(mask)
            for v in neighbor:mask[v-am.tok.node_lo]=1
            for v in neighbor-vis:y[v-am.tok.node_lo]=1
            items.append(dict(x=x,y=y,mask=mask,query=query,stage='saved_natural_state',goal_seen=seen))
        if items:
            te={k:np.asarray([i[k] for i in items]) for k in items[0]}
            result['saved_natural_selected_cases']={'n_states':len(items),'n_queries':len(set(i['query'] for i in items)),
                'selection_note':'at most 8 stored cases per blocking variant, selected partly for answer changes; not representative test frequencies',
                'metrics':evaluate(head,te),'sets':evaluate_sets(head,te)}
        result['checkpoint_unchanged']=sha256(am.run/'checkpoints/best.pt')==entry['checkpoint_sha256']
        assert result['checkpoint_unchanged']
        d['conditional_probe']=result;write_json(f,d)
        print(' DONE',entry['id'],flush=True)
        del am,head,ctrl,data;torch.cuda.empty_cache()


if __name__=='__main__':main()
