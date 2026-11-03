"""Same-test comparison of natural-only, synthetic-only and mixed error heads.

Natural mistakes are often omitted candidates/actions, whereas controlled
corruptions replace graph nodes. A weak MIXED head alone cannot establish
absence of an error signal; this checks the probe-training distribution.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('studies/mechanistic_followup/results'))
    p.add_argument('--gpu',default='0');p.add_argument('--models',nargs='+',default=['context_dfs','context_bfs'])
    args=p.parse_args()
    if args.gpu!='0':p.error('Training is authorized on physical GPU0')
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
    import numpy as np
    import torch
    from .common import AnalysisModel,SEED,write_json,sha256
    from .patching import free_rollouts
    from .probes import fit,subset
    from .set_metrics import evaluate_errors
    from .error_detection import events,extract_events
    torch.set_num_threads(4)
    path=args.input/'error_training_controls.json'
    result=json.loads(path.read_text()) if path.exists() else {'models':{},'description':__doc__}
    selected=json.loads((args.input/'selection.json').read_text())['models']
    for entry in selected:
        if entry['id'] not in args.models:continue
        old=json.loads((args.input/'models'/f'{entry["id"]}.json').read_text())
        if entry['id'] in result['models']:
            assert sha256(Path(entry['run'])/'checkpoints/best.pt')==result['models'][entry['id']]['checkpoint_sha256']
            print('SKIP',entry['id'],flush=True);continue
        print('ERROR TRAINING CONTROL',entry['id'],flush=True)
        am=AnalysisModel(entry,Path.cwd())
        assert am.checkpoint['sha256']==old['checkpoint']['sha256']
        splits={'train':am.examples('train',1024),'val':am.examples('val',256),'test':am.examples('test',256)}
        event_sets={};inventory={}
        for split,exs in splits.items():
            count=128 if split=='train' else 64 if split=='val' else len(exs)
            free,rr=free_rollouts(am,exs[:count],max_new=512 if split!='test' else 0)
            event_sets[(split,'natural')]=[e for ex in exs[:count] if (e:=events(ex,am.protocol,am.tok,free[ex.index]))]
            reference_count=256 if split=='train' else 128 if split=='val' else len(exs)
            event_sets[(split,'synthetic')]=[e for ex in exs[:reference_count]
                if (e:=events(ex,am.protocol,am.tok,ex.row.tokens,synthetic=True,seed=SEED))]
            inventory[split]={'natural_pairs':len(event_sets[(split,'natural')]),
                              'synthetic_pairs':len(event_sets[(split,'synthetic')]),'truncated':sum(r['limit'] for r in rr)}
            assert inventory[split]['natural_pairs']==old['errors']['natural_inventory'][split]['first_errors']
        record={'checkpoint_sha256':am.checkpoint['sha256'],'inventory':inventory,'readouts':{}}
        for offset,readout in [(0,'wrong_token'),(1,'next_token')]:
            print(' FIT',readout,flush=True)
            data={(s,src):extract_events(am,exs,event_sets[(s,src)],offset)
                  for s,exs in splits.items() for src in ['natural','synthetic']}
            # At +1, remembering the previous token's ID is weaker than knowing
            # that it was WRONG. Add a previous-token embedding control rather
            # than treating a win over the identical current-token control as
            # proof of semantic error recognition.
            for key,d in data.items():
                if d is None:continue
                event_map={e['query']:e for e in event_sets[key]}
                ex_map={e.index:e for e in splits[key[0]]}
                controls=[]
                for i,q in enumerate(d['query']):
                    e=event_map[q];ex=ex_map[q]
                    token=e['wrong_token'] if d['y'][i,0] else e['correct_token']
                    emb=am.model.embed.weight[token].float().cpu().numpy()
                    obs=d['x'][i,3,128:].copy()
                    obs[2:4]=[token in ex.nodes,token==ex.nodes[ex.target]]
                    controls.append(np.r_[emb,obs])
                d['x']=np.concatenate([d['x'],np.asarray(controls)[:,None,:]],axis=1)
            te=data[('test','natural')]
            out={'natural_test_pairs':len(te['y'])//2,
                 'layers':[4,5,6,'current_token_control','previous_token_control'],'heads':{}}
            for source in ['natural','synthetic']:
                tr,va=data[('train',source)],data[('val',source)]
                if tr is None or va is None or len(tr['y'])<8 or len(va['y'])<8:
                    out['heads'][source]={'status':'insufficient training/validation errors'};continue
                fitted=fit(tr,va,device=am.device,seed=SEED)
                row={'train_pairs':len(tr['y'])//2,'val_pairs':len(va['y'])//2,
                     'selected_steps':fitted['steps'],'natural_test':evaluate_errors(fitted,te)}
                if offset:
                    valid=te['repair_next_valid'];row['strict_pairs']=int(valid.sum()//2)
                    if valid.any():row['strict_test']=evaluate_errors(fitted,subset(te,valid))
                    semantic=valid & (te['kind']=='in_graph_node')
                    row['strict_in_graph_pairs']=int(semantic.sum()//2)
                    if semantic.any():row['strict_in_graph_test']=evaluate_errors(fitted,subset(te,semantic))
                row['in_graph_pairs']=int((te['kind']=='in_graph_node').sum()//2)
                if row['in_graph_pairs']:row['in_graph_test']=evaluate_errors(fitted,subset(te,te['kind']=='in_graph_node'))
                out['heads'][source]=row
            mixed=old['errors']['readouts'][readout]['tests']['natural']
            out['heads']['mixed']={'natural_test':mixed['all'],
                'strict_pairs':mixed.get('strict_repaired_next_legal_pairs'),
                'strict_test':mixed.get('strict_repaired_next_legal')}
            record['readouts'][readout]=out
            del data
        result['models'][entry['id']]=record
        result['source_sha256']=sha256(Path(__file__))
        write_json(path,result)
        del am
        torch.cuda.empty_cache()
    print('CONTROL COMPLETE',flush=True)


if __name__=='__main__':main()
