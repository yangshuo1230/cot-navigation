"""Factor short/long task distance and heldout endpoint block without training.

All 36 completed checkpoints, native original BF16 evaluator on idle GPU1.
Original core scores reused; only extra diagnostic panels newly evaluated.
Never save activations, weights/checkpoints, W&B runs or full predictions.
"""
from pathlib import Path
import json,sys,importlib.util,random,collections,hashlib
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path.cwd();HERE=ROOT/'studies/exposure_training_diagnosis'
SUITE=ROOT/'studies/fixed_adjacency_suite';MODELS=ROOT/'outputs/exposure_grokking_adj_v3'
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('adj_training',SUITE/'train_suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer

def make_panels(world):
    n=world['n_nodes'];width=world['width'];sourceB=set(world['source_B']);targetB=set(world['target_B'])
    reachable=[]
    for s in range(n):
        seen={s};queue=[s]
        for u in queue:
            for v in world['adj'][u]:
                if v not in seen:seen.add(v);queue.append(v)
        reachable.append(seen)
    shortBB=[];long_nonBB={}
    for s in range(n):
        for t in range(n):
            gap=t//width-s//width
            if gap<=0:continue
            label=int(t in reachable[s]);row=[s,t,label,gap];bb=s in sourceB and t in targetB
            if gap<=2 and bb:shortBB.append(row)
            if gap>=6 and not bb:long_nonBB.setdefault((gap,label),[]).append(row)
    rng=random.Random(7721+world['seed']);long_rows=[]
    for key,count in collections.Counter((r[3],r[2]) for r in world['core_final']).items():
        assert len(long_nonBB[key])>=count
        long_rows.extend(rng.sample(long_nonBB[key],count))
    rng.shuffle(long_rows)
    return dict(full_train=world['base']['train'],short_nonBB=world['base']['final'],short_BB=shortBB,long_nonBB=long_rows)

def summarize(value,world,samples,protocol):
    metrics={k:value[k] for k in ('n','accuracy','balanced_accuracy','yes_recall','no_recall')}
    metrics['by_gap']=value['by_gap']
    if protocol!='path':return metrics
    inv={v:i for i,v in enumerate(samples.nodes)}
    positives=[r for r in value['predictions'] if r['query'][2]==1];valid=[];no_trace=[];length=[];examples=[]
    for row in positives:
        trace=[inv.get(v,-1) for v in row['generated'] if v not in (samples.tok.yes,samples.tok.no,samples.tok.id('<EOS>'))]
        s,t=row['query'][:2];legal=bool(trace and trace[0]==s and trace[-1]==t and all(u>=0 for u in trace))
        if legal:legal=all(v in world['adj'][u] for u,v in zip(trace,trace[1:]))
        valid.append(legal);no_trace.append(not trace);length.append(len(trace))
        if row['correct'] and not legal and len(examples)<2:examples.append(row)
    metrics.update(positive_legal_goal_rate=float(np.mean(valid)),positive_no_trace_rate=float(np.mean(no_trace)),
        positive_mean_trace_nodes=float(np.mean(length)),correct_answer_but_invalid_witness_examples=examples)
    return metrics

@torch.inference_mode()
def main():
    device=torch.device('cuda:1');tok=Tokenizer(256);records=[];panel_meta=[]
    for graph in range(3):
        world=json.loads((SUITE/f'data/graph_{graph}.json').read_text());panels=make_panels(world)
        panel_meta.append(dict(graph=graph,panels={name:dict(n=len(rows),n_positive=sum(r[2] for r in rows),
            gap_counts=dict(collections.Counter(r[3] for r in rows)),sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest()) for name,rows in panels.items()}))
        for seed in range(2):
            for protocol in ('direct','path'):
                samples=suite.Samples(world,protocol)
                for arm in ('R0','R2','R3'):
                    folder=MODELS/f'fixed/g{graph}_s{seed}/{protocol}/{arm}';job=json.loads((folder/'resolved_job.json').read_text())
                    blob=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False);assert blob['step']==125000
                    model=TransformerLM(len(tok),6,1,128,512,256,dropout=job['dropout']).eval().requires_grad_(False)
                    model.load_state_dict(blob['model']);del blob;model.to(device)
                    entry=dict(id=job['id'],graph=graph,model_seed=seed,protocol=protocol,arm=arm,panels={})
                    for name,rows in panels.items():
                        value=suite.evaluate(model,samples,rows,device,batch_size=128)
                        entry['panels'][name]=summarize(value,world,samples,protocol)
                    final=json.loads((folder/'final.json').read_text())
                    entry['panels']['long_BB_core']=summarize(final,world,samples,protocol)
                    records.append(entry);(HERE/'panels.json').write_text(json.dumps(dict(panel_definitions=panel_meta,jobs=records,
                        method='Frozen latest step125k, native original BF16 evaluator; no training. short_nonBB is original base final; short_BB all excluded block pairs gap1/2; long_nonBB gap>=6 matched to core gap/label counts; long_BB is saved original core final.'),indent=2))
                    print(job['id'],'COMPLETE',flush=True)
                    del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
