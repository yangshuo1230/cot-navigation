"""Natural-only prefix classification, distinct from the strict paired test.

Both classes are unedited model generations. Label=1 means a semantic error
has occurred at or before readout; label=0 means the entire prefix is legal.
This includes propagated errors, so it does NOT isolate historical memory.
Match clean readouts by grammatical field, token identity when available,
and position. One positive and at most one negative per query/anchor. All
queries stay in their authoritative train/validation/test split.
"""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import os
from pathlib import Path
import time

ANCHORS=['error','token_1','token_4','token_16','token_64','record_1_vis','record_2_vis','record_4_vis']
FIELDS=['frontier','visited','candidates','action','argument','semicolon']
HEADS=['L1','L2','L3','L4','L5','L6','token_position_control','position_control']


def grammar_field(seq,p,pos,tok):
    """Syntactic field only, without assuming any corrupted state is correct."""
    semi,bar=tok.id(';'),tok.id('|')
    start=pos
    while start>p and seq[start-1]!=semi: start-=1
    bars=seq[start:pos].count(bar)
    if bars<3: return FIELDS[bars]
    if seq[pos]==semi: return 'semicolon'
    return 'action' if pos>0 and seq[pos-1]==bar else 'argument'


def build(am,examples,outputs):
    import numpy as np
    from cot_navigation.analysis.error_detection import semantic_slots
    rng=np.random.default_rng(2026102231)
    pool={f:[] for f in FIELDS}; positives={a:[] for a in ANCHORS}
    sequences={}; errors=Counter()
    terminal={am.tok.yes,am.tok.no,am.tok.id('<EOS>'),am.tok.id('<FOUND>'),am.tok.id('<EXHAUST>')}
    if '<STOP>' in am.tok.stoi: terminal.add(am.tok.id('<STOP>'))
    for ex,out in zip(examples,outputs):
        full=ex.prompt+out['trace_tokens']; p=len(ex.prompt)
        end=next((i for i in range(p,len(full)) if full[i] in terminal),len(full))
        seq=full[:end]; sequences[ex.index]=seq
        slots=list(semantic_slots(ex,am.protocol,am.tok,seq[p:]))
        bad=next(((p+i,f) for i,good,f in slots if seq[p+i] not in good),None)
        # Keep all legal positions as candidates; matching uses no activations
        # and no held-out labels beyond the semantic dataset construction.
        for i,good,field in slots:
            pos=p+i
            if seq[pos] not in good: break
            pool[field].append((ex.index,pos,p,seq[pos]))
        if bad is None: continue
        tau,field=bad; errors[field]+=1
        positions={'error':tau,**{f'token_{k}':tau+k for k in (1,4,16,64)}}
        starts=[i+1 for i in range(tau,len(seq)) if seq[i]==am.tok.id(';')]
        for k in (1,2,4):
            pos=len(seq)
            if len(starts)>=k:
                start=starts[k-1]
                stop=next((i for i in range(start,len(seq)) if seq[i]==am.tok.id(';')),len(seq))
                bars=[i for i in range(start,stop) if seq[i]==am.tok.id('|')]
                if len(bars)>=2: pos=bars[1]
            positions[f'record_{k}_vis']=pos
        for anchor,pos in positions.items():
            if pos<len(seq):
                positives[anchor].append((ex.index,pos,p,seq[pos],grammar_field(seq,p,pos,am.tok),tau))
    rows={}; info={}
    for anchor, items in positives.items():
        rng.shuffle(items); used=set(); selected=[]; exact=0
        for query,pos,p,token,field,tau in items:
            available=[r for r in pool[field] if r[0] not in used]
            if not available: continue
            same=[r for r in available if r[3]==token]
            options=same if same else available
            # Prefer comparable reasoning and prompt lengths. Ties use a fixed
            # randomized pool permutation rather than model feature values.
            q2,pos2,p2,t2=min(options,key=lambda r:abs((r[1]-r[2])-(pos-p))+abs(r[2]-p))
            used.add(q2); exact+=int(t2==token)
            selected.extend([dict(query=query,pos=pos,prompt=p,field=field,y=1,error=tau),
                             dict(query=q2,pos=pos2,prompt=p2,field=field,y=0,error=None)])
        rows[anchor]=selected
        info[anchor]=dict(pairs=len(selected)//2,exact_token_matches=exact,
            distinct_queries=len({r['query'] for r in selected}),
            positive_fields=dict(Counter(r['field'] for r in selected if r['y'])),
            mean_positive_delay=float(np.mean([r['pos']-r['error'] for r in selected if r['y']])) if selected else None)
    return rows,sequences,dict(first_error_fields=dict(errors),anchors=info)


def extract(am,rows,sequences):
    import numpy as np
    import torch
    # Forward each original generated sequence once, retaining only selected
    # endpoints. FP32 avoids the earlier cross-batch BF16 pairing artifact.
    locations={}
    for anchor,items in rows.items():
        for i,r in enumerate(items): locations.setdefault(r['query'],[]).append((anchor,i,r))
    dim=am.cfg.model.d_model; width=dim+4+len(FIELDS)
    output={a:dict(x=np.zeros((len(rs),len(HEADS),width),np.float32),
        y=np.array([r['y'] for r in rs],np.float32),
        query=np.array([r['query'] for r in rs]),
        field=np.array([r['field'] for r in rs]),kind=np.array(['natural_prefix']*len(rs))) for a,rs in rows.items()}
    table=am.model.embed.weight.detach().float().cpu().numpy()
    ids=sorted(locations,key=lambda q:len(sequences[q]))
    for start in range(0,len(ids),8):
        batch=ids[start:start+8]; lengths=[len(sequences[q]) for q in batch]
        tokens=torch.full((len(batch),max(lengths)),am.tok.pad,dtype=torch.long,device=am.inference_device)
        for b,q in enumerate(batch): tokens[b,:lengths[b]]=torch.tensor(sequences[q],device=am.inference_device)
        capture={}
        with torch.inference_mode(): am.model(tokens,capture=capture)
        for b,q in enumerate(batch):
            for anchor,i,r in locations[q]:
                pos,p=r['pos'],r['prompt']; data=output[anchor]['x']
                data[i,:6,:dim]=torch.stack([h[b,pos].float() for h in capture['resid']]).cpu().numpy()
                meta=np.zeros(4+len(FIELDS),np.float32)
                meta[:4]=[pos/4096,(pos-p)/4096,p/4096,sequences[q][p:pos].count(am.tok.id(';'))/100]
                meta[4+FIELDS.index(r['field'])]=1
                data[i,6,:dim]=table[sequences[q][pos]]
                data[i,6:,-len(meta):]=meta
        del capture,tokens
    return output


def metrics(data,logits):
    import numpy as np
    from cot_navigation.analysis.set_metrics import auc_ap
    y=data['y'].astype(bool); result=[]
    for i,head in enumerate(HEADS):
        z=logits[:,i]
        result.append(dict(layer=head,n=len(y),errors=int(y.sum()),queries=len(set(data['query'].tolist())),
            **auc_ap(y,z),balanced_accuracy=float(((z[y]>=0).mean()+(z[~y]<0).mean())/2)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--models',nargs='+',required=True)
    args=parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu); os.environ['WANDB_MODE']='disabled'
    import torch
    from cot_navigation.analysis.common import write_json,sha256
    from cot_navigation.analysis.scale_followup import make_adapter
    from cot_navigation.analysis.error_awareness import train_probe,predict
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    root=Path.cwd(); folder=root/'studies/natural_history_error'
    entries=json.loads((root/'studies/scale_mechanistic/results/selection.json').read_text())['models']
    for entry in entries:
        if entry['id'] not in args.models: continue
        output=folder/'stream_results/models'/f"{entry['id']}.json"
        if output.exists(): continue
        am=make_adapter(entry,root,'cuda:0'); datasets={}; summaries={}; endpoints={}; sets=[]
        for split in ('train','val','test'):
            with gzip.open(folder/'rollouts'/f"{entry['id']}_{split}.json.gz",'rt') as f: cache=json.load(f)
            assert cache['checkpoint_sha256']==entry['checkpoint_sha256']
            examples=[am.example(q,split) for q in cache['query_ids']]; sets.append(set(cache['query_ids']))
            rows,sequences,info=build(am,examples,cache['outputs'])
            datasets[split]=extract(am,rows,sequences); summaries[split]=info; endpoints[split]=rows
            print('DATA',entry['id'],split,info,flush=True)
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        fits=[]
        for anchor in ANCHORS:
            counts={s:len(d[anchor]['y'])//2 for s,d in datasets.items()}
            if counts['train']<20 or counts['val']<8 or counts['test']<8:
                print('SKIP',anchor,counts,flush=True); continue
            for hidden,architecture in ((0,'linear'),(64,'mlp64')):
                for seed in (0,1,2):
                    fitted=train_probe(datasets['train'][anchor],datasets['val'][anchor],hidden,seed,600,512,'cuda:0')
                    fits.append(dict(anchor=anchor,architecture=architecture,seed=seed,sizes=counts,
                        test=metrics(datasets['test'][anchor],predict(fitted,datasets['test'][anchor]))))
                    del fitted
            print('FIT',entry['id'],anchor,counts,flush=True)
        assert sha256(am.run/'checkpoints/best.pt')==entry['checkpoint_sha256']
        write_json(output,dict(complete=True,entry=entry,dataset=summaries,fits=fits,endpoints=endpoints,
            source_sha256=sha256(Path(__file__)),precision='fp32',checkpoint_unchanged=True,
            interpretation='Natural histories: detects prior-or-current semantic violation, including propagated consequences; does not isolate memory of a past error.'))
        print('DONE',entry['id'],flush=True)
        del am,datasets; torch.cuda.empty_cache()


if __name__=='__main__': main()
