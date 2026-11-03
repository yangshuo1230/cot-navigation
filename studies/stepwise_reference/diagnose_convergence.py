"""Read-only full training-set CE and empirical prefix-entropy diagnostic."""
from pathlib import Path
from collections import Counter,defaultdict
import json,math,os,datetime
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn.functional as F
import train
HERE=Path(__file__).resolve().parent
OUT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/outputs/stepwise_reference')
torch.set_num_threads(2)
results=[]
for j in json.loads((HERE/'manifest.json').read_text())['jobs']:
    d=OUT/j['id'];w=json.loads((HERE/'data'/f"graph_{j['graph']}.json").read_text())
    records=w['splits'][j['mode']]['sequences']
    seqs=[train.sequence(r,w['node_names'],j['protocol']) for r in records]
    # The finite empirical distribution is the actual sequence-uniform sampler pool.
    counts=defaultdict(Counter);kind_by_prefix={}
    def kind(seq,i):
        return 'prompt' if i<3 else 'answer' if i==len(seq)-2 else 'eos' if i==len(seq)-1 else 'path'
    for seq in seqs:
        for i in range(1,len(seq)):
            prefix=tuple(seq[:i]);counts[prefix][seq[i]]+=1;kind_by_prefix[prefix]=kind(seq,i)
    floor_sum=defaultdict(float);den=Counter()
    for prefix,c in counts.items():
        n=sum(c.values());k=kind_by_prefix[prefix]
        floor_sum[k]+=sum(-v*math.log(v/n) for v in c.values());den[k]+=n
    blob=torch.load(d/'latest.pt',map_location='cpu',weights_only=False)
    m=train.model_factory(j).cuda().eval();m.load_state_dict(blob['model']);del blob
    sums=defaultdict(float);correct=Counter();total=Counter()
    with torch.no_grad():
        for start in range(0,len(seqs),256):
            part=seqs[start:start+256]
            x=torch.full((len(part),16),train.PAD,dtype=torch.long,device='cuda')
            for i,s in enumerate(part):x[i,:len(s)]=torch.tensor(s,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(x[:,:-1])
            loss=F.cross_entropy(logits.float().reshape(-1,train.VOCAB),x[:,1:].reshape(-1),reduction='none').view(len(part),-1).cpu()
            pred=logits.argmax(-1).cpu()
            for b,s in enumerate(part):
                for i in range(1,len(s)):
                    k=kind(s,i);sums[k]+=float(loss[b,i-1]);correct[k]+=int(pred[b,i-1])==s[i];total[k]+=1
    n=sum(total.values())
    result=dict(job=j['id'],step=125000,n_sequences=len(seqs),n_targets=n,
        full_train_ce=sum(sums.values())/n,empirical_prefix_entropy=sum(floor_sum.values())/n,
        excess_ce=(sum(sums.values())-sum(floor_sum.values()))/n,
        parts={k:dict(n=total[k],ce=sums[k]/total[k],entropy_floor=floor_sum[k]/total[k],teacher_forced_accuracy=correct[k]/total[k]) for k in total})
    results.append(result);print(json.dumps(result),flush=True)
    del m
report=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=results,
    note='Checkpoint read only. CE includes prompt tokens; prefix entropy is the sequence-uniform empirical conditional entropy, not an assertion of global parameter optimization convergence.')
(OUT/'convergence_full_train.json').write_text(json.dumps(report,indent=2))
