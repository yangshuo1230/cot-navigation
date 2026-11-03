"""Paired Linear/MLP error readout, frozen models, probe training on GPU0.

Readouts stop at the injected error token, or immediately before the action
token. Both are genuine input prefixes; no dummy or future token is fed.
All features stay in RAM and probe parameters are discarded after scoring.
"""
import argparse,os,runpy,time,json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',required=True,choices=['clean','error'])
parser.add_argument('--nav-gpu',required=True,type=int)
parser.add_argument('--steps',default=1000,type=int)
args=parser.parse_args()
assert args.nav_gpu!=0
os.environ['CUDA_VISIBLE_DEVICES']='0,'+str(args.nav_gpu)
os.environ['WANDB_MODE']='disabled';os.environ.setdefault('OMP_NUM_THREADS','2')
P=Path(__file__).resolve().parent;C=runpy.run_path(str(P/'injected_error_common.py'))
import numpy as np
import torch
from cot_navigation.analysis.error_awareness import train_probe,predict,evaluate


def extract(data,events,model,tok,device):
    """Each sample's forward input ends at its own readout position."""
    dim=model.d_model+12;n=len(events);x=np.zeros((n*2,10,dim),np.float32)
    rows=[(i,label,anchor,e) for i,e in enumerate(events) for label in (0,1) for anchor in (0,1)]
    def length(r):
        i,label,anchor,e=r
        return (e['action_start'] if anchor else e['anchor']+1)-int(label and e['kind']=='candidate_omission')
    rows.sort(key=length);start=0
    while start<len(rows):
        end=start+1
        while end<len(rows) and end-start<64 and (end-start+1)*length(rows[end])<=16384:end+=1
        batch=rows[start:end];seqs=[];cache={}
        for i,label,anchor,e in batch:
            key=(i,anchor)
            if key not in cache:cache[key]=data.action_pair(e) if anchor else data.pair(e)
            seqs.append(cache[key][label])
        tokens,lengths=C['M']['pad_rows'](seqs,tok.pad,device)
        capture={}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(tokens,capture=capture)
        bi=torch.arange(len(batch),device=device);pos=lengths-1
        with torch.no_grad():
            logp=logits[bi,pos-1].float().log_softmax(-1)
            chosen=tokens[bi,pos];picked=logp.gather(1,chosen[:,None]).squeeze(1)
            lm=torch.stack([-picked,-(logp.exp()*logp).sum(-1),logp.max(-1).values-picked],-1)
            deep=torch.stack([capture['resid'][li][bi,pos].float() for li in (3,4,5)],1)
            embed=model.embed.weight[chosen].float()
            observations=np.zeros((len(batch),9),np.float32)
            chosen_list=chosen.tolist();positions=pos.tolist()
            for b,(i,label,anchor,e) in enumerate(batch):
                token=chosen_list[b];p=positions[b];source=seqs[b][e['prompt_len']-3]
                observations[b,:6]=[p/4096,(p-e['prompt_len'])/4096,
                    float(tok.node_lo<=token<tok.node_lo+tok.n_node_tokens),
                    float(token==e['current']),float(token==e['target']),float(token==source)]
                observations[b,6+(2 if anchor else 0 if e['field']=='candidates' else 1)]=1
            obs=torch.as_tensor(observations,device=device)
            control=torch.cat([embed,obs,torch.zeros((len(batch),3),device=device)],-1)
            surprise=torch.cat([embed,obs,lm],-1)
            features=torch.cat([torch.nn.functional.pad(deep,(0,12)),control[:,None],surprise[:,None]],1).cpu().numpy()
        for b,(i,label,anchor,e) in enumerate(batch):x[2*i+label,5*anchor:5*anchor+5]=features[b]
        del capture,logits,tokens,features,deep
        start=end
        if start%1024<64 or start==len(rows):print('EXTRACT',args.arm,data.domain,start,'/',len(rows),flush=True)
    return dict(x=x,y=np.tile([0.,1.],n).astype(np.float32),
        query=np.repeat([e['query'] for e in events],2),event=np.repeat(np.arange(n),2),
        kind=np.repeat([e['kind'] for e in events],2),field=np.repeat([e['field'] for e in events],2))


def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-60,60)))


def metrics(data,z,val,vz):
    result=evaluate(data,z);p=sigmoid(z);vp=sigmoid(vz);y=data['y'].astype(bool);vy=val['y'].astype(bool)
    for hi,row in enumerate(result):
        threshold=float(np.quantile(vp[~vy,hi],.95))
        row.update(brier=float(((p[:,hi]-y)**2).mean()),
            bce=float((np.logaddexp(0,z[:,hi])-y*z[:,hi]).mean()),
            threshold_val_fpr05=threshold,test_fpr=float((p[~y,hi]>threshold).mean()),
            tpr_at_val_fpr05=float((p[y,hi]>threshold).mean()),
            paired_probability_margin=float((p[1::2,hi]-p[::2,hi]).mean()))
    return result


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.2,0);torch.cuda.set_per_process_memory_fraction(.15,1)
    navdevice=torch.device('cuda:1')
    cfg,tok,model,metadata=C['load_model'](args.arm,navdevice)
    dest=C['OUT']/f'probe_{args.arm}.json';C['OUT'].mkdir(exist_ok=True)
    datasets={d:C['Data'](d) for d in ('main','hard','deep')}
    features={};events={};sampling={};start=time.time()
    for domain in ('main','hard','deep'):
        data=datasets[domain]
        for split in (('train','val','test') if domain!='deep' else ('test',)):
            nq,npairs={'train':(4096,8192),'val':(512,2048),'test':(512,4096)}[split]
            ev,meta=data.events(split,nq,npairs)
            key=domain+'_'+split;events[key]=ev;sampling[key]=meta
            print('SAMPLE',args.arm,key,meta,flush=True)
            features[key]=extract(data,ev,model,tok,navdevice)
        ev,meta=C['repair_events'](data);key='repair_'+domain
        events[key]=ev;sampling[key]=meta;features[key]=extract(data,ev,model,tok,navdevice)
    # Graph/query identities are split within each source, not per token edit.
    for domain in ('main','hard'):
        qs=[set(features[domain+'_'+s]['query'].tolist()) for s in ('train','val','test')]
        assert not (qs[0]&qs[1] or qs[0]&qs[2] or qs[1]&qs[2])
    result=dict(finished=False,arm=args.arm,metadata=metadata,sampling=sampling,heads=C['HEADS'],
        probe_training_gpu=0,nav_feature_gpu=args.nav_gpu,steps=args.steps,seeds=[0,1,2],runs=[],score_pairs={})
    targets=['main_test','hard_test','deep_test','repair_main','repair_hard','repair_deep']
    for training in ('main','hard'):
        result['score_pairs'][training]={}
        for hidden,architecture in ((0,'linear'),(64,'mlp')):
            p_sums={k:np.zeros((len(features[k]['y']),2),np.float64) for k in targets}
            for seed in (0,1,2):
                fitted=train_probe(features[training+'_train'],features[training+'_val'],hidden,seed,args.steps,512,'cuda:0')
                assert fitted[1].device==torch.device('cuda:0')
                vz=predict(fitted,features[training+'_val']);scores={}
                for key in targets:
                    z=predict(fitted,features[key])
                    scores[key]=metrics(features[key],z,features[training+'_val'],vz)
                    if architecture=='mlp':p_sums[key]+=sigmoid(z[:,[2,7]])
                result['runs'].append(dict(training_domain=training,architecture=architecture,seed=seed,
                    selected_steps=fitted[3]['selected_steps'],val_bce=fitted[3]['best_val_bce'],scores=scores))
                print('PROBE FIT',args.arm,training,architecture,seed,flush=True)
                del fitted
            if architecture=='mlp':
                for key in targets:
                    probs=p_sums[key]/3
                    result['score_pairs'][training][key]=dict(
                        query=[e['query'] for e in events[key]],kind=[e['kind'] for e in events[key]],
                        at_error_correct=probs[::2,0].tolist(),at_error_wrong=probs[1::2,0].tolist(),
                        before_action_correct=probs[::2,1].tolist(),before_action_wrong=probs[1::2,1].tolist())
                # Keep a few hard-to-read real synthetic errors, not full activations.
                key=training+'_test';probs=p_sums[key]/3
                result.setdefault('low_error_score_examples',{})[training]=[]
                for i in np.argsort(probs[1::2,1])[:4]:
                    e=events[key][int(i)];good,bad=datasets[training].pair(e)
                    result['low_error_score_examples'][training].append(dict(query=e['query'],kind=e['kind'],
                        at_error_p=float(probs[2*i+1,0]),before_action_p=float(probs[2*i+1,1]),
                        clean_tail=tok.decode(good[-45:]),wrong_tail=tok.decode(bad[-45:])))
            result['seconds']=time.time()-start;C['write'](dest,result)
    result['finished']=True;result['seconds']=time.time()-start;C['write'](dest,result)
    print('PROBE FINISHED',args.arm,round(result['seconds'],1),flush=True)


if __name__=='__main__':main()
