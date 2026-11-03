"""Warm-start pure Direct at one hop and learn two/three hops with replay.

One process per physical GPU1-7. Parent model weights are loaded exactly;
all arms start a fresh optimizer so BEST and LAST parents are comparable.
Only new experiment outputs are written, with resumable optimizer/RNG state.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,math,os,signal,time,uuid
import numpy as np
import torch
import torch.nn.functional as F
from data import Pools,atomic_json,digest
from core_adapter import core
from cot_navigation.config import to_plain
from cot_navigation.models.transformer import build_model
from cot_navigation.training.trainer import _optimizer
from cot_navigation.tracking.logger import Tracker


def cpu_preflight():return core.cpu_preflight()


@torch.inference_mode()
def evaluate(raw,pools,full=False):
    raw.eval();report={}
    for split in ['train','val']+(['test'] if full else []):
        parts={}
        for h in (1,2,3):
            p=pools.get(split,h,full=full)
            m=core.score(raw,p);m['balanced_accuracy']=(m['yes_recall']+m['no_recall'])/2
            parts[str(h)]=m
        report[split]=parts
    report['selection_score']=sum(report['val'][str(h)]['balanced_accuracy'] for h in (1,2,3))/3
    if full:
        report['counterfactual']={}
        report['test_reverse']={}
        for h in (1,2,3):
            report['test_reverse'][str(h)]=core.score(raw,pools.get('test',h,full=True),reverse=True)
            p=pools.counter[h];m,pred,prob=core.score(raw,p,return_arrays=True)
            truth=p['y'].cpu().numpy();ok=(pred==truth).reshape(-1,2)
            m['pair_both_correct']=float(ok.all(-1).mean())
            pp=prob.reshape(-1,2);yy=truth.reshape(-1,2)
            m['signed_delta_p']=float(np.mean((pp[:,0]-pp[:,1])*(2*yy[:,0]-1)))
            report['counterfactual'][str(h)]=m
    raw.train();return report


def gate(report):
    train2=report['train']['2'];val2=report['val']['2'];val1=report['val']['1']
    return (train2['accuracy']>=.99 and train2['ce']<=.02 and val2['balanced_accuracy']>=.95
            and min(val2['yes_recall'],val2['no_recall'])>=.90 and val1['balanced_accuracy']>=.95)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=float)
    args=parser.parse_args();spec=json.loads(args.spec.read_text());out=args.spec.parent
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in [str(i) for i in range(1,8)]:
        raise RuntimeError('Only physical GPU1-7 allowed')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(spec['seed']);np.random.seed(spec['seed'])
    cfg=core.config(spec,out)
    cfg.tracking.notes=json.dumps({'parent':spec['parent_checkpoint'],'scheme':spec['scheme'],
                                  'weights':spec['stages'],'fresh_optimizer':True,'pure_direct':True})
    cfg.train.query_hops='all'
    import yaml
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(to_plain(cfg),sort_keys=False))
    raw=build_model(cfg.model,270,0).cuda()
    latest=out/'checkpoints/latest.pt';(out/'checkpoints').mkdir(exist_ok=True)
    if not latest.exists() and spec['parent_checkpoint']:
        parent=Path(spec['parent_checkpoint'])
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==spec['parent_sha256']
        saved=torch.load(parent,map_location='cpu',weights_only=False)
        raw.load_state_dict(saved['model'],strict=True)
        parent_step=saved['step'];del saved
    else:parent_step=None
    opt,opt_info=_optimizer(raw,cfg,torch.device('cuda'))
    rng=torch.Generator(device='cuda').manual_seed(121000+spec['seed'])
    edge_rng=torch.Generator(device='cuda').manual_seed(122000+spec['seed'])
    pools=Pools(spec['data_dir'])
    state={'step':0,'stage':0,'stage_start_step':0,'consumed_seconds':0.,'gate_streak':0,
           'best_score':-1.,'run_id':uuid.uuid4().hex,'transitions':[],
           'exposure':{'1':0,'2':0,'3':0},'stage_exposure':{},'parent_step':parent_step}
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        assert saved['spec_sha256']==digest(spec)
        raw.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);state=saved['suite_state']
        rng.set_state(saved['data_rng']);edge_rng.set_state(saved['edge_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng']);del saved
    else:
        atomic_json(out/'initialization.json',{'model_sha256':core.hash_model(raw),
                                             'parent_checkpoint':spec['parent_checkpoint'],
                                             'parent_sha256':spec.get('parent_sha256'),
                                             'parent_step':parent_step,'fresh_optimizer':True,
                                             'parameters':raw.report()})
    before=state['consumed_seconds'];started=time.monotonic()
    remaining=max(0,min(spec['budget_seconds']-before,args.budget_seconds if args.budget_seconds is not None else spec['budget_seconds']))
    tracker=Tracker(cfg,state['run_id'],spec.get('parent_run_id'),
                    {'optimizer':opt_info,'parameters':raw.report(),'torch_compile':True},resume=latest.exists())
    if tracker.run:
        tracker.run.define_metric('multihop/*',step_metric='optimizer_step',hidden=True)
        for k in ['multihop/val_h1_accuracy','multihop/val_h2_accuracy','multihop/val_h3_accuracy','multihop/stage']:
            tracker.run.define_metric(k,step_metric='optimizer_step',hidden=False)
    model=torch.compile(raw,dynamic=True)
    stop={'requested':False}
    def signal_stop(*_):stop['requested']=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    def elapsed():return before+time.monotonic()-started
    def save():
        torch.cuda.synchronize();state['consumed_seconds']=elapsed()
        blob={'model':raw.state_dict(),'optimizer':opt.state_dict(),'step':state['step'],
              'suite_state':deepcopy(state),'spec_sha256':digest(spec),'data_rng':rng.get_state(),
              'edge_rng':edge_rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        temp=latest.with_suffix('.tmp');torch.save(blob,temp);temp.replace(latest)
        atomic_json(out/'progress.json',state)
    def assess(final=False,initial=False):
        report=evaluate(raw,pools,full=final)
        report.update(step=state['step'],stage=state['stage'],elapsed_seconds=elapsed(),
                      train_view='all_queries' if final else '2048_fixed_queries_per_hop',
                      val_view='all_queries' if final else '512_fixed_queries_per_hop')
        tag='initial' if initial else 'final' if final else f'step_{state["step"]:09d}'
        atomic_json(out/'eval'/(tag+'.json'),report)
        flat={'optimizer_step':state['step'],'multihop/stage':state['stage'],'multihop/gpu_hours':elapsed()/3600}
        for split in ('train','val','test'):
            for h,m in report.get(split,{}).items():
                flat.update({f'multihop/{split}_h{h}_{k}':v for k,v in m.items() if v is not None})
        tracker.log(state['step'],{'free/overall/answer_accuracy':report['selection_score']})
        if tracker.run:tracker.run.log(flat)
        # Model selection always uses the same balanced 512-query val view.
        # The final full-val report is descriptive and never changes BEST.
        if not final and report['selection_score']>state['best_score']:
            state['best_score']=report['selection_score']
            torch.save({'model':raw.state_dict(),'step':state['step'],'stage':state['stage'],
                        'spec_sha256':digest(spec),'validation_score':state['best_score']},out/'checkpoints/best.pt')
        print('EVAL',json.dumps({k:report[k] for k in ('step','stage','elapsed_seconds','train','val')}),flush=True)
        return report
    ok=False;loss_sum=torch.zeros((),device='cuda');logged=0
    try:
        if not (out/'eval/initial.json').exists():assess(initial=True);save()
        print('START',spec['name'],'parent',spec['parent_checkpoint'],'budget',remaining,flush=True)
        while time.monotonic()-started<remaining and not stop['requested']:
            st=spec['stages'][state['stage']];n=state['step']-state['stage_start_step'];weights=st['weights']
            pair_counts=[math.floor((n+1)*32*w)-math.floor(n*32*w) for w in weights[:2]]
            pair_counts.append(32-sum(pair_counts));xs=[];ys=[]
            for hop,count in enumerate(pair_counts,1):
                if count:
                    p=pools.get('train',hop,full=True)
                    x,y=core.sample(p,count*2,rng)
                    xs.append(x);ys.append(y);state['exposure'][str(hop)]+=count*2
                    k=f'{state["stage"]}/hop{hop}';state['stage_exposure'][k]=state['stage_exposure'].get(k,0)+count*2
            x=core.reorder(torch.cat(xs),edge_rng);y=torch.cat(ys)
            progress=min(1.,elapsed()/(spec['budget_seconds']*.7))
            lr=spec['lr']*(.1+.9*.5*(1+math.cos(math.pi*progress)))*min(1.,(state['step']+1)/1000)
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(x)[:,-1].float(),y)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.);opt.step()
            loss_sum+=loss.detach();logged+=1;state['step']+=1
            if state['step']%500==0:
                value=float(loss_sum/logged)
                if not math.isfinite(value):raise RuntimeError('Non-finite CE')
                tracker.log(state['step'],{'answer_loss':value,'lr':lr,'grad_norm':float(grad)})
                print('TRAIN',state['step'],'stage',state['stage'],'loss',value,'seconds',round(elapsed(),1),flush=True)
                loss_sum.zero_();logged=0
            if state['step'] in (100,500,1000,5000) or state['step']%10000==0:
                report=assess();advance=False
                if state['stage']==0 and len(spec['stages'])>1:
                    if spec['scheme']=='timed':advance=elapsed()>=spec['budget_seconds']/2
                    else:
                        state['gate_streak']=state['gate_streak']+1 if gate(report) else 0
                        advance=state['gate_streak']>=3 and state['step']>=20000
                if advance:
                    state['transitions'].append({'step':state['step'],'seconds':elapsed(),'from':0,'to':1,
                                                  'reason':spec['scheme'],'gate_report':report})
                    state['stage']=1;state['stage_start_step']=state['step'];state['gate_streak']=0
                    print('ADVANCE',state['step'],flush=True)
                save()
        final=assess(final=True);save()
        # Evaluate validation-selected BEST separately without updating weights.
        saved=torch.load(out/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        best=build_model(cfg.model,270,0).cuda().eval().requires_grad_(False)
        best.load_state_dict(saved['model'])
        best_report=evaluate(best,pools,full=True)
        best_report.update(step=saved['step'],stage=saved['stage'],checkpoint='best',
                           selection='macro balanced accuracy on fixed 512 val queries per hop')
        atomic_json(out/'eval/final_best.json',best_report);del best,saved
        status='interrupted' if stop['requested'] else 'completed'
        atomic_json(out/(status+'.json'),{**state,'status':status,'reason':'signal' if stop['requested'] else 'wall_budget',
                                         'tracking_mode':tracker.mode,'test':final['test'],
                                         'not_a_proof_of_impossibility':True})
        ok=True
    finally:
        if not ok:save()
        tracker.finish()


if __name__=='__main__':main()
