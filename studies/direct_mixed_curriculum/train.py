"""Pure-Direct wall-budget trainer using the project's exact model/optimizer.

Offline tensors and GPU edge permutations remove Python graph generation from
the update path. Only the last prompt position predicts the answer. A CPU
preflight verifies equivalence to the project's shifted/masked full sequence
loss. No smoke training or smoke checkpoint is created.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from data import BAR, EDGE, EOS, NODE_LO, NO, QUERY, YES, atomic_json, digest, key
from cot_navigation.config import load_config, to_plain
from cot_navigation.models.transformer import build_model
from cot_navigation.tracking.logger import Tracker
from cot_navigation.training.trainer import _optimizer

BASE = Path('outputs/explore_v1/launches/explore_v1_s012/context/standard/train50000/direct_onehop/direct/resolved_config.yaml')


def config(spec, out):
    return load_config([BASE], [f'output_dir={out}', f'train.model_seed={spec["seed"]}',
        'train.max_steps=10000000', f'train.lr={spec["lr"]}', f'train.min_lr={spec["lr"]/10}',
        'train.warmup_steps=1000', 'train.log_interval=500', 'train.keep_last=1',
        'evaluation.final_checkpoints=[]', 'train.free_gen_samples=0',
        'tracking.mode=online', 'tracking.online_failure=error',
        f'tracking.group={spec["group"]}', f'tracking.name={spec["group"]}/{spec["name"]}',
        f'tracking.variant={spec["name"]}', 'tracking.upload_artifacts=false'])


def hash_model(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def cpu_preflight():
    """One frozen forward/gradient comparison; not an optimizer update."""
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.protocols.render import RenderedExample
    from cot_navigation.data.dataset import collate_examples
    from cot_navigation.training.loss import combine_losses
    cfg = load_config([BASE], ['model.compile=false', 'tracking.mode=disabled'])
    torch.set_num_threads(2);torch.manual_seed(0)
    model = build_model(cfg.model, 270, 0).eval()
    prompt = [EDGE, NODE_LO, NODE_LO+1, EDGE, NODE_LO+2, NODE_LO+3, QUERY, NODE_LO, NODE_LO+1, BAR]
    row = RenderedExample(tokens=prompt+[YES, EOS], prompt_len=10, answer_index=10,
                          trace_indices=[], walk_indices=[], label=1, meta={})
    b = {k:torch.as_tensor(v) if isinstance(v,np.ndarray) else v
         for k,v in collate_examples([row], 0).items()}
    logits = model(b['tokens'])
    a = combine_losses(logits, b['tokens'], b['answer_mask'], b['trace_mask'], b['walk_mask'], 1, 0)['total_loss']
    grad_a = torch.autograd.grad(a, tuple(model.parameters()))
    c = F.cross_entropy(model(torch.tensor([prompt]))[:, -1].float(), torch.tensor([YES]))
    grad_c = torch.autograd.grad(c, tuple(model.parameters()))
    delta = max(float((x-y).abs().max()) for x,y in zip(grad_a, grad_c))
    assert abs(float(a-c)) < 2e-5 and delta < 2e-4, (float(a), float(c), delta)
    assert int(b['answer_mask'].sum()) == 1 and not b['trace_mask'].any() and not b['walk_mask'].any()
    assert len(Tokenizer()) == 270
    return {'masked_ce': float(a), 'prompt_only_ce': float(c), 'max_gradient_difference': delta,
            'optimizer_updates': 0, 'vocab_size': 270, 'model': model.report()}


class Pools:
    def __init__(self, folder):
        self.folder, self.cache = Path(folder), {}

    def get(self, stage, split='train', kind=None):
        kind = kind or stage['kind']
        if kind.startswith('mix_'):
            kind = 'full'
        name = key(kind, stage['vocab'], stage['edges'])
        cache_key = (name, split)
        if cache_key not in self.cache:
            with np.load(self.folder/(name+'.npz')) as f:
                if split+'_x' not in f:
                    return None
                p = {k:torch.as_tensor(f[split+'_'+k], device='cuda') for k in ('x','y','group','variant')}
            p['target'] = torch.where(p['y'].bool(), YES, NO)
            p['yes'] = torch.where(p['y'] == 1)[0]
            p['no'] = torch.where(p['y'] == 0)[0]
            self.cache[cache_key] = p
        return self.cache[cache_key]


def sample(p, count, rng, packed=False):
    if packed:
        assert count == 64 and len(p['y']) % 8 == 0
        groups = torch.randint(len(p['y'])//8, (8,), generator=rng, device='cuda')
        ids = (groups[:,None]*8+torch.arange(8, device='cuda')).flatten()
    else:
        assert count % 2 == 0
        ids = torch.cat([p[k][torch.randint(len(p[k]), (count//2,), generator=rng, device='cuda')]
                         for k in ('yes','no')])
    return p['x'][ids], p['target'][ids]


def reorder(x, rng=None, reverse=False):
    m = (x.shape[1]-4)//3
    triples = x[:, :-4].reshape(len(x), m, 3)
    if reverse:
        triples = triples.flip(1)
    else:
        order = torch.rand((len(x), m), generator=rng, device='cuda').argsort(-1)
        triples = triples.gather(1, order[:,:,None].expand(-1,-1,3))
    return torch.cat([triples.flatten(1), x[:, -4:]], 1)


@torch.inference_mode()
def score(model, p, reverse=False, return_arrays=False):
    if p is None or len(p['y']) == 0:
        return None
    probs, pred, ce = [], [], []
    for start in range(0, len(p['y']), 256):
        x = p['x'][start:start+256]
        if reverse:
            x = reorder(x, reverse=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)[:, -1].float()
        target = p['target'][start:start+256]
        ce.append(F.cross_entropy(logits, target, reduction='none').cpu())
        probs.append(logits[:,[NO,YES]].softmax(-1)[:,1].cpu())
        picked = logits.argmax(-1)
        pred.append(torch.where(picked == YES, 1, torch.where(picked == NO, 0, -1)).cpu())
    prob = torch.cat(probs).numpy();pred = torch.cat(pred).numpy();loss = torch.cat(ce).numpy()
    truth = p['y'].cpu().numpy()
    result = {'n':len(truth), 'accuracy':float(np.mean(pred == truth)), 'ce':float(loss.mean()),
              'yes_recall':float(np.mean(pred[truth == 1] == 1)) if np.any(truth == 1) else None,
              'no_recall':float(np.mean(pred[truth == 0] == 0)) if np.any(truth == 0) else None,
              'format_fail_rate':float(np.mean(pred < 0)), 'p_yes_mean':float(prob.mean()),
              'p_yes_std':float(prob.std())}
    if return_arrays:
        return result, pred, prob
    return result


def pair_metrics(p, pred, prob):
    groups, variants = p['group'].cpu().numpy(), p['variant'].cpu().numpy()
    lookup = {(int(g), int(v)):i for i,(g,v) in enumerate(zip(groups, variants))}
    truth = p['y'].cpu().numpy()
    result = {}
    for name, pairs in {'graph':[(0,4),(1,5),(2,6),(3,7)],
                        'target':[(0,1),(2,3),(4,5),(6,7)],
                        'source':[(0,2),(1,3),(4,6),(5,7)]}.items():
        ids = [(lookup[g,a], lookup[g,b]) for g in np.unique(groups) for a,b in pairs
               if (g,a) in lookup and (g,b) in lookup]
        if not ids:
            continue
        a,b = np.array(ids).T
        result[name+'_pairs'] = len(ids)
        result[name+'_both_correct'] = float(np.mean((pred[a] == truth[a]) & (pred[b] == truth[b])))
        result[name+'_signed_delta_p'] = float(np.mean((prob[a]-prob[b])*(2*truth[a]-1)))
    return result


def evaluate(model, pools, stage, final=False):
    result = {}
    for split in ['train','val'] + (['test','known_graph_query'] if final else []):
        p = pools.get(stage, split)
        if p is None or not len(p['y']):
            continue
        metric, pred, prob = score(model, p, return_arrays=True)
        metric.update(pair_metrics(p, pred, prob))
        result[split] = metric
        alt, alt_pred, _ = score(model, p, reverse=True, return_arrays=True)
        alt['prediction_consistency'] = float(np.mean(alt_pred == pred))
        result[split+'_reverse_order'] = alt
    return result


def gate_pass(report, stage):
    def success(m, acc, ce=None):
        return m['accuracy'] >= acc and (ce is None or m['ce'] <= ce)
    if not success(report['train'], .99, .02) or not success(report['val'], .95):
        return False
    if stage['shuffle'] and (not success(report['train_reverse_order'], .99, .02)
                              or not success(report['val_reverse_order'], .95)):
        return False
    # Graph-flip pairs exist in every non-fixed-world restricted task.
    return report['val'].get('graph_both_correct', 1.0) >= .90


@torch.inference_mode()
def representation_check(raw, pools, stage):
    """Small, fixed validation diagnostic; save scalar statistics only."""
    probe_stage = {**stage,'kind':'full'}
    p = pools.get(probe_stage,'val')
    x = p['x'][:128]
    capture = {}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits = raw(x,capture=capture)[:, -1].float()
    result = {}
    for li,h in enumerate(capture['resid'],1):
        z = h[:, -1].float().reshape(-1,8,h.shape[-1])
        result[f'L{li}'] = {
            'common_rms':float(z.mean((0,1)).square().mean().sqrt()),
            'input_variation_rms':float((z-z.mean((0,1),keepdim=True)).square().mean().sqrt()),
            'graph_change_rms':float((z[:,0]-z[:,4]).square().mean().sqrt()),
            'target_change_rms':float((z[:,0]-z[:,1]).square().mean().sqrt()),
            'source_change_rms':float((z[:,0]-z[:,2]).square().mean().sqrt())}
    result['n'] = len(x)
    result['note'] = 'Complete paired validation groups; last prompt position; no activation arrays saved.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=None, help='Remaining allocation on resume')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text());out = args.spec.parent
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in [str(i) for i in range(1,8)]:
        raise RuntimeError('Only physical GPUs 1-7 are authorized for this queue.')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
    cfg = config(spec, out)
    cfg.tracking.notes = json.dumps({'pure_direct':True,'budget_seconds':spec['budget_seconds'],
                                     'schedule':spec['schedule'],'stages':spec['stages']})
    import yaml
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(to_plain(cfg), sort_keys=False))
    raw = build_model(cfg.model, 270, 0).cuda()
    opt, opt_info = _optimizer(raw, cfg, torch.device('cuda'))
    data_rng = torch.Generator(device='cuda').manual_seed(91000+spec['seed'])
    edge_rng = torch.Generator(device='cuda').manual_seed(92000+spec['seed'])
    pools = Pools(spec['data_dir'])
    latest = out/'checkpoints/latest.pt'
    (out/'checkpoints').mkdir(exist_ok=True)
    state = {'step':0,'stage':0,'stage_start_step':0,'stage_start_seconds':0.,
             'consumed_seconds':0.,'gate_streak':0,'transitions':[], 'exposure':{},
             'run_id':uuid.uuid4().hex, 'best_val':-1.}
    initial_hash = hash_model(raw)
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        assert saved['spec_sha256'] == digest(spec), 'Refusing changed plan on resume'
        raw.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        state = saved['suite_state']
        data_rng.set_state(saved['data_rng']);edge_rng.set_state(saved['edge_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    else:
        atomic_json(out/'initialization.json', {'model_sha256':initial_hash,'model_seed':spec['seed'],'model':raw.report()})
    session_start = time.monotonic()
    consumed_before = state['consumed_seconds']
    remaining = min(spec['budget_seconds']-consumed_before,
                    args.budget_seconds if args.budget_seconds is not None else spec['budget_seconds'])
    remaining = max(0, remaining)
    stop = {'requested':False}
    def on_signal(*_): stop['requested'] = True
    signal.signal(signal.SIGTERM, on_signal);signal.signal(signal.SIGINT, on_signal)
    tracker = Tracker(cfg, state['run_id'], None,
                      {'optimizer':opt_info,'parameters':raw.report(),'torch_compile':True}, resume=latest.exists())
    if tracker.run:
        tracker.run.define_metric('direct/*', step_metric='optimizer_step', hidden=True)
        for name in ['direct/train_accuracy','direct/val_accuracy','direct/train_ce','direct/stage','direct/gpu_hours']:
            tracker.run.define_metric(name, step_metric='optimizer_step', hidden=False)
    model = torch.compile(raw, dynamic=True)
    def elapsed(): return consumed_before + time.monotonic()-session_start
    def save():
        torch.cuda.synchronize()
        state['consumed_seconds'] = elapsed()
        blob = {'model':raw.state_dict(),'optimizer':opt.state_dict(),'step':state['step'],
                'spec_sha256':digest(spec),'suite_state':deepcopy(state),
                'data_rng':data_rng.get_state(),'edge_rng':edge_rng.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        tmp = latest.with_suffix('.tmp');torch.save(blob,tmp);tmp.replace(latest)
        atomic_json(out/'progress.json', {**state,'effective_tracking':tracker.mode})
    def checkpoint_eval(final=False):
        raw.eval()
        st = spec['stages'][state['stage']]
        report = evaluate(raw, pools, st, final=final)
        if final or state['step'] in (100,1000) or state['step']%50000 == 0:
            report['representation'] = representation_check(raw,pools,st)
        report['selection_val'] = score(raw, pools.get(dict(kind='full',vocab=256,edges=24,shuffle=True),'val'))
        report.update(step=state['step'],stage=state['stage'],stage_spec=st,
                      elapsed_seconds=elapsed(), complete_train_pool=True)
        atomic_json(out/'eval'/('final.json' if final else f'step_{state["step"]:09d}.json'), report)
        tracker.log(state['step'], {'free/overall/answer_accuracy':report['val']['accuracy']})
        if tracker.run:
            flat = {'optimizer_step':state['step'],'direct/stage':state['stage'],
                    'direct/gpu_hours':elapsed()/3600}
            for split, metrics in report.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    flat.update({f'direct/{split}_{k}':v for k,v in metrics.items() if v is not None})
            tracker.run.log(flat)
        value = report['selection_val']['accuracy']
        # Best is checkpointed by validation only, not by final test.
        if not final and value > state['best_val']:
            state['best_val'] = value
            torch.save({'model':raw.state_dict(),'step':state['step'],'stage':state['stage'],
                        'spec_sha256':digest(spec),'validation_accuracy':value},out/'checkpoints/best.pt')
        print('EVAL',json.dumps({k:v for k,v in report.items() if k != 'stage_spec'}),flush=True)
        raw.train()
        return report
    last_eval = -1
    last_log = time.monotonic();loss_sum = torch.zeros((),device='cuda');log_n = 0
    run_ok = False
    try:
        print('START',spec['name'],'budget_remaining',remaining,'resume_step',state['step'],flush=True)
        while time.monotonic()-session_start < remaining and not stop['requested']:
            i = state['stage'];st = spec['stages'][i]
            p = pools.get(st)
            if st['kind'].startswith('mix_'):
                n = state['step']-state['stage_start_step'];ratio=st['mixture']
                pairs = math.floor((n+1)*32*ratio)-math.floor(n*32*ratio)
                anchor = 'fixed_'+st['kind'].split('_',1)[1]
                x1,y1 = sample(pools.get(st,kind=anchor),pairs*2,data_rng)
                x2,y2 = sample(p,64-pairs*2,data_rng)
                x,y = torch.cat([x1,x2]),torch.cat([y1,y2])
            else:
                x,y = sample(p,64,data_rng,spec['packed'])
            if st['shuffle']:
                x = reorder(x,edge_rng)
            # Time-cosine through 70% of allocation, then a long constant floor.
            progress = min(1.,elapsed()/(spec['budget_seconds']*.70))
            base_lr = spec['lr']*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            lr = base_lr*min(1.,(state['step']+1)/1000)
            for g in opt.param_groups:g['lr'] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits = model(x)[:, -1].float()
                loss = F.cross_entropy(logits,y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg.train.grad_clip)
            opt.step()
            loss_sum += loss.detach();log_n += 1
            state['step'] += 1
            exposure_key = str(i)
            state['exposure'][exposure_key] = state['exposure'].get(exposure_key,0)+64
            if state['step'] % 500 == 0:
                mean_loss = float(loss_sum/log_n)
                if not math.isfinite(mean_loss): raise RuntimeError('Non-finite training CE')
                tracker.log(state['step'],{'answer_loss':mean_loss,'lr':lr,'grad_norm':float(grad_norm)})
                print('TRAIN',state['step'],'stage',i,'ce',mean_loss,'seconds',round(elapsed(),1),flush=True)
                loss_sum.zero_();log_n=0;last_log=time.monotonic()
            should_eval = state['step'] in (100,500,1000,5000) or state['step']%10000 == 0
            if should_eval:
                report = checkpoint_eval();last_eval=state['step']
                advance = False
                if i < len(spec['stages'])-1:
                    if spec['schedule'] == 'timed':
                        advance = elapsed() >= (i+1)*spec['budget_seconds']/len(spec['stages'])
                    else:
                        state['gate_streak'] = state['gate_streak']+1 if gate_pass(report,st) else 0
                        advance = state['gate_streak'] >= 3 and state['step']-state['stage_start_step'] >= 20000
                if advance:
                    transition = {'from':i,'to':i+1,'step':state['step'],'seconds':elapsed(),
                                  'reason':'time' if spec['schedule']=='timed' else 'three_validation_gates',
                                  'gate_report':report}
                    state['transitions'].append(transition);state['stage']+=1
                    state['stage_start_step']=state['step'];state['stage_start_seconds']=elapsed()
                    state['gate_streak']=0
                    print('ADVANCE',json.dumps(transition),flush=True)
                save()
        final = checkpoint_eval(final=True)
        # Every run receives the same final harder probes, even if a gate stalls.
        common = {}
        for v,m in [(16,6),(64,6),(256,6),(256,24)]:
            st = dict(kind='full',vocab=v,edges=m,shuffle=True)
            p=pools.get(st,'test')
            metric,pred,prob=score(raw,p,return_arrays=True);metric.update(pair_metrics(p,pred,prob))
            common[f'full_v{v}_m{m}']=metric
        atomic_json(out/'eval/common_test.json',common)
        save()
        atomic_json(out/'eval/original_test_last.json', original_test(raw))
        best = torch.load(out/'checkpoints/best.pt', map_location='cpu', weights_only=False)
        raw.load_state_dict(best['model']);raw.eval()
        best_report = evaluate(raw,pools,dict(kind='full',vocab=256,edges=24,shuffle=True),final=True)
        best_report.update(step=best['step'],selection='fixed full-v256-m24 validation, no test selection')
        atomic_json(out/'eval/final_best.json',best_report)
        atomic_json(out/'eval/original_test_best.json', original_test(raw))
        status = 'interrupted' if stop['requested'] else 'completed'
        receipt = {**state,'status':status,'reason':'signal' if stop['requested'] else 'wall_budget',
                   'final_train_accuracy':final['train']['accuracy'],'final_val_accuracy':final['val']['accuracy'],
                   'unreached_stages':len(spec['stages'])-1-state['stage'],
                   'tracking_mode':tracker.mode,'not_a_proof_of_impossibility':True}
        atomic_json(out/(status+'.json'),receipt)
        run_ok=True
    finally:
        if not run_ok:
            save()
        tracker.finish()



# Keep the prior trainer and math, changing only data sampling/diagnostics.
_original_sample = sample
_original_score = score
_original_gate = gate_pass

def sample(p, count, rng, packed=False):
    assert count % 2 == 0 and not packed
    if 'hop_ids' not in p:
        hop=p['group'] % 3 + 1
        p['hop_ids']={(h,c):torch.where((hop==h)&(p['y']==c))[0] for h in (1,2,3) for c in (0,1)}
    # Equal expected hop frequency; each sampled pair has one YES and one NO.
    if 'balanced_hop_ids' not in p:
        p['balanced_hop_ids']={c:torch.stack([p['hop_ids'][h,c] for h in (1,2,3)]) for c in (0,1)}
    hops=torch.randint(3,(count//2,),generator=rng,device='cuda')
    ids=[]
    for c in (0,1):
        table=p['balanced_hop_ids'][c]
        positions=torch.randint(table.shape[1],(count//2,),generator=rng,device='cuda')
        ids.append(table[hops,positions])
    ids=torch.cat(ids)
    return p['x'][ids],p['target'][ids]

def score(model,p,reverse=False,return_arrays=False):
    result=_original_score(model,p,reverse,True)
    if result is None: return None
    metrics,pred,prob=result
    truth=p['y'].cpu().numpy();hop=(p['group']%3+1).cpu().numpy()
    for h in (1,2,3):
        mask=hop==h
        metrics[f'hop{h}_accuracy']=float((pred[mask]==truth[mask]).mean())
        for c,name in ((0,'no'),(1,'yes')):
            use=mask&(truth==c)
            metrics[f'hop{h}_{name}_recall']=float((pred[use]==truth[use]).mean())
    return (metrics,pred,prob) if return_arrays else metrics

def gate_pass(report,stage):
    return _original_gate(report,stage) and all(report[split][f'hop{h}_accuracy']>=threshold
        for split,threshold in [('train',.99),('val',.95)] for h in (1,2,3))

def representation_check(raw,pools,stage):
    return {'note':'Not collected in this data-curriculum experiment.'}


def original_test(raw):
    # Frozen prior-suite test: compare to the previous experiments on identical inputs.
    path=Path('studies/direct_multihop/data_v1/arrays.npz')
    result={}
    with np.load(path) as f:
        for h in (1,2,3):
            use=f['test_hop']==h
            x=torch.as_tensor(f['test_x'][use],device='cuda')
            y=torch.as_tensor(f['test_y'][use],device='cuda')
            p={'x':x,'y':y,'target':torch.where(y.bool(),YES,NO)}
            result[str(h)]=_original_score(raw,p)
    return result

if __name__ == '__main__':main()
