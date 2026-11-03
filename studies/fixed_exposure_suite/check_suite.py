"""Data, loss and phase-gate checks. No formal runs, checkpoints or tracking."""
from pathlib import Path
import copy
import json
import math
import random
import tempfile

import torch
import train_suite

from prepare_suite import digest, make_world, direct_facts, witness
from train_suite import (load_runtime, Samples, gate, logical_loss, balanced,
                         new_model, update, evaluate, fixed_panel)

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    load_runtime()
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert len([j for j in manifest['jobs'] if j['kind']=='fixed'])==36
    assert len([j for j in manifest['jobs'] if j['kind']=='grokking'])==2
    worlds=[]
    for meta in manifest['worlds']:
        w=json.loads((HERE/meta['path']).read_text());worlds.append(w)
        assert digest(w)==meta['sha256']==digest(make_world(meta['seed']))
        assert len(w['edges'])==320 and len(w['adj'])==160
        assert all(v//16-u//16==1 for u,v in w['edges'])
        assert set(map(tuple,w['local_knowledge']))==set(map(tuple,w['edges']))
        for family in ['one_hop','multihop','base']:
            sets=[{tuple(r[:2]) for r in w[family][name]} for name in ['train','monitor','final']]
            assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
        assert len(w['core_final'])==256
        assert sum(r[2] for r in w['core_final'])==128
        assert all(r[3]>=6 for r in w['core_final'])
        for arm in ['R0','R2','R3']:
            paths=w['fragments'][arm]+[witness(w['adj'],r[0],r[1]) for r in w['base']['train'] if r[2]]
            facts=direct_facts(w['base']['train'],paths)
            count=sum(tuple(r[:2]) in facts for r in w['core_final'] if r[2])
            assert count==(128 if arm=='R3' else 0)
            for path in paths:
                assert all(v in w['adj'][u] for u,v in zip(path,path[1:]))
    rule=dict(train_accuracy=.99,monitor_ba=.95,min_class_recall=.95,consecutive=3)
    history=[];train=dict(accuracy=1.);monitor=dict(balanced_accuracy=.98,yes_recall=.98,no_recall=.98)
    assert not gate(history,train,monitor,rule)
    assert not gate(history,train,monitor,rule)
    assert gate(history,train,monitor,rule)
    assert not gate(history,train,dict(monitor,yes_recall=.1),rule)
    samples=Samples(worlds[0],'direct')
    for arm in ['R0','R2','R3']:
        for seed in range(10):
            paths=samples.knowledge_batch(random.Random(seed),arm)
            assert sum(len(p)-1 for p in paths)==64
            for path in paths:
                row=samples.knowledge(path)
                assert len(row.walk_indices)==len(path)-1
                assert row.tokens[-1] not in [row.tokens[i] for i in row.walk_indices]
    # Masked objective remains the same when examples are split into microbatches.
    tokens=torch.randint(0,17,(4,7));logits=torch.randn(4,7,17,requires_grad=True)
    am=torch.zeros(4,7);tm=am.clone();wm=am.clone()
    am[:2,5]=1;tm[1,2:4]=1;wm[2:,2:5]=1
    full=logical_loss(logits,tokens,am,tm,wm,2,6)[0]
    split=sum(logical_loss(logits[i:i+1],tokens[i:i+1],am[i:i+1],tm[i:i+1],wm[i:i+1],2,6)[0] for i in range(4))
    assert torch.allclose(full,split,atol=1e-6)
    grad1=torch.autograd.grad(full,logits,retain_graph=True)[0]
    grad2=torch.autograd.grad(split,logits)[0]
    assert torch.allclose(grad1,grad2,atol=1e-6)
    job=dict(model_seed=0,dropout=.1,weight_decay=.1,lr=1e-4)
    raw,model,opt,fn=new_model(job,torch.device('cpu'),compile_model=False)
    state={k:v.clone() for k,v in raw.state_dict().items()}
    records=balanced(worlds[0]['one_hop']['train'],8,random.Random(42))
    values,norm,_,_=update(raw,model,opt,fn,samples,records,
        samples.knowledge_batch(random.Random(43),'R0',8),torch.device('cpu'),1e-4)
    assert torch.isfinite(values[0]) and torch.isfinite(norm)
    assert any(not torch.equal(v,raw.state_dict()[k]) for k,v in state.items())
    result=evaluate(raw,samples,fixed_panel(worlds[0]['one_hop']['monitor'],8),torch.device('cpu'))
    assert result['n']==8 and 0<=result['balanced_accuracy']<=1
    assert all('predicted_token' in p for p in result['predictions'])
    # Gold reference consistency for PATH and empty-negative policy.
    path_samples=Samples(worlds[0],'path')
    for r in worlds[0]['base']['train'][:100]:
        row=path_samples.query(r)
        if not r[2]:assert row.answer_index==row.prompt_len
    # End-to-end event switch: one-hop snapshot contains no multi-hop QUERY.
    # Temporary regression artifacts are removed when the test exits.
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        from train_suite import save_atomic
        parent=root/'knowledge/g0_s0'
        parent.mkdir(parents=True)
        world_hash=manifest['worlds'][0]['sha256']
        save_atomic(parent/'latest.pt',dict(model=raw.state_dict(),world_hash=world_hash))
        event_job=copy.deepcopy(next(j for j in manifest['jobs'] if j['kind']=='grokking'))
        event_job.update(_output_root=str(root),_world_hash=world_hash,
            eval_interval=1,checkpoint_interval=1,log_interval=1)
        event_job['gate']=dict(train_accuracy=0.,monitor_ba=0.,min_class_recall=0.,consecutive=1)
        train_suite.train(event_job,worlds[0],root/'event_test',torch.device('cpu'),tracking='disabled')
        phase1=torch.load(root/'event_test/one_hop_grokking.pt',weights_only=False)
        final=torch.load(root/'event_test/latest.pt',weights_only=False)
        assert phase1['phase']=='one_hop' and final['phase']=='multi'
        assert phase1['step']==1 and final['step']==2
        assert all(int(key.rsplit(',',1)[-1])==1 for key in phase1['counts'] if key.startswith('q:'))
        assert any(int(key.rsplit(',',1)[-1])>=2 for key in final['counts'] if key.startswith('q:'))
        # Interrupted/resumed optimizer, exposure and RNG agree with uninterrupted CPU training.
        fixed_job=copy.deepcopy(next(j for j in manifest['jobs'] if j['kind']=='fixed'))
        fixed_job.update(_output_root=str(root),_world_hash=world_hash,
            steps=4,eval_interval=2,checkpoint_interval=2,log_interval=1)
        train_suite.train(fixed_job,worlds[0],root/'baseline',torch.device('cpu'),tracking='disabled')
        original_update=train_suite.update
        calls=[0]
        def interrupt(*a,**kw):
            calls[0]+=1
            if calls[0]==3:raise RuntimeError('test interruption')
            return original_update(*a,**kw)
        train_suite.update=interrupt
        try:
            train_suite.train(fixed_job,worlds[0],root/'resumed',torch.device('cpu'),tracking='disabled')
            raise AssertionError('No interruption')
        except RuntimeError as error:
            assert str(error)=='test interruption'
        finally:train_suite.update=original_update
        train_suite.train(fixed_job,worlds[0],root/'resumed',torch.device('cpu'),resume=True,tracking='disabled')
        left=torch.load(root/'baseline/latest.pt',weights_only=False)
        right=torch.load(root/'resumed/latest.pt',weights_only=False)
        assert left['counts']==right['counts'] and left['query_rng']==right['query_rng']
        assert all(torch.equal(v,right['model'][k]) for k,v in left['model'].items())
    report=dict(status='PASS',graphs=3,fixed_tasks=36,grokking_tasks=2,
        checks=['deterministic graph generation','independent traversal/DP labels',
            'query split disjointness','all local edges covered','R0/R2 zero complete-witness exposure',
            'R3 128/128 intentional exposure','exact 64-transition budget','three-evaluation gate',
            'masked loss and gradient additivity','CPU forward/backward','native evaluation interface',
            'event-driven phase switch','no multi-hop QUERY before gate',
            'CPU interrupted resume matches uninterrupted weights/exposure'],
        formal_training_started=False)
    (HERE/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
