"""Withhold the reference terminal action, then greedily predict action and answer."""
from pathlib import Path
import argparse,gzip,json,os,sys,time
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));os.chdir(ROOT)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies/no_bias_diagnostics')]
import diagnose as prior
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore,ParametricWorld
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import context_prompt,parametric_prompt
from cot_navigation.evaluation.rollout import generate
from types import SimpleNamespace

OUT=ROOT/'studies/terminal_action_diagnostic/results'
PREVIOUS=ROOT/'studies/no_bias_diagnostics/results'

def entries():return [e for e in prior.models() if e['protocol'].endswith('_state')]

def withhold_terminal(prefix,tok,label):
    action=tok.id('<FOUND>' if label else '<EXHAUST>')
    assert prefix[-2:]==[action,tok.id(';')]
    supplied=prefix[:-2]
    assert supplied[-1]==tok.id('|')
    assert not set(supplied)&{tok.id('<FOUND>'),tok.id('<EXHAUST>'),tok.label_id(0),tok.label_id(1),tok.id('<EOS>')}
    return supplied,action

def run(entry,batch):
    out=OUT/entry['id'];out.mkdir(parents=True,exist_ok=True)
    folder=ROOT/entry['run'];cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    checkpoint=folder/'checkpoints'/('latest.pt' if entry['tag']=='last' else 'best.pt')
    before=prior.audit.sha(checkpoint)
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    if model.swa_records:model.swa_backend='sdpa'
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(blob['model'],strict=True)
    step=int(blob['step']);del blob
    for source_meta in sorted((PREVIOUS/entry['id']).glob('*.json')):
        meta=json.loads(source_meta.read_text());case=meta['case']
        assert meta['checkpoint_sha256']==before
        receipt=out/(case+'.json')
        if receipt.exists():continue
        with gzip.open(source_meta.with_suffix('.jsonl.gz'),'rt') as f:previous=[json.loads(x) for x in f]
        included=[r for r in previous if not r['oracle_truncated']]
        context=entry['kind']=='context';source=ContextStore(meta['root']) if context else ParametricWorld(meta['root'])
        a=source.arrays if context else source.queries
        roots=cfg.train.scale_roots if context else cfg.train.world_roots
        world=[] if context else [tok.id(f'<WORLD_{[Path(x).resolve() for x in roots].index(Path(meta["root"]).resolve())}>')]
        rows=[];started=time.time()
        for at in range(0,len(included),batch):
            prepared=[]
            for old in included[at:at+batch]:
                idx=old['query'];adj=source.adj(idx) if context else source.adj()
                nodes=source.node_vocab_ids(idx,tok) if context else source.node_vocab_ids(tok)
                s,t=int(a['s_topo'][idx]),int(a['t_topo'][idx]);label=int(a['label'][idx])
                ex=SimpleNamespace(nodes=nodes,adj=adj,source=s,target=t,label=label)
                assert int(t in prior.graph_reached(ex))==label==old['label']
                if context:
                    edges=list(zip(a['edge_src'][idx].astype(int).tolist(),a['edge_dst'][idx].astype(int).tolist()))
                    prompt=context_prompt(tok,nodes,edges,s,t)
                else:prompt=world+parametric_prompt(tok,nodes,s,t)
                prefix,full,cut=prior.oracle_trace(ex,prompt,entry['protocol'],tok,cfg,idx)
                assert not cut
                assert prior.replay(ex,full,entry['protocol'],tok)['complete_valid']
                supplied,action=withhold_terminal(prefix,tok,label)
                assert len(supplied)+4<=model.max_seq_len
                prepared.append((old,prefix,supplied,action))
            # Unrestricted vocabulary, no gold action or separator is fed back.
            outputs=generate(model,[x[2] for x in prepared],entry['protocol'],tok,cfg.train.precision,
                             max_new_tokens=32,cot_margin=0)
            controls=prior.answer_logits(model,[x[1] for x in prepared],tok,cfg.train.precision)
            for (old,prefix,supplied,action),result,control in zip(prepared,outputs,controls):
                emitted=list(result['generated']);label=old['label']
                expected=[action,tok.id(';'),tok.label_id(label)]
                first=emitted[0] if emitted else None
                action_ok=first==action
                separator_ok=len(emitted)>1 and emitted[1]==tok.id(';')
                exact=emitted[:3]==expected
                rows.append(dict(query=old['query'],label=label,gap=old['gap'],
                    expected_action='FOUND' if label else 'EXHAUST',
                    predicted_first_token=tok.name(first) if first is not None else None,
                    action_correct=action_ok,separator_correct=separator_ok,
                    exact_action_separator_answer=exact,answer_correct=result['pred']==label,
                    answer_pred=result['pred'],format_fail=result['format_fail'],
                    generation_limit_reached=result['generation_limit_reached'],
                    gold_terminal_control_correct=control['pred']==label,gold_terminal_control=control,
                    prior_free_correct=old['free_correct'],prior_free_pred=old['free_pred'],
                    generated_tokens=emitted,generated_text=tok.decode(emitted),
                    supplied_tail=tok.decode(supplied[-100:]),prefix_tokens=len(supplied)))
            print(entry['id'],case,min(at+batch,len(included)),'/',len(included),flush=True)
        with gzip.open(out/(case+'.jsonl.gz'),'wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        assert prior.audit.sha(checkpoint)==before
        prior.audit.write(receipt,dict(model=entry,case=case,root=meta['root'],step=step,
            checkpoint_sha256=before,n=len(rows),excluded_truncated=len(previous)-len(rows),
            source_query_ids=[r['query'] for r in previous],included_query_ids=[r['query'] for r in rows],
            input_contract='Complete legal reference up to final state trailing bar; terminal action/semicolon/answer/EOS withheld.',
            decoding='Full-vocabulary greedy, up to 32 tokens, no forced action/separator/answer; exact first 3 is primary joint metric.',
            batch=batch,elapsed_seconds=time.time()-started))
        del source
    del model;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch',type=int,default=4);args=p.parse_args();torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    for e in entries()[args.worker::args.workers]:run(e,args.batch)
    prior.audit.write(OUT/f'worker{args.worker}_done.json',{'done':True})
