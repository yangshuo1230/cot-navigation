"""Manifest-driven natural probes and lenses; reuses existing analysis kernels.

Receiver depth and final checkpoint are data, never hardcoded. Navigation
weights stay frozen. Only compact results and primary natural tokens persist.
"""
import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import runpy
import time

os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'2')
ROOT=Path(__file__).resolve().parents[2]
import numpy as np
import torch
from cot_navigation.analysis.common import AnalysisModel
from cot_navigation.analysis.natural_history import annotate,match_readouts
from cot_navigation.analysis.history_probe import fit_positions,position_scores
from cot_navigation.analysis.state_readout import fit_readout,evaluate_readout
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.experiments.artifacts import atomic_json,sha_file

PREP=runpy.run_path(str(ROOT/'studies/correction_reproduction/natural_history_probes/prepare.py'))
NH=runpy.run_path(str(ROOT/'studies/correction_reproduction/natural_history_probes/run.py'))
STATE=runpy.run_path(str(ROOT/'studies/correction_reproduction/state_readouts/run.py'))
EARLY=runpy.run_path(str(ROOT/'studies/correction_reproduction/state_readouts/early_clean_lens.py'))
LEGAL=runpy.run_path(str(ROOT/'studies/correction_reproduction/state_readouts/legality_transitions.py'))


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


class Model(AnalysisModel):
    def __init__(self,entry):
        self.entry=entry
        super().__init__(dict(entry,checkpoint_name='latest.pt'),ROOT)
        assert self.checkpoint['sha256']==entry['checkpoint_sha256']
        assert not (self.run/'DEPRECATED.json').exists()
        self.checkpoint['path']=entry['checkpoint']

    def sample_ids(self,split,count):
        ids=self.indices(split);ids=ids[self.store.arrays['label'][ids]==1]
        return np.random.default_rng(20261030+{'train':0,'val':1,'test':2}[split]).permutation(ids)[:count].tolist()


def probe_window(row,tok):
    """Truncate an unedited full test generation after first error + 2 records."""
    c=dict(row);c['trace']=list(row['trace'])
    stop=tok.stoi.get('<STOP>')
    if stop is not None and stop in c['trace']:c['trace']=c['trace'][:c['trace'].index(stop)]
    if c['first']:
        semis=[i for i,t in enumerate(c['trace']) if t==tok.id(';')]
        index=c['first']['record']+1
        if index<len(semis):c['trace']=c['trace'][:semis[index]+1]
    marker=tok.stoi.get('<MISTAKE>',-1)
    c['marker_at']=c['trace'].index(marker) if marker in c['trace'] else None
    return c


def sources_for(plan,depth):return [e for e in plan['models'] if e['layers']==depth and e['seed']==0]


def load_cases(plan,out,am,own=True,train_count=1536,val_count=256):
    cases={s:[] for s in ['train','val','test']};provenance=[]
    for split,count in [('train',train_count),('val',val_count)]:
        for e in sources_for(plan,am.model.n_layers):
            path=out/'corpus/native'/f"{e['group']}_s0_{split}.json.gz"
            d=read(path);assert d['complete'] and d['identity']['checkpoint']['sha256']==e['checkpoint_sha256']
            assert d['identity']['source_fingerprint']==am.store.fingerprint
            cases[split].extend(d['rows'][:count]);provenance.append(dict(path=str(path.relative_to(ROOT)),sha256=sha_file(path)))
    entries=[am.entry] if own else [e for e in plan['models'] if e['layers']==am.model.n_layers]
    for e in entries:
        path=Path(plan['manifest']['output'])/'probe_corpus'/f"{e['id']}__test.json.gz"
        d=read(path) if path.exists() else {}
        if d.get('complete'):
            assert d['identity']['model']['checkpoint_sha256']==e['checkpoint_sha256']
            assert d['identity']['fingerprint']==am.store.fingerprint
        else:
            path=out/'corpus/native'/f"{e['group']}_s{e['seed']}_test.json.gz"
            d=read(path);assert d['complete'] and d['identity']['checkpoint']['sha256']==e['checkpoint_sha256']
            assert d['identity']['source_fingerprint']==am.store.fingerprint
        assert {c['query'] for c in d['rows']}==set(plan['tasks']['main']['queries'])
        cases['test'].extend(probe_window(c,am.tok) for c in d['rows'])
        provenance.append(dict(path=str(path),sha256=sha_file(path)))
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {c['query'] for c in cases[a]}&{c['query'] for c in cases[b]}
        assert not {c['graph_hash'] for c in cases[a]}&{c['graph_hash'] for c in cases[b]}
    return cases,provenance


def prepare(plan,out):
    PREP['prepare'].__globals__['P']=out/'corpus'
    for e in plan['models']:
        paths=[out/'corpus/native'/f"{e['group']}_s0_{s}.json.gz" for s in ['train','val']] if e['seed']==0 else []
        baseline=Path(plan['manifest']['output'])/'probe_corpus'/f"{e['id']}__test.json.gz"
        test=out/'corpus/native'/f"{e['group']}_s{e['seed']}_test.json.gz"
        test_ready=any(p.exists() and read(p).get('complete') for p in [baseline,test])
        train_ready=all(p.exists() and read(p)['complete'] for p in paths)
        if train_ready and test_ready:continue
        am=Model(e);compiled=CompiledInference(am.model).eval()
        if not test_ready:
            # Only the missing N100 probe window, not five-graph behavior analysis.
            original_sampler=am.sample_ids
            am.sample_ids=lambda split,count:plan['tasks']['main']['queries'][:count] if split=='test' else original_sampler(split,count)
            PREP['prepare'](am,e['group'],e['seed'],'test',256,32,compiled)
            am.sample_ids=original_sampler
        if e['seed']==0:
            for split,count in [('train',1536),('val',256)]:PREP['prepare'](am,e['group'],0,split,count,32,compiled)
        del compiled,am;gc.collect();torch.cuda.empty_cache()


def state(plan,out):
    destdir=out/'state';destdir.mkdir(exist_ok=True,parents=True)
    for e in plan['models']:
        path=destdir/(e['id']+'.json')
        if path.exists():assert json.loads(path.read_text())['checkpoint']['sha256']==e['checkpoint_sha256'];continue
        begin=time.time();am=Model(e)
        cases,provenance=load_cases(plan,out,am,own=True,train_count=512,val_count=128)
        rows=STATE['select'](am,cases);features={s:STATE['extract'](am,cases[s],rr) for s,rr in rows.items()}
        results=[]
        for task in ['field_node','field_bar']+STATE['SETS']:
            categorical=task.startswith('field_');anchor='state_pre' if task=='successors_pre' else 'state'
            target=3 if task=='successors_pre' else (None if categorical else STATE['SETS'].index(task))
            dd={s:STATE['probe_data'](rr,features[s],task if categorical else anchor,target,am.tok) for s,rr in rows.items()}
            for hidden in [0,64]:
                fitted=fit_readout(dd['train'],dd['val'],outputs=(4 if task=='field_node' else 3) if categorical else am.tok.n_node_tokens,
                    categorical=categorical,hidden=hidden,steps=1500,seed=20261102)
                results.append(dict(task=task,architecture='linear' if hidden==0 else 'mlp64',selected_steps=fitted['steps'],
                    test=evaluate_readout(fitted,dd['test'],categorical)))
                del fitted
        lens,support=EARLY['fit_calibration'](am,cases,rows,1500)
        scored=STATE['lens_scores'](am,lens,features['test'],rows['test']);subsets={}
        for name,mask in EARLY['test_partitions'](am,cases['test'],rows['test']).items():
            ix=np.flatnonzero(mask);subrows=[rows['test'][i] for i in ix]
            subsets[name]=dict(positions=len(ix),queries=len({cases['test'][r['sid']]['query'] for r in subrows}),
                logit_lens=STATE['lens_scores'](am,lens,{k:v[ix] for k,v in features['test'].items()},subrows))
        transitions={}
        for layer in sorted({4,am.model.n_layers-2}):
            transitions[str(layer+1)]=LEGAL['analyse'](am,cases['test'],rows['test'],LEGAL['predictions'](am,lens,features['test'],layer))
        atomic_json(path,dict(model=e,checkpoint=am.checkpoint,sources=provenance,complete=True,
            rows={s:len(rr) for s,rr in rows.items()},heads=[f'L{i+1}' for i in range(am.model.n_layers)]+['token','token_position'],
            probes=results,logit_lens=scored,test_subsets=subsets,calibration_support=support,
            calibration_validation={k:lens[k] for k in ['steps','validation_kl','identity_kl']},
            semantic_transitions=transitions,seconds=time.time()-begin))
        assert sha_file(Path(e['checkpoint']))==e['checkpoint_sha256']
        print('STATE DONE',e['id'],round(time.time()-begin),flush=True)
        del am,features,lens;gc.collect();torch.cuda.empty_cache()


def clean_scores(scores):
    out=[]
    for r in scores:
        assert r['n']==2*r['positive']
        out.append({**{k:v for k,v in r.items() if k not in ['auroc','ap','average_precision','paired_rank']},
            'accuracy':(r['tpr']+1-r['fpr'])/2,'threshold':.5})
    return out


def history(plan,out):
    destdir=out/'history';destdir.mkdir(exist_ok=True,parents=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for e in plan['models']:
        for alignment in ['error_token_0_5','record_0_2']:
            path=destdir/(e['id']+'__'+alignment+'.json')
            if path.exists():assert json.loads(path.read_text())['checkpoint']['sha256']==e['checkpoint_sha256'];continue
            begin=time.time();am=Model(e);raw,provenance=load_cases(plan,out,am,own=False)
            cases={s:annotate(cc,am.tok) for s,cc in raw.items()};plans={};sampling={}
            kw={'error_offsets':range(6)} if alignment=='error_token_0_5' else {}
            for j,split in enumerate(['train','val','test']):plans[split],sampling[split]=match_readouts(cases[split],am.tok,20261030+j,**kw)
            own,own_info=match_readouts(cases['test'],am.tok,20261102,positive_source=e['id'],**kw)
            data={s:NH['extract'](am,cases[s],plans[s]) for s in ['train','val','test']}
            data['own_test']=NH['extract'](am,cases['test'],own) if own else None
            assert all(data[s] is not None for s in ['train','val','test'])
            checkpoint=am.checkpoint;del am;gc.collect();torch.cuda.empty_cache()
            runs=[]
            for hidden,arch in [(0,'linear'),(64,'mlp64')]:
                for seed in [0,1]:
                    fitted=fit_positions(data['train'],data['val'],5000,seed,hidden=hidden)
                    if fitted is None:runs.append(dict(architecture=arch,probe_seed=seed,status='insufficient_query_support'));continue
                    tests={s:clean_scores(position_scores(fitted,d)) for s,d in data.items() if s in ['test','own_test'] and d is not None}
                    exact=NH['subset'](data['test'],data['test']['exact_token'].astype(bool))
                    tests['exact_token_test']=clean_scores(position_scores(fitted,exact))
                    runs.append(dict(architecture=arch,probe_seed=seed,tests=tests,valid_cells=list(fitted['mapping']),selected_steps=fitted['selected_steps']))
                    del fitted;gc.collect();torch.cuda.empty_cache()
            shuffled={}
            for j,s in enumerate(['train','val','test']):
                d=data[s];rng=np.random.default_rng(4477+j);flips={int(event):int(rng.integers(2)) for event in np.unique(d['event'])}
                shuffled[s]={**d,'y':np.array([int(y)^flips[int(event)] for y,event in zip(d['y'],d['event'])],np.float32)}
            fitted=fit_positions(shuffled['train'],shuffled['val'],5000,0,hidden=64)
            runs.append(dict(architecture='mlp64',probe_seed=0,tests={'shuffle_control':clean_scores(position_scores(fitted,shuffled['test']))}))
            atomic_json(path,dict(model=e,checkpoint=checkpoint,sources=provenance,alignment=alignment,complete=True,
                sampling=sampling,own_sampling=own_info,runs=runs,seconds=time.time()-begin))
            assert sha_file(Path(e['checkpoint']))==e['checkpoint_sha256']
            print('HISTORY DONE',e['id'],alignment,round(time.time()-begin),flush=True)
            del data,shuffled,fitted;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','state','history'],required=True);args=parser.parse_args()
    os.chdir(ROOT);torch.set_num_threads(2)
    manifest=json.loads(args.manifest.read_text());assert manifest.get('lifecycle')!='deprecated'
    plan=json.loads((Path(manifest['output'])/'plan.json').read_text());assert plan['manifest']==manifest
    out=ROOT/'outputs/analysis/correction_mechanisms'/manifest['cohort'];out.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest_sha256=sha_file(args.manifest),behavior_plan_sha256=sha_file(Path(manifest['output'])/'plan.json'),
        probe_steps=5000,probe_seeds=[0,1],state_steps=1500,lens_steps=1500,
        corpus='1536 train / 256 val queries per same-depth seed0 donor at p0,p30,p60; shared/own/strict-token tests',
        state='512 train /128 val queries per same-depth donor; own natural test windows; 0.5 set threshold',
        sources={str(p.relative_to(ROOT)):sha_file(p) for p in [Path(__file__),ROOT/'src/cot_navigation/analysis/common.py',
            ROOT/'src/cot_navigation/analysis/state_readout.py',ROOT/'src/cot_navigation/analysis/history_probe.py',
            ROOT/'src/cot_navigation/analysis/within_token.py',ROOT/'studies/correction_reproduction/natural_history_probes/run.py',
            ROOT/'studies/correction_reproduction/state_readouts/run.py',ROOT/'studies/correction_reproduction/state_readouts/early_clean_lens.py',
            ROOT/'studies/correction_reproduction/state_readouts/legality_transitions.py']})
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==identity
    else:atomic_json(out/'plan.json',identity)
    try:globals()[args.phase](plan,out)
    except BaseException as exc:atomic_json(out/'failure.json',dict(phase=args.phase,error=repr(exc)));raise


if __name__=='__main__':main()
