"""Full natural continuations; GPU0 only, no model training or token edits."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['WANDB_MODE']='disabled'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:
    os.environ.setdefault(key,'2')
P=Path(__file__).resolve().parent;ROOT=P.parents[2]

import numpy as np
import torch
from cot_navigation.analysis.error_continuation import audit_continuation
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.evaluation.rollout import generate
from cot_navigation.experiments.artifacts import atomic_json,sha_file
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.checkpoint import load_checkpoint

RUN=ROOT/'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state'
TASKS=['n32','n64','main','deep','hard']


def save(path,data):
    path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix('.tmp')
    with gzip.open(temp,'wt',encoding='utf-8',compresslevel=3) as f:json.dump(data,f,separators=(',',':'))
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--queries',type=int,default=256)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3])
    a=ap.parse_args();os.chdir(ROOT);torch.set_num_threads(2)
    identity=dict(schema=1,queries=a.queries,batch=a.batch,precision='bf16',gpu=0,
        checkpoints='last200k',population='reachable official test queries',
        decoding='natural full greedy; native 4096 context; no token edits or MISTAKE suppression',
        query_seed=20261031,source_sha256={str(f.relative_to(ROOT)):sha_file(f) for f in
            [Path(__file__),ROOT/'src/cot_navigation/analysis/error_continuation.py',ROOT/'src/cot_navigation/evaluation/rollout.py']})
    atomic_json(P/'plan.json',identity)
    for seed in a.seeds:
        for arm in ['matched_clean','correction']:
            files=[P/'results'/f'{arm}_s{seed}_{task}.json.gz' for task in TASKS]
            if all(f.exists() and json.load(gzip.open(f,'rt'))['complete'] for f in files):continue
            folder=RUN/arm/f's{seed}';cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
            ckpt=folder/'checkpoints/latest.pt';checkpoint=dict(path=str(ckpt),sha256=sha_file(ckpt))
            raw=build_model(cfg.model,len(tok),pad_id=tok.pad).cuda().eval().requires_grad_(False)
            blob=load_checkpoint(ckpt,map_location='cpu');assert blob['step']==200000
            raw.load_state_dict(blob['model'],strict=True);del blob
            model=CompiledInference(raw).eval()
            roots=dict(x.split('=',1) for x in cfg.evaluation.extra_sets);roots['main']=cfg.data.root
            for task,path in zip(TASKS,files):
                spec=dict(identity,arm=arm,model_seed=seed,task=task,checkpoint=checkpoint)
                data=json.load(gzip.open(path,'rt')) if path.exists() else dict(identity=spec,complete=False,rows=[])
                assert data['identity']==spec
                if data['complete']:continue
                suffix='' if task=='main' else '-'+task
                historical=[json.loads(s) for s in (folder/f'eval/predictions_test_last{suffix}.jsonl').read_text().splitlines()]
                positives=[r for r in historical if r['label']==1]
                order=np.random.default_rng(20261031).permutation(len(positives))[:a.queries]
                selected=[positives[int(i)] for i in order]
                source=ContextStore(roots[task]);reference=cfg
                if task=='hard':reference=load_config([ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml'])
                reftok=Tokenizer.from_config(reference);assert tok.itos[:len(reftok)]==reftok.itos
                prepared=ScalePreparedRows(source,reference,reftok)
                assert [r['query'] for r in data['rows']]==[r['query_id'] for r in selected[:len(data['rows'])]]
                started=time.time()
                for offset in range(len(data['rows']),len(selected),a.batch):
                    old=selected[offset:offset+a.batch];refs=[prepared.row(r['query_id'],0) for r in old]
                    prompts=[r.tokens[:r.prompt_len] for r in refs]
                    with torch.inference_mode():outputs=generate(model,prompts,'dfs_state',tok,'bf16',max_new_tokens=0,cot_margin=cfg.protocol.cot_margin)
                    for previous,ref,g in zip(old,refs,outputs):
                        q=previous['query_id'];nodes=source.node_vocab_ids(q,tok)
                        adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(source.adj(q))}
                        src=nodes[int(source.arrays['s_topo'][q])];dst=nodes[int(source.arrays['t_topo'][q])]
                        trace=g['trace_tokens'];audit=audit_continuation(trace,adj,src,dst,tok)
                        original=replay(trace,g['pred'],adj,src,dst,tok,label=1)
                        # Independent shared semantic replayer checks the legal prefix.
                        first=next((r for r in audit['records'] if r['bad']),None)
                        if first is not None and original['first'] is not None:
                            assert first['record']==original['first']['record'],(arm,seed,task,q,first,original['first'])
                        data['rows'].append(dict(query=q,distance=previous['distance'],
                            correct=bool(g['pred']==1 and not g['format_fail']),pred=g['pred'],
                            historical_correct=previous['correct'],historical_pred=previous['pred'],
                            interrupted=g['cot_interrupted'],forced_stop=g['forced_stop'],
                            trace_valid=original['trace_valid'],first=original['first'],
                            trace_tokens=len(trace),prompt_tokens=ref.prompt_len,
                            trace_sha256=hashlib.sha256(np.asarray(trace,np.int32).tobytes()).hexdigest(),**audit))
                    data['seconds_current_process']=time.time()-started
                    save(path,data)
                    print('CONTINUATION',arm,seed,task,len(data['rows']),len(selected),round(time.time()-started,1),flush=True)
                data.update(complete=True,compile_fallbacks=sorted(model.compile_fallbacks))
                save(path,data);del prepared,source
            assert sha_file(ckpt)==checkpoint['sha256']
            del model,raw;torch.cuda.empty_cache()
    atomic_json(P/'finished.json',dict(complete=True,files=len(a.seeds)*2*len(TASKS)))


if __name__=='__main__':main()
