"""Frozen-model replay and MISTAKE-logit ablation, exact historical batches.

Only compact positive-query audits and a few excerpts are persisted. All512
historical queries are decoded in batch16 to preserve numerical conditions;
negative traces are discarded after checking reproduction. No training/W&B.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

P=Path(__file__).resolve().parent;ROOT=P.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',required=True,choices=['original_clean','matched_clean','correction'])
parser.add_argument('--seed',required=True,type=int)
parser.add_argument('--gpu',type=int,help='Optional physical GPU; otherwise inherit launcher assignment')
parser.add_argument('--run-root',type=Path,default=ROOT/'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028')
args=parser.parse_args()
if args.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
os.environ['WANDB_MODE']='disabled'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:
    os.environ.setdefault(key,'2')
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch import nn
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.models.transformer import build_model
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate
from cot_navigation.protocols.correction_replay import replay,self_test
from cot_navigation.protocols.trace_audit import validate_states
from cot_navigation.experiments.artifacts import atomic_json as write,sha_file
from cot_navigation.training.checkpoint import load_checkpoint

FOLDER=args.run_root/f'dfs_state/{args.arm}/s{args.seed}'
DEST=P/'results'/f'{args.arm}_s{args.seed}.json'


class MaskMistake(nn.Module):
    def __init__(self,model,token=None):
        super().__init__();self.base=model;self.token=token;self.max_seq_len=model.max_seq_len
    def new_cache(self,*a,**kw):return self.base.new_cache(*a,**kw)
    def forward(self,*a,**kw):
        logits=self.base(*a,**kw)
        if self.token is not None:logits[...,self.token]=-float('inf')
        return logits


def main():
    torch.set_num_threads(2)
    result=json.loads(DEST.read_text()) if DEST.exists() else dict(finished=False,arm=args.arm,seed=args.seed,cases={})
    if result.get('finished'):return
    cfg=load_config([FOLDER/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    checks=self_test(tok)
    ckpt=FOLDER/'checkpoints/latest.pt';stat=ckpt.stat()
    raw=build_model(cfg.model,len(tok),pad_id=tok.pad).cuda().eval().requires_grad_(False)
    blob=load_checkpoint(ckpt,map_location='cpu');assert blob['step']==200000
    raw.load_state_dict(blob['model'],strict=True);del blob
    model=CompiledInference(raw).eval()
    result.update(checkpoint=str(ckpt),checkpoint_sha256=sha_file(ckpt),
        step=200000,batch_size=16,population='positive reachable label1; negatives only for batch reproduction',
        self_tests=checks,intervention='set only MISTAKE logit to -inf at all generation positions; otherwise identical',
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),audit_schema='first_error_v1',unified=True)
    roots=dict(x.split('=',1) for x in cfg.evaluation.extra_sets);roots['main']=cfg.data.root
    total_start=time.time()
    for task in ['main','n64','hard','deep','n32']:
        suffix='' if task=='main' else '-'+task
        original=[json.loads(s) for s in (FOLDER/f'eval/predictions_test_last{suffix}.jsonl').read_text().splitlines()]
        assert len(original)==512
        source=ContextStore(roots[task]);refcfg=cfg;reftok=tok
        if task=='hard':
            refcfg=load_config([ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml'])
            reftok=Tokenizer.from_config(refcfg);assert tok.itos[:len(reftok)]==reftok.itos
        prepared=ScalePreparedRows(source,refcfg,reftok)
        result['cases'].setdefault(task,{})
        if args.arm=='correction' and not result['cases'][task].get('no_mistake',{}).get('finished'):
            # A native/counterfactual pair must share this process's compiled
            # inference state; do not pair a resumed mask with an older decode.
            result['cases'][task].pop('native',None)
        for condition in (['native','no_mistake'] if args.arm=='correction' else ['native']):
            if result['cases'][task].get(condition,{}).get('finished'):continue
            start=time.time();rows=[];examples=[];mismatches=[];reference_checks=0
            inference=MaskMistake(model,tok.id('<MISTAKE>')).eval() if condition=='no_mistake' else model
            for offset in range(0,512,16):
                saved=original[offset:offset+16];refs=[prepared.row(r['query_id'],0) for r in saved]
                prompts=[r.tokens[:r.prompt_len] for r in refs]
                with torch.no_grad():
                    outputs=generate(inference,prompts,'dfs_state',tok,'bf16',max_new_tokens=0,cot_margin=cfg.protocol.cot_margin)
                # Identity proxy control checks that the masking wrapper alone
                # does not change decoded tokens, without an extra model cache.
                if args.arm=='correction' and args.seed==0 and task=='main' and condition=='native' and offset==0:
                    with torch.no_grad():sham=generate(MaskMistake(model).eval(),prompts,'dfs_state',tok,'bf16',max_new_tokens=0,cot_margin=cfg.protocol.cot_margin)
                    assert all(a['generated']==b['generated'] for a,b in zip(outputs,sham))
                    result['identity_proxy_control']=True
                for old,ref,g in zip(saved,refs,outputs):
                    same=(g['pred']==old['pred'] and (not g['format_fail'] and g['pred']==old['label'])==old['correct']
                          and g['cot_interrupted']==old['cot_interrupted'] and g['forced_stop']==old['forced_stop'])
                    if not same:mismatches.append(dict(query_id=old['query_id'],label=old['label'],
                        old_pred=old['pred'],new_pred=g['pred'],pred_changed=g['pred']!=old['pred']))
                    if old['label']!=1:continue
                    q=old['query_id'];nodes=source.node_vocab_ids(q,tok);adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(source.adj(q))}
                    src=nodes[int(source.arrays['s_topo'][q])];dst=nodes[int(source.arrays['t_topo'][q])]
                    seen={src};todo=[src]
                    while todo:
                        for v in adj.get(todo.pop(),[]):
                            if v not in seen:seen.add(v);todo.append(v)
                    assert dst in seen
                    detail=replay(g['trace_tokens'],g['pred'],adj,src,dst,tok,prompt_len=ref.prompt_len,
                        interrupted=g['cot_interrupted'],label=1)
                    # The strict shared validator is an independent sufficient
                    # witness. It has no MISTAKE cancellation and insists on
                    # visited order, so its failures are not our error labels.
                    check=validate_states(g['trace_tokens'],g['pred'],adj,src,dst,tok,False)
                    if check['valid']:
                        assert detail['answer_and_trace_valid'];reference_checks+=1
                    row=dict(query_id=q,distance=old['distance'],pred=g['pred'],interrupted=g['cot_interrupted'],
                        forced_stop=g['forced_stop'],prompt_len=ref.prompt_len,
                        trace_hash=hashlib.sha256(np.asarray(g['trace_tokens'],dtype=np.int32).tobytes()).hexdigest(),**detail)
                    assert row['correct']==(not g['format_fail'] and g['pred']==1)
                    if condition=='no_mistake':assert row['raw_mistake_tokens']==0
                    rows.append(row)
                    if len(examples)<3 and ((detail['first_repaired'] and not detail['correct']) or detail['outcome']=='correct_repaired_valid'):
                        pos=(detail['first'] or {}).get('token',1)-1
                        examples.append(dict(query_id=q,outcome=detail['outcome'],first=detail['first'],
                            excerpt=tok.decode(g['trace_tokens'][max(0,pos-20):pos+90])))
                print(args.arm,args.seed,task,condition,offset+len(saved),round(time.time()-start,1),flush=True)
            assert len(rows)==256
            result['cases'][task][condition]=dict(finished=True,rows=rows,examples=examples,seconds=time.time()-start,
                historical_mismatches=len(mismatches),historical_mismatch_examples=mismatches[:5],
                historical_positive_mismatches=sum(r['label']==1 for r in mismatches),
                historical_positive_pred_changes=sum(r['label']==1 and r['pred_changed'] for r in mismatches),
                historical_positive_correct=sum(r['correct'] for r in original if r['label']==1),
                strict_reference_witnesses=reference_checks)
            write(DEST,result)
        del prepared,source
    result.update(finished=True,seconds=time.time()-total_start,compile_fallbacks=sorted(model.compile_fallbacks))
    assert ckpt.stat().st_mtime_ns==stat.st_mtime_ns
    write(DEST,result);print('FINISHED',args.arm,args.seed,round(result['seconds'],1),flush=True)


if __name__=='__main__':main()
