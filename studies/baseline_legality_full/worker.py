"""Complete frozen baseline rollouts on exact existing dual-attention inputs."""
import argparse,os
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--protocol',required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--batch-indices',nargs='+',type=int);args=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['WANDB_MODE']='disabled'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:os.environ.setdefault(k,'2')
from pathlib import Path
import sys,json,gzip,time,hashlib
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate
from cot_navigation.analysis.common import sha256
FOLDER=Path(__file__).parent;DUAL=ROOT/'outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023_r1'
base=ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum'/args.protocol
cfg=load_config([base/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg);checkpoint=base/'checkpoints/latest.pt';fingerprint=sha256(checkpoint)
blob=torch.load(checkpoint,map_location='cpu',weights_only=False);step=int(blob['step'])
raw=build_model(cfg.model,len(tok),tok.pad).cuda().eval();raw.load_state_dict(blob['model']);raw.requires_grad_(False);del blob
model=CompiledInference(raw);torch.set_num_threads(2)
def atomic(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
for case in args.cases:
 suffix='' if case=='main' else '-'+case
 ref=DUAL/args.protocol/'serial/bidirectional/eval'/f'predictions_test_last{suffix}.jsonl'
 rows=[json.loads(x) for x in ref.read_text().splitlines()];assert len(rows)==256 and len({x['query_id'] for x in rows})==256
 contract={'version':1,'checkpoint_sha256':fingerprint,'checkpoint_step':step,
   'reference_sha256':sha256(ref),'protocol':args.protocol,'case':case,
   'precision':'bf16','compiled':True,'batch':64,'max_new_tokens':0,'cot_margin':cfg.protocol.cot_margin,'max_seq_len':cfg.model.max_seq_len}
 out=FOLDER/'baseline'/args.protocol/case;out.mkdir(parents=True,exist_ok=True)
 manifest=out/'contract.json'
 if manifest.exists():assert json.loads(manifest.read_text())==contract,'incompatible previous evaluation'
 else:atomic(manifest,contract)
 if (out/'COMPLETE.json').exists():print('REUSE',args.protocol,case,flush=True);continue
 start=time.monotonic()
 for at in range(0,len(rows),64):
  if args.batch_indices is not None and at//64 not in args.batch_indices:continue
  path=out/f'batch_{at//64:03d}.json.gz'
  if path.exists():
   with gzip.open(path,'rt') as f:previous=json.load(f)
   assert [x['query_id'] for x in previous['rows']]==[x['query_id'] for x in rows[at:at+64]]
   continue
  atomic(out/'status.json',dict(phase='generation',batch=at//64,processed=at,gpu=args.gpu,pid=os.getpid(),unix=time.time()))
  chunk=rows[at:at+64];t0=time.monotonic()
  outputs=generate(model,[x['prompt_tokens'] for x in chunk],args.protocol,tok,'bf16',max_new_tokens=0,cot_margin=cfg.protocol.cot_margin)
  records=[]
  for row,gen in zip(chunk,outputs):
   records.append({k:row[k] for k in ['query_id','label','shape','distance','prompt_tokens']} | {
      'pred':gen['pred'],'correct':gen['pred']==row['label'],'format_fail':gen['format_fail'],
      'trace_tokens':gen['trace_tokens'],'cot_interrupted':gen['cot_interrupted'],
      'forced_stop':gen['forced_stop'],'generation_limit_reached':gen['generation_limit_reached']})
  tmp=path.with_suffix(path.suffix+'.tmp')
  with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump({'rows':records,'generation_seconds':time.monotonic()-t0,
        'compile_fallbacks':list(model.compile_fallbacks)},f)
  tmp.replace(path)
  print('BATCH',args.protocol,case,at//64,'seconds',round(time.monotonic()-t0,1),'correct',sum(x['correct'] for x in records),'stop',sum(x['cot_interrupted'] for x in records),flush=True)
 if len(list(out.glob('batch_*.json.gz')))!=4:
  print('PARTIAL',args.protocol,case,flush=True);continue
 atomic(out/'COMPLETE.json',dict(n=len(rows),checkpoint_step=step,seconds_this_invocation=time.monotonic()-start,compile_fallbacks=list(model.compile_fallbacks)))
 assert sha256(checkpoint)==fingerprint
 print('DONE',args.protocol,case,flush=True)
