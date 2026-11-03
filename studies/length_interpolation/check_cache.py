"""Check length-specific KV-cache behavior using FP32 full-vs-cached logits."""
import json
from pathlib import Path
import torch
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.models.transformer import build_model
from cot_navigation.evaluation.rollout import pad_rows,cached_logits

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
root=Path('studies/length_generalization/results_n32')
meta=json.loads((root/'metadata.json').read_text())
targets=('id','id_disconnected_pad32','wide32_densityhalf','deep48')
samples={}
for line in (root/'predictions.jsonl').open():
    row=json.loads(line)
    if row['model']=='context/bfs_state' and row['case'] in targets and row['case'] not in samples:samples[row['case']]=row
tok=Tokenizer();rows=[];prompts=[]
for case in targets:
    r=samples[case]
    graph=next(x for x in meta['inputs'][case] if x['index']==r['index'])
    prompt=[x for u,v in graph['edges'] for x in (tok.id('<EDGE>'),graph['nodes'][u],graph['nodes'][v])]
    prompt += [tok.id('<QUERY>'),r['source'],r['target'],tok.id('|')]
    prompts.append(len(prompt));rows.append(prompt+r['trace_tokens'][:32])
cfg=load_config(['outputs/explore_v1/launches/explore_v1_s012/context/standard/train50000/bfs_state/bfs_state/resolved_config.yaml'])
model=build_model(cfg.model,len(tok),tok.pad).cuda().eval()
blob=torch.load('outputs/explore_v1/launches/explore_v1_s012/context/standard/train50000/bfs_state/bfs_state/checkpoints/latest.pt',map_location='cpu',weights_only=False)
model.load_state_dict(blob['model']);del blob
x,lens=pad_rows(rows,tok.pad,'cuda')
with torch.no_grad():
    full=model(x)
    cached=cached_logits(model,x,lens,'fp32')
result={}
for i,case in enumerate(targets):
    a,b=full[i,prompts[i]-1:lens[i]],cached[i,prompts[i]-1:lens[i]]
    error=float((a-b).abs().max())
    margins=a.topk(2,dim=-1).values
    result[case]={'prompt_len':prompts[i],'checked_positions':a.shape[0],
                  'max_abs_logit_difference':error,'max_logit_magnitude':float(a.abs().max()),
                  'mean_abs_logit_difference':float((a-b).abs().mean()),
                  'minimum_top2_margin':float((margins[:,0]-margins[:,1]).min()),
                  'argmax_disagreements':int((a.argmax(-1)!=b.argmax(-1)).sum())}
    # Different FP32 GEMM/SDPA shapes need not be bit-identical. Check decisions
    # and report numeric errors/margins rather than silently rounding logits.
    assert result[case]['argmax_disagreements']==0,result[case]
Path('studies/length_interpolation/cache_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
