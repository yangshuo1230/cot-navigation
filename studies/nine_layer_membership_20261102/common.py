import sys,json,torch,numpy as np,hashlib
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/nine_layer_membership_20261102';PREV=ROOT/'studies/correction_attention_compare_20261102'
def bindings():return [m for m in json.loads((PREV/'models.json').read_text()) if m['name'].startswith('corr9')]
def load(binding):
 torch.set_num_threads(4);run=Path(binding['run']);cp=run/'checkpoints/best.pt';assert hashlib.sha256(cp.read_bytes()).hexdigest()==binding['checkpoint_sha256'];cfg=a.load_config([run/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok
 model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);return cfg,tok,model
def store_plan():return a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']),json.loads((a.OUT/'plan.json').read_text())['panels']['n100']
def generate(model,tok,prefix):
 out=[]
 for _ in range(12):
  with torch.inference_mode():log=model(torch.tensor(prefix+out)[None])
  token=int(log[0,-1].argmax());out.append(token)
  if token==tok.id('|') or not tok.node_lo<=token<tok.node_lo+tok.n_node_tokens:break
 return out
def evaluate(out,gold,tok):
 nodes=out[:-1] if out[-1]==tok.id('|') else out
 return dict(set_correct=out[-1]==tok.id('|') and len(nodes)==len(set(nodes)) and set(nodes)==set(gold),STOP=any('STOP' in tok.name(v) for v in out))
def save(name,value):
 OUT.mkdir(exist_ok=True);(OUT/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False))
