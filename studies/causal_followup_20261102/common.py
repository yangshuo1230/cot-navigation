import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/causal_followup_20261102';PREV=ROOT/'studies/mechanism_probes_20261101'
def setup():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok
 model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
 store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];return cfg,tok,model,store,panel
def save(name,results):
 p=OUT/(name+'.json');p.write_text(json.dumps(results,ensure_ascii=False));print(json.dumps(dict(file=name,conditions=len(results))),flush=True)
def candidate(model,tok,seq):
 generated=[]
 for _ in range(12):
  with torch.inference_mode():log=model(torch.tensor(seq+generated)[None])
  v=int(log[0,-1].argmax());generated.append(v)
  if v==tok.id('|') or not tok.node_lo<=v<tok.node_lo+tok.n_node_tokens:break
 return generated
