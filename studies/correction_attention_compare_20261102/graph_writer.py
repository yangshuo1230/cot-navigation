"""Check L1 reads at the model-specific graph carrier offsets, graph prefix only."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/correction_attention_compare_20261102'
def main():
 torch.set_num_threads(4);models=json.loads((OUT/'models.json').read_text());extra=json.loads((OUT/'extra_routes.json').read_text());plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100'];store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);results=[]
 for model_spec in models:
  run=Path(model_spec['run']);name=model_spec['name'];cfg=a.load_config([run/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model']);offset=extra[name]['offset_scan']['offset']
  for split in ['val','test']:
   for spec in plan[split]:
    row=store.render(spec['index'],'dfs_state',tok);n=row.prompt_len-4;ids=torch.tensor(row.tokens[:n])[None];positions=torch.arange(n)[None];block=model.blocks[0]
    with torch.inference_mode():_,w=block.attn(block.norm1(model.embed(ids)),positions,return_attn=True)
    ds=[3*i+2 for i in range(n//3) if 0<=3*i+2+offset<n];m=float(np.mean([float(w[0,0,d+offset,d]) for d in ds]));results.append(dict(model=name,index=spec['index'],split=split,offset=offset,L1_carrier_reads_original_destination=m,n_edges=len(ds)))
  print(json.dumps(dict(model=name,offset=offset)),flush=True)
 (OUT/'graph_writer.json').write_text(json.dumps(results))
if __name__=='__main__':main()
