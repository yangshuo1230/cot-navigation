"""Read-only config and expected error-pool sampling distribution audit."""
import runpy,json
from pathlib import Path
import numpy as np
from dataclasses import asdict
D=runpy.run_path(str(Path(__file__).with_name('run.py')))
from cot_navigation.data.natural_mistakes import pool_path

def flatten(d,prefix=''):
 out={}
 for k,v in d.items():
  key=prefix+k
  if isinstance(v,dict):out.update(flatten(v,key+'.'))
  else:out[key]=v
 return out

def main():
 result={}
 import yaml
 for proto in ['dfs','bfs']:
  a,b=[flatten(yaml.safe_load((D['FOLDERS'][proto+'_'+x]/'resolved_config.yaml').read_text())) for x in ['clean','error']]
  differences={k:dict(clean=a.get(k),error=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
  effective=[flatten(asdict(D['load_config']([D['FOLDERS'][proto+'_'+x]/'resolved_config.yaml']))) for x in ['clean','error']]
  ea,eb=effective
  effective_diff={k:dict(clean=ea.get(k),error=eb.get(k)) for k in ea.keys()|eb.keys() if ea.get(k)!=eb.get(k)}
  cfg=D['load_config']([D['FOLDERS'][proto+'_error']/'resolved_config.yaml'])
  tok=D['Tokenizer'].from_config(cfg)
  sources=[]
  for root in cfg.train.scale_roots:
   store=D['ContextStore'](root)
   p=pool_path(store,cfg,tok)
   with np.load(p) as f:
    split=f['split'];fields=f['fields'];meta=[json.loads(str(x)) for x in f['metadata']]
    flat=f['tokens'];offsets=f['offsets']
   train=np.flatnonzero(split==0)
   for i,m in enumerate(meta):
    assert int(m['split'])==int(split[i])==int(store.arrays['split_id'][int(m['index'])])
   ids=store.indices('train')
   canonical=D['ScalePreparedRows'](store,cfg,tok)
   stop=tok.id('<STOP>');original_stop={};augmented_stop={}
   for i in train:
    m=meta[int(i)];r=canonical.row(int(m['index']),int(m['natural_variant']))
    original_stop[int(i)]=r.tokens[r.answer_index-1]==stop
    augmented_stop[int(i)]=int(flat[int(offsets[i])+int(fields[i,1])-1])==stop
   source={"root":root,"episodes":len(train),"by_label":{}}
   for label in [0,1]:
    baseline=ids[store.arrays['label'][ids]==label]
    pool=[int(i) for i in train if fields[i,2]==label]
    complex_ids=[i for i in pool if meta[i]['natural_complex']]
    weights={i:0.5/len(pool)+(0.5/len(complex_ids) if i in complex_ids else 0) for i in pool}
    assert abs(sum(weights.values())-1)<1e-9
    histogram={str(h):sum(w for i,w in weights.items() if meta[i]['distance']==h) for h in sorted(set(int(meta[i]['distance']) for i in pool))}
    base_hist={str(h):float(np.mean(store.arrays['distance'][baseline]==h)) for h in sorted(set(store.arrays['distance'][baseline].tolist()))}
    source['by_label'][label]=dict(pool_size=len(pool),complex_size=len(complex_ids),
      baseline_distance_mean=float(store.arrays['distance'][baseline].mean()),
      pool_expected_distance_mean=sum(weights[i]*meta[i]['distance'] for i in pool),
      pool_expected_events=sum(weights[i]*len(meta[i]['natural_events']) for i in pool),
      pool_expected_canonical_steps=sum(weights[i]*meta[i]['steps'] for i in pool),
      pool_expected_original_stop=sum(weights[i]*original_stop[i] for i in pool),
      pool_expected_augmented_stop=sum(weights[i]*augmented_stop[i] for i in pool),
      baseline_distance_hist=base_hist,pool_expected_distance_hist=histogram)
   sources.append(source)
  result[proto]=dict(config_differences=differences,effective_config_differences=effective_diff,
                     split_consistency_verified=True,sources=sources)
 D['write'](D['OUT']/'training_audit.json',result)
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
