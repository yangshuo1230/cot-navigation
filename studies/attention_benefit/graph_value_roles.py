"""Separate graph marker and node-endpoint V contextualization, baseline."""
import runpy,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
G=runpy.run_path(str(P/'graph_values_visited.py'));M=G['M']
G['evaluate'].__globals__['CONDITIONS']=['identity','marker_contextfree_V','endpoint_contextfree_V']
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
cfg,tok,model=M['load_model']('clean')
source=M['D']['ContextStore'](cfg.data.root);cache=M['D']['ScalePreparedRows'](source,cfg,tok)
rows=json.loads((M['AUDIT']/'strict/dfs_clean_main.json').read_text())['rows']
cases=M['build_cases'](source,cache,tok,[r['query_id'] for r in rows])
result=G['evaluate'](model,tok,cases);result['finished']=True
M['write'](P/'graph_value_roles.json',result)
