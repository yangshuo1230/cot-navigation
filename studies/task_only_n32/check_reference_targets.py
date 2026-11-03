"""Validate every cached N32 task target, every split and all four variants."""
from pathlib import Path
from collections import Counter
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies/scale_trace_audit')]
from audit import semantic
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.protocols.tokenizer import Tokenizer

counts=Counter()
for p in ['direct','path','dfs_state','bfs_state','dfs_nodes','bfs_nodes']:
    cfg=load_config([ROOT/f'configs/experiments/task_only_n32_v1/{p}.yaml'])
    tok=Tokenizer.from_config(cfg)
    source=ParametricWorld(Path(cfg.data.root)/'main-seed0')
    cache=ScalePreparedRows(source,cfg,tok)
    nodes=source.node_vocab_ids(tok);raw=source.adj()
    adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(raw)}
    for idx in range(len(source.queries['label'])):
        s,t=nodes[int(source.queries['s_topo'][idx])],nodes[int(source.queries['t_topo'][idx])]
        for variant in range(cache.variants(idx)):
            row=cache.row(idx,variant);body=row.tokens[row.prompt_len:row.answer_index]
            if p=='direct':assert not body
            else:
                verdict=semantic(body,row.label,p,adj,s,t,tok,cfg.protocol.negative_random_path)
                assert verdict['complete_valid'] or verdict['stop'] and verdict['prefix_valid'],(p,idx,variant,verdict)
            counts[p]+=1
output=ROOT/'studies/task_only_n32/reference_audit.json'
output.write_text(json.dumps(dict(counts),indent=2))
print('PASS: all N32 targets, all splits and all variants:',dict(counts))
