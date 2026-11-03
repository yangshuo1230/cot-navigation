"""Compare same held-out query predictions to the batch's existing full test."""
import json
from pathlib import Path
from collections import defaultdict
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('output')
args=parser.parse_args()
out=Path(args.output)
groups=defaultdict(list)
for line in (out/'predictions.jsonl').open():
    r=json.loads(line)
    if r['case']=='id':groups[r['model']].append(r)
mapping={'direct':'baseline/direct','direct_onehop':'direct_onehop/direct',
         'direct_curriculum':'direct_curriculum/direct','path':'baseline/path',
         'path_negative':'path_negative/path','dfs_state':'baseline/dfs_state',
         'bfs_state':'bfs_state/bfs_state','dfs_nodes':'dfs_nodes/dfs_nodes','bfs_nodes':'bfs_nodes/bfs_nodes'}
root=Path('outputs/explore_v1/launches/explore_v1_s012/context/standard/train50000')
result={}
for model,rows in groups.items():
    path=root/mapping[model.split('/')[1]]/'eval/predictions_test_last.jsonl'
    old={r['query_id']:r for line in path.open() for r in [json.loads(line)]}
    matched=[r for r in rows if r['index'] in old]
    diff=[r['index'] for r in matched if r['pred']!=old[r['index']]['pred']]
    result[model]={'matched_queries':len(matched),'different_final_predictions':len(diff),'different_ids':diff,
                   'note':'new batch32 versus original batch16 BF16; compared only eligible original test IDs'}
(out/'same_id_consistency.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
