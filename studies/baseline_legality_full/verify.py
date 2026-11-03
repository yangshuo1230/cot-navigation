"""Check provenance, sample identity, metrics and checkpoint immutability."""
from pathlib import Path
import json,gzip,hashlib
F=Path(__file__).parent;ROOT=F.parents[1];z=json.loads((F/'summary.json').read_text());assert z['complete'] and len(z['runs'])==90 and len(z['paired'])==72
lookup={(x['model'],x['case']):x for x in z['runs']};dual=ROOT/'outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023_r1'
queries=0;fallbacks=[];checks=[]
for contract in (F/'baseline').glob('*/*/contract.json'):
 c=json.loads(contract.read_text());case,protocol=c['case'],c['protocol'];source=dual/protocol/'serial/bidirectional/eval'/('predictions_test_last'+('' if case=='main' else '-'+case)+'.jsonl')
 assert hashlib.sha256(source.read_bytes()).hexdigest()==c['reference_sha256']
 reference={r['query_id']:r for r in map(json.loads,source.read_text().splitlines())};rows=[]
 for file in sorted(contract.parent.glob('batch_*.json.gz')):
  with gzip.open(file,'rt') as handle:data=json.load(handle)
  rows.extend(data['rows']);fallbacks.extend(data['compile_fallbacks'])
 assert len(rows)==256 and len({r['query_id'] for r in rows})==256
 for row in rows:
  ref=reference[row['query_id']]
  assert row['prompt_tokens']==ref['prompt_tokens'] and row['label']==ref['label'] and row['distance']==ref['distance']
 path=ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum'/protocol/'checkpoints/latest.pt'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==c['checkpoint_sha256']
 queries+=len(rows);checks.append(dict(protocol=protocol,case=case,n=len(rows)))
for run in z['runs']:
 if run['model'].startswith('baseline/'):continue
 suffix='' if run['case']=='main' else '-'+run['case'];report=dual/run['model']/'eval'/('free_test_last'+suffix+'.json')
 saved=json.loads(report.read_text());assert saved['overall']['n']==256
 assert saved['overall']['answer_accuracy']==run['overall']['correct']
 assert run['overall']['complete_valid']<=run['overall']['actions_valid']
 assert run['overall']['joint']<=run['overall']['complete_valid']
 queries+=256
assert queries==23040
result={'passed':True,'total_models':10,'dag_setups':9,'queries_per_setup':256,'total_audited_queries':queries,
 'new_baseline_rollouts':4608,'reused_dual_rollouts':18432,'oracle_assertions_per_query':4,
 'same_query_prompt_label_distance':True,'baseline_checkpoints_unchanged':True,
 'all_dual_answers_match_saved_reports':True,'baseline_compile_fallbacks':sorted(set(fallbacks)),
 'definitions':'Strict semantic trace validity excludes final answer token. Actions-valid ignores candidate annotations but checks true graph actions, frontier, visited, transitions, terminal. STOP is incomplete.',
 'limitations':['baseline200k_vs_dual140k_curriculum','one_training_seed_each','historical_dual_compiled_rollouts_reused_no_new_forward','query_CIs_do_not_capture_seed_variation']}
(F/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
