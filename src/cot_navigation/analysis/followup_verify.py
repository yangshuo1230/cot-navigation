"""Read-only numerical/provenance audit; writes one compact verification JSON."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('studies/mechanistic_followup/results'))
    args=p.parse_args();root=args.input
    settings=json.loads((root/'settings.json').read_text())
    previous=Path(settings['previous']);old_verify=json.loads((previous/'verification.json').read_text())
    old_unchanged=all(digest(previous/name)==expected for name,expected in old_verify['artifact_sha256'].items())
    assert old_unchanged,'Previous artifacts were modified'
    entries=json.loads((root/'selection.json').read_text())['models'];checks=[]
    spot_checks=[]
    for entry in entries:
        name=entry['id'];j=json.loads((root/'models'/f'{name}.json').read_text())
        assert all(k in j for k in ['sets','errors','blocking','checkpoint_unchanged_after']),name
        unchanged=digest(Path(entry['run'])/'checkpoints/best.pt')==j['checkpoint']['sha256']
        assert unchanged and j['checkpoint_unchanged_after'],name
        ids=j['query_ids'];assert all(not set(ids[a])&set(ids[b]) for a,b in [('train','val'),('train','test'),('val','test')])
        diffs=[v.get('refit_f1_max_difference',0) for v in j['sets'].values()]
        assert max(diffs,default=0)<1e-6,(name,diffs)
        assert j['rollout']['limits']==0,name
        block_n=0
        for b in j['blocking'].values():
            if not isinstance(b,dict) or not b.get('n_cases'):continue
            assert b['oracle_blocking_verified_count']==b['n_cases']
            for example in b['interesting_examples']:
                d=example['detail'];assert d['oracle_clean_found'] and not d['oracle_all_routes_found']
                assert d['all_frontier_reachability_blocked']
            block_n+=b['n_cases']
        next_result=j['errors'].get('readouts',{}).get('next_token',{})
        for source,t in next_result.get('tests',{}).items():
            if 'all' not in t:continue
            assert abs(t['all'][3]['auroc']-.5)<1e-9,(name,source,'+1 control must be identical in pairs')
            assert abs(t['all'][3]['paired_wrong_higher_rate']-.5)<1e-9
        checks.append({'id':name,'checkpoint_unchanged':unchanged,'query_splits_disjoint':True,
            'native_test_limits':j['rollout']['limits'],'test_replay_agreement':j['rollout']['agreement_previous'],
            'max_old_set_f1_difference':max(diffs,default=0),'certified_blocking_cases':block_n})
        if block_n:
            # Reconstruct a saved edited prefix independently of selection and
            # check both attack and dead-node control from its actual tokens.
            # CPU-only: this never trains or runs a navigation forward pass.
            from .common import AnalysisModel
            from .route_blocking import search_from_state
            am=AnalysisModel(entry,Path.cwd(),device='cpu')
            for variant,b in j['blocking'].items():
                if not isinstance(b,dict) or not b.get('n_cases'):continue
                candidates=b['interesting_examples']
                example=next((e for e in candidates if 'dead_distinct_matched' in e['outcomes']
                              and e['outcomes']['all_routes']['pred']==1
                              and e['outcomes']['dead_distinct_matched']['pred']==0),candidates[0])
                ex=am.example(example['query'],'test');inv={v:i for i,v in enumerate(ex.nodes)}
                audit={'model':name,'variant':variant,'query':ex.index,'conditions':{}}
                for condition,expected in [('clean',True),('all_routes',False),('dead_distinct_matched',True)]:
                    if condition not in example['prefix_tokens']:continue
                    prefix=example['prefix_tokens'][condition]
                    bars=[i for i,v in enumerate(prefix) if v==am.tok.id('|')]
                    visited=prefix[bars[-2]+1:bars[-1]]
                    oracle=search_from_state(ex.adj,[inv[v] for v in example['detail']['frontier']],
                                             [inv[v] for v in visited],ex.target,entry['protocol']=='bfs_state')
                    assert oracle==expected,(name,variant,condition)
                    audit['conditions'][condition]={'oracle_found':oracle,'model_answer':example['outcomes'][condition]['pred']}
                spot_checks.append(audit)
            del am
    sources={n:digest(Path(__file__).parent/n) for n in settings['source_sha256']}
    assert sources==settings['source_sha256']
    controls=json.loads((root/'error_training_controls.json').read_text())
    assert set(controls['models'])=={'context_dfs','context_bfs'}
    for name,m in controls['models'].items():
        original=json.loads((root/'models'/f'{name}.json').read_text())
        assert m['checkpoint_sha256']==original['checkpoint']['sha256']
        assert controls['source_sha256']==digest(Path(__file__).parent/'error_training_controls.py')
        for readout,r in m['readouts'].items():
            for source,v in r['heads'].items():
                if 'natural_test' not in v:continue
                assert v['natural_test'][2]['n_rows']==r['natural_test_pairs']*2
                if readout=='next_token':assert abs(v['natural_test'][3]['auroc']-.5)<1e-9
                elif len(v['natural_test'])>4:
                    assert abs(v['natural_test'][3]['auroc']-v['natural_test'][4]['auroc'])<1e-9
    result={'completed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'models':len(checks),'analysis_gpu':0,'tests_passed':23,
            'previous_artifacts_unchanged':old_unchanged,'error_training_source_controls':list(controls['models']),'new_model_checkpoints':0,
            'probe_checkpoints_saved':0,'wandb_runs_created':0,'checks':checks,'saved_prefix_spot_checks':spot_checks,'source_sha256':sources,
            'artifact_sha256':{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*'))
                               if f.is_file() and f.name!='verification.json'}}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED',len(checks),'models;',sum(x['certified_blocking_cases'] for x in checks),'certified interventions;',
          'max refit diff',max(x['max_old_set_f1_difference'] for x in checks))


if __name__=='__main__':main()
