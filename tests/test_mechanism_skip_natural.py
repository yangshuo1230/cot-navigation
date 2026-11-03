"""Do not resume user-cancelled natural analysis; keep paired oracle controls."""
import gzip
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]


def test_persisted_skip_never_invokes_natural(tmp_path,monkeypatch):
    pipeline=runpy.run_path(str(ROOT/'studies/correction_mechanisms/pipeline.py'))
    manifest=dict(cohort='test',output='behavior')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    out=tmp_path/'outputs/analysis/correction_mechanisms/test';out.mkdir(parents=True)
    (out/'execution_choices.json').write_text(json.dumps(dict(skip_natural=True)))
    namespace=pipeline['main'].__globals__;monkeypatch.setitem(namespace,'ROOT',tmp_path)
    commands=[]
    monkeypatch.setattr(namespace['subprocess'],'run',lambda command,**kw:commands.append(command))
    monkeypatch.setattr(namespace['sys'],'argv',['pipeline.py','--manifest','manifest.json'])
    pipeline['main']()
    assert len(commands)==5
    assert not any(command[1]=='studies/correction_behavior/run.py' for command in commands)
    oracle=next(c for c in commands if c[1].endswith('oracle_mistake.py'))
    assert '--allow-partial-reference' in oracle
    status=json.loads((out/'pipeline_status.json').read_text())
    assert status['state']=='complete' and status['skipped']==['natural_behavior']
    assert 'natural_behavior' not in status['completed']


def test_oracle_missing_historical_reference_is_not_mismatch(tmp_path):
    oracle=runpy.run_path(str(ROOT/'studies/correction_behavior/oracle_mistake.py'))
    (tmp_path/'results').mkdir()
    (tmp_path/'plan.json').write_text(json.dumps(dict(models=[{}],manifest=dict(tasks=['main']))))
    pairs=[]
    for match in [None,True,False]:
        pair=dict(correct=True,valid=True,interrupted=False,forced=0,unsupported=0,
            original_hash_match=match,hash='unchanged')
        pairs.append(dict(oracle_pair=pair,correct=True,trace_valid=True,interrupted=False,repairs=0,trace_sha256='unchanged'))
    with gzip.open(tmp_path/'results/test.json.gz','wt') as f:
        json.dump(dict(complete=True,identity=dict(model=dict(group='g'),task='main'),rows=pairs),f)
    oracle['report'](tmp_path)
    result=json.loads((tmp_path/'comparison.json').read_text())['groups']['main']['g']
    assert result['n']==3 and result['reference_queries']==2 and result['native_original_hash_match']==1


def test_probe_reads_window_without_full_natural_analysis(tmp_path,monkeypatch):
    from types import SimpleNamespace
    runner=runpy.run_path(str(ROOT/'studies/correction_mechanisms/run.py'))
    monkeypatch.setitem(runner['load_cases'].__globals__,'ROOT',tmp_path)
    monkeypatch.chdir(tmp_path)
    entry=dict(id='l9_p30_s0',group='l9_p30',layers=9,seed=0,checkpoint_sha256='pinned')
    plan=dict(models=[entry],manifest=dict(output='unfinished_behavior'),tasks=dict(main=dict(queries=[3])))
    out=tmp_path/'mechanisms';folder=out/'corpus/native';folder.mkdir(parents=True)
    for split,q in [('train',1),('val',2),('test',3)]:
        blob=dict(complete=True,identity=dict(checkpoint=dict(sha256='pinned'),source_fingerprint='same'),
            rows=[dict(query=q,split=split,trace=[10,2],first=None,graph_hash=split,source=entry['id'])])
        with gzip.open(folder/f'l9_p30_s0_{split}.json.gz','wt') as f:json.dump(blob,f)
    am=SimpleNamespace(entry=entry,model=SimpleNamespace(n_layers=9),
        store=SimpleNamespace(fingerprint='same'),tok=SimpleNamespace(stoi={}))
    cases,provenance=runner['load_cases'](plan,out,am)
    assert cases['test'][0]['query']==3 and len(provenance)==3
    assert 'corpus/native' in provenance[-1]['path']


def test_probe_then_natural_with_forced_deferred(tmp_path,monkeypatch):
    pipeline=runpy.run_path(str(ROOT/'studies/correction_mechanisms/pipeline.py'))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(cohort='test',output='behavior',models=[])))
    out=tmp_path/'outputs/analysis/correction_mechanisms/test';out.mkdir(parents=True)
    (out/'execution_choices.json').write_text(json.dumps(dict(skip_natural=False,natural_after_probes=True,
        skip_forced=True,adopt_running=dict(natural_history_probes=123))))
    (out/'pipeline_status.json').write_text(json.dumps(dict(completed=['natural_probe_corpus','state_probes_and_lenses'],skipped=['natural_behavior'])))
    namespace=pipeline['main'].__globals__;monkeypatch.setitem(namespace,'ROOT',tmp_path)
    events=[]
    monkeypatch.setattr(namespace['subprocess'],'run',lambda command,**kw:events.append(command[1]))
    monkeypatch.setitem(namespace,'adopt_history',lambda *args:events.append('adopt_existing_history'))
    monkeypatch.setattr(namespace['sys'],'argv',['pipeline.py','--manifest','manifest.json'])
    pipeline['main']()
    assert events==['adopt_existing_history','studies/correction_behavior/run.py','studies/correction_mechanisms/report.py']
    status=json.loads((out/'pipeline_status.json').read_text())
    assert status['state']=='complete' and status['skipped']==['forced_mistake']
    assert status['completed'].index('natural_history_probes')<status['completed'].index('natural_behavior')


def test_adopt_worker_cannot_hide_incomplete_results(tmp_path,monkeypatch):
    import pytest
    pipeline=runpy.run_path(str(ROOT/'studies/correction_mechanisms/pipeline.py'))
    monkeypatch.setitem(pipeline['adopt_history'].__globals__,'process_running',lambda pid:False)
    manifest=dict(models=[dict(id='test',run='run',checkpoint_step=100)])
    with pytest.raises(FileNotFoundError):pipeline['adopt_history'](123,tmp_path,manifest)
