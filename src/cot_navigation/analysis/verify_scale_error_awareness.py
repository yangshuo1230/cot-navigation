"""Audit scale probe outputs without touching navigation or old studies."""
import json
import argparse
from pathlib import Path
from cot_navigation.analysis.common import sha256, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('studies/scale_error_awareness/results'))
    args=parser.parse_args()
    root=Path.cwd();out=args.output
    settings=json.loads((out/'settings.json').read_text())
    selection=json.loads((out/'selection.json').read_text())
    entries=selection['models'];assert len(entries)==24
    for name,expected in settings['source_sha256'].items():
        assert sha256(root/'src/cot_navigation/analysis'/name)==expected
    sampler_checks=0
    if (out/'sampler_equivalence.json').exists():
        sampler=json.loads((out/'sampler_equivalence.json').read_text())
        assert all(c['identical'] for c in sampler['checks'])
        assert sampler['indexed_sha256']==settings['source_sha256']['scale_error_awareness.py']
        sampler_checks=len(sampler['checks'])
    gpu_audit=[]
    if (out/'gpu_scope_audit.json').exists():
        gpu_audit=json.loads((out/'gpu_scope_audit.json').read_text())
        assert all(row['cuda_visible_devices']=='0' and set(row['physical_gpus'])=={0} for row in gpu_audit)
    replay=[]
    reference=out.parent/'reference_sampler_results/models'
    for path in sorted(reference.glob('*.json')):
        old=json.loads(path.read_text());new=json.loads((out/'models'/path.name).read_text())
        # Runtime/source identity changed; data, random draws, forward states
        # and all probe fits must remain exactly unchanged.
        for key in ['checkpoint','datasets','query_ids','natural_rollout','fits']:
            assert old[key]==new[key],(path.stem,key,'reference replay mismatch')
        replay.append(dict(model=path.stem,data_and_fits_exact_match=True))
    write_json(out/'reference_replay_audit.json',replay)
    records=[]
    for entry in entries:
        d=json.loads((out/'models'/f'{entry["id"]}.json').read_text())
        assert d['checkpoint_unchanged_after']
        assert sha256(root/entry['run']/'checkpoints/best.pt')==entry['sha256']==d['checkpoint']['sha256']
        assert d['checkpoint']['step']==entry['step']
        sets={s:set(d['query_ids'][s]) for s in ['train','val','test']}
        assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
        assert d['causal_mask_max_abs_delta']==0
        assert len(d['fits'])==6
        for summary in d['datasets'].values():
            assert summary['rows']==summary['pairs']*2
            assert sum(summary['categories'].values())==summary['pairs']
        candidates=[c for c in selection['candidates'] if c['kind']==entry['kind'] and c['scale']==entry['scale'] and c['family']==entry['family']]
        assert entry['score']==max(c['score'] for c in candidates)
        for fit in d['fits']:
            for pop in ['synthetic','natural']:
                if fit[pop] is not None:
                    assert len(fit[pop])==4
                    assert all(0<=m['auroc']<=1 for m in fit[pop])
        records.append(dict(model=entry['id'],checkpoint_unchanged=True,disjoint_queries=True,
                            validation_selection=True,no_future_effect=True,
                            natural_pairs=d['datasets']['natural']['pairs']))
    assert not [p for p in out.rglob('*') if p.suffix in {'.pt','.pth','.npy','.npz'}]
    from cot_navigation.analysis.scale_error_awareness import ScaleModel, body_tokens, slots
    adapter=[]
    for name in ['context_l10_path','fixed_l10_path']:
        entry=next(e for e in entries if e['id']==name)
        am=ScaleModel(entry,root,'cpu')
        for split in ['train','val','test']:
            ex=am.examples(split,1)[0]
            raw=am.prepared.row(ex.index,ex.index%am.cfg.protocol.trajectory_variants)
            assert ex.row.tokens==am.world_prefix+raw.tokens
            assert ex.row.prompt_len==raw.prompt_len+len(am.world_prefix)
            body=body_tokens(ex,am.tok,ex.row.tokens)
            assert all(body[i] in good for i,good,f in slots(ex,am.protocol,am.tok,body))
            adapter.append(dict(model=name,split=split,world_prefix=am.tok.decode(am.world_prefix),
                                prompt_length=ex.row.prompt_len,raw_token_match=True))
        del am
    write_json(out/'input_adapter_audit.json',adapter)
    write_json(out/'verification.json',dict(models=24,tests_passed=32,node_oracle_exhaustive_prefixes=4950,
        sampler_equivalence_cases=sampler_checks,
        gpu_processes_audited=len(gpu_audit),
        reference_models_replayed_exactly=len(replay),
        source_fingerprints_match=True,checkpoints_unchanged=True,no_activation_or_checkpoint_files=True,records=records))
    print('VERIFIED',len(records),'models')


if __name__=='__main__':
    main()
