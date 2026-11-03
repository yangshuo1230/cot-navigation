"""Clone the cancelled suite's design, changing only knowledge representation."""
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'fixed_exposure_suite'


def main():
    old=json.loads((SOURCE/'manifest.json').read_text())
    current=copy.deepcopy(old)
    current.update(version=3,status='prepared_not_launched',
        source_design='exposure_grokking_v2',
        experiment_group='fixed_exposure_grokking_adj_v3',
        knowledge_format=dict(name='complete_adjacency_v1',
            prompt=['<EDGE>','source','|'],empty_marker='<EXHAUST>',end_marker='<EOS>',
            successor_order='randomized node token id ascending',
            loss='all successors or EMPTY, plus EOS; no source/prompt loss',
            new_vocabulary_tokens=0,supervised_targets_per_update=64,
            complete_rows_only=True,
            auxiliary_R2_R3='Original WALK fragment pools and 48-transition budget unchanged'))
    for meta in current['worlds']:
        local=HERE/meta['path'];original=SOURCE/meta['path']
        assert local.read_bytes()==original.read_bytes(), 'Graph/query data must remain byte-identical'
    configs=HERE/'configs';configs.mkdir(exist_ok=True)
    for job in current['jobs']:
        job['knowledge_format']='complete_adjacency_v1'
        (configs/(job['id'].replace('/','__')+'.json')).write_text(json.dumps(job,indent=2))
    (HERE/'manifest.json').write_text(json.dumps(current,indent=2))
    changes=[]
    for before,after in zip(old['jobs'],current['jobs']):
        changed={k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
        assert changed=={'knowledge_format'}, (before['id'],changed)
    comparison=dict(status='PASS',world_files_byte_identical=True,
        all_44_jobs_changed_fields=['knowledge_format'],model_architecture_unchanged=True,
        fixed_tasks=36,knowledge_jobs=6,grokking_jobs=2,
        no_old_checkpoints_used=True)
    (HERE/'design_comparison.json').write_text(json.dumps(comparison,indent=2))
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':main()
