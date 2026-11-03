"""Temporary end-to-end check of stage, retention and native final reports."""
import json,tempfile
from pathlib import Path
import torch,yaml
from cot_navigation.config import load_config,dump_config
from cot_navigation.experiments.fine_node_curriculum import ROOT,MANIFEST,FineNodeTrainer,FineJob
from cot_navigation.experiments.scheduler import job_complete

def main():
    torch.set_num_threads(2)
    manifest=json.loads(MANIFEST.read_text())
    j=next(j for j in manifest['jobs'] if j['id']=='s0/bfs_nodes/hard')
    roots=[manifest['roots']['hard'][1],manifest['roots']['hard'][-1]]
    with tempfile.TemporaryDirectory(prefix='fine_nodes_persistence_') as tmp:
        path=Path(tmp);out=path/'run'
        cfg=load_config([ROOT/j['config']],[f'output_dir={json.dumps(str(out))}',
            'runtime.device=cpu','runtime.fused_adamw=disabled','model.compile=false',
            'train.compile_loss=false','train.precision=fp32','tracking.mode=disabled',
            'train.max_steps=4','train.effective_query_batch_size=4',
            'train.scale_stage_ends=[2,4]','train.scale_stage_weights=[1,0,0.3,0.7]',
            'train.scale_roots='+json.dumps(roots),'train.eval_interval=2','train.free_eval_interval=2',
            'train.save_interval=2','train.log_interval=1','train.eval_samples=4','train.free_gen_samples=4',
            'evaluation.test_samples=4','evaluation.compile_generation=false',
            'evaluation.max_new_tokens=32'])
        config=path/'config.yaml';config.write_text(dump_config(cfg))
        t=FineNodeTrainer(cfg);t.fit()
        assert (out/'checkpoints/stage_00000002.pt').is_file()
        assert (out/'checkpoints/stage_00000004.pt').is_file()
        stage1=json.loads((out/'eval/free_00000002.json').read_text())
        stage2=json.loads((out/'eval/free_00000004.json').read_text())
        for report in [stage1,stage2]:assert report['semantic']['n']==4
        retention=json.loads((out/'eval/retention_00000004_level0.json').read_text())
        assert retention['data_root']==roots[0] and retention['semantic']['n']==256
        final=json.loads((out/'eval/free_test_last.json').read_text())
        assert final['training_step']==4 and final['semantic']['n']==4
        assert job_complete(FineJob('ephemeral',config,[],out,4))
        assert not (out/'checkpoints/best.pt').exists()
    receipt=dict(passed=True,temporary_directory_cleaned=True,stage_checkpoints=2,
                 retention_queries=256,final_last_queries=4,native_scheduler_completion=True)
    (ROOT/'studies/fine_node_curriculum/verification_persistence.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':main()
