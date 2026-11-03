"""Ephemeral CPU updates on real prepared data plus exact boundary resume."""
import json, tempfile, gc
from pathlib import Path
from unittest.mock import patch
import torch
from cot_navigation.config import load_config
from cot_navigation.experiments.fine_node_curriculum import (
    FineNodeTrainer, ROOT, MANIFEST, semantic_report)
from cot_navigation.protocols.tokenizer import Tokenizer

def main():
    torch.set_num_threads(2)
    manifest=json.loads(MANIFEST.read_text());results=[]
    # Check every stage boundary before constructing any data/model.
    for job in manifest['jobs']:
        cfg=load_config([ROOT/job['config']],[])
        t=FineNodeTrainer.__new__(FineNodeTrainer);t.cfg=cfg;t.scale_members=[None]*9
        for i,end in enumerate(cfg.train.scale_stage_ends):
            t.completed=end;t._evaluation_weights=True
            assert max(range(9),key=lambda j:t._scale_weights()[j])==i
            t._evaluation_weights=False
            assert max(range(9),key=lambda j:t._scale_weights()[j])==min(i+1,8)
    with tempfile.TemporaryDirectory(prefix='fine_nodes_check_') as tmp:
        for protocol in ['dfs_nodes','bfs_nodes']:
            for family in ['simple','hard']:
                job=next(j for j in manifest['jobs'] if j['seed']==0 and j['protocol']==protocol and j['family']==family)
                roots=[manifest['roots'][family][1],manifest['roots'][family][-1]]
                overrides=['runtime.device=cpu','runtime.fused_adamw=disabled','runtime.smoke=true',
                           'model.compile=false','train.compile_loss=false','train.precision=fp32',
                           'train.persist_outputs=false','tracking.mode=disabled','train.max_steps=4',
                           'train.effective_query_batch_size=4','train.scale_stage_ends=[2,4]',
                           'train.scale_stage_weights=[1,0,0.3,0.7]',
                           'train.scale_roots='+json.dumps(roots),
                           'evaluation.compile_generation=false','evaluation.final_checkpoints=[]']
                cfg=load_config([ROOT/job['config']],overrides)
                t=FineNodeTrainer(cfg)
                # After loading, all training examples must come from offline tokens.
                with patch('cot_navigation.data.dataset.ContextStore.render',side_effect=AssertionError('online graph search')):
                    for _ in range(2):t._update();t.completed+=1
                    ckpt=Path(tmp)/'resume.pt';torch.save(t._payload(),ckpt)
                    reference=t._update();t.completed+=1
                    expected={k:v.detach().clone() for k,v in t.raw.state_dict().items()}
                    t.tracker.finish();del t;gc.collect()
                    resumed=FineNodeTrainer(cfg,resume=str(ckpt))
                    actual=resumed._update();resumed.completed+=1
                delta=max((expected[k]-v).abs().max().item() for k,v in resumed.raw.state_dict().items())
                assert delta<1e-7,delta
                assert abs(reference['total_loss']-actual['total_loss'])<1e-7
                assert len(resumed.render_cache)<=32768
                results.append(dict(protocol=protocol,family=family,resume_max_weight_difference=delta,
                                    resumed_step=resumed.completed,reference_loss=reference['total_loss']))
                resumed.tracker.finish();del resumed;gc.collect()
        tok=Tokenizer(enable_stop=True)
        path=Path(tmp)/'predictions.jsonl'
        prompt=[tok.id('<EDGE>'),tok.node_id(0),tok.node_id(1),tok.id('<QUERY>'),tok.node_id(0),tok.node_id(1),tok.id('|')]
        records=[dict(prompt_tokens=prompt,trace_tokens=[tok.node_id(0),tok.node_id(1)],label=1,correct=True),
                 dict(prompt_tokens=prompt,trace_tokens=[tok.node_id(0),tok.node_id(0),tok.node_id(1)],label=1,correct=True)]
        path.write_text('\n'.join(json.dumps(x) for x in records))
        for fifo in [True,False]:
            report=semantic_report(path,tok,fifo)
            assert report['positive_valid_rate']==0.5 and report['positive_answer_accuracy']==1
            assert report['positive_errors']=={'repeated_node':1}
    receipt=dict(stage_boundary_config_checks=12*9,real_data_updates=12,
                 online_graph_search_in_updates=False,formal_runs_created=0,results=results)
    (ROOT/'studies/fine_node_curriculum/verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
