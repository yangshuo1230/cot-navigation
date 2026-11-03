"""Real batch-64 compiled updates and cached inference; no formal run files."""
import argparse,os,json,time,gc
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--protocol',required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2)
    from cot_navigation.config import load_config
    from cot_navigation.experiments.fine_node_curriculum import ROOT,MANIFEST,FineNodeTrainer
    m=json.loads(MANIFEST.read_text());results=[]
    config=next(j['config'] for j in m['jobs'] if j['seed']==0 and j['family']=='hard' and j['protocol']==a.protocol)
    for level in [1,8]:
        root=m['roots']['hard'][level]
        n=m['levels'][level]
        data_cfg=load_config([ROOT/f'configs/generated/fine_node_curriculum_20261028/data_hard_n{n}.yaml'],[])
        cfg=load_config([ROOT/config],['train.persist_outputs=false','tracking.mode=disabled',
            'train.max_steps=4','train.scale_stage_ends=[4]','train.scale_stage_weights=[1]',
            'data.root='+json.dumps(root),'data.n_nodes='+str(n),'data.n_edges='+str(n*2),
            'data.shapes='+json.dumps({'main':dict(layers=data_cfg.data.shapes['main'].layers,width=data_cfg.data.shapes['main'].width)}),
            'train.scale_roots='+json.dumps([root]),'evaluation.final_checkpoints=[]'])
        t=FineNodeTrainer(cfg);losses=[];t0=time.monotonic()
        for _ in range(3):
            metrics=t._update();t.completed+=1;losses.append(metrics['total_loss'])
        t.raw.eval();t.model.eval()
        rows=[t._render_task(int(i)) for i in t._selected_ids('val',8)]
        report=t._free(rows)
        # Prefix-cache logits must agree with the full causal forward.
        x=torch.tensor([rows[0].tokens[:rows[0].prompt_len]],device='cuda')
        model=t._inference_model();cache=t.raw.new_cache(1,'cuda',torch.bfloat16)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            full=t.raw(x)
            cached=model(x,cache=cache,cache_mode='prefill')
            torch.testing.assert_close(cached,full,atol=.006,rtol=.03)
        assert report['overall']['n']==8
        assert not getattr(model,'compile_fallbacks',set())
        results.append(dict(protocol=a.protocol,root=root,losses=losses,
                            update_plus_generation_seconds=time.monotonic()-t0,
                            free_queries=8,compiled_generation=report['compiled_generation'],
                            cache_full_logits_match=True,compile_fallbacks=0))
        t.tracker.finish();del t,model,cache;gc.collect();torch.cuda.empty_cache()
    receipt=ROOT/'studies/fine_node_curriculum'/f'verification_cuda_{a.protocol}.json'
    receipt.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
