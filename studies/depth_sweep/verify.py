"""Ephemeral compiled GPU checks at every depth on real N16/N100 examples."""
import argparse,os,sys,json,gc
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--protocol',required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(Path(__file__).parent))
    from experiment import ROOT,HERE,Trainer
    from cot_navigation.config import load_config
    manifest=json.loads((HERE/'manifest.json').read_text());results=[]
    for depth in [3,6,9,12]:
        torch._dynamo.reset()
        job=next(j for j in manifest['jobs'] if j['family']=='hard' and j['seed']==0 and j['depth']==depth and j['protocol']==a.protocol)
        original=load_config([job['config']],[]);roots=[original.train.scale_roots[1],original.train.scale_roots[-1]]
        cfg=load_config([job['config']],['train.persist_outputs=false','tracking.mode=disabled','train.max_steps=2',
            'train.scale_roots='+json.dumps(roots),'train.scale_stage_ends=[1,2]','train.scale_stage_weights=[1,0,0,1]',
            'evaluation.final_checkpoints=[]','evaluation.max_new_tokens=32'])
        t=Trainer(cfg);losses=[]
        for _ in range(2):
            m=t._update();t.completed+=1;losses.append(m['total_loss'])
        t.raw.eval();t.model.eval();rows=[t._render_task(int(i)) for i in t._selected_ids('val',2)]
        report=t._free(rows);assert report['overall']['n']==2
        x=torch.tensor([rows[0].tokens[:rows[0].prompt_len]],device='cuda')
        cache=t.raw.new_cache(1,'cuda',torch.bfloat16)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            full=t.raw(x);cached=t._inference_model()(x,cache=cache,cache_mode='prefill')
            torch.testing.assert_close(cached,full,atol=.006,rtol=.03)
        assert not t._inference_model().compile_fallbacks
        results.append(dict(depth=depth,protocol=a.protocol,losses=losses,
                            cache_logits_match=True,compiled_generation=True,compile_fallbacks=0))
        print(results[-1],flush=True);t.tracker.finish();del t,cache;gc.collect();torch.cuda.empty_cache()
    (HERE/f'verification_{a.protocol}.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
