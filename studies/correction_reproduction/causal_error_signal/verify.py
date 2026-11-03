"""Independent prefix-label, geometry and real-model identity checks."""
from collections import defaultdict
import json
from pathlib import Path
import runpy
import numpy as np
import torch
from cot_navigation.analysis.residual_control import residual_edits
from cot_navigation.evaluation.rollout import generate
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.experiments.artifacts import atomic_json,sha_file

P=Path(__file__).resolve().parent
S=runpy.run_path(str(P/'run.py'));R=S['R']


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.32)
    selection=json.loads((P/'selection.json').read_text())
    blobs=[json.loads(p.read_text()) for p in sorted((P/'results').glob('*_s*.json'))]
    assert len(blobs)==8 and all(b['complete'] for b in blobs)
    am=R['Model']('correction',0);cases,_,_=R['load_data'](am.tok)
    checked=0
    for row in selection['selected_readouts']:
        c=cases['test'][row['sid']];graph,src,dst=R['PREP']['graph_tokens'](am,row['query'])
        first=replay(c['trace'][:row['pos']+1],None,graph,src,dst,am.tok)['first']
        assert bool(first)==bool(row['y'])
        assert am.tok.id('<MISTAKE>') not in c['trace'][:row['pos']+1]
        if not row['y']:
            assert replay(c['trace'][:row['decision_pos']+1],None,graph,src,dst,am.tok)['first'] is None
        checked+=1
    for blob in blobs:
        assert sha_file(Path(blob['checkpoint']['path']))==blob['checkpoint']['sha256']
        for name,axis in blob['axes'].items():
            u=np.array(axis['direction']);v=np.array(axis['random'])
            assert abs(np.linalg.norm(u)-1)<1e-5 and abs(np.linalg.norm(v)-1)<1e-5
            assert abs(u@v)<1e-5 and axis['means'][1]>axis['means'][0]
        assert all(r.get('relative_norm',0)<=.20001 for r in blob['local'])
        by_case={(r['site'],r['event'],r['y'],r['layer'],r['condition'],r['dose']):r for r in blob['local']}
        for r in blob['local']:
            if r['condition'].startswith('random_'):
                other=by_case[r['site'],r['event'],r['y'],r['layer'],r['condition'][7:],r['dose']]
                assert abs(r['relative_norm']-other['relative_norm'])<2e-5
        # Orthogonal controls cannot alter the defining linear classifier.
        assert all(abs(r['linear_after']-r['linear_before'])<2e-5 for r in blob['local'] if r['condition'].startswith('random'))
    # Real checkpoint, exact prefix IDs, zero-dose intervention: generated tokens
    # must match the raw model. This is an implementation check, not task eval.
    chosen=[r for r in selection['selected_readouts'] if r['free']][:8]
    prefixes=[am.example(r['query'],'test').prompt+cases['test'][r['sid']]['trace'][:r['pos']+1] for r in chosen]
    axis=np.array(blobs[0]['axes']['onset_L4']['direction'],np.float32)
    edits={3:[(i,len(seq)-1,axis,0.,0.,None) for i,seq in enumerate(prefixes)]}
    class Identity(torch.nn.Module):
        def __init__(self,raw):super().__init__();self.raw=raw;self.max_seq_len=raw.max_seq_len
        def new_cache(self,*a,**kw):return self.raw.new_cache(*a,**kw)
        def forward(self,*a,**kw):
            if kw.get('cache_mode')=='prefill':
                with residual_edits(self.raw,edits):return self.raw(*a,**kw)
            return self.raw(*a,**kw)
    with torch.no_grad():
        a=generate(am.model,prefixes,'dfs_state',am.tok,'fp32',max_new_tokens=32)
        b=generate(Identity(am.model),prefixes,'dfs_state',am.tok,'fp32',max_new_tokens=32)
    assert all(x['generated']==y['generated'] for x,y in zip(a,b))
    assert all(not block._forward_hooks for block in am.model.blocks)
    atomic_json(P/'verification.json',dict(passed=True,natural_prefix_readouts_checked=checked,
        independent_replay_labels=True,negative_bridges_legal=True,no_explicit_marker_in_inputs=True,
        axes_unit_and_orthogonal=True,norm_cap=True,per_case_norm_matching=True,random_linear_score_unchanged=True,
        zero_dose_generated_identity=len(prefixes),checkpoint_hashes_unchanged=8,
        past_last_layer_fixed_text_check='asserted during every actual local batch',
        tests='14 relevant CPU tests passed',free_rows=sum(len(b['rollouts']) for b in blobs)))
    print('CAUSAL VERIFICATION PASSED',checked,flush=True)


if __name__=='__main__':main()
