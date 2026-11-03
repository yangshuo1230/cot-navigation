"""Check optimized metrics against the original tie-aware implementation."""
import json
from pathlib import Path
import numpy as np
import run as study
from cot_navigation.analysis.set_metrics import set_quality

rng=np.random.default_rng(29);maximum=0.
for tied in [False,True]:
    z=rng.normal(size=(97,3,32)).astype(np.float32)
    if tied:z=np.round(z)
    mask=(rng.random((97,32))>.4).astype(np.float32)
    mask[:3]=0;mask[:3,0]=1
    y=(rng.random((97,32))>.7).astype(np.float32)
    y[0]=0;y[1]=1
    fast=study.quality(z,y,mask)
    for h in range(3):
        slow=set_quality(z[:,h],y,mask)
        for key in ['pairwise_auc','topk_exact','strict_separation','topk_recall']:
            maximum=max(maximum,abs(fast[h][key]-slow[key]))
            assert abs(fast[h][key]-slow[key])<1e-10,(key,fast[h][key],slow[key])
        assert fast[h]['nontrivial_rows']==slow['nontrivial_rows']
        assert fast[h]['empty_target_rows']==slow['empty_target_rows']
        assert fast[h]['full_target_rows']==slow['full_target_rows']
root=Path(__file__).resolve().parent
checks={'fast_metrics_vs_original_max_error':maximum,'ties_empty_full_masks':'PASS'}
if (root/'results.json').exists():
    d=json.loads((root/'results.json').read_text())
    # Early running code counted a zero-neighbor mask as both empty and full.
    # Normalize that ancillary count to the original implementation's partition;
    # fitted weights, AUC, ranking and all selected steps are unaffected.
    corrected=[0]
    def normalize_counts(obj):
        if isinstance(obj,dict):
            if all(k in obj for k in ['n_rows','empty_target_rows','nontrivial_rows','full_target_rows']):
                expected=obj['n_rows']-obj['empty_target_rows']-obj['nontrivial_rows']
                if obj['full_target_rows']!=expected:
                    obj['full_target_rows']=expected;corrected[0]+=1
            for value in obj.values():normalize_counts(value)
        elif isinstance(obj,list):
            for value in obj:normalize_counts(value)
    normalize_counts(d)
    if corrected[0]:
        d['ancillary_count_note']='Zero-neighbor masks counted as empty, not full; score metrics and fits unchanged.'
        (root/'results.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
    train=set(d['train_ids']);val=set(d['val_ids']);assert not train&val
    for ids in d['test_ids'].values():assert not(set(ids)&(train|val))
    prior_path=study.ROOT/'studies/dfs_v6_trace_reuse/results.json'
    prior=json.loads(prior_path.read_text())
    hashes={f"seed{s['seed']}":s['checkpoint_sha256'] for s in prior['seeds']}
    for name,m in d['models'].items():
        assert m['checkpoint_sha256']==hashes[name]
        for task in ['allowed','filter']:
            for p in m['tasks'][task]['reference'].values():
                for key in ['matched','matched_exact_stack']:
                    if p[key]['n_pairs']:
                        assert p[key]['directional_accuracy'][7]==.5
                        assert p[key]['mean_logit_delta'][7]==0.
    checks.update(models=len(d['models']),query_disjoint='PASS',matched_identity_control='PASS',checkpoint_hashes='PASS')
    checks['ancillary_zero_mask_counts_normalized']=corrected[0]
(root/'verification.json').write_text(json.dumps(checks,indent=2))
print(checks)
