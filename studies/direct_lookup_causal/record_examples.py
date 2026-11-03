"""Keep a few reproducible counterexamples, without activation dumps."""
from study import *

def run(setting):
    pack=load(setting);m=pack['m'];panel=squares(pack,2);a,b,d,c=panel['indices'].T
    sp=[np.flatnonzero(m['split']==i) for i in range(3)]
    y=pack['reach'][:,m['target']].T.astype(np.float32)
    w,bias,_=coefficient_fit(pack['x'][:,1,1:2],y,sp);w=w[0];bias=bias[0]
    u=np.linalg.svd(w,full_matrices=False)[0][:,:32]
    base=pack['x'][a,1,1];donor=pack['x'][d,1,1];delta=(donor-base)@u@u.T
    semantic=base+delta;complement=donor-delta
    p,pvalid=forward_patch(pack,a,1,semantic);q,qvalid=forward_patch(pack,a,1,complement)
    source=m['source'][a];which=np.arange(len(a))
    score0=(base@w+bias)[which,source];scorep=(semantic@w+bias)[which,source];scoreq=(complement@w+bias)[which,source]
    expected=m['label'][b];eligible=panel['all_four_correct']&pvalid&qvalid
    chosen=[]
    for label in (0,1):
        ids=np.flatnonzero(eligible&(m['label'][a]==label)&((p>0)==m['label'][a])&((q>0)==expected)&((scorep>.5)==expected)&((score0>.5)==m['label'][a]))
        chosen.extend(ids[:3])
    result=[]
    for i in chosen:
        def name(v):return pack['tok'].name(pack['nodes'][int(v)])
        result.append(dict(source=name(source[i]),old_target=name(m['target'][a[i]]),new_target=name(m['target'][b[i]]),
            donor_source=name(m['source'][d[i]]),original_query_id=int(m['query_ids'][a[i]]),
            counterfactual_query_id=int(m['query_ids'][b[i]]),donor_query_id=int(m['query_ids'][d[i]]),
            old_graph_label=int(m['label'][a[i]]),new_graph_label=int(expected[i]),donor_label=int(m['label'][d[i]]),
            native_answer=int(m['native_pred'][a[i]]),ancestor_direction_answer=int(p[i]>0),complement_answer=int(q[i]>0),
            ancestor_readout_before=float(score0[i]),ancestor_readout_after_direction_edit=float(scorep[i]),
            ancestor_readout_after_complement_edit=float(scoreq[i]),native_margin=float(pack['margin'][a[i]]),
            ancestor_direction_margin=float(p[i]),complement_margin=float(q[i]),
            matched_depth_band=bool(panel['matched'][i]),layer=1,rank=32,
            note='.5 is an illustrative ridge-score cutoff, not a calibrated probability. Selection is descriptive, not an independent test.'))
    del pack;torch.cuda.empty_cache();return result

if __name__=='__main__':
    write(HERE/'counterexamples.json',{s:run(s) for s in ('simple','hard')})
