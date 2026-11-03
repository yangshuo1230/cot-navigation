"""Model-first aggregation, compact report and one figure."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ds=[json.loads((HERE/f'seed{s}.json').read_text()) for s in range(4)]
stages=ds[0]['stages'];results=[]
def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None
for role,task in [('target','ancestors'),('source','descendants'),('source','ancestors'),('target','descendants'),('source','successors')]:
    for setup in ('pooled','A_full','A_matched','T'):
        for group in ('A','T'):
            for feature in stages+['random101','random202','random303','layer_oracle','candidate_prior','shuffled_E','shuffled_L1F','shuffled_L6F']:
                seed_stats=[]
                for d in ds:
                    records=[r for r in d['records'] if r['role']==role and r['setup']==setup]
                    values=[]
                    for r in records:
                        key='shuffled_features' if feature.startswith('shuffled_') else 'features'
                        name=feature.removeprefix('shuffled_')
                        v=r[key][name][task][group]
                        values.append(dict(column_auc=v['column']['auc'],row_auc=v['row']['auc'],
                            topk_recall=v['row']['recall_at_true_k'],random_recall=v['row']['uniform_recall'],
                            long_column_auc=v['long_column']['auc'],long_row_auc=v['long_row']['auc'],
                            long_topk_recall=v['long_row']['recall_at_true_k'],
                            nonempty_exact=v['recovery']['nonempty_exact_set'],f1=v['recovery']['macro_f1'],
                            eligible_columns=v['column']['eligible_columns'],long_eligible_columns=v['long_column']['eligible_columns'],
                            n_train=len(r['train']),n_val=len(r['validation']),n_test=len(r['test_'+group])))
                    seed_stats.append({k:mean([v[k] for v in values]) for k in values[0]})
                entry=dict(role=role,task=task,setup=setup,test_group=group,feature=feature,
                           mean={k:mean([s[k] for s in seed_stats]) for k in seed_stats[0]},
                           seed_stats=seed_stats)
                results.append(entry)
def get(role,task,setup,group,feature,metric='column_auc'):
    return next(r for r in results if (r['role'],r['task'],r['setup'],r['test_group'],r['feature'])==(role,task,setup,group,feature))['mean'][metric]
out={'scope':'v6 Direct final step50000; one N80 graph, four model seeds','rows':results}
(HERE/'summary.json').write_text(json.dumps(out,indent=2))
fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
for row,(role,task) in enumerate([('target','ancestors'),('source','descendants')]):
    for col,metric in enumerate(['column_auc','long_column_auc']):
        ax=axes[row,col]
        for setup,group,label,color,ls in [('A_matched','A','A -> held-out A','#2677b5','-'),('T','T','T -> held-out T','#d66a22','-'),('A_matched','T','A -> T (cross-group)','#2677b5','--'),('T','A','T -> A (cross-group)','#d66a22','--')]:
            vals=[get(role,task,setup,group,s,metric) for s in stages]
            ax.plot(stages,vals,label=label,color=color,ls=ls,marker='o',ms=3)
        ax.axhline(.5,color='gray',ls=':',lw=1);ax.set_ylim(.3,1);ax.grid(alpha=.2)
        ax.set_title(f'{role}: {task} / '+('all feasible distances' if col==0 else 'distance >= 4'))
        ax.tick_params(axis='x',rotation=45)
        if col==0:ax.set_ylabel('Fixed-candidate AUROC')
axes[0,1].legend(fontsize=8)
fig.suptitle('Node-held-out set probes: A and T; matched training sample counts')
fig.tight_layout();fig.savefig(HERE/'AT_set_probes.png',dpi=180);fig.savefig(HERE/'AT_set_probes.pdf');plt.close(fig)
lines=['# v6 Direct: A/T node-held-out structural probes','',
       'Four final step50000 Direct checkpoints, one N80 graph. A40/T20: only A nodes have long-query endpoint training. Original model saw all nodes in short queries and adjacency knowledge. V is not used for probe fitting; independent within-A/T folds supply validation.','',
       '## Design','',
       '- Frozen native BF16 QUERY forward. One focal-node vector per stage, averaging over all legal forward query contexts; nodes in the same layer have the same other-endpoint pool. This includes train/held-out query INPUTS, never output labels. Source state is causally independent of later target. Target context average can wash out query-specific information. E is the raw embedding.','- Exclude source nodes in last layer and target nodes in first layer, which have no forward contexts; A36/T18 per role remain. Test focal nodes never enter standardization, fit, ridge or threshold selection. Candidate output identities are fixed all80 nodes; this is not unseen-candidate decoding.','- Three node assignments17/31/47, five rotating folds: three folds train, one val, one test. Each A test fold has7–8 nodes; T3–4. A_matched has identical train/val sample counts to T-only, about10–11 train and3–4 val; A_full has about21–22 train and7–8 val. Matching is on sample counts, not every graph attribute. Pooled fits A+T jointly, then evaluates each group separately.','- Cross-group curves use the same fitted decoder: A->T and T->A. Within-group test focal nodes are also held out. Results averaged within each model first, then across four initializations; fold repetitions do not create new nodes.','- Ancestors, strict reachable descendants, direct successors; BFS independently verifies DP labels. 128->80 affine ridge, FP64 dual solve on GPU0, train-only scaling, validation MSE selection. Predicted scores are not probabilities.','- Main fixed-candidate AUROC compares different held-out focal nodes for a fixed candidate, then averages eligible candidates WITHIN each fold. Scores are never pooled across folds. Restrict to correct layer direction; successors next-layer only. Long metrics additionally restrict layer distance>=4, not a different decoder. Few T test nodes limit eligible candidate support.','- Row AUROC and top-k separately check true members outrank wrong candidates. Top-k uses true set size; empty/all-positive rows excluded. Full set recovery uses val-F1-selected threshold and excludes self, but no layer mask.','- Controls: three fixed random128-dimensional identity vectors, oracle layer one-hot, candidate prior only, and training labels permuted within each layer (validation/test true). Random identity may learn candidate priors, so high row AUC alone is not a node-specific signal.','- Intermediate set readout is not a causal-use test. Higher ancestor AUROC does not by itself show native YES/NO uses ancestors.','',
       '## Equal-sample within-group main result','',
       '|Role/task|Stage|A AUROC|T AUROC|A long AUROC|T long AUROC|','|---|---|---:|---:|---:|---:|']
fmt=lambda v:'NA' if v is None else f'{v:.3f}'
for role,task in [('target','ancestors'),('source','descendants')]:
    for stage in stages:
        vals=[get(role,task,setup,g,stage,m) for m in ('column_auc','long_column_auc') for setup,g in [('A_matched','A'),('T','T')]]
        lines.append('|'+role+'/'+task+'|'+stage+'|'+'|'.join(map(fmt,vals))+'|')
lines += ['', '## Common pooled decoder, separate A/T test nodes','', '|Role/task|Stage|A AUROC|T AUROC|','|---|---|---:|---:|']
for role,task in [('target','ancestors'),('source','descendants')]:
    for stage in ['E','L1F','L3F','L6F']:
        lines.append('|'+role+'/'+task+'|'+stage+'|'+fmt(get(role,task,'pooled','A',stage))+'|'+fmt(get(role,task,'pooled','T',stage))+'|')
lines += ['', '## Embedding controls, equal-sample within-group','', '|Role/task|Group|Embedding column AUC|Random column AUC|Layer column AUC|Shuffled column AUC|Embedding row AUC|Prior row AUC|Top-k recall|Uniform recall|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for role,task in [('target','ancestors'),('source','descendants')]:
    for setup,g in [('A_matched','A'),('T','T')]:
        vals=[get(role,task,setup,g,'E'),mean([get(role,task,setup,g,f'random{s}') for s in (101,202,303)]),get(role,task,setup,g,'layer_oracle'),get(role,task,setup,g,'shuffled_E'),get(role,task,setup,g,'E','row_auc'),get(role,task,setup,g,'candidate_prior','row_auc'),get(role,task,setup,g,'E','topk_recall'),get(role,task,setup,g,'E','random_recall')]
        lines.append('|'+role+'/'+task+'|'+g+'|'+'|'.join(map(fmt,vals))+'|')
lines += ['', '## Cross-group transfer (sample-matched A training)','', '|Role/task|Stage|A->A|A->T|T->T|T->A|','|---|---|---:|---:|---:|---:|']
for role,task in [('target','ancestors'),('source','descendants')]:
    for stage in ['E','L1F','L3F','L6F']:
        vals=[get(role,task,s,g,stage) for s,g in [('A_matched','A'),('A_matched','T'),('T','T'),('T','A')]]
        lines.append('|'+role+'/'+task+'|'+stage+'|'+'|'.join(map(fmt,vals))+'|')
lines += ['', '## Interpretation and limitations','',
          '- T does not lack linearly readable ancestor signal: matched within-group E is T .739 vs A .688, L1F .754 vs .779. A-trained decoder transfers to T at E .713 and L1F .721. Neither a missing representation nor completely incompatible coding is established.',
          '- The SAME pooled decoder reads target ancestors better on A than T after L1F (.861 vs .791), but this gap is not uniform across all four initializations. The matched A/T training experiment gives a much smaller L1F difference. Training support and group structure matter.',
          '- Source descendants remain readable on T, including L1F matched .820 vs A .796; no ancestor-only conclusion. Ancestor/descendant AUCs are different classification problems, not interchangeable information quantities.',
          '- Main fixed-candidate AUC exceeds random-vector and layer-only controls in pooled L1F probes. Nevertheless row ranking can be dominated by candidate priors: at E, matched target ancestor row AUC is A .798 vs prior .797, T .740 vs prior .743. High row AUC does not establish complete node-specific sets.',
          '- Complete nonempty-set recovery is near zero, including pooled target ancestors E 0% and L1F A1.7%/T0%. These probes recover partial ranking signal, not exact stored tables.',
          '- Long-only T AUROC can be very high but is based on only about6-8 eligible candidate columns per fold, each with3-4 test focal nodes; it is unstable and must not be described as near-perfect general decoding. Same-layer shuffled controls are weak for T-only training because many layer groups are singletons; some labels cannot change. Pooled shuffled controls are more informative.',
          '- Context means may hide query-specific representations, and late target states mix source information. Probe failure or decreasing AUC is not evidence that information is absent. None of these tests establish causal mediation of native answers.',
          'Detailed support, row rankings, direct successors, reciprocal role/task probes, full-A fits, exact-set recovery and per-model values are in summary.json and four seed JSONs. Only one key figure, no activation/probe weights/checkpoints or W&B runs.']
(HERE/'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[lines.index('## Equal-sample within-group main result'):]))
