"""Compact four-initialization summary; uncertainty is seed variation only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ds=[json.loads((HERE/f'seed{s}.json').read_text()) for s in range(4)]
stages=ds[0]['stages']; panels=list(ds[0]['native'])
result={'scope':'v6 N80 graph0, four initializations, fixed final step50000','native':{},'curves':{}}
for p in panels:
    result['native'][p]={k:float(np.mean([d['native'][p][k] for d in ds])) for k in ('yes_recall','no_recall','ba')}
    result['native'][p]['n']=ds[0]['native'][p]['n']
    result['curves'][p]={}
    sources={'original_head':lambda d:d['native_head'][p],
             'query_probe':lambda d:d['probes']['query_holdout']['panels'][p],
             'nodeheld_probe':lambda d:d['probes']['both_endpoint_node_holdout']['panels'][p]}
    for name,get in sources.items():
        vals=np.array([[v['balanced_accuracy'] for v in get(d)] for d in ds])
        result['curves'][p][name]={'mean':vals.mean(0).tolist(),'seed_sd':vals.std(0).tolist(),'per_seed':vals.tolist()}
max_delta=max(abs(d['native'][p]['ba']-d['native'][p]['official_ba']) for d in ds for p in panels)
result['verification']={'max_ba_difference_from_official_BF16':max_delta,'observation_hooks_change_logits':max(d['hook_max_error'] for d in ds)}
result['shuffled_nodeheld_BA'] = np.mean([[v['balanced_accuracy'] for v in d['probes']['both_endpoint_node_holdout']['shuffled_heldout']] for d in ds],0).tolist()
(HERE/'summary.json').write_text(json.dumps(result,indent=2))
fig,axes=plt.subplots(1,2,figsize=(14,4.5),sharey=True)
for ax,p in zip(axes,['short_test','long_TT_test']):
    for key,label,color in [('original_head','Original YES/NO head','#555555'),('query_probe','Linear probe: query split','#e08a22'),('nodeheld_probe','Linear probe: both endpoints held out','#2677b5')]:
        c=result['curves'][p][key];m=np.array(c['mean'])*100;s=np.array(c['seed_sd'])*100
        ax.plot(stages,m,label=label,color=color,marker='o',ms=3);ax.fill_between(stages,m-s,m+s,color=color,alpha=.10)
    ax.axhline(50,color='gray',ls=':',lw=1);ax.set_title(p);ax.tick_params(axis='x',rotation=60);ax.set_ylim(40,102);ax.grid(alpha=.2)
axes[0].set_ylabel('Balanced accuracy (%)');axes[1].legend(fontsize=8,loc='lower right')
fig.suptitle('Fixed DAG Direct v6: separator residuals, four seeds, N=80; A=attention, F=FFN')
fig.tight_layout();fig.savefig(HERE/'answer_layers.png',dpi=180);fig.savefig(HERE/'answer_layers.pdf');plt.close(fig)
lines=['# v6 Fixed DAG Direct: final results and sublayer answer readouts','',
       'One N80 adjacent-layer DAG, four random initializations, final step50000; no test checkpoint selection. Frozen native BF16 inference and all probe fits on GPU0. No saved activations, weights, checkpoints or W&B runs.','',
       'Native final BA is computed from full-vocabulary argmax. Intermediate original-head BA compares YES/NO logits after final LayerNorm (uncalibrated logit lens), not causal truncation or proof later layers are unnecessary.', '',
       'The strict probe fits only original training queries whose two endpoints are A nodes (40); validation uses V/V queries (20 nodes), testing T/T queries (20 nodes). Node sets are disjoint. Main-model training still saw V/T nodes in short QUERY/ADJ. Query-split probes use short+long training queries and short+AA validation; therefore no probe node holdout. Controls shuffle labels separately within each split.','',
       '|Panel|n|YES recall|NO recall|BA|','|---|---:|---:|---:|---:|']
for p,r in result['native'].items():lines.append(f"|{p}|{r['n']}|{r['yes_recall']:.2%}|{r['no_recall']:.2%}|{r['ba']:.2%}|")
lines += ['', '|Stage|TT original head|TT query probe|TT node-held probe|','|---|---:|---:|---:|']
for i,s in enumerate(stages):
    c=result['curves']['long_TT_test'];lines.append('|'+s+'|'+'|'.join(f"{c[k]['mean'][i]:.2%}" for k in ('original_head','query_probe','nodeheld_probe'))+'|')
lines += ['',f'Native BF16 maximum BA discrepancy from official report: {max_delta:.8f}; observation hooks max logit change: 0.',
          'Shuffled strict-node test mean BA by stage: '+', '.join(f'{v:.1%}' for v in result['shuffled_nodeheld_BA']),
          'Four initializations are not four independent graphs. AA has only seven negatives; TT has 57 positives/27 negatives. This v6 design is endpoint-role holdout, not the prior strictly witness-restricted R0/R2/R3 setup.']
(HERE/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(result['native'],indent=2));print('\n'.join(lines[-18:]))
