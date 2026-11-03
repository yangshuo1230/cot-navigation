"""Presentation figure from existing Direct probes; no model/probe retraining."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'direct_lookup_causal/relation_orientation'
TASKS=('ancestors','successors','descendants')
TITLES=dict(ancestors='all ancestors',successors='direct successors',descendants='reachable descendants')
def curves(result,role,task,random=False):
    records=result['roles'][role][task]['heldout']
    return np.array([[q['random'][str(seed)][l]['direction_valid_candidates']['fixed_candidate_across_targets']['auc']
                      for l in range(7)] for q in records for seed in (201,303,407)]) if random else np.array(
        [[q['actual'][l]['direction_valid_candidates']['fixed_candidate_across_targets']['auc'] for l in range(7)] for q in records])

fig,axes=plt.subplots(2,3,figsize=(12.5,6.6),layout='constrained',sharex=True,sharey=True)
for row,setting in enumerate(('simple','hard')):
    result=json.loads((OLD/f'{setting}.json').read_text())
    for col,task in enumerate(TASKS):
        ax=axes[row,col]
        for role,color in [('source','#2675b9'),('target','#d87929')]:
            values=curves(result,role,task);control=curves(result,role,task,True)
            ax.plot(range(7),values.mean(0),'o-',ms=4,color=color,label=role+' state')
            ax.fill_between(range(7),values.mean(0)-values.std(0),values.mean(0)+values.std(0),color=color,alpha=.13)
            ax.plot(range(7),control.mean(0),':',color=color,alpha=.75,label=role+' random identity')
        ax.axhline(.5,color='gray',lw=.8,ls='--');ax.grid(alpha=.16)
        ax.set_title(('Simple' if row==0 else 'Hard')+' N100 / '+TITLES[task],fontsize=11)
        ax.set_xticks(range(7),['E']+[f'L{i}' for i in range(1,7)])
        ax.set_ylim(.35,1.025)
        if col==0:ax.set_ylabel('Held-out-node AUROC')
        if row==1:ax.set_xlabel('Residual after transformer block')
        ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(fontsize=8,loc='lower right')
fig.suptitle('Direct: graph-relation decoding across layers\nFocal-node + query holdout; shading = SD across 3 node splits',fontsize=13)
fig.savefig(HERE/'direct_previous.png',dpi=200)
fig.savefig(HERE/'direct_previous.pdf')
print(HERE/'direct_previous.png')
