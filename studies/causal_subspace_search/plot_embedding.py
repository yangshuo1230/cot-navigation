"""Standalone embedding-intervention figure from completed causal search.

No new inference or fitting. Main curves are query-test donor-label success;
random shading is SD across 3 matched-edit-norm random directions, not a CI.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
fig,axes=plt.subplots(2,2,figsize=(10.8,7.5),layout='constrained',sharex=True,sharey=True)
export={}
for row,setting in enumerate(('simple','hard')):
    export[setting]={}
    for col,role in enumerate(('source','target')):
        ax=axes[row,col];d=DATA[setting];ranks=d['ranks']
        entries=[next(c for c in d['candidates'] if c['site']==[role,'E'] and c['rank']==r) for r in ranks]
        learned=np.array([c['test']['swap']['success'] for c in entries])*100
        complement=np.array([c['test']['complement']['success'] for c in entries])*100
        random=np.array([[v['success'] for v in c['test']['random_equal_norm']] for c in entries])*100
        full=np.array([c['test']['full']['success'] for c in entries])*100
        assert np.max(abs(full-full[0]))<1e-8
        x=np.arange(len(ranks))
        ax.plot(x,learned,'o-',color='#2675b9',lw=2.2,label='Learned subspace')
        ax.plot(x,complement,'s-',color='#d87929',lw=1.7,ms=4,label='Complement swap')
        ax.plot(x,random.mean(1),'o--',color='#777777',lw=1.6,ms=4,label='Equal-norm random')
        ax.fill_between(x,random.mean(1)-random.std(1),random.mean(1)+random.std(1),color='#777777',alpha=.15)
        ax.axhline(full[0],ls=':',color='#31915c',lw=1.5,label='Full embedding swap')
        for i,value in enumerate(learned):
            ax.annotate(f'{value:.1f}',(i,value),xytext=(0,-16),textcoords='offset points',ha='center',fontsize=9,color='#185d96')
        n=entries[0]['test']['swap']['n'];queries=d['panels'][role][2]['unique_queries']
        ax.set_title(f'{setting.capitalize()} DAG / {role} embedding\n{n} directed pairs, {queries} unique queries',fontsize=11)
        ax.set_xticks(x,ranks);ax.set_ylim(-3,108);ax.grid(alpha=.16)
        ax.spines[['top','right']].set_visible(False)
        if row==1:ax.set_xlabel('Subspace dimension')
        if col==0:ax.set_ylabel('Test counterfactual success (%)')
        if row==0 and col==0:ax.legend(fontsize=8,loc='center right')
        export[setting][role]=[dict(rank=r,learned=float(learned[i]),complement=float(complement[i]),
            random_mean=float(random[i].mean()),random_sd=float(random[i].std()),full=float(full[i]),
            n_pairs=n,n_unique_queries=queries) for i,r in enumerate(ranks)]
fig.suptitle('Direct N100: causal intervention in node embeddings\nShared projection; natural-donor coefficients; query train/val/test disjoint; model weights frozen',fontsize=12)
fig.savefig(HERE/'embedding_interventions.png',dpi=200)
fig.savefig(HERE/'embedding_interventions.pdf');plt.close(fig)
(HERE/'embedding_interventions.json').write_text(json.dumps(dict(
    metric='Original-correct directed test pairs: full-vocabulary argmax matches opposite-label natural donor.',
    note='Source edits fix target; target edits fix source. Random SD over 3 seeds, not CI. '
         'Embedding sweep has query holdout; no separate focal-node-holdout replication in this figure. '
         'Different roles have different pair cohorts. Percentages, not fractions.',results=export),indent=2))
print(HERE/'embedding_interventions.png')
