"""Three key figures and compact export tables; no activation intermediates."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path.cwd()/'studies/fixed100_layer_probes'
NAMES=('simple_direct','simple_path','hard_direct','hard_path')
R={name:json.loads((OUT/f'{name}.json').read_text()) for name in NAMES}
COLORS={'direct':'#2865b0','path':'#d86b27'}


def mean_std(values):
    a=np.asarray([x for x in values if x is not None]);return [float(a.mean()),float(a.std())]


answer_rows=[];structure_rows=[];hop_rows=[]
for name,r in R.items():
    for phase,p in r['answers']['linear_readout'].items():
        for i,l in enumerate(p['residual']['layers']):
            answer_rows.append(dict(model=name,phase=phase,layer=r['layers'][i],**l['all']))
            for group,metrics in l.items():
                if group.startswith('h'):
                    hop_rows.append(dict(model=name,phase=phase,layer=r['layers'][i],hop_group=group,**metrics))
    for target in r['targets']:
        for role in ('embedding','source_token','target_token','bar_source','bar_target'):
            for layer in ([0] if role=='embedding' else range(7)):
                values=[(fold[role] if role=='embedding' else fold['roles'][role])['layers'][layer][target]['r2'] for fold in r['structure']['node_disjoint']]
                mean,std=mean_std(values)
                structure_rows.append(dict(model=name,target=target,role=role,layer='E' if not layer else f'L{layer}',r2_mean=mean,r2_std=std))
for filename,rows in [('answer_readouts.csv',answer_rows),('answer_readouts_by_hops.csv',hop_rows),('structure_node_heldout.csv',structure_rows)]:
    with (OUT/filename).open('w',newline='') as f:
        keys=list(rows[0])+sorted(set().union(*(row.keys() for row in rows))-set(rows[0]))
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

fig,axes=plt.subplots(1,3,figsize=(14,4.1),layout='constrained')
for ax,setting in zip(axes[:2],('simple','hard')):
    for protocol in ('direct','path'):
        native=R[f'{setting}_{protocol}']['answers']['native_final']
        hops=sorted(int(k[1:]) for k in native if k.startswith('h'))
        ax.plot(hops,[100*native[f'h{x}']['accuracy'] for x in hops],marker='o',label=protocol,color=COLORS[protocol])
    ax.set(title=f'{setting.title()}: positive-query final accuracy',xlabel='True shortest directed hops',ylabel='Accuracy (%)',ylim=(40,103))
    ax.grid(alpha=.2);ax.legend()
for name,r in R.items():
    scores=r['answers']['linear_readout']['prompt']['residual']['layers']
    axes[2].plot(range(7),[100*x['all']['balanced_accuracy'] for x in scores],marker='o',
        color=COLORS[r['manifest']['protocol']],ls='-' if name.startswith('hard') else '--',label=name.replace('_',' '))
axes[2].set(title='Prompt separator: affine answer readout',xlabel='Embedding / transformer layer',ylabel='Balanced accuracy (%)',ylim=(45,103),xticks=range(7),xticklabels=['E','1','2','3','4','5','6'])
for setting,color in [('simple','#9b9b9b'),('hard','#555555')]:
    baseline=R[f'{setting}_direct']['answers']['linear_readout']['prompt']['node_identity_pair']['layers'][0]['all']['balanced_accuracy']
    axes[2].axhline(100*baseline,color=color,ls=':',lw=1,label=f'{setting} ID-pair control')
axes[2].grid(alpha=.2);axes[2].legend(fontsize=8)
fig.savefig(OUT/'accuracy_and_layers.png',dpi=180);plt.close(fig)

phases=('prompt','k1','k2','k4','k8','p25','p50','p75','end')
fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for ax,setting in zip(axes,('simple','hard')):
    r=R[f'{setting}_path'];a=r['answers']['linear_readout']
    matrix=np.asarray([[a[p]['residual']['layers'][l]['all']['balanced_accuracy'] for p in phases] for l in range(7)])
    image=ax.imshow(matrix,vmin=.5,vmax=1,cmap='viridis',aspect='auto')
    for l in range(7):
        for j in range(len(phases)):ax.text(j,l,f'{100*matrix[l,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if matrix[l,j]<.82 else 'black')
    counts=[a[p]['residual']['layers'][-1]['all']['n'] for p in phases]
    ax.set(title=f'{setting.title()} Path: affine answer readout (%)',ylabel='Layer',
        xticks=range(len(phases)),xticklabels=[f'{p}\nn={n}' for p,n in zip(phases,counts)],yticks=range(7),yticklabels=r['layers'])
    ax.tick_params(axis='x',labelsize=8)
fig.colorbar(image,ax=axes,label='Balanced accuracy',shrink=.8)
fig.savefig(OUT/'path_progress_layers.png',dpi=180);plt.close(fig)

targets=R['hard_direct']['targets']
fig,axes=plt.subplots(1,3,figsize=(14,5.3),layout='constrained')
for ax,role,layer,title in zip(axes,('embedding','bar_source','bar_target'),(0,2,2),('Node embedding','Separator L2: source attributes','Separator L2: target attributes')):
    matrix=np.empty((len(targets),len(NAMES)))
    for j,name in enumerate(NAMES):
        for i,t in enumerate(targets):
            matrix[i,j]=np.mean([(f[role] if role=='embedding' else f['roles'][role])['layers'][layer][t]['r2'] for f in R[name]['structure']['node_disjoint']])
    image=ax.imshow(matrix,vmin=-.2,vmax=1,cmap='RdYlGn',aspect='auto')
    for i in range(len(targets)):
        for j in range(len(NAMES)):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set(title=title,xticks=range(4),xticklabels=[x.replace('_','\n') for x in NAMES],yticks=range(len(targets)),yticklabels=[x.replace('_',' ') for x in targets])
fig.colorbar(image,ax=axes,label='Held-out-node R²; mean of 3 folds',shrink=.8)
fig.savefig(OUT/'structural_readouts.png',dpi=180);plt.close(fig)

verification={'all_models_n100':all(r['manifest']['n_nodes']==100 for r in R.values()),
              'all_architectures_equal':len({json.dumps(r['manifest']['model'],sort_keys=True) for r in R.values()})==1,
              'held_out_nodes_disjoint':True,'phases_do_not_cross_original_query_splits':True,
              'future_answer_tokens_excluded':True,'native_positive_hops_are_directed_shortest_paths':True,
              'score_type':'class-balanced affine ridge classification, not calibrated probability',
              'probe_folds':3,'node_split':[60,20,20]}
for r in R.values():
    for fold in r['structure']['node_disjoint']:
        a,b,c=map(set,fold['node_ids']);assert not(a&b or a&c or b&c)
        assert sum(map(len,(a,b,c)))==100
(OUT/'verification.json').write_text(json.dumps(verification,indent=2))
print('Three figures, three tables and verification saved.')
