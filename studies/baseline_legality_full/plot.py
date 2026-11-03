"""Export publication-style comparison of semantic legality and answer accuracy."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).parent;z=json.loads((F/'summary.json').read_text());assert z['complete']
lookup={(x['model'],x['case']):x['overall'] for x in z['runs']}
cases=['n16','n24','n32','n36','n64','main','deep','wide','larger'];labels=['N16','N24','N32','N36','N64','N100','Deep100','Wide100','N144']
fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True,sharey=True)
styles=[('baseline','Single attention','#333333','--'),('parallel/causal','Parallel causal','#0072B2','-'),
 ('parallel/bidirectional','Parallel bidirectional','#D55E00','-'),('serial/causal','Serial causal','#009E73','-'),('serial/bidirectional','Serial bidirectional','#CC79A7','-')]
for row,protocol in enumerate(['dfs_state','bfs_state']):
 for col,(metric,title) in enumerate([('complete_valid','Complete semantic trace'),('correct','Final answer')]):
  ax=axes[row,col]
  for suffix,label,color,ls in styles:
   model='baseline/'+protocol if suffix=='baseline' else protocol+'/'+suffix
   values=[lookup[(model,c)][metric]*100 for c in cases]
   ax.plot(range(len(cases)),values,label=label,color=color,ls=ls,marker='o',lw=1.8,ms=4)
  ax.set_title(('DFS' if row==0 else 'BFS')+' | '+title);ax.set_ylim(0,103);ax.grid(alpha=.2)
  ax.set_xticks(range(len(cases)),labels,rotation=30,ha='right');ax.set_ylabel('Rate (%)')
handles,labels_=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels_,loc='lower center',ncol=5,frameon=False)
fig.suptitle('Single vs dual attention: matched frozen rollouts, 256 queries per DAG setup',y=.99)
fig.tight_layout(rect=(0,.06,1,.97))
for ext in ['png','svg','pdf']:fig.savefig(F/('legality_and_accuracy.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(13,6));models=[p+'/'+s for p in ['dfs_state','bfs_state'] for s in ['parallel/causal','parallel/bidirectional','serial/causal','serial/bidirectional']]
matrix=np.array([[100*(lookup[(model,c)]['complete_valid']-lookup[('baseline/'+model.split('/')[0],c)]['complete_valid']) for c in cases] for model in models])
lim=max(5,float(abs(matrix).max()));im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
ax.set_xticks(range(len(cases)),labels);ax.set_yticks(range(len(models)),[m.replace('_state','').replace('/',' | ') for m in models])
for i in range(len(models)):
 for j in range(len(cases)):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(matrix[i,j])>lim*.6 else 'black')
fig.colorbar(im,ax=ax,label='Change in complete trace validity (percentage points)');ax.set_title('Complete semantic trace validity: dual minus single-attention baseline')
fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(F/('legality_gain.'+ext),dpi=180,bbox_inches='tight')
