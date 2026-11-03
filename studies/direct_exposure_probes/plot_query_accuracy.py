"""Plot existing Direct answer-probe results; no new fitting or inference."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'fixed100_layer_probes/answer_readouts.csv'
with SOURCE.open(newline='',encoding='utf-8-sig') as stream:
    rows=list(csv.DictReader(stream))
layers=['E']+[f'L{i}' for i in range(1,7)]
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained',sharey=True)
selected={}
for model,label,color in [('simple_direct','Simple DAG, N100','#2675b9'),
                          ('hard_direct','Hard DAG, N100','#d87929')]:
    items={r['layer']:r for r in rows if r['model']==model and r['phase']=='prompt'}
    assert set(items)==set(layers)
    selected[model]=[items[l] for l in layers]
    for ax,metric,title in zip(axes,('accuracy','balanced_accuracy'),
                               ('Query accuracy','Balanced query accuracy')):
        values=np.array([float(items[l][metric]) for l in layers])*100
        ax.plot(range(7),values,'o-',color=color,label=label,lw=2,ms=5)
        for i,value in enumerate(values):
            other='hard_direct' if model=='simple_direct' else 'simple_direct'
            other_value=100*float(next(r for r in rows if r['model']==other and r['phase']=='prompt' and r['layer']==layers[i])[metric])
            if model=='hard_direct' and abs(value-other_value)<.01:continue
            offset=9 if value>=other_value else -14
            ax.annotate(f'{value:.1f}',(i,value),xytext=(0,offset),textcoords='offset points',
                        ha='center',fontsize=8,color=color)
        ax.set_title(title,fontsize=12)
for ax in axes:
    ax.set_xticks(range(7),layers);ax.set_ylim(42,104)
    ax.set_xlabel('Separator state: embedding / after block')
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=9,loc='lower right')
axes[1].axhline(50,ls='--',lw=1,color='gray',alpha=.7)
axes[0].set_ylabel('Test score (%)')
fig.suptitle('Direct: linear answer readout across layers\nOriginal query holdout; independent probe per layer; validation-selected threshold',fontsize=12)
fig.savefig(HERE/'direct_query_accuracy_layers.png',dpi=200)
fig.savefig(HERE/'direct_query_accuracy_layers.pdf')
plt.close(fig)
(HERE/'direct_query_accuracy_layers.json').write_text(json.dumps(dict(
    source=str(SOURCE),position='prompt separator |',
    definition='Existing answer classifier per layer. Query holdout only; not focal-node holdout. '
               'These scores are not truncated-model native accuracy. E is constant separator embedding.',
    results=selected),indent=2),encoding='utf-8')
print(HERE/'direct_query_accuracy_layers.png')
