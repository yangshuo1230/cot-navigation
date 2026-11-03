"""One concrete fixed-DAG candidate-tail read, all arrows from layer four."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

HERE=Path(__file__).resolve().parent
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family']='Microsoft YaHei'
r=next(r for r in json.loads((HERE/'candidate_token_examples.json').read_text()) if r['model']=='fixed_hard_n100')
s=next(s for s in r['states'] if s['record']==3)
assert s['autoreg_generated']==['V72','V76','V80'] and s['autoreg_closed']
fig,ax=plt.subplots(figsize=(13,7.8));fig.subplots_adjust(left=.03,right=.98,top=.85,bottom=.08)
ax.set(xlim=(0,13),ylim=(0,8));ax.axis('off')
fig.suptitle('Fixed 候选生成实例：回溯后读取旧列表的剩余部分',fontsize=20,y=.97)
fig.text(.5,.90,'困难固定图 query 780：V70 → V91；给定历史已经探索 V82 并 POP 回 V70。',ha='center',fontsize=11)
def box(x,y,token,pos,color):
    ax.add_patch(FancyBboxPatch((x-.58,y),1.16,.8,boxstyle='round,pad=.04,rounding_size=.08',edgecolor=color,facecolor='#F4F8FB',lw=1.8,zorder=3))
    ax.text(x,y+.47,token,ha='center',fontsize=15,zorder=4)
    ax.text(x,y+.13,f'位置 {pos}',ha='center',fontsize=9,color='#777777',zorder=4)
    return x,y+.83
old={};new={}
ax.text(.5,7.3,'第1条：V70 | V70 | V82 V72 V76 V80 | PUSH V82 ;',fontsize=13)
ax.text(.5,6.75,'旧候选列表',fontsize=12)
for x,(t,p) in zip([2,4,6,8,10],[('V82',8),('V72',9),('V76',10),('V80',11),('|',12)]):
    old[p]=box(x,5.5,t,p,'#B55B5B' if p==8 else '#0072B2')
ax.text(2,5.05,'已探索',ha='center',fontsize=11,color='#B55B5B')
ax.text(.5,2.9,'第3条前缀：V70 | V70 V82 |　　准备输出剩余候选',fontsize=13)
for x,(t,p) in zip([2,4,6,8,10],[('|',29),('V72',30),('V76',31),('V80',32),('|',33)]):new[p]=box(x,1.6,t,p,'#D99A00' if p<33 else '#009E73')
colors=['#D55E00','#0072B2','#009E73','#8064A2']
for j,e in enumerate(s['events']):
    q=e['q'];key=e['top_keys'][3][0];k=key['position'];mass=key['mass']
    assert k==[9,10,11,12][j]
    ax.add_patch(FancyArrowPatch(new[q],(old[k][0],5.45),arrowstyle='-|>',mutation_scale=15,lw=2,color=colors[j],shrinkA=4,shrinkB=3,zorder=2))
    ax.text(new[q][0]+.68,3.85,f'L4 {mass:.3f}',ha='center',fontsize=11,color=colors[j],bbox=dict(facecolor='white',edgecolor='none',pad=2))
ax.text(.7,.8,'|@29 后输出 V72；V72@30 后输出 V76；V76@31 后输出 V80；V80@32 后输出 |。',fontsize=12)
fig.text(.035,.025,'箭头表示读取方向。前缀历史由参考轨迹给定；新候选字段由模型自行生成。此图不证明它计算了通用的集合差。',fontsize=10,color='#666666')
fig.savefig(HERE/'fixed_candidate_tail.png',dpi=180);fig.savefig(HERE/'fixed_candidate_tail.pdf');plt.close(fig)
print('Saved candidate-tail example')
