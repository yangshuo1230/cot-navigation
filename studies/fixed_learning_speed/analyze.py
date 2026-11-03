"""Summarize observed validation crossing times without interpolating checkpoints."""
from pathlib import Path
import json
import csv
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
runs=json.loads((HERE/'inventory.json').read_text())
PROTOCOLS=['direct','path','dfs_nodes','bfs_nodes','dfs_state','bfs_state','filler']
COLORS=dict(zip(PROTOCOLS,['#222222','#d95f02','#1b9e77','#7570b3','#e7298a','#1f78b4','#a6761d']))

def crossing(curve, threshold, count=1):
    """Return the first point of a qualifying run, and its confirmation point."""
    for i in range(len(curve)-count+1):
        if all(v['balanced_accuracy']>=threshold for v in curve[i:i+count]):
            return dict(start=curve[i]['step'],confirmed=curve[i+count-1]['step'])
    return None

rows=[]
for run in runs:
    if 'config' in run:
        curve=run['curve']
        protocol=run['config']['protocol']['name']
        train=run['config']['train']
        data=run['config']['data']
        assert len({v['step'] for v in curve})==len(curve)
        assert [v['step'] for v in curve]==sorted(v['step'] for v in curve)
        assert all(0<=v['balanced_accuracy']<=1 for v in curve)
        row=dict(path=run['path'],protocol=protocol,n_nodes=data['n_nodes'],
                 graph_seed=data.get('graph_seed'),points=len(curve),
                 val_n=curve[0]['n'],max_steps=train['max_steps'],
                 query_batch=train['effective_query_batch_size'],
                 median_step_seconds=run['median_step_seconds'],
                 peak_ba=max(v['balanced_accuracy'] for v in curve),last_ba=curve[-1]['balanced_accuracy'])
        for t in (.9,.95):
            for count in (1,3):
                result=crossing(curve,t,count)
                row[f't{int(t*100)}_{count}']=result
    else:
        row=dict(path=run['path'],protocol=run['job']['protocol'],arm=run['job']['arm'])
        for view in ('train','monitor'):
            curve=[dict(step=v['step'],balanced_accuracy=v[view]['balanced_accuracy']) for v in run['curve']]
            row[view]=dict(first95=crossing(curve,.95),sustained95=crossing(curve,.95,3),
                           peak_ba=max(v['balanced_accuracy'] for v in curve),last_ba=curve[-1]['balanced_accuracy'])
        row['final']=run['final']
    rows.append(row)

def pick(fragment):
    return [r for r in runs if fragment in r['path']]

summary=dict(total_runs=len(runs),standard_runs=sum('config' in r for r in runs),
             exposure_runs=sum('job' in r for r in runs),runs=rows)
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (HERE/'thresholds.csv').open('w',newline='',encoding='utf-8') as stream:
    names=['path','protocol','val_n','peak_ba','last_ba','first90','sustained90_start','first95','sustained95_start','sustained95_confirmed','median_step_seconds']
    writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
    for r in rows:
        if 't90_1' not in r:continue
        flat={k:r.get(k) for k in names}
        for key,source,part in [('first90','t90_1','start'),('sustained90_start','t90_3','start'),('first95','t95_1','start'),('sustained95_start','t95_3','start'),('sustained95_confirmed','t95_3','confirmed')]:
            flat[key]=(r[source] or {}).get(part)
        writer.writerow(flat)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
panels=[('short3_n100_s0/all_hops/','Simple N100: all-hop query training'),
        ('hard_dag_v1_s0/fixed/n100/stage_b/','Hard N100'),
        ('hard_dag_v1_s0/fixed/n36/stage_b/','Hard N36')]
for ax,(fragment,title) in zip(axes.flat,panels):
    for run in pick(fragment):
        name=run['config']['protocol']['name'];curve=run['curve']
        ax.plot([v['step']/1000 for v in curve],[100*v['balanced_accuracy'] for v in curve],
                color=COLORS[name],label=name,linewidth=1.6)
    ax.set(title=title,xlabel='Task optimizer steps (thousands)',ylabel='Free-generation validation BA (%)',ylim=(45,102))
    ax.axhline(95,color='grey',ls=':',lw=.9);ax.grid(alpha=.18)
ax=axes[1,1]
for arm,color in zip(['R0','R2','R3'],['#1b9e77','#d95f02','#7570b3']):
    for protocol,style in [('direct','-'),('path','--')]:
        group=[r for r in runs if 'job' in r and r['job']['arm']==arm and r['job']['protocol']==protocol]
        steps=[v['step']/1000 for v in group[0]['curve']]
        means=np.mean([[v['monitor']['balanced_accuracy'] for v in r['curve']] for r in group],axis=0)
        ax.plot(steps,100*means,color=color,ls=style,label=f'{protocol} {arm}')
ax.set(title='Exposure-limited N160: 6-run mean',xlabel='Task optimizer steps (thousands)',ylabel='Held-out core monitor BA (%)',ylim=(45,102))
ax.axhline(95,color='grey',ls=':',lw=.9);ax.grid(alpha=.18);ax.legend(ncol=2,fontsize=8,loc='upper right')
axes[0,0].legend(ncol=3,fontsize=8,loc='lower right')
fig.suptitle('Fixed DAG learning speed: existing free-generation validation histories',fontsize=14)
fig.savefig(HERE/'fixed_dag_learning_speed.png',dpi=190)
fig.savefig(HERE/'fixed_dag_learning_speed.pdf')
plt.close(fig)

def fmt(v):return '未达到' if v is None else f"{v['start']//1000}k"

lines=['# Fixed DAG 学习速度检查（2026-10-26）','',
       f'只读检查服务器现存 {len(runs)} 条 Fixed DAG 任务曲线：170 条通用训练器任务，36 条限制暴露率任务。没有启动训练、评估或修改 checkpoint。', '',
       '主要结论：部分设置中 Path/DFS node 更早达到高答案准确率；全状态 CoT 不具有普遍加速优势。困难 N100 的 Direct 和 Path 在首次 5k 验证时均已超过 95%，但 Path 更早连续保持该水平。限制暴露率任务则两者均没有达到高泛化准确率。','',
       '## 指标和限制','',
       '- 使用已有 val 自由生成最终答案 balanced accuracy，格式错误计错；不使用 teacher-forced 答案、轨迹匹配率或训练 loss。',
       '- 同时列首次达标、连续三个验证点达标。连续达标列为这三个点的首点，须到第三点才能确认；不代表此后永不下降。',
       '- 标为 5k 且首次评估在 5k 的结果，只能说不晚于 5k，无法比较 5k 以内谁先学会。没有对保存点之间插值。',
       '- 大部分任务每步 64 个查询；同批同图对照比较任务阶段。共享知识阶段不计入相对速度，CoT 每查询监督 token 和计算量不同。',
       '- 单步耗时来自 step>=1000 的日志中位数，不是端到端墙钟时间；不包括完整验证、checkpoint、编译和排队，存在 GPU 并发差异。',
       '- 验证面板固定但较小：simple N100 为128，hard N100为64；scale_v1 常只有16条且多规模混合/课程改变分布，不能据此精确排序速度。',
       '- 标准模型均为6层、hidden128，但不把不同批数据设置的差异当作图复杂度的单因素效应。主要为单图单初始化，结果不是置信区间。',
       '- 该指标只衡量答案正确；高准确率不代表每步 CoT 合法，也不排除固定图上的查询记忆或其他 shortcut。', '',
       '## 主要配对任务','']
for fragment,title in panels+[('short3_n100_s0/short3/','Simple N100: 训练查询跨度≤3'),('block_n100_s0/within_only','N100: 仅块内查询训练'),('block_n100_s0/cross_allowed','N100: 允许跨块查询训练')]:
    group=[r for r in rows if fragment in r['path']]
    lines += [f'### {title}','','|协议|首次≥90%|连续三次≥90% 起点|首次≥95%|连续三次≥95% 起点|最高验证BA|日志单步中位数|','|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(group,key=lambda r:PROTOCOLS.index(r['protocol'])):
        lines.append(f"|{r['protocol']}|{fmt(r['t90_1'])}|{fmt(r['t90_3'])}|{fmt(r['t95_1'])}|{fmt(r['t95_3'])}|{100*r['peak_ba']:.2f}%|{1000*r['median_step_seconds']:.1f} ms|")
    lines.append('')
lines += ['## 限制暴露率：训练拟合与泛化分开看','',
          'N160，3个图×2个初始化，Direct/Path各R0/R2/R3。仅有20k、40k…125k的评估记录，所有36条运行首次20k训练面板已≥95%，没有更早面板可用于区分谁更快拟合训练查询。', '',
          '|协议|暴露条件|monitor 历史最高BA的6-run均值|monitor 达到90%的运行数|最终独立测试BA均值|','|---|---|---:|---:|---:|']
for protocol in ['direct','path']:
    for arm in ['R0','R2','R3']:
        group=[r for r in rows if r.get('arm')==arm and r['protocol']==protocol]
        assert len(group)==6
        assert all(r['train']['first95']['start']==20000 for r in group)
        lines.append(f"|{protocol}|{arm}|{100*st.mean(r['monitor']['peak_ba'] for r in group):.2f}%|{sum(r['monitor']['peak_ba']>=.9 for r in group)}/6|{100*st.mean(r['final']['balanced_accuracy'] for r in group):.2f}%|")
lines += ['', 'monitor 历史最高值存在选择偏差，仅用来确认未到高分；最终测试没有用于挑选最快 checkpoint。', '',
          '## 早期和其他批次','',
          '- first_round_v2：24个任务中没有任何任务达到95%；只有一个Path运行到过90%。因此该批不能比较“达到95%谁更快”。',
          '- retrain_v3：3张图的Path均到过95%（15k/25k/10k），Direct均没有到95%；DFS state和Filler均未到95%。',
          '- explore_v1：baseline、walk_balanced、walk_none内有Direct/Path/DFS state对照；总体Path较强，但搜索顺序、负例路径和WALK配方之间不可混为单个协议效应。完整逐任务阈值见CSV。',
          '- scale_v1：涵盖多世界混合和课程，曲线总体支持节点版较快、全状态较慢；验证样本小且分布会变，不纳入严格同一图学习速度结论。',
          '- task_only_n32_v1：Direct在三种配方均未到90%；DFS node均在首次5k验证已到90%，但这同时是最终可达到水平的差异，不能只解释成有限速度倍数。', '',
          '## 解释','',
          '现有证据支持“某些轨迹监督能改善固定图上的学习效率和可达到水平”，不支持“输出越完整的CoT越快学会”或“CoT普遍减少计算量”。Hard N100上连续95%区间的起点：Path/BFS node 5k，DFS node/DFS state 10k，Direct15k，BFS state45k。全状态单步成本较高，步数优势未必转化成时间优势。', '',
          '要严格比较首次形成高准确率的时间，应在相同知识checkpoint、查询采样与学习率下，从早期更密集地保存固定验证面板成绩，并记录累计训练时间/查询曝光/token；本次没有为此重训。', '',
          '## 文件','', '- inventory.json：原始配置及压缩曲线。', '- summary.json / thresholds.csv：逐任务阈值和连续达标时间。', '- fixed_dag_learning_speed.png / .pdf：主要曲线。']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote report, thresholds and figure for',len(runs),'runs')
print('validation counts',sorted({r['val_n'] for r in rows if 'val_n' in r}))
