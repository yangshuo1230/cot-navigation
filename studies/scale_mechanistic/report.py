"""Aggregate only completed analysis records into three figures and a report."""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'models').glob('*.json'))]
    rows=[r for r in rows if 'elapsed_seconds' in r]
    def p(x):return '—' if x is None else f'{x:.1%}'
    def num(r,task,metric,source='generated',layer=2):
        rec=r.get('probes',{}).get(task,{})
        key=source+('_sets' if metric in {'pairwise_auc','strict_separation','mean_true_probability','mean_false_probability','false_positive_rate_at_half'} else '')
        data=rec.get(key,[])
        return data[layer].get(metric) if len(data)>layer else None
    flat=[]
    for r in rows:
        e=r['entry']
        out={k:e[k] for k in ['id','kind','scale','protocol','n_nodes','step','validation_score']}
        out.update(test_n=r['rollout']['n'],answer_accuracy=r['rollout']['accuracy'],
                   train_queries=len(r['query_ids']['train']),val_queries=len(r['query_ids']['val']))
        for task in ['successors','visited','visited_after_field','allowed_successors']:
            for metric in ['f1','pairwise_auc','strict_separation','mean_true_probability','mean_false_probability']:
                out[task+'_'+metric]=num(r,task,metric)
            out[task+'_input_auc']=num(r,task,'pairwise_auc',layer=3)
            out[task+'_reference_auc']=num(r,task,'pairwise_auc',source='test')
        conditional=r.get('conditional_probe',{}).get('reference_sets',[])
        out['neighbor_filter_reference_auc']=conditional[2].get('pairwise_auc') if conditional else None
        out['neighbor_filter_input_auc']=conditional[3].get('pairwise_auc') if conditional else None
        out['neighbor_filter_mixed_states']=conditional[2].get('nontrivial_rows') if conditional else None
        flat.append(out)
    if flat:
        with (root/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    byid={r['entry']['id']:r for r in rows}
    scales=['l4','l6','l8','l10','curriculum'];labels=['16','36','64','100','100\nCurriculum']
    protocols=['direct','path','dfs_state','bfs_state','dfs_nodes','bfs_nodes']
    selection=[]
    for kind in ['context','fixed']:
        for scale in scales:
            for family in ['direct','path','dfs','bfs']:
                options=[r['entry'] for r in rows if r['entry']['kind']==kind and r['entry']['scale']==scale and r['entry']['protocol'].split('_')[0]==family]
                if options:
                    options.sort(key=lambda e:(-e['validation_score'],e['id']))
                    selection.append({'kind':kind,'scale':scale,'family':family,'winner':options[0]['id'],
                                      'validation_score':options[0]['validation_score'],
                                      'ranking':[{'id':e['id'],'validation_score':e['validation_score']} for e in options]})
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,kind in enumerate(['context','fixed']):
        for col,(task,metric,title) in enumerate([('visited','pairwise_auc','Visited history: L6 minus input AUC'),('allowed_successors','pairwise_auc','Neighbor filtering (reference): L6 minus input AUC'),('visited','strict_separation','Visited history: strict set separation')]):
            ax=axes[row,col]
            for proto in protocols:
                vals=[]
                for scale in scales:
                    r=byid.get(f'{kind}_{scale}_{proto}')
                    if col==1:
                        conditional=r.get('conditional_probe',{}).get('reference_sets',[]) if r else []
                        a=conditional[2].get(metric) if conditional else None
                        b=conditional[3].get(metric) if conditional else None
                    else:
                        a=num(r,task,metric) if r else None
                        b=num(r,task,metric,layer=3) if r and col<2 else 0
                    vals.append(np.nan if a is None or b is None else a-b)
                if np.isfinite(vals).any():ax.plot(range(5),vals,marker='o',label=proto)
            ax.set_xticks(range(5),labels);ax.axhline(0,color='gray',linewidth=.7)
            ax.set_title(kind+': '+title,fontsize=10);ax.set_xlabel('Graph nodes');ax.grid(alpha=.2)
            if col==0:ax.legend(fontsize=7)
    fig.savefig(root/'probes.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,kind in enumerate(['context','fixed']):
        for col,proto in enumerate(['dfs_state','bfs_state']):
            ax=axes[row,col]
            # Distinct-dead controls can use a different subset; paired effects
            # belong in the table, rather than an unmatched NO-rate curve.
            for condition,style in [('clean','--'),('all_routes','-'),('duplicate_padding','-.'),('all_unvisited','-')]:
                vals=[]
                for scale in scales:
                    rec=byid.get(f'{kind}_{scale}_{proto}',{}).get('blocking',{}).get('excluding_goal',{}).get('conditions',{}).get(condition,{})
                    vals.append(rec.get('counterfactual_NO_rate',np.nan))
                ax.plot(range(5),vals,style,marker='o',label=condition)
            ax.set_xticks(range(5),labels);ax.set_ylim(-.03,1.03);ax.set_title(f'{kind}: {proto}; route blocking excludes goal')
            ax.set_ylabel('NO fraction');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(root/'blocking.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,kind in zip(axes,['context','fixed']):
        for proto in ['path','dfs_state','bfs_state','bfs_nodes']:
            vals=[]
            for scale in scales:
                rec=byid.get(f'{kind}_{scale}_{proto}',{}).get('errors',{}).get('readouts',{}).get('next_token',{}).get('natural',{})
                data=rec.get('strict',[])
                vals.append(data[2]['auroc'] if len(data)>2 and rec.get('strict_pairs',0)>=10 and data[2]['auroc'] is not None else np.nan)
            ax.plot(range(5),vals,marker='o',label=proto)
        ax.axhline(.5,color='gray',linestyle='--');ax.set_xticks(range(5),labels);ax.set_ylim(0,1.03)
        ax.set_title(kind+': natural-error +1 AUC; >=10 legal repaired pairs');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(root/'errors.png',dpi=160);plt.close(fig)
    lines=['# Scale 模型：线性探针与因果干预','',f'已完成 {len(rows)}/60 个模型。每个 run 使用按验证选出的 best.pt；未按 test 排序或选模型。',
           '', '导航权重冻结；仅物理 GPU0 拟合仿射探针。无导航训练、W&B run、探针 checkpoint 或 activation 缓存。',
           '', '主规模 N16/36/64/100，另列 curriculum N100；context/fixed 各六种协议。本批没有 Filler 模型。',
           '训练/验证/测试查询互斥；上限256/96/128，固定小图使用实际可用数量。新图预生成池与固定WORLD保持原数据划分。',
           '参考前缀取实际预存随机轨迹 variant 0；固定图添加正确 WORLD token；使用训练的4096上下文、STOP/答案策略。',
           '本批每查询仅4条离线随机轨迹，区别于旧探索的在线随机顺序；跨批差异不能只归因于图规模。',
           '节点版 visited 是已输出的展开历史；全状态版 visited 是此前 PUSH 推导的发现集合。两者不作同语义比较。',
            '输入对照只使用当前节点/当前token embedding、位置、目标出现特征；不是完整输入。所有集合分数为独立 sigmoid，非互斥token概率。',
            '图变大也会使整集合严格恢复更困难；不能把严格分离率下降直接等同为搜索能力下降。探针样本预算固定，阴性结果也可能受读出训练量影响。',
           '', '## 验证选择的协议变体','',
           '每个run均使用best；DFS/BFS再按验证成绩确定主变体，同时完整保留另一变体。visited干预仅适用于全状态版本，节点版获胜时对应全状态结果属于补充分析。',
           '新批次仅有seed0，没有额外训练seed1/2。验证并列时按ID确定展示顺序，完整排序记录在findings.json。',
           '', '|任务|规模|DFS主变体 / 验证分数|BFS主变体 / 验证分数|',
           '|---|---|---|---|']
    for kind in ['context','fixed']:
        for scale in scales:
            cells=[]
            for family in ['dfs','bfs']:
                s=next((s for s in selection if (s['kind'],s['scale'],s['family'])==(kind,scale,family)),None)
                cells.append((s['winner'].replace(kind+'_'+scale+'_','')+' / '+p(s['validation_score'])) if s else '—')
            lines.append('|'+ '|'.join([kind,scale,*cells])+'|')
    lines+=['', '## 探针：自由生成前缀、L6','',
           'visited 在列表前读出历史；visited-after 在列表后读出外显集合；allowed 在列表后读出图后继减去外显visited。',
           'AUC为每个非空非全集状态内正确/错误节点配对排序的均值；严格分离要求每个正确节点都高于每个错误节点。',
           '', '|模型|n|答案准确率|后继F1|visited F1|visited AUC / 输入|visited严格分离|外显VIS AUC|合法候选 AUC / 输入|',
           '|---|---:|---:|---:|---:|---|---:|---:|---|']
    for r in rows:
        lines.append('|'+ '|'.join([r['entry']['id'],str(r['rollout']['n']),p(r['rollout']['accuracy']),p(num(r,'successors','f1')),p(num(r,'visited','f1')),p(num(r,'visited','pairwise_auc'))+' / '+p(num(r,'visited','pairwise_auc',layer=3)),p(num(r,'visited','strict_separation')),p(num(r,'visited_after_field','pairwise_auc')),p(num(r,'allowed_successors','pairwise_auc'))+' / '+p(num(r,'allowed_successors','pairwise_auc',layer=3))])+'|')
    lines+=['','![probes](probes.png)','','## 严格后继内过滤探针','',
            '前表合法候选AUC在全图节点上计算，可能仅靠识别邻接关系获得高分，不能直接证明使用visited。',
            '此处另拟合仿射头，训练和评估均只保留当前节点的真实出邻居，标签为未访问/可选；所有非邻居完全排除。参考测试使用完整probe-test查询。',
            '状态内AUC仅在同时存在已访问和未访问后继的混合状态上定义。自然状态来自保存的干预案例，部分按答案变化选择，仅作案例诊断，不代表总体频率。',
            '', '|模型|参考状态n / 混合n|参考L6 AUC / 输入|参考L6平衡准确率|保存自然混合n|自然L6 AUC / 输入|',
            '|---|---:|---|---:|---:|---|']
    for r in rows:
        cp=r.get('conditional_probe',{});rs=cp.get('reference_sets',[])
        if not rs:continue
        ns=cp.get('saved_natural_selected_cases',{}).get('sets',[])
        lines.append('|'+ '|'.join([r['entry']['id'],f"{rs[2]['n_rows']} / {rs[2]['nontrivial_rows']}",p(rs[2].get('pairwise_auc'))+' / '+p(rs[3].get('pairwise_auc')),p(cp['reference'][2]['balanced_accuracy']),str(ns[2]['nontrivial_rows']) if ns else '0',p(ns[2].get('pairwise_auc'))+' / '+p(ns[3].get('pairwise_auc')) if ns else '—'])+'|')
    lines+=['','## 全路线阻断','',
            '原图标签仍为YES；NO率表示对篡改后搜索状态的响应，不是任务准确率。修改可能与此前历史矛盾。',
            '每组最多40个图算法及完整搜索器共同认证的状态；含/不含目标两组可能不是同一查询或同一位置。死节点对照只在可构造的配对子集计算。',
            '候选/转移正确率的分母仅包括能解析的对应记录，须结合解析样本数；后续PUSH比例不依赖记录能否解析。',
            '', '|模型|阻断|n|原样NO|阻断NO|全图VIS NO|死节点对照n|阻断−死节点YES差 [95%区间]|重复PUSH插入节点|首候选正确(n)|首转移正确(n)|',
            '|---|---|---:|---:|---:|---:|---:|---|---:|---|---|']
    findings=[]
    for r in rows:
        for variant in ['including_goal','excluding_goal']:
            b=r.get('blocking',{}).get(variant,{})
            if not b.get('n_cases'):continue
            c=b['conditions'];a=c['all_routes'];contrast=b.get('paired_contrasts',{}).get('all_routes_minus_dead_distinct_matched',{})
            diff=contrast.get('YES_difference');ci=contrast.get('ci95')
            cs=p(diff)+(f' [{p(ci[0])}, {p(ci[1])}]' if ci else '')
            lines.append('|'+ '|'.join([r['entry']['id'],variant,str(b['n_cases']),p(c['clean']['counterfactual_NO_rate']),p(a['counterfactual_NO_rate']),p(c.get('all_unvisited',{}).get('counterfactual_NO_rate')),str(contrast.get('n',0)),cs,p(a['pushed_inserted_rate']),p(a['first_candidates_correct_rate'])+f" ({a['first_record_parseable_n']})",p(a['first_transition_correct_rate'])+f" ({a['transition_checked_n']})"])+'|')
            if ci and (ci[0]>0 or ci[1]<0):findings.append({'model':r['entry']['id'],'variant':variant,'contrast':contrast})
    lines+=['','![blocking](blocking.png)','','## CoT 错误检测','',
            '用参考轨迹的图内节点错误/修复配对及最多32条训练、验证自然生成轨迹训练错误探针；训练生成上限512 token，测试使用完整上下文。',
            'STOP为合法中断，不算错误。+1两臂追加同一个token，并另列修复后该token仍合法的严格子集。高分不证明原模型自主发现/修复错误。',
            'DFS节点版未运行错误探针：历史节点oracle仅支持BFS层级，不能误用作DFS判定。',
            '', '|模型|自然错误n|错误token AUC / 输入|+1 AUC / 输入|+1严格n|严格AUC / 输入|',
            '|---|---:|---|---|---:|---|']
    for r in rows:
        es=r.get('errors',{}).get('readouts',{})
        if not es:continue
        w=es.get('wrong_token',{}).get('natural',{});n=es.get('next_token',{}).get('natural',{})
        def auc(rec,key,layer):
            x=rec.get(key,[]);return x[layer].get('auroc') if len(x)>layer else None
        lines.append('|'+ '|'.join([r['entry']['id'],str(w.get('n_pairs',0)),p(auc(w,'all',2))+' / '+p(auc(w,'all',3)),p(auc(n,'all',2))+' / '+p(auc(n,'all',3)),str(n.get('strict_pairs',0)),p(auc(n,'strict',2))+' / '+p(auc(n,'strict',3))])+'|')
    lines+=['','![errors](errors.png)','','## 其他干预','',
            'Direct替换QUERY目标，比较改变标签与保持标签的同层节点；Path/节点搜索替换中间节点，比较到目标可达性相反与相同的替代节点。后者是token敏感性测试，不保证修改后前缀合法。',
            '', '|模型|n|原样准确率|目标翻转/相反路线准确率|同标签/相同路线准确率|',
            '|---|---:|---:|---:|---:|']
    for r in rows:
        z=r.get('patching',{});c=z.get('conditions',{})
        if not c:continue
        a='target_flip' if r['entry']['protocol']=='direct' else 'node_opposite_route';b='target_same_label' if a=='target_flip' else 'node_same_route'
        lines.append('|'+ '|'.join([r['entry']['id'],str(z['n_cases']),p(c['clean']['accuracy']),p(c.get(a,{}).get('accuracy')),p(c.get(b,{}).get('accuracy'))])+'|')
    lines+=['','## 解释边界','',
            '各规模训练步数、混合数据和WORLD数量不同；课程不等预算。仅一个模型初始化/每形状一个固定图，不把查询bootstrap区间当跨模型重复。',
            'bootstrap区间为探索性逐项区间，未作多重比较校正。输入对照只排除所列简单线索，不排除所有图/查询身份捷径。',
            'best由较小验证集选择，尤其N16固定图验证样本少；本表与此前last-checkpoint报告不能直接作训练前后比较。',
            '读出失败只针对所选位置、线性头和样本量；读出成功不证明因果使用。外显状态干预也不能单独区分历史修复与忽略字段。',
            '完整分层指标、原样/修改轨迹实例、查询ID、生成中断和checkpoint哈希保存在models/*.json；summary.csv用于后续汇总。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'findings.json').write_text(json.dumps({'completed':len(rows),'validation_variant_selection':selection,'nonzero_paired_contrasts':findings},indent=2),encoding='utf-8')
    print('reported',len(rows),'models')


if __name__=='__main__':main(Path(sys.argv[1]))
