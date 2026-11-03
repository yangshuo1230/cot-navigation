"""Key figures and a Chinese report from compact, positive-only replay metrics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
TASKS=['n32','n64','main','deep','hard']
NAMES=['N32','N64','普通N100','窄深N100','困难DAG']
ARMS=['original_clean','matched_clean','correction']
LABELS=['原始baseline','匹配对照','纠错组']
COLORS=['#999999','#4477aa','#ee8844']
OUTCOMES=['correct_clean_valid','correct_repaired_valid','correct_execution_fields_invalid','correct_incomplete_or_illegal']
OLABELS=['无错完整合法','显式修复后完整合法','动作可执行但字段不合法','其余答对（缺少合法完整执行）']
OCOLORS=['#55a868','#eeaa55','#6a9ec1','#b7b7b7']
CNAMES=['图外候选','图内非边候选','已访问候选','候选重复','候选遗漏','visited错误','栈错误','动作/其他格式']


def causes(a):
    v=np.zeros(8,dtype=int)
    mapping={'candidate_out_of_graph':0,'candidate_nonedge':1,'candidate_visited':2,
             'candidate_duplicate':3,'candidate_omission':4}
    for k,n in a['first_outcome_counts'].items():
        if k.startswith('no_error'):continue
        j=mapping.get(k,5 if k.startswith('visited_') else 6 if k.startswith('frontier_') else 7)
        v[j]+=n
    assert v.sum()==a['first_error_queries']
    return v


def pct(n,d=1024):return n/d*100
def cp(n,d=1024):return f'{n}（{pct(n,d):.2f}%）'
def signed(n,d=1024):return f'{pct(n,d):+.2f}'


def figures(d):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Noto Sans CJK SC','DejaVu Sans'],
                        'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    out=P/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(15.6,5.2),gridspec_kw={'width_ratios':[1.25,1]})
    x=np.arange(5)
    for ai,arm in enumerate(['matched_clean','correction']):
        bottom=np.zeros(5)
        for key,label,color in zip(OUTCOMES,OLABELS,OCOLORS):
            vals=np.array([pct(d['tasks'][t]['groups'][arm]['pooled']['outcomes'][key]) for t in TASKS])
            axes[0].bar(x+(ai-.5)*.35,vals,.33,bottom=bottom,color=color,label=label if ai==0 else None)
            bottom+=vals
    axes[0].set(xticks=x,xticklabels=NAMES,ylim=(0,103),ylabel='占全部正样本比例（%）',
                title='正确答案的轨迹组成：每对左为匹配、右为纠错')
    axes[0].legend(fontsize=8,frameon=False,loc='lower left')
    for ai,(key,label,color) in enumerate([('groups','正常生成','#ee8844'),('blocked','禁止 MISTAKE','#4477aa')]):
        values=[];std=[]
        for t in TASKS:
            item=d['tasks'][t]
            vals=[pct(s['native_correct'] if key=='groups' else s['blocked_correct'],256) for s in item['mistake_ablation']['seeds']]
            values.append(np.mean(vals));std.append(np.std(vals,ddof=1))
        axes[1].bar(x+(ai-.5)*.35,values,.33,yerr=std,capsize=3,color=color,label=label,alpha=.8)
    axes[1].set(xticks=x,xticklabels=NAMES,ylim=(40,103),ylabel='正样本答案准确率（%）',
                title='因果干预：同一纠错模型禁用显式纠错动作')
    axes[1].legend(frameon=False,fontsize=9,loc='lower left')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('正样本收益来源：无错生成、有效修复与动作干预',fontsize=14)
    fig.text(.5,.01,'每任务4个seed × 相同256个正查询；左图是结果组成，不是因果贡献分摊；右图误差棒为seed标准差。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    for ext in ['png','svg']:fig.savefig(out/f'benefit_sources.{ext}',dpi=170)
    plt.close(fig)

    before=np.array([causes(d['tasks'][t]['groups']['matched_clean']['pooled']) for t in TASKS])
    after=np.array([causes(d['tasks'][t]['groups']['correction']['pooled']) for t in TASKS])
    delta=(after-before)/1024*100
    fig,ax=plt.subplots(figsize=(14.5,4.4))
    limit=max(5,float(np.abs(delta).max()))
    im=ax.imshow(delta,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    for i in range(5):
        for j in range(8):
            ax.text(j,i,f'{before[i,j]} → {after[i,j]}\n{delta[i,j]:+.2f}pp',ha='center',va='center',fontsize=9,
                    color='white' if abs(delta[i,j])>limit*.65 else 'black')
    ax.set(xticks=range(8),xticklabels=CNAMES,yticks=range(5),yticklabels=NAMES,
           title='首个错误 token 的类型：匹配对照 → 纠错（每行分母均为1024）')
    fig.colorbar(im,ax=ax,label='纠错−匹配：占全部正样本的百分点变化')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'first_error_causes.{ext}',dpi=170)
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,7.4));flat=axes.ravel()
    peak=max(rate for t in TASKS for a in ARMS for rate,risk in zip(
        d['tasks'][t]['groups'][a]['pooled']['hazard']['rates'],
        d['tasks'][t]['groups'][a]['pooled']['hazard']['risk_states']) if risk>=30)*100
    for ti,t in enumerate(TASKS):
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            hazard=d['tasks'][t]['groups'][arm]['pooled']['hazard']
            v=np.array(hazard['rates'])*100
            v[np.asarray(hazard['risk_states'])<30]=np.nan
            flat[ti].plot(range(7),v,'o-',ms=4,lw=1.5,color=color,label=label)
        flat[ti].set(title=NAMES[ti],xticks=range(7),xticklabels=['1','2','3–4','5–8','9–16','17–32','33+'],
                     ylim=(0,max(15,peak*1.1)),ylabel='每个仍无错搜索状态的首错率（%）',xlabel='已执行真实动作后的搜索步位置')
        flat[ti].grid(alpha=.2)
    flat[5].axis('off');handles,labels=flat[0].get_legend_handles_labels()
    flat[5].legend(handles,labels,frameon=False,loc='upper left')
    flat[5].text(.02,.55,'分母：该区间实际到达且此前未出错的状态数。\n风险状态少于30的点不画，原始计数保留于报告。\n重试不重复计数；合法截断前缀不强行记为错误。\n这是状态曝光后的描述性发生率，非因果效应。',va='top',fontsize=10)
    fig.suptitle('首错位置：控制轨迹长度与幸存状态曝光',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ['png','svg']:fig.savefig(out/f'first_error_positions.{ext}',dpi=170)
    plt.close(fig)


def report(d):
    lines=['# 正样本导航优势的来源：首错、修复与 MISTAKE 干预','',
      '## 本轮最重要的结果','',
      '1. 显式改错的作用随任务变化。N32正常918/1024，禁止MISTAKE后898，几乎失去相对匹配对照897的平均优势；但仍高于原始baseline875。因此这是相对匹配对照的依赖，不是所有纠错训练收益都由显式动作产生。',
      '2. N64、普通N100禁用后仍保留大部分优势：N64正常926、禁止907、匹配851；普通N100正常856、禁止853、匹配806。困难DAG禁止后反而从731提高到750，不能把主动改错视为普遍增益。',
      '3. 普通N100首错由608降到523，无错完整合法答对由415升到501。首错中图外候选269→229、遗漏105→69是主要数量变化，已访问候选87→88没有改善。首错步中位数9→10，token中位数128→145，后移幅度有限，需结合曝光校正发生率。',
      '4. 主动识别不等于稳定修复。普通N100首错被检测182次，正确修复36次，最终整条仍合法的只有5次；修复后又失败的31次全部再出现候选错误。N32修复后21/36能保持完整合法，长程场景明显较低。',
      '5. 并非所有新增答对都提供了合法搜索见证。普通N100相对匹配对照新增154次答对、同时损失104次；新增中只有62次无错合法、2次修复后合法，其余90次没有完整合法状态轨迹。不能把全部正例答案增益直接解释为更强规划。',
      '6. 窄深图的错误会转移：候选首错减少，但visited首错从30增至111；无token错误却提前结束的前缀从1增至29。不能把单纯“首错率降低”全部视为任务完成能力提高。',
      '7. 全部15,360次正常正样本评估中，没有观察到完整合法CoT之后单独答错最终标签；错误答案都伴随轨迹不完整或不合法。这不证明CoT错误是答案错误的唯一原因，但不支持答案读出头单独失败是本轮主要现象。','',
      '## 结论的判断顺序','',
      '先看同权重禁止MISTAKE是否降低正样本准确率，再看完整合法修复实际占多少，最后检查无错生成、首错类型和位置。正确答案不被不合法轨迹否决，但它也不能单独作为规划或完整搜索成功的证据。结果组成的加减只作记账，不等同训练效应的因果中介分摊。','',
      '## 范围与复现','',
      '12个last200k checkpoint，原始baseline、匹配干净对照、纠错各4seed；五套图。生成仅输入边列表与查询，冻结权重，无训练/探针/W&B。每任务分析相同256个可达正查询，4seed合计1024次模型—查询评估，不是1024个独立图。','',
      '为保持历史batch16的浮点条件，仍按历史顺序生成全部512题，生成后丢弃负例轨迹；只保存正样本紧凑指标和少数片段。所有组预算均为原生4K；困难DAG只使用已有32K正确参考缓存提取图/查询，不向模型输入参考CoT，不按参考长度设生成预算。','',
      f"重放检查：{d['reproduction']['native_reports']}份正常生成、{d['reproduction']['masked_reports']}份禁用MISTAKE结果；历史512题层面的预测或中断标记差异共{d['reproduction']['historical_native_mismatches']}条。15,360条正样本预测中答案变化{d['reproduction']['positive_pred_changes']}条（{d['reproduction']['positive_pred_changes']/15360*100:.2f}%）。原测试没有保存全量轨迹，不能声称轨迹逐token复现。统计分解使用重放本身的答案与轨迹，不把历史答案拼到新轨迹。一个16题校验批次通过identity proxy检查，排除该包装器本身改动解码。",'',
      '历史与重放少量差异不能唯一归因于某个内核。代码、权重、图、精度、批量及输入顺序已对齐，但新进程的BF16编译推理仍可能敏感；下面明确列出正样本准确率变化。','',
      '|任务|组别|历史答对/1024|本次答对/1024|本次完整合法/1024|','|---|---|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        for a,label in zip(ARMS,LABELS):
            g=d['tasks'][t]['groups'][a];s=g['pooled']
            lines.append(f"|{name}|{label}|{g['historical_correct']}|{s['correct']}|{s['trace_valid']}|")
    lines += ['', '## 1. 禁用显式 MISTAKE 的因果干预','',
      '保留纠错模型全部权重、输入、贪心解码及预算，仅将MISTAKE的输出logit设为负无穷。允许模型自己选择替代动作，不提供oracle修复。正常与干预生成在同一推理进程成对执行，避免把不同编译初始化混在一对里。',
      '“帮助”指正常答对、禁止后答错；“伤害”指正常答错、禁止后答对。这衡量当前模型使用该显式动作的作用，不等同删除纠错训练数据的效应。','',
      '|任务|正常答对|禁止后答对|帮助/伤害次数|正常−禁止（pp）|正常/禁止的完整合法数|正常/禁止预算中断数|','|---|---:|---:|---:|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        a=d['tasks'][t]['mistake_ablation'];s=a['pooled'];n=d['tasks'][t]['groups']['correction']['pooled'];b=a['blocked']
        lines.append(f"|{name}|{s['native_correct']}|{s['blocked_correct']}|{s['helped']} / {s['harmed']}|{signed(s['native_correct']-s['blocked_correct'])}|{n['trace_valid']} / {b['trace_valid']}|{s['native_interrupted']} / {s['blocked_interrupted']}|")
    lines += ['', '### 干预按seed分解','', '|任务|seed|正常答对/256|禁止后答对/256|帮助|伤害|净变化（正常−禁止）|','|---|---:|---:|---:|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        for s in d['tasks'][t]['mistake_ablation']['seeds']:
            lines.append(f"|{name}|{s['seed']}|{s['native_correct']}|{s['blocked_correct']}|{s['helped']}|{s['harmed']}|{s['native_correct']-s['blocked_correct']:+d}|")
    lines += ['', '### 未触发MISTAKE查询的数值控制','',
      '如果正常轨迹完全没产生MISTAKE，禁用该token本不应改变这条轨迹；但同batch其他查询的长度变化可改变KV裁剪/finished compaction的浮点路径。显式检查这些非触发查询的答案与轨迹漂移。','',
      '|任务|非触发查询数|非触发答案变化|非触发轨迹hash变化|','|---|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        s=d['tasks'][t]['mistake_ablation']['pooled'];lines.append(f"|{name}|{s['nontrigger_queries']}|{s['nontrigger_pred_drift']}|{s['nontrigger_trace_drift']}|")
    lines += ['', '## 2. 正确答案的轨迹组成','',
      '四个互斥的答对类别：①无错误且完整合法；②包含错误、全部被显式取消并修复后完整合法；③动作骨架在真实图上完整可执行，但状态字段或候选优先级不合法；④其他答对，但没有合法完整执行轨迹。④不能直接叫猜测，隐藏推理仍可能存在；它只是当前输出未提供完整合法见证。','',
      '|任务|组别|无错合法答对|修复后合法答对|动作可执行/字段不合法答对|其余答对|答错|','|---|---|---:|---:|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        for a,label in zip(ARMS,LABELS):
            o=d['tasks'][t]['groups'][a]['pooled']['outcomes']
            values=[str(o[k]) for k in OUTCOMES]+[str(o['wrong_answer'])]
            lines.append(f"|{name}|{label}|{'|'.join(values)}|")
    lines += ['', '### 相同查询上新增答对与新损失','',
      '这些是逐seed、逐查询配对的变化，能区分新增正确和同时损失的查询。新增答对的组成是观察证据，不能直接证明某个token或隐藏状态是原因。','',
      '|任务|对照|新增答对|新损失|净收益|新增中无错合法|新增中修复合法|新增中曾有真实修复|','|---|---|---:|---:|---:|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        for a,label in zip(ARMS[:2],LABELS[:2]):
            c=d['tasks'][t]['contrasts'][a]
            lines.append(f"|{name}|{label}|{c['gains']}|{c['losses']}|{c['delta_correct']:+d}|{c['gain_outcomes']['correct_clean_valid']}|{c['gain_outcomes']['correct_repaired_valid']}|{c['gains_any_repair']}|")
    lines += ['', '## 3. 首错的数量与类型','',
      '首错按token顺序定位，允许visited/candidate合法重排，不要求照抄唯一参考。重复节点在重复token处报错；集合遗漏在提前闭合的分隔符处报错。栈、visited、候选和动作全部相对真实搜索状态审计。MISTAKE可取消错误记录而不推进真实状态；因此首个原始错误与首个未取消错误分开保存。发生非法真实动作后，后面的状态列表不再独立判错。',
      '下表按最早非法token互斥分类。一个候选字段可能同时有额外节点与遗漏，但该查询只按最先出现的错误归入一类。修好更早的错误会暴露更晚的错误，因此首错类别之间存在竞争，不把每类差值当作独立因果贡献。计数单位均为1024次正样本模型评估。frontier_wrong_order表示输出了栈中存在、但不应在此位置的节点，含重复/顺序错位；不是只指一个完整列表的纯排列错误。','']
    for t,name in zip(TASKS,NAMES):
        lines += [f'### {name}','', '|首错类别|原始baseline：数量（占全部正例）|匹配：数量（占全部正例）|纠错：数量（占全部正例）|纠错−匹配数量|','|---|---:|---:|---:|---:|']
        aa=[d['tasks'][t]['groups'][a]['pooled'] for a in ARMS];cc=[causes(a) for a in aa]
        for j,label in enumerate(CNAMES):
            lines.append(f"|{label}|{cp(cc[0][j])}|{cp(cc[1][j])}|{cp(cc[2][j])}|{int(cc[2][j]-cc[1][j]):+d}|")
        for key,label in [('first_error_queries','有首错合计'),('first_before_goal','抵达目标前首错'),('first_at_goal','当前节点已是目标时首错')]:
            values=[cp(a[key]) for a in aa];lines.append(f"|{label}|{'|'.join(values)}|{aa[2][key]-aa[1][key]:+d}|")
        values=[cp(1024-a['first_error_queries']) for a in aa];lines.append(f"|未观察到CoT token错误（含未完成前缀）|{'|'.join(values)}|—|")
    lines += ['', '### 首错候选字段中“有用后继”的缺失','',
      '诊断时独立计算哪些合法候选仍可到达目标；不向模型提供。以下是首错所在候选字段的完整内容特征，彼此可重叠，不是互斥首错类型；字段尚未闭合时不判定遗漏。缺失死路也违反协议，但不能与缺失所有通往目标的候选混为一谈。','',
      '|任务|组别|缺失有用后继|有用后继全部缺失|只缺死路|多出非邻居|重复项|','|---|---|---:|---:|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        for a,label in zip(ARMS,LABELS):
            f=d['tasks'][t]['groups'][a]['pooled']['candidate_first_record_features']
            values=[str(f.get(k,0)) for k in ['missing_useful','all_useful_missing','missing_only_deadends','extra_nonedge','duplicate']]
            lines.append(f"|{name}|{label}|{'|'.join(values)}|")
    lines += ['', '## 4. 首错位置与长度控制','',
      'token位置从CoT第一个token开始，排除输入图/查询。搜索步位置按实际执行的动作推进；MISTAKE重试不推进。中位数只针对发生首错的轨迹，存在幸存者选择，不能单凭中位数更大就判断更强。配合每个仍无错状态的首错率（hazard）比较；预算停止但前缀合法者右删失，不强行放一个语义错误。','',
      '|任务|组别|首错token中位数 [Q1,Q3]|首错搜索步中位数 [Q1,Q3]|目标前首错步中位数|首错时context占用中位数|','|---|---|---:|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        for a,label in zip(ARMS,LABELS):
            p=d['tasks'][t]['groups'][a]['pooled']['positions'];v=[]
            for k in ['token','search_step']:
                q=p[k];v.append(f"{q['median']:g} [{q['q25']:g},{q['q75']:g}]")
            lines.append(f"|{name}|{label}|{'|'.join(v)}|{p['search_step_before_goal']['median']:g}|{p['context_fraction']['median']*100:.1f}%|")
    for t,name in zip(TASKS,NAMES):
        lines += ['',f'### {name}的状态曝光校正首错率','', '|搜索步区间|原始baseline：错误/风险状态|匹配：错误/风险状态|纠错：错误/风险状态|','|---|---:|---:|---:|']
        for i,binname in enumerate(['1','2','3–4','5–8','9–16','17–32','33+']):
            vals=[]
            for a in ARMS:
                h=d['tasks'][t]['groups'][a]['pooled']['hazard'];vals.append(f"{h['first_errors'][i]}/{h['risk_states'][i]}（{h['rates'][i]*100:.2f}%）")
            lines.append(f"|{binname}|{'|'.join(vals)}|")
    lines += ['', '### 前缀没有token错误，但搜索尚未完成','',
      '预算截断和过早终止单列，不能与完整无错轨迹合并。过早终止指模型在尚无合法FOUND/EXHAUST时结束或回答，属于完成性失败，只是未观察到更早的字段/动作token错误。','',
      '|任务|组别|无更早token错的预算截断|无更早token错的过早终止|完整合法CoT之后标签答错|','|---|---|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        for a,label in zip(ARMS,LABELS):
            s=d['tasks'][t]['groups'][a]['pooled'];c=s['first_outcome_counts']
            lines.append(f"|{name}|{label}|{c.get('no_error_prefix_budget_cutoff',0)}|{c.get('no_error_prefix_early_end',0)}|{s['wrong_answer_after_valid_trace']}|")
    lines += ['', '## 5. 主动检错与修复是否兑现','',
      '检测首错=首错所在记录主动用MISTAKE取消；首错修复=在不推进真实状态的条件下，后续首条非取消记录的字段与动作全部正确。仅输出标记、只修对一个字段、在非法动作后的尾部反复输出MISTAKE，都不算完整修复。','',
      '|任务|有原始MISTAKE查询|可审计状态下有真检错|首错被检测|首错被正确修复|修复首错后答对|修复首错后整条合法|可审计误报记录|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for t,name in zip(TASKS,NAMES):
        a=d['tasks'][t]['groups']['correction']['pooled']
        vals=[str(a[k]) for k in ['any_raw_mistake','any_true_mark','first_detected','first_repaired','first_repaired_correct','first_repaired_full_valid','false_mark_records']]
        lines.append(f"|{name}|{'|'.join(vals)}|")
    lines += ['', '### 修复首错后再次失败的原因','',
      '这里统计已经正确修复首错、但最终仍没有完整合法轨迹的查询。按之后首次未被取消的token错误分类；若没有这种token错误但未完成/答案错，单列。','',
      '|任务|修复后再次失败数|后续首次未修复错误的数量分解|','|---|---:|---|']
    for t,name in zip(TASKS,NAMES):
        a=d['tasks'][t]['groups']['correction']['pooled'];counts=a['postrepair_failure_first_types']
        lines.append(f"|{name}|{a['first_repaired']-a['first_repaired_full_valid']}|{'; '.join(k+'='+str(v) for k,v in counts.items())}|")
    examples_path=P/'counterexamples.json'
    if examples_path.exists():
        examples=json.loads(examples_path.read_text())
        helped=next((e for e in examples if e['effect']=='helped' and e['native']['outcome']=='correct_repaired_valid'),None)
        harmed=max((e for e in examples if e['effect']=='harmed'),key=lambda e:e['native']['raw_mistake_tokens'],default=None)
        lines += ['', '### 具体可复查案例','']
        if helped:
            a,b=helped['native'],helped['blocked']
            lines.append(f"- 真正有帮助：{helped['task']}、seed{helped['seed']}、query {helped['query_id']}。首错为{a['first']['kind']}，出现在CoT token {a['first']['token']}；正常生成{a['raw_mistake_tokens']}次MISTAKE后修复且整条合法，答YES；禁止后答NO。")
        if harmed:
            a,b=harmed['native'],harmed['blocked']
            lines.append(f"- 反例：{harmed['task']}、seed{harmed['seed']}、query {harmed['query_id']}。正常生成有{a['raw_mistake_tokens']}次MISTAKE，长度{a['trace_length']}，中断={a['interrupted']}，答{a['pred']}；禁止后长度{b['trace_length']}，中断={b['interrupted']}，答{b['pred']}。禁止后的类别为{b['outcome']}，因此这里只证明答案变好，不声称搜索轨迹变合法。")
    lines += ['', '## 6. 对机制结论的限制','',
      '- 禁止MISTAKE检验当前模型对显式动作的依赖；它不能单独区分训练中错误暴露、重采样、额外token监督或内部表示改变的作用。matched_clean控制词表/初始化与查询曝光，但纠错训练实际token量和每条长度权重可以不同。',
      '- 不把MISTAKE出现次数当检错正确率；非法动作后真实状态已失去可审计性，尾部标记另计。一次修复也可能随后再错，故完整合法修复单列。',
      '- 正例回答YES即可计为答对。即使没有完整合法轨迹，模型也可能有隐藏推理；本分析既不把这类回答算作合法规划，也不直接称为猜测。仅正例准确率不能证明整体判别能力增强；原完整结果中一些正例收益伴随负例准确率下降，需保留这一背景。',
      '- 4个seed仍不足以对所有规模作普遍稳健性主张。首错统计是竞争风险下的描述性分解；“某类首错减少”不等于该类错误对答案的因果贡献。','',
      '## 代码与复用','',
      '公共增量：`src/cot_navigation/protocols/correction_replay.py`，支持DFS/BFS、正负标签、token首错与MISTAKE取消。独立小图测试覆盖全部4节点DAG及端点、DFS/BFS、错误答案、取消重写和前缀删失。相关测试17 passed。现有trace_audit契约不改。',
      '研究入口：`studies/correction_reproduction/breakdown/`，复用公共ContextStore、ScalePreparedRows、build_model、CompiledInference、generate、checkpoint、artifacts及idle-GPU launcher；没有复制第二套推理/调度器。旧结果按输入来源和checkpoint哈希导入，运行不依赖旧项目或research档案。','',
      '```bash\ncd ~/cot-navigation-unified\n.venv/bin/python -m cot_navigation study correction_reproduction/breakdown\n# 已授权分析的实际执行入口，默认只显示计划\n.venv/bin/python -m cot_navigation study correction_reproduction/breakdown -- --execute\n.venv/bin/python -m cot_navigation study correction_reproduction/breakdown_summary\n```','',
      '没有启动模型训练、W&B或保存新checkpoint；不保存激活和全量轨迹。12份紧凑模型结果、一个汇总、三张关键图及少量例子足以复查。','',
      '## 附表：全部首错细类型与双分母占比','',
      '每格为数量（占全部1024次正例评估%；占该组发生首错的评估%）。第二个分母只用于看错误构成，不能代替首错发生率。',
      'candidate_nonedge指图内、未访问的非邻居；已访问节点优先归candidate_visited。frontier_wrong_order含栈内重复/错位，frontier_wrong_node指该位置输出了不属于真实栈的节点。']
    for t,name in zip(TASKS,NAMES):
        aa=[d['tasks'][t]['groups'][a]['pooled'] for a in ARMS]
        keys=sorted({k for s in aa for k in s['first_outcome_counts'] if not k.startswith('no_error')})
        lines += ['',f'### {name}细类型','', '|类型|原始baseline|匹配对照|纠错组|','|---|---:|---:|---:|']
        for k in keys:
            values=[]
            for s in aa:
                n=s['first_outcome_counts'].get(k,0);errors=s['first_error_queries']
                values.append(f'{n}（{pct(n):.2f}%；{pct(n,errors):.2f}%）')
            lines.append(f"|{k}|{'|'.join(values)}|")
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    d=json.loads((P/'summary.json').read_text());assert d['finished']
    figures(d);report(d)


if __name__=='__main__':main()
