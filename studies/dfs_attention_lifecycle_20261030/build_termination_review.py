"""Build a narrowly scoped, evidence-graded field-termination follow-up."""
import json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/(name+'.json')).read_text(encoding='utf8'))
F=read('field_termination');C=read('field_termination_candidate');P=read('field_termination_probe');B=read('field_boundary_carriers');G=read('visited_successor_graft')
all_swaps=F['results']+[dict(field='candidates',**r) for r in C['results']]
def node(t):return t.startswith('<V') and t.endswith('>')
def flip(r):return r['prediction']=='|' if r['direction']=='continue_to_stop' else node(r['prediction'])
def balanced(xs,route):
    groups=collections.defaultdict(list)
    for r in xs:groups[r['index']].append(r['routes'][route])
    return np.mean([np.mean(v,axis=0) for v in groups.values()],axis=0).tolist()

def main():
    assert all(j['complete'] for j in [F,C,P,B,G])
    assert len(all_swaps)==432 and len(P['identity_edits'])==36 and len(C['length_identity_probes'])==54
    assert all(r.get('baseline_correct',r.get('baseline_class_correct')) for r in all_swaps)
    assert all(r['earlier_logit_max_difference']<.01 for r in all_swaps)
    assert all(r['follows_donor_class']==flip(r) for r in all_swaps)
    assert all(r['identical'] for r in P['shams']+C['shams'])
    assert all(r['earlier_logits_unchanged'] for r in B['results'])
    assert all(not r['changed_prediction'] for r in B['results'] if r['condition'] in ['sham','low_attention'])
    assert all(r['sham_logit_max_difference']==0 for r in G['results'] if r['condition']=='sham')
    assert read('field_permutation_validation')['passed']
    matrices={};counts={}
    for component in ['attention','ffn']:
        matrix=[]
        for field in ['stack','visited','candidates']:
            values=[]
            for layer in range(1,7):
                rs=[r for r in all_swaps if r['field']==field and r['layer']==layer and r['component']==component]
                assert len(rs)==12
                values.append(sum(flip(r) for r in rs))
            matrix.append(values)
        matrices[component]=matrix
    for phase in sorted(set(r['phase'] for r in P['routes'])):
        rs=[r for r in P['routes'] if r['phase']==phase]
        counts[phase]=dict(n_queries=len(set(r['index'] for r in rs)),n_events=len(rs),
            means={route:balanced(rs,route) for route in rs[0]['routes']})
    summary=dict(reference_query_scope=18,intervention_queries=6,query_ids=sorted(set(r['index'] for r in all_swaps)),
        receiver_output_swaps=432,output_swap_matrices=matrices,route_audit=counts,
        graft_success={kind:sum(r['follows_inserted_successor'] for r in G['results'] if r['condition']==kind) for kind in ['sham','K_only','V_only','K_and_V']},
        identity_edits=36,length_identity_probes=54,legal_candidate_permutation_prefixes=6,
        candidate_permutation_complete_stop=sum(r['set_correct'] for r in P['candidate_permutations'] if r['kind']=='all_emitted'),
        candidate_permutation_one_remaining_correct=sum(r['set_correct'] for r in P['candidate_permutations'] if r['kind']=='one_remaining'))
    (ROOT/'field_termination_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11.8,3.6),layout='constrained')
    for ax,component,title in zip(axes,['attention','ffn'],['交换 receiver 的 attention 输出','交换 receiver 的 FFN 输出']):
        m=np.array(matrices[component]);ax.imshow(m,vmin=0,vmax=12,cmap='Blues')
        ax.set_xticks(range(6),[f'L{i}' for i in range(1,7)]);ax.set_yticks(range(3),['stack','visited','candidates']);ax.set_title(title,fontsize=12)
        for i in range(3):
            for k in range(6):ax.text(k,i,f'{m[i,k]}/12',ha='center',va='center',fontsize=11,color='white' if m[i,k]>7 else '#24415a')
        ax.tick_params(length=0)
    fig.suptitle('同一可见节点：停止与继续的双向局部交换\n每字段 6 条长轨迹 × 2 个方向；只有正确的节点 / | 类别切换计入',fontsize=12)
    fig.savefig(ROOT/'field_termination_swaps.png',dpi=160);plt.close(fig)
    report='''# 字段终止复查：停止信号在哪里，哪些判据尚未识别

这次把“在哪层出现可交换的停止信号”和“如何算出应该停止”分开检查。结果最清楚的是 **visited 的 L5 历史发现关系检索**；stack 更像受目标长度/阶段边界控制；candidates 的阶段控制涉及 L3，但完整集合耗尽电路仍不能宣称已找到。

## 范围与读法

- 原来的 52 条参考查询不变。本轮精查其中 **18 条 N100 test**，共 1,331 个续写/结束事件；干预用其中全部 **6 条 long 查询**，不按干预结果筛选。
- 所有基础轨迹均为正确参考前缀，不新增自由生成成功率结论。6 是查询数；12 是每字段的 6 对停止/继续位置、两个交换方向；432 是局部 attention/FFN 输出交换条件数。
- attention 行 q 消费 q、预测 q+1；终止实验的 receiver 在分隔符 `|` **之前**，没有读取尚未生成的分隔符。
- stack/visited 选同一 query 内、同一可见节点、一个应结束另一个应继续的位置。candidates 通过同一记录的合法候选排列构造相同可见节点的停止/继续对照。共 36 个 receiver 条件的基线停止/继续类别全部正确。
- 交换只替换选定层、选定 receiver 的 attention 或 FFN 输出，保留其他位置并重算下游。不同位置提供的上下文不同；这是局部充分性诊断，不是合法替代搜索。接收者之前的 logits 已核验无实质变化；SDPA 前缀舍入差记录在原始结果中。独立 sham 36 个均精确不变。

## 1. stack：节点身份不是这组样本的直接停止开关

固定长度，只把末节点改成倒数第二节点：6/6 仍输出 `|`。把倒数第二节点改成末节点：6/6 仍继续输出原本应补的节点。把整个 stack 字段改成重复首节点：6/6 仍在原边界停止。因而不能把它解释成“当前字面节点等于目标栈顶就结束”。

长度探针则有明确差异：删去倒数第二节点、保留原末节点后，6/6 **继续补写**；在末节点前多插一个重复节点，6/6 仍停止。这与“达到状态所规定的字段长度/阶段阈值”相容，也与偏移复制或位置相关控制相容。插删会改变位置且使状态非法，故尚未区分显式计数器、相对位置和复制边界。

局部交换不是单层 attention 就能完整控制：L6 attention 仅 4/12 个方向切换类别；L6 FFN 为 10/12（继续→停止 6/6，停止→节点 4/6）。这是高层输出阶段已含停止信息的证据，不能反推 FFN 独自计算目标深度。

专门试过“前一分号寄存了目标深度”假设：固定 Q/K/attention，把前一 `;` 的 V 换成同动作、不同目标深度记录的 `;`。L2 平均读取 47.3%，L4 为 23.2%，但两层各 0/6 改变停止输出。**这条单独 value 路线没有得到支持**，不能因为分号很亮就画成已知深度计数器。

## 2. visited：最符合发现顺序的后继检索，但中间重编码仍有缺口

把每次 PUSH 后的新记录 stack-root 记成发现关系载体：上一轮 visited 尾节点为 x，新 PUSH 节点为 v；该 root 的低层已知会读取 v。可以把它作为候选 `x → v` 关系的上下文载体，其可见字符依旧是起点 s。

观察到的路线：visited 续写的 L5 平均 **67.6%** attention 落在“前一轮 visited 尾节点等于当前输出节点”的历史 root 上；所有续写事件都存在这样的后继载体。visited 结束时，当前输出是截至此时最新发现的节点，历史中没有对应后继 root，匹配组为 0；L5 转而把平均 **47.7%** 分配给已出现的字段分隔符。这一“后继是否存在”的事实部分由正确 DFS 前缀保证，不能单凭相关性当成模型已使用该算法。

干预补上了更直接的证据：交换相同可见节点处的 **L5 attention 输出，两个方向 12/12 均切换为供体的停止/节点类别**；L1–L4、L6 attention 的同类交换均 0/12。

更强的局部反事实：在原本结束的位置，向本轮 root 的 L5 单个 K/V 槽接入“之后才发现的后继”表示；只重算当前 attention 行，其他位置保持原样。这个表示来自同一参考轨迹的未来 root，所以是刻意加入额外信息的诊断，**不代表当时合法可知的信息**。

| 接入内容 | 改为输出所接入后继节点 |
| --- | --- |
| 不改（sham） | 0/6 |
| 只换 K | 0/6 |
| 只换 V | 2/6 |
| 同时换 K 与 V | **5/6** |

例如 query50717，q=1076 原本输出 `|`；只在可见起点 root 槽接入后继寄存器，预测变为 `<V206>`。对照中单独 K 或 V 都仍输出 `|`。query50785 是明确反例：K+V 接入后仍停止。

因此目前支持的机制是：**L5 借上下文键值关系续写发现序列；到了没有可检索后继的末尾，接收状态进入可输出 `|` 的分支。** 尚未分离“检索失败触发默认分隔符”、长度/阶段信号与其他冗余线路；也没有逐层解码 K/V 从 L1 到 L4 的全部内容。

## 3. candidates：不能把高亮的 visited-bar 等同于“候选已耗尽”

为了控制图、当前节点和候选全集，保持同一记录的候选集合不变，交换首末候选顺序。6 个排列都通过独立 replay 的合法前缀检查；若继续执行 PUSH，参数必须随排列改为首候选，不能复用原搜索的后续轨迹。

- 还剩一个候选时，6/6 都没有提前输出 `|`，但仅 4/6 输出正确剩余节点。
- 全部候选已经输出时，5/6 正确停止；query50821 仍重复输出 `<V105>`。

这说明模型有很强的字段进度控制，也会受顺序影响；**尚不具备在这些干预前缀上完全可靠的集合去重/耗尽行为**。

在相同可见候选节点的停止/继续对照中，L3 attention 输出交换有 **7/12 个方向切换为合法的节点/分隔符类别**。另有 1 次由 `|` 变为非法 `<PUSH>`，不能算成功续写。L6 FFN 交换为 12/12。L3 所读取的 visited 结束符在续写和终止时都很亮，因此更适合暂称阶段/上下文接口。

直接检验这个 bar 是否保存候选目标长度：用同一当前节点较早访问、候选更多的 visited-bar V 替换本轮 bar，固定 receiver Q/K 与权重。L3 平均权重 **88.2%**，仍 **0/6** 从停止改为续写；L6 权重 67.3%，仅 **2/6** 由停止改为节点。对应 sham 与低 attention 同字符对照都未改变输出；低权重对照并未匹配扰动幅度。

结论是：L3 与候选阶段有关，某些上下文信息也可经 L6 的 bar-value 路线改变结束输出；但 **“L3 的 | 存着候选数，读一下就知道耗尽”没有得到支持**。

## 4. 共同高亮模式与最终读出

L1 关注 q−3 并非独立的停止信号：visited 续写为 90.6%，结束为 85.3%；candidates 续写为 89.4%，结束为 82.5%。stack 续写为 39.0%，结束为 79.3%。这些是完整行的精确路线统计，按 query 内平均后跨 query 平均，非因果贡献百分比。

最后 FFN 的交换容易改变输出类别：stack 10/12，visited 12/12，candidates 12/12。由于它接近输出层，这只说明最终读出阶段已包含可操纵的停止/节点信息；**不能以此宣称找到了三个字段共享的完整计数算法**。

![六层交换对照](field_termination_swaps.png)

## 证据边界与复现

这是一轮探索性机制分析，使用已有 test panel 产生假设再做干预，不冒充新的预注册确认；每个字段只有 6 条长查询，比例只描述这些案例。对于非法 token/长度探针，只陈述网络反应；合法排列前缀另用独立 replay 检查。所有阴性、非法输出与例外均保留。

- `field_termination.py/.json`：stack/visited 同节点双向局部输出交换（288 条件）。
- `field_termination_candidate.py/.json`：候选同节点交换（144 条件）及 54 个长度/全字段身份探针。
- `field_termination_probe.py/.json`：18 条查询精确路线、36 个单节点编辑、12 个候选排列预测和 24 个 sham。
- `field_boundary_carriers.py/.json`：24 个符号 carrier 目标案例 × 主交换/sham/低权重对照。
- `visited_successor_graft.py/.json`：6 个后继接入案例 × 4 种条件，sham 精确不变。
- `verify_field_permutations.py`、`field_permutation_validation.json`：候选排列合法前缀检查。
- `field_termination_summary.json`、`field_termination_validation.json`：汇总和本地完整性检查。

CPU FP32，同一冻结 checkpoint；未训练、未新增自由生成批次、未生成 MP4。
'''
    (ROOT/'FIELD_TERMINATION.md').write_text(report,encoding='utf8')
    validation=dict(passed=True,queries=summary['query_ids'],reference_route_events=len(P['routes']),
        receiver_swaps=len(all_swaps),same_receiver_baselines_correct=True,sham_exact=36+24+6,
        legal_permutation_prefixes=6,non_node_outputs_excluded_from_continuation=True,
        maximum_output_swap_earlier_logit_difference=max(r['earlier_logit_max_difference'] for r in all_swaps))
    (ROOT/'field_termination_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
    print(json.dumps(dict(validation=validation,swap_matrices=matrices,graft=summary['graft_success'])))

if __name__=='__main__':main()
