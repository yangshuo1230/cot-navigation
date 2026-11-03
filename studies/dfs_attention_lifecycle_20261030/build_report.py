"""A behavior atlas built AFTER the measured lifecycle and intervention study."""
import json,gzip,html,math,base64
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
S=json.loads((ROOT/'statistics.json').read_text(encoding='utf-8'))
T=S['panels']['n100']['test'];GROUPS=S['groups']
SW=json.loads((ROOT/'register_swaps.json').read_text())['results']
VS=json.loads((ROOT/'visited_register_swaps.json').read_text())['results']
TT=json.loads((ROOT/'terminal_tests.json').read_text())['results']
CA=json.loads((ROOT/'carrier_audit.json').read_text())
WR=json.loads((ROOT/'writer_audit.json').read_text())
FREE=json.loads((ROOT/'free_rollouts.json').read_text())
PLAN=json.loads((ROOT/'plan.json').read_text())
QUERIES=json.loads((ROOT/'summary_queries.json').read_text())

def metric(title,phase,feature,layer,group=False,sub=None):
    panel=S['subphases'][sub] if sub else T[phase]
    stats=panel['groups'] if group else panel['routes'][feature]
    key=(layer-1,GROUPS.index(feature)) if group else layer-1
    def get(name):return float(np.asarray(stats[name])[key])
    return dict(title=title,phase=phase,feature=feature,layer=layer,mean=get('mean'),low=get('low'),high=get('high'),
                n_queries=stats['n_queries'],n_events=stats['n_events'],sub=sub,group=group)

def pct(v):return f'{v*100:.1f}%'
def fmt(m):return f"L{m['layer']} {pct(m['mean'])}（95%区间 {pct(m['low'])}–{pct(m['high'])}；{m['n_queries']}查询/{m['n_events']}事件）"

CARDS=[
dict(id='init',name='字段初始化',headline='起点不是每次重新从 QUERY 直接搬运：prompt 的 | 是稳定回读位置。',level='局部干预支持',
 story='第一个 stack 节点从 QUERY source 开始。后续 stack 首节点以及 visited 首节点，在 L5 几乎全部回读同一个 prompt 结束符 |。该 | 在 L4 已读取 query source；这是一条可直接观察的两段读取路线。',
 boxes=['QUERY source s\nL3 表示','prompt 结束符 |\nL4 含上下文信息','新字段的预测位置\nL5 读取 → 输出 s'],links=['L4 读取 source','L5 回读 |'],
 metrics=[metric('prompt | 读取 QUERY source','stack_init_ROOT','query_source',4),metric('PUSH 后 stack 初始化读 prompt |','stack_init_PUSH','prompt_bar',5),metric('POP 后 stack 初始化读 prompt |','stack_init_POP','prompt_bar',5),metric('visited 初始化读 prompt |','visited_init','prompt_bar',5)],
 intervention='6个长查询×stack/visited两种初始化：同样可见字符 | 的 value 交换，12/12 改变预测；sham与低权重位置交换为0/12。',
 limit='“寄存器”是功能性描述：这里证明此局部 value 读取会影响初始化，不等于整段网络只沿这一条路线工作。'),
dict(id='stack',name='stack 续写',headline='新栈顶与 POP 后保留的栈节点，应分开看。',level='稳定 attention 路由',
 story='PUSH 后的新内容会读上次动作/参数；POP 后恢复保留栈节点时，L5 强读历史 first-visited 位置。这些位置的可见字母常是起点 s，但其上下文表示与历史栈的父节点有关。按“同名节点”追踪会漏掉这类载体。',
 boxes=['过去的搜索状态\nstack 的父子关系','历史 first-visited s\n不同记录，表示不同','POP 后保留栈节点\nL5 回读载体'],links=['关系匹配的载体假设','平均约 94.4%'],
 metrics=[metric('POP 后：匹配父节点关系的 first-visited 载体','stack_continue','parent_matched_first_visited_carrier',5,sub='stack_continue::POP'),metric('PUSH 后：上次 PUSH 参数（所有续写位置）','stack_continue','previous_push_argument',1,sub='stack_continue::PUSH')],
 intervention='此恢复路线本轮以 attention 统计为主；没有把平均读取质量直接解释成栈操作的完整因果回路。',
 limit='这里的94.4%仅指POP后保留节点的条件事件；不能用于所有stack续写，也不等于每个位置都在复制同一字母。'),
dict(id='visited',name='visited 续写',headline='可见的 s 也可以成为“新发现节点”的信息载体。',level='探索性局部干预支持',
 story='PUSH(v) 后，下一条记录的 stack 首节点仍写 s，但 L1 对前一 PUSH 参数 v 的平均读取达98.4%。visited 后续续写时，L5 约80.3%注意力落在历史 stack；top-5中有77.2%落在历史 stack-root s 上。它们并非都携带相同信息。',
 boxes=['前一动作 PUSH v\n参数 token v','下一记录 stack-root s\nL1 读 v；L2–L4 混合','后续 visited 的预测位置\nL5 读取历史 s 表示'],links=['L1 读参数 98.4%','中间层未逐段分解'],
 metrics=[metric('visited续写读取历史stack','visited_continue','history_stack',5,group=True),metric('visited续写直接读历史同名visited','visited_continue','previous_same_field_target',6)],
 intervention='六个长查询的中段visited续写：替换“最近发现节点=待输出节点”的历史root value，6/6 改变预测；全部变为donor携带的起点。低权重同字母root和sham为0/6。',
 limit='历史root的“最近发现节点”解释由本轮数据探索得到；其top-5匹配下界为52.4%，不是所有历史root都已解释。L1到L4的残差/FFN混合尚未拆成唯一通路。'),
dict(id='candidate',name='候选生成',headline='首次展开从输入图取信息；回溯重访转向历史动作。',level='稳定 attention 路由',
 story='首次处理节点时，L5 读取该节点出边destination后移3个位置的载体；这些位置的字面token可能是下一条边的另一个节点。回溯后再处理同一节点时，L5转而读取它以前的PUSH和参数。不是每次都执行同一种“查邻接表”。',
 boxes=['首次展开当前节点\n或：回溯重访','首次：出边的 +3 载体\n重访：旧 PUSH / 参数','候选字段的首节点\n不同条件，读取路线不同'],links=['输入图 / 历史选择接口','L5：约73% / 91%'],
 metrics=[metric('根节点候选：相关出边destination+3','candidate_init_root','current_adjacency_destination_shift3',5),metric('fresh候选：相关出边destination+3','candidate_init_fresh','current_adjacency_destination_shift3',5),metric('revisit候选：同一当前节点的旧PUSH/参数','candidate_init_revisit','previous_same_current_PUSH_pair',5)],
 intervention='本轮没有把出边+3的高attention当成“图知识已被复制”的因果证明。此前核查显示L1的+3读取本身有很强的格式/距离成分。',
 limit='候选合法顺序不唯一；下一token与canonical参考不同不必然是语义错误。fresh与revisit、空与非空列表均单独统计。'),
dict(id='ending',name='字段结束',headline='visited 已定位到 L5 后继检索；三个字段的停止控制不同。',level='新增局部干预支持 · 完整判据未解',
 story='复查18条N100 test，其中6条长轨迹做匹配干预。visited在相同可见节点处交换L5 attention输出，停止/节点双向切换12/12；向应结束的位置接入后继root的K/V，5/6输出所接入后继。stack对当前节点身份不敏感，更符合目标长度/阶段边界控制，但计数器未解码。候选L3交换7/12切换到合法类别，另1次变为非法PUSH，未计成功。',
 boxes=['当前字段已输出的前缀\n及历史记录','字段边界 | / 历史 ;\n不同层汇总不同内容','预测新的 |\n不能倒看已生成的 |'],links=['字段特异的回读','停止判据：仍待拆解'],
 metrics=[metric('visited结束：读取本轮visited','visited_end','current_visited',1,group=True),metric('候选结束：读visited结束符','candidate_end','current_visited_bar',3),metric('候选结束：读visited结束符','candidate_end','current_visited_bar',6),metric('stack结束：读取本轮stack','stack_end','current_stack',6,group=True)],
 intervention='新增432个局部输出交换条件、36个单节点编辑、54个长度/全字段身份探针和6个后继接入案例；均为已有6条long查询上的条件，不是新增查询。图中只把节点与分隔符的正确类别切换计入。',
 limit='L3读取候选visited-bar达88.2%，但同字符、更多候选的旧bar-value交换0/6改变停止，不能画成已证实的候选计数器。最后FFN可改变类别也不等于它独立完成停止判断。细节和反例见下方“字段终止复查”。'),
dict(id='action',name='action 与选分支',headline='决定动作与选择 PUSH 的节点，是两个不同的预测事件。',level='稳定 attention 路由',
 story='action由第三个 | 的行预测；PUSH参数由已经消费PUSH的行预测。前者L4强读当前stack，后者L1强读候选。长栈上的FOUND并不总是直读字面栈顶：L4平均89.8%读stack，但仅约0.9%落在真实栈顶token。',
 boxes=['本轮 stack / 候选\n包含已混合的状态信息','第三个 | 的表示\n预测 PUSH / POP / FOUND','若为 PUSH\n下一行再选候选参数'],links=['L4状态读取','L1候选读取'],
 metrics=[metric('预测POP：当前stack','action_POP','current_stack',4,group=True),metric('预测FOUND：当前stack','action_FOUND','current_stack',4,group=True),metric('预测FOUND：字面stack top','action_FOUND','current_stack_top',4),metric('PUSH参数：实际参考选中的候选','push_argument','chosen_candidate',1)],
 intervention='不能再用四节点例子中“FOUND只看栈顶”的路线代表长CoT。动作优先级的完整比较计算仍未定位，已有证据是读取位置的稳定差异。',
 limit='PUSH参数的参考选中候选不一定是自由生成时相同的候选顺序；判断动作与参数的receiver必须严格错开一位。'),
dict(id='answer',name='最终 YES / NO',headline='关键的末尾决策通路在 L1，而不是最亮的 L5 self-attention。',level='局部 value 干预支持',
 story='最终分号 ; 的L1直接读取终止动作FOUND/EXHAUST。只替换这条局部value贡献、保持该层Q/K和attention权重不变，18/18次使最终答案跟随替换后的终止动作翻转。',
 boxes=['FOUND / EXHAUST\n终止动作 token','最终分号 ;\nL1 读取动作的 value','后续层与读出\nYES / NO'],links=['局部替换：18/18翻转','后续计算整体重算'],
 metrics=[metric('YES前最终分号：读FOUND','answer_YES','terminal_action',1),metric('NO前最终分号：读EXHAUST','answer_NO','terminal_action',1),metric('YES前L5：self-attention','answer_YES','self',5,group=True)],
 intervention='反例：L5 self平均99.3%，但把该value换成前一记录同字面分号，6/6仍输出YES。高attention并不自动等于决定答案的语义通路。',
 limit='FOUND/EXHAUST替换是故意不一致的终止前缀诊断，不是另一条合法DFS轨迹；它说明动作到答案的局部读取有作用，不证明此前搜索过程正确。')]

def font(size,bold=False):return ImageFont.truetype('C:/Windows/Fonts/'+('msyhbd.ttc' if bold else 'msyh.ttc'),size)
def diagram(card,idx):
    if card['id']=='ending':
        im=Image.open(ROOT/'field_termination_swaps.png');im.resize((1320,round(im.height*1320/im.width))).save(ROOT/f'{idx:02d}_{card["id"]}.png');return
    im=Image.new('RGB',(1320,310),'white');d=ImageDraw.Draw(im)
    d.text((22,14),card['name']+' / '+card['headline'],font=font(20,True),fill='#20384e')
    fills=['#eaf2fa','#fff3dc','#e4f2ef'];xs=[25,475,925]
    for i,(x,text) in enumerate(zip(xs,card['boxes'])):
        d.rounded_rectangle((x,85,x+355,215),14,fill=fills[i],outline='#cbd8e4',width=2)
        for j,line in enumerate(text.split('\n')):d.text((x+19,119+j*32),line,font=font(19, j==0),fill='#28465f')
    for i in range(2):
        x=xs[i]+355;xx=xs[i+1]
        if '支持' in card['level']:d.line((x+7,151,xx-13,151),fill='#278a92',width=3)
        else:
            for left in range(x+7,xx-13,12):d.line((left,151,min(left+7,xx-13),151),fill='#6a96ad',width=3)
        d.polygon([(xx-9,151),(xx-19,145),(xx-19,157)],fill='#278a92')
        # Labels are placed BELOW the edge, never as weighted attribution.
        words=card['links'][i];d.text((x-5,232),words,font=font(12),fill='#59798d')
    d.text((25,276),'箭头按信息来源 → 读取位置排列；统计读取、局部干预与未解析的中间混合，在正文中区分。',font=font(14),fill='#7c8c9c')
    im.save(ROOT/f'{idx:02d}_{card["id"]}.png')

def make_heatmap():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import PowerNorm
    f=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    columns=[CARDS[0]['metrics'][0],CARDS[0]['metrics'][1],CARDS[0]['metrics'][3],CARDS[1]['metrics'][0],CARDS[2]['metrics'][0],
             CARDS[3]['metrics'][1],CARDS[3]['metrics'][2],CARDS[4]['metrics'][0],CARDS[4]['metrics'][1],CARDS[5]['metrics'][0],CARDS[5]['metrics'][1],CARDS[5]['metrics'][3],CARDS[6]['metrics'][0],CARDS[6]['metrics'][1]]
    labels=['首个stack\n读QUERY source','后续stack初始化\n读prompt |','visited初始化\n读prompt |','POP后stack续写\n读父关系载体','visited续写\n读历史stack','fresh候选初始化\n读图+3载体','revisit候选初始化\n读旧PUSH/参数','visited结束\n读本轮visited','候选结束\n读visited结束符','预测POP\n读本轮stack','预测FOUND\n读本轮stack','PUSH参数\n读所选候选','最终YES\n读终止动作','最终NO\n读终止动作']
    values=[]
    for m in columns:
        panel=S['subphases'][m['sub']] if m['sub'] else T[m['phase']]
        values.append(np.asarray(panel['groups']['mean'])[:,GROUPS.index(m['feature'])] if m['group'] else panel['routes'][m['feature']]['mean'])
    mat=np.array(values).T;fig,ax=plt.subplots(figsize=(19,6.8));image=ax.imshow(mat,cmap='YlGnBu',norm=PowerNorm(.5,0,1),aspect='auto')
    ax.set_xticks(range(len(labels)),labels,fontproperties=f,fontsize=10);ax.set_yticks(range(6),[f'L{i+1}' for i in range(6)],fontsize=11)
    for i in range(6):
        for j in range(len(labels)):ax.text(j,i,f'{mat[i,j]*100:.0f}%',ha='center',va='center',fontsize=10,color='white' if mat[i,j]>.5 else '#294258')
    ax.set_title('同一模型的长 CoT 行为读取图谱 / N100 独立 test 查询，先按 query 平均',fontproperties=f,fontsize=15,pad=18)
    cb=fig.colorbar(image,ax=ax,fraction=.025,pad=.02);cb.set_label('Attention mass (shared 0–1 scale)')
    fig.text(.03,.02,'不同列是不同条件事件与语义目标，列间不相加；仅为读取质量。完整区间、分母和干预结果见报告。',fontproperties=f,fontsize=10,color='#65788c')
    fig.subplots_adjust(left=.04,right=.94,top=.87,bottom=.22);fig.savefig(ROOT/'behavior_attention_map.png',dpi=160);plt.close(fig)

def write_report():
    correct=sum(r['correct'] for r in FREE);valid=sum(r['full_valid'] for r in FREE)
    long=[q for q in QUERIES if q['tag']=='n100'];maxlen=max(q['length'] for q in long);maxrec=max(q['records'] for q in long)
    lines=['# In Context DFS-state：长CoT的行为与attention规律','',
      '字段终止已增加专项复查：[FIELD_TERMINATION.md](FIELD_TERMINATION.md)。包含18条test路线与其中6条长查询的局部干预；下文主统计的52条查询分母不变。','',
      '**结论：该模型把可见符号和重复节点同时用作上下文相关载体；首先应按行为分组，而不是按字面token追单条亮线。**',
      '',f'本轮冻结同一185k课程模型：52条完整参考轨迹（N16 16条、N100 36条），7255个按行为对齐的事件；N100最长{maxlen}个输入token、{maxrec}条状态记录。另生成12条不按表现筛选的原生轨迹，答案正确{correct}/12，完整严格合法{valid}/12。',
      '', '主统计是teacher-forced完整参考轨迹，不能解释成全部自由生成正确。自由生成统计仅使用首次语义错误前的完整合法记录；所有失败保留。独立val用于选择传统route的展示层，test用于报告；新发现的root-register解释与局部干预明确标为探索性。',
      '', '模型：`outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state/checkpoints/best.pt`，step185000，SHA256见plan.json。CPU FP32；未训练、未占用GPU。',
      '', '## 实验契约', '',
      '- N100按label×完整参考长度（≤1024、1025–2048、2049–4096）分层，各cell每split取3条；N16按label×记录数（≤7、>7）各取2条。选择只看标签和长度，不看模型成绩。超出4096的参考轨迹显式排除，不截断后冒充完整轨迹。',
      '- 每query每phase最多12个均匀分布的机会，先query内平均，再跨query平均；95%区间为600次query bootstrap，不把同一长CoT中的token当成独立样本。',
      '- receiver q消费token q、预测q+1；action在第三个bar的行分析，PUSH参数在PUSH行分析，字段结束在新bar之前分析。',
      '- adjacent-layer的2/3跳attention乘积仅是描述性组合，未纳入残差、FFN和Q/K依赖，不作为因果贡献。不同route可重叠，不能相加成分布。',
      '- 语义group把self单列；单独定义的位置route可以包含self，两种统计不混用。低权重交换不是扰动幅度匹配控制，不能仅靠它证明语义特异性；相同可见token、不同上下文value的定向替换结果提供了额外内容证据。',
      '- 传统位置路线已参考并复核项目既有`studies/dfs_pattern_review`；本轮新样本、统计和干预单独保存。', '']
    for i,c in enumerate(CARDS,1):
        lines+=['## '+str(i)+'. '+c['name'],'', '**'+c['headline']+'**','',c['story'],'','![模式图]('+f'{i:02d}_{c["id"]}.png'+')','',
         '| 检查项 | N100独立test结果 |','| --- | --- |']
        for m in c['metrics']:lines.append('| '+m['title']+' | '+fmt(m)+' |')
        lines+=['',c['intervention'],'','适用范围：'+c['limit'],'']
    lines+=['## 符号与节点载体的分工','',
      '| 位置 | 当前证据 | 不能据此声称 |','| --- | --- |',
      '| prompt结束 `|` | L4读source；L5在字段初始化回读；局部value交换12/12改变首节点 | 唯一source通路，或每层都只处理source |',
      '| 每记录stack-root `s` | PUSH后L1读新参数98.4%；visited续写L5回读历史root；探索性交换6/6改变下一节点 | 字面`s`意味着只携带起点；所有中间层已逐段解释 |',
      '| first-visited `s` | POP后保留栈节点L5读匹配父关系载体94.4% | 与stack-root是同一种寄存器，或同一token身份即同一信息 |',
      '| 历史分号 `;` | 多类续写在L2/L4强回读，作为记录级摘要位置很稳定 | 已证明它保存了一个确定的stack/visited全集 |',
      '| 最终分号 `;` | L1动作value交换18/18使答案跟随终止动作；L5self交换0/6改变YES | 最亮L5self就是决策路径；答案正确即可保证轨迹正确 |','',
      '## 原生长CoT对照','',f'12条原生轨迹中，答案正确{correct}/12、完整轨迹严格合法{valid}/12。主模型的长自由搜索仍存在候选和字段错误，因此参考条件下的稳定模式不能直接升级为可靠搜索算法。','',
      '| query index | 输出token数 | 首错前完整合法记录 | 完整合法 | 答案正确 | 首个语义错误 |','| --- | --- | --- | --- | --- | --- |']
    for r in FREE:lines.append(f'| {r["index"]} | {r["generated"]} | {r["valid_records"]} | {r["full_valid"]} | {r["correct"]} | {r["semantic_error"]} |')
    lines+=['','完整成功的两条原生实例为43条记录的NO（query50951）和27条记录的YES（query50708）；它们是此panel中保留的成功结果，不以此替换全部panel分母。','',
      '## 本轮尚未解决','',
      '1. 字段结束已补做局部干预：visited支持L5后继检索，stack更符合长度/阶段边界控制；候选完整耗尽判据与显式计数器仍未拆解，见FIELD_TERMINATION.md。',
      '2. 长栈上的action比较：L4读stack很稳定，但不等于直接读取真实栈顶；跨token状态整合和目标比较的完整回路仍不明。',
      '3. root-register从L1到L4如何经残差、FFN和其他attention保持/重编码：已有读写与局部干预证据，不能画成中间内容完全不变的直线。',
      '4. “模型只有一条符合人类DFS的数据流”没有得到证明；有多个条件路线与分布依赖。',
      '', '## 可复现文件','',
      '- `plan.json`：checkpoint哈希、全部query ID、分层规则与超长样本排除数。',
      '- `events.jsonl.gz` / `free_events.jsonl.gz`：全部事件、分组、top-key、位置路线、符号2/3跳组合。',
      '- `statistics.json`：query-balanced均值、区间、val选层/test检查、子条件和长轨迹切片。',
      '- `register_swaps.json`、`visited_register_swaps.json`、`terminal_tests.json`：全部局部反事实与阴性结果。',
      '- `writer_audit.json`、`carrier_audit.json`：低层写入候选和高层读取的独立位置核查。',
      '- `behavior_atlas.html`：按行为组织的可视化；`behavior_attention_map.png`：全层总览。',
      '', '未生成MP4。所有原始数据和checkpoint均保留，未修改训练模型或既有结果。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    make_heatmap();write_report()
    for i,c in enumerate(CARDS,1):diagram(c,i)
    stats=dict(cards=CARDS,free=[{k:v for k,v in r.items() if k!='tokens'} for r in FREE],
        reference_queries=52,reference_events=S['reference_events'],free_events=S['free_events'],writer=WR['query_balanced_mean'],
        carrier=CA['summary'],terminal_flip=sum(r['local_swap_follows_alternate'] for r in TT),terminal_n=len(TT))
    (ROOT/'atlas_data.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    # Keep the research JSON compact; make the HTML independently viewable.
    # Local-file previews may not resolve sibling image paths reliably.
    embedded_cards=[]
    for i,c in enumerate(CARDS,1):
        picture=ROOT/f'{i:02d}_{c["id"]}.png'
        embedded_cards.append(dict(c,diagram_src='data:image/png;base64,'+base64.b64encode(picture.read_bytes()).decode('ascii')))
    payload=dict(stats,cards=embedded_cards)
    template=(ROOT/'atlas_template.html').read_text(encoding='utf-8')
    heatmap='data:image/png;base64,'+base64.b64encode((ROOT/'behavior_attention_map.png').read_bytes()).decode('ascii')
    template=template.replace('src="behavior_attention_map.png"',f'src="{heatmap}"')
    (ROOT/'behavior_atlas.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
    print(json.dumps(dict(cards=len(CARDS),reference_events=S['reference_events'],free_events=S['free_events'],terminal_flip=stats['terminal_flip'],free_correct=sum(r['correct'] for r in FREE),free_valid=sum(r['full_valid'] for r in FREE))))

if __name__=='__main__':main()
