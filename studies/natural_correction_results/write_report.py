"""Summarize saved results; no new model evaluation or checkpoint selection."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
data=json.loads((root/'summary.json').read_text())
refs=json.loads((root/'baselines.json').read_text())
audit=json.loads((root/'natural_preview_audit.json').read_text())
runs={r['job']:r for r in data['runs']}
ordered=[runs[f'context/{p}/p{rate}_l{lam}'] for p in ['dfs_state','bfs_state'] for rate in [30,50] for lam in [1,16]]
def pct(x):return f'{100*x:.2f}%'

text=['# 自然候选错误训练结果（2026-10-23）','','8 个分支均已完成 200k 步和全部最终评估，共 144 份最终自由生成测试报告。此次检查只读取已有结果，并独立重放保存的轨迹预览，没有启动新训练或 GPU 评估。','',
'## 设置与口径','','DFS/BFS × 30%/50% 错误样本 × 轨迹相对权重 λ=1/16。所有轨迹 token 均计算 loss。损失为 `2*(answer + λ*mean_trace)/(1+λ)`，λ 作用于整条轨迹，不是仅作用于 MISTAKE。','',
'沿用简单分层图课程、6 层 128 维模型、4K 上下文、200k 步；干净自由生成验证增为 64 查询，用于选择 best；最终主测试为 N100/M150、256 查询，正负各 128。last 固定为最终权重，不依据测试成绩重新选择 best。','',
'错误候选是旧干净模型在正确搜索前缀条件下实际完成的列表，保留原 token；包含单次多节点错误，也可在同一正确轨迹组合多个已观测错误与修正点。这不是一次完全自由生成、历史也可能出错的整条原始轨迹。训练错误仅来自训练图，诊断错误来自独立留出图。','',
'## 主测试 accuracy','','|协议|掺入比例|λ|best 步数|best|last|','|---|---:|---:|---:|---:|---:|']
for r in ordered:
    p,arm=r['job'].split('/')[-2:];rate,lam=arm.split('_')
    b,l=r['tests']['best'],r['tests']['last']
    text.append(f"|{p}|{rate[1:]}%|{lam[1:]}|{b['checkpoint_step']}|{pct(b['overall']['answer_accuracy'])}|{pct(l['overall']['answer_accuracy'])}|")
text+=['','DFS 30% λ16 最突出，best=240/256，last=246/256；BFS 50% λ16 为 best=218/256，last=217/256。','',
'比例和权重存在交互：DFS 在 30% 下提高 λ 明显改善，50% 下反而下降；BFS 在 50% 下提高 λ 有明显收益，30% 下主测试基本持平。不能概括成错误越多或轨迹权重越高越好。','',
'## 历史干净训练参考','','|参考实验|best|last|','|---|---:|---:|']
for key in ['clean_dfs_full_l1','clean_dfs_full_l16','original_clean_dfs_state','original_clean_bfs_state']:
    text.append(f"|{key}|{pct(refs[key]['best']['overall']['answer_accuracy'])}|{pct(refs[key]['last']['overall']['answer_accuracy'])}|")
text+=['','同架构、同图数据的历史干净 DFS λ16，last 已达 93.36%。本轮 DFS 30% λ16 的 96.09% 相比它增加 2.73 个百分点，即净多答对 7/256 条；不应把相对更早 λ1 模型的全部收益都归因于错误数据。历史参考的词表/初始化及验证规模不完全相同，且均为单 seed，仍不能严格分离错误数据和其他因素。','',
'## 留出自然错误上的局部纠错','','以下固定使用验证选择的 best，每组同一协议对应相同的 64 个留出错误案例。输入到错误候选字段末尾为止，MISTAKE 不由评估器提供；后续修复须由模型自主生成。','',
'|协议/比例/λ|识错数|逐 token 精确修复|允许候选重排的状态修复|干净前缀误报|','|---|---:|---:|---:|---:|']
for r in ordered:
    d=r['diagnostics']['test_best']['overall']
    text.append(f"|{r['job'].removeprefix('context/')}|{d['detected']}/64|{d['exact_repair']}/64|{d['state_repair']}/64|{d['false_alarm']}/64|")
text+=['','这里的状态修复要求 stack/queue 与 visited 恢复为参考值、候选成员完整且无重复、操作合法；允许候选顺序不同及选择不同的合法首候选。精确修复还要求整条记录逐 token 相同，因此两者差距可能很大。局部修复不等同于后续完整导航成功。','',
'DFS 30% λ16：best 识错 42/64，状态修复 25/64；last 识错同为 42/64，状态修复 29/64（45.31%）。BFS 50% λ16：best 识错 22/64，状态修复 17/64。仍存在大量漏检和检出后的修复失败。','',
'DFS 50% λ16 的诊断明显异常：只检出 1/64，修复 0/64；主任务仍有 82.42%，说明“会回答导航”和“会显式识错修复”并不一致。末步 trace loss 也高于 DFS 30% λ16，但单个训练 batch 的 loss 不能用于断言训练崩溃或容量不足。','',
'与上一轮诊断不可直接比较：上轮是每组 16 个人工单节点错误，本轮是 64 个模型实际产生的复杂候选错误，测试分布不同。','',
'## 自然自由生成中的实际修正','','独立重放 best 的 9 组测试、每组首 4 条保存预览，共 288 条；只在第一次未承认的非法状态/操作之前审计可信前缀。共确认 4 个真实局部修复事件：BFS 30% λ16 一个、BFS 50% λ16 两个、DFS 30% λ1 一个。这是预览审计，不是总体纠错率，也不能根据未发现事件断言其他模型从不纠错。','',
'一个来自 BFS 50% λ16、N100 主测试 query 50543 的例子：','',
'```text','队列 V34 V64，visited V170 V34 V64','错误候选：V220 V1 V207 V58 | MISTAKE ;','下一条候选：V1 V58 V207 | PUSH V1 ;','```','',
'根据输入图与此前合法操作，真实候选集合为 {V1,V58,V207}；模型删除了错误的 V220，保持队列与 visited 正确，并执行合法 PUSH。这个样本最后也答对了 NO，未触发强制 STOP。但没有反事实干预，不能据此证明最终答对是由修正造成的。','',
'DFS 30% λ16 的 last 主测试只有 3.91% 查询输出 MISTAKE，却达到 96.09%；其收益也可能体现为更好的常规状态生成，不应全部解释成显式纠错救回答案。','',
'## 跨规模和形状：best accuracy','','|协议/比例/λ|N16|N64|N100|深图 N100|宽图 N100|N144|','|---|---:|---:|---:|---:|---:|---:|']
for r in ordered:
    tags=['best-n16','best-n64','best','best-deep','best-wide','best-larger']
    text.append('|'+r['job'].removeprefix('context/')+'|'+'|'.join(pct(r['tests'][t]['overall']['answer_accuracy']) for t in tags)+'|')
text+=['','DFS 30% λ16 在 N144 达到 87.50%，BFS 50% λ16 为 83.98%，两者在本轮跨规模测试中也较突出。深图仍相对困难，分别为 82.81% 和 80.47%。不同集合不是同一批图，不能把准确率随 N 的变化直接当作纯规模效应。','',
'## 核验与结论','','- 8 个训练日志均记录到 200000 步，末步 loss 有限；每个分支的 18 份报告均有 256 查询，训练步数为 200000。',
'- 所有分支独立保留验证选择 best 和固定 last，没有用测试成绩回头改选。',
'- 值得保留的主要候选是 DFS 30% λ16，以及 BFS 50% λ16；不是所有自然错误训练组合都有收益。',
'- 相比上一轮，已经观察到可验证的自然局部修正案例，但仍不足以声称形成了可靠、稳定的自然自纠错。',
'- 后续若要确定能力提升来自纠错机制，需要同词表/初始化的同期干净 λ16 对照、多 seed，以及对 MISTAKE 前后状态或 activation 的干预；本次检查没有启动这些额外实验。','']
for r in ordered:
    assert len(r['tests'])==18 and r['last_train']['step']==200000
    for t in r['tests'].values():
        o=t['overall'];assert t['training_step']==200000 and o['n']==256
        assert abs(o['answer_accuracy']-o['n_correct']/256)<1e-12
(root/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
print('Verified 144 saved final reports; report written.')
