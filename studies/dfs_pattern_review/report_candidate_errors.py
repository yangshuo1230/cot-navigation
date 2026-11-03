"""Render compact counts and real examples from the independent graph audit."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'candidate_error_breakdown.json').read_text())
lines=['# Fixed DFS state：候选首错的具体类型','',
       '来自原全测试池的首次语义错误：simple1091个查询中44条candidate_mismatch；hard555个查询中52条。没有新生成或训练。当前错误记录的stack与visited先通过独立重放检查，再检查候选，故不是将已知错误的显式visited继续当正确状态统计。','',
       '真实合法候选定义为出邻居减去visited。不限定候选排列顺序，但不允许重复。每类计数单位是“某查询首个候选错误记录是否出现该现象”，不是节点数；同一记录可以进入多类。','',
       '|类型|simple /44|hard /52|','|---|---:|---:|']
for label,key in [('漏掉合法后继','missing'),('加入已visited的真实出邻居','extra_visited_neighbor'),('加入非出邻居','extra_non_neighbor'),('候选重复','duplicate')]:
    lines.append('|'+label+'|'+'|'.join(f"{r[s]['counts'].get(key,0)} ({100*r[s]['counts'].get(key,0)/r[s]['candidate_first_errors']:.1f}%)" for s in ('simple','hard'))+'|')
lines += ['', '严格区分“visited中的真实出邻居”与“非邻居”。此前simple extra_visited_any=13，有1条额外节点既visited又不是出邻居；纯粹呈现真实邻居排重失败的记录为12条。hard的20条均为真实出邻居。','',
          '## 不重叠的组合类型','', '|组合|simple|hard|','|---|---:|---:|']
for key in sorted(set(r['simple']['disjoint_patterns'])|set(r['hard']['disjoint_patterns'])):
    lines.append(f"|{key}|{r['simple']['disjoint_patterns'].get(key,0)}|{r['hard']['disjoint_patterns'].get(key,0)}|")
lines += ['', 'simple主要是漏候选，且24/44同时漏掉合法节点并加入错误节点。hard的漏候选和额外候选都突出；30条出现额外候选，其中20条含已visited真实邻居、10条含非邻居。7条hard错误含重复，3条是集合内容正确但重复打印节点。','',
          '## 真实例子（Vn为图内拓扑编号，与近几轮图一致）','']
for setting,kind,title in [('hard','missing','漏候选'),('hard','extra_visited_neighbor','未排除已访问后继'),('simple','extra_non_neighbor','非边节点混入'),('hard','duplicate','重复并漏掉另一候选')]:
    e=r[setting]['examples'][kind]
    lines += [f"### {title}：{setting} query {e['query_id']}，第{e['record']}条",'',
              f"当前节点：{e['current']}；真实出邻居：{e['neighbors']}。",'',
              f"正确候选：{e['expected']}；生成候选：{e['shown']}。",'',
              f"遗漏：{e['missing']}；额外：{e['extra']}；重复：{e['duplicate']}；打印的PUSH参数：{e['chosen']}。",'']
    if e['extra_visited_neighbor']:lines.append(f"错误加入的真实邻居已在visited中：{e['extra_visited_neighbor']}。\n")
lines += ['## 是否影响行动','']
for setting in ('simple','hard'):
    p=r[setting]['shown_push']
    lines.append(f"- {setting}：候选首错记录中{p['PUSH']}条同时打印PUSH，其中{p['nonedge']}条选择非邻居、{p['revisited']}条选择已访问节点；{p['valid']}条的PUSH参数仍合法。")
lines += ['', '未打印PUSH的候选错误还可能引起过早POP/EXHAUST，不能将“没有非法PUSH”解释成无害。', '',
          '## 其他观察与边界','']
for setting in ('simple','hard'):
    a=r[setting]
    lines.append(f"- {setting}：第一条记录已经出现候选首错的有{a['counts']['first_record']}条；列表长度相对正确候选为{a['list_lengths']}。")
lines += ['', '这不是单一的visited排重失败：包含出邻居列举、排重、重复输出和枚举完整性问题。已有错误类型只能定位行为表现，不能单独确定FFN邻接知识缺失、历史检索错误或隐藏状态未编码visited。']
(HERE/'CANDIDATE_ERRORS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
assert sum(r['simple']['disjoint_patterns'].values())==44
assert sum(r['hard']['disjoint_patterns'].values())==52
print('Wrote independently checked candidate-error report')
