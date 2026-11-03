"""Locate first annotation and executable-search failures in saved generations."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import statistics

ROOT=Path(__file__).parent
OLD=ROOT.parent/'length_generalization'
if not (OLD/'metadata.json').exists():OLD=OLD/'results_n32'
meta=json.loads((OLD/'metadata.json').read_text())
inputs={(c,r['index']):r for c,rs in meta['inputs'].items() for r in rs}
names=['<PAD>','<EOS>','<EDGE>','<QUERY>','|',';','<YES>','<NO>','<PUSH>','<POP>','<FOUND>','<EXHAUST>','<WALK>','<DOT>']+[f'<V{i}>' for i in range(256)]
decode=lambda xs:' '.join(names[x] for x in xs)


def records(tokens):
    out=[];start=0
    for end,x in enumerate(tokens):
        if x==5:out.append(tokens[start:end]);start=end+1
    return out,tokens[start:]


def split(rec):
    parts=[[]]
    for x in rec:
        if x==4:parts.append([])
        else:parts[-1].append(x)
    return parts


def expected(frontier,seen,adj,target,fifo):
    cur=frontier[0] if fifo else frontier[-1]
    eligible=set(adj[cur])-set(seen)
    action=10 if cur==target else 8 if eligible else 9 if len(frontier)>1 else 11
    return cur,eligible,action


def apply(frontier,seen,tail,fifo):
    if tail[0]==8:frontier.append(tail[1]);seen.append(tail[1])
    elif tail[0]==9:frontier.pop(0 if fifo else -1)


def skeleton_failure(r,adj):
    frontier,seen=[r['source']],[r['source']]
    fifo=r['protocol']=='bfs_state';terminal=None
    rs,unfinished=records(r['trace_tokens'])
    for i,rec in enumerate(rs):
        if terminal:return {'type':'record_after_terminal','record':i}
        actions=[j for j,x in enumerate(rec) if x in (8,9,10,11)]
        if len(actions)!=1:return {'type':'missing_or_multiple_actions','record':i}
        tail=rec[actions[0]:];actual=tail[0]
        if len(tail)!=(2 if actual==8 else 1):return {'type':'malformed_action','record':i}
        cur,eligible,want=expected(frontier,seen,adj,r['target'],fifo)
        evidence={'record':i,'current':decode([cur]),'expected_action':names[want],
                  'actual_action':names[actual],'model_record':decode(rec),
                  'expected_frontier':decode(frontier),'expected_visited':decode(seen),
                  'eligible':decode(sorted(eligible))}
        if actual!=want:
            error=('false_found' if actual==10 else 'missed_target' if want==10 else
                   'premature_pop' if want==8 and actual==9 else
                   'premature_exhaust_with_candidates' if want==8 and actual==11 else
                   'premature_exhaust_with_frontier' if want==9 and actual==11 else
                   'push_without_candidate' if actual==8 else 'wrong_action')
            return {'type':error,**evidence}
        if actual==8 and tail[1] not in eligible:
            error='push_revisited_node' if tail[1] in seen else 'push_non_edge'
            return {'type':error,**evidence,'arg':decode(tail[1:])}
        if actual in (10,11):terminal=actual
        else:apply(frontier,seen,tail,fifo)
    if unfinished:return {'type':'unfinished_record','record':len(rs),'model_record':decode(unfinished)}
    if terminal is None:return {'type':'answer_before_terminal','record':len(rs)}
    if r['pred']!=int(terminal==10):return {'type':'answer_disagrees_with_terminal','record':len(rs)}
    return None


def strict_failure(r,adj):
    if r['trace_valid']:return None
    rs,unfinished=records(r['trace_tokens']);n=r['valid_records'] or 0
    frontier,seen=[r['source']],[r['source']]
    fifo=r['protocol']=='bfs_state'
    for rec in rs[:n]:apply(frontier,seen,split(rec)[3],fifo)
    evidence={'record':n,'type':r['trace_error'],'n_records':len(rs)}
    if n>=len(rs):return evidence
    parts=split(rs[n]);cur,eligible,want=expected(frontier,seen,adj,r['target'],fifo)
    evidence.update(model_record=decode(rs[n]),expected_frontier=decode(frontier),
                    expected_visited=decode(seen),eligible=decode(sorted(eligible)))
    if len(parts)!=4:
        evidence['subtypes']=[f'fields_{len(parts)}']
        return evidence
    shown,vis,cands,tail=parts
    if r['trace_error'] in ('visited_mismatch','frontier_mismatch'):
        actual,truth=(vis,seen) if r['trace_error']=='visited_mismatch' else (shown,frontier)
        sub=[]
        if len(actual)!=len(set(actual)):sub.append('duplicate')
        if set(actual)-set(truth):sub.append('extra')
        if set(truth)-set(actual):sub.append('missing')
        if Counter(actual)==Counter(truth) and actual!=truth:sub.append('reordered')
        evidence['subtypes']=sub
    elif r['trace_error']=='candidate_mismatch':
        sub=[]
        if eligible-set(cands):sub.append('missing_eligible')
        if set(cands)&set(seen):sub.append('includes_visited')
        if set(cands)-set(adj[cur]):sub.append('includes_non_neighbor')
        if len(cands)!=len(set(cands)):sub.append('duplicate')
        evidence['subtypes']=sub
        evidence['missing_candidates']=decode(sorted(eligible-set(cands)))
        evidence['shown_candidates']=decode(cands)
    return evidence


summary={};examples={};detailed=[]
for line in (OLD/'predictions.jsonl').open():
    r=json.loads(line)
    if r['protocol'] not in ('bfs_state','dfs_state'):continue
    src=inputs[r['case'],r['index']]
    adj={v:[] for v in src['nodes']}
    for u,v in src['edges']:adj[src['nodes'][u]].append(src['nodes'][v])
    strict=strict_failure(r,adj);skeleton=skeleton_failure(r,adj)
    key=r['model']+'/'+r['case']
    if key not in summary:summary[key]={'n':0,'strict':Counter(),'subtypes':Counter(),
        'skeleton':Counter(),'strict_first_record':Counter(),'skeleton_first_record':Counter(),
        'answer_correct':0,'strict_valid':0,'skeleton_valid':0,'bad_strict_but_good_answer':0}
    s=summary[key];s['n']+=1;s['answer_correct']+=r['correct'];s['strict_valid']+=strict is None;s['skeleton_valid']+=skeleton is None
    if strict:
        s['strict'][strict['type']]+=1;s['strict_first_record'][strict['record']]+=1
        for z in strict.get('subtypes',[]):s['subtypes'][strict['type']+'/'+z]+=1
        s['bad_strict_but_good_answer']+=r['correct']
    if skeleton:
        s['skeleton'][skeleton['type']]+=1;s['skeleton_first_record'][skeleton['record']]+=1
    detail={'model':r['model'],'case':r['case'],'index':r['index'],'hop':r['hop'],
            'label':r['label'],'pred':r['pred'],'strict_failure':strict,'skeleton_failure':skeleton}
    detailed.append(detail)
    for error in (strict,skeleton):
        if error:
            for subtype in error.get('subtypes',[error['type']]):
                eid=(r['model'],r['case'],error['type'],subtype)
                if eid not in examples:examples[eid]=detail
(ROOT/'error_breakdown.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(ROOT/'detailed_errors.json').write_text(json.dumps(detailed,indent=2),encoding='utf-8')


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


lines=['# 轨迹错误细分：首次状态错误与首次动作错误\n',
       '基于上一轮28736条已保存生成结果，不重新生成。这里每条轨迹只对首次相应错误分类，'
       '避免把一次早期错误后的整段连锁偏差重复计数；状态子类型可能同时出现。\n',
       '## 新图BFS\n']
for model in ('context/bfs_state','context/dfs_state','fixed/dfs_state','fixed/bfs_state','fixed/dfs_random'):
    lines += ['## '+model+'\n']
    rows=[]
    for key,s in summary.items():
        if not key.startswith(model+'/'):continue
        first=s['strict_first_record']
        rows.append([key[len(model)+1:],s['n'],s['strict_valid'],s['skeleton_valid'],
                     str(dict(s['strict'])),str(dict(s['skeleton'])),
                     first.get(0,0)])
    lines += [table(['条件','n','状态完全合法数','动作完全合法数','首次状态错误','首次动作错误','状态首错在record0'],rows)]
lines += ['## 可核对的具体例子\n']
wanted=[('context/bfs_state','id','candidate_mismatch','missing_eligible'),
        ('context/bfs_state','id_disconnected_pad32','visited_mismatch','duplicate'),
        ('context/bfs_state','id_disconnected_pad32','visited_mismatch','extra'),
        ('context/bfs_state','deep16_sameprompt','candidate_mismatch','missing_eligible'),
        ('context/dfs_state','id_disconnected_pad32','malformed_fields','fields_5'),
        ('fixed/dfs_state','fixed0','wrong_action','wrong_action')]
for key in wanted:
    e=examples.get(key)
    if e:lines += ['### '+' / '.join(key)+'\n','```json\n'+json.dumps(e,ensure_ascii=False,indent=2)+'\n```\n']
lines += ['## 含义\n',
    '原分布BFS的首错以漏候选为主；padding后首错转为visited维护，且大量样本第一或很早的record就出错。'
    '动作重放能进一步识别提前POP/EXHAUST、错误FOUND、重复入队或走不存在的边；它们不是只与参考排序不同。'
    '固定图普通DFS/BFS还有错误终止的问题；随机DFS的主要错误明显减少。\n',
    '输入prompt原先固定76token，但CoT输出及训练总序列长度本来是可变的。固定的是图输入和查询边界，'
    '不应称为模型从未见过位置148。补长同时改变边-查询距离、无关节点数和注意力竞争，'
    '因此长度内插能检验训练支持范围，但不能单独证明绝对位置编码是原因。']
(ROOT/'ERROR_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
nonstate={};nonexamples={}
for line in (OLD/'predictions.jsonl').open():
    r=json.loads(line)
    if r['model'] not in ('context/path_negative','context/dfs_nodes','context/bfs_nodes'):continue
    key=r['model']+'/'+r['case']
    if key not in nonstate:nonstate[key]={'n':0,'strict_valid':0,'route_prefix_valid':0,'errors':Counter(),'invalid_token_subtypes':Counter()}
    s=nonstate[key];s['n']+=1;s['strict_valid']+=r['trace_valid']
    s['errors'][r['trace_error'] or 'valid']+=1
    src=inputs[r['case'],r['index']];adj={v:[] for v in src['nodes']}
    for u,v in src['edges']:adj[src['nodes'][u]].append(src['nodes'][v])
    body=r['trace_tokens']
    if r['trace_error']=='non_node_token':
        if any(v<14 for v in body):s['invalid_token_subtypes']['reserved_token']+=1
        if any(v>=14 and v not in adj for v in body):s['invalid_token_subtypes']['node_not_in_current_graph']+=1
    if r['model']=='context/path_negative':
        legal=bool(body) and body[0]==r['source'] and len(body)==len(set(body)) and all(v in adj for v in body)
        legal=legal and all(v in adj[u] for u,v in zip(body,body[1:]))
        s['route_prefix_valid']+=legal
    if r['trace_error'] and r['case'] in ('id','deep16_sameprompt','deep48'):
        k=r['model'],r['case'],r['trace_error']
        if k not in nonexamples:
            nonexamples[k]={'model':r['model'],'case':r['case'],'index':r['index'],'hop':r['hop'],
                'label':r['label'],'pred':r['pred'],'source':decode([r['source']]),'target':decode([r['target']]),
                'trace':decode(body),'error':r['trace_error'],
                'last_node_out_neighbors':decode(adj.get(body[-1],[])) if body else ''}
(ROOT/'nonstate_errors.json').write_text(json.dumps(nonstate,indent=2),encoding='utf-8')
text=['# Path与节点序列错误\n',
      'Path负例的“必须到sink”是当前训练生成器的规则，不是图路径本身合法的必要条件。'
      '下面另统计只要求起点正确、不重复、逐边合法的route prefix。它仍不能证明NO。'
      '若允许负例随机路径在任意位置停止，不应把negative_path_not_at_sink计为走错边。\n']
text += ['错误码non_node_token也包含“合法词表节点，但不在本张输入图中”，不一定是输出了特殊token。'
         'JSON中的invalid_token_subtypes进一步区分了这两种情况。\n']
for m in ('context/path_negative','context/dfs_nodes','context/bfs_nodes'):
    text += ['## '+m+'\n',table(['条件','n','当前训练协议完整合法','路径前缀合法（仅Path）','首次错误分布'],
       [[key[len(m)+1:],s['n'],f"{s['strict_valid']/s['n']:.2%}",
         f"{s['route_prefix_valid']/s['n']:.2%}" if m=='context/path_negative' else '不按路径评判',str(dict(s['errors']))]
        for key,s in nonstate.items() if key.startswith(m+'/')])]
text += ['## 具体例子\n']
for key in [('context/path_negative','id','non_node_token'),
            ('context/path_negative','deep16_sameprompt','negative_path_not_at_sink'),
            ('context/dfs_nodes','id','not_dfs_order'),('context/bfs_nodes','id','not_bfs_order')]:
    if key in nonexamples:text += ['```json\n'+json.dumps(nonexamples[key],ensure_ascii=False,indent=2)+'\n```\n']
(ROOT/'NONSTATE_ERRORS.md').write_text('\n'.join(text),encoding='utf-8')
for k,s in summary.items():
    if k.startswith('context/bfs_state/') or k=='fixed/dfs_state/fixed0':
        print(k,'n',s['n'],'subtypes',dict(s['subtypes']),'actions',dict(s['skeleton']))
