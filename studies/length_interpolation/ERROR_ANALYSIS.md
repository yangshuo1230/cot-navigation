# 轨迹错误细分：首次状态错误与首次动作错误

基于上一轮28736条已保存生成结果，不重新生成。这里每条轨迹只对首次相应错误分类，避免把一次早期错误后的整段连锁偏差重复计数；状态子类型可能同时出现。

## 新图BFS

## context/bfs_state

| 条件 | n | 状态完全合法数 | 动作完全合法数 | 首次状态错误 | 首次动作错误 | 状态首错在record0 |
| --- | --- | --- | --- | --- | --- | --- |
| id | 192 | 149 | 149 | {'candidate_mismatch': 43} | {'premature_pop': 25, 'push_non_edge': 10, 'push_without_candidate': 4, 'push_revisited_node': 4} | 0 |
| id_disconnected_pad32 | 192 | 17 | 73 | {'visited_mismatch': 144, 'candidate_mismatch': 30, 'malformed_fields': 1} | {'premature_pop': 67, 'wrong_action': 10, 'push_non_edge': 21, 'push_revisited_node': 8, 'push_without_candidate': 9, 'false_found': 1, 'missed_target': 2, 'missing_or_multiple_actions': 1} | 31 |
| wide32_branch2 | 192 | 24 | 77 | {'visited_mismatch': 130, 'candidate_mismatch': 36, 'malformed_fields': 2} | {'premature_pop': 71, 'push_without_candidate': 8, 'push_non_edge': 29, 'missed_target': 1, 'wrong_action': 1, 'premature_exhaust_with_candidates': 1, 'false_found': 2, 'missing_or_multiple_actions': 1, 'push_revisited_node': 1} | 39 |
| wide32_densityhalf | 192 | 0 | 3 | {'candidate_mismatch': 113, 'visited_mismatch': 68, 'malformed_fields': 3, 'frontier_mismatch': 8} | {'premature_pop': 118, 'premature_exhaust_with_candidates': 4, 'push_without_candidate': 4, 'push_non_edge': 49, 'missing_or_multiple_actions': 5, 'false_found': 2, 'push_revisited_node': 6, 'malformed_action': 1} | 155 |
| deep16_sameprompt | 448 | 246 | 262 | {'candidate_mismatch': 191, 'visited_mismatch': 3, 'frontier_mismatch': 7, 'malformed_fields': 1} | {'premature_pop': 126, 'push_non_edge': 12, 'push_without_candidate': 27, 'premature_exhaust_with_candidates': 14, 'false_found': 2, 'push_revisited_node': 4, 'missing_or_multiple_actions': 1} | 0 |
| deep24 | 320 | 48 | 102 | {'visited_mismatch': 243, 'candidate_mismatch': 29} | {'premature_pop': 126, 'push_non_edge': 45, 'push_revisited_node': 19, 'push_without_candidate': 25, 'wrong_action': 2, 'false_found': 1} | 26 |
| deep32 | 448 | 64 | 100 | {'visited_mismatch': 288, 'candidate_mismatch': 86, 'malformed_fields': 10} | {'premature_pop': 208, 'push_non_edge': 81, 'push_revisited_node': 21, 'push_without_candidate': 24, 'false_found': 1, 'premature_exhaust_with_candidates': 3, 'missing_or_multiple_actions': 10} | 100 |
| deep48 | 704 | 73 | 130 | {'visited_mismatch': 369, 'candidate_mismatch': 222, 'malformed_fields': 32, 'frontier_mismatch': 8} | {'premature_pop': 348, 'push_non_edge': 139, 'push_revisited_node': 24, 'missing_or_multiple_actions': 32, 'missed_target': 2, 'push_without_candidate': 19, 'false_found': 4, 'premature_exhaust_with_candidates': 6} | 328 |
| large64 | 448 | 29 | 57 | {'visited_mismatch': 227, 'candidate_mismatch': 141, 'malformed_fields': 17, 'frontier_mismatch': 30, 'push_not_first_candidate': 2, 'unfinished_record': 2} | {'premature_pop': 238, 'push_non_edge': 94, 'missing_or_multiple_actions': 16, 'push_revisited_node': 15, 'push_without_candidate': 15, 'wrong_action': 2, 'premature_exhaust_with_candidates': 6, 'unfinished_record': 2, 'false_found': 3} | 288 |

## context/dfs_state

| 条件 | n | 状态完全合法数 | 动作完全合法数 | 首次状态错误 | 首次动作错误 | 状态首错在record0 |
| --- | --- | --- | --- | --- | --- | --- |
| id | 192 | 135 | 139 | {'wrong_action': 16, 'candidate_mismatch': 41} | {'false_found': 3, 'premature_pop': 11, 'push_revisited_node': 9, 'push_non_edge': 5, 'missed_target': 13, 'push_without_candidate': 12} | 0 |
| id_disconnected_pad32 | 192 | 0 | 13 | {'malformed_fields': 192} | {'premature_exhaust_with_candidates': 175, 'push_non_edge': 3, 'push_without_candidate': 1} | 192 |
| wide32_branch2 | 192 | 0 | 21 | {'malformed_fields': 191, 'unfinished_record': 1} | {'premature_exhaust_with_candidates': 169, 'unfinished_record': 1, 'push_non_edge': 1} | 192 |
| wide32_densityhalf | 192 | 0 | 3 | {'malformed_fields': 192} | {'premature_exhaust_with_candidates': 189} | 192 |
| deep16_sameprompt | 448 | 141 | 156 | {'candidate_mismatch': 263, 'wrong_action': 35, 'frontier_mismatch': 9} | {'premature_pop': 256, 'push_without_candidate': 1, 'missed_target': 20, 'false_found': 15} | 0 |
| deep24 | 320 | 63 | 75 | {'candidate_mismatch': 237, 'visited_mismatch': 12, 'wrong_action': 7, 'frontier_mismatch': 1} | {'premature_pop': 111, 'push_without_candidate': 7, 'premature_exhaust_with_candidates': 106, 'push_revisited_node': 2, 'push_non_edge': 11, 'missed_target': 7, 'false_found': 1} | 127 |
| deep32 | 448 | 0 | 70 | {'malformed_fields': 448} | {'premature_exhaust_with_candidates': 378} | 448 |
| deep48 | 704 | 0 | 113 | {'malformed_fields': 704} | {'premature_exhaust_with_candidates': 591} | 704 |
| large64 | 448 | 0 | 57 | {'malformed_fields': 448} | {'push_without_candidate': 2, 'premature_exhaust_with_candidates': 370, 'push_non_edge': 19} | 448 |

## fixed/dfs_state

| 条件 | n | 状态完全合法数 | 动作完全合法数 | 首次状态错误 | 首次动作错误 | 状态首错在record0 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed0 | 64 | 37 | 37 | {'wrong_action': 27} | {'missed_target': 21, 'false_found': 6} | 0 |

## fixed/bfs_state

| 条件 | n | 状态完全合法数 | 动作完全合法数 | 首次状态错误 | 首次动作错误 | 状态首错在record0 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed0 | 64 | 35 | 35 | {'wrong_action': 26, 'candidate_mismatch': 2, 'visited_mismatch': 1} | {'false_found': 6, 'missed_target': 21, 'push_without_candidate': 1, 'push_revisited_node': 1} | 1 |

## fixed/dfs_random

| 条件 | n | 状态完全合法数 | 动作完全合法数 | 首次状态错误 | 首次动作错误 | 状态首错在record0 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed0 | 64 | 62 | 63 | {'candidate_mismatch': 1, 'wrong_action': 1} | {'false_found': 1} | 0 |

## 可核对的具体例子

### context/bfs_state / id / candidate_mismatch / missing_eligible

```json
{
  "model": "context/bfs_state",
  "case": "id",
  "index": 62268,
  "hop": 1,
  "label": 0,
  "pred": 0,
  "strict_failure": {
    "record": 15,
    "type": "candidate_mismatch",
    "n_records": 19,
    "model_record": "<V109> <V106> <V198> <V185> | <V91> <V150> <V255> <V177> <V60> <V130> <V109> <V106> <V198> <V185> | | <POP>",
    "expected_frontier": "<V109> <V106> <V198> <V185>",
    "expected_visited": "<V91> <V150> <V255> <V177> <V60> <V130> <V109> <V106> <V198> <V185>",
    "eligible": "<V144>",
    "subtypes": [
      "missing_eligible"
    ],
    "missing_candidates": "<V144>",
    "shown_candidates": ""
  },
  "skeleton_failure": {
    "type": "premature_pop",
    "record": 15,
    "current": "<V109>",
    "expected_action": "<PUSH>",
    "actual_action": "<POP>",
    "model_record": "<V109> <V106> <V198> <V185> | <V91> <V150> <V255> <V177> <V60> <V130> <V109> <V106> <V198> <V185> | | <POP>",
    "expected_frontier": "<V109> <V106> <V198> <V185>",
    "expected_visited": "<V91> <V150> <V255> <V177> <V60> <V130> <V109> <V106> <V198> <V185>",
    "eligible": "<V144>"
  }
}
```

### context/bfs_state / id_disconnected_pad32 / visited_mismatch / duplicate

```json
{
  "model": "context/bfs_state",
  "case": "id_disconnected_pad32",
  "index": 62324,
  "hop": 1,
  "label": 0,
  "pred": 0,
  "strict_failure": {
    "record": 2,
    "type": "visited_mismatch",
    "n_records": 3,
    "model_record": "<V132> | <V70> <V132> <V132> | | <EXHAUST>",
    "expected_frontier": "<V132>",
    "expected_visited": "<V70> <V132>",
    "eligible": "",
    "subtypes": [
      "duplicate"
    ]
  },
  "skeleton_failure": null
}
```

### context/bfs_state / id_disconnected_pad32 / visited_mismatch / extra

```json
{
  "model": "context/bfs_state",
  "case": "id_disconnected_pad32",
  "index": 61794,
  "hop": 1,
  "label": 0,
  "pred": 0,
  "strict_failure": {
    "record": 2,
    "type": "visited_mismatch",
    "n_records": 7,
    "model_record": "<V56> <V236> <V125> | <V56> <V236> <V54> <V125> | <V5> | <PUSH> <V5>",
    "expected_frontier": "<V56> <V236> <V125>",
    "expected_visited": "<V56> <V236> <V125>",
    "eligible": "<V5>",
    "subtypes": [
      "extra"
    ]
  },
  "skeleton_failure": null
}
```

### context/bfs_state / deep16_sameprompt / candidate_mismatch / missing_eligible

```json
{
  "model": "context/bfs_state",
  "case": "deep16_sameprompt",
  "index": 1,
  "hop": 1,
  "label": 0,
  "pred": 0,
  "strict_failure": {
    "record": 9,
    "type": "candidate_mismatch",
    "n_records": 15,
    "model_record": "<V52> <V34> | <V188> <V223> <V127> <V218> <V52> <V34> | <V101> | <PUSH> <V101>",
    "expected_frontier": "<V52> <V34>",
    "expected_visited": "<V188> <V223> <V127> <V218> <V52> <V34>",
    "eligible": "<V101> <V183>",
    "subtypes": [
      "missing_eligible"
    ],
    "missing_candidates": "<V183>",
    "shown_candidates": "<V101>"
  },
  "skeleton_failure": {
    "type": "premature_pop",
    "record": 10,
    "current": "<V52>",
    "expected_action": "<PUSH>",
    "actual_action": "<POP>",
    "model_record": "<V52> <V34> <V101> | <V188> <V223> <V127> <V218> <V52> <V34> <V101> | | <POP>",
    "expected_frontier": "<V52> <V34> <V101>",
    "expected_visited": "<V188> <V223> <V127> <V218> <V52> <V34> <V101>",
    "eligible": "<V183>"
  }
}
```

### context/dfs_state / id_disconnected_pad32 / malformed_fields / fields_5

```json
{
  "model": "context/dfs_state",
  "case": "id_disconnected_pad32",
  "index": 61794,
  "hop": 1,
  "label": 0,
  "pred": 0,
  "strict_failure": {
    "record": 0,
    "type": "malformed_fields",
    "n_records": 1,
    "model_record": "<V240> | | <V59> | | <EXHAUST>",
    "expected_frontier": "<V56>",
    "expected_visited": "<V56>",
    "eligible": "<V5> <V125> <V236>",
    "subtypes": [
      "fields_5"
    ]
  },
  "skeleton_failure": {
    "type": "premature_exhaust_with_candidates",
    "record": 0,
    "current": "<V56>",
    "expected_action": "<PUSH>",
    "actual_action": "<EXHAUST>",
    "model_record": "<V240> | | <V59> | | <EXHAUST>",
    "expected_frontier": "<V56>",
    "expected_visited": "<V56>",
    "eligible": "<V5> <V125> <V236>"
  }
}
```

### fixed/dfs_state / fixed0 / wrong_action / wrong_action

```json
{
  "model": "fixed/dfs_state",
  "case": "fixed0",
  "index": 49,
  "hop": 2,
  "label": 1,
  "pred": 0,
  "strict_failure": {
    "record": 16,
    "type": "wrong_action",
    "n_records": 19,
    "model_record": "<V232> <V29> <V178> | <V232> <V37> <V44> <V72> <V42> <V57> <V77> <V95> <V29> <V178> | | <POP>",
    "expected_frontier": "<V232> <V29> <V178>",
    "expected_visited": "<V232> <V37> <V44> <V72> <V42> <V57> <V77> <V95> <V29> <V178>",
    "eligible": ""
  },
  "skeleton_failure": {
    "type": "missed_target",
    "record": 16,
    "current": "<V178>",
    "expected_action": "<FOUND>",
    "actual_action": "<POP>",
    "model_record": "<V232> <V29> <V178> | <V232> <V37> <V44> <V72> <V42> <V57> <V77> <V95> <V29> <V178> | | <POP>",
    "expected_frontier": "<V232> <V29> <V178>",
    "expected_visited": "<V232> <V37> <V44> <V72> <V42> <V57> <V77> <V95> <V29> <V178>",
    "eligible": ""
  }
}
```

## 含义

原分布BFS的首错以漏候选为主；padding后首错转为visited维护，且大量样本第一或很早的record就出错。动作重放能进一步识别提前POP/EXHAUST、错误FOUND、重复入队或走不存在的边；它们不是只与参考排序不同。固定图普通DFS/BFS还有错误终止的问题；随机DFS的主要错误明显减少。

输入prompt原先固定76token，但CoT输出及训练总序列长度本来是可变的。固定的是图输入和查询边界，不应称为模型从未见过位置148。补长同时改变边-查询距离、无关节点数和注意力竞争，因此长度内插能检验训练支持范围，但不能单独证明绝对位置编码是原因。