# Path与节点序列错误

Path负例的“必须到sink”是当前训练生成器的规则，不是图路径本身合法的必要条件。下面另统计只要求起点正确、不重复、逐边合法的route prefix。它仍不能证明NO。若允许负例随机路径在任意位置停止，不应把negative_path_not_at_sink计为走错边。

错误码non_node_token也包含“合法词表节点，但不在本张输入图中”，不一定是输出了特殊token。JSON中的invalid_token_subtypes进一步区分了这两种情况。

## context/path_negative

| 条件 | n | 当前训练协议完整合法 | 路径前缀合法（仅Path） | 首次错误分布 |
| --- | --- | --- | --- | --- |
| id | 192 | 95.31% | 95.31% | {'valid': 183, 'non_node_token': 8, 'non_edge': 1} |
| id_disconnected_pad32 | 192 | 55.73% | 56.25% | {'non_node_token': 29, 'repeated_node': 22, 'non_edge': 33, 'valid': 107, 'wrong_positive_endpoint': 1} |
| wide32_branch2 | 192 | 39.58% | 40.62% | {'non_edge': 36, 'valid': 76, 'repeated_node': 27, 'non_node_token': 50, 'wrong_positive_endpoint': 1, 'empty_path': 1, 'missing_answer': 1} |
| wide32_densityhalf | 192 | 0.00% | 0.00% | {'empty_path': 185, 'wrong_start': 4, 'non_node_token': 3} |
| deep16_sameprompt | 448 | 50.22% | 96.65% | {'valid': 225, 'missing_answer': 17, 'negative_path_not_at_sink': 190, 'repeated_node': 8, 'non_node_token': 7, 'wrong_positive_endpoint': 1} |
| deep24 | 320 | 39.69% | 68.44% | {'valid': 127, 'negative_path_not_at_sink': 91, 'non_edge': 27, 'non_node_token': 39, 'repeated_node': 28, 'empty_path': 7, 'missing_answer': 1} |
| deep32 | 448 | 12.95% | 29.24% | {'negative_path_not_at_sink': 64, 'repeated_node': 80, 'valid': 58, 'non_edge': 70, 'non_node_token': 153, 'empty_path': 14, 'wrong_positive_endpoint': 6, 'missing_answer': 3} |
| deep48 | 704 | 0.00% | 0.00% | {'empty_path': 701, 'non_node_token': 2, 'wrong_start': 1} |
| large64 | 448 | 0.00% | 0.00% | {'empty_path': 448} |

## context/dfs_nodes

| 条件 | n | 当前训练协议完整合法 | 路径前缀合法（仅Path） | 首次错误分布 |
| --- | --- | --- | --- | --- |
| id | 192 | 4.17% | 不按路径评判 | {'missing_or_inconsistent_terminal': 149, 'not_dfs_order': 32, 'repeated_node': 2, 'valid': 8, 'non_node_token': 1} |
| id_disconnected_pad32 | 192 | 0.52% | 不按路径评判 | {'empty_nodes': 80, 'non_node_token': 85, 'missing_or_inconsistent_terminal': 21, 'not_dfs_order': 1, 'wrong_start': 4, 'valid': 1} |
| wide32_branch2 | 192 | 2.08% | 不按路径评判 | {'empty_nodes': 75, 'missing_or_inconsistent_terminal': 30, 'non_node_token': 76, 'wrong_start': 7, 'valid': 4} |
| wide32_densityhalf | 192 | 1.04% | 不按路径评判 | {'empty_nodes': 72, 'non_node_token': 85, 'wrong_start': 19, 'valid': 2, 'missing_or_inconsistent_terminal': 14} |
| deep16_sameprompt | 448 | 17.19% | 不按路径评判 | {'missing_or_inconsistent_terminal': 309, 'valid': 77, 'non_node_token': 1, 'not_dfs_order': 58, 'repeated_node': 3} |
| deep24 | 320 | 1.56% | 不按路径评判 | {'valid': 5, 'non_node_token': 139, 'empty_nodes': 115, 'missing_or_inconsistent_terminal': 48, 'wrong_start': 12, 'repeated_node': 1} |
| deep32 | 448 | 2.90% | 不按路径评判 | {'empty_nodes': 203, 'missing_or_inconsistent_terminal': 52, 'non_node_token': 164, 'valid': 13, 'wrong_start': 16} |
| deep48 | 704 | 1.99% | 不按路径评判 | {'missing_or_inconsistent_terminal': 80, 'empty_nodes': 280, 'non_node_token': 263, 'wrong_start': 67, 'valid': 14} |
| large64 | 448 | 2.23% | 不按路径评判 | {'wrong_start': 87, 'empty_nodes': 152, 'valid': 10, 'non_node_token': 162, 'missing_or_inconsistent_terminal': 37} |

## context/bfs_nodes

| 条件 | n | 当前训练协议完整合法 | 路径前缀合法（仅Path） | 首次错误分布 |
| --- | --- | --- | --- | --- |
| id | 192 | 21.88% | 不按路径评判 | {'repeated_node': 26, 'not_bfs_order': 98, 'missing_or_inconsistent_terminal': 8, 'non_node_token': 17, 'valid': 42, 'continued_after_target': 1} |
| id_disconnected_pad32 | 192 | 4.69% | 不按路径评判 | {'not_bfs_order': 74, 'non_node_token': 79, 'valid': 9, 'missing_or_inconsistent_terminal': 27, 'continued_after_target': 2, 'repeated_node': 1} |
| wide32_branch2 | 192 | 4.17% | 不按路径评判 | {'non_node_token': 80, 'not_bfs_order': 70, 'repeated_node': 6, 'missing_or_inconsistent_terminal': 25, 'valid': 8, 'continued_after_target': 3} |
| wide32_densityhalf | 192 | 0.52% | 不按路径评判 | {'not_bfs_order': 20, 'non_node_token': 77, 'missing_or_inconsistent_terminal': 79, 'valid': 1, 'continued_after_target': 4, 'wrong_start': 3, 'repeated_node': 8} |
| deep16_sameprompt | 448 | 18.97% | 不按路径评判 | {'not_bfs_order': 193, 'non_node_token': 117, 'repeated_node': 17, 'missing_or_inconsistent_terminal': 36, 'valid': 85} |
| deep24 | 320 | 5.31% | 不按路径评判 | {'repeated_node': 6, 'not_bfs_order': 137, 'non_node_token': 103, 'valid': 17, 'missing_or_inconsistent_terminal': 56, 'continued_after_target': 1} |
| deep32 | 448 | 3.57% | 不按路径评判 | {'non_node_token': 216, 'not_bfs_order': 120, 'missing_or_inconsistent_terminal': 88, 'wrong_start': 1, 'valid': 16, 'repeated_node': 3, 'continued_after_target': 4} |
| deep48 | 704 | 11.93% | 不按路径评判 | {'missing_or_inconsistent_terminal': 301, 'not_bfs_order': 113, 'non_node_token': 192, 'valid': 84, 'wrong_start': 6, 'continued_after_target': 3, 'repeated_node': 5} |
| large64 | 448 | 10.49% | 不按路径评判 | {'valid': 47, 'non_node_token': 115, 'missing_or_inconsistent_terminal': 228, 'not_bfs_order': 42, 'wrong_start': 8, 'continued_after_target': 3, 'repeated_node': 5} |

## 具体例子

```json
{
  "model": "context/path_negative",
  "case": "id",
  "index": 62330,
  "hop": 1,
  "label": 0,
  "pred": 0,
  "source": "<V221>",
  "target": "<V232>",
  "trace": "<V221> <V248> <V85> <V62>",
  "error": "non_node_token",
  "last_node_out_neighbors": ""
}
```

```json
{
  "model": "context/path_negative",
  "case": "deep16_sameprompt",
  "index": 3,
  "hop": 1,
  "label": 0,
  "pred": 0,
  "source": "<V27>",
  "target": "<V202>",
  "trace": "<V27> <V55> <V146> <V244>",
  "error": "negative_path_not_at_sink",
  "last_node_out_neighbors": "<V101> <V24>"
}
```

```json
{
  "model": "context/dfs_nodes",
  "case": "id",
  "index": 62324,
  "hop": 1,
  "label": 0,
  "pred": 0,
  "source": "<V70>",
  "target": "<V140>",
  "trace": "<V70> <V120>",
  "error": "not_dfs_order",
  "last_node_out_neighbors": "<V168> <V140> <V132>"
}
```

```json
{
  "model": "context/bfs_nodes",
  "case": "id",
  "index": 63197,
  "hop": 1,
  "label": 0,
  "pred": 1,
  "source": "<V122>",
  "target": "<V107>",
  "trace": "<V122> <V82> <V16> <V107>",
  "error": "not_bfs_order",
  "last_node_out_neighbors": ""
}
```
