# 自然错误训练与自然测试探针

**8/8完成。所有主训练、验证、主测试均为未经编辑的自由生成前缀；人工错误仅为末尾受控测试。**

读取已消费token位置p的各层残差，预测截至p是否已出现语义错误。各层×记录×token偏移参数独立；线性/MLP各2探针seed，4模型seed；验证BCE独立选步。

|架构|测试|位置|匹配干净|纠错|
|---|---|---|---:|---:|
|linear|共同自然测试|首错记录|0.7015|0.7474|
|linear|共同自然测试|后续第1条|0.5927|0.5836|
|linear|共同自然测试|后续第2条|0.5643|0.5702|
|linear|自身自然测试|首错记录|0.6830|0.7287|
|linear|自身自然测试|后续第1条|0.5967|0.5944|
|linear|自身自然测试|后续第2条|0.5997|0.5697|
|mlp64|共同自然测试|首错记录|0.6877|0.7443|
|mlp64|共同自然测试|后续第1条|0.5703|0.5643|
|mlp64|共同自然测试|后续第2条|0.5514|0.5556|
|mlp64|自身自然测试|首错记录|0.6631|0.7280|
|mlp64|自身自然测试|后续第1条|0.5838|0.5799|
|mlp64|自身自然测试|后续第2条|0.5812|0.5569|

## 结论与逐层检查

首错记录的共同自然测试中，纠错模型L6 MLP平均AUROC更高，且4个模型seed均高于各自匹配对照。第4–5层的提升也明显。后续记录仍有中层错误相关信息，不能根据最后一层接近随机就说所有层都已失去信息；纠错模型在后续记录没有一致优势。

|记录|层|匹配干净MLP|纠错MLP|
|---|---|---:|---:|
|首错记录|L4|0.6729|0.7902|
|首错记录|L5|0.6619|0.7916|
|首错记录|L6|0.6877|0.7443|
|后续第1条|L4|0.6747|0.6532|
|后续第1条|L5|0.6629|0.6336|
|后续第1条|L6|0.5703|0.5643|
|后续第2条|L4|0.6295|0.6217|
|后续第2条|L5|0.6282|0.6211|
|后续第2条|L6|0.5514|0.5556|

自然训练后MLP在中层略有增益，但L6没有全面超过线性探针。先前人工训练、自然donor外测的结果不再作为主判据；本轮同时更换了训练分布与自然测试轨迹，不能把新旧差异单独归因于一个因素。

人工类型控制使用同一批64个测试查询、相同原始记录：图外/非邻接/已访问/遗漏/重复候选的纠错模型首错记录读出较强；visited遗漏较弱，栈顺序接近随机。本次所选1090个自然错误训练查询中，栈顺序首错只有1个，visited遗漏首错为0，不能据此断言模型完全不编码这些错误。

## 自然测试的主要对照（第6层MLP，首错记录）

|读出或测试子集|匹配干净|纠错|
|---|---:|---:|
|完整自然共享测试|0.6877|0.7443|
|严格匹配当前token的自然子集|0.6550|0.7277|
|当前token+位置/字段对照|0.5071|0.5095|
|历史CoT embedding均值对照|0.5221|0.5195|
|标签随机翻转对照|0.5013|0.4899|

## 动作区域读出（含动作前分隔符、操作符、参数）

|位置|匹配干净L6 MLP|纠错L6 MLP|
|---|---:|---:|
|首错记录|0.5968|0.7205|
|后续第1条|0.5328|0.5400|
|后续第2条|0.5648|0.5630|

## 数据和对照

共同测试给不同冻结模型输入完全相同的自然前缀；自身测试只使用该checkpoint自己的自由生成错误。后者模型间错误难度不同，不能独立作为共同输入上能力的比较。

导航查询均为可达正样本；探针的负标签仅指截至当前位置CoT合法，不是不可达导航问题。正标签来自已错的自然前缀；负标签来自另一个仍合法的自然前缀，不经过修复或参考轨迹拼接。每对匹配生成checkpoint、语法字段、记录内token偏移，优先匹配当前token并尽量接近推理位置。测试另汇报exact-token子集和token/位置、历史embedding均值控制。

自然文本用原生BF16推理生成，冻结模型以FP32重放原样前缀提取残差，避免批量BF16数值差异变成探针线索；不声称逐位复现生成时的BF16隐藏状态。

原始查询和带节点名图哈希在train/val/test间互斥；同图多checkpoint轨迹仍为同一个独立图。只保留MISTAKE出现前的主读出，因此后续位置缺失也反映该筛选，不能解释为信息已经遗忘。

自然续写可能继续产生错误或传播之前错误，因而后续记录可解码不能直接证明存在独立的“历史错误记忆”。人工控制保留参考续写，只作补充敏感性分析。

人工控制的类型在自然训练数据中实际出现过；使用相同原始图、查询、参考记录改变对应字段。人工控制与自然测试分布不同，类型敏感性不是自然发生频率。

## 采样统计

```json
{
  "train": {
    "pairs": 49528,
    "exact_token": 45182,
    "absolute_progress_gap": 3877641,
    "unmatched_tokens": 1192,
    "positive_queries": 1090,
    "positive_sources": {
      "correction_s0": 482,
      "matched_clean_s0": 608
    },
    "first_types": {
      "candidate_out_of_graph": 571,
      "candidate_omission": 104,
      "candidate_visited": 67,
      "candidate_nonedge": 337,
      "candidate_duplicate": 9,
      "frontier_wrong_order": 1,
      "action_wrong_operator": 1
    },
    "exact_token_fraction": 0.9122516556291391,
    "mean_progress_gap": 78.29189549345824
  },
  "val": {
    "pairs": 8505,
    "exact_token": 5088,
    "absolute_progress_gap": 799742,
    "unmatched_tokens": 100,
    "positive_queries": 189,
    "positive_sources": {
      "matched_clean_s0": 113,
      "correction_s0": 76
    },
    "first_types": {
      "candidate_nonedge": 76,
      "candidate_out_of_graph": 84,
      "candidate_visited": 14,
      "candidate_omission": 13,
      "frontier_wrong_order": 1,
      "candidate_duplicate": 1
    },
    "exact_token_fraction": 0.5982363315696649,
    "mean_progress_gap": 94.03198118753674
  },
  "test": {
    "pairs": 9901,
    "exact_token": 6213,
    "absolute_progress_gap": 879082,
    "unmatched_tokens": 183,
    "positive_queries": 228,
    "positive_sources": {
      "correction_s1": 33,
      "matched_clean_s1": 32,
      "correction_s3": 35,
      "matched_clean_s3": 37,
      "matched_clean_s2": 22,
      "matched_clean_s0": 30,
      "correction_s0": 27,
      "correction_s2": 12
    },
    "first_types": {
      "candidate_omission": 45,
      "candidate_visited": 39,
      "candidate_nonedge": 41,
      "candidate_out_of_graph": 101,
      "candidate_duplicate": 1,
      "visited_out_of_graph": 1
    },
    "exact_token_fraction": 0.6275123724876275,
    "mean_progress_gap": 88.78719321280678
  }
}
```

每模型原始结果保存逐位置、各字段、自然首错类型、人工错误类型和自身自然测试的细分统计；图表只保留3张关键图。

独立语义检查：5632条自然前缀首错位置全部一致；所有自然读出的标签、生成源、字段、偏移、原样token和MISTAKE前截断通过校验；checkpoint保持冻结。

![主结果](figures/01_natural_primary.png)

![逐层逐token](figures/02_natural_token_layers.png)

![末尾类型控制](figures/03_error_type_controls.png)
