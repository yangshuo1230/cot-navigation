# 六张 In-Context DFS 示意图核查与 Fixed DAG 对照

结论：多个原图实例数值可以复现，但有一处source位移的明确错误，需要区分输入位置与下一token，且“写入/存储/复制”超过attention本身能证明的范围。Fixed有明显不同的具体回读路线，不能将原图作为所有任务的共同电路。

## 原图来源和范围

- 原始脚本 scripts/plot_dfs_field_flow.py；原图数据模型为 outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state/checkpoints/best.pt，step185000；示例来自 data/scale_v1/context_l4w4_n16_m24_train50000_s0 的query50520/50512。
- 这是4×4小图上的正确canonical DFS-state参考轨迹，teacher forcing、FP32，不能解释为自由生成中的无条件表现。
- V11等名称是图内拓扑索引别名，来自node_tokens的inverse映射，不等于词表中实际叫<V11>的token。
- 找到的flow绘图脚本将数字写死，未附图注均值对应的完整样本ID清单；本次对已独立复算的例子背书，不对全部图注均值作逐位确认。新均值来自明确保存的独立面板，不冒充原图的原始统计。

## 可复现的原图实例

|query/事件|本次FP32复算|原图近似值|
|---|---:|---:|
|50520，prompt separator在L1读取query source|0.915|0.92|
|50520，同位置L4读取query source|0.786|0.79|
|50520，PUSH位置L1读取第二个候选V15|0.947|0.95|
|50520，PUSH位置L4读取第一个候选V13|0.364|0.36|
|50520，PUSH位置L5读取第三个候选V14|0.987|0.99|
|50520，预测POP的第三个分隔符在L4读取栈顶|0.834|0.83|
|50520，预测FOUND的第三个分隔符在L4读取栈顶|0.965|0.97|
|50520，预测新栈顶V13的位置在L1读取前一PUSH参数|0.974|0.97|
|50512，POP后预测保留节点V5的位置L5读取匹配的first-visited载体|0.993|0.99|

“每个层都关注不同候选”可以作为现象描述。此例实际输出V13，尽管L1最关注V15、L5最关注V14，因此不能用单层最大attention直接判定模型选择了哪个节点。

## 明确需要修正的地方

### 1. source的偏移量

原图注称L1将source复制到两个token之后，attention约0.99。按字面定义检查query50520与当前源节点有关的3条边：

|原source位置s|A_L1[s+2,s]|A_L1[s+3,s]|A_L1[d+3,d]（d为destination位置）|
|---:|---:|---:|---:|
|31|0.0158|0.9980|0.9985|
|37|0.0158|0.9971|0.9992|
|55|0.0158|0.9815|1.0000|

因此若“两token后”指s+2，数值不成立；s+3才接近0.99。destination后第三个位置的高attention确实存在。修图时应直接注明实际q、k，避免以三元组起始位置和节点位置混用offset。

### 2. 输入位置与预测位置

attention行q使用的是已经输入的token，预测token q+1。PUSH行分析的是下一节点参数，不能拿它说明模型如何决定输出PUSH。action决策应看之前第三个分隔符。

query50512的保留栈节点例子：q127输入V2，下一token为V5，L5读取匹配中介的质量0.993；q128输入V5，下一token为V9，另一匹配中介的质量也0.993。金色框如果写V5/V9，应注明这是“下一输出”，不能同时宣称框内就是attention行的输入token。

### 3. 读取不等于已证明复制/存储

L1关注某节点、L5关注那个接收位置，支持“两段读取路线”的假设；没有证明V/attention输出确实携带目标身份，也没有证明模型靠这一路完成输出。原图最好将writes/stores/copies改为attends to，或标为待因果验证的carrier假设。
原图的destination→+3箭头使用假设的信息流方向，其他箭头使用receiver→key的attention方向，应该统一或分开图例。本次新图全部采用INPUT receiver→KEY。

### 4. 不把小图样例均值推广到其他状态

本次新20-query面板上，In-Context N16读取前一PUSH参数预测新栈顶的均值为L1 76.4%，同模型N100为51.8%；原图一个深度2新栈顶例子的97.4%仍然正确。该差别至少说明需要声明栈深、查询和记录的选择条件，不能将单例/条件均值写成所有新栈顶的普遍规则。

## 新实验与可比性

- 5个设置：原In-Context课程模型分别在N16/N100上观察；原scale fixed N100，以及当前simple short3-all_hops fixed N100、hard fixed N100。均用各自原validation-best，具体路径/step/SHA在JSON。
- 每设置20个val查询估计按route/层/精确相对距离的均值；独立20个test查询统计，按标签与跨度分层。固定图使用正向查询，各输入q≤4095；未覆盖更长的后期位置。
- 与原图一致使用canonical、teacher-forced参考轨迹；不是训练时全部4个随机变体，也不是原生自由生成。
- 每query每类route最多8个事件，先按query平均再汇总；因此不同route的条件样本数不同。所有ID和计数已保存。
- FP32、冻结模型、GPU0，读取QKV计算指定行；带/不带hook输出逐元素相同。在query50520独立调用model(return_attn=True)，6层行与提取结果最大误差均为0。
- 图中注明“最高层”是描述性地选取L1–L6中该route的最高均值，不是预注册的独立显著性比较，不改变checkpoint选择。各箭头可能来自不同层，不能把数字加成一个概率分布。
- 只有注意力观察，没有训练探针或新增因果编辑；不保存大矩阵、激活、checkpoint或W&B。

## 具体路线的关键差异

|读取路线|IC N16|IC N100|Fixed scale N100|Fixed simple N100|Fixed hard N100|
|---|---:|---:|---:|---:|---:|
|后续栈起点：读prompt bar|L5 100.0%|L5 99.7%|L5 1.3%|L5 3.7%|L2 1.3%|
|后续栈起点：读历史同名源节点|L2 4.1%|L2 3.4%|L3 42.2%|L6 98.3%|L4 95.1%|
|保留栈节点：读匹配first-visited载体|L5 95.4%|L5 89.0%|L3 0.4%|L1 0.7%|L5 0.3%|
|保留栈节点：读历史中同名节点|L2 4.1%|L5 6.7%|L6 99.2%|L6 93.9%|L4 98.3%|
|后续visited节点：读历史visited中同名节点|L6 0.4%|L5 0.4%|L6 79.9%|L6 74.4%|L4 67.3%|
|PUSH参数：读实际选中的候选|L1 82.4%|L1 71.2%|L3 99.6%|L6 99.6%|L4 99.2%|
|回到已扩展节点生成CAND：读上次PUSH及参数|L5 93.9%|L5 90.8%|L5 24.0%|L1 1.2%|L6 6.9%|

重点是位置选择差异，不是直接认定谁会搜索：IC强读first-visited中介，Fixed强读历史同名节点。Fixed也使用显式CoT历史，这并不支持“Fixed完全不执行任何状态更新、只输出已记住的整条轨迹”的说法；同样不能用attention大小证明这份状态内容被功能性使用。

## 距离校正补查

|route/模型|原始质量最高层|同层每key距离校正残差|
|---|---:|---:|
|context_n16 / edge_dest_shift3|L1 99.9%|-0.0001|
|context_n100 / edge_dest_shift3|L1 100.0%|+0.0001|
|context_n16 / stack_kept_predict_matching_carrier|L5 95.4%|+0.5561|
|fixed_simple_n100 / stack_kept_history_same|L6 93.9%|+0.0824|
|fixed_hard_n100 / stack_kept_history_same|L4 98.3%|+0.0906|
|fixed_simple_n100 / visited_next_history_same|L6 74.4%|+0.4004|
|fixed_hard_n100 / visited_next_history_same|L4 67.3%|+0.3514|

图内L1的destination+3几乎完全固定在距离3，按同route精确距离扣均值后接近0；因此它本身更像稳定的格式/距离读取，不能单独当成图语义选择性。它之后的内容是否被使用，需要另做V或路径干预。Fixed对历史同名节点的读取、IC对匹配载体的读取在此经验距离校正后仍有正富集，但这也不能消除全部身份、格式与位置混杂。

## 新图文件

- [六页对照图PDF](dfs_routes_comparison.pdf)
- [1 候选列表](01_candidates.png)
- [2 PUSH选择](02_push_choice.png)
- [3 栈起点与新栈顶](03_stack_source_and_push.png)
- [4 POP后保留栈节点](04_retained_stack.png)
- [5 visited生成](05_visited.png)
- [6 POP/FOUND动作](06_actions.png)

三个图列是IC N16、Fixed simple、Fixed hard；原scale fixed及IC N100的补充结果在上表和JSON。主图未把不同图规模、知识来源、训练预算解释为单因素效应。

对原图的结论应保留为：在指定模型、指定正确参考轨迹上，观察到了若干稳定的attention路由；source位移需修正，carrier“存储内容”及其因果作用仍待验证。
