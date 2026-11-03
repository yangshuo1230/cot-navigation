# N100 Fixed DAG：逐步轨迹合法性审计

使用简单图 all_hops 与困难图的验证集 best checkpoint。所有模型冻结，不进行训练。
主表覆盖完整正向测试池，旧图 1,091 对、新图 555 对；同图各协议使用相同 query ID。
主表为首次确定语义错误即停止的严格审计。已证明非法的前缀不会因为后续输出恢复整条合法；
提前停止行不猜测其最终答案，不将“未生成答案”记成 NO。合法率针对本次 audited greedy 轨迹。

## 全测试集结果

| 图 | 协议 | best step | 所有已输出步骤语义合法 | 完整轨迹及答案合法 | 比例 | 正例完整合法 | 负例完整合法 |
|---|---|---:|---:|---:|---:|---:|---:|
| simple | path | 20000 | 1088/1091 | 1075/1091 | 98.5% | 374/386 | 701/705 |
| simple | dfs_nodes | 30000 | 1052/1091 | 1047/1091 | 96.0% | 367/386 | 680/705 |
| simple | bfs_nodes | 15000 | 1015/1091 | 1005/1091 | 92.1% | 329/386 | 676/705 |
| simple | dfs_state | 40000 | 1036/1091 | 1018/1091 | 93.3% | 361/386 | 657/705 |
| simple | bfs_state | 55000 | 1064/1091 | 1015/1091 | 93.0% | 343/386 | 672/705 |
| hard | path | 10000 | 555/555 | 551/555 | 99.3% | 296/300 | 255/255 |
| hard | dfs_nodes | 15000 | 507/555 | 500/555 | 90.1% | 279/300 | 221/255 |
| hard | bfs_nodes | 25000 | 409/555 | 396/555 | 71.4% | 187/300 | 209/255 |
| hard | dfs_state | 30000 | 492/555 | 492/555 | 88.6% | 265/300 | 227/255 |
| hard | bfs_state | 60000 | 526/555 | 526/555 | 94.8% | 273/300 | 253/255 |

## 核心发现

不能把 Path 的高合法性推广到所有 CoT。困难图 Path 的全部已输出路径均逐边合法，完整协议合法率为 551/555；
DFS State 为 492/555，BFS State 为 526/555；DFS Node 为 500/555，BFS Node 为 396/555。
困难 BFS Node 的首错包括 105 条不符合 BFS 顺序、39 条重复节点及 2 条到目标后仍继续输出；另外 13 条未完成。
困难 DFS State 的首错包括 52 条候选集合错误、8 条动作错误、3 条栈状态错误；候选首错中 51 条发生在到达目标之前。
在候选首错位置还可直接确认 6 条非边 PUSH 和 7 条重复访问 PUSH，这两类计数可能重叠。
因此并非只有无关紧要的终点注释问题。

简单 DFS/BFS State 分别有 18/49 条记录始终合法但预算中止的轨迹；它们降低完整搜索率，不能算作某一步的语义错误。

![完整合法率](complete_legality.png)

## 判定标准

* Path：从正确起点出发，每步为真实边，无重复或非节点 token；正例实际到达目标；负例随机路径到汇节点；最终答案与任务标签一致。负例路径不是不可达证明。
* DFS/BFS Node：验证是否存在与整个扩展序列一致的合法 DFS/BFS 遍历；不错误地要求回溯后或 BFS 相邻扩展节点之间也有边。
* DFS/BFS State：从真实起点状态逐条重放。检查 frontier 顺序、visited 集合、候选集合、动作及更新、终止条件和答案。
* 语义合法允许不同合法邻居顺序；严格协议另要求 visited 为发现顺序、PUSH 选择打印候选列表的第一项。

所有已输出步骤合法与完整搜索合法分开报告。STOP 后每条已打印记录可能都合法，但搜索未完成；STOP 是旧训练中允许的预算接口，不能把它说成某一步的图语义错误。
空轨迹不计入“所有已输出步骤合法”的成功项。Path 的正确边序列未到目标、Node 的合法扩展尚未穷尽，也会出现前者高于后者。

## 首次失败类型

* simple dfs_state: candidate_mismatch=44, legal_prefix_stopped=18, wrong_action=5, overlong_record=2, visited_mismatch=2, illegal_push=1, frontier_mismatch=1
  候选首错分解：{"before_goal": 43, "missing_candidates": 39, "extra_candidates": 29, "extra_visited_candidates": 13, "extra_nonedge_candidates": 17, "nonedge_push_at_first_candidate_error": 12, "revisited_push_at_first_candidate_error": 5, "at_goal": 1}
  已检查前缀内的协议排序错误查询数：{}
* simple bfs_state: legal_prefix_stopped=49, candidate_mismatch=18, wrong_action=9
  候选首错分解：{"before_goal": 17, "missing_candidates": 16, "extra_candidates": 13, "extra_nonedge_candidates": 11, "extra_visited_candidates": 3, "at_goal": 1, "revisited_push_at_first_candidate_error": 1, "nonedge_push_at_first_candidate_error": 1}
  已检查前缀内的协议排序错误查询数：{}
* hard dfs_state: candidate_mismatch=52, wrong_action=8, frontier_mismatch=3
  候选首错分解：{"before_goal": 51, "missing_candidates": 28, "extra_candidates": 30, "extra_visited_candidates": 20, "at_goal": 1, "extra_nonedge_candidates": 10, "revisited_push_at_first_candidate_error": 7, "nonedge_push_at_first_candidate_error": 6}
  已检查前缀内的协议排序错误查询数：{}
* hard bfs_state: wrong_action=19, candidate_mismatch=9, overlong_record=1
  候选首错分解：{"before_goal": 9, "missing_candidates": 9}
  已检查前缀内的协议排序错误查询数：{}

## 动作重放诊断面板

在完整测试池中按 rank-gap × label 每桶固定抽取最多 4 条。该面板从一开始就另行运行，
忽略状态注释错误继续生成，直到动作本身非法或完整结束。此处分母不同，不能与全测试主表直接相减。

| 图 | 协议 | 面板数 | 完整状态严格合法 | 忽略注释后的动作执行合法 |
|---|---|---:|---:|---:|
| simple | dfs_state | 64 | 59/64 | 59/64 |
| simple | bfs_state | 64 | 58/64 | 58/64 |
| hard | dfs_state | 63 | 60/63 | 62/63 |
| hard | bfs_state | 63 | 60/63 | 60/63 |

## 实际失败记录

### simple dfs_state，query=159，<V50> → <V147>

首错：`{"kind": "candidate_mismatch", "record": 63, "detail": {"current": 99, "visited": [64, 57, 179, 257, 196, 27, 233, 240, 154, 178, 41, 232, 242, 81, 181, 16, 209, 143, 164, 239, 218, 79, 142, 174, 173, 259, 231, 60, 159, 267, 69, 30, 99], "expected": [129, 161, 229], "shown": [229, 179, 161], "missing": [129], "extra": [179], "shown_action": [8, 229]}}`

```text
前一条：<V50> | <V50> <V43> <V165> <V243> <V182> <V13> <V219> <V226> <V140> <V164> <V27> <V218> <V228> <V67> <V167> <V2> <V195> <V129> <V150> <V225> <V204> <V65> <V128> <V160> <V159> <V245> <V217> <V46> <V145> <V253> <V55> <V16> | <V85> | <PUSH> <V85>
出错条：<V50> <V85> | <V50> <V43> <V165> <V243> <V182> <V13> <V219> <V226> <V140> <V164> <V27> <V218> <V228> <V67> <V167> <V2> <V195> <V129> <V150> <V225> <V204> <V65> <V128> <V160> <V159> <V245> <V217> <V46> <V145> <V253> <V55> <V16> <V85> | <V215> <V165> <V147> | <PUSH> <V215>
```

### simple bfs_state，query=637，<V36> → <V132>

首错：`{"kind": "legal_prefix_stopped", "record": 95, "detail": {}}`

```text
前一条：<V46> <V191> <V67> <V31> <V140> <V148> <V225> <V204> <V129> <V145> | <V36> <V85> <V147> <V165> <V115> <V215> <V255> <V236> <V243> <V164> <V238> <V104> <V74> <V98> <V182> <V15> <V159> <V27> <V103> <V94> <V60> <V13> <V223> <V142> <V93> <V218> <V245> <V224> <V221> <V124> <V219> <V206> <V217> <V189> <V228> <V96> <V57> <V176> <V226> <V89> <V76> <V150> <V46> <V191> <V67> <V31> <V140> <V148> <V225> <V204> <V129> <V145> | <V55> | <PUSH> <V55>
出错条：<V46> <V191> <V67> <V31> <V140> <V148> <V225> <V204> <V129> <V145> <V55> | <V36> <V85> <V147> <V165> <V115> <V215> <V255> <V236> <V243> <V164> <V238> <V104> <V74> <V98> <V182> <V15> <V159> <V27> <V103> <V94> <V60> <V13> <V223> <V142> <V93> <V218> <V245> <V224> <V221> <V124> <V219> <V206> <V217> <V189> <V228> <V96> <V57> <V176> <V226> <V89> <V76> <V150> <V46> <V191> <V67> <V31> <V140> <V148> <V225> <V204> <V129> <V145> <V55> | | <POP>
```

### hard dfs_state，query=312，<V194> → <V15>

首错：`{"kind": "candidate_mismatch", "record": 41, "detail": {"current": 238, "visited": [208, 238, 203, 45, 133, 209, 181, 35, 76, 34, 267, 142, 16, 164, 218, 162, 154, 171, 55, 30, 90, 60], "expected": [156, 168, 174, 232], "shown": [168, 156], "missing": [174, 232], "extra": [], "shown_action": [8, 168]}}`

```text
前一条：<V194> <V224> <V189> | <V194> <V224> <V189> <V31> <V119> <V195> <V167> <V21> <V62> <V20> <V253> <V128> <V2> <V150> <V204> <V148> <V140> <V157> <V41> <V16> <V76> <V46> | | <POP>
出错条：<V194> <V224> | <V194> <V224> <V189> <V31> <V119> <V195> <V167> <V21> <V62> <V20> <V253> <V128> <V2> <V150> <V204> <V148> <V140> <V157> <V41> <V16> <V76> <V46> | <V154> <V142> | <PUSH> <V154>
```

### hard bfs_state，query=330，<V134> → <V147>

首错：`{"kind": "wrong_action", "record": 15, "detail": {"current": 161, "expected": "FOUND", "shown": [8, 222]}}`

```text
前一条：<V74> <V147> <V115> <V204> <V248> <V177> <V27> <V215> <V15> | <V134> <V85> <V132> <V74> <V147> <V115> <V204> <V248> <V177> <V27> <V215> <V15> | | <POP>
出错条：<V147> <V115> <V204> <V248> <V177> <V27> <V215> <V15> | <V134> <V85> <V132> <V74> <V147> <V115> <V204> <V248> <V177> <V27> <V215> <V15> | <V208> <V217> <V91> <V165> | <PUSH> <V208>
```

## 已确认的非边 PUSH 实例

困难 DFS State，query=758，查询 <V147> → <V239>，第 86 条记录：

```text
当前节点：<V217>
真实出边：<V244>
真实未访问后继：空
模型候选：<V132>, <V91>, <V208>
模型动作：PUSH <V132>
```

图中不存在 <V217> → <V132>。本例在非法动作后停止，不推断它后续的最终答案。

## 验证、效率及解释边界

检查器通过 1,536 个全部 4 节点 DAG 的参考查询/搜索案例，并测试错误答案、候选缺失与非边 PUSH。
另与独立 Node 扩展序列判定器进行了 24,576 个小图语言案例的等价性检查；不把合法回溯或 BFS 兄弟跳转误判为路径非边。
与原生 BF16 greedy 的前缀对照覆盖四个 state checkpoint，共八个查询。
仅在审计进程中将 KV 访问裁到可见前缀，去掉原实现中被遮罩的未来缓存槽；权重、上下文上限和训练代码不变。
优化后的实际生成前缀与上述原生对照一致。本次批量为 32，原生训练末评测常为 16；浮点内核与分批可能影响少数近似并列 token。
这些测试是执行合法性审计，不是与某一参考字符串比较，也不是因果证明模型真的用 CoT 计算答案。
简单图与困难图的训练数据、WORLD 前缀和 checkpoint 步数不同；困难 BFS State 可能仍在训练，使用本次读入时的 best 权重。
所有 checkpoint 从一次完整文件读取中冻结，SHA256、步数及逐查询首错都已保存。
提前停止导致后续错误分布不可知；不报告被截断样本的全轨迹逐记录错误率，以避免产生虚假的高“每步正确率”。
