# 完整邻接表知识实验：adj_v3

本轮按用户要求仅替换知识表示；从头重训知识阶段，不使用已取消v2的检查点或恢复其输出。

## 完整邻接表格式

非空表：`<EDGE> u | v1 v2 ... <EOS>`。

空表：`<EDGE> u | <EXHAUST> <EOS>`。

`<EDGE>`在本实验中是ADJ标记，`<EXHAUST>`表示空后继集合，均复用原词表已有token，模型参数量与初始化方式不变。后继按随机化后的节点token ID排序，不按隐藏拓扑编号排序。全部后继或EMPTY及EOS均参与知识损失，输入标记、源节点和分隔符不参与损失。不会把邻居之间的共现误作链上边。

所有160个节点都进入知识语料，包括空后继节点。只使用完整表，绝不截断。通过整表装箱保持每步恰好64个知识监督位置，按实际监督token归一化；64个位置现在包含EOS/EMPTY，不再全部是正边转移。这是替换知识形式带来的直接变化，不能声称与旧WALK有相同边曝光频率。

## 其余设置保持

- 三张图、节点命名、QUERY划分和R0/R2/R3额外路径池均与取消批次逐字节相同，`design_comparison.json`记录核验。
- 3图×2初始化×Direct/PATH×R0/R2/R3＝36个固定任务，每个125k步；6个共享知识阶段仍各1000步。
- GPU0–5按原队列运行；GPU6/7分别Direct dropout0/0.1、weight decay均0.1。
- 两个grokking任务先只训练一跳QUERY＋完整邻接表；连续三次满足训练99%、监控BA95%、两类recall95%后加入2–9跳QUERY，保留25%一跳回放。两个阶段均无步数上限。
- QUERY逻辑batch64、LR=1e-4、warmup1000、损失权重、评估20k、保存10k、日志100、模型6层/1头/d128/FFN512/RoPE/tied embedding均不变。

R0知识阶段及任务回放只有完整邻接表。R2/R3先采样恰好16个监督位置的完整邻接表，再保留原先48个转移位置的额外WALK片段：R2最长4跳，R3包含测试正例完整witness。保留这些WALK是为了不改变暴露对照处理；共同知识阶段和局部知识回放均已替换。

## 诊断

新增完整邻接表的自由生成评估，逐节点从ADJ提示生成至EOS；检查真实完整集合、空集合、格式和顺序。不给真实出度，不使用top-k。知识阶段结束及每次任务验证均记录该指标，grokking切换判据仍仅沿用原QUERY判据，没有额外增加门槛。

## 输出与启动

服务器代码：`research/fixed_adjacency_suite_20261024/`

新输出：`outputs/exposure_grokking_adj_v3/`

W&B组：`fixed_exposure_grokking_adj_v3`，project仍为`cot-graph-reasoning`。

`start_background.py`后台启动调度器；`read_status.py`只读状态。旧v2保持取消，不恢复已删除结果。

CPU检查覆盖完整表/EMPTY/EOS监督、精确64位置预算、原有图与划分审计、自动阶段切换、切换前无多跳QUERY、恢复与连续训练权重一致性。
