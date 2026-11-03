# In-context DAG prefix attention 实验

已获用户授权准备并后台启动。服务器代码：`/mnt/ningxuefei/yangshuo/cot-navigation/research/incontext_prefix_suite_20261024`。输出：`/mnt/ningxuefei/yangshuo/cot-navigation/outputs/incontext_prefix_v1`。W&B group：`incontext_prefix_v1_20261024`。

## 39 次训练

每个条件seed0/1/2。Direct：standard/stepwise × causal/graph/query，共6条件。PATH：单路causal/graph/query、parallel graph/query、serial graph/query，共7条件。合计13×3=39。主课程对照共享数据/曝光seed；双路仅改变结构，不宣称与单路FLOPs相同。

`graph`只使QUERY之前的图token互相可见，图不读取QUERY；`query`使完整图+QUERY/source/target/输入分隔符双向；整个输入始终不能读输出，输出内部始终因果。双路64+64维，总attention投影参数与单路128维一致。并行、串行参数集合一致。图+query模式下KV属于当前查询，更换查询需重新prefill。

Direct预备550k，standard为普通简单N16/M24全跳，stepwise为历史source混合一跳课程的固定步数回放：200k/30k/30k source受限比例80%/50%/20%，随后完整query v16/v64/v256 M2为30k/50k/30k；M4/M8/M16/M24为50k/40k/30k/60k。不是原有门控逻辑的逐行复现。

共同主课程360k：简单N16/36/64/100各40k，再困难N16/36/64各40k、N10080k。逻辑batch64，第一主阶段全当前池，后续48当前+16已完成主阶段回放，回放轮转，逐batch标签32/32。Direct专属预备池不加入共同回放。Direct总910k、PATH总360k，跨协议成绩不是等总预算比较。

## 复用的优化pipeline

独立快照当前`cot_navigation`源码，新增PrefixLM和Trainer适配器；不改写共享仓库源码或旧实验。复用现有Trainer的length-aware microbatch、65536 token微批上限、30% padding阈值、pin-memory非阻塞传输、BF16、SDPA、torch.compile、编译masked loss、fused AdamW、独立full/prefill/decode编译入口与KV移行。预备阶段用GPU池和纯答案CE。主课程使用原有答案+轨迹按例归一化loss，PATH lambda1，无额外辅助目标。

骨干6L/d128/FFN512、RoPE、dropout0；AdamW betas(.9,.99)、weight_decay.01、clip1。桶对齐16。上下文4096。实际词表为271（270个既有token加现有pipeline的STOP）；保留原节点ID及STOP协议，以复用已审计离线轨迹。这是相对初稿270词表的明确实现适配，所有组一致。

预备LR3e-4→3e-5，主课程1e-4→1e-5；各阶段warmup1000，前90%更新完成余弦衰减，末10%维持floor。预备→主课程保留权重、重建优化器；不重置数据和增强RNG之外的主曝光流。主采样器从独立seed开始，保证不同预备经历不会改变共同主课程曝光。

## 数据与评估

复用已有8个50k训练池以及2个N144外推池；实际路径见manifest。逐根校验已有cache完成校验和；PATH的hard缓存原按32k生成，但实际所有行都能完整放进4096-margin，只读复用而不截断或重生成。配置包括4个离线路径变体，负例随机合法游走监督与旧scale pipeline一致。

预备一跳池来自`research/direct_learning_20261020/data_v1`，已重新穷举核对一跳边标签和train/val/test图身份不交。主图池重新独立遍历抽样核对标签，原有完整审计仍为来源。三个模型seed共享数据，不是三个独立数据生成seed。

日志每100；checkpoint每10k及结束；teacher-forced和原生自由生成每20k，主阶段边界补评；主验证256例。Direct预备验证当前课程的完整query验证池（不是只测固定source子任务）。只报告last主测试，关闭best选择用于主比较。

最终简单N100、困难N100各512；N144简单/困难各256。主测试另做反转/随机边序，以及source/target/单边图编辑的配对标签翻转（无可行翻转的样本明确跳过并报告eligible数）。所有标签由实际修改图独立搜索确认，生成不使用oracle。PATH实际生成按输入边列表独立检查完整有效witness和正确YES联合率，不能以答案正确代替轨迹有效。

每次生成保存prompt和完整输出；可后续核查attention、边界载体与逐层图编码。图信息probe和冻结mask消融属于后续分析，不混入训练目标。推理mask消融有分布偏移，训练mask的配对对照是主要因果证据。

## 启动前验证

- `verification.json`：7种实际模型结构，CPU未来输出无泄漏、query对图的依赖范围、padding、native causal等价、参数量/初始化匹配；GPU编译训练损失/梯度、prefill/decode及变长前缀KV移行。
- `runtime_verification.json`：Direct预备、Direct进入主课程后、PATH主课程实际更新的内存中断/恢复，下一步权重差均0；没有落盘测试checkpoint。
- `bootstrap_verification.json`：逐步Direct的10个预备阶段均进行临时实际更新。
- `benchmark_single.json`和`benchmark_serial.json`：通过正式Trainer适配器做临时性能更新；不创建W&B/正式run/训练checkpoint。

## 初步时间估算

按A800实测更新耗时：单路主阶段约10–31ms，串行双路约12–40ms，Direct普通预备约6.6ms。由未测中间规模插值、用串行作为双路代理、用普通M24作为Direct预备代理，估计纯更新约115 GPU小时；当前队列8卡模拟约15.7小时。含编译/验证/保存/并行争用等预留，初估墙钟18–26小时。不是完成时间承诺，需随正式日志更新，详细假设见time_estimate.json。

后台启动器`start.py`拒绝重复receipt；调度器只使用确认空闲GPU，39任务逐卡排队。worker失败则停止新派发，保留其他在跑任务。中断后自动从latest恢复需按原launch队列重新运行调度器；不静默覆盖既有权重。launch_receipt和source_hashes保存在输出根目录。
