# 自然错误 token 即时纠错长训

项目：`/mnt/ningxuefei/yangshuo/cot-navigation-unified`。
组名：`node_token_correction_20261029`。

24 个独立训练：DFS/BFS-node × simple/hard × seed0/1/2 × repair/matched_clean。
复用上一组九级课程 `12→16→24→32→36→48→64→80→100`：前八级各100k更新，N100为200k，总计1M；70%当前规模、30%旧规模回放。模型、batch64、loss权重、AdamW、全程cosine及编译训练配置保持上一组。

两组均增加同一个 `<MISTAKE>` token（词表272），相同seed初始化相同，使用相同自然错误查询池、同样的30%查询回放与边重排。repair额外插入错误token与标记；matched_clean使用该查询的原始正确轨迹。这样控制词表和查询重复曝光的影响。

## 数据与语义

复用原始图与四变体token缓存。每个seed/协议/家族用其历史50k模型作冻结donor，checkpoint复制到新项目内，运行不依赖旧框架代码。

在每级训练集上，给donor正确参考前缀，读取其greedy一步预测。只保留真实输出且可确定非法的节点：错误起点、重复访问，以及与已生成前缀不相容的下一节点。合法的另一种邻居顺序会被过滤。这里是**正确前缀条件下的一步自然误预测**，不是自由运行的完整错误轨迹。

每级最多收集正负各256个事件；不足时扫描完整50k训练查询，至少各8个才启动训练。每个事件保存query_id、错误位置、实际预测、正确token、错误类型、donor与数据root。验证/测试查询不进入池。两实验臂共享同一个池与采样随机流；后续模型seed的池由其worker按同样规则在后台自动准备。

样本形式：`正确前缀 错误节点 <MISTAKE> 正确节点 正确后缀`，错误节点后立即是标记。错误节点的loss屏蔽；标记、正确节点和后缀仍受监督。训练优化器更新中只进行内存索引与token拼装，不运行donor或图搜索。自然错误池只在训练前准备一次并留存，断点重启复用。

自由生成时不强制插入标记，由模型自己输出。评估保留原始预测，额外按 `<MISTAKE>` 撤销前一个token后进行DFS/BFS合法性检查；孤立标记判非法，答案仍独立评分。每20k验证、每10k保存checkpoint，每阶段保存并检查旧规模能力，最终仅last在N100测试512题。

## 实现与验证

新增 `studies/node_token_correction/experiment.py`，复用原生调度器和上一组 FineNodeTrainer。共享框架仅对配置校验做小幅扩展：允许node协议使用MISTAKE及屏蔽真实错误token；完整state协议的原校验规则保持不变。修改前副本保存在本目录 `config.py.before`。

`verification.json` 检查了错误/标记/正确token的紧邻关系、loss mask、答案位置以及撤销后精确恢复匹配干净样本。自然数据生成和训练中还保留了即时断言。

输出：`outputs/node_token_correction_20261029/`。
后台启动入口：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation-unified
.venv/bin/python studies/node_token_correction/experiment.py --execute --wait --background --resume
```

当前调度PID及启动状态见 `launch_receipt.json`。首轮准备因一个小规模池不足而中止，已扩展合法性检测并重新启动；未产生需丢弃的正式训练checkpoint，已完成的自然事件池复用。
