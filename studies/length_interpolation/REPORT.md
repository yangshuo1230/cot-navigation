# 轨迹错误与prompt长度内插：已执行的控制实验

本次在原50k checkpoint基础上完成6个独立5k续训（BFS/DFS×short/long/endpoints），另对两个原checkpoint做相同集合的冻结对照。每组最终评估1760条，共14080条自由生成。没有根据test选择checkpoint；均用预先指定的最终5k权重。没有修改主训练框架、原始数据集或原批次输出。

## 实验针对的假设

原训练prompt固定76token，但CoT输出和总训练序列长度本来就不固定。要检验的是固定图输入规模/查询边界的训练支持范围，而不是“模型从没在148这个位置计算过”。边列表补长同时改变相对位置、无关节点数和注意力竞争；本实验不单独证明某个位置编码机制是原因。

保持查询组件16节点24边、原节点名、原边的相对顺序、查询、标签及整条监督轨迹完全不变；只加入不相连的16节点24边组件并交错边列表。每个组件增加72个prompt token，没有新增CoT token，也不是Filler。

- short：只训练76token，无额外组件。
- long：只训练220token，即额外两个组件。
- endpoints：每个逻辑batch严格32条76token+32条220token。
- 测试148token是额外一个组件，两种训练端点均不包含它；292token是额外三个组件，用于外推。

每组追加5000步×64查询=320000次相同查询曝光；原邻居重排RNG及最终query sampler状态哈希已核对相同；同一协议三组父权重SHA256相同。重置优化器，AdamW/SDPA/bf16/torch.compile沿用原实现，LR=1e-4，前100步warmup。旧训练序列与缓存不修改；新prompt的loss mask整体正确平移。CPU预检48条通过且没有保存smoke记录或checkpoint。

不同组的输入token/FLOPs不同，不是等计算量实验；long比endpoints平均输入更长，可以帮助判断改善是否仅来自混合短长。新图seed仍为单个，续训不等价于从零训练长度混合模型。

## 首次错误细分

详见ERROR_ANALYSIS.md和error_breakdown.json。基于上一轮的同一192条配对ID/padding样本：

- 原分布BFS43条状态非法，其中38条首次错误包含漏合法候选；动作首次错误包括25次过早POP、10次PUSH不存在的边、4次重复PUSH、4次无候选却PUSH。
- padding后BFS首次错误144次为visited不一致，其中127次含重复节点；另30次为候选错误。动作首次错误包括67次过早POP、21次走不存在的边等。
- padding后DFS192/192在第一条状态就出现5字段而非4字段的记录；175/192还在有候选时直接EXHAUST。它不是先正确执行很长再逐步积累误差。
- 固定图seed0普通DFS的64条抽样，动作错误为21次到了目标却未FOUND，以及6次未到目标就FOUND；普通BFS也有相同数量的这两类错误，另有两次PUSH错误。这里主要是目标/终止判断，不等同于新图padding的格式错误。

其他形式也要区分“走错”与“没完成”：新图DFS节点序列192例中，149例是终止不合适、32例不是合法DFS展开顺序；BFS节点序列有98例不满足BFS展开顺序、26例重复节点。Path负例在同prompt长度的8层图中，严格按生成器走到sink仅50.22%合法，但若只要求路径前缀逐边正确则96.65%合法；这类短负路径不应直接当成走错边。相关例子和较宽松指标见NONSTATE_ERRORS.md。错误码non_node_token也可能是生成了本图没有的节点，而非特殊token。

## 未见148长度的内插结果

| 模型 | 训练方式 | n | 148答案正确 | 148动作骨架合法 | 148完整状态合法 |
| --- | --- | --- | --- | --- | --- |
| bfs_state | 原50k模型 | 96 | 88.54% | 43.75% | 15.62% |
| bfs_state | 续训仅76 | 96 | 90.62% | 44.79% | 15.62% |
| bfs_state | 续训仅220 | 96 | 95.83% | 77.08% | 76.04% |
| bfs_state | 续训76/220各半 | 96 | 94.79% | 77.08% | 77.08% |
| dfs_state | 原50k模型 | 96 | 50.00% | 12.50% | 0.00% |
| dfs_state | 续训仅76 | 96 | 48.96% | 10.42% | 0.00% |
| dfs_state | 续训仅220 | 96 | 94.79% | 86.46% | 84.38% |
| dfs_state | 续训76/220各半 | 96 | 95.83% | 84.38% | 83.33% |


每组96条完全相同的held-out查询，正负均衡、hop1..3各32条。不能与上一轮192条或完整10000条test直接相减。

长度覆盖显著缓解骨架与状态错误，尤其DFS。固定长训练也能向较短148泛化，因此不能把改善归功于“两个端点混合”这一项本身。还需要看它是否损伤原76输入：

| 模型 | 方式 | 76答案 | 76动作骨架 | 76完整状态 |
| --- | --- | --- | --- | --- |
| bfs_state | 原50k模型 | 93.75% | 76.04% | 76.04% |
| bfs_state | 续训仅76 | 93.75% | 79.17% | 79.17% |
| bfs_state | 续训仅220 | 95.83% | 78.12% | 75.00% |
| bfs_state | 续训76/220各半 | 94.79% | 76.04% | 76.04% |
| dfs_state | 原50k模型 | 89.58% | 81.25% | 78.12% |
| dfs_state | 续训仅76 | 94.79% | 88.54% | 87.50% |
| dfs_state | 续训仅220 | 76.04% | 53.12% | 29.17% |
| dfs_state | 续训76/220各半 | 94.79% | 87.50% | 86.46% |


DFS仅长训练在76输入上完整状态合法率降至29.17%，端点混合保留86.46%，因此混合长度对避免短长度行为遗忘有实际价值。BFS两组短长度的表现接近，没有看到同样严重的遗忘。

## 真正改变图结构/路径深度的结果

| 条件 | BFS短对照答案/骨架 | BFS端点答案/骨架 | DFS短对照答案/骨架 | DFS端点答案/骨架 |
| --- | --- | --- | --- | --- |
| 实际加宽32/48/3 | 84.38% / 37.50% | 95.83% / 48.96% | 48.96% / 15.62% | 91.67% / 67.71% |
| 实际加宽加密32/96/3 | 56.25% / 0.00% | 60.42% / 10.42% | 50.00% / 2.08% | 76.04% / 26.04% |
| 同长度加深16/24/7 | 83.48% / 57.14% | 82.59% / 57.14% | 71.43% / 37.95% | 71.88% / 38.39% |
| 加深24/40/5 | 78.75% / 28.12% | 88.12% / 38.75% | 72.50% / 25.62% | 76.88% / 31.25% |
| 加深32/56/7 | 69.20% / 25.45% | 79.02% / 36.16% | 50.00% / 16.52% | 69.64% / 33.48% |
| 加深48/88/11 | 58.81% / 17.90% | 71.59% / 30.97% | 48.58% / 14.20% | 62.78% / 27.84% |
| 加宽加深64/112/7 | 61.16% / 12.95% | 69.20% / 27.68% | 50.00% / 12.05% | 64.29% / 25.45% |


完整四组结果见ALL_CONDITIONS.md，逐hop见BY_HOP.md。所有组仍仅训练查询hop1..3，并没有偷偷加入长路径监督。因此真大图/新增hop的改善是迁移，但若仍失败，说明输入长度覆盖不能替代更复杂状态和更长搜索的训练。

尤其同prompt76的8层图：BFS短对照答案83.48%/骨架57.14%，端点组82.59%/57.14%；DFS也基本不变。这是“长度覆盖不能解决所有长路径问题”的直接对照。真实48节点11跳条件下，BFS端点组答案71.59%，但骨架仅30.97%，还不能称为稳定搜索。

## 内插集的状态错误分布

| 模型 | 方式 | 首次状态错误分布 | 首错在record0的条数 |
| --- | --- | --- | --- |
| bfs_state | 原50k模型 | {'visited_mismatch': 67, 'candidate_mismatch': 13, 'malformed_fields': 1} | 13 |
| bfs_state | 续训仅76 | {'visited_mismatch': 66, 'candidate_mismatch': 15} | 14 |
| bfs_state | 续训仅220 | {'candidate_mismatch': 22, 'visited_mismatch': 1} | 0 |
| bfs_state | 续训76/220各半 | {'candidate_mismatch': 22} | 0 |
| dfs_state | 原50k模型 | {'malformed_fields': 96} | 96 |
| dfs_state | 续训仅76 | {'malformed_fields': 96} | 96 |
| dfs_state | 续训仅220 | {'candidate_mismatch': 14, 'wrong_action': 1} | 1 |
| dfs_state | 续训76/220各半 | {'candidate_mismatch': 13, 'wrong_action': 3} | 0 |


BFS固定短对照的visited首错66/96，端点组为0/96；DFS固定短组第一条就格式错误96/96，端点组格式首错为0/96。剩余错误主要回到候选集合或动作/终止判断，而不是查询一结束就进入错误格式。

## 当前可以支持的结论

固定短prompt/查询边界的训练分布是一个重要失败来源：相同额外步数、相同查询和边顺序下，短长度对照不能修复内插骨架，而长长度暴露与端点混合均显著修复。端点混合还能保留DFS短输入能力。

但这不证明全部错误都由位置问题造成。补长同时提供了无关子图干扰训练，且各组FLOPs不同；候选集合、状态规模和超出3跳的执行仍有独立困难。单seed、96例内插集的探索性结果需要重复验证，尤其BFS答案提高几个百分点不应当成精确效应量；骨架和完整状态的提升更大。

附加KV-cache检查：在prompt76/148/268/292各抽查33个预测位置，FP32完整前向与逐token cache前向的argmax全部一致；最大logit差约4.7e-4，最小top2间隔大于1.5。未发现这些例子存在明显的cache长度索引错误，记录见cache_validation.json；不声称这是穷尽的实现正确性证明。

## 结果文件、W&B与复现

W&B group：length_interp_v1_20261017。六个正式训练run已使用原项目的在线配置，训练损失和长度评估指标上传，未上传checkpoint artifact。

- [bfs_state/short](https://wandb.ai/yangshuo1230-sjtu-hpc-center/cot-graph-reasoning/runs/7e77e60bd98b407f89624f343a66ea13)
- [bfs_state/long](https://wandb.ai/yangshuo1230-sjtu-hpc-center/cot-graph-reasoning/runs/e3d0985ab49b43e7aac67b067968bf16)
- [bfs_state/endpoints](https://wandb.ai/yangshuo1230-sjtu-hpc-center/cot-graph-reasoning/runs/f6aca276206a48c0bc20518a832f8a5e)
- [dfs_state/short](https://wandb.ai/yangshuo1230-sjtu-hpc-center/cot-graph-reasoning/runs/39c5d03c76b349e79e71a5598222e63f)
- [dfs_state/long](https://wandb.ai/yangshuo1230-sjtu-hpc-center/cot-graph-reasoning/runs/0e10a01a76c045a491526f9688381ec4)
- [dfs_state/endpoints](https://wandb.ai/yangshuo1230-sjtu-hpc-center/cot-graph-reasoning/runs/16952bfeb1f24cf78ff881b4f96a1a56)

服务器目录research/length_interpolation_20261017/runs/PROTOCOL/MODE/含experiment.json、resolved_config.yaml、checkpoint、生成结果和完成标记。checkpoint中的step5000是追加阶段步数，父checkpoint50000在experiment.json中单独标记。

从项目根目录执行，例如：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python research/length_interpolation_20261017/experiment.py --protocol bfs_state --mode endpoints --steps 5000
```

这会拒绝覆盖已有checkpoint；训练恢复可传--resume PATH。当前完整结果不需要重跑。新重复应使用新output/seed配置，不能把相同seed重跑当独立重复。

本地原始快照evidence.json、汇总summary.json、错误细分ERROR_ANALYSIS.md、图interpolation.png。短输入对照用于排除仅追加训练步数这一解释；更多seed和从零训练的覆盖对照尚未执行。
