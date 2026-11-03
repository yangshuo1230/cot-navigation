# 祖先方向：embedding 与 L1 attention 后的因果干预

结论：在目标 embedding 中移植祖先探针子空间，可产生超过等范数随机方向的定向答案变化；但同时改变节点身份等信息，尚不能单独归因为祖先语义。L1 attention 后 rank32 的祖先移植则在两张图上都只改变约0.8%的答案，即使独立审计解码器明显转向供体；补空间移植几乎完全改变答案。留出节点分支中的强效应与打乱标签对照接近，不能声称特异祖先因果中介。

## 设置与干预位置

- 仅旧 N100 简单/困难固定图 Direct 验证最佳 checkpoint；未分析最新 N160 限制暴露模型。checkpoint 哈希沿用并核验。
- 输入为原 QUERY prompt，只修改目标位置一次；起点不变。embedding 通过输入位置 hook 替换，权重和绑定输出头逐位确认不变。
- L1 attention 后表示 attention 残差相加后、norm2/FFN之前。通过修改attention增量实现目标残差替换，FFN输入和skip残差同时获得新状态；不是只改LayerNorm输入。后续所有计算重新执行。
- 使用旧 crossed-square 测试面板，供体为同起点换目标的自然查询 b=(s,t′)，答案相反。固定原四问均正确子集：简单238对、困难119对；不按干预结果筛选。配对共享节点与查询，不是独立模型重复。
- 祖先标签为严格祖先集合，排除自身。构造方向的probe与独立审计probe用原query-train的不相交两半拟合，各自只用原query-val选正则。query-only分支二者仍共享节点身份，不能把这种独立称为节点独立。
- 另做target节点60/20/20划分（seed17/31/47），与原query split相交；构造probe和审计probe均未训练过干预/供体目标节点；二者都必须为test-node，且沿用同一原面板。身份审计单独使用全query-train拟合，以使节点身份保留可以测量。
- 扫描rank1/2/4/8/16/32/64，rank32为与旧实验对齐的展示切片，rank64为较宽子空间补充；没有根据test选最优rank。

## 干预公式

设原状态h、同起点供体状态hd、训练均值mu、祖先解码器左奇异向量投影P：
- 祖先移植：h + P(hd-h)。
- 补空间移植：h + (I-P)(hd-h)。
- 祖先擦除：h - P(h-mu)，将该子空间设置为训练均值。
- 全替换hd和no-op为正/负控制。三seed随机方向、三seed打乱目标→祖先标签的方向，逐样本匹配祖先移植或擦除的编辑L2范数；未强制保留编辑后状态总范数，JSON中保留其变化。

## 原查询划分：答案定向改变

|图/位置|rank|祖先移植|补空间移植|随机方向均值|打乱标签均值|全替换|
|---|---:|---:|---:|---:|---:|---:|
|simple/embedding|32|45.4%|52.5%|15.7%|3.8%|100.0%|
|simple/embedding|64|78.2%|21.8%|53.9%|42.6%|100.0%|
|simple/L1_attention|32|0.8%|99.6%|3.2%|0.4%|99.2%|
|simple/L1_attention|64|2.9%|97.1%|16.8%|2.5%|99.2%|
|hard/embedding|32|26.9%|72.3%|7.8%|1.1%|100.0%|
|hard/embedding|64|70.6%|24.4%|55.7%|38.9%|100.0%|
|hard/L1_attention|32|0.8%|99.2%|0.8%|0.0%|100.0%|
|hard/L1_attention|64|26.9%|71.4%|18.8%|8.1%|100.0%|

## 原查询划分：擦除后的原答案保留率

|图/位置|rank|祖先擦除|等范数随机擦除|打乱标签擦除|
|---|---:|---:|---:|---:|
|simple/embedding|32|84.5%|96.4%|97.8%|
|simple/embedding|64|63.9%|81.5%|85.3%|
|simple/L1_attention|32|100.0%|99.9%|99.7%|
|simple/L1_attention|64|98.7%|97.3%|98.9%|
|hard/embedding|32|97.5%|97.8%|99.2%|
|hard/embedding|64|74.8%|87.1%|87.4%|
|hard/L1_attention|32|100.0%|99.4%|100.0%|
|hard/L1_attention|64|95.0%|98.3%|96.9%|

## rank32 的独立审计与身份混杂

|图/位置|祖先分数向供体进展|供体祖先集合F1|起点祖先归属按反事实分类正确|原目标身份保留|模型答案改变|
|---|---:|---:|---:|---:|---:|
|simple/embedding|94.9%|64.8%|79.4%|21.8%|45.4%|
|simple/L1_attention|94.1%|78.2%|88.7%|53.8%|0.8%|
|hard/embedding|95.6%|75.8%|81.5%|43.7%|26.9%|
|hard/L1_attention|94.4%|96.9%|95.8%|24.4%|0.8%|

分数进展是沿“审计probe供体分数−原分数”的投影系数；1为完成自然供体差值，允许超过1或为负，不是百分比恢复率或解释方差。集合F1使用val选阈值，排除原目标、新目标、起点；起点祖先归属另单独审计。二者不使用相同候选分母。

在L1 attention后，rank32擦除将审计解码器的起点祖先归属判断推离原标签，但两模型仍全部保留原答案。它说明此解码器可见方向并非可靠的单一必要中介，而非证明全部祖先信息被清除。

## 留出目标节点分支：支持量与打乱控制

|图/位置|节点seed|配对数|祖先移植rank32|随机均值|打乱标签均值|有效W rank|祖先/打乱子空间重叠|
|---|---:|---:|---:|---:|---:|---:|---:|
|simple/embedding|17|14|78.6%|14.3%|47.6%|50|0.688|
|simple/embedding|31|0|—|—|—|—|—|
|simple/embedding|47|6|66.7%|27.8%|61.1%|42|0.716|
|simple/L1_attention|17|14|7.1%|4.8%|4.8%|50|0.697|
|simple/L1_attention|31|0|—|—|—|—|—|
|simple/L1_attention|47|6|0.0%|27.8%|11.1%|42|0.729|
|hard/embedding|17|23|65.2%|31.9%|65.2%|47|0.752|
|hard/embedding|31|7|57.1%|19.0%|57.1%|46|0.748|
|hard/embedding|47|0|—|—|—|—|—|
|hard/L1_attention|17|23|95.7%|40.6%|95.7%|47|0.979|
|hard/L1_attention|31|7|85.7%|19.0%|85.7%|46|0.972|
|hard/L1_attention|47|0|—|—|—|—|—|

这里配对数仅6–23对，且部分seed没有合格配对，不能做稳定跨划分推断。特别是困难图L1 attention后，留出节点的祖先移植成功率很高，但打乱标签方向可复现相同结果，故不具有祖先语义特异性。
重叠定义为 ||U_ancestor^T U_shuffled||_F² / rank；1表示两个投影子空间相同，随机同rank子空间期望约为rank/128。标签、节点身份及样本行空间可导致不同语义probe共享方向。rank超过有效解码器rank时还会包含近零奇异值方向，应结合JSON中的奇异值谱解释，不能把所有选中方向称作祖先语义。

## 验证、实例与解释边界

观察hook/no-op不改变模型结果；embedding完整替换与自然更换目标的查询logits对齐；attention位置核验实际FFN输入与所请求的残差一致；全局embedding与绑定输出头未修改。每个位置的独立审计还检查直接后继、可达后代、深度读出进展与源节点身份副本，详细数值见JSON。

- simple：{"observer_margin_error": 0, "embedding": {"noop_margin_error": 0.0, "patched_residual_error": 0, "full_swap_counterfactual_success": 1.0, "n": 238}, "L1_attention": {"noop_margin_error": 0.0, "patched_residual_error": 5.960464477539063e-08, "full_swap_counterfactual_success": 0.9915966386554622, "n": 238}, "embedding_and_tied_head_unchanged": true}
- hard：{"observer_margin_error": 0, "embedding": {"noop_margin_error": 0.0, "patched_residual_error": 0, "full_swap_counterfactual_success": 1.0, "n": 119}, "L1_attention": {"noop_margin_error": 0.0, "patched_residual_error": 9.5367431640625e-07, "full_swap_counterfactual_success": 1.0, "n": 119}, "embedding_and_tied_head_unchanged": true}

不能把embedding干预有效表述为“证明祖先集合被用于答题”：目标身份已明显被破坏，其他关系特征也可能同时改变；对照支持的是相关方向的部分因果影响。不能把attention后干预阴性表述为“祖先未被使用”：可能有冗余、非线性读法、未清除方向，且L1分隔符已经从原embedding读过信息。

未停止正在进行的训练；冻结前向CPU FP32/2线程，短时ridge拟合GPU0。仅保存simple.json、hard.json、关键PNG/PDF、报告和复用脚本；没有activation、probe权重、模型checkpoint或W&B run。
