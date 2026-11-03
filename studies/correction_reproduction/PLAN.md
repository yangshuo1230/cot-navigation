# 纠错训练复现方案，待确认（2026-10-28）

**当前仅准备和审计，未 launch。** 本轮先研究最近分析的 in-context DFS state。BFS 不混入首轮，避免同时改变算法、纠错比例和轨迹权重，使重复次数不足。

## 研究问题和分组

固定历史自然错误30%；每条样本的整段 CoT loss 根据该样本 CoT 长度单独赋权，范围1–16。每组从头训练4个随机种子0/1/2/3，共12个正式训练。

|组别|训练内容|MISTAKE 词表|错误池查询重采样|
|---|---|---|---|
|original_clean|原始正确 DFS 轨迹|无|无|
|matched_clean|正确 DFS 轨迹|有，词表与纠错组相同|有；与纠错组逐次相同查询、同一参考变体|
|correction|70%正常轨迹，30%自然错误+MISTAKE+正确重写轨迹|有|有，沿用历史机制|

主要比较 **correction − matched_clean**，检验在控制词表、初始权重和查询曝光后，错误—重写监督是否仍有收益。次要比较 correction − original_clean，复现此前实际观察到的总收益。matched_clean − original_clean 只反映词表/初始化和查询重采样的合并差异，不能单独分解这两者。

matched_clean 不是将错误样本丢弃或换成另外的随机正确问题，而是保留纠错组采中的那个图、起终点和 natural_variant，返回其原始正确轨迹。选择复杂错误和选择 episode 的随机数消耗完全保留。正确轨迹在启动时读入 RAM；训练中无图搜索、错误挖掘或磁盘加载。

30%指训练 example 的采样概率，不是错误 token 占比；一条自然错误 episode 可能有多个错误。仅 correction 真正含错误 token；matched_clean 的配置中仍保留0.3选择概率，以复用完全相同的采样分支，但返回的是无错轨迹。运行入口和 checkpoint 均记录 arm，避免混淆。

## 固定设置

- 模型：6层、1 head、d_model128、FFN512、RoPE、4K上下文；BF16、SDPA、编译模型和 loss、fused AdamW auto。
- 优化：每个模型200,000 optimizer updates；逻辑 batch64；学习率1e-3→1e-4、warmup100、weight decay0.01；每样本轨迹权重1–16（下述长度函数）。两组正确/错误 token 均按既有 full-loss 目标监督，不加入 masking 新变量。
- 课程：原6个图尺度 N16/M24、N24/M36、N32/M48、N36/M54、N64/M96、N100/M150；4段各50k步，混合权重与原纠错实验完全一致。
- 每对 matched_clean/correction 采用相同 model_seed、exposure_seed 和错误 episode 采样种子；同一 seed 初始权重逐字节相同。seed之间同时改变这三类训练随机性，报告的是训练重复波动，不是仅初始化波动。
- 原始图数据、4种正确轨迹变体、donor自然错误池均固定复用，错误挖掘种子固定0。**不重新挖掘、不人工扩大错误种类**，先复现当前方法。错误池是旧干净 donor 的真实候选错误，不使用此次待训练模型或测试准确率挑选训练样本。
- 此次沿用当前训练器/加速实现，执行代码和数据记录 SHA256；不声称与旧软件实现逐位一致。
- 训练步数和查询 batch 相同；纠错轨迹更长，token量、计算量和长序列截断不强行等同。这是复现旧设置的比较，不是等计算量实验。

### 每条 CoT 的长度权重（更正）

对第i条训练样本，令Lᵢ为**该样本实际输入监督中的CoT token数**，不计图/查询prompt、最终答案、EOS和padding。先求该样本原有trace监督位置上的平均CE，再给这整段平均loss赋权。相同batch内的不同样本使用不同λ；权重与训练步数、课程阶段和图节点数无关。

默认取 `λᵢ = clip(Lᵢ/128, 1, 16)`；128作为短轨迹长度单位，接近已有最小尺度正确轨迹平均长度约115 token的整数量级。它是可修改的预设，不是最优超参数的结论；不使用测试成绩选择阈值。函数写入 `trace_weight_policy.json`。

|单条CoT长度L|该条整段trace均值loss的相对权重λ|
|---|---:|
|≤128 token|1|
|256 token|2|
|512 token|4|
|1024 token|8|
|≥2048 token|16|

因此即使训练进入后期，短CoT仍使用低权重；早期碰到长CoT也按其真实长度赋权。整条纠错CoT中的错误记录、MISTAKE和正确重写都属于该CoT，均计入长度。纠错轨迹较长时，其权重可以高于同图的干净参考轨迹；这是当前“实际整条CoT长度”定义的结果，不能再声称两者逐样本λ相同。

保留原逐样本归一化公式 `ℓᵢ = 2*(answer CEᵢ + λᵢ*mean(trace CEᵢ))/(1+λᵢ)`，最后对逻辑batch取平均。λ=1时答案和轨迹均值等权；λ=16时相对权重16:1，总系数保持2。整个长度计算、逐样本系数和CE归约在编译loss内完成；本地日志修正为实际逐样本加权项的平均值，W&B显示batch平均λ，不以“batch平均λ×batch平均loss”冒充真实加权loss。checkpoint记录完整长度函数并在恢复时核对。

由于目标函数从固定16改为逐样本长度权重，结果应称为“长度权重设置下的多种子复现”，不再声称逐项复刻历史固定16实验。原全局课程权重调度已移除。

## 评估与收益判定

主结果固定 **last 200k**，不在各 seed 间挑最好的 checkpoint。验证每20k步一次；checkpoint每10k步保存，保留最近3个及已有best/last；W&B每100步记录。验证 free-generation64个查询、teacher-forced128个查询，只用验证集观察学习曲线。

最终在 N32、N64、普通 N100、窄深 N100、真实困难 DAG N100 五套固定测试集各评估512个查询。困难 DAG 是迁移测试，不混入困难图训练 baseline。所有组使用相同查询、顺序、batch16、原生4K生成预算和 STOP 机制，避免此前批量变化导致的 BF16 解码差异。

困难 DAG 的正确参考使用已有32K缓存，生成仍为4K；这不是放宽纠错模型上下文。提前检查缓存可读和输入长度，避免训练完成后才因参考缓存不匹配而失败。

关键结果：答案准确率、语义 CoT 合法率、候选/visited错误查询比例、MISTAKE/修复/误报比例、预算中断与强制STOP比例。语义合法率按真实图重放，接受不同合法邻居顺序，并允许 MISTAKE 取消错误记录；不用与唯一参考轨迹的逐token匹配率替代。预算耗尽案例仍计入结果。

每组报告全部4个seed、均值、标准差、每seed配对差值和胜出seed数量。跨seed不确定性与固定checkpoint下查询bootstrap区间分开；不把数千个测试查询当作数千次独立训练重复。普通N100为主终点，其余为预先指定的泛化诊断。只有原始baseline比较有收益、匹配对照比较没有收益时，应优先怀疑词表/初始化或曝光差异；少数seed收益大但多数无收益时，不称为稳定提高。

本轮固定图生成种子和错误池，先回答训练随机性下是否复现；不涵盖不同图数据种子或不同donor错误池的不确定性。4个seed仍有限，边缘差异不作强显著性结论。

## 产物、效率和启动保护

不改共享训练器、历史配置、历史数据或历史 checkpoint。研究目录 `research/correction_reproduction_20261028/` 中包含 `suite.py`、`worker.py`、`audit.py`、`review_plan.json`、`data_audit.json`。12份正式配置在 `configs/generated/correction_reproduction/dfs_correction_repro_20261028/`。

所有默认命令均不训练：

```bash
cd ~/cot-navigation
PYTHONPATH=src .venv/bin/python research/correction_reproduction_20261028/suite.py
PYTHONPATH=src .venv/bin/python research/correction_reproduction_20261028/audit.py
```

未来确认后才使用以下启动命令（**尚未执行**）：

```bash
PYTHONPATH=src .venv/bin/python research/correction_reproduction_20261028/suite.py --launch --background --wait
```

计划首批GPU0–3跑4个matched_clean，GPU4–7跑4个correction；原始baseline4个自动排队。后台入口自动建目录。启动前再次核对代码SHA256、已审阅的配置与数据审计；发生改变则拒绝启动。恢复需同一入口加 `--resume`，不要改用通用CLI替代匹配对照入口。

W&B沿用原项目 `cot-graph-reasoning`，统一group `dfs_correction_repro_20261028`，按组别与seed命名。每run默认显示答案loss、轨迹loss、验证答案准确率、语义CoT合法率、当前轨迹权重5张图；其他分解保存本地指标。预检查只检查凭据可用，不创建W&B run；上传是否成功要等正式启动时验证。

不保存激活、数据副本或全量生成轨迹，只保存正式训练checkpoint、紧凑逐查询预测、汇总及少量轨迹预览。没有进行训练或保存任何 smoke run。
