# Fixed DAG 暴露控制与一跳 grokking：正式批次 v2

状态：已按用户要求停止，本轮训练输出、结果报告、图表和44条W&B记录已删除。以下仅保留原配置说明，不表示实验仍在运行。

按用户最终确认执行：六张卡跑固定预算的36个暴露任务；两张卡仅在“一跳QUERY＋知识”达到判据后才接入多跳QUERY。旧fixed_grokking_20261023的short3长训练矩阵不启动。

## 资源和预算

- GPU0–5：3张图×2个模型seed×Direct/PATH×R0/R2/R3＝36个任务，每个固定125000次optimizer更新。共享知识阶段共6个，每个1000步，先在六卡执行。
- GPU6：Direct，残差dropout=0，AdamW weight decay=0.1。
- GPU7：Direct，残差dropout=0.1，AdamW weight decay=0.1。
- 两个grokking任务共享graph0/model_seed0知识检查点；不复用已经训练过多跳QUERY的short3模型。阶段一、阶段二均无最大步数，按掌握判据切换/结束。
- 调度器仅占用空闲卡，不停止其他进程；GPU6/7不借给固定任务。任一任务异常会停止新增派发并记录，已经运行的任务不被强杀。

## 数据与暴露

已准备3张独立的160节点、320边、10层×16宽DAG，边仅连接相邻层；节点名随机。选择160节点以保证每图有足量严格测试项，而不是重复少量查询凑数。全图可达性由独立遍历与反拓扑DP交叉检查。

固定任务的共同QUERY训练范围为1–2跳，起点/终点交叉块B×B留出；主测试固定为6–9跳，共256条（128YES/128NO）。每图另留64条同支持监控项。训练PATH正例只用固定合法witness，负例为空路径后NO，不生成会泄漏长关系的负例随机游走。

所有分支的知识阶段只含单边WALK。知识只监督边转移，不监督起点/EOS，不向模型提供测试标签；非边判断由QUERY监督学习。

- R0：任务阶段知识仍只有单边转移。
- R2：额外允许最长4边的路径片段。
- R3：额外提供完整主测试/监控正例witness，明确作为已暴露上界，不能称未见关系泛化。

三张图的R0/R2主测试完整witness暴露数都为0；R3为128/128。审计涵盖训练QUERY正标签、所有允许片段的祖先—后代关系与训练PATH子路径，不只查字符串相等。R0主测试的最低宏关系拼接距离为3–8，R2为2–3，R3为1，逐图计数保存在manifest。

每步64条QUERY，正负各半；知识损失目标预算固定为64个边转移，按转移token归一化，其中至少16个来自单边知识。不同片段条件的序列数和真实token数不同，不能声称计算量、所有单边频率完全相同；实际计数和有效token均记录。

## Grokking 阶段

阶段一仅训练相邻层一跳QUERY＋单边知识，不接入任何多跳QUERY或多跳WALK。相邻层的可达性等于直接邻接，因此一跳负例不会误标为不可达。

一跳QUERY按标签60%训练、20%监控、20%最终留出。graph0分别为1382/460/462条；监控/最终各有64个正例，其边都已在知识语料中提供，但对应QUERY不参与训练。已有一跳QUERY全部训练的历史设置不用于此处。

每20k步用固定训练面板与固定监控集评估。连续三次同时满足：

- 训练面板答案准确率≥99%；
- 监控集balanced accuracy≥95%；
- YES/NO recall分别≥95%。

达到后保存one_hop_grokking.pt，记录独立一跳最终留出成绩，立即进入多跳阶段；不会等待另一张卡。此判据表示掌握，是否存在严格的延迟泛化要看训练/监控曲线，不能因文件名叫grokking就断言出现了典型grokking。

阶段二：75% QUERY来自2–9跳训练池，25%回放原一跳训练池；知识仍为单边。沿用模型和Adam状态、不重启学习率。多跳监控集固定256条正负平衡面板，达到同样连续三次判据后结束；同样不设最大步数，最终留出不参与阶段切换。

## 训练、日志与恢复

沿用当前普通Transformer：6层、1头、hidden128、FFN512、RoPE、tied embedding。所有真实序列均≤16 token；max_seq_len=256留给自由生成错误余量，非长搜索协议。初始warmup1000步，随后固定LR=1e-4。W&B online，project=cot-graph-reasoning，group=fixed_exposure_grokking_v2。

每100步训练日志、每10000步checkpoint、每20000步验证。每次保存模型、优化器、Torch/CUDA/采样器RNG、阶段、判据历史、实际QUERY与知识片段计数；CPU测试已验证中断恢复后的权重和曝光与连续运行一致。dropout只在attention/FFN残差分支，关闭attention内部的第二次输出dropout；评估关闭全部dropout。

相同graph/model seed下的固定任务共享知识权重；QUERY和知识使用独立RNG，验证不抽训练样本流。各条件的QUERY计数应一致；知识片段暴露本就是实验处理，计数分别保存。缺失/格式错误不强制改成NO；最终答案与轨迹有效性应分别分析。

## 运行位置

服务器项目：`/mnt/ningxuefei/yangshuo/cot-navigation`

- 实验代码：`research/fixed_exposure_suite_20261023/`
- 正式输出：`outputs/exposure_grokking_v2/`
- 调度状态：`outputs/exposure_grokking_v2/status.json`
- 调度日志：`outputs/exposure_grokking_v2/launcher.log`
- 后台启动回执：`outputs/exposure_grokking_v2/launch_receipt.json`

launch.py默认只打印计划，--execute启动调度；train_suite.py默认只读检查指定任务。正式启动通过start_background.py使用独立会话，不依赖当前SSH连接。恢复已有任务使用--resume，不覆盖历史目录。

check_suite.py覆盖：图与标签独立核验、划分不相交、暴露边界、知识监督预算、loss/梯度加权、CPU前后向、自动阶段切换、切换前无多跳QUERY、CPU精确断点恢复。A800上已完成不保存正式模型的GPU短基准；没有把短基准称为完成正式训练。
