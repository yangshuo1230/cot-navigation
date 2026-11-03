# scale 错误探针工作记录

任务：在图 scale 后的 checkpoint 重跑最后输入 token 错误识别，扩大人工数据并比较 Linear/MLP。

- 选择范围：scale_v1_s0 新图/固定图，l6(N36)、l8(N64)、l10(N100)、curriculum(N100)，每种 Path/DFS/BFS，共24个设置。
- DFS/BFS 从state/nodes按记录的验证best_score选择，同分优先state；Path直接读取best；没有test挑选。全部40个候选记录保留。
- scale固定图只有seed0；probe初始化为0/1/2，不替代图seed重复。
- 新模块读取 ScalePreparedRows 实际四种离线轨迹，按index%4取样。保留原WORLD映射、STOP/cot_margin；不修改旧源文件或结果。
- CPU独立输入审计：N100新图prompt长454，固定图WORLD_9且prompt长5；两个任务train/val/test共6个样本与原始缓存token完全一致。
- 节点版支持DFS回溯、BFS队列父节点顺序，独立穷举4节点DAG所有优先级，4950个合法前缀的下一节点集合一致。
- 新增5个测试，连同既有27个分析测试共32个通过。
- 训练目标每模型8192人工对，val2048/test4096对；按查询隔离、query逆频率加权。自然首次错误只外部测试。Linear/MLP各3种子，L4/L5/L6及匹配token基线。
- 物理GPU0，导航模型冻结。不保存activation/probe权重，不创建W&B。服务器上其他用户/线程的GPU进程保持不动，不能把墙钟耗时当作独占设备基准。

服务器根目录：`/mnt/ningxuefei/yangshuo/cot-navigation`。
正式分析入口：`python -m cot_navigation.analysis.scale_error_awareness --resume`。
结果路径：`research/scale_error_awareness_20261019/results`。
日志：`research/scale_error_awareness_20261019/run.log`。
首次启动PID：659814；发现枚举所有替换的CPU开销后仅停止该分析进程，原有4个已完成设置留作重放对照。
优化后正式重跑PID：664683。

优化重跑完成7个设置后，将剩余17个设置分成三个互不重叠的worker，均在物理GPU0运行。核心分析源码、设置和随机数种子不变，保留已完成结果；并行计划见服务器parallel_plan.json。监督进程PID667320，最终校验进程PID667321。仅停止了本研究自己的顺序进程及监视进程，未改动其他任务。

抽样优化：把每个slot的非法替换列表压成虚拟扁平索引，使用与展开列表相同的shuffle RNG，然后仅构造实际抽中的tuple。7个真实checkpoint各2种子共14项样本完全一致；低层单元测试同时检查RNG状态。N64 DFS在12查询抽样检查中约5–9倍加速。抽样不改变，正式结果全部重新计算，避免混用源码指纹。

最终检查使用 `verify_scale_error_awareness.py`，关键结果仅生成两张PNG和紧凑JSON/CSV/报告。

完成操作：等待24个models JSON及finalize.log的FINALIZED；自动最终流程会再次运行32项测试、检查原checkpoint哈希/数据隔离/causal mask，并生成报告与manifest。然后同步results、manifest，以及reference_sampler_source.py到本地provenance，按manifest逐项校验。不要重启已完成的实验，不要修改核心分析源码的指纹。

已记录的重要异常：N100新图Path自然总体MLP AUROC约0.639，但图内节点子集仅0.401，图外/格式子集约0.985；不能把总体分数当成语义导航错误检测能力。固定图N36节点版较高的自然读出目前仅8–11对样本，另有协议/训练步数差异，不能解释为纯规模效应。

完成状态：24/24个checkpoint、每个6组probe fit均已完成；32项测试通过；24个原checkpoint哈希不变、查询划分不相交、全部future audit delta=0。14项真实采样等价检查和4个完整设置的拟合重放逐项一致；三个worker均核对只在物理GPU0。两张图经原分辨率视觉检查，44个结果/源码文件本地与服务器SHA256全部匹配。结果约4.1MB，无activation/probe checkpoint文件。全部本研究进程已退出，检查时GPU0显存0MiB。

局部积极信号：固定图课程BFS的51对自然图内节点错误，MLP L6 AUROC0.638，对照0.527；总体自然AUROC0.494。保留子集信号，不声称完整自然错误识别或模型内生错误监控。完整指标、配对胜率、概率差和具体漏报例子见results/REPORT.md。
