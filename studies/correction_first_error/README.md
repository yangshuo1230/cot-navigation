# 自然首错前的候选状态分析

复用60%纠错数据的l6/l9 s0/s1 latest200k与上一实验缓存自然轨迹，无新rollout。
主分析聚焦候选字段首错；其他字段首错计入覆盖统计，不冒充已分析。

读取首错token尚未输入的activation，并比较同轨迹前1/2条正确记录的候选入口。
控制从不同query的完整正确记录选取，要求同为需输出节点或需关闭列表，按动作序号、
栈深、visited数、已列候选数、合法剩余数、前token匹配；成本>4不匹配，记录成本和覆盖。
前序记录与首错位置不是同一种token位置，只分别与各自匹配对照比较。
首错再区分需节点时错误选择、提前关闭列表、应关闭列表时错误输出，避免空候选比例混淆。

训练/验证仍只取自然前4条完整正确记录，此次扩展到候选字段entry/随机middle/closure，
避免仅入口训练外推任意token位置的额外混淆。训练、验证、测试图/query隔离。
512 train/128 val个query专属仿射lens，其余用于线性状态探针，测试不选步/调阈值。
公共fit/evaluate与fit_lens/lens_logits复用上一实验；状态探针stack/visited/successors/
eligible逐层独立、3000步、验证选步、0.5阈值；仿射lens2000步、验证KL选步。
原lens/仿射lens报告合法率及自然token重现率，尤其核对末层是否重现自然首错。
配对对照可能仍不完全平衡；结果是条件关联，不足以证明隐式纠错。

```bash
.venv/bin/python studies/correction_first_error/run.py --background
```

用户已授权共享GPU0；PyTorch显存池上限20%，仅冻结推理/探针训练，禁用W&B。
输出outputs/analysis/correction_first_error_20261102，每个模型一个紧凑JSON；
状态与日志同步更新。不保存activation或probe/lens权重。
