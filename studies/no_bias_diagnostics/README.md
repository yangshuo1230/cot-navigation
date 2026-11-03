# NO偏向的两个冻结诊断

服务器：`/mnt/ningxuefei/yangshuo/cot-navigation/research/no_bias_diagnostics_20261022`。

不训练、不创建checkpoint或W&B run，GPU0保留。四个推理worker使用启动时空闲的1/4/5/7卡。

覆盖17个checkpoint：旧scale课程固定图/新图各6种协议（包括Direct基线），使用验证best；新DFS loss/SWA五种设置使用最终checkpoint。没有把所有历史静态规模、种子或新hard-DAG训练都混入本次诊断。新图测试N100、深窄N100、N144；固定图只测试实际参与训练的N100主图。每个TEST层差×标签格最多4条，同设置各协议共享查询ID。BF16，生成batch16；不同批大小或精度可能引起输出差异。

诊断1：独立生成合法参考轨迹，输入至答案前一位置，只读取下一token。绝不输入YES/NO或EOS。不改变模型权重、注意力窗口或位置编码；SWA仅选择语义一致的SDPA实现避免新Flex编译。完全未截断样本为主要配对结果；超上下文的参考按原STOP规则处理并单列，不能作为完整证据。报告全词表argmax以及YES/NO两选一结果，区分格式边界错误与答案偏好。状态轨迹含FOUND/EXHAUST，可能直接提示答案，故此诊断测的是“正确过程及终止线索给定后的读取”，不是独立推理能力。Direct没有训练过的中间轨迹，其oracle条件等同原prompt，不向它强塞未训练的CoT。

诊断2：同查询自由生成，分开检查：完整状态是否合法；忽略打印状态后，动作是否构成合法完整搜索；负例答对中STOP、合法前缀提前结束、候选漏项、非法动作等。首错严格接续真实历史；首错后局部错误不视为有效历史。DFS/BFS节点版按照首次扩展顺序验证，不能把相邻输出节点当成边。Path负例是从起点走到sink的随机路径，路径合法不能证明不可达，负例搜索完整性为N/A。Direct也为N/A。

`diagnose.py`保存逐查询配对指标与gzip轨迹，`action_errors.py`再次独立回放真实动作并定位首错，`summarize.py`输出CSV、JSON及报告。为区分状态注释与实际动作，状态协议receipt使用audit_version=2。CPU验证覆盖7680项参考断言、24576项节点语言等价检查，以及不喂答案和“注释错但动作正确”的构造例。

```bash
# 仅在当前队列退出后恢复，已完整条件会跳过
bash research/no_bias_diagnostics_20261022/launch.sh

# 推理完成后执行独立动作回放与汇总
.venv/bin/python research/no_bias_diagnostics_20261022/action_errors.py
.venv/bin/python research/no_bias_diagnostics_20261022/summarize.py
```

日志位于`worker0.log`至`worker3.log`；结果在`results/`。语义验证为`verification.json`。
