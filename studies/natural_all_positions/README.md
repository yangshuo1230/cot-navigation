# 全位置自然候选错误池 v2

旧data/natural_correction/natural_correction_s0已退休，文件保留用于历史追溯。
DEPRECATED.json与公共pool_path检查禁止新加载；v2身份也与旧池不同。

新池：data/natural_correction/all_positions_v2_20261102。
DFS/BFS × 原六个课程规模，共12个池。每规模保持8192训练query、512验证、
512测试query、原抽样seed0、每query一个原搜索variant及一个旧clean donor（best/latest）。
唯一位置抽样政策变更：每条现有4K参考轨迹的所有完整记录均尝试候选字段生成，
states_per_query=0表示全量，有限位置参数直接报错，不允许静默回退到4处。
每个位置从对应正确前缀独立生成一次，不是沿错误自由展开整条轨迹。

复用公共mine_one/complete_fields/describe_error/build_episode/write_pool。
生成到实际分隔符、非节点输出或原生上下文边界，不再额外限制64个候选token。
使用compiled prefill/decode和KV缓存。GPU0共享，PyTorch显存池上限20%，禁用W&B。

每个池同时保存：
- .npz：原4K预算内能打包的错误—MISTAKE—原正确记录训练样本。
- .raw.json.gz：所有观察到的错误字段及其query/variant/原始位置、生成token。
  格式错误、未关闭输出也单独保存；暂不冒充完整候选修复样本。
- .json：全部检查位置数、观察错误数、原始错误数、打包事件数、原始文件校验和。

4K可能限制单条轨迹可容纳的错误数，原始错误事件不会因此丢失。采样、格式失败和
打包排除分别计数；完成时检查每个待生成字段恰有一个停止结果。raw快照定期写入，
完整池已完成时复用；未完成池重启会重新采集该池，不把部分结果当成完整池。

```bash
.venv/bin/python studies/natural_all_positions/run.py --background
```

这是纯数据任务，不launch/relaunch训练。准备状态与日志位于：
outputs/data_preparation/natural_all_positions_v2_20261102/。
完成后未来训练配置应使用新mistake_cache_root；不改写历史模型配置。
