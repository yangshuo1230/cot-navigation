# 搜索过程干预：2026-10-20

仅评估已有 20 个 DFS/BFS validation-best checkpoint；不涉及 Path、不训练模型、不保存新 checkpoint、不启动 W&B。

服务器目录：`/mnt/ningxuefei/yangshuo/cot-navigation/research/search_interventions_20261020`。

从项目根目录运行单模型评估，例如：

```bash
.venv/bin/python research/search_interventions_20261020/run.py \
  --gpu 0 --models context_l10_dfs_state --cases 24 --batch 48
```

输出已完成的模型会自动跳过。`launch.py` 分配全部 8 张 GPU，仅在确认这些 GPU 空闲后使用。

```bash
.venv/bin/python research/search_interventions_20261020/verify.py
```

验证只使用 CPU：重建所有实验、比对每条前缀的哈希、用独立语义检查器验证所有 clean 前缀和合法替代分支，并确认 checkpoint 哈希不变。

本地重画：

```powershell
python G:\CoT\search_interventions_20261020\report.py
```

`run.py --self-test` 只执行内存中的 oracle/parser 自检，不生成实验记录。正式评估保存逐查询统计和每类最多两个详细示例，不保存完整激活或解码缓存。

实验包括目标位置、frontier 交换、可达/死分支替换、合法 PUSH 替代分支、CAND 删除/非边替换/重排，以及错误 FOUND/EXHAUST。参见 `REPORT.md` 的语义与解释边界。

图中分母是每种方法的全部配对样本；只有两边都输出有效 YES/NO 且答案不同才记作翻转。生成失败单独记录。原始图标签与修改后状态的 oracle 标签分别保存。

本轮采用参考搜索前缀，上轮 visited 实验采用模型自由生成前缀，不能直接拼接两轮曲线当作同一组样本的比较。

新版主图按干预选用指标，见 `METRIC_GUIDE.md` 和 `metric_figures/`。只重画新版图可运行 `python plot_metrics.py`；`report.py` 同时更新汇总和全部图。
