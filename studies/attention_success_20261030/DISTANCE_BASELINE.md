> 此文件仅记录已撤回的距离校正实验。当前HTML已恢复原始attention分数。

# HTML attention 展示的距离校正

2026-10-31 起，`token_explorer.html`、`search_attention.html`、`search_animation.html` 均显示：

`ΔA[l,q,k] = A[l,q,k] − μ[l,q−k]`

`μ[l,d]` 是本例完整 116-token 序列中所有 `q−k=d` 的因果可见 attention 的算术平均；每层/head单独计算，包含图、查询、CoT、答案位置，距离 d 有 `116−d` 对样本。未来遮罩位置不计入均值，也不显示校正值。此模型每层只有一个 head。

蓝色为正，橙色为负，白色为零。单位 pp 为百分点：`+10 pp` 表示实际 attention 比同距离基线高 0.10。颜色按绝对值平方根映射，token格统一±100 pp；原始语义分类颜色仍用于标识字段。

排序、六层对照、历史路径、完整矩阵、图边读取与边块聚合全部使用校正分数。分类/边块先逐 token 校正再求和或取平均，不对聚合值重新做归一化。每行之和不要求为0；每条同距离对角线的均值为0。

原始 `attention.npz`、`animation_data.json` 和已有 GIF 保留。HTML 重建顺序为 `build_distance_animation.py`、`build_token_explorer.py`；只更新 HTML，不生成视频。

行为图谱 `../dfs_attention_lifecycle_20261030/behavior_atlas.html` 使用相同定义，但每条查询分别估计距离基线，之后按原规则 query 内平均、跨 query 平均，并对18条N100 test查询bootstrap。原始研究报告中的历史统计保留；干预成功次数不属于attention分数，不作距离校正。
