# 长搜索实际实例

入口：[YES搜索过程](search_animation.html)、[NO搜索过程](search_animation_no.html)、[YES逐token](token_explorer.html)、[NO逐token](token_explorer_no.html)。

主实例已从4记录的小图升级为同一185k模型在N100/150边上的复杂搜索。保留原始attention概率，没有距离校正，没有新训练，没有生成视频。

| 实例 | query index | 状态记录 | 最大栈深 | PUSH / POP | 访问节点 | 连续POP最长 |
| --- | --- | --- | --- | --- | --- | --- |
| YES | 50708 | 27 | 8 | 15 / 11 | 16 | 5 |
| NO | 50951 | 43 | 10 | 21 / 21 | 22 | 7 |

两条都包含7条具有多个候选的记录、分支重访和visited过滤。YES的R24/R26在N51过滤此前其他分支访问的N62；NO的R17/R19在N85过滤N96、R42在N11过滤N24，展示的是DAG跨分支去重，非仅树形回溯。在此前12条middle/long test原生生成中，这两条是全部完整合法的成功结果。展示它们是为说明复杂行为，不能估计总体成功率；该批次只有2/12整条合法。原始全部结果见 `../dfs_attention_lifecycle_20261030/free_rollouts.json`。

模型只接收图和查询。`extract.py`直接读取既有自由生成的tokens，独立replay验证完整轨迹，再用完整因果forward提取全部attention；每个生成位置的argmax均与缓存自由生成一致。NO不是截断后的失败：它完整执行21次POP回到根并输出EXHAUST/NO。

节点N01…N100对应数据的拓扑节点编号，只改变显示名，不改token。6层完整FP32 attention以gzip/base64内嵌HTML，在浏览器解压；没有量化或省略低权重。完整矩阵使用Canvas保留每个格子，逐token列表、六层对照、历史路径仍可精确查看原始值。HTML较大，首次打开需载入。

`build.py`生成四个自包含HTML，并把旧 `attention_success_20261030` 的三个入口更新到YES长实例；原小例子保存在 `*_small.html`。后者仍有独立入口用于对照。

复现：在服务器项目根目录运行 `.venv/bin/python studies/attention_long_20261031/extract.py`；同步两组JSON/gzip后，本地运行 `D:\anaconda\python.exe attention_long_20261031/build.py`。`verify.mjs`验证两条实例的全部记录跳转、图节点、150边、六层原始行/列、历史追踪、完整Canvas和旧入口更新，结果保留于 `validation.json`。
