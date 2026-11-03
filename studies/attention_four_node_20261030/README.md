# 四节点 In Context DFS-state attention

图：A→B、B→C、A→D。查询 A→D；邻居顺序 B 在 D 前。

结论：这个图适合解释分支、回退、visited 保留和 FOUND 的序列格式，暂不适合代表所选模型的正常自由搜索行为。它没有汇合/交叉边，因此也不能展示“有出边但后继已访问”的候选过滤情形。

使用现有绘图脚本中的模型：`outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state/checkpoints/best.pt`，step=185000，6层、单头，冻结模型，CPU FP32。该课程最小训练规模为N16；N4是训练规模之外的输入。没有训练新模型，也没有依据结果换模型或筛选节点编号。

自由生成的实际输出（去掉图和查询前缀）：

```
A | A | | EXHAUST EXHAUST ; NO
```

答案错误，轨迹格式也有错误。固定的8组节点编号（原始一组及7组随机重命名）全部预测NO，完整参考轨迹匹配0/8。这只排除了本次所查编号下的偶然成功，不能单独确定失败原因。

完整参考CoT（状态均为动作执行前的快照）：

```
stack | visited | candidates | action [node] ;
A     | A       | B D        | PUSH B ;
A B   | A B     | C          | PUSH C ;
A B C | A B C   |            | POP ;
A B   | A B C   |            | POP ;
A     | A B C   | D          | PUSH D ;
A D   | A B C D |            | FOUND ;
YES
```

`all_layers.png` 是完整参考CoT的teacher-forced注意力，不是模型自发完成正确搜索的证据。包含13个prompt token、64个轨迹token和YES，共78个token；保留所有分隔符。`free_generation_all_layers.png` 才是实际自由生成序列的注意力。模型输出YES/NO后即停止，因此图中不附加EOS。

每张矩阵的行是已消费的query token，列是key token。第i行预测第i+1个token；例如研究PUSH动作的选择应先看它前一位置的行。灰色表示未来不可见，青色线区分prompt/记录。各层共用0–1概率色标，颜色使用平方根映射，没有按行重新缩放。

`interactive.html` 可切换六层及参考/自由生成，悬浮显示精确概率。`layer_1.png`至`layer_6.png`及同名SVG用于逐token查看；`attention.npz`、`free_attention.npz`保留原始矩阵；`metadata.json`保留checkpoint SHA256、token编号映射、记录边界及重命名检查。

验证：逐行attention和为1；未来区域为0；显式math attention和普通SDPA路径的最大logit差为0.0003147125。均在冻结模型CPU FP32下完成，不占用GPU。

下一步若要判断统计代表性，应在原N16 held-out图中寻找包含同类分支/回退的完整合法自由轨迹，再比较字段对齐的注意力；另增加汇合边案例以检查visited过滤。本次未进行该扩展实验。

## 搜索过程动画

- `search_attention.html`：离线交互动画，无外部依赖。参考轨迹65个输出token帧、28个按语法选定的关键帧；另有9个实际自由生成token帧。可播放/暂停、逐帧、拖动时间轴、跳转六条记录、切换六层和参考/实际轨迹；支持键盘空格及方向键。
- `search_attention.gif` / `search_attention.mp4`：36.4秒，参考轨迹的28个关键帧，固定Layer 4。图、状态、所预测token、top-8 key以及六层分组统计同步变化。
- `animation_data.json`：动画所用全部帧、绝对token位置、各层top-8及语义分组质量。分组求和为1，top-8不重新归一化；自身位置单独统计。语义分类是为阅读设计的分组，不是独立验证的功能定位。
- `build_animation.py` / `animation_template.html`：本机渲染脚本和交互模板。视频由ffmpeg编码，字体来自Windows Microsoft YaHei；不需要再次加载模型。

默认Layer 4下的具体变化（均读取所预测token的前一位置）：

| 参考预测事件 | 观察到的主要attention去向 |
| --- | --- |
| R1 PUSH参数 B | 本步候选80.4% |
| R2 PUSH参数 C | 本步候选68.5% |
| R3 POP | 本步stack 91.5% |
| R4 POP | 本步stack 66.6%，历史状态24.7% |
| R5 PUSH参数 D | 本步候选75.7% |
| R6 FOUND | 本步stack 97.2% |

这些是单例条件权重分配，不单独证明网络执行了DFS，也不表示所画参考token一定是模型的greedy选择。图上状态是独立的参考DFS快照，状态均为动作前；自由生成模式不为错误输出补造合法搜索状态。

浏览器验证使用独立headless Edge：参考/实际全部74个预测帧×6层，共444个视图渲染通过；播放、逐token步进、滑块、记录跳转通过；430px移动布局无页面横向溢出；无浏览器运行异常。记录在`animation_verification.json`，页面截图为`interactive_preview.png`。
