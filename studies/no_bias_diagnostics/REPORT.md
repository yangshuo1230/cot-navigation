# NO偏向：正确参考轨迹与负例完整性诊断

冻结推理，不训练、不创建checkpoint或W&B run。查询在TEST内按层差×标签分层，最多每格4条；同图设置各协议共享查询。老规模课程采用验证best，新loss/SWA五组采用最终checkpoint。

## 正确参考轨迹下的答案读取

完整未截断轨迹不包含答案或EOS，但状态协议包含FOUND/EXHAUST，结果只能说明给出正确计算及终止线索后能否读取答案，不代表独立完成推理。截断轨迹从以下oracle对比中排除；自由生成列也使用相同未截断子集。Direct无CoT，oracle只是原始prompt，不能称为有正确轨迹干预。

|模型|条件|配对完整oracle正/负n|自由YES/NO召回|正确轨迹YES/NO召回|
|---|---|---|---|---|
|context_curriculum_bfs_nodes|deep|76/76|50.0%/86.8%|100.0%/100.0%|
|context_curriculum_bfs_nodes|larger|44/44|36.4%/86.4%|97.7%/100.0%|
|context_curriculum_bfs_nodes|n100|36/36|47.2%/91.7%|100.0%/100.0%|
|context_curriculum_bfs_state|deep|55/58|67.3%/98.3%|100.0%/100.0%|
|context_curriculum_bfs_state|larger|31/29|83.9%/93.1%|100.0%/100.0%|
|context_curriculum_bfs_state|n100|33/33|87.9%/100.0%|100.0%/100.0%|
|context_curriculum_dfs_nodes|deep|76/76|61.8%/97.4%|100.0%/94.7%|
|context_curriculum_dfs_nodes|larger|44/44|2.3%/100.0%|43.2%/63.6%|
|context_curriculum_dfs_nodes|n100|36/36|58.3%/97.2%|100.0%/94.4%|
|context_curriculum_dfs_state|deep|69/57|53.6%/100.0%|100.0%/100.0%|
|context_curriculum_dfs_state|larger|39/30|71.8%/100.0%|100.0%/100.0%|
|context_curriculum_dfs_state|n100|35/34|85.7%/94.1%|100.0%/100.0%|
|context_curriculum_direct|deep|76/76|0.0%/100.0%|0.0%/100.0%|
|context_curriculum_direct|larger|44/44|0.0%/100.0%|0.0%/100.0%|
|context_curriculum_direct|n100|36/36|0.0%/100.0%|0.0%/100.0%|
|context_curriculum_path|deep|76/76|22.4%/100.0%|98.7%/100.0%|
|context_curriculum_path|larger|44/44|27.3%/97.7%|97.7%/100.0%|
|context_curriculum_path|n100|36/36|27.8%/100.0%|97.2%/100.0%|
|context_loss_full_l1|deep|69/57|60.9%/100.0%|100.0%/100.0%|
|context_loss_full_l1|larger|39/30|87.2%/100.0%|100.0%/100.0%|
|context_loss_full_l1|n100|35/34|82.9%/100.0%|100.0%/100.0%|
|context_loss_full_l16|deep|69/57|52.2%/98.2%|100.0%/100.0%|
|context_loss_full_l16|larger|39/30|74.4%/96.7%|100.0%/100.0%|
|context_loss_full_l16|n100|35/34|97.1%/97.1%|100.0%/100.0%|
|context_loss_full_l4|deep|69/57|53.6%/96.5%|100.0%/100.0%|
|context_loss_full_l4|larger|39/30|82.1%/100.0%|100.0%/100.0%|
|context_loss_full_l4|n100|35/34|85.7%/97.1%|100.0%/100.0%|
|context_loss_swa1_l1|deep|69/57|68.1%/82.5%|100.0%/100.0%|
|context_loss_swa1_l1|larger|39/30|71.8%/93.3%|100.0%/100.0%|
|context_loss_swa1_l1|n100|35/34|82.9%/94.1%|100.0%/100.0%|
|context_loss_swa2_l1|deep|69/57|56.5%/98.2%|100.0%/100.0%|
|context_loss_swa2_l1|larger|39/30|64.1%/100.0%|100.0%/100.0%|
|context_loss_swa2_l1|n100|35/34|80.0%/100.0%|100.0%/100.0%|
|fixed_curriculum_bfs_nodes|n100|32/32|87.5%/96.9%|90.6%/100.0%|
|fixed_curriculum_bfs_state|n100|30/30|90.0%/96.7%|100.0%/100.0%|
|fixed_curriculum_dfs_nodes|n100|32/32|96.9%/93.8%|96.9%/100.0%|
|fixed_curriculum_dfs_state|n100|32/30|93.8%/100.0%|100.0%/100.0%|
|fixed_curriculum_direct|n100|32/32|81.2%/84.4%|81.2%/84.4%|
|fixed_curriculum_path|n100|32/32|71.9%/100.0%|93.8%/100.0%|

## 负例答对时，过程是否完整

分母是各条件中真实负例且自由生成答对的查询数。动作完整性要求忽略打印注释后仍能合法执行并穷尽搜索；完整状态还要求frontier、visited和候选正确。STOP按未完成处理，不自动解释为算法错误。Path只提供路径、不承诺穷尽；Direct没有过程，两者搜索完整性为N/A。

|模型|条件|答对负例n|完整状态不合法|实际动作不完整/非法|仅状态注释错|其中STOP|
|---|---|---:|---:|---:|---:|---:|
|context_curriculum_bfs_nodes|deep|66|45.5%|45.5%|0.0%|0.0%|
|context_curriculum_bfs_nodes|larger|38|50.0%|50.0%|0.0%|0.0%|
|context_curriculum_bfs_nodes|n100|33|51.5%|51.5%|0.0%|0.0%|
|context_curriculum_bfs_state|deep|73|41.1%|41.1%|0.0%|9.6%|
|context_curriculum_bfs_state|larger|38|39.5%|39.5%|0.0%|13.2%|
|context_curriculum_bfs_state|n100|36|36.1%|36.1%|0.0%|0.0%|
|context_curriculum_dfs_nodes|deep|74|44.6%|44.6%|0.0%|0.0%|
|context_curriculum_dfs_nodes|larger|44|56.8%|56.8%|0.0%|0.0%|
|context_curriculum_dfs_nodes|n100|35|42.9%|42.9%|0.0%|0.0%|
|context_curriculum_dfs_state|deep|74|41.9%|40.5%|1.4%|2.7%|
|context_curriculum_dfs_state|larger|40|75.0%|72.5%|2.5%|5.0%|
|context_curriculum_dfs_state|n100|33|33.3%|33.3%|0.0%|3.0%|
|context_curriculum_direct|deep|76|N/A|N/A|N/A|0.0%|
|context_curriculum_direct|larger|44|N/A|N/A|N/A|0.0%|
|context_curriculum_direct|n100|36|N/A|N/A|N/A|0.0%|
|context_curriculum_path|deep|76|N/A|N/A|N/A|0.0%|
|context_curriculum_path|larger|43|N/A|N/A|N/A|0.0%|
|context_curriculum_path|n100|36|N/A|N/A|N/A|0.0%|
|context_loss_full_l1|deep|75|42.7%|41.3%|1.3%|2.7%|
|context_loss_full_l1|larger|43|39.5%|39.5%|0.0%|9.3%|
|context_loss_full_l1|n100|35|31.4%|31.4%|0.0%|0.0%|
|context_loss_full_l16|deep|75|40.0%|40.0%|0.0%|6.7%|
|context_loss_full_l16|larger|42|40.5%|40.5%|0.0%|19.0%|
|context_loss_full_l16|n100|35|28.6%|28.6%|0.0%|5.7%|
|context_loss_full_l4|deep|73|39.7%|39.7%|0.0%|0.0%|
|context_loss_full_l4|larger|41|41.5%|41.5%|0.0%|2.4%|
|context_loss_full_l4|n100|35|28.6%|28.6%|0.0%|2.9%|
|context_loss_swa1_l1|deep|56|26.8%|26.8%|0.0%|5.4%|
|context_loss_swa1_l1|larger|38|39.5%|39.5%|0.0%|0.0%|
|context_loss_swa1_l1|n100|34|38.2%|38.2%|0.0%|0.0%|
|context_loss_swa2_l1|deep|72|44.4%|44.4%|0.0%|1.4%|
|context_loss_swa2_l1|larger|43|60.5%|60.5%|0.0%|2.3%|
|context_loss_swa2_l1|n100|36|36.1%|36.1%|0.0%|0.0%|
|fixed_curriculum_bfs_nodes|n100|31|19.4%|19.4%|0.0%|0.0%|
|fixed_curriculum_bfs_state|n100|29|13.8%|13.8%|0.0%|10.3%|
|fixed_curriculum_dfs_nodes|n100|30|26.7%|26.7%|0.0%|0.0%|
|fixed_curriculum_dfs_state|n100|32|34.4%|34.4%|0.0%|3.1%|
|fixed_curriculum_direct|n100|27|N/A|N/A|N/A|0.0%|
|fixed_curriculum_path|n100|32|N/A|N/A|N/A|0.0%|

结果限于每格最多4条的单checkpoint诊断面板，不能替代完整测试准确率。状态首错之后的局部错误只是打印内容检查，不是有效执行历史；节点被提及也不证明曾合法扩展。可支持定位错误发生在过程还是最终答案，但不能单独证明NO偏向的唯一因果机制。

[逐条件统计](summary.json)；[CSV](summary.csv)；[少量配对例子](examples.json)。
