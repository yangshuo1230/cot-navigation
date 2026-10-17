# 数据覆盖与一跳监督重训入口

本轮保持模型、图规模、优化器、WALK语料不变，修改训练查询覆盖与新图独立实例数量。2026-10-16起撤下Filler，当前默认只运行Direct、Path、DFS三种协议。默认每个固定图模型仍只学习一张图。旧数据与旧run保留，新批次不复用旧checkpoint。

## 一条命令

在服务器项目目录运行：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
bash scripts/retrain.sh --run-name retrain_v3_no_filler_s012_50k --gpus 0 1 2 3 4 5 6 7 --wait --background
```

同一run-name不允许覆盖。后台任务先准备/审计数据，再调度空闲GPU，每个任务一张卡，不抢占已有进程。运行状态在 `outputs/retrain_v3/launches/<run-name>/status.json`，总日志在相邻的 `<run-name>.launch.log`；W&B继续使用已配置账号、`cot-graph-reasoning`项目和run-name分组。在线上传失败直接报错。

默认21个任务：新图10k/50k/150k三种数据量×三协议=9；固定图seed0/1/2各一个共享知识阶段及三个combined分支=12。Stage A各1000步，所有任务阶段50000步，每步64查询；固定图B另混16条WALK。没有自动正式训练，执行以上命令后才启动。

Filler的底层协议、旧配置与旧结果保留供历史复查；不进入默认训练或数据准备。恢复历史四协议run时须显式传入原协议列表`--protocols direct path dfs_state filler`，避免更改原plan。当前新批次无需指定`--protocols`。

## 预览、仅准备、单组与恢复

```bash
# 只打印配置/命令；即使带--background也不创建run或数据。
bash scripts/retrain.sh --run-name inspect_v3 --dry-run

# 准备所有默认数据和三协议token缓存，不训练、不占GPU、不建run。
bash scripts/retrain.sh --prepare-only

# 先跑新图Direct/Path数据量对照。
bash scripts/retrain.sh --setting context --protocols direct path --context-sizes 10000 50000 150000 --run-name context_coverage_v3 --wait --background

# 只跑固定图默认combined设置。
bash scripts/retrain.sh --setting fixed --run-name fixed_combined_v3 --wait --background

# 同一批比较固定图四种数据设置；同图四组共用一个Stage A。
bash scripts/retrain.sh --setting fixed --protocols direct path --fixed-variants legacy coverage onehop combined --run-name fixed_ablation_v3 --wait --background

# 恢复默认整批：保留原命令的设置、数据量、协议、种子和run-name。
bash scripts/retrain.sh --run-name retrain_v3_no_filler_s012_50k --gpus 0 1 2 3 4 5 6 7 --wait --background --resume
```

`--context-steps`、`--task-steps`、`--knowledge-steps`设置步数；`--model-seed`和`--data-seed`分别控制初始化和生成；`--one-hop-fraction`默认0.25，只影响onehop/combined。例如0.5则每64查询中32条一跳。0一跳用legacy/coverage配置表示。`--set train.lr=0.0005`等支持已有类型配置；脚本管理的数据路径、协议、数据量、采样设置不能通过--set绕过。

修改图大小、形状或边数时建议复制YAML并使用新的data.root直接运行CLI，避免脚本内用于本次对照的N16/N32路径命名产生歧义。

## 固定图数据与采样

新数据目录 `data/retrain_v3/parametric_n32_m48_onehop_s0`。在原336个多跳查询后，附加全部112个相邻层节点对：48个YES、64个NO，全部属于train。图、节点映射、WALK、多跳查询及其划分与旧设置逐项对齐；原平衡验证子集也保持一致。一跳数据不会落到val/test。

| variant | 多跳训练池 | 标签采样 | 一跳占查询批次 |
| --- | --- | --- | ---: |
| legacy | 原按距离下采样的102/120/92对 | 原均匀循环 | 0 |
| coverage | 原始全部164/166/166对 | 正负分别循环、批次匹配 | 0 |
| onehop | 原102/120/92对，加全部112对一跳 | 每种hop类别内正负匹配 | 25% |
| combined | 原始164/166/166对，加全部112对一跳 | 每种hop类别内正负匹配 | 25% |

combined共276/278/278个唯一训练查询。标签平衡通过重采样实现，不永久丢弃池内查询；距离分布由池决定，不声称各距离等频。各池遍历完才重新洗牌。小数批次配额采用累计取整，标签奇数配额交替分配。各协议的查询/WALK顺序一致；Path选路单独用RNG，仍为先抽查询，再在该查询的全部合法路径中均匀选一条。

legacy对照保留旧的训练池/顺序，但为公平评估使用本轮的full验证/测试视图和选择指标；因此不是旧run的逐位复现。Stage A仍不加载或监督查询，新增一跳任务在Stage B参与。

## 新图数据

默认三个独立数据目录分别为10k/50k/150k训练图，验证/测试各10k图；N16/M24不变。启用`data.context_holdout_first`，先生成val、test，再生成train，保证同种子、相同val/test数量时，三档的held-out图逐例相同且均与各训练集去重。三档训练集不承诺严格嵌套。

所有数据离线生成并检查BFS、查询标签、划分重复及完整轨迹，再准备token缓存；训练update只取内存数据。新图默认2%独立BFS审计并对每例执行结构/bitset/DFS一致性检查；固定图全量BFS审计。数据身份包含新增语义开关，配置不同会拒绝复用旧root。

## 评估、保存与W&B

- 新图每5000步在固定256例val上验证；固定图用全部原始held-out val查询（62/64/61例）。报告accuracy、balanced accuracy、正负召回与距离分桶。
- 固定图按val balanced accuracy选择best；新图按val answer accuracy。选checkpoint不读取test。
- 训练完成后自动对best和last分别评估全部test，保存`eval/free_test_best.json`和`free_test_last.json`，含实际checkpoint步数；固定图完整test为110/106/109例，和旧balanced test的74/78/66例不同，不直接混表。
- `eval/predictions_test_{best,last}.jsonl`逐例保存query_id、真值、预测与距离，便于跨协议配对比较；`traces/free_test_{best,last}.txt`为少量预览，不覆盖val轨迹。
- W&B上传`test/best/overall/{answer_accuracy,balanced_accuracy}`和`test/last/...`。测试与训练共享原run，不额外开W&B任务。
- 如需旧balanced视图或一跳诊断：

```bash
.venv/bin/python -m cot_navigation.cli evaluate --config <run>/resolved_config.yaml --checkpoint <run>/checkpoints/best.pt --split val --query-view balanced --set train.persist_outputs=false
.venv/bin/python -m cot_navigation.cli evaluate --config <run>/resolved_config.yaml --checkpoint <run>/checkpoints/best.pt --split train --query-view one_hop --set train.persist_outputs=false
```

## 验证与范围

`cot-nav smoke --steps 2 --config configs/experiments/retrain_stage_b.yaml`仍是零持久化检查：临时数据退出清理，不创建训练日志、checkpoint或W&B run。正式重训入口不会偷偷启动smoke。

本次没有更改Path/DFS语法或loss，也没有加入新搜索图族或N200论文锚点。新增采样和数据量对照先隔离当前baseline泛化问题；搜索难度图族与严格原文复现仍是后续独立实验。
