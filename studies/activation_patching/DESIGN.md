# 同 token、不同隐藏状态的 CoT 干预

实验目标：检验一个显式字段的深层表示是否携带该字段之外的历史信息，以及这些信息是否因果影响后续搜索。保持 token 序列相同的激活替换，能够证明隐藏表示存在因果作用；单凭这一实验不能给“显式文本 vs 隐藏状态”排出重要性高低，因为历史文本本来就是通过隐藏表示和 KV 进入后续计算的。

## 样本与 donor

首轮使用已有 N=100 的 DFS-state/BFS-state，在新图和固定图两个任务上分别取该完整状态变体的 validation-best checkpoint。不能复用 node-only 的最佳变体，因为它没有这些字段。固定图首轮只覆盖 scale 实验的 seed 0；不把它写成三图 seed 结论。

记录格式：`frontier | visited | candidates | action [arg] ;`。DFS 的 frontier 是栈，BFS 是队列。

recipient 为 TEST 中预先分层抽取的 256 个查询，donor 为 TRAIN 中预先分层抽取的最多 2048 个查询。使用真实离线数据中的轨迹；所有记录经图语义重放验证。查询标签、模型输出不参与挑选配对。使用训练集作为 donor 不包含训练、梯度或标签泄露，模型参数始终冻结。仅保留前缀不超过 1024 token、后面还有完整记录的非终止位置。

对 frontier、visited、candidates 分别配对：目标字段的**完整序列逐 token 相同**，另外两个字段各自语义不同：visited/candidates 必须集合不同，frontier 必须完整有序序列不同。即使只 patch 一个分隔符，也坚持这个条件。默认每个字段最多四个不同 recipient 查询。

两种来源：

1. **no_history**：recipient 的图/查询 prompt + 单个同名 token，独立计算每个被替换 token 的 activation。不是拼上一整个字段。仍保留图/查询信息，去掉此前 CoT。prompt 位置不变，追加 token 的 RoPE 位置设为 recipient 中的原位置，避免追加 token 的绝对位置变化成为额外因素。
2. **cross_example**：另一查询中匹配字段及对应语法位置的 activation，保留 donor 的自然历史。目标字段逐 token 对齐；`|` 按对应字段边界对齐，`;` 按对应记录结束位置对齐。动作操作符还要求 token 身份相同。

严格原生匹配的覆盖率本身要报告。禁止用部分字段相同、相同长度或集合相同替代序列完全相同。操作符不同的配对仍可用于节点/分隔符；只跳过操作符及包含它的组合，以免不必要地删掉 visited 配对。不同组分母可能不同，比较组合效应时必须取同一批 case。

## 干预与消融

冻结模型，在第 2、4、6 层 block 结束后的 residual 上做完整替换，层号从 1 开始。只干预当前一条记录中的相应位置，之前的历史、prompt 和全部输入 token 保持不变。

| 组 | 替换位置 |
| --- | --- |
| nodes | 匹配字段内全部节点 token |
| bar | 匹配字段末尾的一个 `|` |
| operator | 同 token 的 `<PUSH>` / `<POP>`；首轮采非终止记录，无 FOUND/EXHAUST |
| semicolon | 当前记录的 `;` |
| nodes_bar | nodes + bar |
| all | nodes + bar + operator + semicolon |

PUSH 的节点参数不在上述组内。动作操作符按用户补充要求单独干预，然后纳入组合。首轮未覆盖终止操作符；后续需增加 FOUND/EXHAUST cohort 并观察 YES/NO，而不能混入下一条状态记录的指标。

每个干预同时计算同 batch、同 padded 长度、同精度的未干预 twin。另有 self-patch，验证替换原激活不改变输出。

真正执行时对 recipient 重新 prefill：在指定 block 返回处替换 residual，后面的 block 按干预后状态重建 K/V。进入续写前移除 hook，不反复覆盖新生成 token。没有修改训练模型源码，也不直接复制 donor 的 RoPE 旋转后的 K。

**最后一层的因果连接必须分开解释**：过去位置的 post-final-block residual 不再进入任何后续层的 K/V，因此对后续位置没有通路；这是实现负对照。若替换的是当前前缀最后一个 token，例如字段 cut 的 `|` 或记录 cut 的 `;`，其 final-layer residual 直接决定下一个 token logits，此时有效果完全合理。不能把这两种情况混在一起解释成层功能差异。

## 两种读取位置

**current_record**：匹配字段末尾 `|` 读入后立刻停止 prefill，进行自由续写直至当前记录结束。只测 nodes、bar、nodes_bar。读取剩余字段与尚未生成的动作；已经作为前缀输入的字段不计入生成正确性。这个实验最直接对应“当前字段的隐藏状态是否影响后面的动作”。

**next_record**：固定当前整条记录的 token，直到 `;`，再自由生成下一条记录。此时可比较所有单组与组合。当前动作已经输入，不能称为动作变化；统计的是下一条记录的字段与动作。

每个条件最多生成 320 token，遇到第一个 `;` 或终止 token 停止；缺记录、提前回答、格式错误和达到长度上限均单列。这是局部响应实验，**没有评测干预后的最终任务准确率**，也没有把下一条记录的变化写成完整规划行为变化。后续若做长程续写，需另用统一全上下文预算报告最终答案、轨迹合法率和恢复时间。

## 指标与判读

- immediate next-token 分布 JS、最大概率变化、argmax 翻转。
- 配对自由续写是否变化；记录能否解析；字段是否与 recipient 的参考相符；动作是否为合法替代，而非只认固定第一候选。
- 对 recipient 和 donor 各自的后续字段计算相似度。frontier 用完整顺序精确匹配；visited/candidates 用 Jaccard，另保留序列精确匹配。
- `donor_similarity_shift` 必须相对同 batch 未干预 twin。recipient 与 donor 的目标字段本来就相同、或者后续字段也相同的 case，不提供方向性迁移证据；应在预先定义的不同目标子集上解释。
- 若只看到格式破坏/合法率下降，不能称为 donor 语义迁移。若观察到朝 donor 特定候选、visited、栈或动作方向变化，且 recipient 正确动作支持相应区分，才是更强证据。

统计单位是 recipient 查询，不能把三个 layer、多个组的干预次数当独立样本。首轮每格 2–4 个 case 只作方法验证与发现异常，不能作为稳定的群体效应估计。

## 首轮之后的必要控制

以下是进一步归因所需控制，**不属于首轮已执行结果**：

1. 更大查询池，继续坚持另外两个字段的集合/序列语义不同，覆盖深浅、分支、PUSH/POP，不只测早期短状态。无法配对的格子保留为空。
2. 区分“同图不同查询”与“不同图不同查询”。新图跨 example 会同时改变图和目标信息；其 donor 效应不能自动归因于 visited 单一功能。
3. 在各组共同支持的同一批 case 上比较单组/组合；增加 operator+semicolon、nodes+operator 等有针对性的组合，避免一开始全排列。
4. 严格前缀长度匹配、record ordinal 匹配及不同长度分层。no_history 已匹配追加 token 位置，但缺失整个历史本身改变了可注意位置集合，不能声称 RoPE/距离影响被完全去除。cross_example 的历史长度仍可能不同。
5. 记录 patch 前后激活范数，比较范数匹配 donor、同状态不同历史 donor、同 token 随机 donor，并做替换强度扫描；区分内容迁移和一般分布偏移。
6. 验证“文本能否纠正隐藏状态”：字段之后立即自由续写 vs 固定其余正确字段再续写，使用共同 case；局部 logit 响应消失可能是后续文本纠正，也可能是本层因果通路不覆盖该位置，需与层/位置控制一起解释。

不引入 RL、不训练新模型、不生成 W&B run、不写 checkpoint，不保存激活矩阵或 KV 缓存。结果只保存配对索引/前缀哈希、统计和少量异常续写。
