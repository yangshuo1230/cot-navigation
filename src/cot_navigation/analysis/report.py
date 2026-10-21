"""Compact aggregate plots and evidence tables; no activation dumps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROLES, write_json


FAMILIES = ("direct", "path", "filler", "dfs", "bfs")
COLORS = dict(zip(FAMILIES, ["#4878A8", "#EE854A", "#8C6BB1", "#4C9C78", "#CF597E"]))


def primary(models, kind, family):
    return [r for r in models if r["entry"]["kind"] == kind and r["entry"]["family"] == family
            and not r["entry"]["auxiliary"]]


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def attention_plot(models, out, kind="context_graph"):
    fig, axes = plt.subplots(5, 6, figsize=(22, 16), squeeze=False)
    vmax = .035
    for row, fam in enumerate(FAMILIES):
        selected = primary(models, kind, fam)
        values = np.mean([np.asarray(model["attention"]["generated"]["corrected_per_pair"]).mean(1) for model in selected], axis=0)
        counts = np.sum([np.asarray(model["attention"]["generated"]["role_pair_counts"]) for model in selected], axis=0)
        for layer in range(6):
            ax = axes[row, layer]
            masked = np.ma.array(values[layer], mask=counts == 0)
            image = ax.imshow(masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            ax.set_xticks(range(len(ROLES)), ROLES, rotation=90, fontsize=7)
            ax.set_yticks(range(len(ROLES)), ROLES if layer == 0 else [""]*len(ROLES), fontsize=7)
            ax.set_title(f"{fam.upper()} / Layer {layer+1}", fontsize=10)
    label = "New-graph" if kind == "context_graph" else "Fixed-graph (equal mean of 3 worlds)"
    fig.suptitle(f"{label} attention: actual minus distance-conditioned mean\n"
                 "Freely generated prefixes; rows receive, columns supply information; gray/blank = no pairs",
                 fontsize=15)
    fig.subplots_adjust(top=.93, bottom=.09, wspace=.25, hspace=.8, right=.91)
    cax = fig.add_axes([.93, .35, .012, .3])
    fig.colorbar(image, cax=cax, label="Mean residual per token pair (clipped display)")
    save(fig, out/("attention_flow.png" if kind == "context_graph" else "attention_fixed_flow.png"))


def answer_plot(models, out):
    fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharey=True, squeeze=False)
    stages = ["prompt", "early", "middle", "late"]
    for row, kind in enumerate(["context_graph", "parametric_graph"]):
        for col, fam in enumerate(FAMILIES):
            ax = axes[row, col]
            rs = primary(models, kind, fam)
            for idx, label in [(0, "L4"), (1, "L5"), (2, "L6"), (3, "Local token control")]:
                vals = []
                for stage in stages:
                    scores = [r["probes"]["targets"]["answer"]["phase_specific"][stage]["generated_test"][idx]["balanced_accuracy"]
                              for r in rs if stage in r["probes"]["targets"]["answer"].get("phase_specific", {})
                              and "generated_test" in r["probes"]["targets"]["answer"]["phase_specific"][stage]]
                    scores = [score for score in scores if score is not None]
                    vals.append(np.mean(scores) if scores else np.nan)
                ax.plot(range(4), vals, marker="o", ms=4, label=label,
                        linestyle="--" if idx == 3 else "-")
            ax.axhline(.5, color="gray", lw=.8, ls=":")
            ax.set_ylim(0, 1.04)
            ax.set_xticks(range(4), stages, rotation=30, fontsize=8)
            ax.set_title(f"{'New graph' if row == 0 else 'Fixed graph'} / {fam.upper()}")
            if col == 0:
                ax.set_ylabel("Answer probe balanced accuracy")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Answer decodability along freely generated CoT (fixed graph: equal mean of 3 worlds)\n"
                 "Separate probe per progress point; trained on reference prefixes; input control includes observable cues",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, out/"answer_probes.png")


def structure_plot(models, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for row, kind in enumerate(["context_graph", "parametric_graph"]):
        for col, task in enumerate(["successors", "visited"]):
            ax = axes[row, col]
            for fam in FAMILIES:
                rs = [r for r in primary(models, kind, fam)
                      if "generated_test" in r["probes"]["targets"].get(task, {})]
                if not rs:
                    continue
                scores = np.asarray([[v["f1"] for v in r["probes"]["targets"][task]["generated_test"]] for r in rs])
                ax.errorbar(range(4), scores.mean(0), yerr=scores.std(0, ddof=1) if len(rs)>1 else None,
                            color=COLORS[fam], label=fam.upper(), marker="o", capsize=3)
            ax.set_xticks(range(4), ["L4", "L5", "L6", "Node control"])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Within-graph multilabel F1")
            ax.set_title(f"{'New graph' if row == 0 else 'Fixed graph'} / {task}")
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
    fig.suptitle("Structural linear probes on freely generated prefixes\n"
                 "VIS read before its current record is emitted; node-only search uses expansion history instead",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, out/"structural_probes.png")


def patch_plot(models, out):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), squeeze=False)
    for row, kind in enumerate(["context_graph", "parametric_graph"]):
        for col, fam in enumerate(FAMILIES):
            ax = axes[row, col]
            rs = primary(models, kind, fam)
            names = sorted({name for r in rs for name in r.get("patching", {}).get("conditions", {})},
                           key=lambda n: (n != "clean", n))
            for i, name in enumerate(names):
                vals = [r["patching"]["conditions"][name]["accuracy"] for r in rs
                        if name in r.get("patching", {}).get("conditions", {})]
                ax.bar(i, np.mean(vals), color=COLORS[fam], alpha=.95 if name == "clean" else .55)
                if len(vals)>1:
                    ax.errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1), color="black", capsize=3)
            ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{'New graph' if row == 0 else 'Fixed graph'} / {fam.upper()}")
            if col == 0:
                ax.set_ylabel("Continuation answer accuracy")
    fig.suptitle("Token interventions: graph labels unchanged for state/path/DOT edits\n"
                 "Direct changes query target; route-vs-irrelevant VIS inserts have identical token counts",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, out/"token_interventions.png")


def scalar_tables(models, out):
    rows = []
    def score(target, mode, layer, metric):
        values = target.get(mode, [])
        return values[layer].get(metric) if len(values) > layer else None
    for r in models:
        targets = r["probes"]["targets"]
        answer = targets["answer"]
        phase = "middle" if "middle" in answer.get("phase_specific", {}) else "prompt"
        answer = answer.get("phase_specific", {}).get(phase, answer)
        row = {"model": r["entry"]["id"], "kind": r["entry"]["kind"],
               "family": r["entry"]["family"], "auxiliary": r["entry"]["auxiliary"],
               "protocol": r["entry"]["protocol"], "answer_probe_phase": phase,
               "rollout_accuracy": r["rollout"]["accuracy"],
               "clean_replay_agreement": r.get("patching", {}).get("clean_replay_agreement")}
        for task, target in [("answer", answer), ("successors", targets.get("successors", {})),
                             ("visited", targets.get("visited", {}))]:
            metric = "balanced_accuracy" if task == "answer" else "f1"
            for suffix, mode, layer in [("L6", "generated_test", 2), ("input", "generated_test", 3),
                                        ("shuffled", "shuffled_control_test", 2), ("reference", "reference_test", 2)]:
                row[f"{task}_{suffix}"] = score(target, mode, layer, metric)
        rows.append(row)
    with (out/"summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| 任务 | 协议 | 最终答案 | 中段答案探针 L6 / 输入 / 打乱 | 后继 F1：L6 / 输入 / 打乱 | visited F1：L6 / 输入 / 打乱 |",
             "|---|---|---:|---|---|---|"]
    def mean_format(rs, key):
        values = [r[key] for r in rs if r.get(key) is not None]
        return f"{np.mean(values):.1%}" if values else "—"
    for kind in ["context_graph", "parametric_graph"]:
        for fam in FAMILIES:
            rs = [r for r in rows if r["kind"] == kind and r["family"] == fam and not r["auxiliary"]]
            cells = ["新图" if kind == "context_graph" else "固定图三图均值", fam,
                     mean_format(rs, "rollout_accuracy")]
            for task in ["answer", "successors", "visited"]:
                cells.append(" / ".join(mean_format(rs, f"{task}_{suffix}") for suffix in ["L6", "input", "shuffled"]))
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def attention_table(models):
    lines = ["| 新图协议 | L1 | L2 | L3 | L4 | L5 | L6 |", "|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        r = primary(models, "context_graph", fam)[0]
        a = r["attention"]["generated"]
        means = np.asarray(a["corrected_per_pair"]).mean(1)
        counts = np.asarray(a["role_pair_counts"])
        cells = [fam]
        for layer in range(means.shape[0]):
            eligible = (counts >= 30) & ~np.eye(len(ROLES), dtype=bool)
            candidates = np.where(eligible, means[layer], -np.inf)
            i, j = np.unravel_index(candidates.argmax(), candidates.shape)
            cells.append(f"{ROLES[j]}→{ROLES[i]} ({means[layer,i,j]:+.3f})")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def unseen_answer_table(models):
    lines = ["| 新图模型 | 中段样本数（正例） | L4 | L5 | L6 | 输入对照 |",
             "|---|---:|---:|---:|---:|---:|"]
    for fam in FAMILIES:
        r = primary(models, "context_graph", fam)[0]
        answer = r["probes"]["targets"]["answer"]["phase_specific"]
        phase = "middle" if "middle" in answer else "prompt"
        scores = [x.get("goal_not_yet_in_cot", {}) for x in answer[phase]["generated_test"]]
        n = scores[0].get("n_rows", 0)
        positive = scores[0].get("n_positive", 0)
        values = [f"{s['balanced_accuracy']:.1%}" if s.get("balanced_accuracy") is not None else "—" for s in scores]
        lines.append("| " + " | ".join([fam + ("（prompt）" if phase == "prompt" else ""),
                                        f"{n} ({positive})", *values]) + " |")
    return lines


def report(models, selection, out):
    lines = ["# 导航模型机理分析", "", "本次冻结已有模型，只训练线性探针；所有计算均在物理 GPU 0。",
             "", "## checkpoint 与样本", "",
             "在 explore_v1 与 retrain_v3 两个已完成主批次、相同架构/图规模的候选中，只用验证成绩选择最佳变体；纳入新图 10k/50k/150k 训练池。固定图按三个世界的平均验证成绩选变体，每个世界使用自己的 best.pt；测试集未参与选择。所有候选任务训练总日程为 50k 更新，best checkpoint 可以更早。Filler 选中 10k 训练池版本，原 50k 版保留作数据规模对照。不同批次的数据呈现方式有所不同，因此本分析不是重新控制变量后的协议因果比较。", "",
             "| 模型 | 实际协议 | checkpoint 步数 | 验证选择分数 | 分析样本最终准确率 |",
             "|---|---|---:|---:|---:|"]
    for r in models:
        e = r["entry"]
        lines.append(f"| {e['id']}{'（补充对照）' if e['auxiliary'] else ''} | {e['protocol']} | {r['checkpoint']['step']} | {e['score']:.2%} | {r['rollout']['accuracy']:.2%} ({r['rollout']['n']}) |")
    lines += ["", "新图为一组模型初始化；固定图三个 seed 是三个图世界，不是同一图上的三次模型初始化。新图样本按标签和图距离分层，分析子集准确率不等同于完整测试准确率。", "",
              "## 注意力", "", "![注意力流向](attention_flow.png)", "",
              "校正定义：每层每头分别在 train 校准样本上计算 μ(d)=mean(A[i,j] | i-j=d, j≤i)，在 test 上计算 A−μ(d)。排除 padding、未来位置、最终答案及 EOS。图中行为信息接收位置，列为信息来源；显示每对 token 的平均残差，而不是概率。原始权重、残差、计数、距离支持及参考/自由生成两种结果保留在 models/*.json。最后预测答案位置的按来源注意力质量也单独保存。此校正控制经验距离均值，不能保证消除 RoPE 的所有影响；注意力本身也不构成因果信息流证明。", "",
              "## 线性探针", "", "![答案探针](answer_probes.png)", "", "![结构探针](structural_probes.png)", "",
              "只读最后三层 block 后 residual。答案同时训练跨位置共享的仿射分类器，以及 prompt/early/middle/late 各位置独立的仿射分类器；图中使用后者，避免共享读出方向掩盖位置差异。多标签探针预测当前节点全部后继节点或搜索历史。每张图只在其真实节点集合上评分，报告 F1、平衡准确率和整集合准确率，避免大量不存在的节点抬高准确率。train/val/test 查询严格分离；相同查询的不同位置不跨集合。超参数固定，训练迭代点只按 probe-val BCE 选择。", "",
              "输入对照使用当前 token/节点 embedding 加位置和目标出现特征；另外训练打乱标签的控制探针。结构输入对照尤其重要：固定图上当前节点身份本来就可以用于记忆其邻接表。高于对照才支持训练后的 activation 增添了可读信息。state 协议的 visited 标签来自已生成 PUSH 的并集，读出位置在本轮 visited 字段输出前；nodes 协议的标签是已输出的展开节点集合，语义不同，不能直接比较成同一种 visited。Direct/Filler/Path 没有通用搜索 visited 集，不强行添加这一任务。", "",
              "参考前缀与自由生成前缀分开评分，并保留目标尚未在 CoT 出现时的答案成绩。CoT 出现目标节点或终止动作后容易解码答案，不等于提前规划；能够线性读出，也不意味着原模型使用了这一信息。", "",
              "## Token 因果干预", "", "![Token 干预](token_interventions.png)", "",
              "对 DFS/BFS 全状态，在实际生成的中间状态 visited 中插入一个当前节点可达、且能通往目标的未访问节点；等长对照插入一个当前可达但不能通往目标的节点。插入后在 visited 字段结束处截断，重新生成 candidates/action 和所有后续 token，不保留过期的参考后缀。另设 visited 顺序反转的同集合对照。控制节点尽量匹配出度；匹配率及失败/长度上限单独记录。", "",
              "Path/节点序列用等长节点替换，比较通往目标性质改变与保持的对照。Direct 没有 CoT，改 query target 并由图算法重算标签；Filler 没有 visited，比较原始、半长和无 DOT。Direct 与 Filler 的干预与搜索状态干预不是同一实验，不能直接横向比较柱高。", "",
              "所有结果保存 paired bootstrap 区间、答案翻转率，以及 clean 本来正确样本中的受损率；图表按同一批可干预样本比较。前缀篡改可能与之前历史矛盾，属于分布外状态压力测试。只有相对匹配对照的特异性差异，才较支持模型依赖特定内容；不能把任意损坏后的掉分直接归因于搜索机制。这里执行的是 token 干预，并非 activation interchange patching。", "",
              "## 值得跟进的现象", ""]
    findings = []
    for r in models:
        name = r["entry"]["id"]
        for task, t in r["probes"]["targets"].items():
            if "reference_test" not in t:
                continue
            ref = t["reference_test"][2]
            free = t.get("generated_test", t["reference_test"])[2]
            base = t.get("generated_test", t["reference_test"])[3]
            metric = "balanced_accuracy" if task == "answer" else "f1"
            if ref[metric]-free[metric] > .15:
                findings.append(f"{name} 的 {task} 探针：参考前缀 {ref[metric]:.1%}，自由生成前缀 {free[metric]:.1%}。存在较大的前缀分布/执行差距。")
            if free[metric]-base[metric] > .15:
                findings.append(f"{name} 的 {task}：L6 {free[metric]:.1%}，输入对照 {base[metric]:.1%}，存在较明显的额外线性可读信息。")
            if base[metric]-free[metric] > .1:
                findings.append(f"{name} 的 {task}：输入对照 {base[metric]:.1%} 高于 L6 {free[metric]:.1%}。不能把此探针任务的成功全部归因于深层表示。")
        conditions = r.get("patching", {}).get("conditions", {})
        if "visited_route" in conditions:
            critical, control = conditions["visited_route"], conditions["visited_irrelevant"]
            contrast = r.get("patching", {}).get("paired_contrasts", {}).get("visited_route_minus_visited_irrelevant", {})
            ci = contrast.get("paired_ci95")
            ci_text = f"，paired bootstrap 95% 区间 [{ci[0]:+.1%}, {ci[1]:+.1%}]" if ci else ""
            findings.append(f"{name} 的 visited 干预（n={critical['n']}）：clean {conditions['clean']['accuracy']:.1%}，路径节点插入 {critical['accuracy']:.1%}，无关分支插入 {control['accuracy']:.1%}；两种插入的准确率差 {critical['accuracy']-control['accuracy']:+.1%}{ci_text}。")
        if r["rollout"]["limits"]:
            findings.append(f"{name} 有 {r['rollout']['limits']}/{r['rollout']['n']} 个生成达到分析长度上限，需要将这部分与模型自身错误区分。")
        replay = r.get("patching", {}).get("clean_replay_agreement")
        if replay is not None and replay < .98:
            findings.append(f"{name} 的原始前缀无修改重放，与首次生成最终答案的一致率为 {replay:.1%}。BF16 cached-decode 与重新 prefill、批内长度等数值差异可能改变轨迹；干预结论应以等长匹配对照为主，不能把 clean 重放差异算作干预损伤。")
        if "visited_route" in conditions:
            for cond in ["visited_route", "visited_irrelevant"]:
                c = conditions[cond]
                retained = c.get("inserted_node_retained_next_record")
                if retained is not None:
                    findings.append(f"{name}/{cond}：插入节点在下一条完整 visited 中保留 {retained:.1%}（n={c['next_record_available_n']}）；后续仍 PUSH 该已标记节点的比例 {c['inserted_node_pushed_after_edit_rate']:.1%}。")
    lines.extend(f"- {f}" for f in findings)
    lines += ["", "具体变化案例最多每模型保留 8 个，包含 query ID、被改前缀、原始/干预生成结果；见相应 models/*.json 的 patching.interesting_examples。没有保存大规模 activation、完整注意力张量或额外模型 checkpoint。", "",
              "## 复现", "", "```bash", ".venv/bin/python scripts/analyze_mechanisms.py --gpu 0 --resume",
              ".venv/bin/python -m cot_navigation.analysis.report --input research/mechanistic_20261018/results", "```", "",
              "脚本支持选择模型和分析阶段、调整样本数与探针迭代数。修改样本/方法配置时应指定新的 --output，避免混合旧结果。", "",
              "方法参考：[探针与控制任务](https://aclanthology.org/D19-1275/)、[patching 的指标与扰动方法](https://arxiv.org/abs/2309.16042)、[patching 结果解释](https://arxiv.org/abs/2404.15255)。"]
    index = lines.index("## 线性探针")
    lines[index:index] = ["逐层最强跨角色正向富集如下，仅为描述性摘要；箭头从被读取位置指向接收位置，括号为每对 token 的平均残差。", "", *attention_table(models), ""]
    index = lines.index("## Token 因果干预")
    lines[index:index] = ["关键探针数值如下；Direct 没有中段 CoT，使用 prompt 末尾。固定图等权平均三个世界，节点版 BFS 的 visited 是展开历史集合。", "", *scalar_tables(models, out), "",
                          "为排除目标已出现在生成过程中的简单线索，下面只保留 CoT 尚未出现目标节点的中段样本（平衡准确率）：", "", *unseen_answer_table(models), "",
                          "后继探针读取 candidates 字段之前的 residual；visited 探针读取 visited 字段之前的 residual。空缺表示任务不适用或样本只含一个类别。当前读出针对单一位置的 residual；列表也可能分布式保存在多个 CoT 位置及其 KV 中。整集合读出较差，不等于模型没有这些信息。探针失败只构成有限样本、有限线性读出方法下的阴性结果。", ""]
    index = lines.index("## 值得跟进的现象")
    lines[index:index] = ["Filler 还加入保持全部 token 与位置不变的注意力消融：block_dot_chain 禁止 DOT 读取更早的 DOT，保留 prompt 与自身；block_dot_prompt 禁止 DOT 读取 prompt，保留 DOT 间通信与自身。它们与缩短 DOT 的实验不同，原始 DOT 缩短会同时改变位置和计算。", "",
                          "节点对照尽量匹配出度，但不保证与当前节点的距离相同。插入节点下一轮仍出现，也可能是被模型重新 PUSH 的结果，因此同时记录后续 PUSH 率，不能只把保留率当成使用 visited 的证据。若 40 对样本中两种干预的答案完全相同，bootstrap 区间会退化为 [0,0]，这不是总体无差异的证明。", ""]
    audits = [r for r in models if "short_limit_audit" in r]
    if audits:
        audit_lines = ["", "## 生成长度审计", "",
                       "最终结果使用模型原生上下文窗口（本批为 4096；CLI 的 max-new-tokens=0 表示由上下文决定，不是无限长度）。初轮 1024-token 上限曾截断部分固定图轨迹，已对所有触发生成或干预上限的模型延长并重算。未触发上限的结果在更宽松上限下保持有效。", "",
                       "| 模型 | 旧截断数 | 原生上下文截断数 | 旧准确率 | 原生上下文准确率 |", "|---|---:|---:|---:|---:|"]
        for r in audits:
            a = r["short_limit_audit"]
            audit_lines.append(f"| {r['entry']['id']} | {a['rollout_limits']} | {r['rollout']['limits']} | {a['rollout_accuracy']:.2%} | {r['rollout']['accuracy']:.2%} |")
        index = lines.index("## 注意力")
        lines[index:index] = audit_lines+[""]
    # Concrete observations first; the full per-model diagnostics remain below.
    dfs = primary(models, "context_graph", "dfs")[0]
    middle = dfs["probes"]["targets"]["answer"]["phase_specific"]["middle"]["generated_test"]
    unseen = [v.get("goal_not_yet_in_cot", {}) for v in middle]
    filler = primary(models, "parametric_graph", "filler")
    average = lambda key: np.mean([r["patching"]["conditions"][key]["accuracy"] for r in filler])
    path = primary(models, "context_graph", "path")[0]
    dfs_contrast = dfs["patching"]["paired_contrasts"]["visited_route_minus_visited_irrelevant"]
    bfs_contrast = primary(models, "context_graph", "bfs")[0]["patching"]["paired_contrasts"]["visited_route_minus_visited_irrelevant"]
    query_to_path = np.asarray(path["attention"]["generated"]["corrected_per_pair"])[3, 0, ROLES.index("cot_node"), ROLES.index("query_dst")]
    intro = ["", "## 主要观察", "",
             f"- 新图 Path 第 4 层出现明显的 query target→路径节点注意力富集，每对 token 平均残差约 {query_to_path:+.3f}。目标条件在路径生成中的读取很突出，但单凭注意力不能证明搜索算法。",
             f"- 新图 DFS 的中段、目标尚未出现在 CoT 的样本上，答案平衡准确率：L4 {unseen[0]['balanced_accuracy']:.1%}，L6 {unseen[2]['balanced_accuracy']:.1%}，输入对照 {unseen[3]['balanced_accuracy']:.1%}（n={unseen[0]['n_rows']}，正例 {unseen[0]['n_positive']}）。后层线性可读性并不单调；这不是信息被删除或功能集中到某层的证明。",
             "- 固定图的后继节点集合在当前节点 embedding 与简单可见特征的输入对照下接近满分。静态邻接表查找足以解释这类高分，不能直接将其当作动态搜索证据。",
             f"- 固定图 Filler 的干预子集三图均值：原始 {average('clean'):.1%}，阻断 DOT→后续 DOT 通信 {average('block_dot_chain'):.1%}，阻断 prompt→DOT 读取 {average('block_dot_prompt'):.1%}。在该消融下，结果对 prompt 读取的依赖比对 DOT 间通信更明显；不能据此宣布所有 Filler 机制均无效。",
             f"- 新图 DFS 的路径节点 visited 插入与无关分支插入准确率差 {dfs_contrast['accuracy_difference']:+.1%}（n={dfs_contrast['n']}，paired bootstrap 区间 [{dfs_contrast['paired_ci95'][0]:+.1%}, {dfs_contrast['paired_ci95'][1]:+.1%}]）；新图 BFS 的相同对比为 {bfs_contrast['accuracy_difference']:+.1%}（n={bfs_contrast['n']}）。模型还会在后续 PUSH 已提前标记的节点，需区分忽略、修复矛盾历史与真正执行修改后的搜索状态。", ""]
    lines[3:3] = intro
    scope_index = lines.index("## checkpoint 与样本")+2
    lines[scope_index:scope_index] = ["架构为 6 层、1 个注意力头、d_model=128、FFN=512，约 1.22M 参数，RoPE/GELU、上下文 4096。新图为 N16/E24、4×4 分层图；固定图为 N32/E48、8×4 分层图。跨图任务差异同时包含图规模、深度和知识来源变化；协议间 token/FLOPs 也不同。新图探针使用 train/val/test=1024/256/256 条分层查询；固定图使用现有 split 的全部可用查询。注意力以 64 条 train 查询校准、64 条 test 查询观察。", ""]
    control_path = out/"probe_controls.json"
    if control_path.exists():
        controls = json.loads(control_path.read_text())["models"]
        assert all(r["entry"]["id"] in controls for r in models if not r["entry"]["auxiliary"])
        ctrl_lines = ["## 更强的输入对照与 Filler 读出", "",
                      "另训练只接收起点、终点 embedding 拼接的仿射分类器（256 维），使用相同 probe-train/val/test 查询，不读取图边或 CoT。这是排除节点身份捷径的偏保守对照；它也接受训练查询监督，其高分可能来自探针拟合节点边际关系，不能直接当作预训练模型已经完成推理的证据。", "",
                      "| 任务 | 协议 | 起终点身份对照（平衡准确率） | 打乱标签 |", "|---|---|---:|---:|"]
        for kind in ["context_graph", "parametric_graph"]:
            for fam in FAMILIES:
                records = [controls[r["entry"]["id"]] for r in primary(models, kind, fam)]
                score = np.mean([r["query_only_test"]["balanced_accuracy"] for r in records])
                shuffled = np.mean([r["shuffled_query_test"]["balanced_accuracy"] for r in records])
                ctrl_lines.append(f"| {'新图' if kind=='context_graph' else '固定图三图均值'} | {fam} | {score:.1%} | {shuffled:.1%} |")
        ctrl_lines += ["", "新图中段且目标未出现的精确同一子集上：", "",
                       "| 模型 | n | L4 | L5 | L6 | 起终点身份对照 |", "|---|---:|---:|---:|---:|---:|"]
        for fam in ["dfs", "bfs"]:
            r = primary(models, "context_graph", fam)[0]
            target = [x["goal_not_yet_in_cot"] for x in r["probes"]["targets"]["answer"]["phase_specific"]["middle"]["generated_test"]]
            ctrl = controls[r["entry"]["id"]]["query_only_middle_goal_unseen"]
            assert ctrl["n_rows"] == target[0]["n_rows"]
            ctrl_lines.append(f"| {fam} | {ctrl['n_rows']} | {target[0]['balanced_accuracy']:.1%} | {target[1]['balanced_accuracy']:.1%} | {target[2]['balanced_accuracy']:.1%} | {ctrl['balanced_accuracy']:.1%} |")
        ctrl_lines += ["", "Filler 的下表严格比较同一批干预查询。原始输出按完整词表 argmax 评分，YES/NO 限定输出检查格式问题，线性头在任何 DOT 之前的 prompt 激活上训练并读出。表中均为平衡准确率：", "",
                       "| 模型 | n | 无 DOT 原始输出 | 无 DOT 限定 YES/NO | 无 DOT 线性头 L6 | 起终点对照 | 原始 DOT 输出 | 无 DOT 合法答案率 |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in models:
            if r["entry"]["family"] != "filler" or r["entry"]["auxiliary"]:
                continue
            c = controls[r["entry"]["id"]]["matched_filler_readout"]
            values = [c['native_before_dot']['balanced_accuracy'], c['native_binary_before_dot']['balanced_accuracy'],
                      c['linear_before_dot'][2]['balanced_accuracy'], c['query_only']['balanced_accuracy'],
                      c['native_after_dot']['balanced_accuracy'], c['native_before_dot']['valid_answer_rate']]
            ctrl_lines.append("| "+" | ".join([r['entry']['id'], str(c['native_before_dot']['n']), *[f"{v:.1%}" for v in values]])+" |")
        ctrl_lines += ["", "固定图的起终点身份对照本身已经很强。因此，Filler 的 prompt 激活能被新线性头读出答案，尚不足以证明 CoT 提取了深层模型独有的隐藏推理能力。部分无 DOT 原始输出的失败还包含答案格式/读出问题。新图的相应结果接近机会水平。", "",
                       "中段按最终生成长度的分位点选择，仍可能带有轨迹长度相关线索。目标尚未出现与输入对照能排除部分简单解释；若要进一步证明提前规划，还需按固定生成步数及状态规模匹配比较。", ""]
        index = lines.index("## Token 因果干预")
        lines[index:index] = ctrl_lines
    info_index = lines.index("## 线性探针")
    lines[info_index:info_index] = ["字段级流向的解释：Direct 主要显示图边/QUERY 与语法标记之间的读取；Path 的中层对起点和目标 token 的读取突出；Filler 单独列出 DOT 的读入及 DOT 间路由，并与保持位置不变的消融对照；DFS/BFS 则出现历史 action→后续 candidates/frontier 等跨记录读取。这里的 action 来源可能是前一条记录，不是未来动作。它们描述路由模式，具体内容是否被使用仍需结合探针和匹配干预。", ""]
    if all("attention" in r for r in models if r["entry"]["kind"] == "parametric_graph" and not r["entry"]["auxiliary"]):
        index = lines.index("## 线性探针")
        lines[index:index] = ["固定图采用同样的按距离校正方法，各世界独立估计基线后再等权平均：", "", "![固定图注意力对照](attention_fixed_flow.png)", "",
                             "固定图没有输入边列表，因此图边角色为空；此图观察 QUERY 与 CoT 之间的路由。参数内知识的提取还涉及 embedding/FFN，不能只凭注意力定位。", ""]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(out/"findings.json", {"findings": findings, "models": len(models)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((args.input/"selection.json").read_text())
    models = [json.loads((args.input/"models"/f"{e['id']}.json").read_text()) for e in selection["models"]]
    for model in models:
        assert "probes" in model and "patching" in model, model["entry"]["id"]
    attention_plot(models, args.input)
    fixed_attention = all("attention" in r for r in models if r["entry"]["kind"] == "parametric_graph" and not r["entry"]["auxiliary"])
    if fixed_attention:
        attention_plot(models, args.input, "parametric_graph")
    answer_plot(models, args.input)
    structure_plot(models, args.input)
    patch_plot(models, args.input)
    report(models, selection, args.input)
    print(f"Saved {5 if fixed_attention else 4} figures and REPORT.md for {len(models)} models")


if __name__ == "__main__":
    main()
