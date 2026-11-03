"""Summarize saved trials and draw legacy plus intervention-specific figures."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
SCALES = ["l4", "l6", "l8", "l10", "curriculum"]
LABELS = ["N16", "N36", "N64", "N100", "课程至 N100"]
GROUPS = [
    ("Frontier 内容：目标位置与分支替换", [
        ("target_position", "target_current", "目标加入当前节点位置"),
        ("target_position", "target_noncurrent", "目标加入非当前位置"),
        ("target_position", "ordinary_current", "普通节点加入当前位置（对照）"),
        ("target_position", "target_vis_only", "目标仅加入 VIS（对照）"),
        ("frontier_branch", "branch_reachable", "当前节点替换为可达分支"),
        ("frontier_branch", "branch_dead", "当前节点替换为死分支（对照）"),
    ]),
    ("Frontier 顺序与合法 PUSH 分支", [
        ("frontier_order", "swap_current", "交换当前与非当前节点"),
        ("frontier_order", "swap_noncurrent", "仅交换非当前节点（对照）"),
        ("legal_push", "push_alternative", "强制 PUSH 另一个合法候选"),
        ("legal_push", "push_alternative_reordered", "PUSH 替代候选并同步重排 CAND"),
    ]),
    ("候选列表与终止动作", [
        ("candidates", "candidate_remove", "删除首个候选"),
        ("candidates", "candidate_nonedge", "首个候选替换为非出邻居"),
        ("candidates", "candidate_reorder", "交换前两个候选（合法对照）"),
        ("false_found", "force_found", "不可达样本强制 FOUND"),
        ("false_exhaust", "force_exhaust", "到达目标后强制 EXHAUST"),
    ]),
]
COLORS = ["#2878B5", "#D65A42", "#279683", "#8B5DA8", "#AD8633", "#626F80"]
MARKERS = ["o", "s", "D", "^", "v", "P"]
plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "svg.fonttype": "path", "figure.facecolor": "white",
    "savefig.facecolor": "white", "text.color": "#233247"})

models = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE / "results/models").glob("*.json"))]
assert len(models) == 20 and all(d["complete"] and d["checkpoint_unchanged"] for d in models)
trials, summary = [], []
for model in models:
    entry = model["entry"]
    groups = defaultdict(list)
    for row in model["rows"]:
        trials.append({"model": entry["id"], **row})
        groups[(row["family"], row["method"])].append(row)
    for (family, method), rows in groups.items():
        record = {"model": entry["id"], "protocol": entry["protocol"], "kind": entry["kind"],
                  "scale": entry["scale"], "family": family, "method": method, "n": len(rows)}
        for key in ("binary_flip", "valid_answer_pair", "oracle_correct", "generation_failure",
                    "interrupted", "forced_stop", "suffix_complete_valid", "premature_found", "immediate_answer", "first_record_found",
                    "first_candidates_correct", "first_action_legal", "next_state_correct", "pushed_first_displayed", "pushed_injected_nonedge"):
            values = [r[key] for r in rows if r.get(key) is not None]
            record[key + "_n"] = len(values)
            record[key + "_count"] = sum(values) if values else 0
            record[key + "_rate"] = sum(values) / len(values) if values else None
        record["yes_rate"] = sum(r["pred"] == 1 for r in rows) / len(rows)
        record["clean_yes_rate"] = sum(r["clean_pred"] == 1 for r in rows) / len(rows)
        summary.append(record)


def csv_write(name, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


csv_write("summary.csv", summary)
csv_write("trials.csv", trials)
(HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lookup = {(r["protocol"], r["kind"], r["scale"], r["family"], r["method"]): r for r in summary}
for protocol in ("dfs_state", "bfs_state"):
    fig = plt.figure(figsize=(14, 14.5))
    gs = fig.add_gridspec(6, 2, height_ratios=[1, .21, 1, .17, 1, .21],
                          left=.07, right=.985, top=.91, bottom=.075, hspace=.39, wspace=.10)
    fig.suptitle(protocol[:3].upper() + "：搜索过程干预后的答案翻转率", fontsize=20, y=.978, fontweight="bold")
    fig.text(.07, .943, "新图", fontsize=14, fontweight="bold")
    fig.text(.55, .943, "固定图", fontsize=14, fontweight="bold")
    for group_index, (title, methods) in enumerate(GROUPS):
        for j, kind in enumerate(("context", "fixed")):
            ax = fig.add_subplot(gs[group_index * 2, j])
            for k, (family, method, label) in enumerate(methods):
                values = [lookup.get((protocol, kind, scale, family, method)) for scale in SCALES]
                y = [r["binary_flip_rate"] if r else np.nan for r in values]
                ax.plot(range(5), y, label=label, color=COLORS[k], marker=MARKERS[k],
                        lw=1.9, markersize=5.5, markeredgecolor="white", markeredgewidth=.6)
            ax.set_title(title, fontsize=11, loc="left", pad=9)
            ax.set_ylim(-.025, 1.04)
            ax.set_yticks(np.arange(0, 1.01, .25))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_xticks(range(5), LABELS, fontsize=9.5)
            ax.set_xlim(-.18, 4.18)
            ax.grid(axis="y", alpha=.22)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("答案翻转率")
            else:
                ax.tick_params(labelleft=False)
            if group_index == 2:
                ax.set_xlabel("DAG setup", labelpad=8)
        legend_ax = fig.add_subplot(gs[group_index * 2 + 1, :])
        legend_ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center", ncol=3 if len(methods) >= 5 else 2,
                         frameon=False, fontsize=9.5, handlelength=2.4, columnspacing=2.0)
    fig.text(.07, .045, "各曲线均与同一实验、同一截断位置的未改动前缀比较；仅统计 YES / NO 互换，分母为全部配对样本。", fontsize=9.5, color="#66758A")
    fig.text(.07, .025, "每模型每类实验最多 24 个查询；曲线来自不同适用样本组，具体 n、生成失败及局部正确率见数据表。仅合法 PUSH 和 CAND 重排保持完整历史合法。", fontsize=9, color="#66758A")
    fig.savefig(HERE / (protocol[:3] + ".png"), dpi=200)
    fig.savefig(HERE / (protocol[:3] + ".svg"))
    plt.close(fig)

def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


lines = ["# DFS/BFS 显式搜索过程干预", "",
    "使用原 20 个 validation-best 模型：新图/固定图 × N16/N36/N64/N100/课程至 N100 × DFS/BFS；冻结推理，不训练、不修改 checkpoint。",
    "",
    "每类实验按原测试查询顺序选择最多 24 个适用查询，每查询只选一个参考状态。前缀由合法参考搜索生成，选择过程不读取模型输出；从可容纳的记录中优先选轨迹中点。完整合法分支由独立 semantic_slots 检查。",
    "",
    "答案翻转率 = 有效 YES/NO 配对中发生翻转的数量 / 全部干预配对数量。失败单列，不计作翻转。模型继续生成至答案或原上下文/STOP 边界，不额外缩短生成预算。不同实验的样本组不同，不能把曲线差直接解释为干预强度的因果差异。",
    "",
    "## 实验语义", "",
    "- 目标位置：只用原始 NO 查询；把目标/普通死节点分别加入当前位置或非当前位置，并同步加入 VIS。目标在当前位置应本步 FOUND；目标在非当前位置应等到它成为当前节点才 FOUND。只加入 VIS 是目标 token 提示对照。",
    "- Frontier 分支：只用原始 NO 查询，替换当前节点为一个到目标可达/不可达的节点，同时更新 VIS。两个状态由搜索模拟器及独立图可达性检查认证为 YES/NO。",
    "- Frontier 顺序：交换当前位置与另一端；可用时增加仅交换两个非当前节点的对照。frontier 和 VIS 集合不变，历史可能不一致。",
    "- 合法 PUSH：同一合法前缀有至少两个候选，分别强制 PUSH 第一个/第二个；另设同步重排 CAND 的分支。任意合法候选均允许作为搜索优先级，不要求固定排序。两条分支均可从原查询合法到达，最终正确标签应相同。",
    "  单改 PUSH 参数仍是图搜索合法动作，但偏离训练中 PUSH 首个展示候选的序列惯例；同步重排 CAND 的分支则同时遵守该惯例。独立语义检查器验证的是图搜索合法性，不把邻居排序固定为唯一合法优先级。",
    "- CAND：删除首项/把首项替换为未访问非出邻居/交换前两项。前两种制造冗余字段冲突；不能把跟随或纠正字段直接等同于算法忠实性。另记录是否 PUSH 注入的非邻居。",
    "- 终止动作：NO 查询非终止处强制 FOUND；YES 查询真正到达目标的记录强制 EXHAUST。最终输出跟随动作只能说明终止提示控制，不能证明此前搜索正确。",
    "",
    "除合法 PUSH 和 CAND 重排外，干预通常与已有历史或图结构冲突。候选列表干预的局部合法动作仍按图/VIS 判定；强制给定的候选字段不重复计为模型候选错误。强制错误终止动作本身也不归为模型生成错误。",
    "",
    "完整后缀合法率要求完成搜索并给出相符答案；STOP 中断记为未完成，不能把所有未完成都解释为算法错误。各方法的 interrupted/forced_stop/first_error 可在逐查询表中分开检查。",
    "",
    "## N100 关键结果", "",
    "| 设置 | 干预 | n | 答案翻转 | YES | 首动作合法 | 下一状态正确 | 完整后缀合法 |",
    "|---|---|---:|---:|---:|---:|---:|---:|"]
for kind in ("context", "fixed"):
    for protocol in ("dfs_state", "bfs_state"):
        for family, method, name in [
            ("target_position", "target_current", "目标→当前"),
            ("target_position", "target_noncurrent", "目标→非当前"),
            ("frontier_branch", "branch_reachable", "替换为可达分支"),
            ("legal_push", "push_alternative", "合法 PUSH 替代分支"),
            ("candidates", "candidate_nonedge", "CAND 注入非邻居"),
            ("false_found", "force_found", "强制 FOUND"),
            ("false_exhaust", "force_exhaust", "强制 EXHAUST"),
        ]:
            r = lookup.get((protocol, kind, "l10", family, method))
            if not r:
                continue
            lines.append("| " + " | ".join([("新图" if kind == "context" else "固定图") + " " + protocol[:3].upper(), name, str(r["n"]),
                pct(r["binary_flip_rate"]), pct(r["yes_rate"]), pct(r["first_action_legal_rate"]),
                pct(r["next_state_correct_rate"]), pct(r["suffix_complete_valid_rate"])]) + " |")
lines += ["", "首动作合法：模型生成的当前动作是否符合注入状态；强制动作不计入该指标。下一状态正确：首动作后第一条完整状态的 frontier/VIS 是否正确；所有需要非终止状态转移的样本均进入分母，缺失或非法转移记为失败。终止状态和强制终止动作没有该指标。强制错误终止输入下完整后缀合法率为零是构造结果，不单独作模型错误证据。",
          "", "## 结果解释", "",
          "- N100 的四种模型中，强制 FOUND 后均输出 YES（24/24），强制 EXHAUST 后均输出 NO（24/24）。翻转率有时低于 100%，是因为 clean 续写本身可能已经回答错误，不能混淆 YES 率与翻转率。",
          "- 目标加入当前节点位置并不稳定触发终止。N100 新图 DFS/BFS 的本步 FOUND 比例分别为 10/24、0/24；固定图 DFS/BFS 均为 0/24。最终 YES 的比例与此不同，说明答案翻转不能替代逐步执行检查。此类输入和原历史冲突，失败不能单独证明模型在自然轨迹中忽略 frontier。",
          "- N100 固定图 BFS：只改 PUSH 参数，下一状态正确为 0/24；同步重排 CAND 后为 24/24。后者完整合法续写为 18/24，clean 为 19/24。这支持模型对候选顺序/动作参数一致性的强依赖，不能直接推广为所有合法搜索分支都执行失败。",
          "- N100 固定图 DFS：两种 PUSH 替代的下一状态都正确（24/24），但只改参数时后续全部出现候选错误；同步重排 CAND 后完整合法续写为 16/24，clean 为 19/24。局部状态更新正确与完整搜索正确是不同能力。",
          "- CAND 首项替换为非出邻居后，N100 新图 DFS/BFS 均在 24/24 样本 PUSH 该假候选，固定图 DFS/BFS 为 23/24、20/24。显式候选表确实能控制动作，但这种控制不保证图搜索正确；低答案翻转率也不代表过程完全未受影响。",
          "", "这些结果表明不同 CoT 字段的作用不同：终止动作强控制最终答案，候选表强控制部分局部动作，而 frontier 的任意反事实修改未被稳定执行。尚不能据此声称自然生成的全部 CoT 都不忠实，也没有证据支持显式记录与搜索过程完美逐步对应。",
          "", "本轮使用合法参考搜索前缀，上轮 visited 实验使用模型自由生成前缀，不能直接拼接两轮曲线当作同一组样本比较。",
          "", "## 文件", "", "- METRIC_GUIDE.md / metric_figures：按干预选取指标的新版主图、SVG 和逐点绘图数据。", "- dfs.png / bfs.png：保留的旧版答案翻转图。",
          "- summary.csv：全部模型、全部实验及局部指标的汇总，含分母。",
          "- trials.csv：每个查询的预测、翻转、局部检查与前缀哈希。",
          "- results/models：逐模型结果与每类最多两个完整示例。",
          "- results/verification.json：独立合法性与确定性复核结果。", ""]
(HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote two figures (PNG/SVG), tables, and REPORT.md", len(trials), "trials", len(summary), "summary rows")

# The intervention-specific figures are the primary result presentation.
from plot_metrics import draw_figures
draw_figures()
