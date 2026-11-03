"""Plot intervention-specific metrics from saved trials, without inference."""
from collections import defaultdict
import csv
import hashlib
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
COLORS = ["#2878B5", "#D65A42", "#279683", "#8B5DA8", "#AD8633"]
MARKERS = ["o", "s", "D", "^", "v"]
METHOD_COLORS = {
    "clean": "#626F80", "target_current": "#2878B5", "target_noncurrent": "#D65A42",
    "ordinary_current": "#279683", "target_vis_only": "#8B5DA8",
    "branch_reachable": "#2878B5", "branch_dead": "#279683",
    "swap_current": "#2878B5", "swap_noncurrent": "#279683",
    "push_alternative": "#D65A42", "push_alternative_reordered": "#279683",
    "candidate_remove": "#D65A42", "candidate_nonedge": "#8B5DA8", "candidate_reorder": "#279683",
    "force_found": "#2878B5", "force_exhaust": "#D65A42",
}

# Each curve is (family, method, label, optional expected final answer).
SPECS = [
    ("01_target_found", "目标位置干预：是否在当前步骤终止", "本步 FOUND 率", "found", [
        ("target_position", "target_current", "目标加入当前位置", None),
        ("target_position", "target_noncurrent", "目标加入非当前位置", None),
        ("target_position", "ordinary_current", "普通节点加入当前位置", None),
        ("target_position", "target_vis_only", "目标仅加入 VIS", None),
    ], "预期：目标在当前位置时应为 100%；另外三种条件本步均不应 FOUND。该指标只检查终止动作。"),
    ("02_target_state_answer", "目标位置干预：最终答案是否服从修改后的状态", "状态答案正确率", "oracle", [
        ("target_position", "clean", "未改动前缀（应 NO）", None),
        ("target_position", "target_current", "目标加入当前位置（应 YES）", None),
        ("target_position", "target_noncurrent", "目标加入非当前位置（应 YES）", None),
        ("target_position", "ordinary_current", "普通节点加入当前位置（应 NO）", None),
        ("target_position", "target_vis_only", "目标仅加入 VIS（应 NO）", None),
    ], "预期答案由修改后的状态机计算，区别于原始图标签；各曲线都以高值表示服从状态。"),
    ("03_branch_state_answer", "当前节点替换：能否区分可达分支与死分支", "状态答案正确率", "oracle", [
        ("frontier_branch", "clean", "未改动前缀（应 NO）", None),
        ("frontier_branch", "branch_reachable", "替换为可达分支（应 YES）", None),
        ("frontier_branch", "branch_dead", "替换为死分支（应 NO）", None),
    ], "可达/死分支由状态模拟器及独立可达性检查认证；生成失败记为不正确。"),
    ("04_frontier_next_state", "Frontier 顺序干预：下一条状态是否正确更新", "下一状态正确率", "next", [
        ("frontier_order", "clean", "未改动前缀", None),
        ("frontier_order", "swap_current", "交换当前与非当前节点", None),
        ("frontier_order", "swap_noncurrent", "仅交换两个非当前节点", None),
    ], "同时核对下一条 frontier 顺序和 VIS 集合；所有应有非终止转移的样本进入分母，缺失/非法转移记失败。"),
    ("05_push_next_state", "PUSH 分支干预：动作是否传递到下一条状态", "下一状态正确率", "next", [
        ("legal_push", "clean", "原始 PUSH 首个候选", None),
        ("legal_push", "push_alternative", "仅 PUSH 另一个合法候选", None),
        ("legal_push", "push_alternative_reordered", "PUSH 替代候选并同步重排 CAND", None),
    ], "分母为全部分支样本。仅改 PUSH 参数偏离训练中选首个展示候选的惯例；同步重排分支保持该惯例。"),
    ("06_push_complete_search", "PUSH 分支干预：后续搜索是否完整且合法", "完整后缀合法率", "complete", [
        ("legal_push", "clean", "原始 PUSH 首个候选", None),
        ("legal_push", "push_alternative", "仅 PUSH 另一个合法候选", None),
        ("legal_push", "push_alternative_reordered", "PUSH 替代候选并同步重排 CAND", None),
    ], "要求每步状态/候选/动作合法并完成搜索、回答正确；STOP 中断记为未完成，不等同于算法错误。"),
    ("07_candidate_following", "候选列表干预：动作是否跟随首个展示候选", "PUSH 首个展示候选的比例", "first_candidate", [
        ("candidates", "clean", "原始候选列表", None),
        ("candidates", "candidate_remove", "删除首个候选", None),
        ("candidates", "candidate_nonedge", "首项换成非出邻居", None),
        ("candidates", "candidate_reorder", "交换前两个候选", None),
    ], "此图测行为跟随，不是正确率：非出邻居曲线越高，表示越常 PUSH 注入的假节点。"),
    ("08_candidate_action_legal", "候选列表干预：下一动作是否符合图搜索规则", "首动作合法率", "action", [
        ("candidates", "clean", "原始候选列表", None),
        ("candidates", "candidate_remove", "删除首个候选", None),
        ("candidates", "candidate_nonedge", "首项换成非出邻居", None),
        ("candidates", "candidate_reorder", "交换前两个候选", None),
    ], "按图的真实出邻居、VIS 和当前节点验证动作；强制给定的错误 CAND 字段本身不计为模型动作错误。"),
    ("09_terminal_answer_following", "终止动作干预：最终答案是否跟随指定的终止含义", "指定答案输出率", "answer", [
        ("false_found", "force_found", "强制 FOUND 后输出 YES", 1),
        ("false_exhaust", "force_exhaust", "强制 EXHAUST 后输出 NO", 0),
        ("false_found", "clean", "原 NO 样本未改动：输出 NO", 0),
        ("false_exhaust", "clean", "原 YES 样本未改动：输出 YES", 1),
    ], "强制动作曲线测终止提示控制；两条未改动曲线提供各自样本组的原答案正确率对照。"),
]


def score(rows, metric, expected):
    if metric == "next":
        values = [r["next_state_correct"] for r in rows if r["next_state_correct"] is not None]
    elif metric == "answer":
        values = [r["pred"] == expected for r in rows]
    else:
        key = {"found": "first_record_found", "oracle": "oracle_correct",
               "complete": "suffix_complete_valid", "first_candidate": "pushed_first_displayed",
               "action": "first_action_legal"}[metric]
        # Missing records/failures must not disappear from a success-rate denominator.
        values = [r.get(key) is True for r in rows]
    return sum(values), len(values)


def draw_figures():
    output = HERE / "metric_figures"
    output.mkdir(exist_ok=True)
    groups = defaultdict(list)
    sources = []
    for path in sorted((HERE / "results/models").glob("*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        assert data["complete"] and data["independent_prefix_annotations"]
        sources.append({"path": str(path.relative_to(HERE)), "sha256": hashlib.sha256(raw).hexdigest()})
        for row in data["rows"]:
            groups[(data["entry"]["id"], row["family"], row["method"])].append(row)
    assert len(sources) == 20
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "svg.fonttype": "path", "figure.facecolor": "white",
        "savefig.facecolor": "white"})
    points = []
    guide = ["# 按干预目的选择指标", "", "沿用已有冻结评估结果，未重新推理。每张图：行是 DFS/BFS，列是新图/固定图；横轴为 DAG setup。", "",
             "n 为该面板各曲线的指标分母范围；不同实验样本组不同，具体分母见 plot_data.csv。无适用样本保留空缺。", "",
             "Frontier/终止动作等反事实修改可能与历史冲突；仅改 PUSH 参数也偏离训练的首候选惯例。", ""]
    guide.extend(["| 图 | 纵轴指标 | 文件 |", "|---|---|---|"])
    for stem, title, ylabel, _, _, _ in SPECS:
        guide.append(f"| {title} | {ylabel} | [PNG]({(output / (stem + '.png')).as_posix()}) · [SVG]({(output / (stem + '.svg')).as_posix()}) |")
    guide.append("")
    for stem, title, ylabel, metric, curves, note in SPECS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.3))
        fig.subplots_adjust(left=.08, right=.985, top=.86, bottom=.23, hspace=.47, wspace=.15)
        fig.suptitle(title, fontsize=18, y=.965, fontweight="bold")
        fig.text(.08, .91, f"纵轴：{ylabel}。各图的指标定义不同，请按图下注释解读。", fontsize=10, color="#66758A")
        for i, protocol in enumerate(("dfs_state", "bfs_state")):
            for j, kind in enumerate(("context", "fixed")):
                ax = axes[i, j]
                counts = defaultdict(list)
                for k, (family, method, label, expected) in enumerate(curves):
                    y = []
                    for scale in SCALES:
                        model = f"{kind}_{scale}_{protocol}"
                        rows = groups.get((model, family, method), [])
                        successes, n = score(rows, metric, expected)
                        value = successes / n if n else None
                        y.append(value if n else np.nan)
                        if n:
                            counts[scale].append(n)
                        points.append({"figure": stem, "metric": metric, "ylabel": ylabel,
                            "model": model, "family": family, "method": method, "curve_label": label,
                            "sample_n": len(rows), "denominator_n": n, "successes": successes, "value": value})
                    ax.plot(range(5), y, color=METHOD_COLORS[method], marker=MARKERS[k], lw=2,
                            linestyle="--" if family == "false_exhaust" and method == "clean" else "-",
                            markersize=5.7, markeredgecolor="white", markeredgewidth=.6, label=label)
                labels = []
                for scale, label in zip(SCALES, LABELS):
                    ns = counts[scale]
                    nlabel = (str(min(ns)) if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}") if ns else "无样本"
                    labels.append(label + (f"\nn={nlabel}" if ns else "\n无适用样本"))
                ax.set_title(protocol[:3].upper() + " · " + ("新图" if kind == "context" else "固定图"), loc="left", fontsize=12)
                ax.set_xticks(range(5), labels, fontsize=9)
                ax.set_xlim(-.2, 4.2)
                ax.set_ylim(-.035, 1.055)
                ax.set_yticks([0, .25, .5, .75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.grid(axis="y", alpha=.20)
                ax.set_axisbelow(True)
                if j == 0:
                    ax.set_ylabel(ylabel, fontsize=10)
                if i == 1:
                    ax.set_xlabel("DAG setup", labelpad=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .095),
                   ncol=2, frameon=False, fontsize=10, handlelength=2.5, columnspacing=2)
        fig.text(.08, .047, note, fontsize=9, color="#66758A")
        fig.text(.08, .021, "冻结模型；每类最多 24 个查询。数据和原始推理结果不变，完整定义与逐点分母见 METRIC_GUIDE.md / plot_data.csv。", fontsize=9, color="#66758A")
        fig.savefig(output / (stem + ".png"), dpi=200)
        fig.savefig(output / (stem + ".svg"))
        plt.close(fig)
        guide.extend([f"## {title}", "", f"纵轴：**{ylabel}**。{note}", "",
                      f"![{title}]({(output / (stem + '.png')).as_posix()})", ""])
    with (output / "plot_data.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    (output / "source_manifest.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    (HERE / "METRIC_GUIDE.md").write_text("\n".join(guide), encoding="utf-8")
    for source in sources:
        assert hashlib.sha256((HERE / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert len(points) == sum(len(spec[4]) for spec in SPECS) * 20
    print(f"Created {len(SPECS)} metric-specific figures (PNG/SVG), {len(points)} points; source results unchanged.")


if __name__ == "__main__":
    draw_figures()
