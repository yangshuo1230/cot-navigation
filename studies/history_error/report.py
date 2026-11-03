"""Summarize delayed error readout, graph controls and precision controls."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
verification = json.loads((HERE / "verification.json").read_text(encoding="utf-8"))
assert verification["status"] == "PASS" and len(verification["models"]) == 4
NAMES = ["context_l10_dfs_state", "context_l10_bfs_state", "fixed_l10_dfs_state", "fixed_l10_bfs_state"]
TITLES = ["新图 DFS", "新图 BFS", "固定图 DFS", "固定图 BFS"]
TOKEN = ["pre_error", "error", "token_1", "token_4", "token_16", "token_64"]
TOKEN_LABELS = ["错前 1", "错误处", "+1", "+4", "+16", "+64"]
RECORD = ["record_1_vis", "record_2_vis", "record_4_vis"]
RECORD_LABELS = ["后续第 1 条", "后续第 2 条", "后续第 4 条"]
ALL = [*TOKEN[1:], *RECORD]
ALL_LABELS = [*TOKEN_LABELS[1:], "第 1 条 VIS", "第 2 条 VIS", "第 4 条 VIS"]
plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path", "figure.facecolor": "white", "savefig.facecolor": "white"})

models, rows, manifest = {}, [], []
for precision, folder in (("bf16", "results"), ("fp32", "results_fp32")):
    for path in sorted((HERE / folder / "models").glob("*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        assert data["complete"] and data["checkpoint_unchanged"]
        model = path.stem
        models[(precision, model)] = data
        manifest.append({"path": str(path.relative_to(HERE)), "sha256": hashlib.sha256(raw).hexdigest()})
        for probe in data["probes"]:
            domain = "graph_trained" if probe.get("graph_trained") else "main"
            for r in probe["test"]:
                rows.append({"model": model, "precision": precision, "domain": domain,
                             "architecture": probe["architecture"], "seed": probe["seed"], **r})
            for r in probe.get("graph_zero_shot", []):
                rows.append({"model": model, "precision": precision, "domain": "graph_zero_shot",
                             "architecture": probe["architecture"], "seed": probe["seed"], **r})
assert len(models) == 8
groups = defaultdict(list)
for row in rows:
    key = tuple(row[k] for k in ("model", "precision", "domain", "architecture", "anchor", "layer"))
    groups[key].append(row)
summary = []
lookup = {}
for key, values in groups.items():
    record = dict(zip(("model", "precision", "domain", "architecture", "anchor", "layer"), key))
    record.update(pairs=values[0]["pairs"], queries=values[0]["queries"], probe_seeds=len(values))
    for metric in ("auroc", "paired_rank_accuracy", "balanced_accuracy"):
        record[metric + "_mean"] = float(np.mean([r[metric] for r in values]))
        record[metric + "_std"] = float(np.std([r[metric] for r in values]))
    summary.append(record)
    lookup[key] = record


def get(model, precision, domain, architecture, anchor, layer="L6", metric="auroc"):
    row = lookup.get((model, precision, domain, architecture, anchor, layer))
    return (row[metric + "_mean"], row[metric + "_std"]) if row else (np.nan, np.nan)


def save(fig, stem):
    fig.savefig(HERE / (stem + ".png"), dpi=210)
    fig.savefig(HERE / (stem + ".svg"))
    plt.close(fig)


def curves(stem, anchors, labels, title, xlabel, metric="auroc"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.075, right=.985, top=.85, bottom=.22, wspace=.15, hspace=.43)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=.965)
    ylabel = "AUROC" if metric == "auroc" else "配对排序正确率"
    styles = [
        ("bf16", "mlp64", "L6", "MLP · L6 · BF16", "#2878B5", "o", "-"),
        ("fp32", "mlp64", "L6", "MLP · L6 · FP32", "#D65A42", "s", "--"),
        ("bf16", "linear", "L6", "Linear · L6 · BF16", "#279683", "D", "-"),
        ("bf16", "mlp64", "edited_and_current_node", "历史编辑节点 + 当时当前节点身份对照", "#8B5DA8", "^", ":"),
    ]
    fig.text(.075, .91, "同一查询只改一个未被 PUSH 的候选；后续输入 token、frontier 和 VIS 均相同。", fontsize=10, color="#66758A")
    for ax, model, name in zip(axes.flat, NAMES, TITLES):
        for precision, arch, layer, label, color, marker, ls in styles:
            values = [get(model, precision, "main", arch, a, layer, metric) for a in anchors]
            y = np.array([v[0] for v in values]); sd = np.array([v[1] for v in values])
            ax.plot(range(len(anchors)), y, label=label, color=color, marker=marker, linestyle=ls, lw=1.9, markersize=5)
            ax.fill_between(range(len(anchors)), y-sd, y+sd, color=color, alpha=.10)
        ax.axhline(.5, color="#7D8998", ls="--", lw=.8)
        ax.set_ylim(0, 1.04); ax.set_yticks([0,.25,.5,.75,1])
        ax.set_xticks(range(len(labels)), labels, fontsize=9)
        ax.set_title(name + " · 测试 256 对", loc="left", fontsize=12)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.18)
        if ax in axes[1]: ax.set_xlabel(xlabel, labelpad=9)
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, ncol=2, loc="lower center", bbox_to_anchor=(.54,.09), frameon=False, fontsize=10)
    fig.text(.075,.047,"每个位置/层分别拟合探针；阴影仅为 3 次探针初始化标准差，不是样本置信区间。0.5 为随机基线。",fontsize=9,color="#66758A")
    fig.text(.075,.022,"身份对照可访问此前编辑节点及当时当前节点的 embedding；固定图高分可能来自图关系记忆。未测试自然错误或自动纠错。",fontsize=9,color="#66758A")
    save(fig,stem)


curves("01_token_delay", TOKEN, TOKEN_LABELS, "历史候选错误：后续 token 激活还能读出错误吗？", "相对错误 token 的离散读出位置")
curves("02_record_delay", RECORD, RECORD_LABELS, "历史候选错误：在后续 VIS 结束处读出", "各记录 VIS 结束、CAND 开始前")
curves("S1_paired_record_rank", RECORD, RECORD_LABELS, "同一查询内：错误历史的分数是否高于正确历史？", "各记录 VIS 结束、CAND 开始前", metric="paired_rank_accuracy")

fig, axes = plt.subplots(2,2,figsize=(13,8.8))
fig.subplots_adjust(top=.86,bottom=.18,left=.075,right=.89,hspace=.50,wspace=.20)
fig.suptitle("层与读出位置：历史错误的可读性",fontsize=18,fontweight="bold",y=.965)
fig.text(.075,.914,"BF16 · 64 hidden MLP · 三次初始化 AUROC 均值；每个格子单独训练探针。",fontsize=10,color="#66758A")
for ax,model,name in zip(axes.flat,NAMES,TITLES):
    matrix=np.array([[get(model,"bf16","main","mlp64",a,layer)[0] for a in ALL] for layer in ["L1","L2","L3","L4","L5","L6"]])
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap="RdBu_r",aspect="auto")
    ax.set_xticks(range(len(ALL)),ALL_LABELS,rotation=30,ha="right",fontsize=9)
    ax.set_yticks(range(6),["L1","L2","L3","L4","L5","L6"])
    ax.set_title(name,loc="left",fontsize=12)
    for i in range(6):
        for j in range(len(ALL)):
            ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=9,color="white" if abs(matrix[i,j]-.5)>.30 else "#233247")
cax=fig.add_axes([.92,.26,.018,.48]);fig.colorbar(im,cax=cax,label="AUROC")
fig.text(.075,.035,"固定 token 偏移与按记录对齐的位置并不等价；同一个记录锚点对应的实际 token 距离因轨迹而异。",fontsize=9,color="#66758A")
save(fig,"03_layer_heatmap")

fig,axes=plt.subplots(1,2,figsize=(13,5.8))
fig.subplots_adjust(top=.78,bottom=.28,left=.075,right=.985,wspace=.16)
fig.suptitle("图信息对照：相同 CoT，在不同图上合法或违法",fontsize=18,fontweight="bold",y=.97)
fig.text(.075,.88,"每张图均改一条边、保持节点/边数和原查询标签；两类的全部 CoT 文本相同。每设置测试 237 对。",fontsize=10,color="#66758A")
aa=["error","token_1","token_4","record_1_vis"];ll=["错误处","+1 token","+4 token","下一条 VIS 结束"]
for ax,model,name in zip(axes,NAMES[:2],TITLES[:2]):
    for precision,color,marker,label in [("bf16","#2878B5","o","原探针迁移 · BF16"),("fp32","#D65A42","s","原探针迁移 · FP32")]:
        y=[get(model,precision,"graph_zero_shot","mlp64",a)[0] for a in aa]
        ax.plot(range(4),y,color=color,marker=marker,label=label,lw=2,ls="-" if precision=="bf16" else "--")
    y=get(model,"bf16","graph_trained","mlp64","record_1_vis")[0]
    ax.scatter([3],[y],color="#279683",marker="D",s=65,label="该位置单独训练的图关系探针",zorder=5)
    ax.axhline(.5,color="#8B5DA8",ls=":",label="节点身份对照 = 0.5")
    ax.set_title(name,loc="left",fontsize=12);ax.set_xticks(range(4),ll,fontsize=9)
    ax.set_ylim(0,1.04);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel("AUROC");ax.grid(axis="y",alpha=.18)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.53,.085),ncol=2,frameon=False,fontsize=10)
fig.text(.075,.025,"仅绘制在两张图上都没有新增后续违规的位置；不把这一结果解释为模型主动监控或修正错误。",fontsize=9,color="#66758A")
save(fig,"04_graph_control")

for name,data in (("all_probe_runs.csv",rows),("summary.csv",summary)):
    with (HERE/name).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        writer.writeheader();writer.writerows(data)
(HERE/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(HERE/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")

def fmt(model,anchor,domain="main",precision="bf16"):
    mean,sd=get(model,precision,domain,"mlp64",anchor)
    return f"{mean:.3f} ± {sd:.3f}"

text=["# 后续激活中的历史 CoT 错误读出", "",
"完成 N100 新图/固定图 × DFS/BFS 四个 validation-best checkpoint 的冻结读出；每个再做 FP32 数值对照。未训练导航模型，未创建 W&B run，未保存探针权重或 activation 缓存。", "",
"独立CPU验证为 PASS：重建核对4864个主配对、2194个图对照配对，BF16/FP32的查询和位置相同，错误前激活严格一致，checkpoint哈希均未改变。这些计数包含train/val/test，不能当作独立测试集数量。", "",
"## 数据构造与问题范围", "",
"每个模型 train/val/test 分别为 768/192/256 个原查询，每查询一对历史，按既有查询 split 隔离。每对从合法参考轨迹构造：将 CAND 最后一个、未被 PUSH 的合法邻居 b 换成同层未访问非邻居 z；其余 token 完全不变。首个候选 a 与 PUSH a 保持不变，后续记录的实际 frontier/VIS/动作仍合法且两边一致。标签是过去是否存在该次候选注释错误，不是最终答案或当前局部状态是否错误。", "",
"选取的轨迹必须容纳全部读出位置，且不包含 FOUND/EXHAUST/YES/NO。只测人工注释错误和教师提供的共同后续，没有测试自然错误、自由续写或自动纠错。", "",
"读取 L1–L6 block 后 residual，位置为错前1、错误token、+1/+4/+16/+64 token，以及后续第1/2/4条记录 VIS 结束处。各位置/层独立训练 Linear 和64 hidden GELU MLP，最多600更新，验证BCE选状态，三个探针初始化。标准化和超参沿用此前探针实现。", "",
"错前1位置是分支身份负对照：此时两边实际都尚未出错，读出器试图区分哪个分支随后会被编辑；两边前缀完全相同，预期只能达到随机水平。", "",
"初次特征提取被错误前激活一致性断言拦截：BF16 配对跨批可产生数值差异。修正为配对始终同批后正式重跑；首轮未产生探针结果。正式 BF16/FP32 全部配对的错误前激活差为0，探针AUROC均为0.5。", "",
"## L6 主结果", "", "以下为 BF16 MLP AUROC，均值 ± 三次初始化标准差；这是初始化差异，不是样本置信区间。每格测试256对。", "",
"| 设置 | 错误当场 | +1 token | +4 token | +64 token | 第1条 VIS | 第2条 VIS | 第4条 VIS |",
"|---|---:|---:|---:|---:|---:|---:|---:|"]
for model,name in zip(NAMES,TITLES):
    text.append("| "+" | ".join([name,*[fmt(model,a) for a in ["error","token_1","token_4","token_64",*RECORD]]])+" |")
text += ["", "![token距离]("+(HERE/"01_token_delay.png").as_posix()+")", "",
"![记录距离]("+(HERE/"02_record_delay.png").as_posix()+")", "",
"## 图关系与身份对照", "",
"新图另构造相同 CoT 文本的图配对：合法图把当前节点的边 u→b 改成 u→z；错误图在一个未展开节点上做等量无关边替换，保留原候选错误。两边均只有一次图编辑，节点数/边数/层结构不变，原查询的可达标签不变。只读到下一条VIS结束，避免变更的边在后续CAND中引起新错误。", "",
"新图 DFS 的 graph train/val/test 为701/171/237对；BFS为680/168/237对，均来自主实验对应split。原探针直接迁移到该对照；另外在下一条VIS结束位置独立训练图对照探针。", "",
"| 设置 | 原探针迁移：错误处 | 原探针迁移：下一条VIS | 在下一条VIS独立训练 |",
"|---|---:|---:|---:|"]
for model,name in zip(NAMES[:2],TITLES[:2]):
    text.append("| "+" | ".join([name,fmt(model,"error","graph_zero_shot"),fmt(model,"record_1_vis","graph_zero_shot"),fmt(model,"record_1_vis","graph_trained")])+" |")
text += ["", "该对照中两类 CoT 和节点身份完全相同，身份探针AUROC为0.5。因此新图DFS在错误当场约0.99的图对照迁移，支持该位置的错误可读性依赖图关系；同样的证据未延续到下一条VIS结束。", "",
"固定图身份对照可见历史编辑节点和当时当前节点的embedding，能够记忆固定图边关系，MLP多数位置约0.996–1.000。故固定图较高的残差读出不单独证明存在专用错误监控状态。所有后续位置的当前token身份对照均为0.5。", "",
"![图关系对照]("+(HERE/"04_graph_control.png").as_posix()+")", "",
"## 如何解释", "",
"- 新图DFS：错误处很容易读出，而且同一CoT换图后仍能区分合法性；但+4/+64及下一条VIS处接近随机，第2条VIS处约0.60出现弱读出。未发现稳定、持续的历史错误读出。", 
"- 新图BFS：该checkpoint在本实验的即时与延迟读出都弱。不能推广为所有BFS模型都不具备该能力。", 
"- 固定图BFS：下一条VIS结束处AUROC约0.944，第2条约0.693，第4条约0.516，显示较强的短期、字段相关信息；它可能是候选工作记忆或节点关系信息，并不等同主动意识到错误。", 
"- 固定图DFS：独立样本AUROC较弱，但同查询配对排序在第1/2/4条VIS分别约0.798/0.784/0.708，说明不能把跨查询AUROC接近0.5直接解释为表示完全相同或不存在任何痕迹。配对排序需要反事实正确历史作参照，不能当作单条自然历史的检测成功率。", 
"- FP32与BF16的主趋势接近，未恢复稳定的长程读出，低精度舍入不足以解释主要结果。", 
"- 固定token偏移和按字段对齐不是同一位置分布：记录长度不同，不能从两张图拼出单调遗忘曲线。", 
"- 此处只证明有限探针能否读出。高分不证明主模型使用该信息纠错；低分不证明任意读出器都无法恢复信息。", "",
"![层位置热力图]("+(HERE/"03_layer_heatmap.png").as_posix()+")", "",
"## 文件与复现", "", "- summary.csv/json：所有域、层、位置、精度的均值和初始化标准差。",
"- all_probe_runs.csv：每个初始化的AUROC、配对排序与balanced accuracy及分母。",
"- results/models 与 results_fp32/models：原始结果和少量完整配对示例。",
"- verification.json：独立CPU重建、因果负对照、身份对照与checkpoint不变验证。",
"- S1_paired_record_rank.png/svg：逐查询配对排序补充图。", "",
"服务器项目根目录：", "", "```bash", ".venv/bin/python studies/history_error/run.py --gpu 7",
".venv/bin/python studies/history_error/run.py --gpu 7 --precision fp32 --output studies/history_error/results_fp32",
"OMP_NUM_THREADS=2 .venv/bin/python studies/history_error/verify.py", "```", "",
"本地重画：`python G:\\CoT\\history_error\\report.py`。训练目标模型和原始数据保持不变。", ""]
(HERE/"REPORT.md").write_text("\n".join(text),encoding="utf-8")
print("Wrote four main figures, one supplement, report and tables",len(summary),"summary rows")
