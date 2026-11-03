#!/usr/bin/env python3
"""Instance attention during canonical DFS-state traces.

In-context curriculum DFS is shown on held-out 4x4 graphs. Fixed DFS is the
10x10 model on its own graph. Rows are action tokens; cells are last-layer
attention mass. One head. Teacher-forced reference traces, not free generation.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cot_navigation.analysis.common import records  # noqa: E402
from cot_navigation.config import load_config  # noqa: E402
from cot_navigation.data.dataset import ContextStore, ParametricWorld  # noqa: E402
from cot_navigation.models.transformer import build_model  # noqa: E402
from cot_navigation.protocols.tokenizer import Tokenizer  # noqa: E402

OUT = ROOT / "figures/scale_v1/dfs_cot_attention_examples.png"
JSON = ROOT / "studies/block_membership_probe/dfs_cot_attention_examples.json"
DEVICE = "cuda:0"
COLUMNS = [
    "Query source",
    "Query target",
    "Graph",
    "Current stack",
    "Current visited",
    "Current candidates",
    "Earlier nodes",
    "Self",
    "Other",
]
CONTEXT_RUN = ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state"
FIXED_RUN = ROOT / "outputs/scale_v1/launches/scale_v1_s0/fixed/l10/stage_b/dfs_state"
CONTEXT_DATA = ROOT / "data/scale_v1/context_l4w4_n16_m24_train50000_s0"
FIXED_DATA = ROOT / "data/scale_v1/fixed_l10w10_n100_m150_s0/main-seed0"


def load_model(run: Path, device: torch.device):
    cfg = load_config([run / "resolved_config.yaml"])
    tok = Tokenizer.from_config(cfg)
    model = build_model(cfg.model, len(tok), tok.pad).to(device).eval()
    state = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    step = int(state["step"])
    del state
    return cfg, tok, model, step


def world_token(cfg, tok: Tokenizer, world: Path) -> int:
    for index, raw in enumerate(cfg.train.world_roots):
        if (ROOT / raw).resolve() == world.resolve():
            return tok.id(f"<WORLD_{index}>")
    raise KeyError(world)


@torch.inference_mode()
def last_attention(model, tokens: list[int], device: torch.device) -> np.ndarray:
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _logits, weights = model(ids, return_attn=True)
    layer = weights[-1][0, 0].float().cpu().numpy()
    del weights
    return layer


def node_index(tokens_at: list[int], node_tokens: list[int]) -> str:
    inverse = {token: index for index, token in enumerate(node_tokens)}
    names = []
    for token in tokens_at:
        if token in inverse:
            names.append(str(inverse[token]))
    return ",".join(names)


def bucket_of(position: int, token: int, info: dict) -> int:
    if position == info["receiver"]:
        return 7
    if position == info["q_src"]:
        return 0
    if position == info["q_dst"]:
        return 1
    if position in info["graph"]:
        return 2
    if position in info["cur_stack"]:
        return 3
    if position in info["cur_visited"]:
        return 4
    if position in info["cur_cands"]:
        return 5
    if position in info["earlier_nodes"]:
        return 6
    return 8


def masses(attn: np.ndarray, receiver: int, info: dict, tokens: list[int]) -> np.ndarray:
    row = attn[receiver, : receiver + 1]
    row = row / max(float(row.sum()), 1e-8)
    info = {**info, "receiver": receiver}
    out = np.zeros(len(COLUMNS), dtype=np.float64)
    for key in range(receiver + 1):
        out[bucket_of(key, tokens[key], info)] += row[key]
    return out


def graph_positions(tokens, tok: Tokenizer, node_tokens: list[int], prompt_len: int) -> set[int]:
    node_set = set(node_tokens)
    return {
        i for i in range(max(prompt_len - 4, 0))
        if tokens[i] in node_set or tok.name(tokens[i]) in {"<EDGE>", "<WORLD_0>", "<WORLD_9>"}
        or tok.name(tokens[i]).startswith("<WORLD_") or tok.name(tokens[i]) == "<EDGE>"
    }


def example_matrix(attn, tokens, tok: Tokenizer, node_tokens: list[int], prompt_len: int) -> dict:
    node_set = set(node_tokens)
    graph = graph_positions(tokens, tok, node_tokens, prompt_len)
    parsed = records(tokens, prompt_len, tok)
    rows, labels = [], []
    for step_index, record in enumerate(parsed, start=1):
        action_pos = record["separators"][2] + 1
        action = tok.name(tokens[action_pos])
        arg = ""
        if action == "<PUSH>" and action_pos + 1 < record["end"]:
            arg = node_index([tokens[action_pos + 1]], node_tokens)
        label = f"{step_index} {action[1:-1]}" + (f" {arg}" if arg else "")
        a, b, c = record["separators"]
        earlier = range(prompt_len, record["start"])
        info = {
            "q_src": prompt_len - 3,
            "q_dst": prompt_len - 2,
            "graph": graph,
            "cur_stack": set(range(record["start"], a)),
            "cur_visited": set(range(a + 1, b)),
            "cur_cands": set(range(b + 1, c)),
            "earlier_nodes": {i for i in earlier if tokens[i] in node_set},
        }
        # World token is graph context on the fixed prompt; edge nodes already match node_set.
        rows.append(masses(attn, action_pos, info, tokens))
        labels.append(label)
    answer_pos = len(tokens) - 3  # last ';' predicts YES/NO; tokens end with answer, EOS
    # Find the semicolon immediately before the answer more reliably.
    answer_pos = next(i for i in range(len(tokens) - 1, prompt_len - 1, -1) if tokens[i] == tok.id(";"))
    info = {
        "q_src": prompt_len - 3,
        "q_dst": prompt_len - 2,
        "graph": graph,
        "cur_stack": set(),
        "cur_visited": set(),
        "cur_cands": set(),
        "earlier_nodes": {i for i in range(prompt_len, answer_pos) if tokens[i] in node_set},
    }
    rows.append(masses(attn, answer_pos, info, tokens))
    labels.append("Answer")
    return {"matrix": np.vstack(rows), "labels": labels, "n_records": len(parsed)}


def choose_context(store: ContextStore, tok: Tokenizer) -> list[int]:
    ids = store.indices("test")
    scored = []
    for index in ids.tolist():
        row = store.render(int(index), "dfs_state", tok)
        n = len(records(row.tokens, row.prompt_len, tok))
        if 4 <= n <= 10:
            scored.append((n, int(store.arrays["label"][index]), int(store.arrays["distance"][index]), int(index)))
    yes = [item for item in scored if item[1] == 1]
    no = [item for item in scored if item[1] == 0]
    yes.sort()
    no.sort()
    # Two reachable traces of different lengths, then one unreachable.
    picked = [yes[0][3], yes[len(yes) // 2][3], no[0][3]]
    return picked


def choose_fixed(world: ParametricWorld, tok: Tokenizer) -> list[int]:
    ids = world.indices("test", view="full")
    q = world.queries
    scored = []
    for index in ids.tolist():
        row = world.render_query(int(index), "dfs_state", tok)
        n = len(records(row.tokens, row.prompt_len, tok))
        if 3 <= n <= 8:
            scored.append((n, int(q["label"][index]), int(q["distance"][index]), int(index)))
    yes = sorted(item for item in scored if item[1] == 1)
    no = sorted(item for item in scored if item[1] == 0)
    return [yes[0][3], yes[len(yes) // 2][3], no[len(no) // 3][3]]


def prepare_fixed(world, tok, index, world_id):
    row = world.render_query(index, "dfs_state", tok)
    tokens = [world_id, *row.tokens]
    q = world.queries
    return {
        "tokens": tokens,
        "prompt_len": row.prompt_len + 1,
        "node_tokens": world.node_vocab_ids(tok),
        "source": int(q["s_topo"][index]),
        "target": int(q["t_topo"][index]),
        "distance": int(q["distance"][index]),
        "label": int(q["label"][index]),
        "index": index,
    }


def prepare_context(store, tok, index):
    row = store.render(index, "dfs_state", tok)
    return {
        "tokens": row.tokens,
        "prompt_len": row.prompt_len,
        "node_tokens": store.node_vocab_ids(index, tok),
        "source": int(store.arrays["s_topo"][index]),
        "target": int(store.arrays["t_topo"][index]),
        "distance": int(store.arrays["distance"][index]),
        "label": int(store.arrays["label"][index]),
        "index": index,
    }


def draw(panels: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.6))
    image = None
    for ax, panel in zip(axes.ravel(), panels):
        grid = panel["matrix"]
        image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
        ax.set_xticks(range(len(COLUMNS)))
        ax.set_xticklabels(COLUMNS, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(panel["labels"])))
        ax.set_yticklabels(panel["labels"], fontsize=7)
        ax.set_title(panel["title"], fontsize=9, pad=6)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                if value < 0.08:
                    continue
                color = "white" if value < 0.62 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.14, wspace=0.42, hspace=0.55)
    colorbar = fig.add_axes((0.92, 0.28, 0.014, 0.48))
    fig.colorbar(image, cax=colorbar, label="Attention mass")
    fig.suptitle("DFS-state attention at each action, last layer", fontsize=12)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".png.tmp")
    Image.open(buffer).convert("RGB").save(tmp, format="PNG")
    tmp.replace(OUT)
    print(OUT)


def main() -> None:
    device = torch.device(DEVICE)
    panels = []
    saved = []
    ctx_cfg, ctx_tok, ctx_model, ctx_step = load_model(CONTEXT_RUN, device)
    store = ContextStore(CONTEXT_DATA)
    for index in choose_context(store, ctx_tok):
        item = prepare_context(store, ctx_tok, index)
        attn = last_attention(ctx_model, item["tokens"], device)
        parsed = example_matrix(attn, item["tokens"], ctx_tok, item["node_tokens"], item["prompt_len"])
        title = (f"In-context curriculum, 4×4\n"
                 f"{'YES' if item['label'] else 'NO'} {item['source']}→{item['target']}, "
                 f"{item['distance']} hops")
        panels.append({**parsed, "title": title})
        saved.append({"setting": "context_curriculum_4x4", "step": ctx_step, **item_meta(item),
                      "labels": parsed["labels"], "matrix": parsed["matrix"].round(4).tolist()})
        print(title.replace("\n", " "), parsed["labels"])
    del ctx_model
    torch.cuda.empty_cache()

    fix_cfg, fix_tok, fix_model, fix_step = load_model(FIXED_RUN, device)
    world = ParametricWorld(FIXED_DATA)
    world_id = world_token(fix_cfg, fix_tok, world.root)
    for index in choose_fixed(world, fix_tok):
        item = prepare_fixed(world, fix_tok, index, world_id)
        attn = last_attention(fix_model, item["tokens"], device)
        parsed = example_matrix(attn, item["tokens"], fix_tok, item["node_tokens"], item["prompt_len"])
        title = (f"Fixed 10×10\n"
                 f"{'YES' if item['label'] else 'NO'} {item['source']}→{item['target']}, "
                 f"{item['distance']} hops")
        panels.append({**parsed, "title": title})
        saved.append({"setting": "fixed_l10", "step": fix_step, **item_meta(item),
                      "labels": parsed["labels"], "matrix": parsed["matrix"].round(4).tolist()})
        print(title.replace("\n", " "), parsed["labels"])
    draw(panels)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps({
        "receiver": "action token; the Answer row is the semicolon that predicts YES/NO",
        "layer": "last transformer layer",
        "head": 1,
        "columns": COLUMNS,
        "trace": "canonical DFS-state trace, teacher forced",
        "examples": [{k: v for k, v in row.items() if k != "tokens" and k != "node_tokens"} for row in saved],
    }, indent=2))
    print(JSON)


def item_meta(item: dict) -> dict:
    return {key: item[key] for key in ("index", "source", "target", "distance", "label", "prompt_len")}


if __name__ == "__main__":
    main()
