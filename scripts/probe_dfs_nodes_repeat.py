#!/usr/bin/env python3
"""How DFS nodes keeps an already emitted node from winning the next-token score."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import plot_dfs_cot_attention as base  # noqa: E402
import plot_dfs_nodes_attention as probe  # noqa: E402
import plot_path_dag_attention as pathfig  # noqa: E402
from cot_navigation.protocols.render import render_task  # noqa: E402

DEVICE = "cuda:0"


def mean(values):
    return float(sum(values) / len(values)) if values else float("nan")


@torch.no_grad()
def forward(model, tokens, device, hide_layers=None, spans=None):
    patched = []
    if hide_layers:
        def wrapped(orig):
            def attend(q, k, v, keep, force_math, causal=False):
                length = q.shape[-2]
                index = torch.arange(length, device=q.device)
                visible = index.view(1, 1, 1, length) <= index.view(1, 1, length, 1)
                for query, start in spans:
                    if start < query:
                        visible[0, 0, query, start:query] = False
                visible[0, 0].fill_diagonal_(True)
                return orig(q, k, v, visible, True, causal=False)
            return attend
        for layer in hide_layers:
            module = model.blocks[layer].attn
            patched.append((module, module._attend))
            module._attend = wrapped(module._attend)
    try:
        ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
        capture = {}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(ids, return_attn=True, capture=capture)
        logits, weights = out
        lens = []
        for resid in capture["resid"]:
            hidden = model.final_norm(resid.detach().float())
            lens.append(model.lm_head(hidden)[0].cpu())
        attn = torch.stack([weight[0, 0].float().cpu() for weight in weights])
        return logits[0].float().cpu(), attn, lens
    finally:
        for module, original in patched:
            module._attend = original


def decisions(item, prompt_len):
    rows = []
    for dec_pos, last, parent, gold in probe.node_positions(item["steps"], prompt_len):
        rows.append((dec_pos, last, parent, gold, parent != last))
    return rows


def scores(table, emitted, remaining, gold):
    logits = table
    pred = int(logits.argmax())
    graph = emitted + remaining
    graph_logits = logits[graph]
    graph_prob = torch.softmax(graph_logits, dim=0)
    emitted_share = float(graph_prob[: len(emitted)].sum()) if emitted else 0.0
    gold_logit = float(logits[gold])
    emitted_max = float(logits[emitted].max()) if emitted else float("nan")
    return {
        "repeat": float(pred in set(emitted)),
        "self": float(pred == emitted[-1]) if emitted else 0.0,
        "earlier": float(pred in set(emitted[:-1])) if len(emitted) > 1 else 0.0,
        "gold": float(pred == gold),
        "share": emitted_share,
        "gap": gold_logit - emitted_max,
    }


def measure(model, tok, items, device, hide_sets):
    base_rows = {key: [] for key in ("repeat", "self", "earlier", "gold", "share", "gap")}
    by_kind = {kind: {key: [] for key in base_rows} for kind in ("go", "back")}
    lens_share = [[] for _ in range(6)]
    attn_on_earlier = [[] for _ in range(6)]
    hidden = {name: {key: [] for key in ("repeat", "earlier", "gold", "share", "gap")} for name in hide_sets}
    example = None
    for item in items:
        row = render_task(
            tok, "dfs_nodes", item["label"], item["prompt"], item["adj"],
            item["source"], item["target"], item["node_tokens"],
        )
        logits, attn, lens = forward(model, row.tokens, device)
        prompt_len = row.prompt_len
        steps = decisions(item, prompt_len)
        spans = [(dec_pos, prompt_len) for dec_pos, *_ in steps if dec_pos > prompt_len]
        usable = []
        for dec_pos, last, parent, gold, back in steps:
            if dec_pos + 1 >= len(row.tokens) or row.tokens[dec_pos + 1] != item["node_tokens"][gold]:
                break
            order_tokens = [item["node_tokens"][last]]
            # earlier nodes sit at prompt_len .. dec_pos
            earlier_nodes = row.tokens[prompt_len:dec_pos]
            emitted = earlier_nodes + order_tokens
            seen = {last}
            # reconstruct seen from positions already passed; gold is not included
            # emitted tokens are exactly the expansion prefix including current
            remaining = [token for node, token in enumerate(item["node_tokens"]) if token not in emitted and node < len(item["adj"])]
            # node index alignment: item node_tokens[i] is topo i. Filter to real nodes.
            remaining = [token for token in item["node_tokens"] if token not in set(emitted)]
            gold_token = item["node_tokens"][gold]
            stat = scores(logits[dec_pos], emitted, remaining, gold_token)
            kind = "back" if back else "go"
            for key, value in stat.items():
                base_rows[key].append(value)
                by_kind[kind][key].append(value)
            for layer in range(6):
                lens_share[layer].append(scores(lens[layer][dec_pos], emitted, remaining, gold_token)["share"])
                if earlier_nodes:
                    attn_on_earlier[layer].append(float(attn[layer, dec_pos, prompt_len:dec_pos].sum()))
            usable.append((dec_pos, emitted, remaining, gold_token, kind))
            if example is None and back and len(emitted) >= 3:
                show = emitted[-3:] + [gold_token]
                prob = torch.softmax(logits[dec_pos, show], dim=0)
                example = (item["index"], [tok.name(token) for token in show], [round(float(p), 3) for p in prob], kind)
            seen.add(gold)
        del logits, attn, lens
        for name, layers in hide_sets.items():
            masked, _attn, _lens = forward(model, row.tokens, device, hide_layers=layers, spans=spans)
            for dec_pos, emitted, remaining, gold_token, _kind in usable:
                stat = scores(masked[dec_pos], emitted, remaining, gold_token)
                for key in hidden[name]:
                    hidden[name][key].append(stat[key])
            del masked
    return {
        "n": len(base_rows["repeat"]),
        "base": {key: mean(value) for key, value in base_rows.items()},
        "go": {key: mean(value) for key, value in by_kind["go"].items()},
        "back": {key: mean(value) for key, value in by_kind["back"].items()},
        "go_n": len(by_kind["go"]["repeat"]),
        "back_n": len(by_kind["back"]["repeat"]),
        "lens_share": [mean(value) for value in lens_share],
        "prefix_attn": [mean(value) for value in attn_on_earlier],
        "hide": {name: {key: mean(value) for key, value in rows.items()} for name, rows in hidden.items()},
        "example": example,
    }


def main():
    device = torch.device(DEVICE)
    families = [
        ("layered", ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_nodes", [
            ("4×4", ROOT / "data/scale_v1/context_l4w4_n16_m24_train50000_s0"),
            ("6×6", ROOT / "data/scale_v1/context_l6w6_n36_m54_train50000_s0"),
            ("10×10", ROOT / "data/scale_v1/context_l10w10_n100_m150_train50000_s0"),
        ]),
        ("hard", ROOT / "outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_nodes", [
            (name, root) for name, root in pathfig.SIZES
        ]),
    ]
    hide_sets = {
        "L1": [0], "L4": [3], "L5": [4], "L6": [5], "L5+L6": [4, 5], "all": [0, 1, 2, 3, 4, 5],
    }
    for family, run, sizes in families:
        _cfg, tok, model, step = base.load_model(run, device)
        print(f"\n{family} step {step}", flush=True)
        for name, root in sizes:
            items = probe.choose(pathfig.load_test(root), tok)
            result = measure(model, tok, items, device, hide_sets)
            base_row = result["base"]
            print(
                f"{name} n={result['n']} repeat {base_row['repeat']:.3f} earlier {base_row['earlier']:.3f} "
                f"self {base_row['self']:.3f} gold {base_row['gold']:.3f} "
                f"share {base_row['share']:.3f} gap {base_row['gap']:.2f}",
                flush=True,
            )
            print(
                f"  go n={result['go_n']} repeat {result['go']['repeat']:.3f} share {result['go']['share']:.3f} "
                f"back n={result['back_n']} repeat {result['back']['repeat']:.3f} share {result['back']['share']:.3f}",
                flush=True,
            )
            print("  lens share", " ".join(f"{v:.3f}" for v in result["lens_share"]), flush=True)
            print("  prefix attn", " ".join(f"{v:.3f}" for v in result["prefix_attn"]), flush=True)
            for hide_name, rows in result["hide"].items():
                print(
                    f"  hide {hide_name}: repeat {rows['repeat']:.3f} earlier {rows['earlier']:.3f} "
                    f"gold {rows['gold']:.3f} share {rows['share']:.3f} gap {rows['gap']:.2f}",
                    flush=True,
                )
            print("  example", result["example"], flush=True)


if __name__ == "__main__":
    main()
