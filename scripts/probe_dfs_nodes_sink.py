#!/usr/bin/env python3
"""Where the next node is read when the current node has no outgoing edge.

Layered DFS-nodes, one 4×4 trace (test index 2, V204 then V95) and the same
comparison on every sink step of the 4×4 test sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import plot_dfs_cot_attention as base  # noqa: E402
import plot_dfs_nodes_attention_examples as ex  # noqa: E402
import plot_path_dag_attention as pathfig  # noqa: E402

RUN = ROOT / "outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_nodes"
DATA = ROOT / "data/scale_v1/context_l4w4_n16_m24_train50000_s0"


def name_of(tok, item, node):
    return tok.name(item["node_tokens"][node]).strip("<>")


def pos_name(tok, item, pos):
    plen = item["prompt_len"]
    edges = item["edges"]
    if pos < 3 * len(edges):
        edge, slot = divmod(pos, 3)
        src, dst = edges[edge]
        role = ("EDGE", name_of(tok, item, src), name_of(tok, item, dst))[slot]
        return f"e{edge} {role}"
    if pos == plen - 4:
        return "QUERY"
    if pos == plen - 3:
        return "source"
    if pos == plen - 2:
        return "target"
    if pos == plen - 1:
        return "|"
    index = pos - plen
    order = item["order"]
    if index < len(order):
        return name_of(tok, item, order[index])
    return f"t{pos}"


@torch.no_grad()
def run_forward(model, tokens, device, hides=None):
    """hides: list of (layer, query, keys). keys may include the query itself."""
    ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
    positions = torch.arange(ids.shape[1], device=device)[None]
    wrapped = []
    if hides:
        by_layer = {}
        for layer, query, keys in hides:
            by_layer.setdefault(layer, []).append((query, set(keys)))
        for layer, pairs in by_layer.items():
            attn = model.blocks[layer].attn
            original = attn._attend

            def make(pairs, original):
                def wrapped_attend(q, k, v, keep, force_math, causal=False):
                    length = q.shape[-2]
                    index = torch.arange(length, device=q.device)
                    visible = index.view(1, 1, 1, length) <= index.view(1, 1, length, 1)
                    for query, keys in pairs:
                        for key in keys:
                            if 0 <= key < length and key <= query:
                                visible[0, 0, query, key] = False
                    empty = ~visible[0, 0].any(dim=-1)
                    if empty.any():
                        visible[0, 0, empty, :] = False
                        visible[0, 0, empty, empty.nonzero(as_tuple=False)[:, 0]] = True
                    return original(q, k, v, visible, True, causal=False)

                return wrapped_attend

            attn._attend = make(pairs, original)
            wrapped.append((attn, original))
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            x = model.embed(ids)
            layers = []
            for block in model.blocks:
                hidden = block.norm1(x)
                delta, weights = block.attn(hidden, positions, return_attn=True)
                mid = x + delta
                post = mid + block.ffn(block.norm2(mid))
                layers.append((x[0].detach().float(), mid[0].detach().float(), post[0].detach().float(), weights[0, 0].detach().float()))
                x = post
            logits = model.lm_head(model.final_norm(x.float()))[0].detach().float()
        return [(a.cpu(), b.cpu(), c.cpu(), d.cpu()) for a, b, c, d in layers], logits.cpu()
    finally:
        for attn, original in wrapped:
            attn._attend = original


def lens(model, resid, device):
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model.lm_head(model.final_norm(resid.to(device).float())).float().cpu()


def top_text(tok, scores, k=6):
    values, indices = torch.topk(scores, k)
    parts = []
    for value, index in zip(values.tolist(), indices.tolist()):
        parts.append(f"{tok.name(index).strip('<>')} {value:.2f}")
    return ", ".join(parts)


def gold_stats(scores, gold_id):
    prob = torch.softmax(scores, dim=-1)
    order = torch.argsort(scores, descending=True)
    rank = int((order == gold_id).nonzero()[0])
    return float(prob[gold_id]), rank, tok_at(scores)


def tok_at(scores):
    return int(scores.argmax())


def main():
    device = torch.device("cuda:0")
    _cfg, tok, model, step = base.load_model(RUN, device)
    model.eval()
    item = ex.prepare(pathfig.load_test(DATA), tok, 2)
    order = item["order"]
    parent = item["parent"]
    plen = item["prompt_len"]
    print("step", step, "order", [name_of(tok, item, n) for n in order])
    print("query", name_of(tok, item, item["source"]), "->", name_of(tok, item, item["target"]))
    # Decision that writes V95: current is order[2] == V204.
    dec = plen + 2
    cur, gold = order[2], order[3]
    print("decision", name_of(tok, item, cur), "->", name_of(tok, item, gold), "parent", name_of(tok, item, parent[gold]), "pos", dec)
    print("outs of current", item["adj"][cur])
    print("children of parent", [name_of(tok, item, v) for v in item["adj"][parent[gold]]])
    edge_pos = {}
    for index, (src, dst) in enumerate(item["edges"]):
        if src == parent[gold]:
            edge_pos[name_of(tok, item, dst)] = (3 * index + 2, 3 * index + 3)
            print(
                f"  edge {name_of(tok, item, src)}->{name_of(tok, item, dst)} dest {3 * index + 2} "
                f"({pos_name(tok, item, 3 * index + 2)}) +3 {3 * index + 3} ({pos_name(tok, item, 3 * index + 3)}) "
                f"token there {tok.name(item['tokens'][3 * index + 3]).strip('<>')}"
            )
    gold_name = name_of(tok, item, gold)
    gold_id = item["node_tokens"][gold]
    dest, plus = edge_pos[gold_name]
    parent_pos = plen + order.index(parent[gold])
    cur_pos = dec

    layers, logits = run_forward(model, item["tokens"], device)
    print("\n== attention at the decision, top keys ==")
    for layer, (_pre, _mid, _post, attn) in enumerate(layers):
        row = attn[dec, : dec + 1]
        values, indices = torch.topk(row, 8)
        bits = [f"{pos_name(tok, item, int(i))}@{int(i)} {v:.3f}" for v, i in zip(values, indices)]
        print(f"L{layer + 1}", " ".join(bits))
        focus = {
            "self": cur_pos,
            "parent": parent_pos,
            "gold dest": dest,
            "gold +3": plus,
        }
        for child, (dpos, ppos) in edge_pos.items():
            focus[f"{child} dest"] = dpos
            focus[f"{child} +3"] = ppos
        print("   ", " ".join(f"{label} {float(row[p]):.3f}" for label, p in focus.items()))

    print("\n== logit lens at the decision ==")
    for layer, (pre, mid, post, _attn) in enumerate(layers):
        for tag, resid in (("in", pre[dec]), ("attn", mid[dec]), ("ffn", post[dec])):
            scores = lens(model, resid, device)
            prob = torch.softmax(scores, dim=-1)
            rank = int((torch.argsort(scores, descending=True) == gold_id).nonzero()[0])
            kids = " ".join(
                f"P({name_of(tok, item, node)}) {float(prob[item['node_tokens'][node]]):.3f}"
                for node in item["adj"][parent[gold]]
            )
            print(f"L{layer + 1} {tag:4} P({gold_name}) {float(prob[gold_id]):.3f} rank {rank + 1} {kids} top {top_text(tok, scores, 4)}")
    final = logits[dec]
    prob = torch.softmax(final, dim=-1)
    print(f"final P({gold_name}) {float(prob[gold_id]):.3f} argmax {tok.name(int(final.argmax())).strip('<>')}")

    print("\n== what each candidate position unembeds to, after each layer ==")
    watch = {
        "V204 self": cur_pos,
        "V14 prefix": parent_pos,
        "V209 prefix": plen,
        "V95 dest": dest,
        "V95 +3": plus,
        "pos0": 0,
        "bar": plen - 1,
    }
    for child, (dpos, ppos) in edge_pos.items():
        watch[f"{child} dest"] = dpos
        watch[f"{child} +3"] = ppos
    for layer, (_pre, _mid, post, _attn) in enumerate(layers):
        print(f"L{layer + 1}")
        for label, pos in watch.items():
            scores = lens(model, post[pos], device)
            prob = torch.softmax(scores, dim=-1)
            print(f"  {label:12} {top_text(tok, scores, 3)}   P({gold_name}) {float(prob[gold_id]):.3f}")

    print("\n== what V14's own position attends to ==")
    for layer, (_pre, _mid, _post, attn) in enumerate(layers):
        row = attn[parent_pos, : parent_pos + 1]
        values, indices = torch.topk(row, 6)
        bits = [f"{pos_name(tok, item, int(i))} {v:.3f}" for v, i in zip(values, indices)]
        print(f"L{layer + 1}", " ".join(bits))

    print("\n== hide one source from this decision ==")
    trials = {
        "gold dest": [dest],
        "gold +3": [plus],
        "parent prefix": [parent_pos],
        "self": [cur_pos],
        "start prefix": [plen],
        "all parent dest": [d for d, _p in edge_pos.values()],
        "all parent +3": [p for _d, p in edge_pos.values()],
        "sibling dests": [d for child, (d, _p) in edge_pos.items() if child != gold_name],
    }
    for label, keys in trials.items():
        for layer in range(6):
            _layers, out = run_forward(model, item["tokens"], device, hides=[(layer, dec, keys)])
            scores = out[dec]
            prob = torch.softmax(scores, dim=-1)
            print(
                f"{label:16} L{layer + 1} P({gold_name}) {float(prob[gold_id]):.3f} "
                f"argmax {tok.name(int(scores.argmax())).strip('<>')}"
            )
        _layers, out = run_forward(
            model, item["tokens"], device, hides=[(layer, dec, keys) for layer in range(6)]
        )
        scores = out[dec]
        prob = torch.softmax(scores, dim=-1)
        print(
            f"{label:16} all P({gold_name}) {float(prob[gold_id]):.3f} "
            f"argmax {tok.name(int(scores.argmax())).strip('<>')}"
        )


def child_logits(tok, scores, item, nodes):
    return " ".join(
        f"{name_of(tok, item, node)} {float(scores[item['node_tokens'][node]]):.2f}"
        for node in nodes
    )


def followup(model, tok, item, device):
    order, parent, plen = item["order"], item["parent"], item["prompt_len"]
    layers, logits = run_forward(model, item["tokens"], device)
    print("\n== other no-edge steps in this trace ==")
    for index, node in enumerate(order[:-1]):
        gold = order[index + 1]
        if parent[gold] == node or item["adj"][node]:
            continue
        dec = plen + index
        row = layers[5][3][dec, : dec + 1]
        parent_pos = plen + order.index(parent[gold])
        scores = logits[dec]
        prob = torch.softmax(scores, dim=-1)
        kids = [v for v in item["adj"][parent[gold]] if v not in order[: index + 1]]
        print(
            f"{name_of(tok, item, node)} -> {name_of(tok, item, gold)} "
            f"parent {name_of(tok, item, parent[gold])} "
            f"L6 parent {float(row[parent_pos]):.3f} self {float(row[dec]):.3f} "
            f"Pgold {float(prob[item['node_tokens'][gold]]):.3f} "
            f"children {child_logits(tok, scores, item, kids)}"
        )

    dec = plen + 2
    cur, gold = order[2], order[3]
    parent_node = parent[gold]
    parent_pos = plen + order.index(parent_node)
    plus = []
    for edge, (src, dst) in enumerate(item["edges"]):
        if src == parent_node and 3 * edge + 3 < plen - 4:
            plus.append((dst, 3 * edge + 2, 3 * edge + 3))
    print("\n== corrupt the list stored at V14 ==")
    trials = {
        "all +3 at L4": [(3, parent_pos, [p for _d, _s, p in plus])],
        "only V95 +3 at L4": [(3, parent_pos, [p for d, _s, p in plus if d == gold])],
        "only V204 +3 at L4": [(3, parent_pos, [p for d, _s, p in plus if d == cur])],
        "only V0 +3 at L4": [(3, parent_pos, [p for d, _s, p in plus if d == order[4]])],
    }
    # The copy into +3 is layer 1 at the +3 position, reading the destination.
    for dst, dest, slot in plus:
        trials[f"L1 copy of {name_of(tok, item, dst)}"] = [(0, slot, [dest])]
    print(
        "baseline",
        child_logits(tok, logits[dec], item, [d for d, _s, _p in plus]),
        "argmax", tok.name(int(logits[dec].argmax())).strip("<>"),
    )
    for label, hides in trials.items():
        _layers, out = run_forward(model, item["tokens"], device, hides=hides)
        scores = out[dec]
        print(label, child_logits(tok, scores, item, [d for d, _s, _p in plus]), "argmax", tok.name(int(scores.argmax())).strip("<>"))

    print("\n== child logits stored at the V14 token ==")
    for layer, (_pre, _mid, post, _attn) in enumerate(layers):
        scores = lens(model, post[parent_pos], device)
        print(f"L{layer + 1}", child_logits(tok, scores, item, [d for d, _s, _p in plus]))

    print("\n== L6 leave-one-key at the decision ==")
    pre = layers[5][0]
    attn = layers[5][3][dec]
    block = model.blocks[5]
    x = pre.to(device)[None]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden = block.norm1(x)
        qkv = block.attn.qkv(hidden).view(1, x.shape[1], 3, block.attn.n_heads, block.attn.head_dim)
        _q, _k, value = qkv.unbind(dim=2)
        value = value.permute(0, 2, 1, 3).reshape(x.shape[1], -1)
        mapped = block.attn.proj(value).float()
    weight = attn.to(device).float()
    piece = weight[:, None] * mapped
    mixed = piece.sum(dim=0)
    actual = (layers[5][1][dec] - pre[dec]).to(device)
    print("delta match", float((mixed - actual).norm()), "actual", float(actual.norm()))
    wanted = [item["node_tokens"][node] for node in (cur, gold, order[4])]

    def show(resid, label):
        scores = lens(model, resid, device)
        bits = " ".join(f"{tok.name(i).strip('<>')} {float(scores[i]):.2f}" for i in wanted)
        print(f"{label:22} {bits} argmax {tok.name(int(scores.argmax())).strip('<>')}")

    show(pre[dec], "before L6")
    show(pre[dec].to(device) + mixed, "full L6 attn")
    for pos, label in ((0, "pos0"), (parent_pos, "V14"), (dec, "self"), (plen, "V209"), (plen - 1, "bar")):
        show(pre[dec].to(device) + piece[pos], f"only {label}")
        show(pre[dec].to(device) + mixed - piece[pos], f"without {label}")


def aggregate(model, tok, device):
    arrays = pathfig.load_test(DATA)
    taken = {0: 0, 1: 0}
    chosen = []
    for index, label in enumerate(arrays["label"].tolist()):
        label = int(label)
        if taken[label] >= 64:
            continue
        taken[label] += 1
        chosen.append(ex.prepare(arrays, tok, index))
    buckets = {
        key: {"n": 0, "exact": [], "p": [], "parent": [], "self": [], "dest": [], "plus": [], "l4": []}
        for key in ("one", "many")
    }
    harmed = {key: {"parent": [], "slots": [], "direct": []} for key in ("one", "many")}
    for item in chosen:
        order, parent, plen = item["order"], item["parent"], item["prompt_len"]
        layers, logits = run_forward(model, item["tokens"], device)
        sinks = []
        for index, node in enumerate(order[:-1]):
            gold = order[index + 1]
            if parent[gold] == node or item["adj"][node]:
                continue
            dec = plen + index
            parent_pos = plen + order.index(parent[gold])
            slots = [3 * edge + 3 for edge, (src, _dst) in enumerate(item["edges"]) if src == parent[gold] and 3 * edge + 3 < plen - 4]
            dests = [3 * edge + 2 for edge, (src, dst) in enumerate(item["edges"]) if src == parent[gold] and dst == gold]
            row = layers[5][3][dec]
            parent_row = layers[3][3][parent_pos]
            kind = "one" if sum(v not in order[: index + 1] for v in item["adj"][parent[gold]]) == 1 else "many"
            gold_id = item["node_tokens"][gold]
            prob = float(torch.softmax(logits[dec], dim=-1)[gold_id])
            slot = buckets[kind]
            slot["n"] += 1
            slot["exact"].append(float(int(logits[dec].argmax()) == gold_id))
            slot["p"].append(prob)
            slot["parent"].append(float(row[parent_pos]))
            slot["self"].append(float(row[dec]))
            slot["dest"].append(float(row[dests].sum()) if dests else 0.0)
            slot["plus"].append(float(row[slots].sum()) if slots else 0.0)
            slot["l4"].append(float(parent_row[slots].sum()) if slots else 0.0)
            sinks.append((kind, dec, parent_pos, slots, dests, gold_id))
        del layers, logits
        groups = {
            "parent": [(5, dec, [parent_pos]) for _k, dec, parent_pos, _s, _d, _g in sinks],
            "slots": [(3, parent_pos, slots) for _k, _d, parent_pos, slots, _ds, _g in sinks],
            "direct": [(layer, dec, dests + slots) for layer in range(6) for _k, dec, _p, slots, dests, _g in sinks],
        }
        outs = {}
        for name, hides in groups.items():
            if not hides:
                continue
            _layers, out = run_forward(model, item["tokens"], device, hides=hides)
            outs[name] = out
        for kind, dec, _parent_pos, _slots, _dests, gold_id in sinks:
            for name in ("parent", "slots", "direct"):
                scores = outs[name][dec]
                harmed[kind][name].append(float(int(scores.argmax()) == gold_id))
    print("\n== 4×4 no-edge steps, 64 YES + 64 NO ==")
    for kind in ("one", "many"):
        slot = buckets[kind]
        if not slot["n"]:
            continue
        def avg(xs):
            return sum(xs) / len(xs)
        print(
            f"{kind} n={slot['n']} exact {avg(slot['exact']):.3f} P {avg(slot['p']):.3f} "
            f"L6 parent {avg(slot['parent']):.3f} self {avg(slot['self']):.3f} "
            f"dest {avg(slot['dest']):.3f} +3 {avg(slot['plus']):.3f} "
            f"parent L4 on its +3 {avg(slot['l4']):.3f}"
        )
        print(
            "  hide parent at L6", f"{avg(harmed[kind]['parent']):.3f}",
            "hide parent +3 at L4", f"{avg(harmed[kind]['slots']):.3f}",
            "hide edge from decision", f"{avg(harmed[kind]['direct']):.3f}",
        )


if __name__ == "__main__":
    device = torch.device("cuda:0")
    _cfg, tok, model, _step = base.load_model(RUN, device)
    model.eval()
    item = ex.prepare(pathfig.load_test(DATA), tok, 2)
    main_item = item
    # main() reloads; call the detailed prints through the already loaded model by
    # reusing main's body would double-load. Run both phases here.
    aggregate(model, tok, device)
