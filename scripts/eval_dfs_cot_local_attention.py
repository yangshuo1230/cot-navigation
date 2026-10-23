#!/usr/bin/env python3
"""Restrict in-context DFS attention to the current record and the previous one.

The prompt stays visible. Older completed records are masked, including at the
semicolon that predicts YES/NO, where that semicolon still counts as part of
the record it closes. Greedy answer accuracy is compared with full attention.
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
sys.path.insert(0, str(ROOT / "scripts"))

import cot_navigation.models.transformer as transformer_mod  # noqa: E402
import plot_dfs_context_scale_attention as scale  # noqa: E402
import plot_dfs_cot_attention as base  # noqa: E402
from cot_navigation.analysis.common import records  # noqa: E402
from cot_navigation.evaluation.rollout import generate  # noqa: E402
from cot_navigation.models.record_attention import record_mask, record_metadata  # noqa: E402

OUT = ROOT / "figures/scale_v1/dfs_context_local_attention_accuracy.png"
JSON = ROOT / "research/block_membership_probe_20261021/dfs_context_local_attention.json"
DEVICE = "cuda:0"
MAX_LEN = 4096
FREE_PER_CLASS = 64


def two_record_mask(history, positions, keep_records, *, use_flex=False):
    """Prompt, the record containing the query, and one record before it."""
    del keep_records, use_flex
    has_query, prompt_end, key_record, _query_record = record_metadata(history, positions)
    query_id = key_record.gather(1, positions)
    length = history.shape[1]
    keys = torch.arange(length, device=history.device)
    causal = keys[None, None, :] <= positions[:, :, None]
    body = keys[None, None, :] >= prompt_end[:, None, None]
    in_window = key_record[:, None, :] >= query_id[:, :, None] - 1
    visible = (
        (keys[None, None, :] < prompt_end[:, None, None])
        | (body & in_window)
        | ~has_query[:, None, None]
    )
    return (causal & visible)[:, None, :, :]


def check_mask() -> None:
    prompt = [2, 14, 15, 3, 14, 19, 4]
    records_tok = [
        [14, 4, 14, 4, 15, 4, 8, 15, 5],
        [14, 15, 4, 14, 15, 4, 16, 4, 8, 16, 5],
        [14, 15, 16, 4, 14, 15, 16, 4, 4, 9, 5],
        [14, 15, 4, 14, 15, 16, 4, 4, 9, 5],
    ]
    sequence = prompt + sum(records_tok, []) + [11, 5, 7, 1]
    x = torch.tensor([sequence])
    pos = torch.arange(x.shape[1])[None]
    mine = two_record_mask(x, pos, 1)[0, 0]
    window1 = record_mask(x, pos, 1)[0, 0]
    window2 = record_mask(x, pos, 2)[0, 0]
    starts = [len(prompt)]
    for record in records_tok:
        starts.append(starts[-1] + len(record))
    inside = starts[3] + 3
    semicolon = starts[3] - 1
    if not torch.equal(mine[inside], window1[inside]):
        raise RuntimeError("inside a record, current+previous does not match window 1")
    if not torch.equal(mine[semicolon], window2[semicolon]):
        raise RuntimeError("at a semicolon, current+previous does not match window 2")
    if torch.equal(mine[semicolon], window1[semicolon]):
        raise RuntimeError("semicolon window collapsed to the current record only")


def enable(model, local: bool) -> None:
    transformer_mod.record_mask = two_record_mask
    model.swa_records = 1 if local else 0
    model.swa_backend = "sdpa"


def rates(correct, labels) -> dict:
    correct = np.asarray(correct, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    yes = labels == 1
    no = ~yes
    return {
        "n": int(len(labels)),
        "n_yes": int(yes.sum()),
        "n_no": int(no.sum()),
        "accuracy": float(correct.mean()) if len(labels) else None,
        "balanced_accuracy": float(0.5 * (correct[yes].mean() + correct[no].mean())) if yes.any() and no.any() else None,
        "yes_recall": float(correct[yes].mean()) if yes.any() else None,
        "no_recall": float(correct[no].mean()) if no.any() else None,
    }


@torch.inference_mode()
def teacher_forced(model, items, tok, device) -> dict:
    yes_id, no_id = tok.yes, tok.no
    actions = {tok.id(name) for name in ("<PUSH>", "<POP>", "<FOUND>", "<EXHAUST>")}
    answer_correct, action_correct, answer_changed, action_changed, answer_margins = [], [], [], [], []
    labels, distances, long_flags = [], [], []
    skipped = 0
    for item in items:
        tokens = item["tokens"]
        if len(tokens) > MAX_LEN or tokens[-2] not in (yes_id, no_id):
            skipped += 1
            continue
        parsed = records(tokens, item["prompt_len"], tok)
        if not parsed:
            skipped += 1
            continue
        ids = torch.tensor(tokens, device=device, dtype=torch.long)[None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            enable(model, False)
            full = model(ids)
            enable(model, True)
            local = model(ids)
        answer_at = parsed[-1]["end"]
        if tokens[answer_at + 1] not in (yes_id, no_id):
            skipped += 1
            continue
        full_answer = int(full[0, answer_at].argmax())
        local_answer = int(local[0, answer_at].argmax())
        gold = tokens[answer_at + 1]
        answer_correct.append((full_answer == gold, local_answer == gold))
        full_margin = torch.topk(full[0, answer_at].float(), 2).values
        answer_changed.append(full_answer != local_answer)
        answer_margins.append(float(full_margin[0] - full_margin[1]))
        labels.append(item["label"])
        distances.append(item["distance"])
        long_flags.append(len(parsed) >= 8)
        act_full, act_local, act_diff = [], [], []
        for record in parsed:
            pos = record["separators"][2]
            gold_action = tokens[pos + 1]
            if gold_action not in actions:
                continue
            pred_full = int(full[0, pos].argmax())
            pred_local = int(local[0, pos].argmax())
            act_full.append(pred_full == gold_action)
            act_local.append(pred_local == gold_action)
            act_diff.append(pred_full != pred_local)
        action_correct.append((float(np.mean(act_full)), float(np.mean(act_local))))
        action_changed.append(float(np.mean(act_diff)))
        del full, local
    full_ok = [row[0] for row in answer_correct]
    local_ok = [row[1] for row in answer_correct]
    long = np.asarray(long_flags, dtype=bool)
    return {
        "skipped_over_length": skipped,
        "answer_full": rates(full_ok, labels),
        "answer_local": rates(local_ok, labels),
        "answer_changed": float(np.mean(answer_changed)),
        "action_full": float(np.mean([row[0] for row in action_correct])),
        "action_local": float(np.mean([row[1] for row in action_correct])),
        "action_changed": float(np.mean(action_changed)),
        "long_answer_full": rates([ok for ok, flag in zip(full_ok, long) if flag],
                                  [label for label, flag in zip(labels, long) if flag]),
        "long_answer_local": rates([ok for ok, flag in zip(local_ok, long) if flag],
                                   [label for label, flag in zip(labels, long) if flag]),
        "long_answer_changed": float(np.mean([changed for changed, flag in zip(answer_changed, long) if flag])),
        "changed_with_margin": float(np.mean([
            changed and margin > 0.5 for changed, margin in zip(answer_changed, answer_margins)
        ])),
    }


def free_subset(items):
    chosen = []
    for label in (1, 0):
        group = [item for item in items if item["label"] == label]
        group.sort(key=lambda item: item["index"])
        chosen.extend(group[:FREE_PER_CLASS])
    return chosen


@torch.inference_mode()
def free_accuracy(model, items, tok, device, local: bool) -> dict:
    enable(model, local)
    correct, labels, fails, interrupted = [], [], [], []
    batch = 8
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        prompts = [item["tokens"][:item["prompt_len"]] for item in chunk]
        generated = generate(model, prompts, "dfs_state", tok, "bf16", cot_margin=8)
        for item, gen in zip(chunk, generated):
            ok = gen["pred"] == item["label"] and not gen["format_fail"]
            correct.append(ok)
            labels.append(item["label"])
            fails.append(gen["format_fail"])
            interrupted.append(gen["cot_interrupted"])
        print(f"    free {'local' if local else 'full'} {min(start + batch, len(items))}/{len(items)}", flush=True)
    out = rates(correct, labels)
    out["format_fail_rate"] = float(np.mean(fails))
    out["interrupted_rate"] = float(np.mean(interrupted))
    return out


def draw(rows: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    x = np.arange(len(rows))
    width = 0.36
    for ax, key, title in (
        (axes[0], "teacher_forced", "Teacher-forced answer"),
        (axes[1], "free", "Free generation"),
    ):
        full = [row[key]["full"]["balanced_accuracy"] for row in rows]
        local = [row[key]["local"]["balanced_accuracy"] for row in rows]
        ax.bar(x - width / 2, full, width, color="#0072B2", label="Full attention")
        ax.bar(x + width / 2, local, width, color="#D55E00", label="Current + previous record")
        ax.set_xticks(x)
        ax.set_xticklabels([row["size"] for row in rows])
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#666666", linewidth=0.8, linestyle="--")
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11)
        for xpos, value in list(zip(x - width / 2, full)) + list(zip(x + width / 2, local)):
            ax.text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("In-context DFS with older records masked", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.14, wspace=0.16)
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
    check_mask()
    device = torch.device(DEVICE)
    _cfg, tok, model, step = base.load_model(base.CONTEXT_RUN, device)
    saved = []
    for name, root in scale.SIZES:
        arrays = scale.load_test(root)
        items = [scale.render_index(arrays, tok, index) for index in range(len(arrays["label"]))]
        print(f"\n{name} n={len(items)}", flush=True)
        forced = teacher_forced(model, items, tok, device)
        print("  teacher-forced", json.dumps({
            "full": forced["answer_full"],
            "local": forced["answer_local"],
            "answer_changed": forced["answer_changed"],
            "action_full": forced["action_full"],
            "action_local": forced["action_local"],
            "action_changed": forced["action_changed"],
            "long_full": forced["long_answer_full"]["balanced_accuracy"],
            "long_local": forced["long_answer_local"]["balanced_accuracy"],
            "long_changed": forced["long_answer_changed"],
            "changed_with_margin": forced["changed_with_margin"],
            "skipped": forced["skipped_over_length"],
        }), flush=True)
        subset = free_subset(items)
        free_full = free_accuracy(model, subset, tok, device, False)
        free_local = free_accuracy(model, subset, tok, device, True)
        print("  free", json.dumps({"full": free_full, "local": free_local}), flush=True)
        saved.append({
            "size": name,
            "step": step,
            "teacher_forced": forced,
            "free": {"n_per_class": FREE_PER_CLASS, "full": free_full, "local": free_local},
        })
    draw([{
        "size": row["size"],
        "teacher_forced": {
            "full": row["teacher_forced"]["answer_full"],
            "local": row["teacher_forced"]["answer_local"],
        },
        "free": row["free"],
    } for row in saved])
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(saved, indent=2))
    print(JSON)


if __name__ == "__main__":
    main()
