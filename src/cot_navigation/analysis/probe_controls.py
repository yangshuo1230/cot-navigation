"""Query-only controls and matched pre-DOT readout checks.

Extra classifiers see frozen features only. They cannot update the navigation
model. Raw-vocabulary and YES/NO-restricted native readouts are both reported,
so format failure is not confused with lack of binary answer information.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("studies/mechanistic/results"))
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if args.gpu != "0":
        parser.error("This analysis is authorized on physical GPU 0")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["WANDB_MODE"] = "disabled"
    import numpy as np
    import torch
    from .common import AnalysisModel, SEED, anchors, write_json
    from .probes import fit, evaluate, extract, subset
    from .patching import free_rollouts
    torch.set_num_threads(4)
    root = Path.cwd()
    settings = json.loads((args.input/"settings.json").read_text())
    selection = json.loads((args.input/"selection.json").read_text())
    result_path = args.input/"probe_controls.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"models": {}, "seed": SEED}

    def query_features(am, examples):
        x = []
        for ex in examples:
            s, t = am.model.embed.weight[ex.nodes[ex.source]], am.model.embed.weight[ex.nodes[ex.target]]
            x.append(torch.cat([s, t]).detach().float().cpu().numpy()[None])
        return {"x": np.asarray(x), "y": np.asarray([[e.label] for e in examples], np.float32),
                "mask": np.ones((len(examples), 1), np.float32),
                "query": np.asarray([e.index for e in examples]),
                "stage": np.asarray(["query_only"]*len(examples)), "goal_seen": np.zeros(len(examples), bool)}

    def prediction_metrics(pred, labels):
        pred, labels = np.asarray(pred), np.asarray(labels)
        recall = [float((pred[labels == y] == y).mean()) if (labels == y).any() else None for y in (0, 1)]
        return {"n": len(labels), "n_positive": int(labels.sum()), "accuracy": float((pred == labels).mean()),
                "balanced_accuracy": float(np.mean(recall)) if None not in recall else None,
                "valid_answer_rate": float((pred >= 0).mean())}

    def head_predictions(fitted, data):
        x = torch.tensor(data["x"], dtype=torch.float32, device=fitted["weight"].device)
        with torch.no_grad():
            logits = torch.einsum("nld,ldk->nlk", (x-fitted["mean"])/fitted["scale"],
                                  fitted["weight"])+fitted["bias"]
        return (logits[..., 0] >= 0).cpu().numpy().astype(int)

    def native(am, examples, dots):
        raw, constrained, mass = [], [], []
        for begin in range(0, len(examples), 16):
            chunk = examples[begin:begin+16]
            inputs = [e.prompt+[am.tok.id("<DOT>")]*dots for e in chunk]
            logits, _, _ = am.forward(inputs)
            scores = torch.stack([logits[i, len(ids)-1].float() for i, ids in enumerate(inputs)])
            picked = scores.argmax(-1).cpu().tolist()
            raw.extend([1 if p == am.tok.yes else 0 if p == am.tok.no else -1 for p in picked])
            constrained.extend(scores[:, [am.tok.no, am.tok.yes]].argmax(-1).cpu().tolist())
            mass.extend(scores.softmax(-1)[:, [am.tok.no, am.tok.yes]].sum(-1).cpu().tolist())
        return raw, constrained, mass

    for entry in selection["models"]:
        if entry["auxiliary"]:
            continue
        if entry["id"] in result["models"]:
            original = json.loads((args.input/"models"/f"{entry['id']}.json").read_text())
            if result["models"][entry["id"]]["checkpoint_sha256"] != original["checkpoint"]["sha256"]:
                raise ValueError("Main checkpoint changed; choose a new analysis output")
            continue
        print("CONTROL", entry["id"], flush=True)
        am = AnalysisModel(entry, root)
        train = am.examples("train", settings["train_queries"])
        val = am.examples("val", settings["val_queries"])
        test = am.examples("test", settings["test_queries"])
        tr, va, te = [query_features(am, ex) for ex in [train, val, test]]
        fitted = fit(tr, va, device=am.device, seed=SEED, steps=settings["probe_steps"])
        shuffled = fit(tr, va, device=am.device, seed=SEED+17, steps=settings["probe_steps"], shuffled=True)
        record = {"query_only_train": evaluate(fitted, tr)[0], "query_only_test": evaluate(fitted, te)[0],
                  "shuffled_query_test": evaluate(shuffled, te, shuffled=True, seed=SEED+17)[0]}
        original = json.loads((args.input/"models"/f"{entry['id']}.json").read_text())
        assert am.checkpoint["sha256"] == original["checkpoint"]["sha256"]
        record["checkpoint_sha256"] = am.checkpoint["sha256"]
        assert [e.index for e in test] == original["probes"]["query_ids"]["test"]
        if am.context and entry["family"] in {"dfs", "bfs"}:
            free, outcomes = free_rollouts(am, test, max_new=0)
            assert [o["pred"] for o in outcomes] == [o["pred"] for o in original["rollout"]["by_query"]]
            eligible = []
            for ex in test:
                anchor, _ = anchors(free[ex.index], ex, am.protocol, am.tok)
                middle = dict(anchor).get("middle")
                eligible.append(middle is not None and ex.nodes[ex.target] not in free[ex.index][ex.row.prompt_len:middle+1])
            if any(eligible):
                record["query_only_middle_goal_unseen"] = evaluate(fitted, subset(te, np.asarray(eligible)))[0]
        if entry["family"] == "filler":
            selected_ids = set(original["patching"]["query_ids"])
            chosen = [e for e in test if e.index in selected_ids]
            prefix_data = [extract(am, ex, {e.index: e.prompt for e in ex})["answer"] for ex in [train, val, test]]
            linear = fit(prefix_data[0], prefix_data[1], device=am.device, seed=SEED, steps=settings["probe_steps"])
            mask = np.asarray([q in selected_ids for q in prefix_data[2]["query"]])
            chosen_data = subset(prefix_data[2], mask)
            linear_pred = head_predictions(linear, chosen_data)
            labels = [e.label for e in chosen]
            assert chosen_data["query"].tolist() == [e.index for e in chosen]
            raw, restricted, mass = native(am, chosen, 0)
            after, _, _ = native(am, chosen, am.filler)
            query_pred = head_predictions(fitted, subset(te, np.asarray([e.index in selected_ids for e in test])))[:, 0]
            record["matched_filler_readout"] = {
                "query_ids": [e.index for e in chosen], "labels": labels,
                "native_before_dot": prediction_metrics(raw, labels),
                "native_binary_before_dot": prediction_metrics(restricted, labels),
                "native_answer_mass_before_dot": float(np.mean(mass)),
                "native_after_dot": prediction_metrics(after, labels),
                "linear_before_dot": [prediction_metrics(linear_pred[:, i], labels) for i in range(linear_pred.shape[1])],
                "query_only": prediction_metrics(query_pred, labels),
                "predictions": {"raw_before": raw, "restricted_before": restricted, "after": after,
                                "linear": linear_pred.tolist(), "query_only": query_pred.tolist()},
            }
        result["models"][entry["id"]] = record
        write_json(result_path, result)
        del am
        torch.cuda.empty_cache()
    print("COMPLETE supplementary controls", flush=True)


if __name__ == "__main__":
    main()
