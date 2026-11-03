"""Reusable frozen-checkpoint attention, linear probe and token intervention suite.

Run from the project root. Only probes are optimized; model parameters remain
frozen and are fingerprinted. CUDA_VISIBLE_DEVICES is set BEFORE importing
torch, so all optimization and inference use the requested physical GPU.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, default=Path("studies/mechanistic/results"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--models", nargs="*")
    p.add_argument("--stages", nargs="+", choices=["attention", "probes", "patching"], default=["attention", "probes", "patching"])
    p.add_argument("--train-queries", type=int, default=1024)
    p.add_argument("--val-queries", type=int, default=256)
    p.add_argument("--test-queries", type=int, default=256)
    p.add_argument("--attention-queries", type=int, default=64)
    p.add_argument("--attention-fixed", action="store_true", help="Also capture attention in fixed-graph task models")
    p.add_argument("--patch-cases", type=int, default=40)
    p.add_argument("--probe-steps", type=int, default=250)
    p.add_argument("--max-new-tokens", type=int, default=0, help="0 uses the checkpoint's full context window")
    p.add_argument("--inventory", type=Path, help="Optional pre-read inventory; still uses validation scores only")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    stage_versions = {"attention": 2, "probes": 3, "patching": 2}
    if args.gpu != "0":
        p.error("This experiment is authorized on physical GPU 0 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import numpy as np
    import torch
    from cot_navigation.analysis.common import AnalysisModel, SEED, write_json, sha256
    from cot_navigation.analysis.selection import inventory, select
    from cot_navigation.analysis.attention import collect_attention
    from cot_navigation.analysis.probes import run_probes
    from cot_navigation.analysis.patching import free_rollouts, run_patching
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.root = args.root.resolve()
    out = args.output if args.output.is_absolute() else args.root/args.output
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out/"settings.json"
    previous_limit = args.max_new_tokens
    if args.resume and settings_path.exists():
        previous = json.loads(settings_path.read_text())
        previous_limit = previous.get("max_new_tokens", args.max_new_tokens)
        for key in ["train_queries", "val_queries", "test_queries", "attention_queries",
                    "patch_cases", "probe_steps", "max_new_tokens"]:
            if previous.get(key) != getattr(args, key):
                if key == "max_new_tokens" and previous_limit > 0 and (args.max_new_tokens == 0 or args.max_new_tokens > previous_limit):
                    continue  # safely extend only analyses that hit the old cap
                raise ValueError(f"Analysis setting {key} changed; choose a new --output")
    selection_path = out/"selection.json"
    if args.resume and selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        rows = json.loads(args.inventory.read_text()) if args.inventory else inventory(args.root)
        selection = select(rows)
        write_json(selection_path, selection)
    if args.resume and previous_limit != args.max_new_tokens:
        # Persist every old per-model limit BEFORE updating global settings;
        # an interrupted extension must remain safely resumable.
        for entry in selection["models"]:
            saved = out/"models"/f"{entry['id']}.json"
            if saved.exists():
                record = json.loads(saved.read_text())
                record.setdefault("analysis_max_new_tokens", previous_limit)
                write_json(saved, record)
    write_json(settings_path, {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                    "seed": SEED, "device": torch.cuda.get_device_name(0),
                                    "method_version": 1})
    started = time.time()
    for entry in selection["models"]:
        name = entry["id"]
        if args.models and name not in args.models:
            continue
        result_path = out/"models"/f"{name}.json"
        result = json.loads(result_path.read_text()) if args.resume and result_path.exists() else {"entry": entry}
        stored_sha = result.get("checkpoint", {}).get("sha256")
        if stored_sha and sha256(args.root/entry["run"]/"checkpoints/best.pt") != stored_sha:
            raise ValueError(f"Checkpoint changed for {name}; choose a new --output")
        stages = [s for s in args.stages if not (s == "attention" and entry["kind"] != "context_graph"
                  and (not args.attention_fixed or entry.get("auxiliary", False)))]
        versions = result.setdefault("stage_versions", {})
        # Migrate first-pass summaries without rerunning unchanged stages.
        for key in ["attention", "patching", "probes"]:
            if key in result and key not in versions:
                versions[key] = (2 if key == "attention" and "generated" in result[key] else 1)
        if versions.get("probes") == 2 and entry["protocol"] not in {"dfs_state", "bfs_state"}:
            versions["probes"] = 3  # structural read-position change only affects state protocols
        old_limit = result.get("analysis_max_new_tokens", previous_limit)
        extending = old_limit > 0 and (args.max_new_tokens == 0 or args.max_new_tokens > old_limit)
        hit_limit = result.get("rollout", {}).get("limits", 0) > 0 or any(
            c.get("limit_rate", 0) > 0 for c in result.get("patching", {}).get("conditions", {}).values())
        if extending and hit_limit:
            result["short_limit_audit"] = {"max_new_tokens": old_limit,
                "rollout_accuracy": result.get("rollout", {}).get("accuracy"),
                "rollout_limits": result.get("rollout", {}).get("limits"),
                "patching_limits": {k: c.get("limit_rate", 0) for k, c in result.get("patching", {}).get("conditions", {}).items()}}
            for stage in stages:
                versions[stage] = -1
        result["analysis_max_new_tokens"] = args.max_new_tokens
        if args.resume and all(s in result and versions.get(s) == stage_versions[s] for s in stages):
            write_json(result_path, result)
            print(f"SKIP {name}: stages already complete", flush=True)
            continue
        print(f"START {name} {entry['protocol']} {entry['run']}", flush=True)
        am = AnalysisModel(entry, args.root)
        if result.get("checkpoint", {}).get("sha256", am.checkpoint["sha256"]) != am.checkpoint["sha256"]:
            raise ValueError(f"Checkpoint changed for {name}; choose a new --output")
        result["checkpoint"] = am.checkpoint
        train = am.examples("train", args.train_queries)
        val = am.examples("val", args.val_queries)
        test = am.examples("test", args.test_queries)
        assert not ({x.index for x in train} & {x.index for x in val})
        assert not ({x.index for x in train} & {x.index for x in test})
        free_sequences = None
        if any(s not in result or versions.get(s) != stage_versions[s] for s in stages):
            print(f"  free rollout {len(test)} test queries", flush=True)
            free_sequences, rollouts = free_rollouts(am, test, max_new=args.max_new_tokens)
            result["rollout"] = {"n": len(test), "accuracy": float(np.mean([r["correct"] for r in rollouts])),
                                 "limits": sum(r["limit"] for r in rollouts),
                                 "mean_generated_tokens": float(np.mean([r["tokens"] for r in rollouts])),
                                 "by_query": rollouts}
        for stage in stages:
            if args.resume and stage in result and versions.get(stage) == stage_versions[stage]:
                continue
            print(f"  {stage}", flush=True)
            if stage == "attention":
                result[stage] = {
                    "reference": collect_attention(am, train[:args.attention_queries], test[:args.attention_queries]),
                    "generated": collect_attention(am, train[:args.attention_queries], test[:args.attention_queries],
                                                   generated_prefixes=free_sequences),
                }
            elif stage == "probes":
                result[stage] = run_probes(am, train, val, test, free_sequences, steps=args.probe_steps)
            else:
                result[stage] = run_patching(am, test, free_sequences,
                                             max_cases=args.patch_cases, max_new=args.max_new_tokens,
                                             original_predictions={row['query']: row['pred'] for row in result['rollout']['by_query']})
            versions[stage] = stage_versions[stage]
            write_json(result_path, result)
            print(f"  saved {stage}; elapsed={time.time()-started:.0f}s", flush=True)
        del am, train, val, test, free_sequences
        torch.cuda.empty_cache()
    print(f"COMPLETE elapsed={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
