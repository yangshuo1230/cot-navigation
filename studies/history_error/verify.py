"""CPU-only reconstruction of all paired data and precision checks."""
from pathlib import Path
import hashlib
import json
import math
import torch
from run import build_events, ANCHORS
from cot_navigation.analysis.scale_followup import make_adapter
from cot_navigation.analysis.common import sha256, write_json

torch.set_num_threads(2)
root = Path.cwd()
study = root / "studies/history_error"
models = sorted((study / "results/models").glob("*.json"))
assert len(models) == 4


def chance(value):
    # Tie-aware AUROC integrates floating-point FPR increments; mathematically
    # identical score distributions can yield 0.4999999999999999.
    return math.isclose(value, .5, rel_tol=0, abs_tol=1e-12)


checked = []
for path in models:
    bf16 = json.loads(path.read_text())
    fp32 = json.loads((study / "results_fp32/models" / path.name).read_text())
    assert bf16["source_sha256"] == sha256(study / "run_bf16_source.py")
    assert fp32["source_sha256"] == sha256(study / "run.py")
    assert bf16["entry"] == fp32["entry"]
    am = make_adapter(bf16["entry"], root, "cpu")
    count = graph_count = 0
    split_queries = []
    for split, target in (("train", 768), ("val", 192), ("test", 256)):
        events, graph_events, inventory = build_events(am, split, target, max(target * 4, 512))
        qs = {e["query"] for e in events}
        split_queries.append(qs)
        for result in (bf16, fp32):
            saved = result["dataset"][split]
            assert saved["pairs"] == inventory["pairs"] == target
            assert saved["query_ids"] == [e["query"] for e in events]
            assert saved["graph_query_ids"] == [e["query"] for e in graph_events]
            assert saved["pre_error_max_difference"] == 0
            assert saved["mean_error_position"] == sum(e["error_position"] for e in events) / len(events)
            assert all(saved["mean_readout_delays"][a] == sum(e["positions"][a] - e["error_position"] for e in events) / len(events) for a in ANCHORS)
        for e in events:
            differences = [i for i, (a, b) in enumerate(zip(e["correct"], e["wrong"])) if a != b]
            assert differences == [e["error_position"]]
            assert e["correct"][e["error_position"] + 1:] == e["wrong"][e["error_position"] + 1:]
            count += 1
        for e in graph_events:
            assert e["correct"][e["prompt_length"]:] == e["wrong"][e["prompt_length"]:]
            assert e["valid_anchors"]["record_1_vis"]
            graph_count += 1
    assert not (split_queries[0] & split_queries[1] or split_queries[0] & split_queries[2] or split_queries[1] & split_queries[2])
    for result in (bf16, fp32):
        assert result["complete"] and result["checkpoint_unchanged"]
        for probe in result["probes"]:
            if not probe.get("graph_trained"):
                assert all(chance(r["auroc"]) and r["paired_rank_accuracy"] == .5 for r in probe["test"] if r["anchor"] == "pre_error")
                assert all(chance(r["auroc"]) for r in probe["test"] if r["layer"] == "current_token" and r["anchor"] != "error")
            for r in probe.get("graph_zero_shot", []):
                if r["layer"] in {"current_token", "edited_and_current_node"}:
                    assert chance(r["auroc"])
            if probe.get("graph_trained"):
                assert all(chance(r["auroc"]) for r in probe["test"] if r["layer"] in {"current_token", "edited_and_current_node"})
    assert sha256(am.run / "checkpoints/best.pt") == bf16["entry"]["checkpoint_sha256"]
    checked.append({"model": path.stem, "reconstructed_primary_pairs": count,
                    "reconstructed_graph_pairs": graph_count, "checkpoint_unchanged": True,
                    "precisions_share_identical_pairs": True, "query_splits_disjoint": True,
                    "pre_error_activations_identical": True, "identity_controls_verified": True})
    print("VERIFIED", path.stem, count, graph_count, flush=True)
    del am
write_json(study / "verification.json", {"status": "PASS", "models": checked,
    "primary_pairs_reconstructed": sum(r["reconstructed_primary_pairs"] for r in checked),
    "graph_pairs_reconstructed": sum(r["reconstructed_graph_pairs"] for r in checked),
    "notes": ["Reference search and independent semantic_slots certify every pair during reconstruction",
              "Correct/wrong pairs share every subsequent input token and current state",
              "Graph controls share full CoT text and have one graph edit in each class",
              "No answer or terminal action tokens in any readout prefix",
              "No model training, activation cache files or new checkpoints"]})
