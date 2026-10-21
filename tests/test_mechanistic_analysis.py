"""Pure checks for distance correction, causal indices, selection and edits."""
from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cot_navigation.analysis.attention import DistanceBaseline
from cot_navigation.analysis.common import Example, anchors, records
from cot_navigation.analysis.patching import state_case, reachable, dot_attention_ablation
from cot_navigation.analysis.probes import metrics, shuffled_labels
from cot_navigation.analysis.selection import select
from cot_navigation.protocols.render import render_task, parametric_prompt
from cot_navigation.protocols.tokenizer import Tokenizer


def example(protocol="dfs_state"):
    tok = Tokenizer(16)
    # Two-step route to t=3 and a dead branch 2->4.
    adj = [[1, 2], [3], [4], [], []]
    nodes = [tok.node_id(i) for i in range(5)]
    prompt = parametric_prompt(tok, nodes, 0, 3)
    row = render_task(tok, protocol, 1, prompt, adj, 0, 3, nodes)
    return tok, Example(7, "test", 1, adj, nodes, 0, 3, row)


def test_distance_means_exclude_noncausal_and_pair_weight_lengths():
    b = DistanceBaseline(1, 1, 4)
    weights = np.array([[[[1., 999., 999.], [.2, .8, 999.], [.1, .3, .6]]]])
    b.add(weights)
    b.add(np.array([[[[1.]]]]))
    assert b.counts.tolist() == [4, 2, 1, 0]
    assert np.allclose(b.mean[0, 0], [.85, .25, .1, 0])
    for d in range(3):
        residual = np.diagonal(weights, -d, axis1=-2, axis2=-1).sum()
        if d == 0:
            residual += 1.
        assert abs(residual-b.mean[0, 0, d]*b.counts[d]) < 1e-10


def test_records_positions_and_probe_anchor_exclude_answer_and_terminal():
    tok, ex = example()
    prefix = ex.row.tokens[:ex.row.answer_index]
    rs = records(prefix, ex.row.prompt_len, tok)
    assert rs and rs[-1]["action"] == [tok.id("<FOUND>")]
    answer, structures = anchors(prefix, ex, "dfs_state", tok)
    terminal = prefix.index(tok.id("<FOUND>"))
    assert all(pos < terminal for _, pos in answer)
    assert all(s["pos"] < terminal for s in structures)
    # Each target describes history at a read position before VIS is emitted.
    assert structures[0]["visited"] == [ex.nodes[ex.source]]
    assert prefix[structures[0]["pos"]] == tok.id("|")
    assert structures[0]["successor_pos"] == rs[0]["separators"][1]
    assert structures[0]["successor_pos"] > structures[0]["pos"]


def test_visited_intervention_changes_one_token_and_discards_stale_candidates():
    tok, ex = example()
    am = SimpleNamespace(protocol="dfs_state", tok=tok)
    case = state_case(am, ex, ex.row.tokens[:ex.row.answer_index])
    assert case is not None
    clean, bad, control = [case["prefixes"][k] for k in ["clean", "visited_route", "visited_irrelevant"]]
    assert len(bad) == len(control) == len(clean)+1
    assert clean[-1] == bad[-1] == control[-1] == tok.id("|")
    assert len(records(clean, ex.row.prompt_len, tok)) == case["detail"]["record"]
    inv = {v: i for i, v in enumerate(ex.nodes)}
    assert reachable(ex.adj, inv[case["detail"]["critical"]], ex.target)
    assert not reachable(ex.adj, inv[case["detail"]["control"]], ex.target)


def test_masks_and_f1_do_not_reward_absent_nodes():
    y = np.array([[1., 0., 0., 0.]])
    mask = np.array([[1., 1., 0., 0.]])
    out = metrics(np.array([[1., 1., -1., -1.]]), y, mask)
    assert out["accuracy"] == .5 and out["f1"] == 2/3
    assert out["exact_set_accuracy"] == 0


def test_shuffled_multilabel_targets_keep_graph_mask_and_cardinality():
    data = {"y": np.array([[0., 1., 0., 1., 0.]]), "mask": np.array([[0., 1., 1., 1., 0.]])}
    out = shuffled_labels(data, 1)
    assert out.sum() == 2 and out[0, [0, 4]].sum() == 0


def test_fixed_variant_selection_uses_mean_across_graphs_and_no_test():
    rows = []
    for family in ["direct", "path", "filler", "dfs_state", "bfs_state"]:
        rows.append(dict(run=f"context/base/{family}", protocol=family, kind="context_graph", seed=None, score=.7))
        for seed in range(3):
            rows.append(dict(run=f"fixed/seed{seed}/base/{family}", protocol=family,
                             kind="parametric_graph", seed=seed, score=.8))
    # Huge seed0 result must not hide poor seed1/2 scores.
    for seed, score in enumerate([1., .4, .4]):
        rows.append(dict(run=f"fixed/seed{seed}/alternative/direct", protocol="direct",
                         kind="parametric_graph", seed=seed, score=score))
    result = select(rows)
    fixed_direct = [r for r in result["models"] if r["kind"] == "parametric_graph" and r["family"] == "direct"]
    assert len(result["models"]) == 20 and len(fixed_direct) == 3
    assert all("base" in r["run"] for r in fixed_direct)
    assert result["no_test_used_for_selection"]


def test_filler_selection_includes_smaller_training_pool_and_retains_control():
    rows = []
    for proto in ["direct", "path", "filler", "dfs_state", "bfs_state"]:
        rows.append(dict(run=f"context/train50000/{proto}", protocol=proto, kind="context_graph", seed=None, score=.7))
        for seed in range(3):
            rows.append(dict(run=f"fixed/seed{seed}/base/{proto}", protocol=proto,
                             kind="parametric_graph", seed=seed, score=.8))
    rows.append(dict(run="context/train10000/filler", protocol="filler", kind="context_graph", seed=None, score=.75))
    result = select(rows)
    chosen = next(e for e in result["models"] if e["id"] == "context_filler")
    control = next(e for e in result["models"] if e["id"] == "context_filler_50k_control")
    assert chosen["run"] == "context/train10000/filler" and not chosen["auxiliary"]
    assert control["run"] == "context/train50000/filler" and control["auxiliary"]
    assert sum(not e["auxiliary"] for e in result["models"]) == 20


def test_dot_attention_ablation_preserves_self_and_restores_original():
    import torch
    q = torch.zeros(1, 1, 4, 1)
    v = torch.tensor([1., 2., 10., 20.]).reshape(1, 1, 4, 1)
    original = lambda *args, **kwargs: (None, None)
    attn = SimpleNamespace(_attend=original)
    am = SimpleNamespace(model=SimpleNamespace(blocks=[SimpleNamespace(attn=attn)]))
    with dot_attention_ablation(am, 2, "chain"):
        out, _ = attn._attend(q, q, v, None, False, causal=True)
        assert torch.allclose(out[0, 0, 3], torch.tensor([23/3]))
    assert attn._attend is original
    with dot_attention_ablation(am, 2, "prompt"):
        out, _ = attn._attend(q, q, v, None, False, causal=True)
        assert out[0, 0, 3].item() == 15.
    assert attn._attend is original


@pytest.mark.parametrize("hit_limit", [False, True])
def test_resume_native_context_refreshes_only_truncated_results(tmp_path, monkeypatch, hit_limit):
    """No model/GPU is opened; a raised marker simulates interruption safely."""
    import torch
    import cot_navigation.analysis.common as common
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("mechanism_runner", root/"scripts/analyze_mechanisms.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    out = tmp_path/"results"
    (out/"models").mkdir(parents=True)
    entry = {"id": "context_direct", "kind": "context_graph", "protocol": "direct", "run": "unused"}
    (out/"selection.json").write_text(json.dumps({"models": [entry]}))
    keys = dict(train_queries=1024, val_queries=256, test_queries=256,
                attention_queries=64, patch_cases=40, probe_steps=250, max_new_tokens=1024)
    (out/"settings.json").write_text(json.dumps(keys))
    result = {"entry": entry, "attention": {"generated": {}}, "probes": {}, "patching": {"conditions": {}},
              "rollout": {"limits": int(hit_limit), "accuracy": .5},
              "stage_versions": {"attention": 2, "probes": 3, "patching": 2}}
    saved = out/"models/context_direct.json"
    saved.write_text(json.dumps(result))
    class RefreshRequired(Exception):
        pass
    def never_load(*args, **kwargs):
        raise RefreshRequired("refresh requested before opening model")
    monkeypatch.setattr(common, "AnalysisModel", never_load)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "mock-no-GPU")
    monkeypatch.setattr("sys.argv", ["runner", "--output", str(out), "--max-new-tokens", "0", "--resume"])
    if hit_limit:
        with pytest.raises(RefreshRequired):
            runner.main()
        # An interrupted extension retains the original per-model cap even
        # though global settings have already advanced, so resume refreshes it.
        assert json.loads(saved.read_text())["analysis_max_new_tokens"] == 1024
    else:
        runner.main()
        assert json.loads(saved.read_text())["analysis_max_new_tokens"] == 0
    assert json.loads((out/"settings.json").read_text())["max_new_tokens"] == 0
