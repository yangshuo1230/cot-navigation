"""Scale mixture, context truncation, and the scale-v1 launch plan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cot_navigation.config import ConfigError, load_config
from cot_navigation.protocols.render import RenderedExample, context_prompt, render_task, truncate_example
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.augmentation import transform
from cot_navigation.training.sampler import ScaleMixtureSampler

ROOT = Path(__file__).resolve().parents[1]


def _row(tokens, prompt_len, answer_index):
    return RenderedExample(
        tokens=tokens, prompt_len=prompt_len, answer_index=answer_index,
        trace_indices=list(range(prompt_len, answer_index)), walk_indices=[], label=1,
    )


def test_truncate_state_trace_on_record_boundary():
    tok = Tokenizer(4, enable_stop=True)
    prompt = [1, 2, 3]
    body = [tok.node_id(0), tok.id(";"), tok.node_id(1), tok.id(";"), tok.node_id(2), tok.id(";")]
    answer = [tok.yes, tok.id("<EOS>")]
    row = _row(prompt + body + answer, len(prompt), len(prompt) + len(body))
    limit = len(prompt) + 4 + 3
    updated = truncate_example(row, tok, limit + 8, 8)
    assert len(updated.tokens) <= limit
    assert updated.tokens[updated.answer_index - 1] == tok.id("<STOP>")
    assert updated.tokens[updated.answer_index:] == answer
    assert tok.id(";") == updated.tokens[updated.answer_index - 2]
    assert updated.meta["truncated"] is True


def test_truncate_path_on_token_boundary_and_keeps_short_rows():
    tok = Tokenizer(4, enable_stop=True)
    prompt = [9]
    body = [tok.node_id(i % 4) for i in range(6)]
    answer = [tok.no, tok.id("<EOS>")]
    row = _row(prompt + body + answer, 1, 1 + len(body))
    updated = truncate_example(row, tok, 1 + 2 + 3 + 8, 8)
    assert updated.tokens == [9, tok.node_id(0), tok.node_id(1), tok.id("<STOP>"), tok.no, tok.id("<EOS>")]
    assert truncate_example(row, tok, 100, 8) is row


def test_scale_mixture_quotas_and_idle_member():
    sampler = ScaleMixtureSampler([50, 50, 50], 10, 0, [0.3, 0.7, 0.0])
    idle = sampler.samplers[2].cursor
    counts = np.zeros(3, dtype=int)
    for _ in range(1000):
        for member, _local in sampler.next():
            counts[member] += 1
    assert counts.sum() == 10000
    assert abs(counts[0] / counts.sum() - 0.3) < 0.02
    assert abs(counts[1] / counts.sum() - 0.7) < 0.02
    assert counts[2] == 0 and sampler.samplers[2].cursor == idle
    sampler.set_weights([0.0, 0.0, 1.0])
    members = {member for member, _local in sampler.next()}
    assert members == {2}


def test_shuffled_bfs_changes_the_trace():
    tok = Tokenizer(8)
    adj = [[1, 2], [3], [3], []]
    nodes = [tok.node_id(i) for i in range(4)]
    prompt = context_prompt(tok, nodes, [(0, 1), (0, 2), (1, 3), (2, 3)], 0, 3)
    base = render_task(tok, "bfs_state", 1, prompt, adj, 0, 3, nodes)
    changed = False
    for seed in range(40):
        row = transform(base, "bfs_state", tok, adj, nodes, 0, 3, search_rng=np.random.default_rng(seed))
        changed = changed or row.tokens != base.tokens
    assert changed


def test_scale_plan_is_one_seed_on_gpus_1_to_7():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import scale

    args = scale.parse_args(["--dry-run", "--run-name", "scale_v1_s0"])
    assert args.gpus == [1, 2, 3, 4, 5, 6, 7]
    _output, jobs = scale.build_jobs(args, ROOT)
    assert len(jobs) == 68
    by_name = {job.name: job for job in jobs}
    assert by_name["context/l4/direct"].steps == 50000
    assert by_name["context/l6/path"].steps == 75000
    assert by_name["context/l8/dfs_state"].steps == 100000
    assert by_name["context/l10/bfs_nodes"].steps == 125000
    assert by_name["context/curriculum/bfs_state"].steps == 200000
    assert by_name["fixed/l6/stage_a"].steps == 1000
    assert by_name["fixed/curriculum/stage_b/dfs_nodes"].steps == 200000
    assert by_name["fixed/l10/stage_b/path"].depends_on == "fixed/l10/stage_a"
    path = " ".join(by_name["context/l6/path"].overrides)
    bfs = " ".join(by_name["fixed/curriculum/stage_b/bfs_state"].overrides)
    assert "protocol.negative_random_path=true" in path
    assert "protocol.random_search_order=true" in bfs
    assert "evaluation.selection_min_step=150000" in bfs
    assert "train.one_hop_fraction=0.25" in " ".join(by_name["fixed/l4/stage_b/direct"].overrides)
    prepared = scale.geometries()
    assert len(prepared) == 30
    length = next(item for item in prepared if item[0] == "context" and item[1] == 12 and item[2] == 12)
    assert length[3] == 2
    longer = load_config(
        [ROOT / "configs/experiments/retrain_context.yaml"],
        scale.data_overrides(*length),
    )
    assert longer.data.n_nodes == 144 and longer.protocol.cot_margin == 8


def test_scale_weights_must_match_the_roots():
    with pytest.raises(ConfigError, match="scale_weights"):
        load_config(
            [ROOT / "configs/experiments/retrain_context.yaml"],
            [
                "protocol.cot_margin=8",
                'data.root="data/a"',
                'train.scale_roots=["data/a","data/b"]',
                "train.scale_weights=[1.0]",
            ],
        )
