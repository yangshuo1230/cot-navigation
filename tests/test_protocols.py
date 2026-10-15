"""DFS, tokenization, loss masks, and answer scoring."""

from __future__ import annotations

import pytest
import torch

from cot_navigation.data.dataset import collate_examples
from cot_navigation.protocols.dfs import dfs_search, witness_path
from cot_navigation.protocols.render import render_steps, render_task, render_walk, context_prompt, parametric_prompt
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.validate_trace import compare_completed, parse_completed_steps, score_final_answer
from cot_navigation.training.loss import combine_losses


def test_dfs_chain_dead_end_merge_and_empty():
    chain = dfs_search([[1], [2], []], 0, 2)
    assert [step.action for step in chain] == ["PUSH", "PUSH", "FOUND"]
    assert witness_path(chain) == [0, 1, 2]
    dead = dfs_search([[1, 2], [], [3], []], 0, 3)
    assert [step.action for step in dead] == ["PUSH", "POP", "PUSH", "PUSH", "FOUND"]
    assert dead[2].cands == (2,)
    merge = dfs_search([[1, 2], [3], [3], []], 0, 4)
    at_second_branch = [step for step in merge if step.cur == 2][0]
    assert 3 not in at_second_branch.cands
    assert dfs_search([[], []], 0, 1)[0].action == "EXHAUST"
    long = dfs_search([[1], [2], [3], [4], []], 0, 4)
    assert all(step.action == "PUSH" for step in long[:-1]) and long[-1].action == "FOUND"
    # Several pops walking back out of a dead chain toward an empty root.
    popped = dfs_search([[1], [2], [], []], 0, 3)
    assert [step.action for step in popped].count("POP") >= 2
    assert popped[-1].action == "EXHAUST"


def test_compact_vocabulary_and_explicit_edges_queries():
    tok = Tokenizer()
    assert len(tok) == 270 and tok.version == "v2"
    for old in ("<ANS>", "<CUT>", "<BUDGET>", "<DFS>", "<CUR>", "<NONE>"):
        with pytest.raises(KeyError):
            tok.id(old)
    nodes = [tok.node_id(i) for i in range(4)]
    assert context_prompt(tok, nodes, [(0,1),(1,2)], 0, 3) == [
        tok.id("<EDGE>"), nodes[0], nodes[1], tok.id("<EDGE>"), nodes[1], nodes[2],
        tok.id("<QUERY>"), nodes[0], nodes[3], tok.id("|")]


def test_complete_dfs_roundtrip_and_no_repeated_cur_field():
    tok = Tokenizer(16)
    adj = [[1,2], [3], [4], [], [5], []]
    nodes = [tok.node_id(i) for i in range(6)]
    steps = dfs_search(adj, 0, 5)
    prompt = parametric_prompt(tok, nodes, 0, 5)
    row = render_task(tok, "dfs_state", 1, prompt, adj, 0, 5, nodes)
    body = row.tokens[row.prompt_len:row.answer_index]
    parsed = parse_completed_steps(body, tok)
    report = compare_completed(steps, parsed, nodes)
    assert report["complete_match"] and report["matched_records"] == 8
    assert body.count(tok.id("|")) == 3*len(steps)
    assert body.count(tok.id(";")) == len(steps)
    assert row.tokens[-2:] == [tok.yes, tok.id("<EOS>")]
    assert parsed[4]["visited"] == [nodes[v] for v in (0,1,3)]
    assert parse_completed_steps([tok.id("|"), tok.id(";")], tok) == []
    # A correctly terminated prefix is not a complete search.
    assert not compare_completed(steps, parsed[:2], nodes)["complete_match"]


@pytest.mark.parametrize("protocol", ["direct", "path", "dfs_state", "filler"])
def test_negative_outputs_and_loss_masks(protocol):
    tok = Tokenizer(8)
    nodes = [tok.node_id(i) for i in range(4)]
    adj = [[1], [2], [], []]
    prompt = parametric_prompt(tok, nodes, 0, 3)
    row = render_task(tok, protocol, 0, prompt, adj, 0, 3, nodes, filler_tokens=16)
    assert row.tokens[row.answer_index] == tok.no
    assert row.answer_index not in row.trace_indices
    assert all(i >= row.prompt_len for i in row.trace_indices)
    if protocol in {"direct", "path"}:
        assert row.tokens[row.prompt_len:] == [tok.no, tok.id("<EOS>")]
    if protocol in {"direct", "filler"}:
        assert row.trace_indices == []
    if protocol == "filler":
        assert row.tokens[row.prompt_len:row.answer_index] == [tok.id("<DOT>")]*16
    walk = render_walk(tok, nodes, [0,1,2])
    assert walk.walk_indices == [2,3] and len(walk.tokens)-1 not in walk.walk_indices
    batch = collate_examples([row, walk], tok.pad, multiple=8)
    logits = torch.zeros(*batch["tokens"].shape, len(tok), requires_grad=True)
    parts = combine_losses(logits, torch.tensor(batch["tokens"]),
        torch.tensor(batch["answer_mask"]), torch.tensor(batch["trace_mask"]),
        torch.tensor(batch["walk_mask"]), n_task=1, n_walk=1)
    assert torch.isfinite(parts["total_loss"])
    parts["total_loss"].backward()
    assert logits.grad is not None


def test_answer_has_no_causal_leakage():
    from cot_navigation.models.transformer import TransformerLM
    tok = Tokenizer(8)
    nodes = [tok.node_id(i) for i in range(3)]
    row = render_task(tok, "path", 1, parametric_prompt(tok,nodes,0,2), [[1],[2],[]],0,2,nodes)
    tokens = torch.tensor([row.tokens]);changed = tokens.clone()
    changed[0,row.answer_index] = tok.no
    model = TransformerLM(len(tok),2,1,16,32,128,attn_impl="math").eval()
    with torch.no_grad():
        before, after = model(tokens), model(changed)
    assert torch.equal(before[:,:row.answer_index],after[:,:row.answer_index])


def test_final_label_is_independent_of_trace_correctness():
    tok = Tokenizer(8)
    wrong_trace = [tok.id("<POP>"),tok.id(";"),tok.node_id(7)]
    assert score_final_answer(wrong_trace+[tok.yes,tok.id("<EOS>")],0,1,tok)["correct"]
    assert score_final_answer(wrong_trace+[tok.no],0,1,tok)["pred"] == 0
    assert score_final_answer(wrong_trace+[tok.id("<EOS>"),tok.yes],0,1,tok)["format_fail"]
