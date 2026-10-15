"""DFS, tokenization, loss masks, and answer scoring."""

from __future__ import annotations

import torch

from cot_navigation.data.dataset import collate_examples
from cot_navigation.protocols.dfs import dfs_search, witness_path
from cot_navigation.protocols.render import render_steps, render_task, render_walk
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


def test_trace_roundtrip_and_budget_slice_keeps_answer_tail():
    tok = Tokenizer(16)
    adj = [[1, 2], [], [3], []]
    node_token = [tok.node_id(i) for i in range(4)]
    steps = dfs_search(adj, 0, 3)
    body = render_steps(steps, node_token, tok)
    parsed = parse_completed_steps(body, tok)
    broken = [tok.id("<STEP>"), tok.id("<CUR>"), tok.id("<ENDSTEP>")]
    assert parse_completed_steps(broken, tok) == []
    report = compare_completed(steps, parsed, node_token)
    assert report["first_error_record"] is None and report["n_completed"] == len(steps)
    prompt = [tok.id("<QUERY>"), node_token[0], node_token[3], tok.mode_id("dfs_state"), tok.id("<BUDGET>"), tok.budget_id(16), tok.id("<OUT>")]
    full = render_task(tok, "dfs_state", 16, 1, prompt, adj, 0, 3, node_token)
    short = render_task(tok, "dfs_state", 5, 1, prompt, adj, 0, 3, node_token)
    assert short.tokens[short.prompt_len : short.prompt_len + 5] == full.meta["full_body"][:5]
    assert short.cut and short.tokens[short.answer_index - 2] == tok.id("<CUT>")
    assert short.tokens[short.answer_index - 1] == tok.id("<ANS>")
    assert short.tokens[short.answer_index] == tok.yes
    # The tail is outside T, so the answer token is still present.
    assert tok.id("<ANS>") in short.tokens
    finished = render_task(tok, "path", 64, 1, prompt, adj, 0, 3, node_token)
    assert not finished.cut
    assert finished.tokens[finished.answer_index] == tok.yes


def test_loss_masks_shift_and_do_not_supervise_filler_dots_or_walk_eos():
    tok = Tokenizer(16)
    node_token = [tok.node_id(i) for i in range(4)]
    # 0 reaches 2 and does not reach the isolated node 3.
    adj = [[1], [2], [], []]
    prompt = [tok.id("<QUERY>"), node_token[0], node_token[3], tok.mode_id("filler"), tok.id("<BUDGET>"), tok.budget_id(16), tok.id("<OUT>")]
    filler = render_task(tok, "filler", 16, 0, prompt, adj, 0, 3, node_token)
    assert filler.trace_indices == []
    assert filler.tokens[filler.answer_index] == tok.no
    assert filler.tokens[filler.prompt_len : filler.prompt_len + 16] == [tok.id("<DOT>")] * 16
    dfs_prompt = [tok.id("<QUERY>"), node_token[0], node_token[2], tok.mode_id("dfs_state"), tok.id("<BUDGET>"), tok.budget_id(16), tok.id("<OUT>")]
    dfs = render_task(tok, "dfs_state", 16, 1, dfs_prompt, adj, 0, 2, node_token)
    assert dfs.answer_index not in dfs.trace_indices
    assert all(i >= dfs.prompt_len for i in dfs.trace_indices)
    walk = render_walk(tok, node_token, [0, 1, 2])
    assert walk.walk_indices == [2, 3]
    assert walk.tokens[-1] == tok.id("<EOS>")
    assert (len(walk.tokens) - 1) not in walk.walk_indices
    batch = collate_examples([dfs, filler], tok.pad, multiple=4)
    assert batch["tokens"].shape[1] % 4 == 0
    logits = torch.zeros(2, batch["tokens"].shape[1], len(tok))
    # Put mass on the supervised targets so the answer term is finite and the pad targets are irrelevant.
    for row, example in enumerate((dfs, filler)):
        for index, token in enumerate(example.tokens):
            logits[row, max(index - 1, 0), token] = 5
    parts = combine_losses(
        logits,
        torch.tensor(batch["tokens"]),
        torch.tensor(batch["answer_mask"]),
        torch.tensor(batch["trace_mask"]),
        torch.tensor(batch["walk_mask"]),
        n_task=2,
        n_walk=0,
    )
    assert torch.isfinite(parts["total_loss"])
    # Changing the label token must not change the logit that predicts it.
    tokens = torch.tensor(batch["tokens"])
    changed = tokens.clone()
    changed[0, dfs.answer_index] = tok.no if tokens[0, dfs.answer_index] == tok.yes else tok.yes
    from cot_navigation.models.transformer import TransformerLM

    model = TransformerLM(len(tok), 2, 1, 32, 64, 128, attn_impl="math")
    model.eval()
    with torch.no_grad():
        before = model(tokens)
        after = model(changed)
    pos = dfs.answer_index - 1
    assert torch.allclose(before[0, : pos + 1], after[0, : pos + 1])


def test_score_uses_only_the_answer_slot():
    tok = Tokenizer(8)
    yes, no, ans = tok.yes, tok.no, tok.id("<ANS>")
    # YES appears in the trace, but the slot says NO. Label YES is therefore wrong.
    tokens = [tok.id("<STEP>"), yes, ans, no]
    scored = score_final_answer(tokens, prompt_len=0, label=1, tok=tok)
    assert scored["correct"] is False and scored["format_fail"] is False and scored["pred"] == 0
    # Wrong trace tokens, right slot.
    tokens = [1, 2, 3, ans, yes]
    scored = score_final_answer(tokens, 0, 1, tok)
    assert scored["correct"] is True and scored["format_fail"] is False
    # No legal slot.
    scored = score_final_answer([yes, yes], 0, 1, tok)
    assert scored["format_fail"] and scored["correct"] is False
