"""Free generation and cached logits.

Filler's default answer pass is one causal prefill of the fixed DOT block.
That latency is not autoregressive CoT decode. Path and DFS count a prefill of
the prompt, then one cached step per generated token. Decode never uses a
square ``is_causal`` mask: the new query attends to every cache key at an
absolute position less than or equal to its own.
"""

from __future__ import annotations

import torch

from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer


def _device_dtype(model: TransformerLM, precision: str):
    device = next(model.parameters()).device
    if precision == "bf16" and device.type == "cuda":
        return device, torch.bfloat16
    return device, torch.float32


def _autocast(device: torch.device, precision: str):
    enabled = precision == "bf16" and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def pad_rows(rows: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    tokens = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        if not row:
            raise ValueError("empty sequence")
        tokens[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        lengths[i] = len(row)
    return tokens, lengths


def _gather_row(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch, index]


@torch.no_grad()
def full_logits(model: TransformerLM, tokens: torch.Tensor, precision: str = "fp32") -> torch.Tensor:
    device = tokens.device
    with _autocast(device, precision):
        return model(tokens)


@torch.no_grad()
def cached_logits(model: TransformerLM, tokens: torch.Tensor, lengths: torch.Tensor, precision: str = "fp32") -> torch.Tensor:
    """Logits at every position via prefill of token 0 and q_len=1 decode after that.

    Rows may be right-padded. Callers compare only positions ``< lengths``.
    """
    device, dtype = _device_dtype(model, precision)
    batch, length = tokens.shape
    cache = model.new_cache(batch, device, dtype)
    with _autocast(device, precision):
        first = tokens[:, :1]
        positions = torch.zeros(batch, 1, dtype=torch.long, device=device)
        logits0 = model(first, positions=positions, cache=cache, cache_mode="prefill")
        cache.lengths.fill_(1)
        columns = [logits0[:, 0, :]]
        for pos in range(1, length):
            step_pos = cache.lengths.clone().unsqueeze(1)
            step = model(tokens[:, pos : pos + 1], positions=step_pos, cache=cache, cache_mode="decode")
            cache.lengths += 1
            columns.append(step[:, 0, :])
    return torch.stack(columns, dim=1)


def _decode_step(model, cache, token_ids: torch.Tensor, precision: str) -> torch.Tensor:
    device = token_ids.device
    positions = cache.lengths.clone().unsqueeze(1)
    with _autocast(device, precision):
        logits = model(token_ids, positions=positions, cache=cache, cache_mode="decode")
    cache.lengths += 1
    return logits[:, 0, :]


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompts: list[list[int]],
    budgets: list[int],
    protocol: str,
    tok: Tokenizer,
    precision: str = "fp32",
) -> list[dict]:
    """Greedy generation. The true label is not an input.

    Path/DFS insert ``<CUT><ANS>`` once the trace body reaches ``T`` and then
    sample one answer token. Reaching ``T`` does not itself mean NO.
    """
    if protocol == "filler":
        return _generate_filler(model, prompts, budgets, tok, precision)
    # Direct trains only the label conditional on ANS. ANS is a fixed protocol
    # delimiter, not an unsupervised token that the model must invent at test time.
    input_prompts = [p + [tok.id("<ANS>")] for p in prompts] if protocol == "direct" else prompts
    device, dtype = _device_dtype(model, precision)
    model.eval()
    tokens, lengths = pad_rows(input_prompts, tok.pad, device)
    batch = len(prompts)
    cache = model.new_cache(batch, device, dtype)
    positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0).expand(batch, -1)
    with _autocast(device, precision):
        prefill_logits = model(tokens, positions=positions, cache=cache, cache_mode="prefill")
    cache.lengths.copy_(lengths)
    next_logits = _gather_row(prefill_logits, lengths - 1)
    phase = ["sample_answer" if protocol == "direct" else "body" for _ in range(batch)]
    trace_count = [0] * batch
    traces: list[list[int]] = [[] for _ in range(batch)]
    generated: list[list[int]] = [[tok.id("<ANS>")] if protocol == "direct" else [] for _ in range(batch)]
    answers: list[int | None] = [None] * batch
    forced_cut = [False] * batch
    failed = [False] * batch
    ans_id = tok.id("<ANS>")
    cut_id = tok.id("<CUT>")
    guard = 0
    limit = max(budgets) + 6
    while any(item != "done" for item in phase) and guard < limit:
        guard += 1
        feed = [tok.pad] * batch
        need_feed = False
        for i in range(batch):
            if phase[i] == "done":
                continue
            if phase[i] == "sample_answer":
                pred = int(next_logits[i].argmax().item())
                answers[i] = pred
                generated[i].append(pred)
                phase[i] = "done"
                continue
            if phase[i] == "force_ans":
                feed[i] = ans_id
                generated[i].append(ans_id)
                phase[i] = "sample_answer"
                need_feed = True
                continue
            if phase[i] == "body" and trace_count[i] >= budgets[i]:
                feed[i] = cut_id
                generated[i].append(cut_id)
                forced_cut[i] = True
                phase[i] = "force_ans"
                need_feed = True
                continue
            pred = int(next_logits[i].argmax().item())
            generated[i].append(pred)
            feed[i] = pred
            need_feed = True
            if pred == ans_id:
                phase[i] = "sample_answer"
            else:
                traces[i].append(pred)
                trace_count[i] += 1
        if not need_feed:
            break
        step_tokens = torch.tensor(feed, dtype=torch.long, device=device).unsqueeze(1)
        next_logits = _decode_step(model, cache, step_tokens, precision)
    results = []
    for i, prompt in enumerate(prompts):
        if phase[i] != "done":
            failed[i] = True
        pred = None
        if answers[i] in (tok.yes, tok.no):
            pred = 1 if answers[i] == tok.yes else 0
        elif answers[i] is not None:
            failed[i] = True
        results.append(
            {
                "prompt_len": len(prompt),
                "generated": generated[i],
                "trace_tokens": traces[i],
                "answer_token": answers[i],
                "pred": pred,
                "format_fail": bool(failed[i] or pred is None),
                "forced_cut": forced_cut[i],
                "tokens": prompt + generated[i],
            }
        )
    return results


def _generate_filler(model, prompts, budgets, tok: Tokenizer, precision: str) -> list[dict]:
    """One prefill of prompt + DOT*T + ANS. The DOT block is not decoded token by token."""
    device, _dtype = _device_dtype(model, precision)
    model.eval()
    rows = []
    ans_at = []
    for prompt, budget in zip(prompts, budgets):
        row = prompt + [tok.id("<DOT>")] * int(budget) + [tok.id("<ANS>")]
        ans_at.append(len(row) - 1)
        rows.append(row)
    tokens, lengths = pad_rows(rows, tok.pad, device)
    with _autocast(device, precision):
        logits = model(tokens)
    picked = _gather_row(logits, lengths - 1).argmax(dim=-1)
    results = []
    for i, prompt in enumerate(prompts):
        answer = int(picked[i].item())
        pred = 1 if answer == tok.yes else 0 if answer == tok.no else None
        # rows already end in ANS. Append the sampled label only.
        generated = rows[i][len(prompt) :] + [answer]
        results.append(
            {
                "prompt_len": len(prompt),
                "generated": generated,
                "trace_tokens": [tok.id("<DOT>")] * int(budgets[i]),
                "answer_token": answer,
                "pred": pred,
                "format_fail": pred is None,
                "forced_cut": False,
                "tokens": prompt + generated,
                "filler_prefill": True,
            }
        )
    return results
