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


def _decode_step(model, cache, token_ids: torch.Tensor, precision: str, active=None) -> torch.Tensor:
    device = token_ids.device
    positions = cache.lengths.clamp(max=cache.max_len - 1).unsqueeze(1)
    with _autocast(device, precision):
        logits = model(token_ids, positions=positions, cache=cache, cache_mode="decode")
    cache.lengths += 1 if active is None else active.to(cache.lengths.dtype)
    return logits[:, 0, :]


@torch.no_grad()
def generate(model: TransformerLM, prompts: list[list[int]], protocol: str,
             tok: Tokenizer, precision: str = "fp32", *, filler_tokens: int = 0,
             max_new_tokens: int = 0) -> list[dict]:
    """Greedy free generation; the first YES/NO terminates a response.

    There is no oracle length, cutoff token or forced answer. EOS before an
    answer and exhausted context/safety limits are failures. Trace errors do
    not veto a correct answer. Direct/Filler sample exactly one answer token.
    """
    if not prompts:
        return []
    if protocol not in {"direct", "path", "dfs_state", "bfs_state", "dfs_nodes", "bfs_nodes", "filler"}:
        raise ValueError(protocol)
    if protocol == "filler" and filler_tokens < 1:
        raise ValueError("filler_tokens must be positive")
    device, dtype = _device_dtype(model, precision)
    model.eval()
    dots = [tok.id("<DOT>")] * filler_tokens if protocol == "filler" else []
    inputs = [prompt + dots for prompt in prompts]
    if any(len(row) >= model.max_seq_len for row in inputs):
        raise ValueError("prompt/filler leaves no room for an answer")
    tokens, lengths = pad_rows(inputs, tok.pad, device)
    with _autocast(device, precision):
        if protocol in {"direct", "filler"}:
            logits = model(tokens)
        else:
            cache = model.new_cache(len(prompts), device, dtype)
            positions = torch.arange(tokens.shape[1], device=device).expand(len(prompts), -1)
            logits = model(tokens, positions=positions, cache=cache, cache_mode="prefill")
            cache.lengths.copy_(lengths)
    next_logits = _gather_row(logits, lengths - 1)
    caps = [min(model.max_seq_len - len(p), max_new_tokens or model.max_seq_len) for p in inputs]
    if protocol in {"direct", "filler"}:
        caps = [1] * len(prompts)
    generated = [list(dots) for _ in prompts]
    traces = [list(dots) for _ in prompts]
    answers = [None] * len(prompts)
    active = [True] * len(prompts)
    limits = [False] * len(prompts)
    counts = [0] * len(prompts)
    eos = tok.id("<EOS>")
    while any(active):
        # One transfer per decode step, not per row.
        picked = next_logits.argmax(dim=-1).tolist()
        feed = [tok.pad] * len(prompts)
        for i in range(len(prompts)):
            if not active[i]:
                continue
            pred = picked[i]
            generated[i].append(pred)
            counts[i] += 1
            if protocol in {"direct", "filler"}:
                # Preserve the raw sampled answer for diagnostics even when
                # it is outside YES/NO; never coerce it into a binary answer.
                answers[i] = pred
            if pred in (tok.yes, tok.no):
                answers[i] = pred
                active[i] = False
            elif protocol in {"direct", "filler"} or pred == eos:
                active[i] = False
            else:
                traces[i].append(pred)
                if counts[i] >= caps[i]:
                    limits[i] = True
                    active[i] = False
                else:
                    feed[i] = pred
        if any(active):
            step_tokens = torch.tensor(feed, dtype=torch.long, device=device).unsqueeze(1)
            mask = torch.tensor(active, dtype=torch.bool, device=device)
            next_logits = _decode_step(model, cache, step_tokens, precision, mask)
    results = []
    for i, prompt in enumerate(prompts):
        pred = int(answers[i] == tok.yes) if answers[i] in (tok.yes, tok.no) else None
        results.append({"prompt_len": len(prompt), "generated": generated[i],
                        "trace_tokens": traces[i], "answer_token": answers[i], "pred": pred,
                        "format_fail": pred is None, "generation_limit_reached": limits[i],
                        "tokens": prompt + generated[i], "filler_prefill": protocol == "filler"})
    return results
