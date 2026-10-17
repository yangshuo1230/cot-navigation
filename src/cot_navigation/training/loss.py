"""Per-example losses.

Shift: logits at index ``i`` predict token ``i+1``. The hidden state that
predicts the label is the preceding prefix position, so the label token is not an input to
its own prediction. Padding and non-target tokens have mask 0 and are excluded
before the per-example mean, so a longer microbatch does not change an
example's weight.

Path/DFS task loss is ``answer CE + mean(trace CE)`` for that example.
Direct/filler task loss is the answer CE only. Walk loss is the mean CE of
transition nodes. The logical-batch reduction divides by the number of task
queries and the number of walks separately.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_nll(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Return NLL of shape ``[B, L-1]`` for targets ``tokens[:, 1:]``, in fp32."""
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = tokens[:, 1:].unsqueeze(-1)
    return -log_probs.gather(-1, targets).squeeze(-1)


def masked_mean(nll: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean over supervised target positions. ``target_mask`` is aligned to ``tokens`` (unshifted)."""
    mask = target_mask[:, 1:].to(dtype=nll.dtype)
    denom = mask.sum(dim=-1)
    value = (nll * mask).sum(dim=-1) / denom.clamp(min=1.0)
    return value, denom > 0


def combine_losses(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    answer_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    walk_mask: torch.Tensor,
    n_task: int,
    n_walk: int,
    walk_weight=1.0,
) -> dict[str, torch.Tensor]:
    nll = token_nll(logits, tokens)
    answer, has_answer = masked_mean(nll, answer_mask)
    trace, has_trace = masked_mean(nll, trace_mask)
    walk, has_walk = masked_mean(nll, walk_mask)
    zeros = torch.zeros_like(answer)
    task = torch.where(has_answer, answer + torch.where(has_trace, trace, zeros), zeros)
    walk_term = torch.where(has_walk, walk, zeros)
    total = None
    if n_task > 0:
        total = task.sum() / n_task
    if n_walk > 0:
        walk_mean = walk_term.sum() / n_walk
        total = walk_weight * walk_mean if total is None else total + walk_weight * walk_mean
    if total is None:
        total = logits.new_zeros(())
    # Counts stay tensors so the training step does not synchronize just to log.
    return {
        "total_loss": total,
        "answer": answer,
        "trace": trace,
        "walk": walk_term,
        "has_answer": has_answer,
        "has_trace": has_trace,
        "has_walk": has_walk,
        "n_task": n_task,
        "n_walk": n_walk,
    }
