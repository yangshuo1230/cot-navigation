"""Frozen query/adjacency objective, independent of data storage."""
import torch

def logical_loss(logits, tokens, lengths, answers, groups, denominators, trace_weight, knowledge_weight, use_trace):
    """Each microbatch contributes to ONE global logical objective.

    Normalizers are global across all microbatches. Never divide each chunk
    independently: doing that would overweight long singleton DFS examples.
    """
    import torch.nn.functional as F
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                         tokens[:, 1:].reshape(-1), reduction='none').reshape(tokens.shape[0], -1)
    pos = torch.arange(1, tokens.shape[1], device=tokens.device)[None, :]
    task = groups < 2
    answer_mask = (pos == answers[:, None]) & task[:, None]
    trace_mask = (((pos >= 4) & (pos < answers[:, None])) | (pos == answers[:, None] + 1)) & task[:, None]
    trace_mask = trace_mask & use_trace
    knowledge_mask = (pos >= 3) & (pos < lengths[:, None]) & (~task[:, None])
    ans = (ce * answer_mask).sum(-1)
    trace = (ce * trace_mask).sum(-1) / trace_mask.sum(-1).clamp_min(1)
    a, t = ans.sum() / denominators[0], trace.sum() / denominators[0]
    k = (ce * knowledge_mask).sum() / denominators[1].clamp_min(1)
    values = [a + trace_weight * t + knowledge_weight * k, a, t, k]
    for group, denom in [(0, denominators[2]), (1, denominators[3])]:
        mask = groups == group
        values.extend([(ans * mask).sum() / denom.clamp_min(1), (trace * mask).sum() / denom.clamp_min(1)])
    return torch.stack(values)
