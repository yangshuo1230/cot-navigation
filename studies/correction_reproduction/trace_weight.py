"""Per-example whole-CoT weighting; independent of training step/graph scale."""
import torch
from cot_navigation.training.loss import combine_losses


def build_policy(cfg=None):
    return dict(version=2, basis='per_example_actual_cot_tokens_before_answer',
        tokens_per_weight=128, lower=1., upper=16.,
        formula='lambda_i=clip(L_i/128,1,16)',
        normalization='2*(answer_ce_i+lambda_i*mean_trace_ce_i)/(1+lambda_i)',
        include_error_and_repair_tokens=True, exclude_answer_eos_padding=True)


def weights_from_masks(answer_mask, trace_mask, tokens_per_weight=128):
    """Count CoT targets before the answer, excluding trailing EOS and padding.

    The existing trace loss mask is left intact. In this full-loss suite it
    includes every CoT token, including artificial errors and repairs.
    """
    before_answer = answer_mask[:, 1:].cumsum(dim=-1) == 0
    lengths = (trace_mask[:, 1:] * before_answer).sum(dim=-1).float()
    weights = (lengths / tokens_per_weight).clamp(min=1., max=16.)
    return lengths, weights


def combine_length_losses(logits, tokens, answer_mask, trace_mask, walk_mask,
                          n_task, n_walk, walk_weight=1., trace_lambda=1.,
                          tokens_per_weight=128):
    # Scalar trace_lambda is accepted for the shared trainer signature only.
    lengths, weights = weights_from_masks(answer_mask, trace_mask, tokens_per_weight)
    parts = combine_losses(logits, tokens, answer_mask, trace_mask, walk_mask,
                           n_task, n_walk, walk_weight, trace_lambda=weights)
    factor = torch.where(parts['has_trace'], 2. / (1. + weights), 1.)
    parts.update(trace_lengths=lengths, trace_weights=weights,
        weighted_answer=factor * parts['answer'],
        weighted_trace=factor * weights * parts['trace'])
    return parts
