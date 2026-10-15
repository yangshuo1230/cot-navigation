"""Model numerics: cache, padding, shared FFN, accumulation, reference path."""

from __future__ import annotations

import torch
from torch import nn

from cot_navigation.evaluation.rollout import cached_logits, full_logits, generate
from cot_navigation.models.attention import KVCache
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.loss import combine_losses


def _tiny(attn="math", ffn="standard"):
    model = TransformerLM(80, layers=2, heads=1, d_model=32, d_ff=64, max_seq_len=64, ffn=ffn, attn_impl=attn)
    model.eval()
    return model


def test_cache_matches_full_forward_with_padding_and_q_len_1():
    torch.manual_seed(0)
    model = _tiny("math")
    tokens = torch.randint(1, 40, (2, 7))
    lengths = torch.tensor([4, 7])
    tokens[0, 4:] = 0
    full = full_logits(model, tokens)
    cached = cached_logits(model, tokens, lengths)
    for row in range(2):
        assert torch.allclose(full[row, : lengths[row]], cached[row, : lengths[row]], atol=1e-4, rtol=1e-4)
    # A single unpadded row matches the padded batch on its valid prefix.
    single = full_logits(model, tokens[0:1, :4])
    assert torch.allclose(single[0], full[0, :4], atol=1e-4, rtol=1e-4)


def test_sdpa_matches_math_logits_and_grads():
    torch.manual_seed(1)
    math_model = _tiny("math")
    sdpa = _tiny("sdpa")
    sdpa.load_state_dict(math_model.state_dict())
    tokens = torch.randint(1, 40, (3, 9))
    math_model.train()
    sdpa.train()
    logits_m = math_model(tokens)
    logits_s = sdpa(tokens)
    assert torch.allclose(logits_m, logits_s, atol=1e-4, rtol=1e-4)
    loss_m = logits_m[:, -1].float().square().mean()
    loss_s = logits_s[:, -1].float().square().mean()
    loss_m.backward()
    loss_s.backward()
    for p_m, p_s in zip(math_model.parameters(), sdpa.parameters()):
        assert torch.allclose(p_m.grad, p_s.grad, atol=1e-4, rtol=1e-4)


def test_return_attn_matches_math_and_is_not_stored_by_default():
    torch.manual_seed(2)
    model = _tiny("sdpa")
    tokens = torch.randint(1, 20, (1, 5))
    with torch.no_grad():
        plain = model(tokens)
        weighted, weights = model(tokens, return_attn=True)
    assert torch.allclose(plain, weighted, atol=1e-4, rtol=1e-4)
    assert weights[0].shape[-1] == 5
    assert torch.allclose(weights[0].sum(-1), torch.ones_like(weights[0].sum(-1)), atol=1e-4)


def test_shared_ffn_is_one_parameter_in_the_optimizer():
    model = _tiny(ffn="shared_dense")
    assert model.blocks[0].ffn is model.blocks[1].ffn
    ids = [id(p) for p in model.parameters()]
    assert len(ids) == len(set(ids))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt_ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(opt_ids) == len(set(opt_ids))
    report = model.report()
    assert report["ffn"] == "shared_dense"
    dense = _tiny(ffn="standard")
    assert report["unique_parameters"] < dense.report()["unique_parameters"]


def test_microbatch_gradients_match_full_batch():
    torch.manual_seed(3)
    model = _tiny("math")
    tokens = torch.randint(1, 30, (4, 8))
    answer = torch.zeros(4, 8)
    trace = torch.zeros(4, 8)
    answer[:, 6] = 1
    trace[:, 1:6] = 1
    walk = torch.zeros(4, 8)

    def grads(chunks):
        model.zero_grad(set_to_none=True)
        total = 0
        for rows in chunks:
            part = combine_losses(model(tokens[rows]), tokens[rows], answer[rows], trace[rows], walk[rows], n_task=4, n_walk=0)
            part["total_loss"].backward()
            total = total + part["total_loss"].detach()
        return total, [p.grad.detach().clone() for p in model.parameters()]

    full, g_full = grads([slice(0, 4)])
    model.zero_grad(set_to_none=True)
    split, g_split = grads([slice(0, 1), slice(1, 4)])
    assert torch.allclose(full, split, atol=1e-5)
    for a, b in zip(g_full, g_split):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_activation_capture_is_optional():
    model = _tiny("math")
    tokens = torch.randint(1, 10, (1, 3))
    with torch.no_grad():
        plain = model(tokens)
        captured = {}
        other = model(tokens, capture=captured)
    assert "resid" in captured and len(captured["resid"]) == 2
    assert torch.allclose(plain, other)


class _Scripted(nn.Module):
    def __init__(self, script, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, 4)
        self.script = list(script)
        self.cursor = 0
        self.vocab = vocab
        self.max_seq_len = 32

    def new_cache(self, batch, device, dtype):
        return KVCache(1, batch, 1, 32, 4, device, torch.float32)

    def forward(self, tokens, positions=None, cache=None, cache_mode=None, capture=None, return_attn=False):
        logits = torch.full((tokens.shape[0], tokens.shape[1], self.vocab), -30.0)
        index = min(self.cursor, len(self.script) - 1)
        logits[:, -1, self.script[index]] = 30
        self.cursor += 1
        return logits

    def eval(self):
        return self


def test_free_generation_stops_on_label_and_never_injects_answer():
    tok = Tokenizer(8)
    trace = tok.node_id(1)
    prompt = [tok.id("<QUERY>"),tok.node_id(0),tok.node_id(2),tok.id("|")]
    [ok] = generate(_Scripted([trace,tok.yes],len(tok)),[prompt],"path",tok)
    assert ok["generated"] == [trace,tok.yes] and ok["pred"] == 1
    [limit] = generate(_Scripted([trace],len(tok)),[prompt],"path",tok,max_new_tokens=2)
    assert limit["generation_limit_reached"] and limit["format_fail"] and limit["pred"] is None
    assert limit["generated"] == [trace,trace]
    [bad] = generate(_Scripted([trace,tok.yes],len(tok)),[prompt],"direct",tok)
    assert bad["format_fail"] and bad["pred"] is None
    [good] = generate(_Scripted([tok.no],len(tok)),[prompt],"direct",tok)
    assert good["generated"] == [tok.no] and good["pred"] == 0
    [eos] = generate(_Scripted([tok.id("<EOS>"),tok.yes],len(tok)),[prompt],"dfs_state",tok)
    assert eos["format_fail"] and eos["answer_token"] is None


def test_filler_prefill_matches_teacher_forced_answer_without_ans():
    tok=Tokenizer(16)
    model=TransformerLM(len(tok),2,1,16,32,128,attn_impl="math").eval()
    prompt=[tok.id("<QUERY>"),tok.node_id(0),tok.node_id(2),tok.id("|")]
    prefix=prompt+[tok.id("<DOT>")]*16
    with torch.no_grad():
        expected=int(model(torch.tensor([prefix]))[0,-1].argmax())
    [row]=generate(model,[prompt],"filler",tok,filler_tokens=16)
    assert row["generated"]==[tok.id("<DOT>")]*16+[expected]
    assert row["filler_prefill"]


def test_mixed_prompt_lengths_can_finish_at_different_context_limits():
    torch.manual_seed(23)
    tok=Tokenizer(8)
    model=TransformerLM(len(tok),2,1,16,32,32,attn_impl="math").eval()
    prompts=[[tok.id("|")]*30,[tok.id("|")]*4]
    batched=generate(model,prompts,"path",tok,max_new_tokens=6)
    separate=[generate(model,[prompt],"path",tok,max_new_tokens=6)[0] for prompt in prompts]
    for got,want in zip(batched,separate):
        assert got["generated"]==want["generated"]
        assert got["generation_limit_reached"]==want["generation_limit_reached"]
