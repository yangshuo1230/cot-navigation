"""Read-only CPU audit. Does not instantiate Trainer or start a W&B run."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.natural_mistakes import NaturalCorrectionPool, pool_path
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.data.mistakes import records, audit_reference
from cot_navigation.data.shards import file_sha256
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from worker import MatchedCleanPool, weight_hash, load_semantic_audit
from trace_weight import combine_length_losses


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-existing', action='store_true')
    args = p.parse_args()
    audit_file = HERE / 'data_audit.json'
    previous = json.loads(audit_file.read_text()) if args.verify_existing else None
    torch.set_num_threads(2)
    plan = json.loads((HERE / 'review_plan.json').read_text())
    jobs = plan['jobs']
    # Config paths are derived from the immutable prepared job names.
    generated = ROOT / 'configs/generated/correction_reproduction' / plan['run_name']
    configs = {(arm, seed): load_config([generated / f'{arm}_s{seed}.yaml'])
               for arm in ['original_clean', 'matched_clean', 'correction'] for seed in plan['seeds']}
    report = dict(training_launched=False, gpu_used=False, wandb_run_created=False,
                  torch_version=torch.__version__, python_version=sys.version.split()[0],
                  jobs=len(jobs), seeds=plan['seeds'], scales=[], initial_weights={}, eval_sets={})
    cfg = configs['correction', plan['seeds'][0]]
    policy = json.loads((HERE / 'trace_weight_policy.json').read_text())
    assert all(c.train.trace_loss_lambda == 1. for c in configs.values())
    assert policy['formula'] == 'lambda_i=clip(L_i/128,1,16)'
    # Mixed short/long rows in ONE batch must have different weights. Check
    # values/gradients, exclusion of EOS/padding, and compiled-graph reuse.
    from torch._dynamo.testing import CompileCounter
    counter = CompileCounter()
    loss_fn = torch.compile(combine_length_losses, backend=counter, fullgraph=True)
    rng = torch.Generator().manual_seed(7919)
    tokens = torch.randint(11, (3, 4103), generator=rng)
    walk_mask=torch.zeros(3,4103)
    for lengths in [[32,256,2048], [64,512,3072], [128,1024,4096]]:
        answer_mask=torch.zeros(3,4103); trace_mask=torch.zeros(3,4103)
        for row, length in enumerate(lengths):
            trace_mask[row,3:3+length]=1
            answer_mask[row,3+length]=1
            trace_mask[row,4+length]=1  # EOS is supervised but not CoT length
        logits=torch.randn(3,4103,11,generator=rng).requires_grad_(True)
        parts=loss_fn(logits,tokens,answer_mask,trace_mask,walk_mask,
                      n_task=3,n_walk=0)
        expected_weights=torch.tensor(lengths).float().div(128).clamp(1,16)
        assert torch.equal(parts['trace_lengths'],torch.tensor(lengths).float())
        assert torch.equal(parts['trace_weights'],expected_weights)
        expected=(2*(parts['answer']+expected_weights*parts['trace'])/(1+expected_weights)).mean()
        assert torch.allclose(parts['total_loss'],expected,atol=1e-6)
        actual_grad=torch.autograd.grad(parts['total_loss'],logits,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(expected,logits)[0]
        assert torch.allclose(actual_grad,expected_grad,atol=1e-6)
        # Length grouping/microbatching must leave the logical-batch objective unchanged.
        chunk_sum = sum(combine_length_losses(logits[i:i+1],tokens[i:i+1],
            answer_mask[i:i+1],trace_mask[i:i+1],walk_mask[i:i+1],n_task=3,n_walk=0)['total_loss']
            for i in range(3))
        assert torch.allclose(parts['total_loss'],chunk_sum,atol=1e-6)
    assert counter.frame_count == 1, counter.frame_count
    report['trace_weight_policy']=policy
    report['dynamic_loss_checks']=dict(mixed_length_batches=3,values_and_gradients=3,
        microbatch_invariance=True,eos_padding_excluded=True,compiled_graphs=counter.frame_count)
    tok = Tokenizer.from_config(cfg)
    donors = {}
    for root in cfg.train.scale_roots:
        source = ContextStore(root); prepared = ScalePreparedRows(source, cfg, tok)
        pool = NaturalCorrectionPool(source, cfg, tok)
        matched = MatchedCleanPool(source, cfg, tok)
        path = pool_path(source, cfg, tok)
        header = json.loads(path.with_suffix('.json').read_text())
        assert header['provenance']['settings']['seed'] == 0  # frozen mining seed
        counts = Counter()
        for i, split in enumerate(pool.arrays['split']):
            row = pool.row(i, 0); q = int(row.meta['index']); variant = int(row.meta['natural_variant'])
            clean = prepared.row(q, variant)
            assert int(source.arrays['split_id'][q]) == int(split) == int(row.meta['split'])
            assert row.label == clean.label and row.tokens[:row.prompt_len] == clean.tokens[:clean.prompt_len]
            assert len(row.tokens) <= cfg.model.max_seq_len
            audit_reference(clean, records(clean, tok), tok, fifo=False)
            for event in row.meta['natural_events']:
                assert row.tokens[event['mistake_action']] == tok.id('<MISTAKE>')
                assert row.tokens[event['repair_start']:event['repair_end']] == clean.tokens[event['original_start']:event['original_end']]
                assert row.tokens[event['candidate_start']:event['candidate_end']] == event['predicted']
                counts['events'] += 1
            counts[['train', 'val', 'test'][int(split)]] += 1
        rng_a = np.random.default_rng(4829); rng_b = np.random.default_rng(4829)
        for j in range(1000):
            a = pool.sample(j % 2, rng_a, .5); b = matched.sample(j % 2, rng_b, .5)
            assert a.meta['index'] == b.meta['index'] and a.label == b.label
            assert b.tokens == prepared.row(int(a.meta['index']), int(a.meta['natural_variant'])).tokens
        assert rng_a.bit_generator.state == rng_b.bit_generator.state
        report['scales'].append(dict(root=root, pool=str(path), sha256=file_sha256(path),
                                    counts=dict(counts), paired_sampling_checks=1000))
        for donor in header['provenance']['donors']:
            donors[donor['checkpoint']] = donor['sha256']
        del pool, matched, prepared, source
    report['donors'] = {}
    for name, expected in donors.items():
        assert file_sha256(Path(name)) == expected, f'donor changed: {name}'
        report['donors'][name] = expected
    for seed in plan['seeds']:
        hashes = []
        for arm in ['matched_clean', 'correction']:
            c = configs[arm, seed]; t = Tokenizer.from_config(c)
            torch.manual_seed(seed)
            model = build_model(c.model, len(t), pad_id=t.pad)
            hashes.append(weight_hash(model)); del model
        assert hashes[0] == hashes[1]
        report['initial_weights'][str(seed)] = hashes[0]
    assert len(set(report['initial_weights'].values())) == len(plan['seeds'])
    # Ensure all final held-out caches can actually be read BEFORE long training.
    sets = [('main', cfg.data.root)] + [s.split('=', 1) for s in cfg.evaluation.extra_sets]
    for name, root in sets:
        source = ContextStore(root)
        reference_cfg = cfg
        if name == 'hard':
            reference_cfg = load_config([ROOT / 'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml'])
        reference_tok = Tokenizer.from_config(reference_cfg)
        prepared = ScalePreparedRows(source, reference_cfg, reference_tok)
        from cot_navigation.evaluation.sampling import stratified_ids
        a = source.arrays
        ids = stratified_ids(source.indices('test'), a['shape_id'], a['distance'], a['label'], 512, 1729)
        assert len(ids) == 512
        for q in ids:
            row = prepared.row(int(q), 0)
            assert row.prompt_len < 4096 and int(row.meta['split']) == 2
        report['eval_sets'][name] = dict(root=root, n=len(ids),
            query_ids_sha256=hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
            reference_budget=reference_cfg.model.max_seq_len, generation_budget=4096, generation_batch=16)
        del source, prepared
    audit = load_semantic_audit()
    a, b, c = [tok.node_id(i) for i in range(3)]
    def rec(stack, vis, cs, action):
        return stack + [tok.id('|')] + vis + [tok.id('|')] + cs + [tok.id('|')] + action + [tok.id(';')]
    good = rec([a], [a], [b], [tok.id('<PUSH>'), b]) + rec([a,b], [a,b], [], [tok.id('<FOUND>')])
    bad = rec([a], [a], [c], [tok.id('<MISTAKE>')])
    assert audit(good, 1, {a:[b]}, a, b, tok, False)['literal_complete']
    assert audit(bad + good, 1, {a:[b]}, a, b, tok, False)['effective_complete']
    illegal = rec([a], [a], [c], [tok.id('<PUSH>'), c])
    assert not audit(illegal, 1, {a:[b]}, a, b, tok, False)['execution_complete']
    # Counterexample: alternate valid neighbor order must pass semantic replay.
    alt = rec([a], [a], [c,b], [tok.id('<PUSH>'), c]) + rec([a,c], [a,c], [], [tok.id('<FOUND>')])
    assert audit(alt, 1, {a:[b,c]}, a, c, tok, False)['literal_complete']
    report['semantic_oracle_checks'] = 4
    # Exercise the real trainer _free wrapper with scripted generations on CPU;
    # this checks the metrics integration without any optimizer/model forward.
    from worker import install
    install('correction')
    import cot_navigation.training.trainer as training
    from types import SimpleNamespace
    prompt = [tok.id('<EDGE>'), a, b, tok.id('<QUERY>'), a, b, tok.id('|')]
    rows = [SimpleNamespace(tokens=prompt + good, prompt_len=len(prompt),
        answer_index=len(prompt)+len(good), label=1, filler_tokens=0,
        meta={'index':i}) for i in range(2)]
    original_generate = training.generate
    def scripted_generate(*args, **kwargs):
        return [dict(pred=1,format_fail=False,generation_limit_reached=False,
                     trace_tokens=trace,tokens=prompt+trace) for trace in [good,bad+good]]
    training.generate = scripted_generate
    try:
        trainer = training.Trainer.__new__(training.Trainer)
        trainer.cfg=cfg; trainer.tok=tok; trainer.persist=False
        trainer.device=torch.device('cpu'); trainer.raw=object()
        result = trainer._free(rows)
        assert result['semantic']['n'] == 2
        assert result['semantic']['trace_valid_rate'] == 1.
        assert result['semantic']['repair_rate'] == .5
        assert result['overall']['answer_accuracy'] == 1.
        report['scripted_cpu_metrics_integration'] = True
    finally:
        training.generate = original_generate
    from cot_navigation.tracking.logger import _has_wandb_credentials
    import wandb
    report['wandb_credentials_available'] = bool(_has_wandb_credentials(wandb))
    assert report['wandb_credentials_available'], 'online tracking requested but credentials unavailable'
    if previous is not None:
        assert previous == report, 'reviewed data/environment audit changed; prepare for review again'
    else:
        audit_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
