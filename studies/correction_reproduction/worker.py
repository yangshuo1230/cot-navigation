"""Isolated training entry point for the reviewed DFS reproduction suite.

No shared trainer/data files are changed. The matched-clean arm samples the
same observed-error episode, then returns its original canonical trajectory.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
from cot_navigation.data.natural_mistakes import NaturalCorrectionPool
from cot_navigation.data.scale_prepared import ScalePreparedRows
from trace_weight import combine_length_losses


class MatchedCleanPool(NaturalCorrectionPool):
    """Keep error-pool sampling/RNG identical; remove only the repair episode."""
    def __init__(self, source, cfg, tok):
        super().__init__(source, cfg, tok)
        prepared = ScalePreparedRows(source, cfg, tok)
        # Resolve once at startup; sampling remains O(1), with no DFS/disk I/O.
        self.clean_rows = {id(row): prepared.row(int(row.meta['index']),
                          int(row.meta['natural_variant'])) for row in self.train_rows}
        for row in self.train_rows:
            clean = self.clean_rows[id(row)]
            assert row.tokens[:row.prompt_len] == clean.tokens[:clean.prompt_len]
            assert row.label == clean.label and int(clean.meta['split']) == 0
            assert tok.id('<MISTAKE>') not in clean.tokens

    def sample(self, label, rng, complex_share):
        observed = super().sample(label, rng, complex_share)
        return self.clean_rows[id(observed)]


def weight_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_semantic_audit():
    # Reuse the previously validated pure replayer without executing that
    # research script's environment changes (notably WANDB_MODE=disabled).
    path = ROOT / 'studies/accuracy_source/run.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['audit']


def install(arm):
    import cot_navigation.data.natural_mistakes as natural
    import cot_navigation.training.trainer as training
    import cot_navigation.tracking.logger as logger
    if arm == 'matched_clean':
        natural.NaturalCorrectionPool = MatchedCleanPool
    original_snapshot = training._arch_snapshot
    policy = json.loads((Path(__file__).parent / 'trace_weight_policy.json').read_text())

    def snapshot(cfg, tok, manifest_hash):
        return {**original_snapshot(cfg, tok, manifest_hash),
                'reproduction_arm': arm, 'reproduction_version': 3,
                'trace_weight_policy': policy}
    training._arch_snapshot = snapshot
    # Identical small dashboards; semantic validity allows alternative legal
    # neighbor orders and MISTAKE cancellation, unlike exact trace matching.
    logger.visible_metrics = lambda cfg: ('answer_loss', 'trace_loss',
        'free/overall/answer_accuracy', 'free/semantic/trace_valid_rate', 'trace_weight/mean')
    from functools import partial
    training.combine_losses = partial(combine_length_losses,
                                      tokens_per_weight=policy['tokens_per_weight'])
    original_update = training.Trainer._update

    def length_weighted_update(self, collect_metrics=True):
        import torch
        if not hasattr(self, '_length_weight_loss_installed'):
            compiled_loss = self.loss_fn

            def observed_loss(*args, **kwargs):
                parts = compiled_loss(*args, **kwargs)
                if self._collect_length_metrics:
                    take = parts['has_answer']
                    self._length_metric_chunks.append(torch.stack([
                        parts['weighted_answer'].detach().sum(),
                        parts['weighted_trace'].detach().sum(),
                        torch.where(take, parts['trace_weights'], 0.).sum(),
                        torch.where(take, parts['trace_lengths'], 0.).sum(), take.sum()]))
                return parts
            self.loss_fn = observed_loss
            self._length_weight_loss_installed = True
        self._collect_length_metrics = collect_metrics
        self._length_metric_chunks = []
        metrics = original_update(self, collect_metrics=collect_metrics)
        if collect_metrics and self._length_metric_chunks:
            # One tiny transfer per LOGGED update; no per-row GPU synchronizing.
            a, t, w, length, n = torch.stack(self._length_metric_chunks).sum(0).cpu().tolist()
            n = max(1., n)
            metrics.update(weighted_answer_loss=a/n, weighted_trace_loss=t/n,
                           task_loss=(a+t)/n,
                           **{'trace_weight/mean':w/n, 'trace_length/mean':length/n})
            metrics.pop('trace_loss_lambda', None)
        self._length_metric_chunks = []
        return metrics
    training.Trainer._update = length_weighted_update
    audit = load_semantic_audit()
    original_free = training.Trainer._free
    original_extra = training.Trainer._evaluate_extra

    def semantic_free(self, rows, *, record_name=None):
        counts = Counter()
        generate = training.generate

        def observed_generate(model, prompts, protocol, tok, *args, **kwargs):
            outputs = generate(model, prompts, protocol, tok, *args, **kwargs)
            for prompt, g in zip(prompts, outputs):
                assert prompt[-4] == tok.id('<QUERY>') and prompt[-1] == tok.id('|')
                triples = np.asarray(prompt[:-4]).reshape(-1, 3)
                assert all(int(t[0]) == tok.id('<EDGE>') for t in triples)
                adj = {}
                for _, u, v in triples: adj.setdefault(int(u), []).append(int(v))
                a = audit(g['trace_tokens'], g['pred'], adj, prompt[-3], prompt[-2], tok, False)
                counts['n'] += 1
                counts['trace_valid'] += int(a['effective_prefix_valid'] and a['terminal'] in ('FOUND', 'EXHAUST'))
                counts['answer_and_trace_correct'] += int(a['effective_complete'])
                for name, key in [('candidate_error', 'candidate_bad'), ('visited_error', 'visited_bad'),
                                  ('mistake', 'mistake_records'), ('repair', 'repairs'), ('false_mark', 'false_marks')]:
                    counts[name] += int(a[key] > 0)
            return outputs
        # Synchronous, process-local wrapper; never changes logits/decoding.
        training.generate = observed_generate
        try:
            report = original_free(self, rows, record_name=record_name)
        finally:
            training.generate = generate
        report['semantic'] = dict(n=counts['n'], **{k + '_rate': v / max(1, counts['n'])
            for k, v in counts.items() if k != 'n'})
        return report

    def hard_extra(self, tag, step, spec):
        if spec.split('=', 1)[0] != 'hard':
            return original_extra(self, tag, step, spec)
        # Hard-DAG gold rows were prepared at 32K. Use that immutable reference
        # cache, while generation still obeys this model's native 4K limit.
        from cot_navigation.config import load_config
        from cot_navigation.data.dataset import ContextStore
        from cot_navigation.protocols.tokenizer import Tokenizer
        saved = (self.store, self.world, self.prepared_tasks, self.prepared_walks,
                 self._active_member, self.render_cache)
        try:
            self.store = ContextStore(spec.split('=', 1)[1]); self.world = None
            refcfg = load_config([ROOT / 'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml'])
            reference_tok = Tokenizer.from_config(refcfg)
            assert self.tok.itos[:len(reference_tok)] == reference_tok.itos
            self.prepared_tasks = ScalePreparedRows(self.store, refcfg, reference_tok)
            self.prepared_walks = None; self._active_member = 'extra:hard'; self.render_cache = {}
            self.evaluate_split('test', checkpoint_tag=tag + '-hard', checkpoint_step=step)
        finally:
            (self.store, self.world, self.prepared_tasks, self.prepared_walks,
             self._active_member, self.render_cache) = saved
    training.Trainer._free = semantic_free
    training.Trainer._evaluate_extra = hard_extra


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True, choices=['original_clean', 'matched_clean', 'correction'])
    p.add_argument('--config', required=True)
    p.add_argument('--resume')
    p.add_argument('--execute-training', action='store_true')
    args = p.parse_args()
    if not args.execute_training:
        p.error('training is disabled unless --execute-training is explicitly provided')
    from cot_navigation.config import load_config
    install(args.arm)
    from cot_navigation.training.trainer import Trainer
    cfg = load_config([args.config])
    assert cfg.tracking.variant == args.arm, 'arm and reviewed configuration disagree'
    trainer = Trainer(cfg, resume=args.resume)
    try:
        if not args.resume:
            proof = dict(arm=args.arm,model_seed=cfg.train.model_seed,
                         initial_weight_sha256=weight_hash(trainer.raw),from_scratch=True,
                         trace_weight_policy=trainer.snapshot['trace_weight_policy'])
            (Path(cfg.output_dir) / 'reproduction_identity.json').write_text(json.dumps(proof, indent=2))
        trainer.fit()
    finally:
        trainer.tracker.finish()


if __name__ == '__main__':
    main()
