"""Frozen checkpoint comparison; no Trainer, W&B, run records or checkpoints."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ContextStore
from cot_navigation.evaluation.sampling import stratified_ids
from cot_navigation.evaluation.rollout import generate, pad_rows

torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('cot_navigation.models.decode_reference',
    ROOT / 'studies/decode_shape_fix/before/src/cot_navigation/models/dual_attention.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
reports = []
for protocol, order, graph in [('dfs_state', 'parallel', 'causal'), ('bfs_state', 'serial', 'causal')]:
    # Each production worker hosts one architecture; isolate benchmark cases.
    torch._dynamo.reset()
    run = ROOT / 'outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023_r1' / protocol / order / graph
    cfg = load_config([run / 'resolved_config.yaml'])
    tok = Tokenizer.from_config(cfg)
    ck = torch.load(run / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    models = [reference.DualAttentionLM(cfg.model, len(tok), tok.pad).cuda().eval(),
              build_model(cfg.model, len(tok), tok.pad).cuda().eval()]
    for m in models:
        m.load_state_dict(ck['model'])
    del ck
    store = ContextStore(cfg.data.root)
    a = store.arrays
    ids = stratified_ids(store.indices('val'), a['shape_id'], a['distance'], a['label'], 8, cfg.evaluation.seed)
    rows = [store.render(i, protocol, tok) for i in ids]
    prompts = [r.tokens[:r.prompt_len] for r in rows]
    tokens, lengths = pad_rows(prompts, tok.pad, torch.device('cuda'))
    # Equal forced token work checks numerical agreement independently of
    # greedy branch choices, and compares steady-state per-step latency.
    latencies, outputs = [], []
    for m in models:
        cache = m.new_cache(len(rows), 'cuda', torch.float32)
        with torch.no_grad():
            m(tokens, cache=cache, cache_mode='prefill')
            cache.lengths.copy_(lengths)
            step_outputs = []
            for i in range(72):
                cache.valid_length = tokens.shape[1] + i + 1
                token = torch.full((len(rows), 1), tok.node_id(0), device='cuda', dtype=torch.long)
                step_outputs.append(m(token, positions=cache.lengths[:, None], cache=cache, cache_mode='decode'))
                cache.lengths += 1
            outputs.append(torch.cat(step_outputs, dim=1))
        del cache
    fp32_max_diff = (outputs[0] - outputs[1]).abs().max().item()
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-5, rtol=2e-4)
    del outputs
    wrapped = CompiledInference(models[1])
    generate(wrapped, prompts[:2], protocol, tok, 'bf16', max_new_tokens=16, cot_margin=cfg.protocol.cot_margin)
    timed = []
    for label, m in [('old_eager', models[0]), ('new_compiled_cold', wrapped), ('new_compiled_warm', wrapped)]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = generate(m, prompts, protocol, tok, 'bf16', max_new_tokens=0,
                          cot_margin=cfg.protocol.cot_margin)
        torch.cuda.synchronize()
        timed.append(dict(mode=label, seconds=time.perf_counter() - start, outputs=result))
        print(json.dumps(dict(protocol=protocol, mode=label, seconds=timed[-1]['seconds'],
                              generated_tokens=sum(len(g['generated']) for g in result))), flush=True)
    old, cold, new = [x['outputs'] for x in timed]
    report = dict(protocol=protocol, queries=len(rows), fp32_max_diff=fp32_max_diff,
        old_seconds=timed[0]['seconds'], new_cold_seconds=timed[1]['seconds'], new_seconds=timed[2]['seconds'],
        old_tokens=sum(len(g['generated']) for g in old), new_tokens=sum(len(g['generated']) for g in new),
        same_predictions=sum(x['pred'] == y['pred'] for x, y in zip(old, new)),
        same_traces=sum(x['trace_tokens'] == y['trace_tokens'] for x, y in zip(old, new)),
        old_correct=sum(g['pred'] == r.label and not g['format_fail'] for g, r in zip(old, rows)),
        new_correct=sum(g['pred'] == r.label and not g['format_fail'] for g, r in zip(new, rows)),
        fallbacks=sorted(wrapped.compile_fallbacks), max_new_tokens=0,
        concurrent_training=True)
    assert not wrapped.compile_fallbacks
    reports.append(report)
    print(json.dumps(report), flush=True)
    (ROOT / 'studies/decode_shape_fix/benchmark.json').write_text(json.dumps(reports, indent=2) + '\n')
    del wrapped, models, store
    torch.cuda.empty_cache()
