"""Compare identical real prompts; no training or checkpoint/log creation."""
import json
import time
from pathlib import Path

import torch
from cot_navigation.config import load_config
from checkpoint_evaluator import CheckpointEvaluator
from cot_navigation.evaluation.rollout import generate

torch.set_num_threads(2)
root = Path('outputs/hard_dag_v1/launches/hard_dag_v1_s0')
results = []
for task, path in [('context', root/'context/curriculum/bfs_state'),
                   ('fixed', root/'fixed/n100/stage_b/bfs_state')]:
    cfg = load_config([str(path/'resolved_config.yaml')])
    ev = CheckpointEvaluator(cfg, path/'checkpoints/best.pt')
    ids = ev._selected_ids('test', 16)
    rows = [ev._render_task(i) for i in ids]
    prompts = [r.tokens[:r.prompt_len] for r in rows]
    # Warm-up kernels outside the timed comparison.
    generate(ev.raw, prompts[:1], cfg.protocol.name, ev.tok, 'bf16', max_new_tokens=8)
    outputs, timings = {}, {}
    for name, crop, compact in [('reference',False,False), ('cropped',True,False), ('optimized',True,True)]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs[name] = generate(ev.raw, prompts, cfg.protocol.name, ev.tok, 'bf16',
                                 max_new_tokens=1024, cot_margin=cfg.protocol.cot_margin,
                                 crop_cache=crop, compact_finished=compact)
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - start
        print(json.dumps(dict(task=task, mode=name, seconds=timings[name],
                              lengths=[len(x['generated']) for x in outputs[name]])), flush=True)
    report = dict(task=task, batch=16, max_new_tokens=1024, seconds=timings,
                  speedup=timings['reference']/timings['optimized'],
                  exact_rows={k:sum(a==b for a,b in zip(outputs['reference'],v)) for k,v in outputs.items()},
                  answer_agreement={k:sum(a['pred']==b['pred'] for a,b in zip(outputs['reference'],v)) for k,v in outputs.items()})
    results.append(report)
    print(json.dumps(report), flush=True)
    del ev
Path('studies/eval_speedup/benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
