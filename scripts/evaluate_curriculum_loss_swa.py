#!/usr/bin/env python3
"""Summarize native best/last predictions; optionally replay semantic traces.

Answer accuracy is split by true label, scale and exact layer gap. Optional
trace replay reuses the previous validated graph oracle and shared TEST panel;
it never selects a checkpoint using test performance.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',default='curriculum_dfs_loss_swa_s0')
    p.add_argument('--checkpoint',choices=['best','last'],default='best')
    p.add_argument('--audit-traces',action='store_true')
    p.add_argument('--per-cell',type=int,default=8)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    from cot_navigation.config import load_config
    from train_curriculum_loss_swa import ARMS
    generated=ROOT/'configs/generated/curriculum_loss_swa'/args.run_name
    manifest=json.loads((generated/'manifest.json').read_text())
    entries=[]
    for job in manifest['jobs']:
        cfg=load_config([job['config']]);run=Path(cfg.output_dir)
        cases=['main']+[s.split('=',1)[0] for s in cfg.evaluation.extra_sets]
        entries.append((job,cfg,run,cases))
        if args.dry_run:print(job['name'],args.checkpoint,cases)
    if args.dry_run:return
    # Refuse incomplete comparisons instead of silently reporting only finished arms.
    for job,cfg,run,cases in entries:
        for case in cases:
            path=run/'eval'/f"predictions_test_{args.checkpoint}{'' if case=='main' else '-'+case}.jsonl"
            if not path.exists():raise FileNotFoundError(path)
    out=ROOT/'studies/curriculum_loss_swa/results'/args.run_name/args.checkpoint
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for job,cfg,run,cases in entries:
        arm=job['name'].split('/')[-1]
        for case in cases:
            path=run/'eval'/f"predictions_test_{args.checkpoint}{'' if case=='main' else '-'+case}.jsonl"
            data=[json.loads(s) for s in path.read_text().splitlines()]
            for gap in ['all']+sorted({r['distance'] for r in data}):
                for label in [0,1]:
                    subset=[r for r in data if r['label']==label and (gap=='all' or r['distance']==gap)]
                    if not subset:continue
                    correct=sum(bool(r['correct']) for r in subset)
                    rows.append({'arm':arm,'case':case,'gap':gap,'label':label,'n':len(subset),'correct':correct,'accuracy':correct/len(subset)})
    with (out/'answer_accuracy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    if args.audit_traces:
        if args.checkpoint!='best':raise ValueError('Semantic replay currently targets validation-best only')
        sys.path.insert(0,str(ROOT/'studies/curriculum_trace_errors'))
        import audit
        audit.OUT=out/'trace_audit'
        for job,cfg,run,cases in entries:
            checkpoint=run/'checkpoints/best.pt'
            entry={'id':job['name'].replace('/','_'),'run':str(run),'kind':'context','protocol':'dfs_state',
                   'scale':'curriculum','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            audit.run(entry,args.per_cell,args.batch)
        metrics=[]
        for path in sorted((audit.OUT/'models').glob('*/*.json')):
            m=json.loads(path.read_text())
            with gzip.open(path.with_suffix('.jsonl.gz'),'rt') as f:data=[json.loads(s) for s in f]
            for label in [0,1]:
                rr=[r for r in data if r['label']==label]
                metrics.append({'model':m['model']['id'],'case':m['case'],'label':label,'n':len(rr),
                    'answer_correct':sum(r['correct'] for r in rr),'complete_trace_valid':sum(r['complete_valid'] for r in rr),
                    'action_skeleton_valid':sum(r['action_valid'] for r in rr),'stop':sum(r['stop'] for r in rr)})
        with (out/'trace_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,list(metrics[0]));w.writeheader();w.writerows(metrics)
    print('Saved',out)

if __name__=='__main__':main()
