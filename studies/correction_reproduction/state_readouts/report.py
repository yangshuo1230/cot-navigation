"""Compact per-seed tables; incomplete cohorts are always labelled."""
import json
from pathlib import Path
import numpy as np
from cot_navigation.experiments.artifacts import atomic_json


def report(out):
    out=Path(out);files=sorted(out.glob('l6_p*_s*.json'))
    data=[json.loads(p.read_text()) for p in files]
    groups={g:[d for d in data if d['model']['group']==g] for g in ['l6_p0','l6_p30']}
    lines=['# Natural field/state readouts and logit lens',
        f'Completed {len(data)}/8 models. Main N100 positive queries only.',
        'Natural windows stop after first error + two records. Train/validation/test graphs and queries are disjoint.',
        'State labels describe emitted frontier/visited/candidates; successors are actual graph neighbours of the emitted current node.',
        'Legal candidates = actual neighbours minus emitted visited. These are not an oracle-correct DFS state after prior errors.',
        'State anchor: third bar, before action. successors_pre uses first bar, before current visited/candidate fields.',
        'Set decisions: probability >= 0.5, evaluated over graph nodes. Exact-set accuracy, F1 and false-positive counts are retained.',
        'Set labels are sparse: compare ordinary accuracy with the all-negative baseline as well as balanced accuracy.',
        'Each receiver uses its own natural test text on shared held-out queries; these are not identical-prefix representation comparisons.',
        'Field probes: nodes only (4 classes), or identical | tokens (3 classes). Each layer has independent parameters.',
        'Heads 1–6 = model layers; 7 = token embedding; 8 = token embedding + absolute/relative position.',
        'Calibration: affine lens distilled from final logits; no graph-state labels. Probe decodability alone is not causal evidence.',
        '\n## Probe test performance',
        '| group | seeds | task | probe | metric | L1 | L2 | L3 | L4 | L5 | L6 | token | token+pos |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    summary=[]
    for group,dd in groups.items():
        keys=sorted({(p['task'],p['architecture']) for d in dd for p in d['probes']})
        for task,arch in keys:
            pp=[next(p for p in d['probes'] if (p['task'],p['architecture'])==(task,arch)) for d in dd]
            for p in pp:
                for h in p['test']:
                    if 'tn' in h:
                        h['all_negative_accuracy']=(h['tn']+h['fp'])/max(1,h['tp']+h['tn']+h['fp']+h['fn'])
            metrics=['accuracy','balanced_accuracy'] if task.startswith('field') else ['accuracy','all_negative_accuracy','balanced_accuracy','exact_set_accuracy','f1']
            for metric in metrics:
                values=np.array([[h[metric] for h in p['test']] for p in pp])
                row=dict(group=group,task=task,architecture=arch,metric=metric,seeds=[d['model']['seed'] for d in dd],
                    mean=values.mean(0).tolist(),std=values.std(0).tolist())
                summary.append(row)
                cells=[f'{100*m:.1f}±{100*s:.1f}' for m,s in zip(values.mean(0),values.std(0))]
                lines.append('| '+' | '.join([group,str(len(dd)),task,arch,metric]+cells)+' |')
    lines+=['\n## Logit lens: actual next-token accuracy / agreement with final layer',
        'Natural tokens were greedily sampled from this model. Next-token accuracy therefore duplicates final-layer agreement on these prefixes; neither is correctness against a semantic oracle.',
        '| group | seeds | position | lens | metric | L1 | L2 | L3 | L4 | L5 | L6 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    lens_summary=[]
    for group,dd in groups.items():
        keys=sorted({(p['task'],p['kind']) for d in dd for p in d['logit_lens']})
        for task,kind in keys:
            for metric in ['accuracy','final_agreement']:
                values=np.array([[p[metric] for p in d['logit_lens'] if p['task']==task and p['kind']==kind] for d in dd])
                lens_summary.append(dict(group=group,task=task,kind=kind,metric=metric,mean=values.mean(0).tolist(),std=values.std(0).tolist()))
                lines.append('| '+' | '.join([group,str(len(dd)),task,kind,metric]+[f'{100*m:.1f}±{100*s:.1f}' for m,s in zip(values.mean(0),values.std(0))])+' |')
    atomic_json(out/'summary.json',dict(complete=len(data)==8,models=len(data),probes=summary,logit_lens=lens_summary))
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();report(a.output)
