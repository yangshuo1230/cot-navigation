"""Compact report for native trace replay and certified legal history pairs."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    results = root / 'results'
    data = [json.loads(p.read_text()) for p in sorted((results/'models').glob('*.json'))]
    expected = json.loads((results/'selection.json').read_text())['models']
    assert len(data) == len(expected) == 60
    assert all(d.get('complete') and d['checkpoint_unchanged'] for d in data)
    rows = []
    for d in data:
        e, a = d['entry'], d['audit']
        n = a['counts']['n']
        rows.append(dict(model=e['id'], kind=e['kind'], protocol=e['protocol'], n=n,
            accuracy=a['accuracy'], valid=a['complete_valid_rate'],
            correct_and_valid=(a['counts']['answer_correct']-a['counts']['answer_correct_invalid_trace'])/n,
            correct_invalid=a['correct_answer_invalid_trace_rate'],
            nonedge_push=a['counts']['nonedge_push_queries']/n,
            frontier_jump=a['counts']['frontier_jump_queries']/n,
            visited_update=a['counts']['visited_update_error_queries']/n,
            stopped=a['counts']['stop']/n))
    with (results/'audit_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pair_rows=[]
    for d in data:
        for task,p in d['pairs'].items():
            if not isinstance(p,dict) or not p.get('n'): continue
            for match in sorted({r['match'] for r in p['by_query']}):
                obs=[r for r in p['by_query'] if r['match']==match]
                pair_rows.append(dict(model=d['entry']['id'],task=task,match=match,n=len(obs),
                    equal_length=sum(r['equal_length'] for r in obs),
                    candidate_a=np.mean([r['a']['candidate_set_correct'] for r in obs]),
                    candidate_b=np.mean([r['b']['candidate_set_correct'] for r in obs]),
                    action_a=np.mean([r['a']['action_legal'] for r in obs]),
                    action_b=np.mean([r['b']['action_legal'] for r in obs]),
                    focus_eligible=np.mean([r['a']['focus_listed'] is True for r in obs]) if task=='changed' else '',
                    focus_visited=np.mean([r['b']['focus_listed'] is True for r in obs]) if task=='changed' else ''))
    with (results/'pair_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
    protocols=['direct','path','dfs_state','bfs_state','dfs_nodes','bfs_nodes']
    scales=['l4','l6','l8','l10','curriculum']
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for i,kind in enumerate(['context','fixed']):
        for j,metric in enumerate(['accuracy','correct_and_valid']):
            ax=axes[i,j]
            values=np.array([[next(r[metric] for r in rows if r['model']==f'{kind}_{s}_{p}') for s in scales] for p in protocols])
            ax.imshow(values,vmin=0,vmax=1,cmap='viridis')
            for y in range(6):
                for x in range(5):ax.text(x,y,f'{values[y,x]:.0%}',ha='center',va='center',color='white' if values[y,x]<.6 else 'black',fontsize=9)
            ax.set_xticks(range(5),['N16','N36','N64','N100','Curr.\nN100'])
            ax.set_yticks(range(6),protocols)
            ax.set_title(f'{kind}: '+('answer accuracy' if j==0 else 'correct answer + complete valid trace'))
    fig.savefig(results/'trace_replay.png',dpi=170);plt.close(fig)
    states=[d for d in data if d['entry']['protocol'].endswith('_state')]
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for i,kind in enumerate(['context','fixed']):
        for j,protocol in enumerate(['dfs_state','bfs_state']):
            ax=axes[i,j]
            for arm,label,style in [('a','focus included when eligible','-o'),('b','focus included when visited','--s')]:
                vals=[]
                for s in scales:
                    d=next(d for d in states if d['entry']['id']==f'{kind}_{s}_{protocol}')
                    p=d['pairs']['changed']
                    vals.append(p.get('focus_listed_when_eligible' if arm=='a' else 'focus_listed_when_visited',np.nan))
                ax.plot(range(5),vals,style,label=label)
            ax.set_xticks(range(5),['N16','N36','N64','N100','Curr.'])
            ax.set_ylim(-.04,1.04);ax.grid(alpha=.2)
            ax.set_title(f'{kind} {protocol}'+(' (queue also changes)' if protocol=='bfs_state' else ' (same frontier)'))
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.savefig(results/'legal_history_pairs.png',dpi=170);plt.close(fig)
    lines=['# Native trace replay and legal history pairs','',
        '60 validation-selected scale checkpoints; same test query IDs as the previous analysis. '
        'Each static scale and curriculum model is reported separately. No new probe fitting or checkpoint edits.', '',
        '## Full replay','',
        '| Model | n | Answer | Answer + valid full trace | Correct answer / invalid trace | Non-edge PUSH query | State jump query | STOP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['n']} | {r['accuracy']:.1%} | {r['correct_and_valid']:.1%} | {r['correct_invalid']:.1%} | {r['nonedge_push']:.1%} | {r['frontier_jump']:.1%} | {r['stopped']:.1%} |")
    lines += ['', 'Semantic replay stops at the first error. Non-edge PUSH and state-jump columns are local checks across printed records, including records after the first semantic error; these are not independently valid execution histories. '
        'STOP traces count as incomplete, not automatically as algorithm errors. Node-only traces are expansion orders, not edge-by-edge paths. Direct has no intermediate trace.', '',
        '## Legal history pairs','',
        '| Model | Pair type | Match | n | Same length | Candidate set A/B | Legal action A/B | Focus included eligible/visited |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in pair_rows:
        focus=f"{r['focus_eligible']:.1%}/{r['focus_visited']:.1%}" if r['task']=='changed' else '—'
        lines.append(f"| {r['model']} | {r['task']} | {r['match']} | {r['n']} | {r['equal_length']} | {r['candidate_a']:.1%}/{r['candidate_b']:.1%} | {r['action_a']:.1%}/{r['action_b']:.1%} | {focus} |")
    lines += ['',
        'Both prefixes are independently replay-certified legal histories of the same graph/query, generated from 64 seeded neighbor permutations. '
        'Each query contributes at most one pair per type (maximum 40). Selection never uses model outputs. '
        'Prefixes end after the VIS separator, before CAND. We score the next-token legal probability mass and free-generate up to 32 tokens to inspect the current candidate/action record. '
        'This 32-token local diagnostic is not a final-answer test.', '',
        'DFS changed pairs have the identical full frontier and current node. A focus successor is unvisited in A and visited in B. '
        'Prefix length and VIS count are matched where possible, not universally. Changes include the entire legal history and VIS, so this does not isolate a single internal representation or a single token.', '',
        'On this layered DAG family, BFS changed pairs require relaxed queue matching: the current node, queue length and VIS count match, but queue contents can differ. '
        'These demonstrate response to a valid state/history change and cannot attribute the effect to VIS alone. Same-state controls preserve frontier and VIS set across histories. '
        'Candidate filtering success is evidence of local state use, not proof of complete search or planning.', '',
        'Small fixed graphs have fewer held-out queries (N16: 19; N36: 114); other cells use 128. '
        'Results are one graph seed per scale and must not be treated as replicated graph-level confidence estimates. '
        'No Filler checkpoint is present in this scale selection.', '',
        'Raw per-query counts, first errors, limited decoded examples, exact checkpoints/hashes, and all paired measurements are in results/models/*.json. '
        'Only two key figures are produced; no activation dumps or probe weights.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Reported',len(rows),'models and',len(pair_rows),'pair groups')


if __name__=='__main__':main()
