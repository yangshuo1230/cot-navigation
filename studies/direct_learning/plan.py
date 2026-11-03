"""Explicit, reviewable 80 GPU-hour matrix. No architecture or CoT changes."""
from copy import deepcopy

GROUP = 'direct_learning_20261020_v1'
SECONDS = 7200


def stage(kind='full', vocab=16, edges=2, shuffle=True, mixture=None):
    return dict(kind=kind, vocab=vocab, edges=edges, shuffle=shuffle, mixture=mixture)


def matrix():
    jobs = []
    def add(name, stages, seed=0, lr=3e-4, schedule='static', packed=False):
        jobs.append(dict(name=name, seed=seed, lr=lr, schedule=schedule, packed=packed,
                         budget_seconds=SECONDS, stages=deepcopy(stages)))
    controls = {
        'A_fixed_graph16': stage('fixed_graph'),
        'B_fixed_query16': stage('fixed_query'),
        'C_fixed_source16': stage('fixed_source'),
        'D_fixed_target16': stage('fixed_target'),
        'E_full16_fixed_order': stage(shuffle=False),
        'F_full16_shuffled': stage(),
        'full64_fixed_order': stage(vocab=64, shuffle=False),
        'full64_shuffled': stage(vocab=64),
        'full256_fixed_order': stage(vocab=256, shuffle=False),
        'full256_shuffled': stage(vocab=256),
        'ordinary_m2_16': stage('ordinary'),
        'ordinary_m24_256': stage('ordinary', vocab=256, edges=24),
        'C_fixed_source256': stage('fixed_source', vocab=256),
        'D_fixed_target256': stage('fixed_target', vocab=256),
        'A_fixed_graph256': stage('fixed_graph', vocab=256),
    }
    for name, spec in controls.items():
        add(name+'_s0', [spec])
    add('full16_packed_s0', [stage()], packed=True)
    add('full16_lr1e3_s0', [stage()], lr=1e-3)
    add('full256_lr1e3_s0', [stage(vocab=256)], lr=1e-3)
    # Same data, independent initialization/sampling: graph-level and model seeds differ.
    for name in ['A_fixed_graph16', 'B_fixed_query16', 'C_fixed_source16',
                 'D_fixed_target16', 'E_full16_fixed_order', 'F_full16_shuffled',
                 'full64_shuffled', 'full256_shuffled', 'ordinary_m2_16', 'ordinary_m24_256']:
        add(name+'_s1', [controls[name]], seed=1)
    tail = [stage(vocab=64), stage(vocab=256)] + [stage(vocab=256, edges=m) for m in (4, 8, 16, 24)]
    for anchor in ['source', 'target']:
        chain = [stage('fixed_'+anchor), stage(shuffle=False), stage()] + tail
        add('curr_'+anchor+'_gate_s0', chain, schedule='gated')
        add('curr_'+anchor+'_gate_s1', chain, seed=1, schedule='gated')
        add('curr_'+anchor+'_timed_s0', chain, schedule='timed')
        mix = [stage('mix_'+anchor, mixture=p) for p in (.8, .5, .2)]
        add('curr_mix_'+anchor+'_gate_s0', mix+[stage()]+tail, schedule='gated')
    add('curr_shuffle_gate_s0', [stage(shuffle=False), stage()]+tail, schedule='gated')
    add('curr_vocab_gate_s0', [stage()]+tail, schedule='gated')
    for seed in (0, 1):
        add('final_m24_paired_s'+str(seed), [stage(vocab=256, edges=24)], seed=seed)
    assert len(jobs) == 40 and sum(j['budget_seconds'] for j in jobs) == 80*3600
    assert len({j['name'] for j in jobs}) == len(jobs)
    return jobs
