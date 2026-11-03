"""Layer-5 to final-layer SEMANTIC legality transitions at a fixed position.

Use the same frozen early/correct calibrated lens. Only the consumed prefix
must be legal; positions predicting the first natural error remain eligible.
No natural-next-token equality is used as the legality label.
"""
import gc
import json
import os
from pathlib import Path
import runpy
import time
from collections import defaultdict

P=Path(__file__).resolve().parent
E=runpy.run_path(str(P/'early_clean_lens.py'))
R=E['R'];ROOT=R['ROOT'];np=R['np'];torch=R['torch']
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.analysis.within_token import lens_logits
from cot_navigation.experiments.artifacts import atomic_json,sha_file


def next_legal(prefix,token,graph,src,dst,tok):
    """Semantic prefix extension, including incomplete action punctuation.

    replay censors an unfinished empty action, so explicitly reject a fourth
    field separator even before any action token has been emitted.
    """
    if token==tok.id('|'):
        begin=len(prefix)-1
        while begin>=0 and prefix[begin]!=tok.id(';'):begin-=1
        if prefix[begin+1:].count(tok.id('|'))>=3:return False
    return replay(prefix+[int(token)],None,graph,src,dst,tok)['first'] is None


def counts(a,b):
    a=np.asarray(a,bool);b=np.asarray(b,bool)
    ll=int((a&b).sum());il=int((~a&b).sum());li=int((a&~b).sum());ii=int((~a&~b).sum())
    n=len(a)
    return dict(n=n,legal_legal=ll,illegal_legal=il,legal_illegal=li,illegal_illegal=ii,
        illegal_to_legal_fraction=il/max(1,n),legal_to_illegal_fraction=li/max(1,n),
        final_legal_given_l5_illegal=il/(il+ii) if il+ii else None,
        final_illegal_given_l5_legal=li/(li+ll) if li+ll else None,
        l5_legal_fraction=(ll+li)/max(1,n),final_legal_fraction=(ll+il)/max(1,n))


@torch.no_grad()
def predictions(am,lens,feature,layer=4):
    out={k:[] for k in ['raw','calibrated','final']}
    for start in range(0,len(feature['x']),256):
        h=torch.as_tensor(feature['x'][start:start+256,:am.model.n_layers,:am.model.d_model],device=am.device)
        for kind in ['raw','calibrated']:
            out[kind].extend(lens_logits(am.model,lens,h[:,layer],layer,kind).argmax(-1).cpu().tolist())
        out['final'].extend(lens_logits(am.model,None,h[:,-1],am.model.n_layers-1,'raw').argmax(-1).cpu().tolist())
    return out


def analyse(am,cases,rows,pred):
    graphs={};oracle={};labels={k:[] for k in pred};eligible=[];excluded=defaultdict(int)
    for i,r in enumerate(rows):
        c=cases[r['sid']];p=r['pos'];tau=c['first']['token']-1 if c['first'] else len(c['trace'])
        if p>=tau:excluded['already_erroneous_prefix']+=1;continue
        if p+1>=len(c['trace']):excluded['no_observed_next_token']+=1;continue
        eligible.append(i);q=c['query']
        if q not in graphs:graphs[q]=R['R']['graph_tokens'](am,q)
        graph,src,dst=graphs[q];prefix=c['trace'][:p+1]
        def legal(token):
            key=(r['sid'],p,int(token))
            if key not in oracle:
                oracle[key]=next_legal(prefix,int(token),graph,src,dst,am.tok)
            return oracle[key]
        # Independent onset consistency check against the cached natural audit.
        natural_legal=legal(c['trace'][p+1])
        assert natural_legal==(p+1<tau),(c['query'],p,tau)
        for kind in pred:labels[kind].append(legal(pred[kind][i]))
    report=[]
    for population in ['all_legal_prefix','early_legal_prefix','later_legal_prefix']:
        for task in ['all_unique_positions','field_node','field_bar','state','state_pre','candidate_node_position','candidate_prediction']:
            ix=[]
            for j,i in enumerate(eligible):
                r=rows[i]
                if population=='early_legal_prefix' and r['record']>=4:continue
                if population=='later_legal_prefix' and r['record']<4:continue
                if task=='all_unique_positions':
                    # Node samples and the three boundaries are disjoint;
                    # state/state_pre repeat these same boundaries.
                    if r['task'] not in ['field_node','field_bar']:continue
                elif task=='candidate_prediction':
                    if not ((r['task']=='field_node' and r['field']=='candidates') or
                            (r['task']=='field_bar' and r['y']==1)):continue
                elif task=='candidate_node_position':
                    if r['task']!='field_node' or r['field']!='candidates':continue
                elif r['task']!=task:continue
                ix.append(j)
            if not ix:continue
            for kind in ['raw','calibrated']:
                a=np.asarray(labels[kind])[ix];b=np.asarray(labels['final'])[ix]
                report.append(dict(population=population,task=task,kind=kind,
                    queries=len({cases[rows[eligible[j]]['sid']]['query'] for j in ix}),**counts(a,b)))
    return dict(transitions=report,excluded=dict(excluded),eligible_anchor_rows=len(eligible),oracle_cache=len(oracle))


def report(out,manifest):
    dd=[json.loads((out/(m['id']+'.json')).read_text()) for m in manifest['models'] if (out/(m['id']+'.json')).exists()]
    dd=[d for d in dd if d.get('oracle_version')==2]
    summary=[];lines=['# Layer5 → final semantic prefix legality',f'Completed {len(dd)}/8 models.',
        'The consumed prefix must be legal. First-error PREDICTION positions are retained. Whole-record correctness is NOT a test filter.',
        'Legality uses shared semantic replay on prefix + proposed token; noncanonical legal visited/candidate order is accepted.',
        'Previously erroneous prefixes are excluded, not labelled illegal in both layers. This is next-token prefix legality, not whole-rollout legality.',
        'Counts sum all four seeds; positions and queries across seeds are repeated measurements. Calibration is unchanged.',
        '| group | population | task | n | illegal→legal | legal→illegal | both legal | both illegal | illegal→legal / all | legal→illegal / all | repair / L5 illegal | damage / L5 legal |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for group in ['l6_p0','l6_p30']:
        groupdata=[d for d in dd if d['model']['group']==group]
        keys=sorted({(r['population'],r['task'],r['kind']) for d in groupdata for r in d['transitions']})
        for population,task,kind in keys:
            rr=[r for d in groupdata for r in d['transitions'] if (r['population'],r['task'],r['kind'])==(population,task,kind)]
            total={k:sum(r[k] for r in rr) for k in ['n','legal_legal','illegal_legal','legal_illegal','illegal_illegal']}
            n=total['n'];il=total['illegal_legal'];li=total['legal_illegal'];ll=total['legal_legal'];ii=total['illegal_illegal']
            item=dict(group=group,population=population,task=task,kind=kind,models=len(rr),**total,
                illegal_to_legal_fraction=il/n,legal_to_illegal_fraction=li/n,
                final_legal_given_l5_illegal=il/(il+ii) if il+ii else None,
                final_illegal_given_l5_legal=li/(li+ll) if li+ll else None,
                per_seed=[dict(seed=d['model']['seed'],**r) for d in groupdata for r in d['transitions']
                    if (r['population'],r['task'],r['kind'])==(population,task,kind)])
            summary.append(item)
            if kind=='calibrated' and task in ['all_unique_positions','field_node','candidate_prediction']:
                fmt=lambda v:'NA' if v is None else f'{100*v:.2f}%'
                lines.append('| '+' | '.join([group,population,task,str(n),str(il),str(li),str(ll),str(ii)]+[
                    fmt(item[k]) for k in ['illegal_to_legal_fraction','legal_to_illegal_fraction','final_legal_given_l5_illegal','final_illegal_given_l5_legal']])+' |')
    atomic_json(out/'summary.json',dict(complete=len(dd)==8,models=len(dd),results=summary))
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    os.chdir(ROOT);torch.set_num_threads(2)
    base=ROOT/'outputs/analysis/state_readouts/legacy6_s0123_supplement_20261102'
    previous=base/'early_clean_lens';out=base/'legality_transitions';out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((base/'plan.json').read_text());manifest=json.loads((ROOT/plan['args']['manifest']).read_text())
    atomic_json(out/'plan.json',dict(schema='prefix_legality_transition_v2',base_plan_sha256=sha_file(base/'plan.json'),
        calibration_plan_sha256=sha_file(previous/'plan.json'),layer=5,final_layer=6,
        test_filter='consumed prefix before first semantic error; include first-error prediction',oracle='correction_replay.replay'))
    for m in manifest['models']:
        dest=out/(m['id']+'.json')
        if dest.exists() and json.loads(dest.read_text()).get('oracle_version')==2:continue
        started=time.time();print('MODEL',m['id'],flush=True)
        am=R['R']['Model'](m['training_arm'],m['seed']);old=json.loads((previous/(m['id']+'.json')).read_text())
        assert am.checkpoint==old['checkpoint']
        cases,sources=R['load_cases'](m['training_arm'],m['seed'],plan['args']['train_queries'],plan['args']['val_queries'])
        assert sources==old['sources'];selected=R['select'](am,cases)
        lens,support=E['fit_calibration'](am,cases,selected,plan['args']['steps']);assert support==old['support']
        feature=R['extract'](am,cases['test'],selected['test'])
        full=R['lens_scores'](am,lens,feature,selected['test'])
        for a,b in zip(full,old['logit_lens']):assert abs(a['final_agreement']-b['final_agreement'])<1e-7
        result=analyse(am,cases['test'],selected['test'],predictions(am,lens,feature))
        atomic_json(dest,dict(model=m,checkpoint=am.checkpoint,sources=sources,complete=True,oracle_version=2,elapsed=time.time()-started,**result))
        report(out,manifest);print('DONE',m['id'],round(time.time()-started),flush=True)
        del am,lens,feature;gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'status.json',dict(complete=True,models=len(manifest['models'])))


if __name__=='__main__':main()
