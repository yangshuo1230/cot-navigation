"""Compact structural evidence; missing links are not labelled independent."""
from collections import Counter
import gzip
import json
from pathlib import Path

P=Path(__file__).resolve().parent
TASKS=['n32','n64','main','deep','hard'];ARMS=['matched_clean','correction']


def rate(n,d):return n/d if d else None


def main():
    length_file=P.parent/'length_analysis.json'
    length=json.loads(length_file.read_text()) if length_file.exists() else {}
    native_file=P/'native_window_summary.json'
    out=dict(complete=False,total_tasks=20,completed_tasks=0,tasks={},length=length,
        native_window=json.loads(native_file.read_text()) if native_file.exists() else None)
    for task in TASKS:
        blobs={}
        for arm in ARMS:
            for seed in [0,1]:
                f=P/'results'/f'{arm}_s{seed}_{task}.json.gz'
                if not f.exists():continue
                with gzip.open(f,'rt') as h:b=json.load(h)
                if b['complete']:blobs[arm,seed]=b;out['completed_tasks']+=1
        seeds=[s for s in [0,1] if all((arm,s) in blobs for arm in ARMS)]
        out['tasks'][task]=dict(paired_seeds=seeds,groups={})
        for arm in ARMS:
            ts=[r for s in seeds for r in blobs[arm,s]['rows']]
            dd=[tr['dependencies'] for tr in ts]
            revisits=[r for d in dd for r in d['revisits']]
            extras=[r for r in revisits if r['original_unused_extras']]
            missing=[r for r in revisits if r['missing_still_needed']]
            clean=[r for r in revisits if r['previously_clean']]
            transitions=[r for d in dd for r in d['annotation_transitions']]
            local=[]
            for offset in [1,2]:
                rs=[r for d in dd for r in d['after_illegal'] if r['offset']==offset]
                supported=[r for r in rs if r['follows_executed'] is not None]
                local.append(dict(offset=offset,observed_next_records=len(rs),supported=len(supported),
                    **{field:sum(r['follows_executed'][field] for r in supported) for field in ['frontier','visited','candidate','action','all']},
                    reject_all=sum(r['follows_rejected']['all'] for r in rs if r['follows_rejected'] is not None),
                    printed_supported=sum(r['follows_printed'] is not None for r in rs),
                    printed_all=sum(r['follows_printed']['all'] for r in rs if r['follows_printed'] is not None)))
            following=[]
            for field in ['frontier','visited']:
                for cancel in [False,True]:
                    rs=[r for r in transitions if r['field']==field and r['cancel']==cancel]
                    following.append(dict(field=field,cancel=cancel,n=len(rs),counts=dict(Counter(r['following'] for r in rs))))
            revisit=dict(total_pairs=len(revisits),unused_extra_anchors=sum(d['unused_extra_anchors'] for d in dd),
                unused_extra_without_observed_return=sum(d['unused_extra_without_observed_return'] for d in dd),
                unused_extra_return_pairs=len(extras),same_extra_repeated=sum(bool(r['repeated_unused_extras']) for r in extras),
                same_extra_repeat_rate=rate(sum(bool(r['repeated_unused_extras']) for r in extras),len(extras)),
                bad_again=sum(r['candidate_bad'] for r in extras),
                still_missing_pairs=len(missing),same_missing_repeated=sum(bool(r['repeated_missing']) for r in missing),
                initially_clean_return_pairs=len(clean),clean_then_bad=sum(r['candidate_bad'] for r in clean))
            links=[r for d in dd for r in d['candidate_action_links']]
            out['tasks'][task]['groups'][arm]=dict(trajectories=len(ts),local=local,following=following,revisit=revisit,
                illegal_push=sum(r['chosen_illegal'] for r in links),
                illegal_push_from_candidate=sum(r['chosen_illegal'] and r['chosen_in_printed_candidates'] for r in links))
    out['complete']=out['completed_tasks']==20
    (P/'summary.json').write_text(json.dumps(out,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    template=(P/'dashboard_template.html').read_text(encoding='utf-8')
    (P/'dependency_dashboard.html').write_text(template.replace('__DATA__',json.dumps(out,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    print('DEPENDENCIES',out['completed_tasks'],20,flush=True)


if __name__=='__main__':main()
