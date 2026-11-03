"""Build a portable, offline HTML dashboard; no web server/CDN/dependencies."""
import argparse
import json
import math
from pathlib import Path

P=Path(__file__).resolve().parent


def accuracy_metrics(value):
    """Recover exact 0.5-threshold accuracy of balanced matched readout pairs.

    Stored TPR/FPR use logit >= 0 (probability >= 0.5). Every evaluated cell
    contains both members of each wrong/legal pair. This affine identity also
    preserves the existing position/probe/model macro averages; no refit or
    test-set threshold tuning is needed. Keep historical source JSON intact.
    """
    if isinstance(value,list):return [accuracy_metrics(v) for v in value]
    if not isinstance(value,dict):return value
    result={k:accuracy_metrics(v) for k,v in value.items()
            if k not in {'auroc','auroc_seed_std'}}
    if value.get('tpr') is not None and value.get('fpr') is not None:
        assert 0<=value['tpr']<=1 and 0<=value['fpr']<=1
        if 'positive' in value and 'n' in value:
            assert value['n']==2*value['positive'],'Expected balanced matched readouts'
        result['accuracy']=(value['tpr']+1-value['fpr'])/2
    return result


def compact(value):
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Non-finite dashboard metric')
        return round(value,6)
    if isinstance(value,list):return [compact(v) for v in value]
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=P/'summary.json')
    parser.add_argument('--output',type=Path,default=P/'natural_probe_dashboard.html')
    parser.add_argument('--early-summary',type=Path,default=P/'early_offsets_summary.json')
    args=parser.parse_args();data=json.loads(args.summary.read_text(encoding='utf-8'))
    assert data['complete'] and data['aggregates'] and data['curves']
    assert {'verification','readout_verification'}<=data['audits'].keys()
    assert {'train','val','test'}<=data['shared_sampling'].keys()
    # Keep all scored cells/model-seed metrics, omitting repeated server paths.
    data['shared_sampling'].pop('source_files',None)
    if args.early_summary.exists():
        early=json.loads(args.early_summary.read_text(encoding='utf-8'))
        assert early['complete'] and all(early['audits'].values())
        data['early_offsets']=early
    data=accuracy_metrics(data)
    data['decision_rule']=dict(probability_threshold=.5,positive_if='probability >= threshold',
        aggregation='macro over matched token positions, probe seeds and model seeds',
        population='balanced natural wrong/legal readout pairs; not free-generation prevalence')
    payload=json.dumps(compact(data),ensure_ascii=False,separators=(',',':'),allow_nan=False)
    payload=payload.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    template=(P/'dashboard_template.html').read_text(encoding='utf-8')
    assert template.count('__DASHBOARD_DATA__')==1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(template.replace('__DASHBOARD_DATA__',payload),encoding='utf-8')
    print(f'Wrote {args.output} ({args.output.stat().st_size:,} bytes; fully offline)')


if __name__=='__main__':main()
