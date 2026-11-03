"""Deduplicate resumed logs and compare within the same final curriculum stage."""
from pathlib import Path
import json,math,statistics,hashlib
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
base=ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum'
dual=ROOT/'outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023_r1'
specs=[('baseline/'+p,base/p) for p in ['dfs_state','bfs_state']]+[(str(p.parent.relative_to(dual)),p.parent) for p in sorted(dual.glob('*/*/*/resolved_config.yaml'))]
results=[]
for name,run in specs:
    cfg=yaml.safe_load((run/'resolved_config.yaml').read_text());raw=[];by_step={}
    for line in (run/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line)
        if 'total_loss' not in row:continue
        raw.append(row);by_step[int(row['step'])]=row
    # Original baseline logged every10, dual every100. Use the same 100-step grid.
    rows=[by_step[s] for s in sorted(by_step) if s%100==0]
    final=cfg['train']['max_steps'];ends=cfg['train']['scale_stage_ends'];start=ends[-2]
    assert len(rows)==final//100 and rows[-1]['step']==final
    def window(a,b):
        chunk=[x for x in rows if a<x['step']<=b]
        keys=['total_loss','trace_loss','answer_loss','grad_norm','lr']
        return {'start':a,'end':b,'n':len(chunk),**{k:{'mean':statistics.mean(x[k] for x in chunk),
                    'median':statistics.median(x[k] for x in chunk),'std':statistics.stdev(x[k] for x in chunk) if len(chunk)>1 else 0.} for k in keys}}
    windows={'last20k':window(final-20000,final),'last10k':window(final-10000,final),
             'previous10k':window(final-20000,final-10000),'last5k':window(final-5000,final),
             'previous5k':window(final-10000,final-5000),
             'phase_first10k':window(start,start+10000)}
    bins=[]
    for end in range(1000,final+1,1000):
        chunk=[x for x in rows if end-1000<x['step']<=end]
        bins.append({'step':end,'phase_elapsed':end-start,'mean_loss':statistics.mean(x['total_loss'] for x in chunk),
                     'mean_trace_loss':statistics.mean(x['trace_loss'] for x in chunk),
                     'mean_grad_norm':statistics.mean(x['grad_norm'] for x in chunk),
                     'mean_lr':statistics.mean(x['lr'] for x in chunk)})
    tail=[x for x in bins if x['step']>final-20000];xx=np.array([x['step'] for x in tail],float)/10000;yy=np.array([x['mean_loss'] for x in tail])
    slope,intercept=np.polyfit(xx,yy,1);pred=intercept+slope*xx
    r2=1-float(((yy-pred)**2).sum())/float(((yy-yy.mean())**2).sum()) if yy.var() else 0.
    validation=[]
    for file in sorted((run/'eval').glob('free_[0-9]*.json')):
        report=json.loads(file.read_text());step=report.get('step')
        if step is None:continue
        validation.append({'step':step,'phase_elapsed':step-start,**report['overall']})
    result={'model':name,'run':str(run),'protocol':cfg['protocol']['name'],'steps':final,'stage_ends':ends,
        'train_config':cfg['train'],'raw_training_records':len(raw),'unique_training_steps':len(by_step),
        'discarded_duplicate_rows':len(raw)-len(by_step),'common_grid':100,'windows':windows,
        'tail20k_slope_per10k':float(slope),'tail20k_slope_r2':r2,
        'loss_change_last10k_vs_previous10k_percent':100*(windows['last10k']['total_loss']['mean']/windows['previous10k']['total_loss']['mean']-1),
        'loss_change_last5k_vs_previous5k_percent':100*(windows['last5k']['total_loss']['mean']/windows['previous5k']['total_loss']['mean']-1),
        'loss_change_phase_first10k_to_last10k_percent':100*(windows['last10k']['total_loss']['mean']/windows['phase_first10k']['total_loss']['mean']-1),
        'bins_1k':bins,'validation':validation,'source_sha256':hashlib.sha256((run/'metrics.jsonl').read_bytes()).hexdigest()}
    results.append(result)
    vals=[(v['step'],v['answer_accuracy']) for v in validation if v['step']>=final-20000]
    print(name,'tail10k',round(windows['last10k']['total_loss']['mean'],6),
      'delta10k%',round(result['loss_change_last10k_vs_previous10k_percent'],2),
      'delta5k%',round(result['loss_change_last5k_vs_previous5k_percent'],2),
      'slope/10k',round(float(slope),6),'r2',round(r2,3),'vals',vals,flush=True)
(OUT/'summary.json').write_text(json.dumps({'runs':results,'note':'Training losses only; held-out teacher-forced reports contain answer accuracy, not trace NLL. Resume duplicates use final logged attempt per optimizer step; grid100 equalizes logging density. Final phase is N100-dominated mixture, not N100-only.'},indent=2)+'\n')
