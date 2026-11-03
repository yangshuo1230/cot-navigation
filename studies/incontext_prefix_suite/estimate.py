"""Initial estimate from actual production update measurements, not a guarantee."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
single=json.loads((HERE/'benchmark_single.json').read_text());dual=json.loads((HERE/'benchmark_serial.json').read_text())
def duration(report):
    d={r['name']:r['median_update_s'] for r in report['measurements']}
    # Linear interpolation is an approximation for the two unmeasured middle sizes.
    simple=[d['simple16']+(d['simple100']-d['simple16'])*r for r in (0,20/84,48/84,1)]
    hard=[d['hard16']+(d['hard100']-d['hard16'])*r for r in (0,20/84,48/84,1)]
    return sum(t*40000 for t in simple+hard[:3])+hard[3]*80000
s=duration(single);d=duration(dual);boot=550000*single['measurements'][0]['median_update_s']
manifest=json.loads((HERE/'manifest.json').read_text());gpu=[0.]*8;costs=[]
for j in manifest['jobs']:
    seconds=(s if j['order']=='single' else d)+(boot if j['protocol']=='direct' else 0)
    slot=min(range(8),key=lambda i:gpu[i]);gpu[slot]+=seconds
    costs.append(dict(job=j['id'],training_hours_estimate=seconds/3600))
result=dict(status='ESTIMATE',single_main_hours=s/3600,dual_main_hours=d/3600,direct_boot_hours_proxy=boot/3600,
    aggregate_gpu_hours=sum(gpu)/3600,simulated_training_only_wall_hours=max(gpu)/3600,
    preliminary_wall_hours_range=[max(gpu)/3600*1.15,max(gpu)/3600*1.65],jobs=costs,
    assumptions=['A800x8 continuously available','single-route query-bidirectional proxy for all single masks',
        'serial query-bidirectional proxy for both dual orders','N36/N64 linearly interpolated from N16/N100',
        'standard M24 bootstrap step cost proxies the full 550k Direct bootstrap',
        'range includes an allowance for compilation/validation/checkpoints/contention, not a measured guarantee',
        'stage boot time and model failure can shift completion; estimates will need refresh from formal logs'])
(HERE/'time_estimate.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='jobs'},indent=2))
