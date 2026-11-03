"""Withdraw exactly one user-rejected cohort; retain data for provenance."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import time

NOTICE='已废弃：本批6/9层×30/60/90%模型存在问题，全部派生结果不得作为有效结论或对照。'


def write_json(path,value):
    tmp=path.with_suffix('.withdraw.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--stop-pid',type=int);a=ap.parse_args();root=a.root.resolve()
    record=dict(lifecycle='deprecated',date='2026-11-02',reason='User reported this experimental cohort has a problem; specific cause not supplied.',
        scope='6/9 layers x 30/60/90 percent x seeds0/1; checkpoints and all derived analyses, figures, and claims',
        preserve='Raw files retained for provenance; no conclusions, baseline reuse, donors or initialization reuse permitted.')
    if a.stop_pid is not None:
        path=Path(f'/proc/{a.stop_pid}/cmdline')
        cmd=path.read_bytes() if path.exists() else b''
        if cmd:
            assert b'correction_behavior/oracle_mistake.py' in cmd and b'depth_rate_20261101' in cmd
            os.kill(a.stop_pid,signal.SIGTERM)
            for _ in range(50):
                if not path.exists() or not path.read_bytes():break
                time.sleep(.1)
            assert not path.exists() or not path.read_bytes(),'Target did not terminate'
            record['stopped_oracle_pid']=a.stop_pid
    train=root/'outputs/correction_depth_rate/launches/dfs_depth_rate_20261031'
    analysis=root/'outputs/analysis/correction_behavior/depth_rate_20261101'
    targets=[p for p in [train,analysis,analysis/'oracle_mistake',analysis/'marker_census',analysis/'figures'] if p.exists()]
    if train.exists():targets+=list(train.glob('dfs_state/l*_p*/s[01]'))
    for p in targets:
        write_json(p/'DEPRECATED.json',record)
        status=p/'status.json'
        if status.exists():
            obj=json.loads(status.read_text(encoding='utf-8'));obj['lifecycle']='deprecated'
            if p==analysis/'oracle_mistake':obj['phase']='stopped_deprecated'
            write_json(status,obj)
    manifest=root/'studies/correction_behavior/cohorts/depth_rate_20261101.json'
    if manifest.exists():
        obj=json.loads(manifest.read_text(encoding='utf-8'));obj.update(lifecycle='deprecated',withdrawal=record);write_json(manifest,obj)
    if analysis.exists():
        for path in analysis.rglob('*.md'):
            text=path.read_text(encoding='utf-8')
            if not text.startswith('> '+NOTICE):path.write_text('> '+NOTICE+'\n> 撤回日期：2026-11-02；原始内容保留供追溯。\n\n'+text,encoding='utf-8')
        banner='<div style="background:#b71c1c;color:white;padding:20px;font-size:20px;font-weight:bold">'+NOTICE+'</div>'
        for path in analysis.rglob('*.html'):
            text=path.read_text(encoding='utf-8')
            if NOTICE not in text:path.write_text(text.replace('<main>','<main>'+banner,1).replace('<title>','<title>【已废弃】',1),encoding='utf-8')
        for name in ['summary.json','plot_snapshot.json']:
            path=analysis/name
            if path.exists():
                obj=json.loads(path.read_text(encoding='utf-8'));obj.update(lifecycle='deprecated',withdrawal=record);write_json(path,obj)
        data=analysis/'figures/figure_data.csv'
        if data.exists():
            with data.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            if rows:
                for row in rows:row['lifecycle']='deprecated'
                with data.open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    legacy=root/'studies/correction_depth_rate/run.py'
    if legacy.exists():
        text=legacy.read_text(encoding='utf-8')
        old="    output = ROOT / 'outputs/correction_depth_rate/launches' / args.run_name\n"
        new=old+"    if (output / 'DEPRECATED.json').exists():\n        raise ValueError('Cohort withdrawn by user; training/resume disabled')\n"
        if old in text and "training/resume disabled" not in text:legacy.write_text(text.replace(old,new,1))
    write_json(root/'studies/correction_behavior/withdrawal_depth_rate_20261102.json',record)
    print('WITHDRAWN',len(targets),'artifact/model directories')


if __name__=='__main__':main()
