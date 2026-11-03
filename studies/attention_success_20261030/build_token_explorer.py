"""Full-token inspection with layer-correct upstream/downstream navigation."""
import json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    original=json.loads((ROOT/'animation_data.json').read_text(encoding='utf-8'))
    trace=original['reference'];labels=trace['labels'];info=trace['info'];prompt=original['prompt_len']
    for p in range(72):
        role=['graph_marker','graph_source','graph_target'][p%3]
        info[p]['role']=role
    info[72]['description']='查询标记 QUERY'
    info[75]['description']='查询结束 |'
    for rec in trace['records']:
        bars=[p for p in range(rec['start'],rec['end']) if labels[p]=='|']
        for p,label in zip(bars,['stack 结束 |','visited 结束 |','候选结束 |']):
            info[p]['description']=f'R{rec["record"]} {label}'
        info[rec['end']-1]['description']=f'R{rec["record"]} 记录结束 ;'
    att=np.load(ROOT/'attention.npz')['attention']
    assert att.shape==(6,len(labels),len(labels))
    assert np.allclose(att.sum(-1),1,atol=2e-6)
    raw_ids=np.load(ROOT/'attention.npz')['tokens'].astype(int).tolist()
    data=dict(labels=labels,raw_ids=raw_ids,info=info,attention=att.tolist(),prompt_len=prompt,records=trace['records'],
              graph=original['graph'],display_aliases=original['display_aliases'],
              default_position=next(f['row'] for f in trace['frames'] if f['record']==2 and f['phase']=='argument'))
    data['graph']={k:v for k,v in data['graph'].items() if k not in ['attention','blocks']}
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':'))
    html=(ROOT/'token_explorer_template.html').read_text(encoding='utf-8').replace('__DATA__',payload)
    # Keep the existing animation, make the focused inspector the main entry.
    if not (ROOT/'search_animation.html').exists():shutil.copyfile(ROOT/'search_attention.html',ROOT/'search_animation.html')
    (ROOT/'token_explorer.html').write_text(html,encoding='utf-8')
    (ROOT/'search_attention.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(tokens=len(labels),layers=6,weights=int(att.size),default_position=data['default_position'])))

if __name__=='__main__':main()
