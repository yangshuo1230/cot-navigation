"""Update interactive HTML only; keep original raw data and GIF/MP4 artifacts."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def main():
    data=json.loads((ROOT/'animation_data.json').read_text(encoding='utf8'))
    raw=np.load(ROOT/'attention.npz')['attention'];n=raw.shape[-1]
    baseline=np.array([[a.diagonal(-d).mean(dtype=np.float64) for d in range(n)] for a in raw])
    score=np.zeros(raw.shape,dtype=float)
    for q in range(n):score[:,q,:q+1]=raw[:,q,:q+1]-baseline[:,np.arange(q,-1,-1)]
    trace=data['reference'];info=trace['info'];edge_n=len(data['graph']['edges']);edge_tokens=edge_n*3
    def group(key,q,ri):
        if key==q:return 10
        role=info[key]['role']
        if role=='graph':return 0
        if role=='source':return 1
        if role=='target':return 2
        if role=='syntax':return 9
        current=info[key]['record']==ri
        if role in ['stack','visited','candidates']:return {'stack':3,'visited':4,'candidates':5}[role] if current else 7
        if role in ['action','argument']:return 6 if current else 8
        return 9
    for f in trace['frames']:
        q=f['row'];g=np.array([group(k,q,f['record']) for k in range(q+1)]);f['masses']=[];f['top']=[]
        for layer in range(6):
            z=score[layer,q,:q+1]
            f['masses'].append([float(z[g==i].sum()) for i in range(len(data['categories']))])
            f['top'].append([dict(position=int(k),token=trace['labels'][k],description=info[k]['description'],value=float(z[k]),group=int(g[k])) for k in np.argsort(-z,kind='stable')[:8]])
        f['edge_weights']=score[:,q,:edge_tokens].reshape(6,edge_n,3).tolist()
    data['graph']['attention']=score[:,:edge_tokens,:edge_tokens].tolist()
    data['graph']['blocks']=score[:,:edge_tokens,:edge_tokens].reshape(6,edge_n,3,edge_n,3).sum(4).mean(2).tolist()
    data['distance_baseline']=baseline.tolist();data['score_definition']='A(layer,q,k)-mean(A(layer,i,j) where i-j=q-k), all causal pairs in the complete sequence; per layer/head, no renormalization.'
    template=(ROOT/'animation_template.html').read_text(encoding='utf8')
    (ROOT/'search_animation.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))),encoding='utf8')
    print(json.dumps(dict(frames=len(trace['frames']),layers=6,html_only=True)))

if __name__=='__main__':main()
