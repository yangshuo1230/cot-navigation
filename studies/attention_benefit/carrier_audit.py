"""Check whether edge information is read at the following boundary token.

Read-only attention alignment plus query-local, position-matched lesions.
Offset 3 is the next EDGE marker, except the last edge is followed by QUERY.
No claim that attention at raw endpoint positions exhausts edge-information
retrieval. Same common hard-DAG prefixes as the main retest.
"""
import runpy,time
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
R=runpy.run_path(str(P/'hard_retest.py'));M=R['M'];H=R['H']
OFFSETS=[-3,0,1,2,3,6]
CONDITIONS=['identity','raw_relevant_edges','next_carriers','matched_carriers']


class Editor(M['EndQueryLesion']):
    def prepare(self,batch):
        self.tokens,self.lengths=M['pad_rows']([c['prefix'] for c in batch],self.model.pad_id,'cuda')
        B,L=self.tokens.shape
        self.masks={c:torch.zeros(B,L,dtype=torch.bool,device='cuda') for c in CONDITIONS[1:]}
        for i,c in enumerate(batch):
            relevant=c['relevant'];others=set(range(c['edge_end']//3))-set(relevant)
            for e in relevant:
                self.masks['raw_relevant_edges'][i,3*e:3*e+3]=True
                assert 3*e+3<len(c['prefix'])
                self.masks['next_carriers'][i,3*e+3]=True
                j=min(others,key=lambda j:(abs(j-e),j));others.remove(j)
                self.masks['matched_carriers'][i,3*j+3]=True


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    _,_,source,cache,old=R['test_source']();ids=[r['query_id'] for r in old]
    out=dict(offsets=OFFSETS,conditions=CONDITIONS,results={})
    for name in R['FOLDERS']:
        cfg,tok,model,meta=R['load'](name)
        cases,coverage=H['bounded_cases'](source,cache,tok,ids,4096,hard=True)
        # Only nonempty candidates: every analyzed state has a real target edge.
        selected=[]
        for c in cases:
            ps=[p for p in c['probes'] if M['ROLES'][p['role']]=='candidate_nonempty']
            if ps:selected.append(dict(c,probes=ps))
        counts=np.zeros((len(ids),1,1));sums=np.zeros((len(ids),model.n_layers,len(OFFSETS)))
        index={q:i for i,q in enumerate(ids)}
        reader=M['SelectedRows'](model)
        try:
            for lo in range(0,len(selected),8):
                batch=selected[lo:lo+8];weights=reader.forward(batch)
                for bi,c in enumerate(batch):
                    qi=index[c['query_id']]
                    for pi,p in enumerate(c['probes']):
                        relevant=np.flatnonzero(c['edges'][:,1]==p['cur']);counts[qi]+=1
                        for li,w in weights.items():
                            row=w[bi,:,pi,:p['q']+1].mean(0)
                            for oi,offset in enumerate(OFFSETS):
                                keys=np.unique(3*relevant+offset);keys=keys[(keys>=0)&(keys<=p['q'])]
                                sums[qi,li,oi]+=row[keys].sum()
        finally:reader.close()
        mass=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
        probes=[]
        for c in selected:
            for p in c['probes']:
                probes.append(dict(query_id=c['query_id'],prefix=c['tokens'][:p['q']+1],edge_end=c['edge_end'],
                    relevant=np.flatnonzero(c['edges'][:,1]==p['cur']).tolist(),allowed=p['eligible']))
        totals=np.zeros((len(ids),len(CONDITIONS)));n=np.zeros_like(totals);drift=0
        for lo in range(0,len(probes),16):
            batch=probes[lo:lo+16];editor=Editor(model,tok);editor.prepare(batch)
            try:
                for ci,condition in enumerate(CONDITIONS):
                    logits=editor.forward(condition);pred=logits.argmax(-1).tolist()
                    for i,c in enumerate(batch):
                        qi=index[c['query_id']];totals[qi,ci]+=pred[i] in c['allowed'];n[qi,ci]+=1
            finally:editor.close()
        values=np.divide(totals,n,out=np.full_like(totals,np.nan),where=n>0)
        rng=np.random.default_rng(20261024);diff=values-values[:,:1]
        boot=np.stack([np.nanmean(diff[rng.integers(0,len(ids),len(ids))],axis=0) for _ in range(1000)])
        out['results'][name]=dict(metadata=meta,n_nonempty_fields=len(probes),
            n_queries=int((n[:,0]>0).sum()),attention_mass=np.nanmean(mass,axis=0).tolist(),
            legal_next_token=np.nanmean(values,axis=0).tolist(),
            delta_ci_low=np.nanquantile(boot,.025,axis=0).tolist(),delta_ci_high=np.nanquantile(boot,.975,axis=0).tolist())
        H['write'](R['OUT']/'carrier_audit.json',out)
        print('CARRIER',name,out['results'][name],flush=True)
        del model;torch.cuda.empty_cache()
    out['finished']=True;H['write'](R['OUT']/'carrier_audit.json',out)


if __name__=='__main__':main()
