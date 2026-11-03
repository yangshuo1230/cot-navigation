"""Restore graph-informed historical node residuals after early graph cuts.

Clean and ablated runs use native SDPA with identical explicit-mask shapes.
Restoring current-node positions excludes the decision position. Matched
controls restore equal numbers of nearby other historical tokens.
"""
import runpy,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
R=runpy.run_path(str(P/'recheck.py'));M=R['M']
CONDITIONS=['identity','patch_identity_current']
for k in (1,2,5):
    CONDITIONS += [f'cut_{k}',f'current_{k}',f'matched_{k}']
CONDITIONS+=['all_restore_2']


def evaluate(model,tok,cases,first_only=False):
    probes=[]
    for c in cases:
        for p in c['probes']:
            if M['ROLES'][p['role']] not in ('candidate_nonempty','candidate_empty'):continue
            probes.append(dict(query_id=c['query_id'],prefix=c['tokens'][:p['q']+1],
                edge_end=c['edge_end'],cur=p['cur'],allowed=p['eligible'] or [tok.id('|')],
                empty=not p['eligible'],relevant_edges=np.flatnonzero(c['edges'][:,1]==p['cur']).tolist()))
    conditions=['identity','cut_1','cut_2','cut_6'] if first_only else CONDITIONS
    sums=np.zeros((len(cases),3,len(conditions),2));counts=np.zeros_like(sums)
    index={c['query_id']:i for i,c in enumerate(cases)}
    checks=dict(identity_patch_maxdiff=0.,all_restore_maxdiff=0.)
    for lo in range(0,len(probes),16):
        batch=probes[lo:lo+16]
        tokens,lengths=M['pad_rows']([c['prefix'] for c in batch],model.pad_id,'cuda')
        B,L=tokens.shape
        cur=torch.zeros((B,L),dtype=torch.bool,device='cuda');matched=cur.clone();nongraph=cur.clone()
        for bi,c in enumerate(batch):
            last=len(c['prefix'])-1
            current=[i for i in range(c['edge_end'],last) if c['prefix'][i]==c['cur']]
            available=set(range(c['edge_end'],last))-set(current)
            for i in current:
                j=min(available,key=lambda j:(abs(j-i),j));available.remove(j)
                cur[bi,i]=True;matched[bi,j]=True
            nongraph[bi,c['edge_end']:last+1]=True
        editor=R['NativeMask'](model,batch,tokens,lengths,'explicit_identity')
        capture={}
        try:
            with torch.no_grad(),M['_autocast'](torch.device('cuda'),'bf16'):
                baseline=M['_gather_row'](model(tokens,capture=capture),lengths-1).float()
        finally:editor.close()
        for ci,condition in enumerate(conditions):
            handle=None
            if condition=='identity':logits=baseline
            else:
                if condition=='patch_identity_current':k=2;kind='current';maskcondition='explicit_identity'
                elif condition=='all_restore_2':k=2;kind='all';maskcondition='global_first_2'
                else:
                    kind,ks=condition.rsplit('_',1);k=int(ks);maskcondition=f'global_first_{k}'
                editor=R['NativeMask'](model,batch,tokens,lengths,maskcondition)
                if kind!='cut':
                    mask={'current':cur,'matched':matched,'all':nongraph}[kind]
                    reference=capture['resid'][k-1]
                    def patch(module,args,out,mask=mask,reference=reference):
                        x,w=out;x=x.clone();x[mask]=reference[mask];return x,w
                    handle=model.blocks[k-1].register_forward_hook(patch)
                try:
                    with torch.no_grad(),M['_autocast'](torch.device('cuda'),'bf16'):
                        logits=M['_gather_row'](model(tokens),lengths-1).float()
                finally:
                    if handle:handle.remove()
                    editor.close()
                if condition=='patch_identity_current':checks['identity_patch_maxdiff']=max(checks['identity_patch_maxdiff'],float((logits-baseline).abs().max()))
                if condition=='all_restore_2':checks['all_restore_maxdiff']=max(checks['all_restore_maxdiff'],float((logits-baseline).abs().max()))
            pred=logits.argmax(-1).tolist();prob=logits.softmax(-1)
            for bi,c in enumerate(batch):
                qi=index[c['query_id']]
                for state in (0,2 if c['empty'] else 1):
                    sums[qi,state,ci]+=[float(pred[bi] in c['allowed']),float(prob[bi,c['allowed']].sum())]
                    counts[qi,state,ci]+=1
        print('rescue',first_only,lo,'/',len(probes),flush=True)
    values=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
    return dict(conditions=conditions,states=['all','nonempty','empty'],metrics=['legal_argmax','legal_probability'],
                means=np.nanmean(values,axis=0).tolist(),n_fields=len(probes),checks=checks)


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    result={}
    for arm in ('clean','error'):
        cfg,tok,model=M['load_model'](arm)
        source=M['D']['ContextStore'](cfg.data.root);cache=M['D']['ScalePreparedRows'](source,cfg,tok)
        rows=json.loads((M['AUDIT']/'strict/dfs_clean_main.json').read_text())['rows']
        cases=M['build_cases'](source,cache,tok,[r['query_id'] for r in rows])
        result[arm]=evaluate(model,tok,cases)
        # Independently test the first record: no previous trace can reveal
        # the current adjacency list. Include it even if not sampled above.
        for c in cases:
            row=cache.row(c['query_id'],0);rec=M['records'](row,tok)[0]
            p=dict(c['probes'][0],q=rec['spans']['cands'][0]-1,cur=rec['stack'][-1],
                   eligible=rec['cands'],record=rec)
            p['role']=M['ROLES'].index('candidate_nonempty' if rec['cands'] else 'candidate_empty')
            c['probes']=[p]
        result[arm+'_first_record']=evaluate(model,tok,cases,first_only=True)
        M['write'](P/'rescue.json',result)
        del model;torch.cuda.empty_cache()
    result['finished']=True;M['write'](P/'rescue.json',result)


if __name__=='__main__':main()
