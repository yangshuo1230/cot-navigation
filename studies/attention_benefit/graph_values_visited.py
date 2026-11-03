"""Test contextual graph values and visited reads at candidate decisions.

Context-free graph V donors are from the same token/position/model with
graph queries restricted to self in preceding layers. Normal query/CoT
computations remain available. V patching preserves recipient Q/K/weights.
Only aggregated results are saved; no attention/activation dumps.
"""
import runpy,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
M=runpy.run_path(str(P/'run.py'))
CONDITIONS=['identity','identity_L2','top3_contextfree_V','matched3_contextfree_V','all_graph_contextfree_V',
            'top3_zero_V','visited_all_L2','visited_current_L2','visited_other_L2']


class GraphSelf:
    def __init__(self,model,batch,tokens,through):
        self.saved=[];B,L=tokens.shape;dev=tokens.device
        q=torch.arange(L,device=dev)[None,:,None];k=torch.arange(L,device=dev)[None,None,:]
        ends=torch.tensor([c['edge_end'] for c in batch],device=dev)[:,None,None]
        keep=((k<=q)&((q>=ends)|(k==q)))[:,None]
        for b in model.blocks[:through]:
            self.saved.append((b.attn,b.attn._attend))
            def attend(q,k,v,original_keep,force_math=False,causal=False,mask=keep):
                assert original_keep is None and causal and not force_math
                return F.scaled_dot_product_attention(q,k,v,attn_mask=mask),None
            b.attn._attend=attend
    def close(self):
        for a,fn in self.saved:a._attend=fn


class Editor(M['EndQueryLesion']):
    def __init__(self,model,tok,layer,donor):
        self.layer=layer;self.donor=donor;self.last_stats=None
        super().__init__(model,tok)
    def replace(self,li,attn):
        def hook(module,args):
            if li!=self.layer:return None
            out=args[0];B,L,Dm=out.shape;H=attn.n_heads;dim=attn.head_dim
            q,k,v=self.qkv[li].view(B,L,3,H,dim).unbind(2)
            assert H==1
            bi=torch.arange(B,device=out.device);ix=self.lengths-1
            qs=M['apply_rope'](q[bi,ix].unsqueeze(2),ix[:,None],attn.rope_base)
            ks=M['apply_rope'](k.transpose(1,2),torch.arange(L,device='cuda').expand(B,-1),attn.rope_base)
            keep=torch.arange(L,device='cuda')[None,:]<self.lengths[:,None]
            with torch.autocast(device_type='cuda',enabled=False):
                score=torch.matmul(qs.float(),ks.float().transpose(-1,-2))/math.sqrt(dim)
                w=torch.softmax(score.masked_fill(~keep[:,None,None,:],-torch.inf),-1)
                selected=torch.zeros(B,L,dtype=torch.bool,device='cuda');stats=[]
                for i,c in enumerate(self.batch):
                    n=c['edge_end'];wg=w[i,0,0,:n]
                    top=wg.topk(3).indices.tolist();other=set(range(n))-set(top);matched=[]
                    for t in top:
                        candidates=[j for j in other if j%3==t%3]
                        j=min(candidates,key=lambda j:(abs(j-t),j));matched.append(j);other.remove(j)
                    which=top if self.condition.startswith('top3') else matched
                    if self.condition=='all_graph_contextfree_V':which=list(range(n))
                    if self.condition=='marker_contextfree_V':which=list(range(0,n,3))
                    if self.condition=='endpoint_contextfree_V':which=[j for j in range(n) if j%3!=0]
                    selected[i,which]=True
                    role=torch.arange(n,device='cuda')%3
                    topkey=top[0]
                    stats.append([float(wg.sum()),float(wg[role==0].sum()),float(wg[role!=0].sum()),
                                  float(wg[top].sum()),float(topkey%3==0),float(topkey%3==1),float(topkey%3==2)])
                self.last_stats=np.asarray(stats)
                vv=v.transpose(1,2).float()
                if 'contextfree' in self.condition or self.condition=='top3_zero_V':
                    replacement=torch.zeros_like(vv) if self.condition=='top3_zero_V' else self.donor.view(B,L,3,H,dim)[:,:,2].transpose(1,2).float()
                    vv=torch.where(selected[:,None,:,None],replacement,vv)
                elif self.condition.startswith('visited_'):
                    deny=torch.zeros_like(selected)
                    for i,c in enumerate(self.batch):
                        a,b=c['vis'];mask=torch.arange(L,device='cuda');span=(mask>=a)&(mask<b)
                        current=self.tokens[i]==c['cur']
                        if self.condition.startswith('visited_current_'):span&=current
                        if self.condition.startswith('visited_other_'):span&=~current
                        deny[i]=span
                    w=w.masked_fill(deny[:,None,None,:],0);w=w/w.sum(-1,keepdim=True)
                elif self.condition!='identity':raise ValueError(self.condition)
                changed=torch.matmul(w,vv).to(out.dtype).reshape(B,Dm)
            result=out.clone();result[bi,ix]=changed;return (result,)
        return hook


def evaluate(model,tok,cases,graph_layer=5):
    graph_li=graph_layer-1
    graph_check=f'identity_argmax_drift_L{graph_layer}'
    probes=[]
    for c in cases:
        for p in c['probes']:
            if M['ROLES'][p['role']] not in ('candidate_nonempty','candidate_empty'):continue
            visited=c['tokens'][p['vis_span'][0]:p['vis_span'][1]]
            outgoing=c['edges'][c['edges'][:,1]==p['cur'],2]
            probes.append(dict(query_id=c['query_id'],prefix=c['tokens'][:p['q']+1],edge_end=c['edge_end'],
                cur=p['cur'],vis=p['vis_span'],allowed=p['eligible'] or [tok.id('|')],empty=not p['eligible'],
                has_visited_neighbor=bool(set(visited)&set(outgoing.tolist()))))
    sums=np.zeros((len(cases),3,len(CONDITIONS),2));counts=np.zeros_like(sums)
    statsum=np.zeros((len(cases),7));statn=np.zeros((len(cases),1))
    index={c['query_id']:i for i,c in enumerate(cases)}
    checks={graph_check:0,'donor_same_token_maxdiff':0.}
    if 'identity_L2' in CONDITIONS:checks['identity_argmax_drift_L2']=0
    if 'identity_L4' in CONDITIONS:checks['identity_argmax_drift_L4']=0
    fieldgroups=np.zeros(3,dtype=int)
    for lo in range(0,len(probes),16):
        batch=probes[lo:lo+16];tokens,lengths=M['pad_rows']([c['prefix'] for c in batch],model.pad_id,'cuda')
        # At the chosen layer graph keys, donors have never read other positions.
        values={};h=model.blocks[graph_li].attn.qkv.register_forward_hook(lambda mod,args,out:values.update(donor=out.detach()))
        donor_editor=GraphSelf(model,batch,tokens,graph_li)
        try:
            with torch.no_grad(),M['_autocast'](torch.device('cuda'),'bf16'):model(tokens)
        finally:h.remove();donor_editor.close()
        # Same-token graph values must be identical regardless of surrounding
        # edges after self-only graph attention. Check within each prefix.
        dv=values['donor'].view(tokens.shape[0],tokens.shape[1],3,model.d_model)[:,:,2]
        for i,c in enumerate(batch):
            ids=tokens[i,:c['edge_end']]
            for token in ids.unique():
                vv=dv[i,:c['edge_end']][ids==token]
                checks['donor_same_token_maxdiff']=max(checks['donor_same_token_maxdiff'],float((vv-vv[:1]).abs().max()))
        with torch.no_grad(),M['_autocast'](torch.device('cuda'),'bf16'):
            standard=M['_gather_row'](model(tokens),lengths-1).float()
        for ci,condition in enumerate(CONDITIONS):
            layer=1 if condition.startswith('visited_') or condition=='identity_L2' else graph_li
            if condition.endswith('_L4'):layer=3
            editor=Editor(model,tok,layer,values['donor']);editor.batch=batch
            editor.tokens=tokens;editor.lengths=lengths
            try:logits=editor.forward('identity' if condition in ('identity_L2','identity_L4') else condition)
            finally:editor.close()
            if condition=='identity':
                checks[graph_check]+=int((logits.argmax(-1)!=standard.argmax(-1)).sum())
                for i,c in enumerate(batch):
                    qi=index[c['query_id']];statsum[qi]+=editor.last_stats[i];statn[qi]+=1
            if condition=='identity_L2':
                checks['identity_argmax_drift_L2']+=int((logits.argmax(-1)!=standard.argmax(-1)).sum())
            if condition=='identity_L4':
                checks['identity_argmax_drift_L4']+=int((logits.argmax(-1)!=standard.argmax(-1)).sum())
            pred=logits.argmax(-1).tolist();prob=logits.softmax(-1)
            for i,c in enumerate(batch):
                qi=index[c['query_id']];groups=[0]+([1] if not c['empty'] else [])+([2] if c['has_visited_neighbor'] else [])
                for group in groups:
                    sums[qi,group,ci]+=[float(pred[i] in c['allowed']),float(prob[i,c['allowed']].sum())]
                    counts[qi,group,ci]+=1
                    if ci==0:fieldgroups[group]+=1
        print('values/visited',lo,'/',len(probes),flush=True)
    result=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
    stats=np.divide(statsum,statn,out=np.full_like(statsum,np.nan),where=statn>0)
    return dict(graph_layer=graph_layer,conditions=CONDITIONS,groups=['all','nonempty','has_visited_neighbor'],n_fields=fieldgroups.tolist(),
                metrics=['legal_argmax','legal_probability'],means=np.nanmean(result,axis=0).tolist(),
                graph_stats_names=['graph_mass','edge_marker_mass','endpoint_mass','top3_graph_mass',
                                  'top_graph_is_marker','top_graph_is_source','top_graph_is_dest'],
                graph_stats=np.nanmean(stats,axis=0).tolist(),checks=checks)


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    out={}
    for arm in ('clean','error'):
        cfg,tok,model=M['load_model'](arm)
        source=M['D']['ContextStore'](cfg.data.root);cache=M['D']['ScalePreparedRows'](source,cfg,tok)
        rows=json.loads((M['AUDIT']/'strict/dfs_clean_main.json').read_text())['rows']
        cases=M['build_cases'](source,cache,tok,[r['query_id'] for r in rows])
        out[arm]=evaluate(model,tok,cases);M['write'](P/'graph_values_visited.json',out)
        del model;torch.cuda.empty_cache()
    out['finished']=True;M['write'](P/'graph_values_visited.json',out)


if __name__=='__main__':main()
