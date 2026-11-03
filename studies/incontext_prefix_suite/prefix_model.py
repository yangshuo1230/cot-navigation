"""Query-independent or query-conditioned prefix attention; unchanged backbone sizes."""
import torch
from torch import nn
import torch.nn.functional as F
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.models.attention import KVCache,apply_rope
from cot_navigation.models.ffn import build_ffn

class PrefixCache(KVCache):
    def select_batch(self,indices):
        super().select_batch(indices)
        self.prompt_end=self.prompt_end.index_select(0,indices)
        self.bi_end=self.bi_end.index_select(0,indices)

def visibility(positions,keys,bi_end):
    q=positions[:,None,:,None];k=keys[None,None,None,:]
    return (k<=q)|((q<bi_end[:,None,None,None])&(k<bi_end[:,None,None,None]))

class PrefixAttention(nn.Module):
    def __init__(self,width,rope_base):
        super().__init__();self.rope_base=rope_base
        self.qkv=nn.Linear(width,3*width,bias=False);self.proj=nn.Linear(width,width,bias=False)
    def forward(self,x,positions,cache=None,layer_idx=0,cache_mode=None,return_attn=False,capture=None,record_keep=None):
        q,k,v=self.qkv(x).chunk(3,-1)
        q=apply_rope(q[:,None],positions,self.rope_base);k=apply_rope(k[:,None],positions,self.rope_base);v=v[:,None]
        if cache is not None:
            cache.write(layer_idx,k,v,positions)
            if cache_mode=='decode':
                end=min(cache.max_len,((int(cache.valid_length or cache.max_len)+7)//8)*8)
                k,v=cache.k[layer_idx][:,:,:end],cache.v[layer_idx][:,:,:end]
        bi=record_keep
        keep=visibility(positions,torch.arange(k.shape[2],device=x.device),bi)
        if return_attn:
            logits=(q.float()@k.float().transpose(-2,-1))*(q.shape[-1]**-.5)
            weights=logits.masked_fill(~keep,float('-inf')).softmax(-1)
            out=(weights@v.float()).to(q.dtype)
        else:
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=keep);weights=None
        return self.proj(out[:,0]),weights

class Branch(nn.Module):
    def __init__(self,width,rope):
        super().__init__();self.dim=width//2;self.rope=rope
        self.qkv=nn.Linear(width,3*self.dim,bias=False);self.proj=nn.Linear(self.dim,width,bias=False)
    def project(self,x,positions):
        q,k,v=self.qkv(x).chunk(3,-1)
        return apply_rope(q[:,None],positions,self.rope),apply_rope(k[:,None],positions,self.rope),v[:,None]

class DualBlock(nn.Module):
    def __init__(self,width,ff,order,rope,cap):
        super().__init__();self.order=order;self.cap=cap
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.graph=Branch(width,rope);self.history=Branch(width,rope);self.ffn=build_ffn('standard',width,ff)
    def forward(self,x,positions,prompt,bi,cache=None,layer_idx=0,cache_mode=None,capture=None):
        norm=self.norm1(x);hq,hk,hv=self.history.project(norm,positions)
        decode=cache is not None and cache_mode=='decode'
        if cache is not None:
            b=torch.arange(x.shape[0],device=x.device)[:,None]
            cache.k[layer_idx][b,1,positions]=hk[:,0];cache.v[layer_idx][b,1,positions]=hv[:,0]
            if decode:
                end=min(cache.max_len,((int(cache.valid_length or cache.max_len)+7)//8)*8)
                hk,hv=cache.k[layer_idx][:,1:2,:end],cache.v[layer_idx][:,1:2,:end]
        keys=torch.arange(hk.shape[2],device=x.device)
        hkeep=(keys[None,None,None,:]>=prompt[:,None,None,None])&(keys[None,None,None,:]<=positions[:,None,:,None])
        hout=F.scaled_dot_product_attention(hq,hk,hv,attn_mask=hkeep)
        delta=self.history.proj(hout[:,0])*(positions>=prompt[:,None])[:,:,None]
        intermediate=x+delta
        gnorm=self.norm1(intermediate) if self.order=='serial' else norm
        gq,gk,gv=self.graph.project(gnorm,positions)
        if cache is not None and not decode:
            cache.k[layer_idx][b,0,positions]=gk[:,0];cache.v[layer_idx][b,0,positions]=gv[:,0]
        if decode:
            size=min(cache.valid_length or cache.max_len,self.cap)
            gk,gv=cache.k[layer_idx][:,0:1,:size],cache.v[layer_idx][:,0:1,:size]
        else:
            size=min(x.shape[1],self.cap);gk,gv=gk[:,:,:size],gv[:,:,:size]
        keys=torch.arange(size,device=x.device)
        keep=(keys[None,None,None,:]<prompt[:,None,None,None])&visibility(positions,keys,bi)
        gout=F.scaled_dot_product_attention(gq,gk,gv,attn_mask=keep)
        x=intermediate+self.graph.proj(gout[:,0]);x=x+self.ffn(self.norm2(x))
        if capture is not None:capture.setdefault('resid',[]).append(x)
        return x

class PrefixLM(TransformerLM):
    def __init__(self,cfg,vocab,pad_id,mode='graph',order='single'):
        self.mode=mode;self.order=order;self.cap=1024
        super().__init__(vocab,cfg.layers,1,cfg.d_model,cfg.d_ff,cfg.max_seq_len,
                         dropout=0.,rope_base=cfg.rope_base,pad_id=pad_id)
        if order=='single':
            # Replacement consumes RNG, so restore the exact vanilla initialization.
            for block in self.blocks:
                old=block.attn
                replacement=PrefixAttention(cfg.d_model,cfg.rope_base)
                replacement.load_state_dict(old.state_dict());block.attn=replacement
        else:
            self.n_heads=2
            self.blocks=nn.ModuleList([DualBlock(cfg.d_model,cfg.d_ff,order,cfg.rope_base,self.cap) for _ in range(cfg.layers)])
            self.reset_parameters()
    def new_cache(self,batch,device,dtype):
        return PrefixCache(self.n_layers,batch,self.n_heads,self.max_seq_len,self.d_model//self.n_heads,device,dtype)
    def forward(self,tokens,positions=None,cache=None,cache_mode=None,capture=None,return_attn=False):
        if positions is None:positions=torch.arange(tokens.shape[1],device=tokens.device)[None].expand(tokens.shape[0],-1)
        if cache is not None and cache_mode=='decode':prompt,bi=cache.prompt_end,cache.bi_end
        else:
            markers=tokens==3;query=markers.to(torch.int64).argmax(1);prompt=query+4
            torch._assert_async(markers.any(1).all(),'QUERY required')
            torch._assert_async((prompt<=self.cap).all(),'graph prefix capacity exceeded')
            bi=query if self.mode=='graph' else prompt if self.mode=='query' else torch.zeros_like(query)
            if cache is not None:cache.prompt_end=prompt;cache.bi_end=bi
        x=self.embed(tokens);weights=[]
        for i,block in enumerate(self.blocks):
            if self.order=='single':
                x,w=block(x,positions,cache,i,cache_mode,return_attn,capture,bi);weights.append(w)
            else:
                if return_attn:raise ValueError('Use graph/history branch hooks for dual attention diagnostics')
                x=block(x,positions,prompt,bi,cache,i,cache_mode,capture)
        logits=self.lm_head(self.final_norm(x))
        return (logits,weights) if return_attn else logits
    def report(self):
        return {**super().report(),'prefix_mask':self.mode,'attention_order':self.order,'prefix_cap':self.cap}
