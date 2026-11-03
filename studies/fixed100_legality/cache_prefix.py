"""Audit-local KV prefix views. Storage/context/weights are unchanged.

The original decoder attends over its entire allocated 32k KV buffer with
future slots masked out. These views remove only masked future slots, rounded
up to 128 positions. No training process or project attention file is edited.
"""
from types import MethodType


class PrefixList:
    def __init__(self,values,cache):self.values=values;self.cache=cache
    def __getitem__(self,index):return self.values[index][:,:,:self.cache.prefix_visible,:]


def enable(model):
    original=model.new_cache
    def new_cache(self,batch,device,dtype):
        cache=original(batch,device,dtype)
        cache.prefix_visible=cache.max_len
        cache.k=PrefixList(cache.k,cache);cache.v=PrefixList(cache.v,cache)
        write=cache.write
        def bounded_write(layer,k,v,positions):
            if layer==0:
                needed=int(positions.max().item())+1
                cache.prefix_visible=min(cache.max_len,((needed+127)//128)*128)
            write(layer,k,v,positions)
        cache.write=bounded_write
        return cache
    model.new_cache=MethodType(new_cache,model)
    return model
