import torch
from torch import nn
from cot_navigation.models.attention import KVCache
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.analysis.oracle_mistake import MistakeOracle
from cot_navigation.evaluation.rollout import generate


def test_completed_wrong_action_is_rewound_without_gold_repair():
    tok=Tokenizer(8,enable_stop=True,enable_mistake=True)
    a,b,c=[tok.node_id(i) for i in range(3)]
    def rec(s,v,cs,action):return s+[tok.id('|')]+v+[tok.id('|')]+cs+[tok.id('|')]+action+[tok.id(';')]
    bad=rec([a],[a],[c],[tok.id('<PUSH>'),c])
    cancelled=rec([a],[a],[c],[tok.id('<MISTAKE>')])
    clean=rec([a],[a],[b],[tok.id('<PUSH>'),b]);found=rec([a,b],[a,b],[],[tok.id('<FOUND>')])
    class Script(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.zeros(1));self.max_seq_len=128
        def new_cache(self,batch,device,dtype):
            cache=KVCache(1,batch,1,128,4,device,dtype);cache.record_tokens=torch.zeros(batch,128,dtype=torch.long);return cache
        def forward(self,tokens,positions=None,cache=None,cache_mode=None):
            z=torch.full((*tokens.shape,len(tok)),-30.)
            for i in range(len(tokens)):
                cache.record_tokens[i,positions[i]]=tokens[i]
                pos=int(positions[i,-1]);prefix=cache.record_tokens[i,:pos+1].tolist()
                stream=(cancelled+clean+found if tok.id('<MISTAKE>') in prefix else bad)+[tok.yes]
                z[i,-1,stream[min(pos,len(stream)-1)]]=30
            return z
    prompts=[[a],[a]]
    native=generate(Script(),prompts,'dfs_state',tok,cot_margin=8)
    noop=generate(Script(),prompts,'dfs_state',tok,cot_margin=8,action_override=lambda i,t:None)
    assert native==noop
    oracles=[MistakeOracle({a:[b],b:[],c:[]},a,b,tok) for _ in prompts]
    forced=generate(Script(),prompts,'dfs_state',tok,cot_margin=8,action_override=lambda i,t:oracles[i](t))
    for i,g in enumerate(forced):
        assert g['trace_tokens']==cancelled+clean+found
        assert replay(g['trace_tokens'],g['pred'],{a:[b],b:[],c:[]},a,b,tok)['trace_valid']
        assert oracles[i].counts['forced']==1
    # Fields are correct but PUSH's argument is wrong: whole-action replacement.
    oracle=MistakeOracle({a:[b],b:[],c:[]},a,b,tok)
    trace=rec([a],[a],[b],[tok.id('<PUSH>'),c])
    assert oracle(trace)==6 and oracle.stack==[a]
