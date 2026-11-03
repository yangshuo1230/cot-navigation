"""True-graph completed-record oracle; supplies cancellation, never gold repair."""
from collections import Counter


class MistakeOracle:
    def __init__(self,adj,src,dst,tok):
        self.adj,self.dst,self.tok=adj,dst,tok
        self.stack=[src];self.visited={src};self.start=0;self.disabled=False;self.terminal=False
        self.counts=Counter();self.force_positions=[]

    def __call__(self,trace):
        if self.disabled:return None
        tok=self.tok;begin=self.start;body=trace[begin:-1];bars=[j for j,t in enumerate(body) if t==tok.id('|')]
        if len(bars)<3 or self.terminal:
            self.disabled=True;self.counts['unsupported']+=1;return None
        cut=begin+bars[2]+1
        groups=[body[:bars[0]],body[bars[0]+1:bars[1]],body[bars[1]+1:bars[2]],body[bars[2]+1:]]
        stack,visited=self.stack,self.visited;cur=stack[-1];eligible=set(self.adj.get(cur,[]))-visited
        fields=(len(bars)==3 and groups[0]==stack and set(groups[1])==visited and len(groups[1])==len(set(groups[1])) and set(groups[2])==eligible and len(groups[2])==len(set(groups[2])))
        action=groups[3]
        if action==[tok.id('<MISTAKE>')]:
            self.start=len(trace);self.counts['cancel_records']+=1;return None
        wanted='<FOUND>' if cur==self.dst else '<PUSH>' if eligible else '<POP>' if len(stack)>1 else '<EXHAUST>'
        legal=(len(action)==(2 if wanted=='<PUSH>' else 1) and action[0]==tok.id(wanted) and
               (wanted!='<PUSH>' or action[1] in eligible and bool(groups[2]) and action[1]==groups[2][0]))
        if not fields or not legal:
            self.counts['forced']+=1;self.counts['bad_fields']+=not fields;self.counts['bad_action']+=not legal
            self.force_positions.append(cut);return cut
        if wanted=='<PUSH>':stack.append(action[1]);visited.add(action[1])
        elif wanted=='<POP>':stack.pop()
        else:self.terminal=True
        self.start=len(trace);return None
