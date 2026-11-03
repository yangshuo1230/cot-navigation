"""Query/adjacency sample encoding and exact token-budget packing."""
class Samples:
    def __init__(self, world, protocol):
        from cot_navigation.protocols.tokenizer import Tokenizer
        from cot_navigation.protocols.render import RenderedExample
        self.world=world; self.protocol=protocol; self.tok=Tokenizer(256)
        self.nodes=[self.tok.node_id(n) for n in world['node_names']]
        self.row_type=RenderedExample; self.cache={}
        self.transition_cache={}
        self.adjacency_packing={}

    def query(self, record):
        from cot_navigation.protocols.render import render_task,parametric_prompt
        key=tuple(record)
        if key not in self.cache:
            s,t,y,gap=record
            row=render_task(self.tok,self.protocol,y,
                parametric_prompt(self.tok,self.nodes,s,t),self.world['adj'],s,t,self.nodes)
            self.cache[key]=row
        return self.cache[key]

    def knowledge(self, path):
        key=tuple(path)
        if key not in self.transition_cache:
            if path[0]=='adj':
                source=path[1]
                successors=sorted(self.world['adj'][source],key=lambda v:self.nodes[v])
                body=[self.nodes[v] for v in successors] or [self.tok.id('<EXHAUST>')]
                prompt=[self.tok.id('<EDGE>'),self.nodes[source],self.tok.id('|')]
                tokens=prompt+body+[self.tok.id('<EOS>')]
                self.transition_cache[key]=self.row_type(tokens,len(prompt),-1,[],
                    list(range(len(prompt),len(tokens))),-1,
                    meta=dict(knowledge_kind='adjacency',source=source,successors=successors))
            else:
                # Only R2/R3's additional path exposure uses the historical WALK format.
                tokens=[self.tok.id('<WALK>')]+[self.nodes[v] for v in path]+[self.tok.id('<EOS>')]
                self.transition_cache[key]=self.row_type(tokens,2,-1,[],list(range(2,len(path)+1)),-1,
                    meta=dict(knowledge_kind='auxiliary_walk'))
        return self.transition_cache[key]

    def adjacency_batch(self,rng,budget):
        # Pack complete rows into exactly the original target-token budget.
        # A table is never truncated to fit, including its explicit EMPTY/EOS.
        if budget not in self.adjacency_packing:
            costs=[max(1,len(vs))+1 for vs in self.world['adj']]
            possible=[True]+[False]*budget
            for amount in range(1,budget+1):
                possible[amount]=any(c<=amount and possible[amount-c] for c in set(costs))
            if not possible[budget]:raise ValueError('Cannot pack complete adjacency rows into budget')
            choices={amount:[u for u,c in enumerate(costs) if c<=amount and possible[amount-c]]
                     for amount in range(2,budget+1)}
            self.adjacency_packing[budget]=(costs,choices)
        costs,choices=self.adjacency_packing[budget]
        remaining=budget;rows=[]
        while remaining:
            source=rng.choice(choices[remaining])
            rows.append(('adj',source));remaining-=costs[source]
        return rows

    def knowledge_batch(self,rng,arm,budget=64):
        if arm=='R0':return self.adjacency_batch(rng,budget)
        local_budget=min(16,budget)
        paths=self.adjacency_batch(rng,local_budget)
        pool=self.world['fragments'][arm]
        remaining=budget-local_budget
        while remaining:
            path=rng.choice(pool)
            path=path[:remaining+1]
            paths.append(path)
            remaining-=len(path)-1
        return paths
