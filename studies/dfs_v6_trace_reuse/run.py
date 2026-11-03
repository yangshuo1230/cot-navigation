"""Audit actual-exposure trajectory reuse; no training or activation dumps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['WANDB_MODE']='disabled'
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
V6=ROOT/'studies/fixed_navigation_protocols_v6'
sys.path[:0]=[str(V6),str(ROOT/'src')]
from packed_bank import PackedBank
from semantics import replay_nodes
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

OUT=Path(__file__).resolve().parent
PANELS=['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']


class Corpus:
    """Exact sequence set, prefix trie, suffix automaton and subsequence table.

    Shared separator is outside the node alphabet, preventing matches across
    training-row boundaries. Every positive exposure count, and only those,
    contributes to the corpus. Repeated training draws do not multiply rows.
    """
    def __init__(self,sequences,n):
        self.seqs=list(set(map(tuple,sequences)))
        self.exact=set(self.seqs)
        self.trie=[{}]
        self.edges=[{}];self.link=[-1];self.length=[0];last=0
        def extend(c):
            nonlocal last
            cur=len(self.edges);self.edges.append({});self.length.append(self.length[last]+1);self.link.append(0)
            p=last
            while p>=0 and c not in self.edges[p]:
                self.edges[p][c]=cur;p=self.link[p]
            if p>=0:
                q=self.edges[p][c]
                if self.length[p]+1==self.length[q]:self.link[cur]=q
                else:
                    clone=len(self.edges);self.edges.append(self.edges[q].copy())
                    self.length.append(self.length[p]+1);self.link.append(self.link[q])
                    while p>=0 and self.edges[p].get(c)==q:
                        self.edges[p][c]=clone;p=self.link[p]
                    self.link[q]=self.link[cur]=clone
            last=cur
        self.positions=np.full((len(self.seqs),n),-1,np.int16)
        self.starts=np.asarray([s[0] for s in self.seqs])
        for i,s in enumerate(self.seqs):
            t=0
            for k,c in enumerate(s):
                if c not in self.trie[t]:
                    self.trie[t][c]=len(self.trie);self.trie.append({})
                t=self.trie[t][c]
                self.positions[i,c]=k
                extend(c)
            extend(-1)

    def compare(self,seq):
        seq=tuple(seq);size=len(seq)
        if not size:return dict(length=0,exact=False,whole_contiguous=False,whole_prefix=False,whole_subsequence=False,
                                longest_prefix_fraction=0.,longest_contiguous_fraction=0.)
        t=0;p=0
        for c in seq:
            if c not in self.trie[t]:break
            t=self.trie[t][c];p+=1
        state=0;l=0;best=0;endings=[]
        for c in seq:
            while state and c not in self.edges[state]:
                state=self.link[state];l=min(l,self.length[state])
            if c in self.edges[state]:
                state=self.edges[state][c];l+=1
            else:state=l=0
            best=max(best,l);endings.append(l)
        pos=self.positions[:,list(seq)]
        matches=(pos>=0).all(1)&(np.diff(pos,axis=1)>0).all(1)
        result=dict(length=size,exact=seq in self.exact,whole_contiguous=best==size,whole_prefix=p==size,
                    whole_subsequence=bool(matches.any()),
                    same_source_subsequence=bool((matches&(self.starts==seq[0])).any()),
                    longest_prefix_fraction=p/size,longest_contiguous_fraction=best/size)
        for k in [4,8,16]:
            result[f'window{k}_coverage']=float(np.mean(np.asarray(endings[k-1:])>=k)) if size>=k else None
        return result


def states(adj,source,target,seq,include_goal=False,include_pops=False):
    """Reconstruct implicit DFS, stopping before any invalid expansion.

    Compare canonical stack order + visited membership, not token order of VIS.
    Exclude the first two expansions and goal states from novelty claims.
    """
    if not seq or seq[0]!=source:return []
    stack=[source];seen={source};mask=1<<source;result=[]
    for k,v in enumerate(seq[1:],1):
        if stack[-1]==target:break
        while stack and not any(w not in seen for w in adj[stack[-1]]):
            stack.pop()
            if include_pops and stack:result.append((tuple(stack),mask))
        if not stack or v in seen or v not in adj[stack[-1]]:break
        stack.append(v);seen.add(v);mask|=1<<v
        if k>=2 and (v!=target or include_goal):result.append((tuple(stack),mask))
    if include_pops and stack and stack[-1]!=target:
        while len(stack)>1 and not any(w not in seen for w in adj[stack[-1]]):
            stack.pop();result.append((tuple(stack),mask))
    return result


def average(rows):
    if not rows:return dict(n=0)
    out={'n':len(rows)}
    for key in rows[0]:
        if key.startswith('_'):continue
        values=[r[key] for r in rows if r.get(key) is not None]
        if values:out[key]=float(np.mean(values))
    union=set().union(*(r['_states'] for r in rows))
    new=set().union(*(r['_new_states'] for r in rows))
    out.update(unique_states=len(union),unique_novel_states=len(new))
    return out


def self_test():
    c=Corpus([[0,1,2],[2,3]],5)
    assert c.compare([0,1,2])['exact']
    assert c.compare([1,2])['whole_contiguous']
    assert not c.compare([1,2,3])['whole_contiguous']
    assert c.compare([0,2])['whole_subsequence']
    assert not c.compare([2,1])['whole_subsequence']
    # Validate automaton against brute force, including repeated/unknown nodes.
    rng=np.random.default_rng(31)
    for _ in range(100):
        seq=rng.integers(0,5,size=5).tolist()
        brute=max([0]+[j-i for i in range(5) for j in range(i+1,6)
                       if any(tuple(seq[i:j])==s[k:k+j-i] for s in c.seqs for k in range(len(s)-j+i+1))])
        assert abs(c.compare(seq)['longest_contiguous_fraction']-brute/5)<1e-9


def main():
    self_test();torch.set_num_threads(2)
    manifest=json.loads((V6/'manifest.json').read_text())
    meta=next(m for m in manifest['worlds'] if m['name']=='n80_g0')
    bank=PackedBank(meta,'dfs_nodes',torch.device('cpu'))
    world=bank.world;n=world['n_nodes'];tok=Tokenizer(256)
    tokens=[tok.node_id(i) for i in world['node_names']];inverse={t:i for i,t in enumerate(tokens)}
    doc={'audit_version':2,'setting':'v6 N80 graph0; seeds0..3; final50k; actual exposure only',
         'generation':'compiled BF16; batch128; original native cap; GPU0; no graph constraints',
         'state_definition':'ordered implicit DFS stack + visited membership; exclude first2 and goal states',
         'self_test':'suffix automaton vs brute force, row boundaries, subsequence controls PASS','seeds':[]}
    path=OUT/'results.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('audit_version')==2:doc=previous
    for seed in range(4):
        if any(s['seed']==seed for s in doc['seeds']):continue
        run=ROOT/f'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed}/dfs_nodes'
        counts=json.loads((run/'exposure_counts.json').read_text())
        train=[];trainstates=set();used=[]
        trainqueries=set(world['panels']['short_train'])|set(world['panels']['long_train'])
        for row,key in enumerate(bank.keys):
            if not key.startswith('q:') or counts.get(key,0)<=0:continue
            qid=int(bank.cpu['qids'][row]);assert qid in trainqueries
            a=int(bank.cpu['offsets'][row])+4;b=int(bank.cpu['offsets'][row])+int(bank.cpu['answers'][row])
            seq=[inverse[int(v)] for v in bank.cpu['tokens'][a:b]]
            r=world['records'][qid]
            assert replay_nodes(world['adj'],r['s'],r['t'],[tokens[v] for v in seq],inverse)['valid']
            train.append(seq);used.append((row,qid))
            # Include terminal states in the training bank: a state previously
            # seen at another query's goal must not be mislabeled as novel.
            trainstates.update(states(world['adj'],r['s'],r['t'],seq,include_goal=True,include_pops=True))
        corpus=Corpus(train,n)
        print('CORPUS',seed,len(train),len(corpus.seqs),len(trainstates),flush=True)
        blob=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        assert blob['step']==50000
        model=TransformerLM(len(tok),**blob['job']['model'],dropout=0.).to('cuda:0')
        model.load_state_dict(blob['model']);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        compiled=torch.compile(model,dynamic=True)
        original=json.loads((run/'final.json').read_text())
        result=dict(seed=seed,checkpoint_sha256=hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest(),
                    actual_training_variants=len(train),unique_training_trajectories=len(corpus.seqs),
                    unique_training_states=len(trainstates),panels={},examples=[])
        for name in PANELS:
            ids=world['panels'][name]
            assert not(set(ids)&trainqueries)
            generated=[];refs=[];checks=Counter();errors=Counter()
            for at in range(0,len(ids),128):
                qids=ids[at:at+128]
                prompts=[[tok.id('<QUERY>'),tokens[world['records'][q]['s']],tokens[world['records'][q]['t']],tok.id('|')] for q in qids]
                outputs=generate(compiled,prompts,'dfs_nodes',tok,'bf16',max_new_tokens=bank.info['max_new_tokens'])
                for qid,out in zip(qids,outputs):
                    r=world['records'][qid]
                    semantic=replay_nodes(world['adj'],r['s'],r['t'],out['trace_tokens'],inverse)
                    legal=semantic['valid'] and not out['generation_limit_reached']
                    errors[f"y{r['y']}/{semantic['error'] or 'valid'}"]+=1
                    checks['n']+=1;checks['trace_valid']+=legal
                    checks['correct']+=out['answer_token']==tok.label_id(r['y'])
                    checks[f"correct{r['y']}"]+=out['answer_token']==tok.label_id(r['y'])
                    checks['truncated']+=out['generation_limit_reached']
                    seq=[inverse.get(t,n) for t in out['trace_tokens']]
                    # Off-graph tokens cannot match training sequences or be replayed as nodes.
                    if n in seq:
                        rr=dict(length=len(seq),exact=False,whole_contiguous=False,whole_prefix=False,
                                whole_subsequence=False,same_source_subsequence=False,
                                longest_prefix_fraction=None,longest_contiguous_fraction=None,
                                window4_coverage=None,window8_coverage=None,window16_coverage=None)
                        ss=states(world['adj'],r['s'],r['t'],seq[:seq.index(n)])
                    else:
                        rr=corpus.compare(seq);ss=states(world['adj'],r['s'],r['t'],seq)
                    new=set(ss)-trainstates
                    rr.update(legal=bool(legal),has_novel_state=bool(new),
                              novel_state_and_not_subsequence=bool(new) and not rr['whole_subsequence'],state_occurrences=len(ss),
                              novel_state_occurrences=sum(s not in trainstates for s in ss),
                              _states=set(ss),_new_states=new)
                    generated.append(rr)
                    if legal and new and not rr['whole_subsequence'] and sum(e['panel']==name for e in result['examples'])<2:
                        example=dict(panel=name,query=qid,source=tok.name(tokens[r['s']]),target=tok.name(tokens[r['t']]),
                                     label=r['y'],trace=[tok.name(tokens[u]) for u in seq],
                                     novel_states=len(new),whole_subsequence=False,
                                     longest_contiguous_fraction=rr['longest_contiguous_fraction'])
                        stack,vis=next(iter(sorted(new)))
                        example['novel_stack']=[tok.name(tokens[u]) for u in stack]
                        example['novel_visited']=[tok.name(tokens[u]) for u in range(n) if vis>>u&1]
                        result['examples'].append(example)
                    for row in range(int(bank.qptr[qid]),int(bank.qptr[qid+1])):
                        a=int(bank.cpu['offsets'][row])+4;b=int(bank.cpu['offsets'][row])+int(bank.cpu['answers'][row])
                        refseq=[inverse[int(t)] for t in bank.cpu['tokens'][a:b]]
                        rf=corpus.compare(refseq);rs=states(world['adj'],r['s'],r['t'],refseq)
                        rn=set(rs)-trainstates
                        rf.update(legal=True,has_novel_state=bool(rn),state_occurrences=len(rs),
                                  novel_state_occurrences=sum(s not in trainstates for s in rs),_states=set(rs),_new_states=rn)
                        refs.append(rf)
            old=original['panels'][name]
            expected=old['metrics']['all']
            diff={k:dict(new=v,old=expected.get(k,0)) for k,v in checks.items() if v!=expected.get(k,0)}
            same_errors=dict(errors)==old['errors']
            result['panels'][name]=dict(generated=average(generated),legal_generated=average([r for r in generated if r['legal']]),
                                      reference=average(refs),native_checks=dict(checks),original_count_differences=diff,
                                      original_error_counts_match=same_errors)
            print('PANEL',seed,name,'legal',checks['trace_valid'],'exact',sum(r['exact'] for r in generated),
                  'new legal',sum(r['legal'] and r['has_novel_state'] for r in generated),'diff',diff,flush=True)
        doc['seeds'].append(result)
        path.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        del compiled,model,blob,corpus
        torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':main()
