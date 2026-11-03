"""Grounded action-only first errors, independent of printed annotations."""
from pathlib import Path
import gzip,json,os,sys
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ContextStore,ParametricWorld

def action_replay(body,adj,src,dst,label,tok,fifo):
    frontier=[src];seen={src};terminal=None;start=0;record=0
    actions={tok.id('<'+a+'>'):a for a in ('PUSH','POP','FOUND','EXHAUST')}
    def fail(kind):return {'valid':False,'first_error':kind,'record':record,'visited_before_error':len(seen)}
    for end,x in enumerate(body):
        if x!=tok.id(';'):continue
        rec=body[start:end];start=end+1
        if terminal:return fail('continued_after_terminal')
        positions=[i for i,v in enumerate(rec) if v in actions]
        if len(positions)!=1:return fail('malformed_action')
        tail=rec[positions[0]:];name=actions[tail[0]]
        if len(tail)!=(2 if name=='PUSH' else 1):return fail('malformed_action')
        cur=frontier[0 if fifo else -1];eligible=set(adj[cur])-seen
        expected='FOUND' if cur==dst else 'PUSH' if eligible else 'POP' if len(frontier)>1 else 'EXHAUST'
        if name!=expected:
            return fail('missed_target' if expected=='FOUND' else 'false_found' if name=='FOUND'
                        else 'premature_exhaust' if name=='EXHAUST' else 'pop_with_unvisited_neighbors'
                        if name=='POP' and eligible else 'wrong_action')
        if name=='PUSH':
            if tail[1] not in adj[cur]:return fail('nonedge_push')
            if tail[1] in seen:return fail('revisited_push')
            seen.add(tail[1]);frontier.append(tail[1])
        elif name=='POP':frontier.pop(0 if fifo else -1)
        else:terminal=name
        record+=1
    if start!=len(body):return fail('unfinished_record')
    if terminal is None:return fail('unfinished_search')
    if terminal!=('FOUND' if label else 'EXHAUST'):return fail('wrong_terminal')
    return {'valid':True,'first_error':None,'visited_before_error':len(seen)}

def main():
    out=ROOT/'studies/no_bias_diagnostics/results'
    checked=0;skipped=0
    for p in sorted(out.glob('*/*.jsonl.gz')):
        meta=json.loads(p.with_suffix('').with_suffix('.json').read_text())
        entry=meta['model']
        if not entry['protocol'].endswith('_state'):continue
        cfg=load_config([ROOT/entry['run']/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
        context=entry['kind']=='context';source=ContextStore(meta['root']) if context else ParametricWorld(meta['root'])
        a=source.arrays if context else source.queries
        with gzip.open(p,'rt') as f:rows=[json.loads(x) for x in f]
        for row in rows:
            if 'trace_body' not in row:skipped+=1;continue
            idx=row['query'];adj=source.adj(idx) if context else source.adj()
            nodes=source.node_vocab_ids(idx,tok) if context else source.node_vocab_ids(tok)
            adjtok={nodes[u]:{nodes[v] for v in vs} for u,vs in enumerate(adj)}
            r=action_replay(row['trace_body'],adjtok,nodes[int(a['s_topo'][idx])],nodes[int(a['t_topo'][idx])],row['label'],tok,entry['protocol']=='bfs_state')
            assert (r['valid'] and not row['stop'])==row['execution_valid']
            row['action_replay']=r;checked+=1
        tmp=p.with_name(p.name+'.tmp')
        with gzip.open(tmp,'wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        tmp.replace(p)
    print('Action-only replays checked:',checked,'skipped missing traces:',skipped)

if __name__=='__main__':main()
