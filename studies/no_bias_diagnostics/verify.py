"""CPU semantic checks; creates no model, optimizer or checkpoint."""
from pathlib import Path
import os,sys,json
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));os.chdir(ROOT)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies/curriculum_trace_errors')]
import audit
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import context_prompt,render_task
from action_errors import action_replay
tok=Tokenizer()
result=audit.test_validators(tok)
nodes=[tok.node_id(i) for i in range(4)];adj=[[1,2],[3],[],[]]
prompt=context_prompt(tok,nodes,[(0,1),(0,2),(1,3)],0,3)
row=render_task(tok,'dfs_state',1,prompt,adj,0,3,nodes)
prefix=row.tokens[:row.answer_index]
assert not set(prefix)&{tok.label_id(0),tok.label_id(1),tok.id('<EOS>')}
body=row.tokens[row.prompt_len:row.answer_index]
adjtok={nodes[u]:{nodes[v] for v in vs} for u,vs in enumerate(adj)}
assert action_replay(body,adjtok,nodes[0],nodes[3],1,tok,False)['valid']
assert not action_replay(body,adjtok,nodes[0],nodes[3],0,tok,False)['valid']
# Dropping the first candidate annotation invalidates the printed state,
# while the actual action sequence can still execute correctly.
broken=list(body);bars=[i for i,v in enumerate(broken) if v==tok.id('|')]
assert bars[2]>bars[1]+1
del broken[bars[1]+1]
ex=type('Example',(),dict(nodes=nodes,adj=adj,source=0,target=3,label=1))()
assert not audit.full_state_audit(ex,broken,tok,False)['complete_valid']
assert action_replay(broken,adjtok,nodes[0],nodes[3],1,tok,False)['valid']
result.update(no_answer_or_eos_in_oracle_prefix=True,annotation_vs_execution_separated=True,
              oracle_validated_independently=True,optimizer_updates=0)
out=ROOT/'studies/no_bias_diagnostics/verification.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
