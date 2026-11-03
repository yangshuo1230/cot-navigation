"""Compare checked decoding prefixes to native greedy generation on real models."""
from pathlib import Path
import sys,json
import torch

ROOT=Path.cwd();OUT=ROOT/'studies/fixed100_legality'
sys.path[:0]=[str(ROOT/'src'),str(OUT)]
from replay import StateReplay
from stream import generate_checked
from cache_prefix import enable
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.models.transformer import build_model
from cot_navigation.evaluation.rollout import generate
from cot_navigation.protocols.render import parametric_prompt

torch.set_num_threads(2)
class Limited(StateReplay):
    def check(self,body):
        bad=super().check(body)
        if not bad and len(body)>=512:return self.fail('fidelity_test_cap')
        return bad

results=[]
for setting,base in [('simple','outputs/fixed_short3_v1/launches/short3_n100_s0/all_hops'),('hard','outputs/hard_dag_v1/launches/hard_dag_v1_s0/fixed/n100/stage_b')]:
    for protocol in ('dfs_state','bfs_state'):
        folder=Path(base)/protocol;cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
        world=ParametricWorld(Path(cfg.data.root)/'main-seed0');nodes=world.node_vocab_ids(tok)
        raw=world.adj();adj={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(raw)}
        model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
        blob=torch.load(folder/'checkpoints/best.pt',map_location='cpu',weights_only=False);model.load_state_dict(blob['model']);del blob
        ids=world.indices('test')[:2];header=[tok.id('<WORLD_0>')] if cfg.train.world_roots else []
        prompts=[];checkers=[]
        for i in ids:
            s,t=int(world.queries['s_topo'][i]),int(world.queries['t_topo'][i]);prompts.append(header+parametric_prompt(tok,nodes,s,t))
            checkers.append(Limited(adj,nodes[s],nodes[t],tok,protocol=='bfs_state'))
        original=generate(model,prompts,protocol,tok,cfg.train.precision,max_new_tokens=512,cot_margin=cfg.protocol.cot_margin)
        enable(model)
        checked=generate_checked(model,prompts,tok,cfg.train.precision,checkers,cfg.protocol.cot_margin)
        for a,b in zip(original,checked):
            stop=min(len(a['trace_tokens']),len(b['trace']))
            assert a['trace_tokens'][:stop]==b['trace'][:stop],(setting,protocol)
        results.append(dict(setting=setting,protocol=protocol,queries=len(ids),prefix_matches=True))
        del model;torch.cuda.empty_cache()
Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified/migration/standalone/research_tests/stream_fidelity.json').write_text(json.dumps(results,indent=2));print(results)
