"""Materialize a native unified-framework curriculum, without training."""
import json
from pathlib import Path
import yaml

ROOT = Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
NAME = 'fine_node_curriculum_20261028'
LEVELS = [(12,3,4),(16,4,4),(24,4,6),(32,4,8),(36,6,6),
          (48,6,8),(64,8,8),(80,8,10),(100,10,10)]

def main():
    from cot_navigation.config import load_config
    generated = ROOT/'configs/generated'/NAME
    generated.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT/'studies/fine_node_curriculum/base.yaml').read_text())
    roots = {}
    preparations = []
    for family in ['simple','hard']:
        roots[family] = []
        for n,l,w in LEVELS:
            m = n*3//2 if family == 'simple' else n*2
            prefix,shape = ('scale_v1','l') if family == 'simple' else ('hard_dag_v1','r')
            width = 'w' if family == 'simple' else 'b'
            root = f'data/{prefix}/context_{shape}{l}{width}{w}_n{n}_m{m}_train50000_s0'
            roots[family].append(root)
            cfg = json.loads(json.dumps(base))
            cfg['data'].update(n_nodes=n,n_edges=m,shapes={'main':dict(layers=l,width=w)},
                               graph_family='layered' if family=='simple' else 'multiscale_dag_v1',
                               root=root,workers=4)
            existing=ROOT/root/'manifest.json'
            if existing.is_file():
                # Preserve the complete immutable identity, including fields
                # unused by context generation but bound by its fingerprint.
                cfg['data']=json.loads(existing.read_text())['config']
                cfg['data'].update(root=root,workers=4)
            cfg['train'].update(scale_roots=[],scale_stage_ends=[],scale_stage_weights=[])
            cfg['evaluation'].update(extra_sets=[],final_checkpoints=['last'])
            path = generated/f'data_{family}_n{n}.yaml'
            path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            load_config([path],[])
            preparations.append(str(path.relative_to(ROOT)))
    jobs = []
    for seed in range(3):
        for protocol in ['bfs_nodes','dfs_nodes']:
            # Prioritize difficult-graph replications in the GPU queue.
            for family in ['hard','simple']:
                cfg = yaml.safe_load((generated/f'data_{family}_n100.yaml').read_text())
                ident = f's{seed}/{protocol}/{family}'
                cfg['output_dir'] = str(ROOT/'outputs'/NAME/ident)
                ends = [100000*i for i in range(1,9)]+[1000000]
                weights = []
                for i in range(len(LEVELS)):
                    weights += [1.0 if j == 0 else 0.0 for j in range(len(LEVELS))] if i == 0 else [0.3/i if j<i else 0.7 if j==i else 0.0 for j in range(len(LEVELS))]
                cfg['protocol']['name'] = protocol
                cfg['train'].update(max_steps=1000000,model_seed=seed,
                                    scale_roots=roots[family],scale_stage_ends=ends,
                                    scale_stage_weights=weights,keep_last=1,
                                    eval_interval=20000,free_eval_interval=20000,
                                    save_interval=10000,log_interval=100)
                cfg['evaluation'].update(test_samples=512,extra_sets=[],final_checkpoints=['last'],
                                         selection_min_step=1000001,save_trace_tokens=True,
                                         compile_generation=True)
                cfg['tracking'].update(group=NAME,name=NAME+'/'+ident,
                                       variant='fine-fixed-budget-'+family+'-'+protocol,
                                       notes='Nine fixed stages, 100k each except N100 200k; 70/30 replay. No validation-based stage gating or test selection.',
                                       tags=['fine_node_curriculum','unified','causal',family,protocol])
                path=generated/(ident.replace('/','__')+'.yaml')
                path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_config([path],[])
                jobs.append(dict(id=ident,config=str(path.relative_to(ROOT)),seed=seed,
                                 protocol=protocol,family=family))
    manifest=dict(name=NAME,levels=[n for n,_,_ in LEVELS],stage_ends=ends,
                  roots=roots,preparations=preparations,jobs=jobs)
    (generated/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(configs=len(jobs),data_roots=len(preparations),levels=manifest['levels'],steps=1000000)))

if __name__=='__main__':main()
