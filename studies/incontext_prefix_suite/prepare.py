"""Audit existing data, resolve 39 immutable configurations; no training."""
from pathlib import Path
import sys,json,hashlib,copy
HERE=Path(__file__).resolve().parent
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
sys.path.insert(0,str(HERE/'src'))
import yaml,numpy as np
from cot_navigation.config import load_config,to_plain
from cot_navigation.data.dataset import ContextStore
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.shards import file_sha256,is_complete
from runtime import read_prepared

ROOTS=[f'data/scale_v1/context_l{l}w{w}_n{n}_m{m}_train50000_s0' for l,w,n,m in [(4,4,16,24),(6,6,36,54),(8,8,64,96),(10,10,100,150)]]+[
    f'data/hard_dag_v1/context_r{l}b{w}_n{n}_m{m}_train50000_s0' for l,w,n,m in [(4,4,16,32),(6,6,36,72),(8,8,64,128),(10,10,100,200)]]
OOD=dict(simple144='data/scale_v1/context_l12w12_n144_m216_train2_s0',hard144='data/hard_dag_v1/context_r12b12_n144_m288_s0')
BASE=ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/path/resolved_config.yaml'
def config(job):
    boot=550000 if job['protocol']=='direct' else 0
    ends=[boot+x for x in (40000,80000,120000,160000,200000,240000,280000,360000)]
    weights=[float(i==j) for i in range(8) for j in range(8)]
    return load_config([BASE,ROOT/'configs/train/throughput.yaml'],[
        f'output_dir={ROOT/"outputs/incontext_prefix_v1"/job["id"]}',
        'data.include_one_hop_queries=true',f'model.max_seq_len=4096','model.architecture=standard','model.heads=1',
        f'protocol.name={job["protocol"]}','protocol.trajectory_variants=4',f'protocol.negative_random_path={str(job["protocol"]=="path").lower()}',
        f'train.model_seed={job["seed"]}',f'train.exposure_seed={job["seed"]}',f'train.max_steps={boot+360000}',
        'train.lr=0.0001','train.min_lr=0.00001','train.warmup_steps=1000',
        'train.effective_query_batch_size=64','train.bucket_multiple=16','train.eval_samples=256','train.free_gen_samples=256',
        'train.trace_loss_lambda=1','train.shuffle_context_edges=true','train.query_hops=all',
        'train.scale_roots='+json.dumps(ROOTS),'train.scale_stage_ends='+json.dumps(ends),
        'train.scale_stage_weights='+json.dumps(weights),'train.scale_weights=[]','train.scale_walk_weights=[]',
        'evaluation.batch_size=64','evaluation.test_samples=512','evaluation.final_checkpoints=[last]',
        'evaluation.selection_min_step=10000000','evaluation.save_trace_tokens=true','evaluation.trace_diagnostics=false',
        'tracking.mode=online','tracking.online_failure=error','tracking.group=incontext_prefix_v1_20261024',
        'tracking.entity=yangshuo1230-sjtu-hpc-center',f'tracking.name={job["id"]}',
        'tracking.notes=Isolated snapshot of optimized pipeline; exact replay quotas; graph-only vs graph+query prefix; no output leakage.'])

def main():
    import os
    os.chdir(ROOT);(HERE/'configs').mkdir(exist_ok=True)
    cells=[]
    for recipe in ('standard','stepwise'):
        for mask in ('causal','graph','query'):cells.append(dict(protocol='direct',recipe=recipe,mask=mask,order='single'))
    for mask,order in [('causal','single'),('graph','single'),('query','single'),('graph','parallel'),('query','parallel'),('graph','serial'),('query','serial')]:
        cells.append(dict(protocol='path',recipe='standard',mask=mask,order=order))
    jobs=[]
    # Dispatch a mix of protocols from seed 0 first, then all remaining repeats.
    order=[0,1,2,6,7,8,9,10,11,12,3,4,5]
    for seed in range(3):
        for k in order:
            j=dict(**cells[k],seed=seed);j['id']=f"s{seed}/{j['protocol']}_{j['recipe']}/{j['mask']}_{j['order']}"
            cfg=config(j);p=HERE/'configs'/(j['id'].replace('/','__')+'.yaml');p.write_text(yaml.safe_dump(to_plain(cfg),sort_keys=False))
            j['config']=str(p);j['updates']=cfg.train.max_steps;jobs.append(j)
    audits=[];tok=Tokenizer.from_config(config(jobs[0]))
    for root in ROOTS+list(OOD.values()):
        store=ContextStore(root);entry=dict(root=root,fingerprint=store.fingerprint,splits={},prepared={})
        rng=np.random.default_rng(783)
        for split in ('train','val','test'):
            ids=store.indices(split);entry['splits'][split]=len(ids)
            for i in rng.choice(ids,min(32,len(ids)),replace=False):
                adj=store.adj(int(i));s,t=int(store.arrays['s_topo'][i]),int(store.arrays['t_topo'][i]);seen={s};todo=[s]
                for u in todo:
                    for v in adj[u]:
                        if v not in seen:seen.add(v);todo.append(v)
                assert (t in seen)==bool(store.arrays['label'][i])
        for protocol in ('direct','path'):
            cfg=config(dict(jobs[0],protocol=protocol));prepared=read_prepared(store,cfg,tok)
            entry['prepared'][protocol]=dict(path=getattr(prepared,'source_path','canonical'),
                max_length=int(np.diff(prepared.arrays['offsets']).max()),n_rows=len(prepared.arrays['fields']))
            source_path=getattr(prepared,'source_path',None)
            if source_path:
                marker=Path(source_path+'.complete');expected=json.loads(marker.read_text()) if marker.read_text().lstrip().startswith('{') else None
                entry['prepared'][protocol]['sha256']=file_sha256(source_path)
        audits.append(entry)
    boot_hashes={}
    for key in ['fixed_source_v16_m2','full_v16_m2','full_v64_m2','full_v256_m2','full_v256_m4','full_v256_m8','full_v256_m16','full_v256_m24']:
        p=ROOT/'studies/direct_learning/data_v1'/(key+'.npz');boot_hashes[key]=file_sha256(p)
        with np.load(p) as data:
            seen=[]
            for split in ('train','val','test'):
                graphs=set()
                for prompt,label in zip(data[split+'_x'],data[split+'_y']):
                    edges=[tuple(map(int,e[1:])) for e in prompt[:-4].reshape(-1,3)];s,t=map(int,prompt[-3:-1])
                    assert int((s,t) in edges)==int(label)
                    graphs.add(tuple(sorted(edges)))
                seen.append(graphs)
            assert not(seen[0]&seen[1] or seen[0]&seen[2] or seen[1]&seen[2])
    manifest=dict(group='incontext_prefix_v1_20261024',jobs=jobs,roots=ROOTS,ood=OOD,data_audit=audits,boot_hashes=boot_hashes,
        data_labels_independently_verified=True,bootstrap_splits_graph_disjoint=True,total_updates=sum(j['updates'] for j in jobs),
        main_steps=360000,direct_bootstrap_steps=550000,status='prepared_not_launched')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(jobs=len(jobs),updates=manifest['total_updates'],data_roots=len(audits)),indent=2))
if __name__=='__main__':main()
