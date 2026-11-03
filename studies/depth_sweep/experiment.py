"""Clean in-context DFS state/node depth comparison using native long curriculum."""
import os,sys,json,argparse,dataclasses,multiprocessing as mp
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.experiments import scheduler
from cot_navigation.experiments.fine_node_curriculum import FineNodeTrainer
from cot_navigation.training.trainer import Trainer as NativeTrainer
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.prepared import PreparedRows,_preparation_lock
from cot_navigation.data.scale_prepared import identity,prepare_variants
from cot_navigation.data.shards import is_complete,load_npz,file_sha256,atomic_savez,atomic_json,mark_complete
from cot_navigation.protocols.render import truncate_example
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.trace_audit import validate_trace

ROOT=Path(__file__).resolve().parents[2];NAME='depth_sweep_4096_20261101';HERE=ROOT/'studies/depth_sweep'
OUT=ROOT/'outputs'/NAME;GEN=ROOT/'configs/generated'/NAME

def rows_for(store,cfg,tok,verify=False):
    spec=identity(store,cfg,tok)
    own=OUT/'derived'/store.root.name/(cfg.protocol.name+'.npz')
    if own.exists() and (not verify or is_complete(own)):
        metadata=json.loads(own.with_suffix('.json').read_text())
        if metadata['identity']==spec:
            obj=PreparedRows.__new__(PreparedRows);obj.protocol=cfg.protocol.name;obj.arrays=load_npz(own);obj.source_path=str(own);return obj
    for metadata in sorted((store.root/'prepared').glob(f'scale_{cfg.protocol.name}_*.json')):
        old=json.loads(metadata.read_text())
        if any(old.get(k)!=v for k,v in spec.items() if k!='max_seq_len'):continue
        archive=metadata.with_suffix('.npz')
        if not archive.exists() or (verify and not is_complete(archive)):continue
        arrays=load_npz(archive)
        if np.any(arrays['tokens']==tok.id('<STOP>')):continue
        if int(np.diff(arrays['offsets']).max())>cfg.model.max_seq_len-cfg.protocol.cot_margin:continue
        obj=PreparedRows.__new__(PreparedRows);obj.protocol=cfg.protocol.name;obj.arrays=arrays;obj.source_path=str(archive)
        return obj
    raise FileNotFoundError(f'Missing complete {cfg.protocol.name} variants {store.root}')

def derive(full,cfg,tok,path,store):
    arrays=full.arrays;lengths=np.diff(arrays['offsets']);long=np.flatnonzero(lengths>cfg.model.max_seq_len-cfg.protocol.cot_margin)
    result=arrays
    if len(long):
        chunks=[];offsets=[0];fields=arrays['fields'].copy();metadata=[str(x) for x in arrays['metadata']];longset=set(map(int,long))
        for i in range(len(lengths)):
            if i in longset:
                query=int(np.searchsorted(arrays['query_offsets'],i,side='right')-1)
                variant=i-int(arrays['query_offsets'][query]);row=full.row(query,variant)
                row=truncate_example(row,tok,cfg.model.max_seq_len,cfg.protocol.cot_margin)
                chunk=np.asarray(row.tokens,dtype=np.int32);fields[i]=[row.prompt_len,row.answer_index,row.label,row.filler_tokens]
                metadata[i]=json.dumps(row.meta,separators=(',',':'))
            else:chunk=arrays['tokens'][arrays['offsets'][i]:arrays['offsets'][i+1]]
            chunks.append(chunk);offsets.append(offsets[-1]+len(chunk))
        result=dict(tokens=np.concatenate(chunks),offsets=np.asarray(offsets,dtype=np.int64),fields=fields,
                    metadata=np.asarray(metadata,dtype=str),query_offsets=arrays['query_offsets'])
    assert int(np.diff(result['offsets']).max())<=4088
    atomic_savez(path,**result);atomic_json(path.with_suffix('.json'),dict(identity=identity(store,cfg,tok),
        policy='native_record_boundary_STOP',full_source=full.source_path,truncated_rows=len(long),rows=len(lengths)))
    mark_complete(path,file_sha256(path))

def configure():
    import yaml
    GEN.mkdir(parents=True,exist_ok=True);jobs=[];preps=[]
    base=ROOT/'configs/generated/fine_node_curriculum_20261028'
    for family in ['hard','simple']:
        for seed in [0,1]:
            for depth in [3,6,9,12]:
                for protocol in ['dfs_nodes','dfs_state']:
                    cfg=yaml.safe_load((base/f's{seed}__dfs_nodes__{family}.yaml').read_text())
                    name=f'{family}/{protocol}/L{depth}/s{seed}'
                    cfg['model'].update(layers=depth,max_seq_len=4096)
                    cfg['protocol'].update(name=protocol,enable_mistake=False,mistake_rate=0,mistake_source='synthetic')
                    cfg['output_dir']=str(OUT/name)
                    cfg['tracking'].update(mode='offline',group=NAME,name=NAME+'/'+name,variant=f'L{depth}-{family}-{protocol}',
                        notes='Clean full-state DFS versus first-expansion nodes; depth 3/6/9/12; two model seeds; shared 4096 context; nine-stage 1M curriculum; native record-boundary STOP truncation when needed.')
                    path=GEN/(name.replace('/','__')+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_config([path],[])
                    jobs.append(dict(name=name,config=str(path),family=family,protocol=protocol,depth=depth,seed=seed))
        template=yaml.safe_load((base/f's0__dfs_nodes__{family}.yaml').read_text())
        for i,root in enumerate(template['train']['scale_roots']):
            n=[12,16,24,32,36,48,64,80,100][i]
            cfg=yaml.safe_load((base/f'data_{family}_n{n}.yaml').read_text())
            cfg['model']['max_seq_len']=4096
            path=GEN/f'data_{family}_n{n}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_config([path],[]);preps.append(str(path))
    scheduler.write_json(HERE/'manifest.json',dict(jobs=jobs,preparations=preps))
    print('Configured 32 models and 18 shared datasets',flush=True)

def prepare_one(config):
    torch.set_num_threads(2);cfg=load_config([config],[]);tok=Tokenizer.from_config(cfg)
    with _preparation_lock(Path(cfg.data.root)):
        store=ContextStore(cfg.data.root);results=[]
        for protocol in ['dfs_nodes','dfs_state']:
            c=dataclasses.replace(cfg,protocol=dataclasses.replace(cfg.protocol,name=protocol))
            if protocol=='dfs_state':
                fullcfg=dataclasses.replace(c,model=dataclasses.replace(c.model,max_seq_len=32768))
                full=rows_for(store,fullcfg,tok,verify=True)
                own=OUT/'derived'/store.root.name/(protocol+'.npz')
                if not is_complete(own):derive(full,c,tok,own,store)
                del full
            rows=rows_for(store,c,tok,verify=True)
            checked=0
            for split in ['train','val','test']:
                ids=store.indices(split)
                for index in ids[np.linspace(0,len(ids)-1,8,dtype=int)]:
                    index=int(index);nodes=store.node_vocab_ids(index,tok);adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(store.adj(index))}
                    src=nodes[int(store.arrays['s_topo'][index])];dst=nodes[int(store.arrays['t_topo'][index])]
                    for variant in range(rows.variants(index)):
                        row=rows.row(index,variant);trace=row.tokens[row.prompt_len:row.answer_index]
                        result=validate_trace(trace,row.label,protocol,adj,src,dst,tok)
                        if tok.id('<STOP>') in trace:
                            assert protocol=='dfs_state' and trace[-1]==tok.id('<STOP>') and trace[-2]==tok.id(';')
                            prefix=validate_trace(trace[:-1],row.label,protocol,adj,src,dst,tok)
                            assert prefix.get('error')=='missing_or_inconsistent_terminal',prefix
                        else:assert result['valid'],(store.root,protocol,index,result)
                        checked+=1
            results.append(dict(protocol=protocol,cache=rows.source_path,sha256=file_sha256(rows.source_path),
                                max_tokens=int(np.diff(rows.arrays['offsets']).max()),reference_checks=checked,truncated_rows=int(np.count_nonzero(rows.arrays['tokens']==tok.id('<STOP>')))))
            del rows
        print('Prepared/reused',cfg.data.root,flush=True)
        return dict(root=cfg.data.root,fingerprint=store.fingerprint,caches=results)

def prepare_all(jobs,output,env):
    receipt=output/'data_ready.json'
    if receipt.exists():return
    manifest=json.loads((HERE/'manifest.json').read_text())
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(prepare_one,manifest['preparations']))
    scheduler.write_json(receipt,results)
    cleanup=HERE/'superseded_cache_cleanup.json'
    removed=[]
    if cleanup.exists():
        for name in json.loads(cleanup.read_text()):
            path=Path(name).resolve();assert path.is_relative_to(ROOT/'data') and path.name.startswith('scale_dfs_state_')
            for target in [path,path.with_suffix('.json'),Path(str(path)+'.complete')]:
                if target.exists():target.unlink();removed.append(str(target))
    scheduler.write_json(output/'superseded_cache_cleanup_done.json',removed)

class Trainer(FineNodeTrainer):
    def _preload_member(self,member):member.prepared_tasks=rows_for(member.store,self.cfg,self.tok)
    def _render_task(self,index,*,path_index=0):
        row=super()._render_task(index,path_index=path_index)
        if len(self.render_cache)>2048:self.render_cache.pop(next(iter(self.render_cache)))
        return row
    def _free(self,rows,*,record_name=None):
        report=NativeTrainer._free(self,rows,record_name=record_name)
        if not self.persist:return report
        name=record_name or f'val_{self.completed:08d}'
        records=[json.loads(x) for x in (self.out/'eval'/f'predictions_{name}.jsonl').read_text().splitlines()]
        valid=positive=posvalid=joint=0;errors=Counter()
        for r in records:
            prompt=r['prompt_tokens'];q=prompt.index(self.tok.id('<QUERY>'));adj={self.tok.node_id(i):[] for i in range(256)}
            for i in range(0,q,3):adj[prompt[i+1]].append(prompt[i+2])
            result=validate_trace(r['trace_tokens'],r['label'],self.cfg.protocol.name,adj,prompt[q+1],prompt[q+2],self.tok)
            valid+=result['valid'];positive+=r['label']==1;posvalid+=result['valid'] and r['label']==1;joint+=result['valid'] and r['correct']
            if not result['valid']:errors[result['error']]+=1
        report['semantic']=dict(n=len(records),n_positive=positive,valid_rate=valid/len(records),
                                positive_valid_rate=posvalid/max(1,positive),joint_rate=joint/len(records),errors=dict(errors))
        return report

@dataclasses.dataclass
class Job(scheduler.Job):
    def command(self,checkpoint=None,parent=None):
        argv=[sys.executable,str(HERE/'experiment.py'),'--worker',self.name]
        if checkpoint:argv+=['--checkpoint',str(checkpoint)]
        return argv

def main():
    p=argparse.ArgumentParser();p.add_argument('--configure',action='store_true');p.add_argument('--worker');p.add_argument('--checkpoint')
    p.add_argument('--run-name',default=NAME);p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    for flag in ['execute','wait','background','resume']:p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.configure:configure();return
    specs=json.loads((HERE/'manifest.json').read_text())['jobs']
    if a.worker:
        j=next(j for j in specs if j['name']==a.worker);cfg=load_config([j['config']],[])
        with scheduler.file_lock(Path(cfg.output_dir)/'.worker.lock',blocking=False):
            t=Trainer(cfg,resume=a.checkpoint);t.fit();scheduler.write_json(t.out/'COMPLETE.json',dict(step=t.completed))
        return
    jobs=[Job(j['name'],Path(j['config']),[],Path(load_config([j['config']],[]).output_dir),1000000) for j in specs]
    a.dry_run=not a.execute
    scheduler.execute(a,OUT,jobs,sys.argv[1:],script_path=HERE/'experiment.py',data_preparer=prepare_all)

if __name__=='__main__':main()
