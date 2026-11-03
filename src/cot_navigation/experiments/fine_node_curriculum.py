"""Fine DFS/BFS-node curriculum using the native data, Trainer and scheduler.

Only experiment hooks live here: compact cache reuse, stage-boundary evaluation,
semantic rollout reporting and retention validation. Core framework unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

import numpy as np
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.prepared import PreparedRows, _preparation_lock
from cot_navigation.data.provenance import prepare_generation
from cot_navigation.data.scale_prepared import identity, prepare_variants
from cot_navigation.data.shards import is_complete, load_npz, file_sha256
from cot_navigation.data.generate import generate_context
from cot_navigation.data.validate import validate_context
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.trace_audit import validate_nodes
from cot_navigation.experiments import scheduler
from cot_navigation.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[3]
NAME = 'fine_node_curriculum_20261028'
MANIFEST = ROOT/'configs/generated'/NAME/'manifest.json'

def flat_rows(store, cfg, tok, *, verify=False):
    """Reuse larger-budget legacy caches only if every full row fits unchanged."""
    spec = identity(store,cfg,tok)
    candidates=[]
    for path in (store.root/'prepared').glob(f'scale_{cfg.protocol.name}_*.json'):
        old=json.loads(path.read_text())
        if all(old.get(k)==v for k,v in spec.items() if k!='max_seq_len') and old['max_seq_len']>=cfg.model.max_seq_len:
            candidates.append(path)
    for path in sorted(candidates,key=lambda p:json.loads(p.read_text())['max_seq_len']):
        archive=path.with_suffix('.npz')
        if verify and not is_complete(archive):
            continue
        if not archive.is_file() or not Path(str(archive)+'.complete').is_file():
            continue
        arrays=load_npz(archive)
        if int(np.diff(arrays['offsets']).max())>cfg.model.max_seq_len-cfg.protocol.cot_margin:
            continue
        obj=PreparedRows.__new__(PreparedRows)
        obj.protocol=cfg.protocol.name;obj.arrays=arrays;obj.source_path=str(archive)
        return obj
    raise FileNotFoundError(f'No complete fitting variants: {store.root} {cfg.protocol.name}')

def prepare_one(config):
    import dataclasses
    cfg=load_config([ROOT/config],[]);root=Path(cfg.data.root)
    with _preparation_lock(root):
        generation=prepare_generation(cfg.data)
        receipt=ROOT/'outputs'/NAME/'data_receipts'/(root.name+'.json')
        if receipt.is_file():
            old=json.loads(receipt.read_text())
            if old.get('generation')==generation['fingerprint'] and unchanged_inventory(old['inventory']):
                print('Reused verified '+str(root),flush=True)
                return old['report']
        if not is_complete(root/'manifest.json'):
            generate_context(cfg.data,resume=True)
        # Native validation independently checks labels, splits and shard checksums.
        validation=validate_context(cfg.data)
        store=ContextStore(root);tok=Tokenizer.from_config(cfg)
        caches=[]
        for protocol in ('dfs_nodes','bfs_nodes'):
            pcfg=dataclasses.replace(cfg,protocol=dataclasses.replace(cfg.protocol,name=protocol))
            try:rows=flat_rows(store,pcfg,tok,verify=True)
            except FileNotFoundError:
                prepare_variants(store,pcfg,tok,protocol)
                rows=flat_rows(store,pcfg,tok,verify=True)
            # Sample every split and every variant; validate against true graph/label.
            checked=0
            for split in ('train','val','test'):
                ids=store.indices(split)
                for i in ids[np.linspace(0,len(ids)-1,12,dtype=int)]:
                    i=int(i);nodes=store.node_vocab_ids(i,tok)
                    adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(store.adj(i))}
                    for variant in range(rows.variants(i)):
                        row=rows.row(i,variant)
                        trace=row.tokens[row.prompt_len:row.answer_index]
                        result=validate_nodes(trace,row.label,adj,nodes[int(store.arrays['s_topo'][i])],
                                              nodes[int(store.arrays['t_topo'][i])],tok,protocol=='bfs_nodes')
                        assert result['valid'],(root,protocol,i,variant,result)
                        checked+=1
            caches.append(dict(protocol=protocol,path=rows.source_path,
                               sha256=file_sha256(rows.source_path),semantic_rows_checked=checked,
                               max_row_tokens=int(np.diff(rows.arrays['offsets']).max())))
        result=dict(root=str(root),generation=generation['fingerprint'],validation=validation,caches=caches)
        receipt.parent.mkdir(parents=True,exist_ok=True)
        inventory={}
        for path in root.rglob('*'):
            if path.is_file():
                stat=path.stat();inventory[str(path)]=[stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        scheduler.write_json(receipt,dict(generation=generation['fingerprint'],inventory=inventory,report=result))
        print('Prepared/reused '+str(root),flush=True)
        return result

def unchanged_inventory(inventory):
    for path,signature in inventory.items():
        try:stat=Path(path).stat()
        except FileNotFoundError:return False
        if [stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]!=signature:return False
    return True

def prepare_all(jobs,output,env):
    receipt=output/'data_ready.json'
    # Workers only consume verified flat caches, never run graph generation.
    manifest=json.loads(MANIFEST.read_text())
    if receipt.is_file():
        previous=json.loads(receipt.read_text())
        if previous.get('manifest_sha256')==file_sha256(MANIFEST):
            if unchanged_inventory(previous['inventory']):
                print('Verified data receipt unchanged; reusing all datasets and variants.',flush=True)
                return
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(prepare_one,manifest['preparations']))
    inventory={}
    for report in results:
        for p in Path(report['root']).rglob('*'):
            if p.is_file():
                stat=p.stat();inventory[str(p)]=[stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
    scheduler.write_json(receipt,dict(manifest_sha256=file_sha256(MANIFEST),inventory=inventory,datasets=results))

def semantic_report(path,tok,fifo):
    records=[json.loads(line) for line in path.read_text().splitlines()]
    valid=pos_valid=joint=positive=correct_pos=0;errors=Counter();positive_errors=Counter()
    for r in records:
        prompt=r['prompt_tokens'];q=prompt.index(tok.id('<QUERY>'))
        assert q%3==0 and len(prompt)==q+4
        adj={tok.node_id(i):[] for i in range(tok.n_node_tokens)}
        for i in range(0,q,3):
            assert prompt[i]==tok.id('<EDGE>')
            adj[prompt[i+1]].append(prompt[i+2])
        result=validate_nodes(r['trace_tokens'],r['label'],adj,prompt[q+1],prompt[q+2],tok,fifo)
        valid+=result['valid'];joint+=result['valid'] and r['correct']
        positive+=r['label']==1;pos_valid+=result['valid'] and r['label']==1
        correct_pos+=r['correct'] and r['label']==1
        if not result['valid']:
            errors[result['error']]+=1
            if r['label']==1:positive_errors[result['error']]+=1
    return dict(n=len(records),n_positive=positive,valid_rate=valid/max(1,len(records)),
                positive_valid_rate=pos_valid/max(1,positive),
                positive_answer_accuracy=correct_pos/max(1,positive),
                joint_rate=joint/max(1,len(records)),errors=dict(errors),positive_errors=dict(positive_errors))

class FineNodeTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        self._evaluation_weights=False
        super().__init__(*args,**kwargs)
        self.render_cache=OrderedDict()

    def _preload_member(self,member):
        # No eager creation of 9 x 50k x 4 Python token lists per process.
        member.prepared_tasks=flat_rows(member.store,self.cfg,self.tok)

    def _render_task(self,index,*,path_index=0):
        result=super()._render_task(index,path_index=path_index)
        if len(self.render_cache)>32768:
            self.render_cache.pop(next(iter(self.render_cache)))
        return result

    def _scale_weights(self):
        if not self._evaluation_weights:return super()._scale_weights()
        stage=min(int(np.searchsorted(self.cfg.train.scale_stage_ends,max(0,self.completed-1),side='right')),
                  len(self.cfg.train.scale_stage_ends)-1)
        width=len(self.scale_members)
        return self.cfg.train.scale_stage_weights[stage*width:(stage+1)*width]

    def _evaluate(self,kind):
        self._evaluation_weights=True
        try:
            super()._evaluate(kind)
            if kind=='free' and self.completed in self.cfg.train.scale_stage_ends:
                # Validation retention at stage boundaries, never changes training.
                self.raw.eval();self.model.eval()
                current=int(np.argmax(self._scale_weights()))
                for i,member in enumerate(self.scale_members[:current]):
                    self._activate(member)
                    rows=[self._render_task(int(j)) for j in self._selected_ids('val',256)]
                    name=f'retention_{self.completed:08d}_level{i}'
                    report=self._free(rows,record_name=name)
                    report.update(step=self.completed,data_root=member.root)
                    if self.persist:scheduler.write_json(self.out/'eval'/f'{name}.json',report)
                self._activate(self.scale_members[current])
        finally:
            self._evaluation_weights=False
            self.raw.train();self.model.train()

    def _free(self,rows,*,record_name=None):
        report=super()._free(rows,record_name=record_name)
        if self.persist:
            name=record_name or f'val_{self.completed:08d}'
            report['semantic']=semantic_report(self.out/'eval'/f'predictions_{name}.jsonl',self.tok,
                                                 self.cfg.protocol.name=='bfs_nodes')
        return report

    def _save(self):
        super()._save()
        if self.persist and self.completed in self.cfg.train.scale_stage_ends:
            target=self.out/'checkpoints'/f'stage_{self.completed:08d}.pt'
            # Refresh after evaluation, matching native latest state.
            tmp=target.with_suffix('.writing');shutil.copyfile(self.out/'checkpoints/latest.pt',tmp);tmp.replace(target)

@dataclass
class FineJob(scheduler.Job):
    def command(self,checkpoint=None,parent=None):
        assert parent is None
        command=[sys.executable,'-m','cot_navigation.experiments.fine_node_curriculum',
                 '--worker','--config',str(self.config)]
        if checkpoint is not None:command+=['--checkpoint',str(checkpoint)]
        return command

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',default=NAME);p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--wait',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--background',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--config');p.add_argument('--checkpoint')
    a=p.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    if a.worker:
        cfg=load_config([a.config],[])
        with scheduler.file_lock(Path(cfg.output_dir)/'.worker.lock',blocking=False):
            t=FineNodeTrainer(cfg,resume=a.checkpoint)
            if not a.checkpoint:
                h=hashlib.sha256()
                for k,v in t.raw.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
                scheduler.write_json(t.out/'initialization.json',dict(seed=cfg.train.model_seed,weights_sha256=h.hexdigest()))
            t.fit()
            scheduler.write_json(t.out/'COMPLETE.json',dict(step=t.completed,queries_seen=t.queries_seen))
        return 0
    assert a.run_name==NAME,'Configurations are immutable; generate a new suite for another name.'
    manifest=json.loads(MANIFEST.read_text());output=ROOT/'outputs'/NAME
    jobs=[]
    for j in manifest['jobs']:
        cfg=load_config([ROOT/j['config']],[])
        jobs.append(FineJob(j['id'],ROOT/j['config'],[],Path(cfg.output_dir),cfg.train.max_steps))
    if a.prepare_only:
        output.mkdir(parents=True,exist_ok=True)
        with scheduler.file_lock(output/'.data_preparation.lock',blocking=False):prepare_all(jobs,output,os.environ)
        return 0
    # Native scheduler requires resume when prepare-only has created output.
    if a.execute and output.exists() and not (output/'launch_plan.json').exists():
        allowed={'data_ready.json','data_receipts','.data_preparation.lock'}
        if any(path.name not in allowed for path in output.iterdir()):
            raise RuntimeError('Existing output contains records beyond shared data preparation')
        scheduler.write_json(output/'launch_plan.json',scheduler.plan_record(jobs))
        a.resume=True
        if '--resume' not in argv:argv+=['--resume']
    a.dry_run=not a.execute
    # Use a tiny script adapter only for native background re-entry.
    return scheduler.execute(a,output,jobs,argv,
                             script_path=ROOT/'studies/fine_node_curriculum/launch.py',data_preparer=prepare_all)

if __name__=='__main__':raise SystemExit(main())
