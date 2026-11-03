"""Observed wrong-node repair with the native long-curriculum training engine."""
import os,sys,json,argparse,dataclasses,shutil,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.experiments import scheduler
from cot_navigation.experiments.fine_node_curriculum import FineNodeTrainer,flat_rows
from cot_navigation.data.dataset import ContextStore
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.trace_audit import validate_nodes
from cot_navigation.data.dataset import collate_examples
from cot_navigation.models.transformer import build_model
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.training.checkpoint import load_checkpoint

ROOT=Path(__file__).resolve().parents[2];NAME='node_token_correction_20261029'
BASE=ROOT/'configs/generated/fine_node_curriculum_20261028'
HERE=ROOT/'studies/node_token_correction';OUT=ROOT/'outputs'/NAME

def prepare():
    source=json.loads((BASE/'manifest.json').read_text());jobs=[]
    generated=ROOT/'configs/generated'/NAME;generated.mkdir(parents=True,exist_ok=True)
    import yaml
    for job in source['jobs']:
        donor=HERE/'donors'/(job['id'].replace('/','__')+'.pt');donor.parent.mkdir(exist_ok=True)
        if not donor.exists():
            original=ROOT.with_name('cot-navigation')/'outputs/node_family_recheck_20261028'/job['id']/'checkpoints/stage_00050000.pt'
            shutil.copyfile(original,donor)
        for arm in ['repair','matched_clean','repair_full_loss']:
            cfg=yaml.safe_load((ROOT/job['config']).read_text());ident=job['id']+'/'+arm
            cfg['output_dir']=str(OUT/ident)
            cfg['protocol'].update(enable_mistake=True,mistake_source='natural',mistake_mask_errors=arm!='repair_full_loss',mistake_rate=0.3)
            cfg['tracking'].update(group=NAME,name=NAME+'/'+ident,variant=arm,
                notes='Observed wrong-node predictions on train-only correct prefixes. Wrong node immediately followed by MISTAKE then gold node; wrong token loss masked. Shared 30% observed-query replay in both arms.')
            path=generated/(ident.replace('/','__')+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_config([path],[])
            jobs.append({**job,'arm':arm,'config':str(path),'donor':str(donor),'name':ident})
    (HERE/'manifest.json').write_text(json.dumps(dict(name=NAME,jobs=jobs),indent=2))

def collect(job,cfg):
    """GPU preprocessing once; store only model-produced repeated/start errors."""
    cache=OUT/'pools'/job['id'];cache.mkdir(parents=True,exist_ok=True)
    with scheduler.file_lock(cache/'.lock'):
        ready=cache/'READY.json'
        if ready.exists():return
        clean=dataclasses.replace(cfg,protocol=dataclasses.replace(cfg.protocol,enable_mistake=False,mistake_rate=0,mistake_source='synthetic'))
        tok=Tokenizer.from_config(clean)
        model=build_model(clean.model,len(tok),pad_id=tok.pad).cuda().eval()
        blob=load_checkpoint(job['donor']);model.load_state_dict(blob['model']);del blob
        inference=CompiledInference(model);summary=[]
        for level,root in enumerate(cfg.train.scale_roots):
            dest=cache/f'level{level}.json'
            if dest.exists():continue
            store=ContextStore(root);rows=flat_rows(store,clean,tok)
            ids=store.indices('train').copy();np.random.default_rng(20261029).shuffle(ids)
            events=[];counts=Counter()
            for start in range(0,len(ids),64):
                batchids=ids[start:start+64];batch=[rows.row(int(i),0) for i in batchids]
                arrays=collate_examples(batch,tok.pad,64)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=inference(torch.from_numpy(arrays['tokens']).cuda()).argmax(-1).cpu().numpy()
                for bi,(index,row) in enumerate(zip(batchids,batch)):
                    if counts[row.label]>=256:continue
                    adj={tok.node_id(k):[] for k in range(tok.n_node_tokens)}
                    for k in range(0,row.prompt_len-4,3):adj[row.tokens[k+1]].append(row.tokens[k+2])
                    src=row.tokens[row.prompt_len-3]
                    for pos in range(row.prompt_len,row.answer_index):
                        bad=int(pred[bi,pos-1]);gold=row.tokens[pos]
                        # A repeated previously expanded node or wrong source is
                        # unambiguously illegal, not another valid neighbor order.
                        if bad!=gold and tok.node_lo<=bad<tok.node_lo+tok.n_node_tokens and (pos==row.prompt_len or bad in row.tokens[row.prompt_len:pos] or not validate_nodes(row.tokens[row.prompt_len:pos]+[bad],1,adj,src,bad,tok,cfg.protocol.name=='bfs_nodes')['valid']):
                            events.append(dict(query_id=int(index),position=pos,bad=bad,gold=gold,label=row.label,
                                               kind='wrong_start' if pos==row.prompt_len else 'repeated_node' if bad in row.tokens[row.prompt_len:pos] else 'illegal_next_node'))
                            counts[row.label]+=1;break
                if min(counts.get(0,0),counts.get(1,0))>=256:break
            if min(counts.get(0,0),counts.get(1,0))<8:raise RuntimeError(f'Insufficient natural errors {root}: {counts}')
            scheduler.write_json(dest,dict(root=root,donor=job['donor'],donor_step=50000,
                split='train',variant=0,conditioning='gold prefix greedy next-node',events=events,
                counts=dict(counts),queries_scanned=min(start+64,len(ids))))
            summary.append(dict(level=level,counts=dict(counts)))
            print('Observed pool',job['id'],level,dict(counts),flush=True)
        scheduler.write_json(ready,dict(donor=job['donor'],levels=len(cfg.train.scale_roots),summary=summary))
        del model,inference;torch.cuda.empty_cache()

def cancel(tokens,marker):
    out=[];invalid=False
    for token in tokens:
        if token==marker:
            if not out:invalid=True
            else:out.pop()
        else:out.append(token)
    return out,invalid

class Trainer(FineNodeTrainer):
    def __init__(self,cfg,job,**kw):
        self.job=job;self.events={}
        super().__init__(cfg,**kw)
        for i,member in enumerate(self.scale_members):
            data=json.loads((OUT/'pools'/job['id']/f'level{i}.json').read_text())
            self.events[member.root]={label:[e for e in data['events'] if e['label']==label] for label in [0,1]}

    def _training_task(self,index):
        # The native state-episode augmentation is bypassed for node episodes.
        cfg=self.cfg
        self.cfg=dataclasses.replace(cfg,protocol=dataclasses.replace(cfg.protocol,enable_mistake=False))
        try:row=super()._training_task(index)
        finally:self.cfg=cfg
        rng=self.augmentation_rngs['mistake'];inject=rng.random()<0.3
        if not inject:return row
        pool=self.events[self._active_member][row.label];event=pool[int(rng.integers(len(pool)))]
        row=self._render_task(event['query_id'],path_index=0)
        # Identical donor-query exposure and edge permutations in both arms.
        edges=np.asarray(row.tokens[:row.prompt_len-4]).reshape(-1,3)
        prompt=edges[self.augmentation_rngs['edges'].permutation(len(edges))].ravel().tolist()+row.tokens[row.prompt_len-4:row.prompt_len]
        row=dataclasses.replace(row,tokens=prompt+row.tokens[row.prompt_len:])
        if self.job['arm']=='matched_clean':return row
        pos=event['position'];assert row.tokens[pos]==event['gold']
        tokens=row.tokens[:pos]+[event['bad'],self.tok.id('<MISTAKE>')]+row.tokens[pos:]
        trace=[x if x<pos else x+2 for x in row.trace_indices]+[pos+1]
        if not cfg.protocol.mistake_mask_errors:trace.append(pos)
        result=dataclasses.replace(row,tokens=tokens,answer_index=row.answer_index+2,trace_indices=sorted(trace),
            meta={**row.meta,'mistake':True,'observed_error':event})
        assert tokens[pos:pos+3]==[event['bad'],self.tok.id('<MISTAKE>'),event['gold']]
        assert (pos not in result.trace_indices)==cfg.protocol.mistake_mask_errors
        assert pos+1 in result.trace_indices and pos+2 in result.trace_indices
        self._check_length(result);return result

    def _free(self,rows,*,record_name=None):
        # Keep raw predictions; semantic scoring applies token cancellation.
        from cot_navigation.training.trainer import Trainer as Native
        from cot_navigation.experiments.fine_node_curriculum import semantic_report
        report=Native._free(self,rows,record_name=record_name)
        if self.persist:
            name=record_name or f'val_{self.completed:08d}'
            path=self.out/'eval'/f'predictions_{name}.jsonl'
            records=[json.loads(x) for x in path.read_text().splitlines()]
            normalized=[];malformed=0;markers=0
            for r in records:
                trace,bad=cancel(r['trace_tokens'],self.tok.id('<MISTAKE>'));malformed+=bad
                markers+=r['trace_tokens'].count(self.tok.id('<MISTAKE>'))
                normalized.append({**r,'trace_tokens':trace if not bad else [self.tok.pad]})
            temp=path.with_name('normalized_'+path.name);temp.write_text('\n'.join(json.dumps(r) for r in normalized))
            report['semantic']=semantic_report(temp,self.tok,self.cfg.protocol.name=='bfs_nodes')
            report['correction_tokens']=dict(markers=markers,malformed=malformed)
        return report

@dataclasses.dataclass
class Job(scheduler.Job):
    def command(self,checkpoint=None,parent=None):
        argv=[sys.executable,str(HERE/'experiment.py'),'--worker',self.name]
        if checkpoint:argv+=['--checkpoint',str(checkpoint)]
        return argv

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker');p.add_argument('--checkpoint')
    p.add_argument('--full-loss-suite',action='store_true');p.add_argument('--run-name',default=NAME);p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--execute',action='store_true');p.add_argument('--wait',action='store_true');p.add_argument('--background',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.prepare:prepare();return
    jobspec=json.loads((HERE/'manifest.json').read_text())['jobs']
    if a.worker:
        job=next(j for j in jobspec if j['name']==a.worker);cfg=load_config([job['config']],[])
        collect(job,cfg)
        with scheduler.file_lock(Path(cfg.output_dir)/'.worker.lock',blocking=False):
            t=Trainer(cfg,job,resume=a.checkpoint);t.fit()
            scheduler.write_json(t.out/'COMPLETE.json',dict(step=t.completed))
        return
    jobspec=[j for j in jobspec if (j['arm']=='repair_full_loss')==a.full_loss_suite]
    if a.full_loss_suite:a.run_name='node_token_full_loss_20261029'
    jobs=[Job(j['name'],Path(j['config']),[],Path(load_config([j['config']],[]).output_dir),1000000) for j in jobspec]
    a.dry_run=not a.execute
    def prepared(*unused):
        if a.full_loss_suite:
            print('Waiting for original 24-job suite to finish before assigning GPUs.',flush=True)
            while True:
                status=json.loads((OUT/'status.json').read_text())
                if not status['running'] and not status['pending'] and len(status['completed'])==24:break
                time.sleep(15)
    scheduler.execute(a,ROOT/'outputs'/a.run_name,jobs,sys.argv[1:],script_path=HERE/'experiment.py',data_preparer=prepared)

if __name__=='__main__':main()


