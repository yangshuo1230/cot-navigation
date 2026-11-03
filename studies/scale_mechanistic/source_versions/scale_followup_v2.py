"""Frozen scale-model probes and paired token interventions.

Run as cot_navigation.analysis.scale_followup from the repository root.
Navigation checkpoints are read-only; only small JSON summaries are persisted.
The historical analysis modules remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import time


def inventory(root):
    import torch
    import yaml
    from .common import sha256
    base = root / 'outputs/scale_v1/launches/scale_v1_s0'
    entries = []
    # All protocols are retained; comparisons use validation-selected best.pt.
    for kind in ['context', 'fixed']:
        for scale in ['l4', 'l6', 'l8', 'l10', 'curriculum']:
            folder = base / kind / scale
            if kind == 'fixed':
                folder /= 'stage_b'
            for protocol in ['direct', 'path', 'dfs_state', 'bfs_state', 'dfs_nodes', 'bfs_nodes']:
                run = folder / protocol
                cfg = yaml.safe_load((run / 'resolved_config.yaml').read_text())
                path = run / 'checkpoints/best.pt'
                blob = torch.load(path, map_location='cpu', weights_only=False)
                entries.append(dict(id=f'{kind}_{scale}_{protocol}', run=str(run.relative_to(root)),
                                    kind=kind, scale=scale, protocol=protocol,
                                    n_nodes=cfg['data']['n_nodes'], step=int(blob['step']),
                                    validation_score=float(blob['best_score']),
                                    selection_metric=cfg['evaluation']['selection_metric'],
                                    checkpoint_sha256=sha256(path)))
                del blob
    return entries


def make_adapter(entry, root, inference_device='cuda:0'):
    import numpy as np
    from .common import AnalysisModel, Example, SEED
    from cot_navigation.data.dataset import ParametricWorld
    from cot_navigation.data.scale_prepared import ScalePreparedRows

    class ScaleModel(AnalysisModel):
        def __init__(self):
            super().__init__(entry, root, device=inference_device)
            self.inference_device = self.device
            import torch
            self.device = torch.device('cuda:0')  # affine heads always train on physical GPU0
            # Exact offline reference variant and STOP truncation used in training.
            self.prepared = ScalePreparedRows(self.store, self.cfg, self.tok)
            self.world_prefix = []
            if not self.context and self.cfg.train.world_roots:
                worlds = [(root / p).resolve() for p in self.cfg.train.world_roots]
                self.world_prefix = [self.tok.id(f'<WORLD_{worlds.index(self.store.root.resolve())}>')]

        def example(self, index, split, variant=0):
            a = self.store.arrays if self.context else self.store.queries
            row = self.prepared.row(index, variant)
            if self.world_prefix:
                n = len(self.world_prefix)
                row = replace(row, tokens=self.world_prefix + row.tokens,
                              prompt_len=row.prompt_len+n, answer_index=row.answer_index+n,
                              trace_indices=[i+n for i in row.trace_indices],
                              walk_indices=[i+n for i in row.walk_indices])
            adj = self.store.adj(index) if self.context else self.store.adj()
            nodes = self.store.node_vocab_ids(index, self.tok) if self.context else self.store.node_vocab_ids(self.tok)
            assert len(row.tokens) <= self.model.max_seq_len
            return Example(index, split, row.label, adj, nodes,
                           int(a['s_topo'][index]), int(a['t_topo'][index]), row)

        def examples(self, split, count):
            return [self.example(i, split) for i in self.sample_ids(split, count)]

        def forward(self, sequences, *, attention=False):
            # Historical feature extractors use am.device for head fitting.
            # Only the frozen forward temporarily uses the inference device.
            probe_device = self.device
            self.device = self.inference_device
            try:
                return super().forward(sequences, attention=attention)
            finally:
                self.device = probe_device

    return ScaleModel()


def generate(am, prompts, *, cap=0, batch=16):
    from cot_navigation.evaluation.rollout import generate as rollout
    result = []
    for at in range(0, len(prompts), batch):
        result.extend(rollout(am.model, prompts[at:at+batch], am.protocol, am.tok, 'bf16',
                              max_new_tokens=cap, cot_margin=am.cfg.protocol.cot_margin))
    return result


def free_rollouts(am, examples, cap=0):
    seqs, rows = {}, []
    for ex, out in zip(examples, generate(am, [e.prompt for e in examples], cap=cap)):
        seqs[ex.index] = ex.prompt + out['trace_tokens']
        rows.append(dict(query=ex.index, label=ex.label, pred=out['pred'],
                         correct=out['pred']==ex.label, limit=out['generation_limit_reached'],
                         interrupted=out['cot_interrupted'], forced_stop=out['forced_stop'],
                         trace_tokens=len(out['trace_tokens'])))
    return seqs, rows


def extract(am, examples, free=None, batch=6):
    """Affine residual probes plus the historical input-feature control.

    visited = PUSH-implied discovery history before printing VIS.
    visited_after_field = the printed VIS set, just after printing VIS.
    allowed_successors = graph neighbors minus printed VIS at that same point.
    The latter two diagnose reading/filtering the currently explicit state.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .common import anchors, records
    stores = {}
    for at in range(0, len(examples), batch):
        chunk = examples[at:at+batch]
        sequences = [free[e.index] if free is not None else e.row.tokens[:e.row.answer_index] for e in chunk]
        _, residual, _ = am.forward(sequences)
        for b, (ex, seq) in enumerate(zip(chunk, sequences)):
            answer, structural = anchors(seq, ex, am.protocol, am.tok)
            inverse = {v:i for i,v in enumerate(ex.nodes)}
            def add(task, pos, positives, stage, current=None):
                h = F.pad(torch.stack([residual[l][b,pos].float() for l in am.layers]), (0,4))
                token = seq[pos] if current is None else current
                emb = am.model.embed.weight[token].float()
                seen = ex.nodes[ex.target] in seq[ex.row.prompt_len:pos+1]
                obs = emb.new_tensor([pos/am.model.max_seq_len, (pos-ex.row.prompt_len)/am.model.max_seq_len,
                                      float(seen), float(token==ex.nodes[ex.target])])
                x = torch.cat([h, torch.cat([emb,obs])[None]],0).cpu().numpy()
                if task == 'answer':
                    y, mask = np.array([ex.label],np.float32), np.ones(1,np.float32)
                else:
                    y, mask = np.zeros(am.tok.n_node_tokens,np.float32), np.zeros(am.tok.n_node_tokens,np.float32)
                    for v in ex.nodes: mask[v-am.tok.node_lo]=1
                    for v in positives: y[v-am.tok.node_lo]=1
                row = dict(x=x,y=y,mask=mask,query=ex.index,stage=stage,goal_seen=seen)
                s = stores.setdefault(task,{k:[] for k in row})
                for k,v in row.items(): s[k].append(v)
            for stage,pos in answer: add('answer',pos,[],stage)
            for item in structural:
                current=item['current']
                successors=[ex.nodes[v] for v in ex.adj[inverse[current]]]
                add('successors',item.get('successor_pos',item['pos']),successors,'pre_candidates',current)
                if am.protocol.startswith(('dfs_','bfs_')):
                    add('visited',item['pos'],item['visited'],item['stage'],current)
            if am.protocol.endswith('_state'):
                parsed=records(seq,ex.row.prompt_len,am.tok)
                selected={i['pos'] for i in structural}
                for r in parsed:
                    if r['separators'][0] not in selected: continue
                    current=r['frontier'][0 if am.protocol=='bfs_state' else -1]
                    if current not in inverse or not set(r['visited'])<=set(ex.nodes): continue
                    pos=r['separators'][1]
                    add('visited_after_field',pos,set(r['visited']),'pre_candidates',current)
                    allowed={ex.nodes[v] for v in ex.adj[inverse[current]]}-set(r['visited'])
                    add('allowed_successors',pos,allowed,'pre_candidates',current)
        del residual
    return {task:{k:np.asarray(v) for k,v in data.items()} for task,data in stores.items()}


def probe_stage(am, splits, free, steps):
    from .common import SEED
    from .probes import fit,evaluate,shuffled_labels
    from .set_metrics import evaluate_sets
    features={s:extract(am,ex) for s,ex in splits.items()}
    features['generated']=extract(am,splits['test'],free)
    out={}
    for task,tr in features['train'].items():
        if any(task not in features[s] for s in ['val','test']):
            out[task]={'status':'insufficient anchors'};continue
        va=features['val'][task]
        fitted=fit(tr,va,device=am.device,seed=SEED,steps=steps)
        control=fit(tr,va,device=am.device,seed=SEED+17,steps=steps,shuffled=True)
        rec={'selected_steps':fitted['steps'],'n_anchors':{s:len(features[s][task]['y']) for s in features if task in features[s]}}
        for split in ['test','generated']:
            if task not in features[split]: continue
            data=features[split][task]
            rec[split]=evaluate(fitted,data)
            if task!='answer': rec[split+'_sets']=evaluate_sets(fitted,data)
            random={**data,'y':shuffled_labels(data,SEED+219)}
            rec[split+'_shuffled']=evaluate(control,random)
            if task!='answer':rec[split+'_shuffled_sets']=evaluate_sets(control,random)
        out[task]=rec
        del fitted,control
    return out


def blocking_stage(am, examples, free, count):
    import numpy as np
    from .common import records, SEED
    from .route_blocking import blocking_case
    from .patching import summarize_pairs
    result={}
    for exclude in [False,True]:
        cases=[]; dropped=Counter()
        for ex in examples:
            case=blocking_case(am,ex,free[ex.index],exclude_goal=exclude)
            if case is None: continue
            for name in list(case['prefixes']):
                if len(case['prefixes'][name])>=am.model.max_seq_len-am.cfg.protocol.cot_margin-4:
                    del case['prefixes'][name];dropped[name]+=1
            if 'clean' not in case['prefixes'] or 'all_routes' not in case['prefixes']:continue
            cases.append(case)
            if len(cases)>=count:break
        key='excluding_goal' if exclude else 'including_goal'
        if not cases:
            result[key]={'status':'no certified eligible states','n_cases':0};continue
        flat=[];keys=[]
        for i,c in enumerate(cases):
            for name,prefix in c['prefixes'].items():keys.append((i,name));flat.append(prefix)
        grouped=[{} for _ in cases]
        for (i,name),out in zip(keys,generate(am,flat)):grouped[i][name]=out
        base=summarize_pairs(cases,grouped,am.tok)
        rng=np.random.default_rng(SEED)
        for name,condition in base['conditions'].items():
            ids=[i for i,g in enumerate(grouped) if name in g]
            condition['counterfactual_NO_rate']=float(np.mean([grouped[i][name]['pred']==0 for i in ids]))
            condition['clean_accuracy_same_cases']=float(np.mean([grouped[i]['clean']['pred']==1 for i in ids]))
            condition['interrupted_rate']=float(np.mean([grouped[i][name]['cot_interrupted'] for i in ids]))
            condition['forced_stop_rate']=float(np.mean([grouped[i][name]['forced_stop'] for i in ids]))
            if name in {'all_routes','all_unvisited'}:
                pushes=[];retains=[];legal_candidates=[];legal_transition=[]
                for i in ids:
                    c,o=cases[i],grouped[i][name];prefix=c['prefixes'][name]
                    # Unlike the historical helper, audit the FULL actual insertion.
                    inserted=set(c['detail']['inserted']) if name=='all_routes' else set(examples[0].nodes)
                    # Graph node identities vary across context queries.
                    vis_start=len(prefix)-1
                    while vis_start>0 and prefix[vis_start-1]!=am.tok.id('|'):vis_start-=1
                    edited_vis=set(prefix[vis_start:-1])
                    if name=='all_unvisited':inserted=edited_vis-set(c['detail']['visited'])
                    gen=o['generated'];pushes.append(any(a==am.tok.id('<PUSH>') and b in inserted for a,b in zip(gen,gen[1:])))
                    first=[r for r in records(o['tokens'],c['prompt_length'],am.tok) if r['separators'][1]==len(prefix)-1]
                    future=[r for r in records(o['tokens'],c['prompt_length'],am.tok) if r['start']>=len(prefix)]
                    if future:retains.append(float(inserted<=set(future[0]['visited'])))
                    if first:
                        r=first[0]
                        ex=next(e for e in examples if e.index==c['query'])
                        inv={v:j for j,v in enumerate(ex.nodes)}
                        current=c['detail']['frontier'][0 if am.protocol=='bfs_state' else -1]
                        expected={ex.nodes[v] for v in ex.adj[inv[current]]}-edited_vis
                        legal_candidates.append(float(set(r['candidates'])==expected and len(r['candidates'])==len(expected)))
                        if future:
                            action=r['action'];ff=list(c['detail']['frontier']);vv=set(edited_vis)
                            ok=False
                            if len(action)==2 and action[0]==am.tok.id('<PUSH>') and action[1] in expected:
                                ff.append(action[1]);vv.add(action[1]);ok=True
                            elif action==[am.tok.id('<POP>')] and not expected and len(ff)>1:
                                ff.pop(0 if am.protocol=='bfs_state' else -1);ok=True
                            legal_transition.append(float(ok and future[0]['frontier']==ff and set(future[0]['visited'])==vv))
                condition.update(pushed_inserted_rate=float(np.mean(pushes)),
                                 retained_n=len(retains),retained_rate=float(np.mean(retains)) if retains else None,
                                 first_record_parseable_n=len(legal_candidates),
                                 first_candidates_correct_rate=float(np.mean(legal_candidates)) if legal_candidates else None,
                                 transition_checked_n=len(legal_transition),
                                 first_transition_correct_rate=float(np.mean(legal_transition)) if legal_transition else None)
            if name not in {'all_routes','clean'}:
                delta=np.array([int(grouped[i]['all_routes']['pred']==1)-int(grouped[i][name]['pred']==1) for i in ids])
                boots=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(-1)
                base['paired_contrasts']['all_routes_minus_'+name]={'n':len(ids),'YES_difference':float(delta.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist()}
        result[key]={'status':'complete','n_cases':len(cases),'certified_blocked_n':len(cases),
                     'query_ids':[c['query'] for c in cases], 'oversize_condition_skips':dict(dropped),
                     'mean_inserted':float(np.mean([c['detail']['inserted_count'] for c in cases])),**base}
    return result


def error_stage(am,splits,free,count,steps):
    """Use existing first-error machinery only where its semantics are valid.

    STOP is an interruption, not an erroneous search token. The historical
    node-only validator enforces BFS levels and must NOT label DFS-node traces.
    """
    import numpy as np
    from .common import SEED
    from .error_detection import events,extract_events
    from .probes import fit,subset
    from .set_metrics import evaluate_errors
    if am.protocol in {'direct','dfs_nodes'}:
        return {'status':'not applicable' if am.protocol=='direct' else 'omitted: historical error oracle is BFS-only for node traces'}
    def trim(seq):
        stop=am.tok.id('<STOP>')
        return seq[:seq.index(stop)] if stop in seq else seq
    natural={};synthetic={};inventory={}
    for split,examples in splits.items():
        used=examples if split=='test' else examples[:count]
        if split=='test':seqs=free;rows=[]
        else:seqs,rows=free_rollouts(am,used,cap=512)
        natural[split]=[event for ex in used if (event:=events(ex,am.protocol,am.tok,trim(seqs[ex.index])))]
        synthetic[split]=[event for ex in examples if (event:=events(ex,am.protocol,am.tok,trim(ex.row.tokens[:ex.row.answer_index]),synthetic=True,seed=SEED))]
        inventory[split]={'natural_queries':len(used),'natural_errors':len(natural[split]),
                          'synthetic_errors':len(synthetic[split]),'generation_cap_hits':sum(r['limit'] for r in rows)}
    out={'inventory':inventory,'readouts':{}}
    def join(items):
        items=[x for x in items if x is not None]
        return {k:np.concatenate([x[k] for x in items]) for k in items[0]} if items else None
    for offset in [0,1]:
        data={(split,source):extract_events(am,ex,(natural if source=='natural' else synthetic)[split],offset)
              for split,ex in splits.items() for source in ['natural','synthetic']}
        tr=join([data['train',s] for s in ['natural','synthetic']]);va=join([data['val',s] for s in ['natural','synthetic']])
        if tr is None or va is None:continue
        head=fit(tr,va,device=am.device,seed=SEED,steps=steps)
        def random(d,seed):return {**d,'y':np.random.default_rng(seed).permutation(d['y'])}
        ctrl=fit(random(tr,SEED+17),random(va,SEED+118),device=am.device,seed=SEED+17,steps=steps)
        rec={}
        for source in ['natural','synthetic']:
            te=data['test',source]
            if te is None:rec[source]={'status':'no eligible errors'};continue
            item={'n_pairs':len(te['y'])//2,'all':evaluate_errors(head,te),'shuffled':evaluate_errors(ctrl,random(te,SEED+219))}
            if offset:
                valid=te['repair_next_valid'];item['strict_pairs']=int(valid.sum()//2)
                if valid.any():item['strict']=evaluate_errors(head,subset(te,valid))
            rec[source]=item
        out['readouts']['wrong_token' if offset==0 else 'next_token']=rec
        del data,head,ctrl
    out['examples']=[{k:v for k,v in e.items() if k not in {'wrong','correct'}} |
                     {'wrong_tail':am.tok.decode(e['wrong'][-60:]),'correct_tail':am.tok.decode(e['correct'][-60:])}
                     for e in natural['test'][:3]]
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('studies/scale_mechanistic/results'))
    parser.add_argument('--train',type=int,default=256)
    parser.add_argument('--val',type=int,default=96)
    parser.add_argument('--test',type=int,default=128)
    parser.add_argument('--cases',type=int,default=40)
    parser.add_argument('--steps',type=int,default=250)
    parser.add_argument('--error-train',type=int,default=32)
    parser.add_argument('--inference-gpu',type=int,choices=range(8),default=0)
    parser.add_argument('--models',nargs='*')
    parser.add_argument('--stages',nargs='+',default=['probes','blocking','patching','errors'])
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='0' if args.inference_gpu==0 else f'0,{args.inference_gpu}'
    os.environ['WANDB_MODE']='disabled'
    import numpy as np
    import torch
    from .common import write_json,sha256
    from .patching import input_case,node_case,summarize_pairs
    torch.set_num_threads(4)
    root=Path.cwd();args.output.mkdir(parents=True,exist_ok=True)
    selection=args.output/'selection.json'
    entries=json.loads(selection.read_text())['models'] if selection.exists() else inventory(root)
    if not selection.exists():write_json(selection,{'rule':'each run best.pt selected on validation; no test ranking','models':entries})
    settings={k:v for k,v in vars(args).items() if k not in {'models','stages','output','inference_gpu'}}
    settings.update(gpu=0,source_sha256=sha256(Path(__file__)),context=4096,
                    reference='actual prepared variant 0; WORLD prefix; trained STOP policy',
                    no_saved_weights=True)
    sp=args.output/'settings.json'
    if sp.exists():
        previous=json.loads(sp.read_text())
        # Device placement is an execution change; preserve both source hashes.
        old_logical={k:v for k,v in previous.items() if k not in {'source_sha256','source_history'}}
        new_logical={k:v for k,v in settings.items() if k!='source_sha256'}
        if old_logical!=new_logical:raise ValueError('Analysis settings changed; use fresh output')
        history=previous.get('source_history',[previous['source_sha256']])
        if settings['source_sha256'] not in history:
            settings['source_history']=history+[settings['source_sha256']]
            write_json(sp,settings)
    else:write_json(sp,settings)
    for entry in entries:
        name=entry['id']
        if args.models and name not in args.models:continue
        path=args.output/'models'/f'{name}.json'
        result=json.loads(path.read_text()) if path.exists() else {'entry':entry}
        if all(s in result for s in args.stages):continue
        print('MODEL',name,flush=True);start=time.time()
        am=make_adapter(entry,root,'cuda:0' if args.inference_gpu==0 else 'cuda:1')
        result.setdefault('execution_history',[]).append({'source_sha256':settings['source_sha256'],
            'inference_physical_gpu':args.inference_gpu,'probe_physical_gpu':0,
            'resumed_stages':[s for s in args.stages if s in result]})
        assert am.checkpoint['sha256']==entry['checkpoint_sha256']
        splits={s:am.examples(s,getattr(args,s)) for s in ['train','val','test']}
        ids={s:[e.index for e in ex] for s,ex in splits.items()}
        assert all(not set(ids[a])&set(ids[b]) for a,b in [('train','val'),('train','test'),('val','test')])
        result.update(checkpoint=am.checkpoint,query_ids=ids,layers=[i+1 for i in am.layers]+['input_control'])
        free,rows=free_rollouts(am,splits['test'])
        result['rollout']={'n':len(rows),'accuracy':float(np.mean([r['correct'] for r in rows])),
                           'interrupted':sum(r['interrupted'] for r in rows),'forced_stop':sum(r['forced_stop'] for r in rows),'by_query':rows}
        write_json(path,result)
        for stage in args.stages:
            if stage in result:continue
            print(' STAGE',stage,flush=True)
            if stage=='probes':result[stage]=probe_stage(am,splits,free,args.steps)
            elif stage=='blocking':
                result[stage]=blocking_stage(am,splits['test'],free,args.cases) if am.protocol.endswith('_state') else {'status':'no visited field'}
            elif stage=='patching':
                if am.protocol.endswith('_state'):result[stage]={'status':'covered by certified blocking'}
                else:
                    cases=[]
                    for ex in splits['test']:
                        case=input_case(am,ex) if am.protocol=='direct' else node_case(am,ex,free[ex.index])
                        if case:cases.append(case)
                        if len(cases)>=args.cases:break
                    flat=[];keys=[]
                    for i,c in enumerate(cases):
                        for k,p in c['prefixes'].items():flat.append(p);keys.append((i,k))
                    grouped=[{} for _ in cases]
                    for (i,k),o in zip(keys,generate(am,flat)):grouped[i][k]=o
                    result[stage]={'n_cases':len(cases),**summarize_pairs(cases,grouped,am.tok)} if cases else {'status':'no eligible cases'}
            elif stage=='errors':result[stage]=error_stage(am,splits,free,args.error_train,args.steps)
            else:raise ValueError(stage)
            write_json(path,result)
        result['checkpoint_unchanged_after']=sha256(am.run/'checkpoints/best.pt')==entry['checkpoint_sha256']
        assert result['checkpoint_unchanged_after']
        result['elapsed_seconds']=time.time()-start
        write_json(path,result);print(' DONE',name,round(time.time()-start,1),flush=True)
        del am,free;torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
