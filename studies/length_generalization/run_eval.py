"""Frozen-checkpoint graph-size/hop extrapolation. No optimization or W&B.

Run from the real project root. Uses its unchanged generation, rendering and
greedy cached decoder. All methods see identical graphs/prompts per condition.
"""
import os
os.environ['WANDB_MODE']='disabled'
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from cot_navigation.config import load_config
from cot_navigation.data.generate import _generate_context_row
from cot_navigation.data.dataset import ContextStore,ParametricWorld
from cot_navigation.data.prepared import PreparedRows
from cot_navigation.evaluation.sampling import stratified_ids
from cot_navigation.evaluation.rollout import generate
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import context_prompt,render_task,render_steps
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search,first_expansion_order
from cot_navigation.training.augmentation import transform
from trace_audit import validate_trace,self_test

TOK=Tokenizer()
RUNROOT=Path('outputs/explore_v1/launches/explore_v1_s012')
CONTEXT=[('direct','baseline/direct'),('direct_onehop','direct_onehop/direct'),
         ('direct_curriculum','direct_curriculum/direct'),('path','baseline/path'),
         ('path_negative','path_negative/path'),('dfs_state','baseline/dfs_state'),
         ('bfs_state','bfs_state/bfs_state'),('dfs_nodes','dfs_nodes/dfs_nodes'),('bfs_nodes','bfs_nodes/bfs_nodes')]
SHAPES=[('id',4,4,24),('id_disconnected_pad32',4,8,48),
        ('wide32_branch2',4,8,48),('wide32_densityhalf',4,8,96),
        ('deep16_sameprompt',8,2,24),('deep24',6,4,40),('deep32',8,4,56),
        ('deep48',12,4,88),('large64',8,8,112)]


def graph_record(adj,nodes,s,t,h,label,index,edges=None):
    depth=[0]*len(nodes)
    for u,vs in enumerate(adj):
        for v in vs:depth[v]=max(depth[v],depth[u]+1)
    token_adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(adj)}
    return {'adj':adj,'nodes':nodes,'s':s,'t':t,'h':h,'label':label,'index':index,
            'token_adj':token_adj,'longest_graph_path':max(depth),
            'edges':edges if edges is not None else [(u,v) for u,vs in enumerate(adj) for v in vs]}


def make_data(per_label):
    source=ContextStore('data/explore_v1/context_standard_n16_m24_train50000_val10000_test10000_s0')
    allhash=set(map(int,source.arrays['struct_hash']))
    data={}
    for case,layers,width,m in SHAPES:
        rows=[]
        if case=='id':
            for h in range(1,4):
                for y in (0,1):
                    pool=source.indices('test')
                    pool=pool[(source.arrays['distance'][pool]==h)&(source.arrays['label'][pool]==y)]
                    for i in np.random.default_rng(20261017+h*2+y).choice(pool,per_label,replace=False):
                        i=int(i);a=source.arrays
                        edges=list(zip(map(int,a['edge_src'][i]),map(int,a['edge_dst'][i])))
                        rows.append(graph_record(source.adj(i),source.node_vocab_ids(i,TOK),int(a['s_topo'][i]),
                            int(a['t_topo'][i]),h,y,i,edges))
        elif case=='id_disconnected_pad32':
            # Keep query component, labels, node names and within-component
            # neighbor order IDENTICAL to ID. Only add an unreachable component
            # and interleave its edges, increasing prompt length from76 to148.
            from cot_navigation.data.graph import sample_edges
            rng=np.random.default_rng(20261017)
            for old in data['id']:
                fresh=rng.choice([x for x in range(256) if TOK.node_id(x) not in old['nodes']],16,replace=False)
                nodes=[None]*32
                remap=lambda u:(u//4)*8+u%4
                for u in range(16):
                    nodes[remap(u)]=old['nodes'][u]
                    nodes[remap(u)+4]=TOK.node_id(int(fresh[u]))
                original=[(remap(u),remap(v)) for u,v in old['edges']]
                padding=[(remap(u)+4,remap(v)+4) for u,v in sample_edges(4,4,24,rng)]
                rng.shuffle(padding)
                slots=set(map(int,rng.choice(48,24,replace=False)))
                a,b=iter(original),iter(padding)
                edges=[next(a) if i in slots else next(b) for i in range(48)]
                adj=[[] for _ in nodes]
                for u,v in edges:adj[u].append(v)
                rows.append(graph_record(adj,nodes,remap(old['s']),remap(old['t']),
                    old['h'],old['label'],old['index'],edges))
        else:
            forbidden=set(allhash)
            payload={'split':'test','shapes':{'balanced':(layers,width)},'shape_names':['balanced'],
                     'n_nodes':layers*width,'n_edges':m,'node_vocab':256,'max_attempts':10000,
                     'data_seed':20261017+SHAPES.index((case,layers,width,m))*100003,'forbidden':forbidden}
            for h in range(1,layers):
                for y in (0,1):
                    for _ in range(per_label):
                        r=_generate_context_row(payload,(len(rows),'balanced',h,y))
                        forbidden.add(r['struct_hash'])
                        adj=[[] for _ in range(layers*width)]
                        edges=list(zip(r['edge_src'],r['edge_dst']))
                        for u,v in edges:adj[u].append(v)
                        rows.append(graph_record(adj,[TOK.node_id(v) for v in r['node_token']],
                            r['s'],r['t'],h,y,r['sample_id'],edges))
        data[case]=rows
    world=ParametricWorld('data/explore_v1/fixed_standard_n32_m48_onehop_s0/balanced-seed0')
    a=world.queries;ids=world.indices('test')
    chosen=stratified_ids(ids,np.zeros(len(a['label']),dtype=int),a['distance'],a['label'],64,20261017)
    data['fixed0']=[graph_record(world.adj(),world.node_vocab_ids(TOK),int(a['s_topo'][i]),
                        int(a['t_topo'][i]),int(a['distance'][i]),int(a['label'][i]),int(i)) for i in chosen]
    return data


def reference(record,protocol,negative,fixed=False,random_order=False):
    r=record
    prompt=([TOK.id('<QUERY>'),r['nodes'][r['s']],r['nodes'][r['t']],TOK.id('|')] if fixed else
            context_prompt(TOK,r['nodes'],r['edges'],r['s'],r['t']))
    row=render_task(TOK,protocol,r['label'],prompt,r['adj'],r['s'],r['t'],r['nodes'])
    if negative and r['label']==0:
        row=transform(row,protocol,TOK,r['adj'],r['nodes'],r['s'],r['t'],
                      negative_rng=np.random.default_rng(20261017+r['index']))
    if random_order:
        row=transform(row,protocol,TOK,r['adj'],r['nodes'],r['s'],r['t'],
                      search_rng=np.random.default_rng(20261017+r['index']))
    return row


def audit_references(data):
    counts=Counter()
    # Also audit all fixed-world query caches, including all splits, independently
    # of exact-reference comparison and including all prepared path alternatives.
    for seed in range(3):
        w=ParametricWorld(f'data/explore_v1/fixed_standard_n32_m48_onehop_s0/balanced-seed{seed}')
        nodes=w.node_vocab_ids(TOK);adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(w.adj())}
        for protocol in ('path','dfs_state','bfs_state','dfs_nodes','bfs_nodes'):
            cache=PreparedRows(w,protocol,TOK)
            for i in range(len(w.queries['label'])):
                for variant in range(cache.variants(i)):
                    row=cache.row(i,variant)
                    s,t=nodes[int(w.queries['s_topo'][i])],nodes[int(w.queries['t_topo'][i])]
                    verdict=validate_trace(row.tokens[row.prompt_len:row.answer_index],row.label,
                                           protocol,adj,s,t,TOK)
                    assert verdict['valid'],(seed,i,protocol,verdict)
                    counts['fixed_cached_'+protocol]+=1
    for case,records in data.items():
        for name,suffix in CONTEXT:
            if name.startswith('direct'):continue
            protocol=suffix.split('/')[-1]
            for r in records:
                row=reference(r,protocol,name=='path_negative',case=='fixed0')
                verdict=validate_trace(row.tokens[row.prompt_len:row.answer_index],row.label,protocol,
                    r['token_adj'],r['nodes'][r['s']],r['nodes'][r['t']],TOK,name=='path_negative')
                assert verdict['valid'],(case,name,r['index'],verdict)
                counts['generated_reference_'+name]+=1
    return dict(counts)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='studies/length_generalization/results')
    parser.add_argument('--per-label',type=int,default=8)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--models',nargs='*')
    parser.add_argument('--cases',nargs='*')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    data=make_data(args.per_label)
    check=self_test(TOK,dfs_search,bfs_search,render_steps,first_expansion_order)
    audits=audit_references(data)
    print('Reference audit passed',audits,flush=True)
    jobs=[('context/'+name,RUNROOT/'context/standard/train50000'/suffix,
           [x[0] for x in SHAPES]) for name,suffix in CONTEXT]
    fixed_variants=CONTEXT[:1]+CONTEXT[3:]+[('dfs_random','dfs_random/dfs_state')]
    jobs += [('fixed/'+name,RUNROOT/'fixed/standard/seed0'/suffix,['fixed0']) for name,suffix in fixed_variants]
    metadata={'per_label_per_hop':args.per_label,'batch':args.batch,'checkpoint':'latest.pt / 50k',
              'precision':'bf16','max_seq_len':4096,'max_new_tokens':0,'device':torch.cuda.get_device_name(0),
              'validator_checks':check,'reference_audit':audits,'shapes':SHAPES,
              'seed':20261017,'records':{k:len(v) for k,v in data.items()},
              'inputs':{k:[{z:r[z] for z in ('index','s','t','h','label','nodes','edges','longest_graph_path')}
                           for r in rs] for k,rs in data.items()}}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    file=out/'predictions.jsonl'
    done=set()
    if file.exists():
        for line in file.open():
            r=json.loads(line);done.add((r['model'],r['case'],r['index']))
    for modelname,path,cases in jobs:
        if args.models and modelname not in args.models:continue
        cfg=load_config([path/'resolved_config.yaml'])
        protocol=cfg.protocol.name
        model=build_model(cfg.model,len(TOK),TOK.pad).cuda().eval()
        blob=torch.load(path/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert blob['step']==50000
        model.load_state_dict(blob['model']);del blob
        print('MODEL',modelname,flush=True)
        for case in cases:
            if args.cases and case not in args.cases:continue
            records=[r for r in data[case] if (modelname,case,r['index']) not in done]
            if not records:continue
            case_start=time.time()
            correct=valid=0
            for offset in range(0,len(records),args.batch):
                chunk=records[offset:offset+args.batch]
                rows=[reference(r,protocol,cfg.protocol.negative_random_path,case=='fixed0',
                                cfg.protocol.random_search_order) for r in chunk]
                prompts=[row.tokens[:row.prompt_len] for row in rows]
                gens=generate(model,prompts,protocol,TOK,'bf16',max_new_tokens=0)
                with file.open('a') as handle:
                    for r,row,g in zip(chunk,rows,gens):
                        verdict=validate_trace(g['trace_tokens'],g['pred'],protocol,r['token_adj'],
                            r['nodes'][r['s']],r['nodes'][r['t']],TOK,cfg.protocol.negative_random_path)
                        ref=row.tokens[row.prompt_len:row.answer_index]
                        prefix=0
                        for a,b in zip(ref,g['trace_tokens']):
                            if a!=b:break
                            prefix+=1
                        item={'model':modelname,'protocol':protocol,'case':case,'index':r['index'],
                            'hop':r['h'],'label':r['label'],'pred':g['pred'],
                            'correct':g['pred']==r['label'],'format_fail':g['format_fail'],
                            'limit':g['generation_limit_reached'],'trace_valid':verdict['valid'],
                            'trace_error':verdict.get('error'),'valid_records':verdict.get('valid_records'),
                            'semantic_valid_prefix_tokens':verdict.get('valid_prefix_tokens'),
                            'exact_reference':g['trace_tokens']==ref,'reference_prefix_tokens':prefix,
                            'prompt_len':row.prompt_len,'reference_trace_len':len(ref),
                            'reference_fits_context':row.prompt_len+len(ref)+1<=cfg.model.max_seq_len,
                            'generated_trace_len':len(g['trace_tokens']),'trace_tokens':g['trace_tokens'],
                            'source':r['nodes'][r['s']],'target':r['nodes'][r['t']],
                            'longest_graph_path':r['longest_graph_path']}
                        assert not (item['exact_reference'] and item['correct']) or verdict['valid'],item
                        handle.write(json.dumps(item)+'\n')
                        correct+=item['correct'];valid+=item['trace_valid']
                print(f'  {case} {min(offset+args.batch,len(records))}/{len(records)} '
                      f'correct={correct} valid={valid} elapsed={time.time()-case_start:.1f}s',flush=True)
            print('CASE DONE',modelname,case,flush=True)
        del model;torch.cuda.empty_cache()
    print('ALL DONE seconds',time.time()-started,flush=True)


if __name__=='__main__':main()
