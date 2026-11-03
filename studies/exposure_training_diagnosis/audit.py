"""Read-only diagnosis of all completed adj_v3 training outputs."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'studies/exposure_training_diagnosis'
SUITE=ROOT/'studies/fixed_adjacency_suite';MODELS=ROOT/'outputs/exposure_grokking_adj_v3'
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.protocols.tokenizer import Tokenizer

def metrics(value):return {key:value[key] for key in ('n','accuracy','balanced_accuracy','yes_recall','no_recall')}

def main():
    tok=Tokenizer(256);jobs=[];world_stats=[]
    for graph in range(3):
        w=json.loads((SUITE/f'data/graph_{graph}.json').read_text());n=w['n_nodes'];nodes=[tok.node_id(v) for v in w['node_names']];inverse={v:i for i,v in enumerate(nodes)}
        world_stats.append(dict(graph=graph,n_nodes=n,n_edges=w['n_edges'],
            base_counts={key:len(v) for key,v in w['base'].items()},
            base_train_gap_counts={str(g):sum(r[3]==g for r in w['base']['train']) for g in sorted(set(r[3] for r in w['base']['train']))},
            core_gap_counts={str(g):sum(r[3]==g for r in w['core_final']) for g in sorted(set(r[3] for r in w['core_final']))},
            exposure_audit=w['audit']))
        for seed in range(2):
            for protocol in ('direct','path'):
                for arm in ('R0','R2','R3'):
                    folder=MODELS/f'fixed/g{graph}_s{seed}/{protocol}/{arm}'
                    job=json.loads((folder/'resolved_job.json').read_text());final=json.loads((folder/'final.json').read_text())
                    evaluation=json.loads((folder/'eval_000125000.json').read_text())
                    history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
                    curves=[]
                    for file in sorted(folder.glob('eval_*.json')):
                        value=json.loads(file.read_text());curves.append(dict(step=value['step'],train=metrics(value['train']),monitor=metrics(value['monitor'])))
                    adjacency=evaluation['adjacency'];small={k:v for k,v in adjacency.items() if not isinstance(v,(list,dict))}
                    lengths={};legality={}
                    for label in (0,1):
                        generated=[r for r in final['predictions'] if r['query'][2]==label];counts=[];first=[];legal=[];goal=[]
                        for r in generated:
                            trace=[v for v in r['generated'] if v not in (tok.yes,tok.no,tok.id('<EOS>'))]
                            counts.append(len(trace));s,t=r['query'][:2];path=[inverse.get(v,-1) for v in trace]
                            first.append(bool(path and path[0]==s))
                            valid=bool(path and all(v>=0 for v in path))
                            if valid:valid=all(v in w['adj'][u] for u,v in zip(path,path[1:]))
                            legal.append(valid);goal.append(bool(valid and path[0]==s and path[-1]==t))
                        lengths[str(label)]=dict(n=len(counts),no_trace=int(sum(c==0 for c in counts)),mean=float(np.mean(counts)),max=max(counts))
                        legality[str(label)]=dict(first_is_source=float(np.mean(first)),all_edges_legal=float(np.mean(legal)),legal_goal_end=float(np.mean(goal)))
                    jobs.append(dict(id=job['id'],graph=graph,model_seed=seed,protocol=protocol,arm=arm,job=job,
                        final=metrics(final),train_panel=metrics(evaluation['train']),monitor=metrics(evaluation['monitor']),adjacency=small,
                        last_training={k:history[-1][k] for k in ('loss','answer_loss','knowledge_loss','lr','grad_norm')},
                        tail_mean_loss=float(np.mean([r['loss'] for r in history[-20:]])),curves=curves,
                        generated_lengths=lengths,generated_legality=legality))
    grouped={}
    for protocol in ('direct','path'):
        grouped[protocol]={}
        for arm in ('R0','R2','R3'):
            selected=[j for j in jobs if j['protocol']==protocol and j['arm']==arm]
            entry={}
            for source,keys in [('final',('balanced_accuracy','yes_recall','no_recall')),('train_panel',('accuracy',)),('adjacency',('exact_set_accuracy',))]:
                for key in keys:entry[source+'_'+key]=float(np.mean([j[source][key] for j in selected]))
            entry['tail_mean_loss']=float(np.mean([j['tail_mean_loss'] for j in selected]))
            if protocol=='path':
                for label in ('0','1'):
                    for key in ('first_is_source','all_edges_legal','legal_goal_end'):
                        entry['class'+label+'_'+key]=float(np.mean([j['generated_legality'][label][key] for j in selected]))
                    entry['class'+label+'_no_trace_rate']=float(np.mean([j['generated_lengths'][label]['no_trace']/j['generated_lengths'][label]['n'] for j in selected]))
            grouped[protocol][arm]=entry
    result=dict(worlds=world_stats,jobs=jobs,summary=grouped,
        current_train_suite_sha256=hashlib.sha256((SUITE/'train_suite.py').read_bytes()).hexdigest())
    OUT.mkdir(exist_ok=True);(OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(grouped,indent=2))

if __name__=='__main__':main()
