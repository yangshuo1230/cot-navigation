"""Can query target / separator states expose the target's strict ancestors?

Native SDPA frozen inference on GPU 1. Affine ridge fits on GPU 0. No saved
activations/weights. Original query split plus target-identity held-out tests.
Strict graph ancestors exclude the target itself. Metrics additionally exclude
the query source, so reading/copying either endpoint cannot count as success.
"""
from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=HERE/'ancestor_readout';OUT.mkdir(exist_ok=True)

def masks(target,source,n):
    candidate=np.ones((len(target),n),bool)
    candidate[np.arange(len(target)),target]=False
    candidate[np.arange(len(target)),source]=False
    before=candidate&(np.arange(n)[None,:]<target[:,None])
    return {'all_other_nodes':candidate,'earlier_topological_nodes':before}

def metrics(y,s,mask,threshold):
    result=ranking(y,s,mask)
    p=(s>=threshold)&mask;truth=y.astype(bool)&mask
    tp=int((p&truth).sum());fp=int((p&~truth).sum());fn=int((~p&truth).sum())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    result.update(precision=precision,recall=recall,f1=2*tp/max(2*tp+fp+fn,1),
        exact_set=float((p==truth).all(1).mean()),threshold=float(threshold))
    return result

def evaluate(y,p,split,maskdict,target):
    tr,va,te=split;out=[]
    # Compare a FIXED candidate node across different test targets. Aggregate
    # repeated queries by target identity first. Constant ancestor-frequency
    # scores then have exactly AUROC=.5 rather than receiving credit for globally
    # popular ancestors. Only candidates with both reachable/unreachable test
    # targets contribute; report their count explicitly.
    grouped={}
    for name,mask in maskdict.items():
        ss=[];yy=[];valid=[]
        for t in np.unique(target[te]):
            ids=te[target[te]==t];mm=mask[ids];counts=mm.sum(0)
            score=(p[ids].astype(np.float64)*mm[:,None,:]).sum(0)/np.maximum(counts,1)[None,:]
            ss.append(np.round(score,9));yy.append(y[ids[0]]);valid.append(counts>0)
        grouped[name]=(np.asarray(ss),np.asarray(yy),np.asarray(valid))
    for l in range(p.shape[1]):
        s=p[:,l];v=s[va];mask=maskdict['all_other_nodes'][va];yy=y[va].astype(bool)&mask
        # A single validation-only threshold for the complete ancestor set.
        candidates=np.unique(np.r_[0,np.quantile(v[mask],np.linspace(0,1,41))])
        scores=[]
        for threshold in candidates:
            pred=(v>=threshold)&mask;tp=(pred&yy).sum();fp=(pred&~yy).sum();fn=(~pred&yy).sum()
            scores.append(2*tp/max(2*tp+fp+fn,1))
        threshold=candidates[int(np.argmax(scores))]
        entry={}
        for name,mask in maskdict.items():
            value=metrics(y[te],s[te],mask[te],threshold)
            gs,gy,gm=grouped[name];aucs=[]
            for node in range(y.shape[1]):
                keep=gm[:,node];a=linear.auc(gy[keep,node],gs[keep,l,node])
                if a is not None:aucs.append(a)
            value['fixed_candidate_across_targets']=dict(n_candidates=len(aucs),n_targets=len(gy),
                auc=float(np.mean(aucs)) if aucs else None)
            entry[name]=value
        out.append(entry)
    return out

@torch.inference_mode()
def collect(setting):
    old=json.loads((HERE/f'{setting}.json').read_text());run=Path(old['manifest']['run'])
    assert hashlib.sha256((run/'checkpoints/best.pt').read_bytes()).hexdigest()==old['manifest']['checkpoint_sha256']
    cfg=load_config([run/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');adj=world.adj();n=len(adj)
    reach=np.eye(n,dtype=bool)
    for u in reversed(range(n)):
        for v in adj[u]:reach[u]|=reach[v]
    # Independent BFS verifies every reachability target.
    for u in range(n):
        seen={u};todo=[u]
        for v in todo:
            for w in adj[v]:
                if w not in seen:seen.add(w);todo.append(w)
        assert seen==set(np.flatnonzero(reach[u]))
    reach[np.arange(n),np.arange(n)]=False
    rows=[json.loads(l) for l in gzip.open(OLD/f'{setting}_path_queries.jsonl.gz','rt')]
    nodes=world.node_vocab_ids(tok);source=np.array([r['source'] for r in rows]);target=np.array([r['target'] for r in rows])
    split=np.array([r['split'] for r in rows]);y=reach[:,target].T.astype(np.float32)
    header=[tok.id('<WORLD_0>')] if cfg.train.world_roots else []
    seq=[header+[tok.id('<QUERY>'),nodes[s],nodes[t],tok.id('|')] for s,t in zip(source,target)]
    ids=torch.tensor(seq,device='cuda:1');nl=cfg.model.layers+1
    model=build_model(cfg.model,len(tok),tok.pad).to('cuda:1').eval().requires_grad_(False)
    model.load_state_dict(torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
    x=[];lens=[];head=[]
    for start in range(0,len(rows),128):
        tokens=ids[start:start+128];cap={}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            model(tokens,capture=cap)
            hidden=torch.stack([model.embed(tokens)]+cap['resid'])[:,:,-2:,:]
            scores=model.lm_head(model.final_norm(hidden.float()))[...,nodes]
        x.append(hidden.permute(1,2,0,3).float().cpu().numpy())
        head.append(scores.permute(1,2,0,3).float().cpu().numpy())
    x=np.concatenate(x);head=np.concatenate(head);emb=model.embed.weight.detach().float().cpu().numpy()
    del model;torch.cuda.empty_cache()
    return old['manifest'],x,head,y,source,target,split,emb[np.asarray(nodes)],nl

def run(setting):
    manifest,x,head,y,source,target,split,emb,nl=collect(setting);n=y.shape[1]
    maskdict=masks(target,source,n);qsplit=[np.flatnonzero(split==i) for i in range(3)]
    random=np.random.default_rng(2026).normal(0,.02,emb.shape).astype(np.float32)
    features={'target_and_separator':x.reshape(len(x),2*nl,-1),
        'target_embedding':emb[target],'random_target_identity':random[target],
        'source_target_onehot':np.concatenate([np.eye(n)[source],np.eye(n)[target]],1).astype(np.float32)}
    out=dict(manifest=manifest,n_queries=len(x),n_nodes=n,roles=['target','separator'],layers=['E']+[f'L{i}' for i in range(1,nl)],
        target='strict ancestors A(t)={u:u != t and u can reach t}',
        metric_notes='All metrics exclude query source and target. Earlier-topological-node subset rules out simply ranking every later node as negative. Exact set uses a validation-only F1 threshold. Query rows share target labels; do not treat them as independent node samples.',
        query_split={},target_heldout=[],original_output_head=[])
    for name,xx in features.items():
        pred,ridge=fit(xx,y,qsplit);out['query_split'][name]=dict(ridge=ridge,metrics=evaluate(y,pred,qsplit,maskdict,target))
    # The original head requires no probe training; choose set threshold on val.
    out['original_output_head']=evaluate(y,head.reshape(len(x),2*nl,n),qsplit,maskdict,target)
    out['endpoint_identity_probes']={}
    for label,identities in [('source',source),('target',target)]:
        pred,ridge=fit(features['target_and_separator'],np.eye(n,dtype=np.float32)[identities],qsplit)
        te=qsplit[2]
        out['endpoint_identity_probes'][label]=dict(ridge=ridge,
            top1_accuracy=(pred[te].argmax(-1)==identities[te,None]).mean(0).tolist(),
            note='Original query split; diagnoses endpoint identity access, not reachability computation.')
    for seed in (17,31,47):
        rng=np.random.default_rng(seed);order=rng.permutation(n);fold=np.zeros(n,int)
        fold[order[int(.6*n):int(.8*n)]]=1;fold[order[int(.8*n):]]=2
        sp=[np.flatnonzero((split==i)&(fold[target]==i)) for i in range(3)]
        entry=dict(seed=seed,query_counts=list(map(len,sp)),target_counts=[len(np.unique(target[ii])) for ii in sp],features={})
        for name in ('target_and_separator','target_embedding','random_target_identity'):
            pred,ridge=fit(features[name],y,sp);entry['features'][name]=dict(ridge=ridge,metrics=evaluate(y,pred,sp,maskdict,target))
        # No target identity: per-output ancestor frequency in probe training.
        freq=np.broadcast_to(y[sp[0]].mean(0),(len(y),n)).copy()[:,None,:]
        entry['features']['ancestor_frequency']=dict(metrics=evaluate(y,freq,sp,maskdict,target))
        out['target_heldout'].append(entry)
        print(setting,'target heldout',seed,'done',flush=True)
    out['verification']=dict(all_sources_reachability_matches_bfs=True,role_positions='target=-2, separator=-1; prompt only; no generated or answer tokens',
        frozen_checkpoint_hash_matches_prior_study=True)
    write(OUT/f'{setting}.json',out);return out

def summarize(results):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    lines=['# 目标与分隔符：祖先集合读出','',
        '结论：目标 token 的 embedding 和隐藏状态具有可推广到 probe 未见目标的祖先相关线性信号；不能据此声称完整祖先集合被精确编码。分隔符弱得多，尤其困难图 L1 在固定候选、跨目标对照中接近机会水平，因此不能把困难图 L2 读取分隔符直接解释为读取祖先集合。较深层分隔符可出现较弱且受划分影响的目标相关信号。',
        '',
        '问题：是否能从查询目标 token 或分隔符的隐藏状态读出 A(t)={u:u≠t 且 u 能到达 t}？固定 N100 简单图/困难图的原最佳 Path checkpoint。只输入 prompt，提取目标与分隔符位置的 E、各层 block 后 activation；没有 CoT、答案或未来信息。冻结前向 GPU1，仿射 ridge 拟合 GPU0。',
        '', '普通 query 划分下，目标身份查表可能使分数虚高。主要结果采用同时留出原 query 与目标节点身份的探针对照，3 个随机划分 seed；原模型训练仍见过全部节点。所有指标排除当前 query 的 source 和 target；另外只在拓扑序早于目标的节点中评估，减少仅靠拓扑先后关系的捷径。',
        '', '| 图 | 特征 | 前驱候选内 AUROC E/L1/L2/L4/L6 | L1 全节点 AUROC / 集合F1 / 完全集合正确率 |','|---|---|---|---|']
    notes=[]
    for col,(setting,r) in enumerate(results.items()):
        ax=axes[0,col];contrast_ax=axes[1,col]
        nl=len(r['layers'])
        for role,name,color in [(0,'target','tab:blue'),(1,'separator','tab:orange')]:
            curves=np.array([[v['features']['target_and_separator']['metrics'][role*nl+l]['earlier_topological_nodes']['auc'] for l in range(nl)] for v in r['target_heldout']])
            means=curves.mean(0);std=curves.std(0)
            ax.errorbar(range(nl),means,yerr=std,marker='o',color=color,label=name+' (target held out, mean +/- sd)')
            contrast=np.array([[v['features']['target_and_separator']['metrics'][role*nl+l]['earlier_topological_nodes']['fixed_candidate_across_targets']['auc'] for l in range(nl)] for v in r['target_heldout']])
            contrast_ax.errorbar(range(nl),contrast.mean(0),yerr=contrast.std(0),marker='o',color=color,label=name)
            idx=role*nl+1;vals=[]
            for metric in ('auc','f1','exact_set'):
                vals.append(np.mean([v['features']['target_and_separator']['metrics'][idx]['all_other_nodes'][metric] for v in r['target_heldout']]))
            lines.append(f"| {setting} | {name} | {' / '.join(f'{means[l]:.3f}' for l in (0,1,2,4,6))} | {' / '.join(f'{z:.3f}' for z in vals)} |")
        for key,label,color in [('random_target_identity','Random target identity','tab:red'),('ancestor_frequency','Ancestor frequency','grey')]:
            vals=[v['features'][key]['metrics'][0]['earlier_topological_nodes']['auc'] for v in r['target_heldout']]
            ax.axhline(np.mean(vals),ls='--',color=color,label=label)
            notes.append(f"- {setting} {label}：留出目标测试中前驱候选内 AUROC {np.mean(vals):.3f}。")
            cv=[v['features'][key]['metrics'][0]['earlier_topological_nodes']['fixed_candidate_across_targets']['auc'] for v in r['target_heldout']]
            contrast_ax.axhline(np.mean(cv),ls='--',color=color,label=label)
        ax.set(title=setting+': rank candidate nodes within each target',xticks=range(nl),xticklabels=r['layers'],ylim=(.35,1.02),ylabel='AUROC (earlier topological nodes only)')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
        contrast_ax.set(title=setting+': fixed candidate, compare different targets',xticks=range(nl),xticklabels=r['layers'],ylim=(.25,1.05),ylabel='Macro candidate AUROC across held-out targets')
        contrast_ax.grid(alpha=.2);contrast_ax.legend(fontsize=7)
        notes.append(f"- {setting}：三个划分的 train/val/test 目标数量分别为 {[v['target_counts'] for v in r['target_heldout']]}。")
    fig.savefig(OUT/'ancestor_readout.png',dpi=180);plt.close(fig)
    lines+=['']+notes+['','## 更直接的目标依赖对照','',
        '固定候选节点 u，比较不同测试目标 t，检查读出分数是否在 u 能到达 t 时更高；只比较 u 拓扑序早于 t 的情况。先按目标身份聚合重复 query，再对每个有正负目标的候选计算 AUROC。每个节点固定的祖先频率在此指标中恰为 .5。所有测试目标身份都没有参与 probe 拟合；3 seed 均值及标准差，不是显著性检验。',
        '', '| 图 | 特征 | E | L1 | L2 | L4 | L6 |','|---|---|---|---|---|---|---|']
    for setting,r in results.items():
        nl=len(r['layers'])
        for role,name in ((0,'target'),(1,'separator')):
            vals=[]
            for l in (0,1,2,4,6):
                scores=[v['features']['target_and_separator']['metrics'][role*nl+l]['earlier_topological_nodes']['fixed_candidate_across_targets']['auc'] for v in r['target_heldout']]
                vals.append(f'{np.mean(scores):.3f} ± {np.std(scores):.3f}')
            lines.append('| '+setting+' | '+name+' | '+' | '.join(vals)+' |')
    lines+=['','| 图 | 随机目标身份 AUROC | 频率基线 AUROC | L1 分隔符 source / target 身份分类准确率 |','|---|---|---|---|']
    for setting,r in results.items():
        vals=[]
        for key in ('random_target_identity','ancestor_frequency'):
            scores=[v['features'][key]['metrics'][0]['earlier_topological_nodes']['fixed_candidate_across_targets']['auc'] for v in r['target_heldout']]
            vals.append(f'{np.mean(scores):.3f} ± {np.std(scores):.3f}')
            if key=='ancestor_frequency':assert all(abs(s-.5)<1e-9 for s in scores)
        n=len(r['layers']);identity=r.get('endpoint_identity_probes',{})
        access=' / '.join(f"{identity[k]['top1_accuracy'][n+1]:.3f}" for k in ('source','target')) if identity else 'not measured'
        lines.append('| '+setting+' | '+' | '.join(vals)+' | '+access+' |')
    lines+=['','普通 query 留出、原始输出头、完整候选节点指标、前驱子集指标及集合 precision/recall/F1/完全正确率均见 simple.json、hard.json。分数为 affine ridge 读出分数，不是模型概率。AUROC 大于 .5 表示排序上可区分，不代表完整祖先集合被精确恢复。误差条是三个节点划分的标准差，不是置信区间。祖先相关读出还可能部分依赖深度、拓扑秩等结构统计，本次未排除这些解释。',
        '', '注意力层 l 读取的是前一层 l-1 的 token 状态：简单图 L2 读取目标对应目标 L1；困难图 L2 读取分隔符对应分隔符 L1。可读出说明该位置提供相关信息的可能性，尚不能证明该注意力通路实际使用了这个集合，需后续 value/activation patch 做因果验证。',
        '', '复现：.venv/bin/python studies/path_direction/ancestor_readout.py。未保存 activation、probe 或模型 checkpoint。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':
    if '--summarize' in sys.argv:
        summarize({s:json.loads((OUT/f'{s}.json').read_text()) for s in ('simple','hard')})
    else:summarize({s:run(s) for s in ('simple','hard')})
