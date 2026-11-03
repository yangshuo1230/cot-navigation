"""Supplement node-only traces with exact actually exposed WALK matches."""
import json
import hashlib
import run

run.torch.set_num_threads(2)
for setting in ('simple','hard'):
    for protocol in ('dfs_nodes','bfs_nodes'):
        dest=run.OUT/f'{setting}_{protocol}.json';result=json.loads(dest.read_text())
        folder=run.RUNS[setting]/protocol
        cfg=run.load_config([folder/'resolved_config.yaml']);tok=run.Tokenizer.from_config(cfg)
        world=run.ParametricWorld(run.Path(cfg.data.root)/f'{cfg.data.graph_shape}-seed{cfg.data.graph_seed}')
        nodes=world.node_vocab_ids(tok);adj=world.adj();q=world.queries
        blob=run.torch.load(folder/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        weights=run.counts(blob['walk_sampler'])
        assert weights.sum()==blob['walks_seen']
        # All underlying WALK items have already been sampled in Stage B;
        # hence their union also covers the same-world knowledge-stage bank.
        assert len(weights)==world.n_walks() and (weights>0).all()
        bank={run.digest([nodes[v] for v in world.walk_nodes_at(i)]) for i in range(world.n_walks())}
        model=run.build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
        model.load_state_dict(blob['model']);del blob;run.enable(model)
        header=[tok.id('<WORLD_0>')] if cfg.train.world_roots else []
        prompts=[];replays=[];tokenadj={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(adj)}
        for i in result['panel_ids']:
            s,t=int(q['s_topo'][i]),int(q['t_topo'][i])
            prompts.append(header+run.parametric_prompt(tok,nodes,s,t))
            replays.append(run.SequenceReplay(tokenadj,nodes[s],nodes[t],int(q['label'][i]),protocol,tok,run.legality.previous.sequence_audit))
        generated=run.generate_checked(model,prompts,tok,cfg.train.precision,replays,cfg.protocol.cot_margin)
        matched=0;union=0;completed=0
        for old,new in zip(result['rows'],generated):
            assert old['query_id'] in result['panel_ids']
            assert old['tokens']==len(new['trace']) and old['complete']==(new['pred'] is not None)
            old['exact_walk']=run.digest(new['trace']) in bank
            old['exact_cot_or_walk']=old['exact_tokens'] or old['exact_walk']
            if old['complete'] and not old['interrupted']:
                completed+=1;matched+=old['exact_walk'];union+=old['exact_cot_or_walk']
        result['walk_supplement']=dict(actual_task_walk_exposures=int(weights.sum()),walk_items=len(weights),all_items_actually_sampled=True,
            completed=completed,exact_walk_count=matched,exact_walk_rate=matched/completed,
            cot_or_walk_count=union,cot_or_walk_rate=union/completed,
            definition='Exact whole node sequence, not arbitrary substring. All graph WALK items sampled in Stage B.')
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(setting,protocol,result['walk_supplement'],flush=True)
        del model;run.torch.cuda.empty_cache()
