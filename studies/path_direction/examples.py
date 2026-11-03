"""Two matched-target real examples: layerwise probabilities on true neighbors."""
from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@torch.inference_mode()
def main():
    results={};fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,setting in enumerate(('simple','hard')):
        r=json.loads((HERE/f'{setting}.json').read_text());e=r['target_pairs']['examples'][0]
        run=Path(r['manifest']['run']);cfg=load_config([run/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
        world=ParametricWorld(Path(cfg.data.root)/'main-seed0');adj=world.adj();nodes=world.node_vocab_ids(tok)
        names=[tok.name(t) for t in nodes];u=names.index(e['current']);ts=[names.index(e[k]) for k in ('target_a','target_b')]
        model=build_model(cfg.model,len(tok),tok.pad).to('cuda:1').eval().requires_grad_(False)
        model.load_state_dict(torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
        header=[tok.id('<WORLD_0>')] if cfg.train.world_roots else []
        seqs=[header+[tok.id('<QUERY>'),nodes[u],nodes[t],tok.id('|'),nodes[u]] for t in ts]
        ids=torch.tensor(seqs,device='cuda:1');capture={}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            model(ids,capture=capture)
            hidden=torch.stack([model.embed(ids)]+capture['resid'])[:,:,-1]
            logits=model.lm_head(model.final_norm(hidden.float()))
        logits=logits[...,np.asarray(nodes)[adj[u]]].float()
        prob=logits.softmax(-1).cpu().numpy();entry=[]
        for col,t in enumerate(ts):
            ax=axes[row,col]
            for j,v in enumerate(adj[u]):ax.plot(range(7),prob[:,col,j],'o-',label=names[v]+(' (useful)' if names[v] in e['useful_a' if col==0 else 'useful_b'] else ' (dead)'))
            ax.set(title=f'{setting}: {names[u]} -> {names[t]}',xticks=range(7),xticklabels=['E']+[f'L{i}' for i in range(1,7)],ylim=(-.03,1.03),ylabel='Probability conditioned on true successors')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
            entry.append(dict(prompt=tok.decode(seqs[col]),target=names[t],successors=[names[v] for v in adj[u]],
                useful=e['useful_a' if col==0 else 'useful_b'],probabilities_by_layer=prob[:,col].tolist(),
                native_generated_next=e['next_a' if col==0 else 'next_b']))
        results[setting]=entry;del model;torch.cuda.empty_cache()
    write(HERE/'examples.json',results);fig.savefig(HERE/'matched_targets.png',dpi=170)

if __name__=='__main__':main()
