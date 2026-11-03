"""Frozen in-context DFS-state: full token attention on a four-node DAG."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ROOT = Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0, str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import context_prompt, render_task
from cot_navigation.evaluation.rollout import generate

OUT = ROOT/'studies/attention_four_node_20261030'
RUN = ROOT/'outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state'

def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config([RUN/'resolved_config.yaml'])
    tok = Tokenizer.from_config(cfg)
    cfg.model.compile = False
    model = build_model(cfg.model, len(tok), tok.pad).eval()
    cp = RUN/'checkpoints/best.pt'
    state = torch.load(cp, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False)
    print('loaded checkpoint step', state['step'], flush=True)
    adj = [[1,3],[2],[],[]]
    edges = [(0,1),(1,2),(0,3)]
    nodes = [tok.node_id(i) for i in range(4)]
    prompt = context_prompt(tok,nodes,edges,0,3)
    ref = render_task(tok,'dfs_state',1,prompt,adj,0,3,nodes)
    with torch.inference_mode():
        free = generate(model,[prompt],'dfs_state',tok,'fp32',max_new_tokens=256)[0]
    generated = free['tokens']
    # Generation stops at YES/NO; EOS is not necessary for trajectory matching.
    expected = ref.tokens[:-1]
    exact = generated == expected
    print('free output',tok.decode(generated),'reference match',exact,flush=True)
    # Always show the entire canonical search; explicitly label conditioning.
    tokens = expected
    with torch.inference_mode():
        logits, weights = model(torch.tensor(tokens)[None],return_attn=True)
        captured = logits.clone()
        normal = model(torch.tensor(tokens)[None])
    att = np.stack([w[0,0].numpy() for w in weights])
    assert np.allclose(att.sum(-1),1,atol=2e-6)
    assert np.max(np.triu(att,1)) == 0
    maxdiff = float((captured-normal).abs().max())
    print('math/SDPA max logit difference', maxdiff, flush=True)
    assert maxdiff < .002
    labels = [chr(65+nodes.index(t)) if t in nodes else tok.name(t).strip('<>') for t in tokens]
    # Record/field boundaries and short, unambiguous labels.
    groups = [(0,len(prompt),'Graph + query')]
    records = []
    cursor = len(prompt)
    for i, step in enumerate(ref.meta['reference_steps']):
        end = tokens.index(tok.id(';'),cursor)+1
        action = step.action + (' '+chr(65+step.arg) if step.arg is not None else '')
        groups.append((cursor,end,f'R{i+1}: {action}'))
        records.append(dict(record=i+1,start=cursor,end=end,stack=[chr(65+x) for x in step.stack],
            visited=[chr(65+x) for x in step.visited],candidates=[chr(65+x) for x in step.cands],action=action))
        cursor = end
    groups.append((cursor,len(tokens),'YES'))
    # Label every token with its absolute position, including all separators.
    ticks = [f'{i}:{name}' for i,name in enumerate(labels)]
    n = len(tokens)
    norm = PowerNorm(gamma=.5,vmin=0,vmax=1)
    cmap = plt.colormaps['magma'].copy()
    cmap.set_bad('#eeeeee')
    def draw(ax,layer,small=False,annotate_groups=True):
        matrix = np.ma.array(att[layer],mask=np.triu(np.ones((n,n),bool),1))
        im = ax.imshow(matrix,cmap=cmap,norm=norm,interpolation='nearest',aspect='equal')
        fs = 3.8 if small else 7
        ax.set_xticks(range(n),ticks,rotation=90,fontsize=fs)
        ax.set_yticks(range(n),ticks,fontsize=fs)
        ax.tick_params(length=0,pad=1)
        for a,b,name in groups:
            if a:
                ax.axhline(a-.5,color='#2cbac8',lw=.45 if small else .9,alpha=.85)
                ax.axvline(a-.5,color='#2cbac8',lw=.45 if small else .9,alpha=.85)
            if not small and annotate_groups:
                ax.text(n+.3,(a+b-1)/2,name,va='center',fontsize=9)
        ax.set_title(f'Layer {layer+1} / head 1',fontsize=11 if small else 15,pad=8)
        ax.set_xlabel('Key position (token attended to)',fontsize=8 if small else 11)
        ax.set_ylabel('Query position (consumed token; predicts NEXT token)',fontsize=8 if small else 11)
        return im
    mode = 'Free generation matches canonical trace exactly' if exact else 'Teacher-forced canonical trace; free generation DOES NOT match'
    title = 'In-context DFS-state | A -> B, B -> C, A -> D | query A -> D'
    fig,axes = plt.subplots(2,3,figsize=(27,19))
    for layer,ax in enumerate(axes.flat): im=draw(ax,layer,True)
    fig.suptitle(title+'\n'+mode+' | CPU FP32 | shared scale: sqrt color mapping, attention 0-1',fontsize=15)
    fig.subplots_adjust(left=.04,right=.94,bottom=.055,top=.94,wspace=.18,hspace=.22)
    cb=fig.colorbar(im,cax=fig.add_axes([.955,.25,.012,.5]),ticks=[0,.01,.04,.1,.25,.5,1])
    cb.set_label('Attention probability')
    fig.savefig(OUT/'all_layers.png',dpi=170)
    plt.close(fig)
    for layer in range(att.shape[0]):
        fig,ax=plt.subplots(figsize=(18,17))
        im=draw(ax,layer)
        fig.suptitle(title+'\n'+mode,fontsize=14,y=.99)
        fig.subplots_adjust(left=.11,right=.84,bottom=.13,top=.94)
        cb=fig.colorbar(im,cax=fig.add_axes([.94,.25,.014,.5]),ticks=[0,.01,.04,.1,.25,.5,1])
        cb.set_label('Attention probability (sqrt color mapping)')
        fig.savefig(OUT/f'layer_{layer+1}.png',dpi=160)
        fig.savefig(OUT/f'layer_{layer+1}.svg')
        plt.close(fig)
    # Show the actual free rollout as well, including any malformed state.
    reference_att = att
    reference_labels = labels
    reference_groups = groups
    reference_n = n
    reference_ticks = ticks
    with torch.inference_mode():
        _, fw = model(torch.tensor(generated)[None],return_attn=True)
    att = np.stack([w[0,0].numpy() for w in fw])
    labels = [chr(65+nodes.index(t)) if t in nodes else tok.name(t).strip('<>') for t in generated]
    n = len(generated)
    ticks = [f'{i}:{name}' for i,name in enumerate(labels)]
    groups = [(0,len(prompt),'Prompt'),(len(prompt),n,'Actual free output')]
    fig,axes=plt.subplots(2,3,figsize=(21,14))
    for layer,ax in enumerate(axes.flat): im=draw(ax,layer,annotate_groups=False)
    fig.suptitle(title+'\nACTUAL GREEDY FREE GENERATION (no reference states supplied)',fontsize=14)
    fig.subplots_adjust(left=.06,right=.93,bottom=.08,top=.92,wspace=.28,hspace=.32)
    fig.colorbar(im,cax=fig.add_axes([.955,.25,.012,.5]),ticks=[0,.01,.04,.1,.25,.5,1])
    fig.savefig(OUT/'free_generation_all_layers.png',dpi=150)
    plt.close(fig)
    np.savez_compressed(OUT/'free_attention.npz',attention=att,tokens=np.array(generated))
    att, labels, groups, n, ticks = reference_att,reference_labels,reference_groups,reference_n,reference_ticks
    # Deterministic relabeling check, not a population-level generalization test.
    rng=np.random.default_rng(20261030)
    checks=[]
    for trial in range(8):
        raw = list(range(4)) if trial==0 else rng.choice(256,4,replace=False).tolist()
        nt=[tok.node_id(i) for i in raw]
        pr=context_prompt(tok,nt,edges,0,3)
        rr=render_task(tok,'dfs_state',1,pr,adj,0,3,nt)
        with torch.inference_mode():
            result=generate(model,[pr],'dfs_state',tok,'fp32',max_new_tokens=256)[0]
        checks.append(dict(raw_node_ids=raw,pred=result['pred'],exact_canonical=result['tokens']==rr.tokens[:-1],
                           generated_tokens=result['tokens'],limit=result['generation_limit_reached']))
        print('relabel',trial,checks[-1]['exact_canonical'],flush=True)
    np.savez_compressed(OUT/'attention.npz',attention=att,tokens=np.array(tokens),prompt_len=len(prompt))
    metadata=dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
        step=int(state['step']),protocol='dfs_state',knowledge='context_graph',device='cpu',precision='fp32',
        node_mapping=dict(zip('ABCD',nodes)),edges=['A->B','B->C','A->D'],query='A->D',
        mode=mode,free_matches_reference=exact,free_result=free,reference_tokens=tokens,
        token_labels=labels,records=records,sequence_length=n,heads=cfg.model.heads,layers=cfg.model.layers,
        max_math_sdpa_logit_difference=maxdiff,relabel_checks=checks,
        caveats=['Four nodes are smaller than training graphs (minimum N16).',
                 'One branching example does not establish statistical representativeness.',
                 'No merge/cross edge: does not exercise rejection of an already-visited successor.',
                 'Attention row i consumes token i and predicts token i+1.',
                 'Color uses a square-root mapping on probabilities 0..1; values are not row-rescaled.'])
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=[title,mode,'','Prompt: '+' '.join(labels[:len(prompt)]),'',
           'Record format: stack | visited | candidates | action [node] ;']
    for rec in records:
        lines.append(f"R{rec['record']}: {' '.join(rec['stack'])} | {' '.join(rec['visited'])} | {' '.join(rec['candidates'])} | {rec['action']} ;")
    lines+=['YES','',f'Checkpoint: {cp}',f'Step: {state["step"]}',
            f'Relabel exact canonical matches: {sum(c["exact_canonical"] for c in checks)}/8',
            f'Math vs SDPA maximum logit difference: {maxdiff}',*metadata['caveats']]
    (OUT/'trace.txt').write_text('\n'.join(lines)+'\n')
    payload=json.dumps(dict(reference=dict(attention=att.tolist(),labels=labels,prompt_len=len(prompt)),
        free=dict(attention=np.stack([w[0,0].numpy() for w in fw]).tolist(),
                  labels=[chr(65+nodes.index(t)) if t in nodes else tok.name(t).strip('<>') for t in generated],
                  prompt_len=len(prompt))))
    html='''<!doctype html><html><meta charset="utf-8"><title>Four-node DFS attention</title>
<style>body{font:15px system-ui;margin:24px;background:#fafafa;color:#222}select{padding:6px;margin-right:16px}#info{padding:12px;background:#e7eef4;position:sticky;top:0}svg{background:white} .tip{font-size:13px;color:#555}</style>
<h2>In-context DFS-state: A→B, B→C, A→D; query A→D</h2>
<p>Checkpoint: scale_v1 context curriculum DFS-state, best step 185000, CPU FP32, one head/layer.</p>
<p><b>The actual free output is incorrect (NO). Reference mode supplies the correct full CoT.</b><br>
This N4 example is below the N16 minimum training size. All 8 fixed node renamings predicted NO.</p>
<label>Trace <select id="mode"><option value="reference">Full canonical reference (teacher forced)</option><option value="free">Actual greedy free generation</option></select></label>
<label>Layer <select id="layer">''' + ''.join(f'<option value="{i}">{i+1}</option>' for i in range(6)) + '''</select></label>
<p class="tip">Rows = consumed/query tokens; columns = attended/key tokens. Row i predicts token i+1. Hover any cell for exact probability. Colors use sqrt(probability); gray = future masked. Cyan lines = prompt/record boundaries. Scroll or zoom to inspect.</p>
<div id="info">Hover a cell to inspect attention.</div><div id="plot"></div><script>
const DATA=PAYLOAD;
const mode=document.getElementById('mode'),layer=document.getElementById('layer');
function render(){const d=DATA[mode.value],a=d.attention[+layer.value],n=a.length,s=16,p=100;
const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('width',p+n*s+30);svg.setAttribute('height',p+n*s+30);
function el(t,attrs){const e=document.createElementNS(ns,t);for(const k in attrs)e.setAttribute(k,attrs[k]);svg.appendChild(e);return e;}
for(let i=0;i<n;i++){let t=el('text',{x:p-6,y:p+i*s+12,'text-anchor':'end','font-size':10});t.textContent=i+':'+d.labels[i];
t=el('text',{transform:`translate(${p+i*s+12},${p-6}) rotate(-90)`,'font-size':10});t.textContent=i+':'+d.labels[i];
for(let j=0;j<n;j++){let v=a[i][j],q=Math.sqrt(v),c=i<j?'#eee':`rgb(${Math.round(15+240*q)},${Math.round(9+225*q*q)},${Math.round(25+110*q)})`;
let r=el('rect',{x:p+j*s,y:p+i*s,width:s,height:s,fill:c});r.onmouseenter=()=>{document.getElementById('info').textContent=`Layer ${+layer.value+1} | query ${i}:${d.labels[i]} → key ${j}:${d.labels[j]} | ${i<j?'MASKED':v.toFixed(8)} | next token: ${d.labels[i+1]||'(end)'}`;};}}
let boundaries=[d.prompt_len];d.labels.forEach((t,i)=>{if(t===';')boundaries.push(i+1)});
for(const b of boundaries){el('line',{x1:p+b*s,y1:p,x2:p+b*s,y2:p+n*s,stroke:'#27b6c8','stroke-width':1});el('line',{x1:p,y1:p+b*s,x2:p+n*s,y2:p+b*s,stroke:'#27b6c8','stroke-width':1});}
document.getElementById('plot').replaceChildren(svg);}
mode.onchange=render;layer.onchange=render;render();</script></html>'''
    (OUT/'interactive.html').write_text(html.replace('PAYLOAD',payload),encoding='utf-8')
    print(json.dumps({k:metadata[k] for k in ['mode','sequence_length','step','max_math_sdpa_logit_difference']}),flush=True)

if __name__=='__main__': main()
