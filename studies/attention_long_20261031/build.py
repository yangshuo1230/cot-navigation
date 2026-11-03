"""Self-contained long native-search examples; full FP32 attention, no video."""
import base64,gzip,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'attention_success_20261030'

LOADER='''const packed=Uint8Array.from(atob('__PACKED__'),c=>c.charCodeAt(0));
const unpacked=await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
const flat=new Float32Array(unpacked),N=D.labels.length;
if(flat.length!==6*N*N)throw Error('Attention shape mismatch');
D.attention=Array.from({length:6},(_,l)=>Array.from({length:N},(_,q)=>Array.from(flat.subarray((l*N+q)*N,(l*N+q+1)*N))));
document.getElementById('loading')?.remove();
'''

MATRIX='''function renderMatrix(){
if(!$('matrix').closest('details').open||state.l===0){if(state.l===0)$('matrix').innerHTML='<p>L0没有attention矩阵。</p>';return}
if(matrixLayer===state.l)return;matrixLayer=state.l;const n=D.labels.length,mat=D.attention[state.l-1],canvas=document.createElement('canvas');
canvas.width=n;canvas.height=n;canvas.style.cssText='width:'+n+'px;height:'+n+'px;image-rendering:pixelated;cursor:crosshair';
const ctx=canvas.getContext('2d'),im=ctx.createImageData(n,n);
for(let q=0;q<n;q++)for(let k=0;k<n;k++){let i=(q*n+k)*4,v=Math.sqrt(mat[q][k]);im.data[i]=k>q?232:246-226*v;im.data[i+1]=k>q?236:249-114*v;im.data[i+2]=k>q?240:250-98*v;im.data[i+3]=255}
ctx.putImageData(im,0,0);let note=document.createElement('p');note.textContent='完整 '+n+' × '+n+' 矩阵；行=query，列=key。移动查看位置和分数，点击追踪。';
const pair=e=>{let r=canvas.getBoundingClientRect();return [Math.min(n-1,Math.floor((e.clientY-r.top)*n/r.height)),Math.min(n-1,Math.floor((e.clientX-r.left)*n/r.width))]};
canvas.onmousemove=e=>{let [q,k]=pair(e);canvas.title=k>q?'未来遮罩':`L${state.l} #${q} ${label(q)} → #${k} ${label(k)}: ${(mat[q][k]*100).toFixed(6)}%`};
canvas.onclick=e=>{let [q,k]=pair(e);if(k<=q){locate(q);peek({p:k,w:mat[q][k]})}};
$('matrix').replaceChildren(note,canvas);
}
'''

def explorer(data,packed,name):
    s=(OLD/'token_explorer_template.html').read_text(encoding='utf8');n=len(data['labels']);summary=data['summary']
    s=s.replace('实际自由生成 · YES 正确 · 全轨迹合法',f'实际自由生成 · {name.upper()}正确 · {summary["records"]}条记录合法')
    s=s.replace('完整序列 · 116 tokens',f'完整序列 · {n} tokens')
    s=s.replace('<a href="search_animation.html">搜索动画 ↗</a>',f'<a href="{data["animation_url"]}">搜索过程 ↗</a> <a href="token_explorer.html">YES长实例</a> <a href="token_explorer_no.html">NO穷尽实例</a>')
    s=s.replace('选一个具体 token → 看分类和每个来源 → 点“追上游”继续查历史。重复字母用绝对位置、记录、字段区分。',f'N100 / 150边 · {summary["records"]}条记录 · 最大深度{summary["max_depth"]} · {summary["pushes"]}次PUSH/{summary["pops"]}次POP · 访问{summary["visited"]}节点。完整原始attention，不作距离校正。')
    start=s.index('function buildTokenlist()');end=s.index('\nfunction highlightTokens',start)
    s=s[:start]+'''function buildTokenlist(){const list=$('tokenlist');list.replaceChildren();const sections=[];
for(let i=0;i<D.graph.edges.length;i++){let [u,v]=D.graph.edges[i];sections.push({name:`边 ${i+1} · ${D.graph.labels[u]}→${D.graph.labels[v]}`,lo:i*3,hi:i*3+3})}
sections.push({name:`查询 ${D.source}→${D.target}`,lo:D.prompt_len-4,hi:D.prompt_len});
for(let rec of D.records)sections.push({name:`R${rec.record} · ${rec.action}`,lo:rec.start,hi:rec.end});sections.push({name:'最终答案',lo:D.labels.length-1,hi:D.labels.length});
for(let section of sections){let box=document.createElement('details');box.open=section.lo>=D.prompt_len;box.className='tokenblock';box.dataset.search=(section.name+' '+D.labels.slice(section.lo,section.hi).join(' ')+' '+D.info.slice(section.lo,section.hi).map(x=>x.description).join(' ')).toLowerCase();let sm=document.createElement('summary');sm.textContent=section.name;box.appendChild(sm);for(let p=section.lo;p<section.hi;p++)box.appendChild(tokenButton(p,()=>locate(p)));list.appendChild(box)}}''' +s[end:]
    s=s.replace('let state={p:D.default_position,l:4',"let state={p:D.default_position,l:Math.max(0,Math.min(6,+(hash.get('l')??4)))")
    s=s.replace('p<72','p<D.prompt_len-4').replace('state.p<76','state.p<D.prompt_len').replace("if(!/^[A-P]$/.test(token))return","if(!D.graph.labels.includes(token))return")
    start=s.index('function renderMatrix()');end=s.index('\nfunction render()',start);s=s[:start]+MATRIX+s[end:]
    foot=f'N100测试query {summary["index"]}；同一185k冻结模型。只给图和查询的原生生成，整条轨迹replay合法，完整forward逐token argmax与缓存自由生成一致。展示两条已知成功轨迹，不代表总体成功率。N01…N100为拓扑节点展示名，输入token不变。'
    s=re.sub(r'<div class="foot">.*?</div>',f'<div class="foot">{foot}</div>',s,flags=re.S)
    s=s.replace('<script>','<div id="loading" style="padding:12px">正在载入完整长轨迹attention…</div><script>\n(async()=>{',1)
    s=s.replace("const cats=",LOADER.replace('__PACKED__',packed)+"\nconst hash=new URLSearchParams(location.hash.slice(1));if(hash.has('p')){const p=+hash.get('p');if(Number.isInteger(p)&&p>=0&&p<D.labels.length)D.default_position=p;}\nconst cats=",1)
    s=s.replace('</script>','})().catch(e=>{document.getElementById("loading").textContent="载入失败："+e.message;console.error(e)});</script>')
    return s.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')))

def main():
    overview=[]
    for name in ['yes','no']:
        data=json.loads((ROOT/f'{name}.json').read_text(encoding='utf8'));blob=(ROOT/f'{name}.attention.f32.gz').read_bytes();raw=gzip.decompress(blob)
        assert hashlib.sha256(raw).hexdigest()==data['attention_sha256']
        att=np.frombuffer(raw,dtype='<f4').reshape(data['attention_shape']);assert np.allclose(att.sum(-1),1,atol=3e-6)
        suffix='' if name=='yes' else '_no';data['animation_url']='search_animation'+suffix+'.html';data['explorer_url']='token_explorer'+suffix+'.html'
        adjacency={n:[] for n in data['graph']['labels']}
        for u,v in data['graph']['edges']:adjacency[data['graph']['labels'][u]].append(data['graph']['labels'][v])
        filtered=[r['record'] for r in data['records'] if any(v in r['visited'] for v in adjacency[r['stack'][-1]])]
        run=longest=0;pushed={};cross_branch=[]
        for r in data['records']:
            current=r['stack'][-1]
            if (set(adjacency[current])&set(r['visited']))-set(pushed.get(current,[])):cross_branch.append(r['record'])
            if r['action']=='PUSH':pushed.setdefault(current,[]).append(r['argument'])
            run=run+1 if r['action']=='POP' else 0;longest=max(longest,run)
        data['summary'].update(max_consecutive_pop=longest,visited_filter_records=filtered,cross_branch_filter_records=cross_branch)
        packed=base64.b64encode(blob).decode('ascii')
        (ROOT/data['explorer_url']).write_text(explorer(data,packed,name),encoding='utf8')
        story=(ROOT/'story_template.html').read_text(encoding='utf8').replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))).replace('__LOADER__',LOADER.replace('__PACKED__',packed))
        (ROOT/data['animation_url']).write_text(story,encoding='utf8');overview.append(data['summary'])
    (ROOT/'selection.json').write_text(json.dumps(dict(criteria='Both fully legal native rollouts in the existing 12-query middle/long panel; YES for deep search, NO for exhaustive backtracking. No extra generations or outcome filtering within the two successes.',cases=overview),ensure_ascii=False,indent=2),encoding='utf8')
    # Keep the small case available, make existing entry URLs open the long YES.
    for original in ['token_explorer','search_attention','search_animation']:
        archive=OLD/(original+'_small.html')
        if not archive.exists():shutil.copyfile(OLD/(original+'.html'),archive)
        if original!='search_animation':archive.write_text(archive.read_text(encoding='utf8').replace('href="search_animation.html"','href="search_animation_small.html"'),encoding='utf8')
    for target,source in [('token_explorer','token_explorer'),('search_attention','token_explorer'),('search_animation','search_animation')]:
        s=(ROOT/(source+'.html')).read_text(encoding='utf8')
        for url in ['token_explorer.html','token_explorer_no.html','search_animation.html','search_animation_no.html']:
            s=s.replace('"'+url+'"','"../attention_long_20261031/'+url+'"')
        (OLD/(target+'.html')).write_text(s,encoding='utf8')
    print(json.dumps(overview))

if __name__=='__main__':main()
