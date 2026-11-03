"""Build a self-contained search/attention explainer from measured matrices."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
META = json.loads((HERE/'metadata.json').read_text())
# Display aliases only. Preserve all actual model token IDs and input edges.
order=[META['source'],*META['records'][0]['candidates']]
order=list(dict.fromkeys(order+[x for x in META['node_mapping']]))
RENAME={old:chr(65+i) for i,old in enumerate(order)}
ORIGINAL_NODE_NAMES=list(META['node_mapping'])
META['node_mapping']={RENAME[k]:v for k,v in META['node_mapping'].items()}
for rec in META['records']:
    for field in ['stack','visited','candidates']:rec[field]=[RENAME[x] for x in rec[field]]
    words=rec['action'].split();rec['action']=' '.join([words[0]]+[RENAME[x] for x in words[1:]])
SOURCE=RENAME[META['source']];TARGET=RENAME[META['target']]
PROMPT=META['prompt_len']
CATEGORIES = ['图结构', '查询起点', '查询目标', '本步 stack', '本步 visited', '本步候选', '本步动作', '历史状态', '历史动作', '格式符号', '自身']
PALETTE = ['#607d95','#397bf2','#c264d9','#167c9b','#859642','#dc9137','#a368cb','#5a9695','#887da8','#a3acb9','#e26a51']
PHASES = {'stack':'生成栈字段','visited':'生成 visited 字段','candidates':'生成候选字段','action':'决定搜索动作','argument':'选择 PUSH 的节点','syntax':'生成字段分隔符','answer':'生成最终答案'}

def make_trace(file, reference):
    src=np.load(HERE/file)
    att=src['attention']
    tokens=src['tokens'].tolist()
    names=[next((k for k,v in META['node_mapping'].items() if v==t), {2:'EDGE',3:'QUERY',4:'|',5:';',6:'YES',7:'NO',8:'PUSH',9:'POP',10:'FOUND',11:'EXHAUST'}.get(t,str(t))) for t in tokens]
    info=[dict(role='syntax',record=-1,description='格式符号') for _ in tokens]
    for e in range(META['n_edges']):
        for j in range(e*3,e*3+3):
            info[j]=dict(role='graph',record=-1,description=f'图边 {names[e*3+1]}→{names[e*3+2]} / '+['EDGE','起点','终点'][j%3])
    info[PROMPT-3]=dict(role='source',record=-1,description='查询起点 '+SOURCE)
    info[PROMPT-2]=dict(role='target',record=-1,description='查询目标 '+TARGET)
    records=META['records']
    for r,rec in enumerate(records):
        start,end=rec['start'],rec['end']
        bars=[p for p in range(start,end) if names[p]=='|']
        assert len(bars)==3
        a,b,c=bars
        for lo,hi,role in [(start,a,'stack'),(a+1,b,'visited'),(b+1,c,'candidates'),(c+1,end-1,'action')]:
            for p in range(lo,hi):
                rr='argument' if role=='action' and names[p] in META['node_mapping'] else role
                info[p]=dict(role=rr,record=r,description=f'R{r+1} '+{'stack':'stack','visited':'visited','candidates':'候选','action':'动作','argument':'动作参数'}[rr])
        for p in bars+[end-1]: info[p]=dict(role='syntax',record=r,description=f'R{r+1} 分隔符')
    info[-1]=dict(role='answer',record=len(records)-1,description='最终答案')
    def group(key,row,record):
        if key==row:return 10
        role=info[key]['role']
        if role=='graph':return 0
        if role=='source':return 1
        if role=='target':return 2
        if role=='syntax':return 9
        current=info[key]['record']==record
        if role in ['stack','visited','candidates']:
            return {'stack':3,'visited':4,'candidates':5}[role] if current else 7
        if role in ['action','argument']:return 6 if current else 8
        return 9
    frames=[]
    for target in range(PROMPT,len(tokens)):
        row=target-1
        ri=info[target]['record']
        phase=info[target]['role']
        rec=records[ri]
        masses=[];tops=[]
        for layer in range(6):
            weights=att[layer,row,:row+1]
            mass=[float(sum(float(v) for key,v in enumerate(weights) if group(key,row,ri)==g)) for g in range(len(CATEGORIES))]
            assert abs(sum(mass)-1)<2e-6
            ids=np.argsort(-weights,kind='stable')[:8]
            tops.append([dict(position=int(k),token=names[k],description=info[k]['description'],value=float(weights[k]),group=group(int(k),row,ri)) for k in ids])
            masses.append(mass)
        frames.append(dict(target=target,row=row,record=ri,phase=phase,phase_name=PHASES[phase],next=names[target],consumed=names[row],masses=masses,top=tops,state=rec,
                           edge_weights=att[:,row,:META['n_edges']*3].reshape(6,META['n_edges'],3).tolist()))
    # Keyframes are chosen by syntax, never by attention magnitude.
    keys=[]
    for i,f in enumerate(frames):
        if f['phase']=='answer' or f['phase'] in ['action','argument'] or (f['phase'] not in ['syntax'] and (i==0 or (frames[i-1]['record'],frames[i-1]['phase'])!=(f['record'],f['phase']))):
            keys.append(i)
        elif f['phase']=='syntax' and f['next']=='|' and i>0 and frames[i-1]['next']=='|':
            keys.append(i)
    assert all(0<=f['row']<f['target']<len(tokens) for f in frames)
    return dict(labels=names,info=info,frames=frames,keyframes=keys,reference=reference,records=records)

def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('msyhbd.ttc' if bold else 'msyh.ttc'),size)

def render_frame(trace,idx,layer=3):
    f=trace['frames'][idx];state=f['state'];W,H=1440,1290
    im=Image.new('RGB',(W,H),'#eef2f7');d=ImageDraw.Draw(im)
    def text(x,y,t,s=20,c='#20334d',bold=False):d.text((x,y),t,font=font(s,bold),fill=c)
    def box(bounds,fill='#ffffff',outline=None,r=16):d.rounded_rectangle(bounds,r,fill,outline,width=2)
    text(36,22,'DFS / Attention in motion',30,bold=True)
    text(36,68,'成功的自由生成 · A→C · 先探索 B 死端再回退 · 固定 Layer 4 / Head 1',19)
    box((36,106,1404,153),'#e1f3eb',r=8)
    text(52,116,'答案 YES 正确；全部 4 条 DFS-state 严格合法。输入完整 N16 / 24 边；下图放大 3 个访问节点。',18,'#27634c')
    box((36,174,472,653));box((492,174,1404,653));box((36,673,1404,860))
    text(58,191,f'R{f["record"]+1} / 4 · {state["action"]}',22,bold=True)
    text(58,228,'模型生成状态（严格重放验证，动作前）',16,'#61738c')
    pos={'A':(145,315),'B':(340,290),'C':(340,420)}
    def arrow(a,b,color,width):
        x,y=pos[a];xx,yy=pos[b];dx,dy=xx-x,yy-y;length=math.hypot(dx,dy);ux,uy=dx/length,dy/length
        start=(x+26*ux,y+26*uy);end=(xx-30*ux,yy-30*uy)
        d.line([start,end],fill=color,width=width)
        d.polygon([end,(end[0]-10*ux+5*uy,end[1]-10*uy-5*ux),(end[0]-10*ux-5*uy,end[1]-10*uy+5*ux)],fill=color)
    for a,b in [('A','B'),('A','C')]:
        on=any(x==a and y==b for x,y in zip(state['stack'],state['stack'][1:]))
        arrow(a,b,'#168fa1' if on else '#a7b4c6',6 if on else 3)
    for n,(x,y) in pos.items():
        fill='#e0f1f4' if n in state['stack'] else '#edf0dc' if n in state['visited'] else '#fff1da' if n in state['candidates'] else '#f0f3f8'
        d.ellipse((x-25,y-25,x+25,y+25),fill=fill,outline='#13899b' if n==state['stack'][-1] else '#b0becd',width=4 if n==state['stack'][-1] else 2)
        text(x-10,y-16,n,22,bold=True)
    for y,label,values,col in [(482,'STACK',state['stack'],'#167c9b'),(526,'VISITED',state['visited'],'#859642'),(570,'CAND',state['candidates'],'#dc9137')]:
        text(58,y,label,15,col,True);text(168,y-3,'  '.join(values) or '∅',21,col)
    text(58,619,'状态从模型实际生成序列解析；非隐藏状态读出。',13,'#738398')
    text(516,191,f'{f["phase_name"]}  →  实际 token: {f["next"]}',24,bold=True)
    text(516,235,f'读取位置 {f["row"]}:{f["consumed"]} 的 attention；它用于预测位置 {f["target"]}。',17,'#61738c')
    top=f['top'][layer]
    text(516,275,f'Top 8 key · 覆盖 {sum(k["value"] for k in top):.1%} 注意力质量',18,bold=True)
    for j,k in enumerate(top):
        yy=313+j*37;col=PALETTE[k['group']]
        text(518,yy,f'{k["position"]:02d} : {k["token"]}',16,col,True)
        text(650,yy,k['description'],14,'#65778d')
        d.rounded_rectangle((872,yy+3,872+max(1,int(390*k['value'])),yy+21),4,fill=col)
        text(1284,yy,f'{k["value"]:.1%}',16,col)
    dominant=sorted(enumerate(f['masses'][layer]),key=lambda x:-x[1])[:2]
    text(516,617,'当前主要去向：'+'；'.join(f'{CATEGORIES[g]} {v:.1%}' for g,v in dominant),16,'#506c83')
    text(58,689,'同一预测位置 · 六层注意力去向',19,bold=True)
    cw=108;ox=173;oy=754
    for g,name in enumerate(CATEGORIES):text(ox+g*cw,724,name,12,'#65778d')
    for l,mass in enumerate(f['masses']):
        text(64,oy+l*15,f'L{l+1}'+(' *' if l==layer else ''),11,'#168fa1' if l==layer else '#65778d',l==layer)
        for g,v in enumerate(mass):
            c=tuple(int(246*(1-math.sqrt(v))+z*math.sqrt(v)) for z in (24,135,154))
            d.rectangle((ox+g*cw,oy+l*15,ox+(g+1)*cw-6,oy+l*15+12),fill=c)
    text(36,871,f'Checkpoint 185k · CPU FP32 · 单头真实 softmax 权重 · 非因果贡献度    |    输出 token 帧 {idx+1}/{len(trace["frames"])}',13,'#65778d')
    box((36,904,1404,1268))
    total=sum(sum(x) for x in f['edge_weights'][layer])
    text(58,917,f'边列表读取 · Layer {layer+1} · 总质量 {total:.2%}',20,bold=True)
    text(58,949,'每行一条原始输入边：EDGE / 起点 / 终点的 attention。未按图部分重新归一化。',14,'#65778d')
    for edge_i,weights in enumerate(f['edge_weights'][layer]):
        half=edge_i//12;line=edge_i%12;x=58+half*677;y=978+line*23
        u,v=META['edges'][edge_i];name=f'{edge_i+1:02d} {RENAME[ORIGINAL_NODE_NAMES[u]]}→{RENAME[ORIGINAL_NODE_NAMES[v]]}'
        text(x,y,name,13)
        for k,prob in enumerate(weights):
            xx=x+110+k*170;q=math.sqrt(prob)
            col=tuple(int(246*(1-q)+z*q) for z in (24,135,154))
            d.rectangle((xx,y,xx+155,y+20),fill=col)
            text(xx+5,y,['E ','起 ','终 '][k]+f'{prob:.2%}',11,'#ffffff' if prob>.4 else '#3b586f')
    return im

def render_edges(data,layer):
    im=Image.new('RGB',(1060,930),'#eef2f7');draw=ImageDraw.Draw(im)
    draw.text((30,22),f'边列表内部编码 / Layer {layer+1}',font=font(27,True),fill='#20334d')
    draw.text((30,67),'行：编码中的边；列：读取的边。3×3 token 块聚合，逐行合计100%。',font=font(17),fill='#61738c')
    draw.text((30,95),'实际成功案例的输入前缀 · 灰色=未来遮罩 · 统一0–1概率色标（平方根映射）',font=font(15),fill='#61738c')
    p=190;s=28;a=data['graph']['blocks'][layer]
    for i in range(24):
        u,v=META['edges'][i];name=f'{i+1:02d} {RENAME[ORIGINAL_NODE_NAMES[u]]}→{RENAME[ORIGINAL_NODE_NAMES[v]]}'
        draw.text((65,p+i*s+5),name,font=font(13),fill='#526d87')
        draw.text((p+i*s+1,150),f'{i+1:02d}',font=font(12),fill='#526d87')
        for j in range(24):
            q=math.sqrt(a[i][j]);col=tuple(int(246*(1-q)+z*q) for z in (24,135,154)) if j<=i else '#e1e5ea'
            draw.rectangle((p+j*s,p+i*s,p+(j+1)*s-2,p+(i+1)*s-2),fill=col)
    for k in range(200):
        q=math.sqrt(k/199);col=tuple(int(246*(1-q)+z*q) for z in (24,135,154))
        draw.line((920,750-k*2,942,750-k*2),fill=col)
    for y,label in [(350,'100%'),(550,'50%'),(750,'0%')]:draw.text((953,y-6),label,font=font(14),fill='#526d87')
    draw.text((65,890),'交互版可点击每个边块，展开 EDGE / 起点 / 终点的原始3×3权重。',font=font(16),fill='#526d87')
    return im

def main():
    edge_tokens=META['n_edges']*3
    raw=np.load(HERE/'attention.npz')['attention'][:,:edge_tokens,:edge_tokens]
    edge_blocks=raw.reshape(6,META['n_edges'],3,META['n_edges'],3).sum(axis=4).mean(axis=2)
    assert np.allclose(edge_blocks.sum(-1),1,atol=2e-6)
    data=dict(reference=make_trace('attention.npz',True),categories=CATEGORIES,colors=PALETTE,
              prompt_len=PROMPT,source=SOURCE,target=TARGET,
              graph=dict(labels=[RENAME[x] for x in ORIGINAL_NODE_NAMES],edges=META['edges'],active=['A','B','C'],attention=raw.tolist(),blocks=edge_blocks.tolist()),
              display_aliases=RENAME,original_metadata='metadata.json')
    (HERE/'animation_data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    template=(HERE/'animation_template.html').read_text(encoding='utf-8')
    (HERE/'search_animation.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
    ref=data['reference'];frames=[render_frame(ref,i) for i in ref['keyframes']]
    frames[0].save(HERE/'search_attention.gif',save_all=True,append_images=frames[1:],duration=1300,loop=0,optimize=False)
    render_frame(ref,ref['keyframes'][0]).save(HERE/'animation_preview.png')
    edge_frames=[render_edges(data,l) for l in range(6)]
    edge_frames[0].save(HERE/'edge_encoding.gif',save_all=True,append_images=edge_frames[1:],duration=1800,loop=0,optimize=False)
    edge_frames[3].save(HERE/'edge_encoding_layer4.png')
    print(json.dumps(dict(actual_free_frames=len(ref['frames']),keyframes=len(ref['keyframes']),gif_frames=len(frames)),ensure_ascii=False))

if __name__=='__main__':main()
