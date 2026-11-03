"""Build a self-contained search/attention explainer from measured matrices."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
META = json.loads((HERE/'metadata.json').read_text())
CATEGORIES = ['图结构', '查询起点', '查询目标', '本步 stack', '本步 visited', '本步候选', '本步动作', '历史状态', '历史动作', '格式符号', '自身']
PALETTE = ['#607d95','#397bf2','#c264d9','#167c9b','#859642','#dc9137','#a368cb','#5a9695','#887da8','#a3acb9','#e26a51']
PHASES = {'stack':'生成栈字段','visited':'生成 visited 字段','candidates':'生成候选字段','action':'决定搜索动作','argument':'选择 PUSH 的节点','syntax':'生成字段分隔符','answer':'生成最终答案'}

def make_trace(file, reference):
    src=np.load(HERE/file)
    att=src['attention']
    tokens=src['tokens'].tolist()
    names=[next((k for k,v in META['node_mapping'].items() if v==t), {2:'EDGE',3:'QUERY',4:'|',5:';',6:'YES',7:'NO',8:'PUSH',9:'POP',10:'FOUND',11:'EXHAUST'}.get(t,str(t))) for t in tokens]
    info=[dict(role='syntax',record=-1,description='格式符号') for _ in tokens]
    for e in range(3):
        for j in range(e*3,e*3+3):
            info[j]=dict(role='graph',record=-1,description=f'图边 {names[e*3+1]}→{names[e*3+2]} / '+['EDGE','起点','终点'][j%3])
    info[10]=dict(role='source',record=-1,description='查询起点 A')
    info[11]=dict(role='target',record=-1,description='查询目标 D')
    records=META['records'] if reference else [dict(record=1,start=13,end=len(tokens)-1,stack=['A'],visited=['A'],candidates=[],action='EXHAUST EXHAUST（非法重复）')]
    for r,rec in enumerate(records):
        start,end=rec['start'],rec['end']
        bars=[p for p in range(start,end) if names[p]=='|']
        assert len(bars)==3
        a,b,c=bars
        for lo,hi,role in [(start,a,'stack'),(a+1,b,'visited'),(b+1,c,'candidates'),(c+1,end-1,'action')]:
            for p in range(lo,hi):
                rr='argument' if role=='action' and names[p] in 'ABCD' else role
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
    for target in range(13,len(tokens)):
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
        frames.append(dict(target=target,row=row,record=ri,phase=phase,phase_name=PHASES[phase],next=names[target],consumed=names[row],masses=masses,top=tops,state=rec))
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
    f=trace['frames'][idx];state=f['state'];W,H=1440,900
    im=Image.new('RGB',(W,H),'#eef2f7');d=ImageDraw.Draw(im)
    def text(x,y,t,s=20,c='#20334d',bold=False):d.text((x,y),t,font=font(s,bold),fill=c)
    def box(bounds,fill='#ffffff',outline=None,r=16):d.rounded_rectangle(bounds,r,fill,outline,width=2)
    text(36,22,'DFS / Attention in motion',30,bold=True)
    text(36,68,'参考 CoT 条件注意力 · A→D · 先探索 B 分支再回退 · 固定 Layer 4 / Head 1',19)
    box((36,106,1404,153),'#fff0d4',r=8)
    text(52,116,'这是正确参考前缀，不是模型自发完成的搜索。原模型在该四节点输入上自由生成错误 NO。',18,'#805b21')
    box((36,174,472,653));box((492,174,1404,653));box((36,673,1404,860))
    text(58,191,f'R{f["record"]+1} / 6 · {state["action"]}',22,bold=True)
    text(58,228,'参考 DFS 状态（动作执行前）',16,'#61738c')
    pos={'A':(145,315),'B':(275,315),'C':(390,315),'D':(145,420)}
    def arrow(a,b,color,width):
        x,y=pos[a];xx,yy=pos[b];dx,dy=xx-x,yy-y;length=math.hypot(dx,dy);ux,uy=dx/length,dy/length
        start=(x+26*ux,y+26*uy);end=(xx-30*ux,yy-30*uy)
        d.line([start,end],fill=color,width=width)
        d.polygon([end,(end[0]-10*ux+5*uy,end[1]-10*uy-5*ux),(end[0]-10*ux-5*uy,end[1]-10*uy+5*ux)],fill=color)
    for a,b in [('A','B'),('B','C'),('A','D')]:
        on=any(x==a and y==b for x,y in zip(state['stack'],state['stack'][1:]))
        arrow(a,b,'#168fa1' if on else '#a7b4c6',6 if on else 3)
    for n,(x,y) in pos.items():
        fill='#e0f1f4' if n in state['stack'] else '#edf0dc' if n in state['visited'] else '#fff1da' if n in state['candidates'] else '#f0f3f8'
        d.ellipse((x-25,y-25,x+25,y+25),fill=fill,outline='#13899b' if n==state['stack'][-1] else '#b0becd',width=4 if n==state['stack'][-1] else 2)
        text(x-10,y-16,n,22,bold=True)
    for y,label,values,col in [(482,'STACK',state['stack'],'#167c9b'),(526,'VISITED',state['visited'],'#859642'),(570,'CAND',state['candidates'],'#dc9137')]:
        text(58,y,label,15,col,True);text(168,y-3,'  '.join(values) or '∅',21,col)
    text(58,619,'图上状态是参考搜索状态，不是模型内部状态。',13,'#738398')
    text(516,191,f'{f["phase_name"]}  →  参考 token: {f["next"]}',24,bold=True)
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
    return im

def main():
    data=dict(reference=make_trace('attention.npz',True),free=make_trace('free_attention.npz',False),categories=CATEGORIES,colors=PALETTE)
    (HERE/'animation_data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    template=(HERE/'animation_template.html').read_text(encoding='utf-8')
    (HERE/'search_attention.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
    ref=data['reference'];frames=[render_frame(ref,i) for i in ref['keyframes']]
    frames[0].save(HERE/'search_attention.gif',save_all=True,append_images=frames[1:],duration=1300,loop=0,optimize=False)
    render_frame(ref,ref['keyframes'][0]).save(HERE/'animation_preview.png')
    # Export a broadly playable, compact movie as well.
    import subprocess
    proc=subprocess.run(['ffmpeg','-y','-i',str(HERE/'search_attention.gif'),'-vf','fps=20,format=yuv420p','-c:v','libx264','-crf','22','-movflags','+faststart',str(HERE/'search_attention.mp4')],capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stderr[-2000:])
    print(json.dumps(dict(reference_frames=len(ref['frames']),keyframes=len(ref['keyframes']),free_frames=len(data['free']['frames']),gif_frames=len(frames)),ensure_ascii=False))

if __name__=='__main__':main()
