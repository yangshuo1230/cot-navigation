import json,gzip,base64,html
from pathlib import Path
import numpy as np
from summarize import summary
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/(n+'.json')).read_text(encoding='utf8'))
def main():
 S=read('summary');E=read('extra_routes');W=read('graph_writer');I=read('extra_interventions');B=read('bar_pool');names=[m['name'] for m in S['models']];labels=['原6层','纠错6/s0','纠错6/s1','纠错9/s0','纠错9/s1'];pack={}
 for model in S['models']:
  name=model['name'];pack[name]={}
  for title,f in S['results'][name]['features'].items():pack[name][title]=dict(layer=f['layer'],mean=f['test']['mean'],low=f['test']['low'],high=f['test']['high'],route=f['route'],n_queries=f['test']['n_queries'])
  for title in ['init_roots','visited_direct','revisit_tail','fresh_raw_dest']:
   f=E[name][title];pack[name][title]=dict(layer=f['layer'],mean=f['test']['mean'],low=f['test']['low'],high=f['test']['high'],route=f['route'],n_queries=f['test']['n_queries'])
  offset=E[name]['offset_scan'];route='adjacency_offset_'+str(offset['offset'])
  with gzip.open(ROOT/(name+'.supplement.jsonl.gz'),'rt') as f:rows=[json.loads(x) for x in f]
  rs=[r for r in rows if r['split']=='test' and r['phase']=='candidate_init_fresh'];stats=summary(rs,lambda r:r['routes'][route]);pack[name]['graph_best']=dict(layer=offset['layer'],route=route,mean=stats['mean'],low=stats['low'],high=stats['high'],n_queries=stats['n_queries'])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import PowerNorm
 plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
 features=[('stack初始化','初始化\nprompt |'),('init_roots','初始化\n历史stack-root'),('visited续写/root','visited\n历史root'),('visited_direct','visited\n旧同名visited'),('graph_best','首次候选\n各自图偏移'),('回访候选/旧PUSH参数','回访候选\n旧PUSH/参数'),('revisit_tail','回访候选\n旧候选尾部'),('YES答案','答案YES\n终止动作')]
 values=np.array([[pack[n][k]['mean'][pack[n][k]['layer']-1] for k,_ in features] for n in names]);fig,ax=plt.subplots(figsize=(14,5.3),layout='constrained');ax.imshow(values,cmap='YlGnBu',norm=PowerNorm(.5,0,1));ax.set_xticks(range(len(features)),[v for _,v in features],fontsize=10);ax.set_yticks(range(len(names)),labels,fontsize=11)
 for i,n in enumerate(names):
  for j,(k,_) in enumerate(features):ax.text(j,i,f"L{pack[n][k]['layer']}\n{100*values[i,j]:.1f}%",ha='center',va='center',fontsize=11,color='white' if values[i,j]>.5 else '#28485d')
 ax.set_title('同一组正确前缀：val选展示层，test按query平均 / 原始attention，统一0–1色标',fontsize=13);fig.savefig(ROOT/'route_comparison.png',dpi=160);plt.close(fig)
 phases=['stack_init_PUSH','visited_continue','candidate_init_revisit','stack_end','candidate_end'];phase_labels=['stack初始化','visited续写','回访候选','stack结束','候选结束'];fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
 for ax,key,title in zip(axes,['no_attention_agrees','no_ffn_agrees'],['移除最后一层attention','移除最后一层FFN']):
  a=np.array([[1-S['results'][n]['phases'][p]['last_block'][key]['mean'] for p in phases] for n in names]);ax.imshow(a,cmap='OrRd',vmin=0,vmax=1);ax.set_xticks(range(5),phase_labels,rotation=20,ha='right');ax.set_yticks(range(5),labels);ax.set_title(title+'：输出改变比例')
  for i in range(5):
   for j in range(5):ax.text(j,i,f'{a[i,j]*100:.1f}%',ha='center',va='center',color='white' if a[i,j]>.5 else '#543b2d')
 fig.savefig(ROOT/'last_block.png',dpi=160);plt.close(fig)
 # A compact standalone HTML: no external PNG dependency or large raw matrix.
 payload=dict(names=names,labels=labels,models=S['models'],features=features,pack=pack)
 def img(file):return 'data:image/png;base64,'+base64.b64encode((ROOT/file).read_bytes()).decode('ascii')
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>6层与9层纠错模型对照</title><style>*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#243e55;font:14px/1.65 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1400px;margin:auto;padding:24px}section{background:white;border:1px solid #dce5ee;border-radius:12px;padding:20px;margin-bottom:16px}h1{font-size:25px;margin:0}h2{font-size:18px;margin:0 0 10px}p{color:#61798f}a{color:#187f91}select{font:inherit;padding:7px;border:1px solid #dce5ee;border-radius:6px;max-width:100%}img{width:100%;height:auto;display:block}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;min-width:800px;font-size:12px}th,td{padding:9px;border:1px solid #e3eaf0;text-align:center}th{background:#f4f7fa}.selected{outline:2px solid #dc9a36;outline-offset:-2px}.na{color:#a9b5bf;background:#edf1f5}.links{display:flex;gap:15px;flex-wrap:wrap}.notes{background:#f2f7f9;padding:12px;border-radius:7px}li{margin:7px 0}</style><main>
<section><h1>6层与9层纠错模型：attention共性与不同实现</h1><p>30%错误前缀训练，各2个seed；原185k的6层模型作基线。相同18条val+18条test、相同token和参考轨迹。这里的比例是读取分配，不是因果贡献或自由搜索成功率。</p><div class="notes">主要发现：图载体偏移为原模型+3、纠错6层为0、纠错9层为+1；两个6层纠错seed偏向读旧候选尾部，9层内部则有明显seed差异。正确前缀上的pattern不等于纠错时的pattern。</div><div class="links"><a href="REPORT.md">完整报告与限制</a><a href="models.json">checkpoint身份</a><a href="validation.json">验证</a></div></section>
<section><h2>按字段查看全部层</h2><select id="feature"></select><p>橙框=在val选择的展示层；数值=独立test的query均值。颜色统一0–1平方根映射。灰色表示该模型没有该层。悬停显示置信区间及分母。</p><div class="scroll"><table><thead><tr><th>模型 / 读取位置</th><th>L1</th><th>L2</th><th>L3</th><th>L4</th><th>L5</th><th>L6</th><th>L7</th><th>L8</th><th>L9</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section><h2>关键路线概览</h2><img src="__ROUTES__" alt="五个模型的关键读取路线"><p>“各自图偏移”使用各模型val选出的实际载体位置，不是相同物理key；其他列读取位置相同。均值和选层不把两个seed合并。</p></section>
<section><h2>最后一层的功能不固定</h2><img src="__LAST__" alt="移除最后attention或FFN后输出改变的比例"><p>移除attention后重新计算FFN。图中比较与原最终token的不同，不能直接解释为DFS语义错误率。新模型并非简单复制原模型L5/L6的分工。</p></section>
<section><h2>局部因果检查</h2><ul><li>6层纠错：历史stack-root初始化、旧visited节点续写、旧候选尾部的定向value删除，所测两个seed各6个条件都失去正确输出；原模型对这些路径的依赖明显不同。</li><li>9层初始化：prompt | 与历史stack-bar联合删除后，两seed均0/6正确。seed0单删prompt |仍6/6正确，说明信息在多个同类载体中分散。</li><li>9层visited：seed0偏旧同名visited节点，seed1保留历史root路线；对应局部干预也不同。</li><li>图+1路线：seed1的局部删除影响明显；seed0在这组单位置删除中仍5/6候选字段正确，不把其高attention当成已经证明必要。</li></ul></section>
<section><h2>范围</h2><p>4个纠错checkpoint的best步数不同（180k/160k/200k/160k），原基线配方也不同，因此不能将所有差异单独归因于层数或纠错训练。这里只检验了两个seed、已使用过的参考query panel。人工错误visited前缀不等于自然错误的恢复benchmark。</p></section></main>
<script>const D=__DATA__;function render(){let key=document.getElementById('feature').value,body=document.getElementById('rows');body.replaceChildren();D.names.forEach((name,i)=>{let f=D.pack[name][key],tr=document.createElement('tr'),label=document.createElement('td');label.textContent=D.labels[i]+' / '+f.route;tr.appendChild(label);for(let l=0;l<9;l++){let td=document.createElement('td');if(l>=f.mean.length){td.textContent='—';td.className='na'}else{let v=f.mean[l],q=Math.sqrt(v);td.style.background=`rgb(${246-222*q},${248-113*q},${250-96*q})`;td.style.color=v>.5?'white':'#26475b';td.textContent=(v*100).toFixed(1)+'%';td.title='95% query-bootstrap: '+(f.low[l]*100).toFixed(1)+'–'+(f.high[l]*100).toFixed(1)+'%; n='+f.n_queries;if(l+1===f.layer)td.className='selected'}tr.appendChild(td)}body.appendChild(tr)})}for(let [key,label] of D.features){let o=document.createElement('option');o.value=key;o.textContent=label.replaceAll(String.fromCharCode(10),' / ');document.getElementById('feature').appendChild(o)}document.getElementById('feature').onchange=render;render();window.comparisonTest={data:D,render};</script></html>'''
 page=page.replace('__ROUTES__',img('route_comparison.png')).replace('__LAST__',img('last_block.png')).replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')))
 (ROOT/'comparison.html').write_text(page,encoding='utf8')
 print(json.dumps(dict(models=len(names),comparison_features=len(features),html='comparison.html')))
if __name__=='__main__':main()
