"""Concrete nodes and zero-based positions from actual audited queries."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GOLD='#D99A00';GRAY='#777777'

def canvas(title,subtitle):
    fig,ax=plt.subplots(figsize=(15,9));fig.subplots_adjust(left=.025,right=.985,top=.89,bottom=.08)
    ax.set(xlim=(0,16),ylim=(0,10));ax.axis('off')
    fig.suptitle(title,fontsize=20,y=.98)
    fig.text(.5,.925,subtitle,ha='center',fontsize=11,color='#444444')
    return fig,ax

def cell(ax,x,y,token,pos,color=GRAY,w=1.1):
    ax.add_patch(FancyBboxPatch((x,y),w,.67,boxstyle='round,pad=.04,rounding_size=.08',
                              lw=1.8,edgecolor=color,facecolor='#FFF7DF' if color==GOLD else '#F2F7FA',zorder=3))
    ax.text(x+w/2,y+.39,token,ha='center',va='center',fontsize=15,color='#222222',zorder=4)
    ax.text(x+w/2,y+.11,f'位置 {pos}',ha='center',va='center',fontsize=8,color=GRAY,zorder=4)
    return (x+w/2,y+.69)

def row(ax,y,title,items,x=.7):
    ax.text(.5,y+1.04,title,fontsize=12,fontweight='bold',zorder=5,
            bbox=dict(facecolor='white',edgecolor='none',pad=2,alpha=.95))
    result={}
    for token,pos,color in items:
        result[pos]=cell(ax,x,y,token,pos,color);x+=1.28
    return result

def arc(ax,src,dst,label,color=BLUE,rad=.2,textxy=None):
    ax.add_patch(FancyArrowPatch(src,dst,arrowstyle='-|>',connectionstyle=f'arc3,rad={rad}',
                               mutation_scale=15,lw=2,color=color,shrinkA=4,shrinkB=4,zorder=2))
    if textxy:ax.text(*textxy,label,fontsize=12,color=color,bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=2))

def notes(ax,x,y,text):ax.text(x,y,text,fontsize=12,va='top',linespacing=1.7,color='#333333')
def save(fig,name,pdf):
    fig.text(.03,.025,'实线箭头：输入位置读取哪个历史位置。下一输出已核对模型 argmax；“中介携带什么内容”仍属于机制假设。',fontsize=10,color=GRAY)
    fig.savefig(HERE/(name+'.png'),dpi=180,facecolor='white');pdf.savefig(fig);plt.close(fig)

with PdfPages(HERE/'concrete_information_routes.pdf') as pdf:
    fig,ax=canvas('例 1：In-Context 如何在 POP 后重新写出 V5、V9？',
                  '真实 query 50512：V2 → V6。这里只画第 3、4、5 条记录中有关的 token；位置从 0 开始。')
    a=row(ax,7.6,'第 3 条：栈是 V2 V5 V9；随后 PUSH V15',[
        ('V2',100,GRAY),('V5',101,ORANGE),('V9',102,GRAY),('|',103,GRAY),('V2',104,BLUE)])
    b=row(ax,4.7,'第 4 条：栈是 V2 V5 V9 V15；随后 POP V15',[
        ('V2',114,GRAY),('V5',115,GRAY),('V9',116,ORANGE),('V15',117,GRAY),('|',118,GRAY),('V2',119,BLUE)])
    c=row(ax,1.1,'第 5 条：要重新写出栈 V2 V5 V9',[
        ('V2',127,GOLD),('V5',128,GOLD),('V9',129,GREEN),('|',130,GRAY)])
    arc(ax,a[104],a[101],'L1：0.985',BLUE,.45,(3.0,9.15))
    arc(ax,b[119],b[116],'L1：0.997',BLUE,.4,(4.3,6.2))
    arc(ax,c[127],(a[104][0],7.58),'L5：0.993',ORANGE,-.18,(.2,3.8))
    arc(ax,c[128],(b[119][0],4.68),'L5：0.993',GREEN,-.11,(4.6,3.0))
    ax.text(.7,.5,'① 输入 V2@127 → 模型输出 V5@128',fontsize=11,color=ORANGE)
    ax.text(.7,.06,'② 输入 V5@128 → 模型输出 V9@129',fontsize=11,color=GREEN)
    notes(ax,9.0,8.5,'两个蓝色位置字面上都叫 V2，\n但它们位于不同记录：\n\nV2@104 在 L1 强读 V5@101；\nV2@119 在 L1 强读 V9@116。\n\n稍后生成栈时，L5 分别读取\n这两个不同历史位置。\n\n符合“用历史隐藏状态作中介”\n的解释；不等于已证明其中\n只存了 V5 或 V9。')
    save(fig,'concrete_01_context_stack',pdf)

    fig,ax=canvas('例 2：Fixed 在相似的 POP 后，更直接读取历史 V74',
                  '真实 simple Fixed query 151：V65 → V87。同样是在重新写栈，读取的历史位置却不同。')
    a=row(ax,7.6,'第 3 条：栈 V65 V74 V82；随后 PUSH V96',[
        ('V65',30,GRAY),('V74',31,BLUE),('V82',32,GRAY),('|',33,GRAY)])
    b=row(ax,4.7,'第 4 条：栈 V65 V74 V82 V96；随后 POP V96',[
        ('V65',43,GRAY),('V74',44,BLUE),('V82',45,GRAY),('V96',46,GRAY),('|',47,GRAY)])
    c=row(ax,1.1,'第 5 条：要重新写出栈 V65 V74 V82',[
        ('V65',56,GOLD),('V74',57,GREEN),('V82',58,GRAY),('|',59,GRAY)])
    arc(ax,c[56],(a[31][0],7.58),'L6：0.202',BLUE,-.30,(.3,6.2))
    arc(ax,c[56],(b[44][0],4.68),'L6：0.778',ORANGE,-.12,(2.2,3.1))
    ax.text(.7,.5,'输入 V65@56 → 模型实际输出 V74@57',fontsize=12,color=GREEN)
    notes(ax,8.0,8.5,'这次主要读到的 key，\n字面上本来就是 V74：\n\n上一条栈中的 V74@44：0.778\n再前一条的 V74@31：0.202\n\n该层对更早所有 V74 的\n注意力合计为 0.994。\n\n与左述 In-Context 例子的差别：\nIC 读 V2 的历史位置再输出 V5；\nFixed 读历史 V74 再输出 V74。\n\n这是读取路线的差别，\n还不是因果作用的证明。')
    save(fig,'concrete_02_fixed_stack',pdf)

    fig,ax=canvas('例 3：为什么 PUSH 关注 V15，却先输出 V13？',
                  '真实 In-Context query 50520：V11 → V14。V13 是先探索的分支，返回 V11 后再轮到 V15。')
    a=row(ax,7.3,'第 1 条：候选 V13 V15 V14；动作 PUSH V13',[
        ('V13',80,GRAY),('V15',81,BLUE),('V14',82,GRAY),('|',83,GRAY),('PUSH',84,ORANGE),('V13',85,GRAY)])
    ax.text(.6,5.85,'第 2 条：V11 V13 | V11 V13 | 空 | POP ;',fontsize=13)
    b=row(ax,2.0,'第 3 条：回到 V11；已经访问过 V13',[
        ('V11',96,GRAY),('|',97,GRAY),('V11',98,GRAY),('V13',99,GRAY),('|',100,GOLD),('V15',101,GREEN)])
    arc(ax,a[84],a[81],'L1：0.947',BLUE,.45,(3.7,9.0))
    arc(ax,b[100],(a[84][0],7.28),'L5：0.835',ORANGE,-.05,(6.15,5.1))
    arc(ax,b[100],(a[85][0],7.28),'L5：0.152',GRAY,-.16,(7.55,6.1))
    arc(ax,b[100],(a[81][0],7.28),'L6：0.528',GREEN,.24,(1.0,4.25))
    ax.text(.7,1.15,'第 1 条 PUSH@84 的下一输出是 V13，不是它在 L1 最关注的 V15。',fontsize=11)
    ax.text(.7,.65,'第 3 条输入 |@100 后，模型实际输出 V15@101。',fontsize=12,color=GREEN)
    notes(ax,9.0,8.4,'可观察到的回读顺序：\n\n① PUSH@84 的 L1 读取旧候选\n   V15@81，质量为 0.947。\n\n② 回到 V11 后，|@100 的 L5\n   读取旧 PUSH@84 及其参数。\n\n③ 同一位置的 L6 还直接读取\n   旧候选 V15@81。\n\n所以，早先 PUSH 的隐藏状态\n可能参与保留后续候选信息。\n但它不是唯一观察到的通路。')
    save(fig,'concrete_03_candidates',pdf)

print('Saved three concrete diagrams and PDF')
