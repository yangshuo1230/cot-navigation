"""Six measured token-route diagrams, matching the user's six topics."""
from pathlib import Path
import textwrap,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
ALL=['context_n16','context_n100','fixed_scale_n100','fixed_simple_n100','fixed_hard_n100']
DATA={k:json.loads((HERE/f'{k}.json').read_text()) for k in ALL}
SHOW=['context_n16','fixed_simple_n100','fixed_hard_n100']
LABELS={'context_n16':'In-Context N16','fixed_simple_n100':'Fixed simple N100','fixed_hard_n100':'Fixed hard N100'}
BLUE='#0072B2';GREEN='#009E73';ORANGE='#D55E00';PURPLE='#8064A2';GOLD='#D99A00';INK='#222222'

def peak(model,metric):
    m=DATA[model]['metrics'].get(metric)
    if not m:return None
    values=m['query_macro'];li=int(np.argmax(values))
    return li+1,float(values[li]),m['n_events']

def box(ax,x,y,text,w=2.4,h=.62,color=BLUE,face='#F5F9FC',fs=9):
    ax.add_patch(FancyBboxPatch((x-w/2,y),w,h,boxstyle='round,pad=.04,rounding_size=.10',lw=1.4,edgecolor=color,facecolor=face,zorder=3))
    ax.text(x,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,zorder=4)

def arrow(ax,start,end,text,color,rad=0,label_offset=.99):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.6,color=color,connectionstyle=f'arc3,rad={rad}',zorder=2))
    ax.text(end[0],end[1]+label_offset,text,ha='center',va='center',color=color,fontsize=9)

def route(ax,y,title,receiver,keys,model):
    """Arrow orientation is always input receiver -> attended key."""
    ax.text(.3,y+3.0,title,ha='left',fontsize=11,color=INK)
    box(ax,5,y,receiver,w=4.2,h=.76,color=GOLD,face='#FFF6DE')
    xs=np.linspace(1.8,8.2,len(keys)) if len(keys)>1 else [5]
    for j,(x,(label,metric,color)) in enumerate(zip(xs,keys)):
        result=peak(model,metric) if metric else None
        box(ax,x,y+1.75,label,w=2.7,h=.72,color=color if result else '#AAAAAA',face='#F4F6F7')
        if result:
            li,mass,n=result
            arrow(ax,(5+(j-(len(keys)-1)/2)*.16,y+.82),(x,y+1.70),f'L{li}  {mass:.3f}',color,rad=(j-(len(keys)-1)/2)*.11)
        else:ax.text(x,y+2.65,'not applicable',ha='center',fontsize=8,color='#777777')

def canvas(title,subtitle):
    fig,axes=plt.subplots(1,3,figsize=(17.5,7.9))
    fig.subplots_adjust(left=.025,right=.985,top=.86,bottom=.08,wspace=.06)
    for ax,name in zip(axes,SHOW):
        ax.set(xlim=(0,10),ylim=(-.3,8.2));ax.axis('off')
        ax.text(5,7.95,LABELS[name]+f"  (best {DATA[name]['step']//1000}k)",ha='center',va='center',fontsize=12,fontweight='bold')
    fig.suptitle(title,fontsize=17,y=.985)
    fig.text(.5,.93,subtitle,ha='center',fontsize=10,color='#444444')
    return fig,axes

def finish(fig,name,pdf,example=False):
    note='Example arrows: exact raw attention at the indicated layer.' if example else 'Labels: highest query-mean raw attention across L1-L6, shown descriptively; layers differ across arrows.'
    fig.text(.025,.025,note+'  Arrows point from INPUT receiver to read KEY, not proven information transfer.',fontsize=9,color='#444444')
    fig.savefig(HERE/f'{name}.png',dpi=170,facecolor='white');pdf.savefig(fig);plt.close(fig)

with PdfPages(HERE/'dfs_routes_comparison.pdf') as pdf:
    fig,axes=canvas('1. Candidate lists: graph-position reads versus explicit-state reads',
                     'Correct canonical prefixes, FP32; 20 held-out queries per model. Fixed models have no EDGE list in their input.')
    for ax,name in zip(axes,SHOW):
        route(ax,4.2,'First visit to the current node','input visited |\nnext: first candidate',[
            ('current node\nin visited','candidate_fresh_current_visited',GREEN),
            ('current\nstack top','candidate_fresh_stack_top',BLUE),
            ('edge destinations\nshifted by +3','candidate_fresh_shifted_edge_dest',PURPLE)],name)
        route(ax,.1,'Revisiting a current node after backtracking','input visited |\nnext: first candidate',[
            ('current node\nin visited','candidate_repeat_current_visited',GREEN),
            ('previous same-node\nPUSH + argument','candidate_repeat_previous_push_pair',ORANGE)],name)
    finish(fig,'01_candidates',pdf)

    fig,axes=canvas('2. Reading candidates at PUSH: attention is not the selected answer',
                     'One measured three-candidate example per model; topology-index aliases Vn are local to each graph.')
    for ax,name in zip(axes,SHOW):
        r=DATA[name]
        if name=='context_n16':examples=[r['pictured_queries']['50520']['push_three_'+str(j)][0] for j in (1,2,3)]
        else:examples=[r['examples']['push_three_'+str(j)] for j in (1,2,3)]
        first=examples[0]
        assert len({(e['query_id'],e['q']) for e in examples})==1
        ax.text(.3,7.1,f"Query {first['query_id']}, record {first['record']+1}, receiver q={first['q']}",fontsize=10)
        box(ax,5,2.1,'input <PUSH>\nnext model: '+first['native_prediction'],w=4.1,h=.9,color=GOLD,face='#FFF6DE')
        for j,(x,e,color) in enumerate(zip([1.8,5,8.2],examples,[ORANGE,BLUE,GREEN])):
            key=e['keys'][0];a=key['mass'];li=int(np.argmax(a))
            box(ax,x,5.0,key['token']+f'\n(candidate {j+1})',w=2.5,h=.8,color=color)
            arrow(ax,(5+(j-1)*.18,3.1),(x,4.93),f'L{li+1}  {a[li]:.3f}',color,rad=(j-1)*.12,label_offset=1.2)
        ax.text(5,1.5,'Next reference token: '+first['predicted_reference'],ha='center',fontsize=10)
        li,mass,n=peak(name,'push_arg_chosen')
        ax.text(5,.65,f'Across diagnostic queries: chosen candidate\nreceives mean {mass:.3f} at L{li} ({n} events).',ha='center',fontsize=10,color='#444444')
    finish(fig,'02_push_choice',pdf,example=True)

    fig,axes=canvas('3. Stack writing: source recovery and the newly pushed node',
                     'The receiver is the token BEFORE the stack node being predicted. These are distinct prediction substeps.')
    for ax,name in zip(axes,SHOW):
        route(ax,4.2,'Starting a later stack record','input previous ;\nnext: source s',[
            ('query\nsource s','stack_root_query_source',GREEN),
            ('prompt\nseparator |','stack_root_prompt_bar',BLUE),
            ('earlier occurrences\nof source s','stack_root_history_source',PURPLE)],name)
        route(ax,.1,'Appending the new stack top after PUSH','input last parent-stack token\nnext: pushed node u',[
            ('previous\nPUSH argument u','stack_new_top_push_arg',ORANGE)],name)
    finish(fig,'03_stack_source_and_push',pdf)

    fig,axes=canvas('4. Retained stack nodes after POP: carrier positions versus same-node history',
                     'A first-visited-token carrier is a positional hypothesis; attention alone does not establish its stored content.')
    for ax,name in zip(axes,SHOW):
        route(ax,4.2,'First-visited carrier hypothesis','input: first visited token\nhidden state at this position',[
            ('same-record\nsecond-from-top\nstack node','visited_first_second_stack',BLUE)],name)
        route(ax,.1,'Predict a retained stack node','input preceding stack token\nnext: retained node v',[
            ('first-visited\ncarrier positions\nmatching node v','stack_kept_predict_matching_carrier',ORANGE),
            ('earlier positions\nprinting node v','stack_kept_history_same',PURPLE)],name)
    finish(fig,'04_retained_stack',pdf)

    fig,axes=canvas('5. Visited writing: strong differences in the positions being read',
                     'Historical same-name matches refer to visited fields; source-history reads include stack and other earlier CoT positions.')
    for ax,name in zip(axes,SHOW):
        route(ax,4.2,'Writing the first visited node','input separator after stack\nnext: source s',[
            ('prompt\nseparator |','visited_first_prompt_bar',BLUE),
            ('current\nstack source s','visited_first_stack_root',GREEN),
            ('earlier\nsource s positions','visited_first_history_source',PURPLE)],name)
        route(ax,.1,'Writing a later visited node','input previous visited token\nnext: visited node v',[
            ('earlier source s\n(any CoT field)','visited_next_history_source_any',ORANGE),
            ('current stack\nsource s','visited_next_current_stack_source',GREEN),
            ('earlier visited\noccurrences of v','visited_next_history_same',PURPLE)],name)
    finish(fig,'05_visited',pdf)

    fig,axes=canvas('6. Action prediction: inspect the last separator, not the action token',
                     'Each receiver is the THIRD field separator, after candidates. Stack-top attention varies with action and model.')
    for ax,name in zip(axes,SHOW):
        route(ax,4.2,'Predicting POP','input last |\nnext: <POP>',[
            ('current\nstack top','pop_stack_top',ORANGE)],name)
        route(ax,.1,'Predicting FOUND','input last |\nnext: <FOUND>',[
            ('current\nstack top','found_stack_top',GREEN)],name)
    finish(fig,'06_actions',pdf)

print('Saved six diagrams and one six-page PDF')
