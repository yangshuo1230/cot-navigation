"""Checks for linear readouts, intrinsic graph labels and causal capture."""
import numpy as np
import torch
import linear
from study import attrs,phase_positions
from cot_navigation.models.transformer import TransformerLM


def run():
    rng=np.random.default_rng(9);x=rng.normal(size=(240,2,8)).astype(np.float32)
    ids=np.arange(240);rng.shuffle(ids);split=(ids[:140],ids[140:190],ids[190:])
    y=2*x[:,0,0]-3*x[:,0,1]+5
    pred,_=linear.fit(x[:,:1],y,split)
    assert linear.regression(y[split[2]],pred[split[2],0,0])['r2']>.99
    y=(x[:,0,0]>.1).astype(int)
    pred,fit=linear.fit(x[:,:1],y,split,binary=True)
    assert linear.classification(y[split[2]],pred[split[2],0,0],fit['threshold'][0][0])['balanced_accuracy']>.85
    assert linear.auc([0,1,0,1],[0,1,0,1])==1
    assert linear.auc([0,1,0,1],[1,1,1,1])==.5
    a=attrs([[1,2],[3],[3],[]],2)
    assert a[:,2].tolist()==[0,1,1,2]
    assert a[:,3].tolist()==[2,1,1,0]
    assert a[:,6].tolist()==[0,1,1,3]
    assert a[:,7].tolist()==[3,1,1,0]
    assert phase_positions(0)=={'prompt':0}
    assert phase_positions(4)['end']==4 and 'k8' not in phase_positions(4)
    model=TransformerLM(32,2,1,16,32,32).cuda().eval()
    first=torch.tensor([[2,3,4,5,6,7]],device='cuda')
    altered=first.clone();altered[:,4:]=torch.tensor([9,10],device='cuda')
    captures=[]
    with torch.no_grad():
        for tokens in (first,altered):
            c={};model(tokens,capture=c);captures.append(c['resid'])
    for l in range(2):assert torch.allclose(captures[0][l][:,:4],captures[1][l][:,:4],atol=1e-6)
    print('PASS affine recovery, binary readout, tied AUROC, DAG labels, progress anchors, no-future causal capture')


if __name__=='__main__':run()
