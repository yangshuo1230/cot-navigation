"""Prefix timing and leakage checks for history-error readouts."""
import numpy as np
from cot_navigation.analysis.history_probe import history_target, score


def test_target_changes_only_after_consuming_the_error():
    assert history_target([3,4,5,6],5).tolist()==[0,0,1,1]
    # h[4] predicts token 5, which is absent from that activation's prefix.
    assert history_target([4],5)[0]==0


def test_probabilities_and_ranking_include_clean_negatives():
    result=score(np.array([0,0,1,1]),np.array([-2.,-1.,1.,2.]))
    assert result['auroc']==1 and result['fpr']==0 and result['tpr']==1
    assert result['negative_p']<result['positive_p']


def test_identical_features_cannot_rank_paired_labels():
    result=score([0,1,0,1],[.2,.2,-.4,-.4])
    assert result['auroc']==.5


def test_position_probes_learn_opposite_rules_without_sharing():
    import torch
    from cot_navigation.analysis.history_probe import fit_positions,position_scores
    torch.set_num_threads(1)
    def data(n,start):
        y=np.tile([0,1],n*2).astype(np.float32)
        cell=np.repeat([0,10000],n*2)
        x=((y*2-1)*np.where(cell==0,1,-1)).astype(np.float32)
        return dict(x=x[:,None,None],y=y,cell=cell,query=np.tile(np.repeat(np.arange(n)+start,2),2))
    fitted=fit_positions(data(40,0),data(10,100),steps=200,device='cpu')
    scores=position_scores(fitted,data(10,200))
    assert len(scores)==2 and all(r['auroc']==1 for r in scores)
    assert fitted['probe'].w[0,0,0]*fitted['probe'].w[1,0,0]<0


def test_onset_shortcut_removed_before_training_and_scoring():
    from cot_navigation.analysis.history_probe import paired_after_error
    # Event 0 has not yet made its error at cell 5; event 1 has. A token
    # identifying event 1 could otherwise obtain high AUC on shuffled labels.
    d=dict(event=np.array([0,0,1,1]),cell=np.array([5,5,5,5]),
           after=np.array([False,False,True,True]),y=np.array([0,0,0,1]))
    out=paired_after_error(d)
    assert out['event'].tolist()==[1,1] and out['y'].tolist()==[0,1]
