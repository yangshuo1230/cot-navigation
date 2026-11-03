"""Frozen ridge implementation reused from mechanism_probes_20261101."""
import numpy as np
ALPHAS=[1e-5,1e-4,1e-3,.01,.1]
def fit(x,y,alpha,w=None):
 if w is None:w=np.ones(len(x))/len(x)
 w=w/w.sum();mean=(x*w[:,None]).sum(0);scale=np.sqrt(((x-mean)**2*w[:,None]).sum(0));scale[scale<1e-5]=1
 z=np.c_[(x-mean)/scale,np.ones(len(x))];penalty=np.eye(z.shape[1])*alpha;penalty[-1,-1]=0
 b=np.linalg.solve(z.T@(z*w[:,None])+penalty,z.T@(y*w[:,None]));coef=b[:-1]/scale[:,None];intercept=b[-1]-mean@coef
 return np.vstack([coef,intercept])
def pred(x,b):return np.c_[x,np.ones(len(x))]@b
def weights(meta):
 ids=[r['index'] for r in meta];return np.array([1/ids.count(ix) for ix in ids])
