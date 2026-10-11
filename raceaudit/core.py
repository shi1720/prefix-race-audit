from dataclasses import dataclass
from itertools import combinations
import numpy as np

@dataclass(frozen=True)
class Record:
    """Only revealed responses; inclusion[k]=Pr(reveal rank k+1)."""
    endpoint: tuple
    reward: tuple
    inclusion: tuple
    def __post_init__(self):
        if not len(self.endpoint)==len(self.reward)==len(self.inclusion) or not self.endpoint:
            raise ValueError('unaligned or empty prefix')
        if len(set(self.endpoint)) != len(self.endpoint):
            raise ValueError('duplicate endpoint')
        q=np.asarray(self.inclusion)
        if q[0]!=1 or np.any(q<=0) or np.any(q>1) or np.any(np.diff(q)>1e-12):
            raise ValueError('invalid cumulative inclusion probabilities')
        if not all(0<=y<=1 for y in self.reward):
            raise ValueError('rewards must be in [0,1]')

def targets(m,r):
    if not 0<r<m: raise ValueError('require 0<r<m')
    return [frozenset(s) for k in range(1,r+1) for s in combinations(range(m),k)]

def sample_depth(q,u):
    q=np.asarray(q,float)
    if q.ndim!=1 or q[0]!=1 or np.any(q<=0) or np.any(q>1) or np.any(np.diff(q)>1e-12) or not 0<=u<1:
        raise ValueError('invalid q or uniform variate')
    return int(np.count_nonzero(u<q))

def estimate_record(record,removed):
    """A censored record contributes zero, never an imputed hidden response."""
    removed=frozenset(removed)
    for j,a in enumerate(record.endpoint):
        if a not in removed:
            return (record.reward[j]-record.reward[0])/record.inclusion[j]
    return 0.0

def replay(order,reward,removed):
    for j,a in enumerate(order):
        if a not in removed: return j,float(reward[j]-reward[0])
    raise ValueError('all endpoints removed')

def coefficients(order,reward,menu,r):
    n=len(order); A=np.zeros((len(menu),r)); J=np.zeros((n,len(menu)),dtype=int); D=np.zeros_like(J,dtype=float)
    for k,S in enumerate(menu):
        keep=~np.isin(order,list(S)); j=np.argmax(keep,axis=1)
        delta=reward[np.arange(n),j]-reward[:,0]
        J[:,k]=j; D[:,k]=delta
        for z in range(1,r+1): A[k,z-1]=np.mean((j==z)*delta**2)
    return A,J,D
