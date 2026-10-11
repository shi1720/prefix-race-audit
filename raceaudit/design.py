"""Convex rank-only allocation. Dual values independently certify solutions."""
import numpy as np
from scipy.optimize import minimize

def exchangeable_coefficients(m,r):
    if not 0<r<m: raise ValueError('require 0<r<m')
    survival=1.0; out=[]
    for j in range(1,r+2):
        p=survival*(m-r)/(m-j+1)
        if j>=2: out.append(p)
        survival*=max(0,r-j+1)/(m-j+1)
    return np.asarray(out)

def pooled_allocation(a,c,b,floor=0.0):
    """Weighted PAVA on a/c followed by square-root budget water filling."""
    a=np.asarray(a,float);c=np.asarray(c,float)
    if a.shape!=c.shape or np.any(a<0) or np.any(c<=0) or not floor>=0 or floor>1 or b < floor*c.sum()-1e-10:
        raise ValueError('infeasible or invalid allocation')
    if b>=c.sum(): return np.ones(len(a))
    blocks=[]
    for j,(x,y) in enumerate(zip(a,c)):
        blocks.append([j,j+1,x,y])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3] < blocks[-1][2]/blocks[-1][3]-1e-15:
            v=blocks.pop();u=blocks.pop();blocks.append([u[0],v[1],u[2]+v[2],u[3]+v[3]])
    z=np.zeros(len(a))
    for l,h,x,y in blocks:z[l:h]=np.sqrt(x/y)
    if not np.any(z): return np.full(len(a),floor)
    def alloc(scale):return np.clip(scale*z,floor,1)
    lo=0.;hi=1.
    while c@alloc(hi)<b and hi<1e18:hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if c@alloc(mid)<b:lo=mid
        else:hi=mid
    return alloc((lo+hi)/2)

def solve_design(A,b,c=None,floor=.005):
    """Minimize max_S sum_j A_Sj*(1/q_j-1). Return a dual certificate."""
    A=np.asarray(A,float); k,r=A.shape;c=np.ones(r) if c is None else np.asarray(c,float)
    if np.any(A<0) or len(c)!=r or np.any(c<=0) or b < floor*c.sum()-1e-10:raise ValueError('invalid design')
    b=min(b,float(c.sum()))
    if not np.any(A):return {'q':np.full(r,floor),'value':0.,'lower':0.,'gap':0.,'lambda':np.ones(k)/k}
    # Maximizing concave dual over a simplex; PAVA computes its inner minimizer.
    def calc(lam):
        q=pooled_allocation(lam@A,c,b,floor)
        v=A@(1/q-1)
        return -float(lam@v),-v
    best=None
    for init in [np.ones(k)/k,np.eye(1,k,int(np.argmax(A.sum(axis=1))))[0]]:
        sol=minimize(lambda x:calc(x)[0],init,jac=lambda x:calc(x)[1],method='SLSQP',bounds=[(0,1)]*k,
                     constraints=[{'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(k)}],
                     options={'ftol':1e-11,'maxiter':1000})
        lam=np.maximum(sol.x,0);lam/=lam.sum();q=pooled_allocation(lam@A,c,b,floor)
        v=A@(1/q-1); lower=float(lam@v);gap=float(v.max()-lower)
        item={'q':q,'value':float(v.max()),'lower':lower,'gap':gap,'lambda':lam}
        if best is None or gap<best['gap']:best=item
        if gap<1e-8:break
    # Primal fallback if nonsmooth dual leaves a material gap.
    if best['gap']>1e-7:
        q0=best['q'];x0=np.r_[q0,best['value']]
        cons=[{'type':'ineq','fun':lambda x:b-c@x[:-1]},
              {'type':'ineq','fun':lambda x:x[:-2]-x[1:-1]},
              {'type':'ineq','fun':lambda x:x[-1]-A@(1/x[:-1]-1)}]
        sol=minimize(lambda x:x[-1],x0,method='SLSQP',bounds=[(floor,1)]*r+[(0,None)],constraints=cons,
                     options={'ftol':1e-12,'maxiter':2000})
        if sol.success:
            q=sol.x[:-1];val=float((A@(1/q-1)).max())
            if val<best['value']:best.update(q=q,value=val,gap=val-best['lower'])
    if np.any(np.diff(best['q'])>1e-7) or c@best['q']>b+1e-7: raise RuntimeError('infeasible numerical result')
    return best
