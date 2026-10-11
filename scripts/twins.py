"""Exact winner-log indistinguishability witnesses and allocation separation."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from raceaudit import *
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[]
 for m in [4,6,8,16,32,64,128,256,1024]:
  b=1.;p=exchangeable_coefficients(m,2);q=pooled_allocation(p,np.ones(2),b)
  paired_opt=2/m*(2/b-1);wrong=2/m*(1/q[1]-1)
  ex_opt=float(p@(1/q-1))
  assert abs(wrong/paired_opt-np.sqrt(m-2))<1e-10
  rows.append({'m':m,'q_exchangeable':q.tolist(),'q_paired':[.5,.5],'exchangeable_opt_variance':ex_opt,'paired_opt_variance':paired_opt,'paired_wrong_variance':wrong,'wrong_design_ratio':wrong/paired_opt,'optimal_risk_ratio':paired_opt/ex_opt})
 for m in [4,6]:
  menu=targets(m,2);orders=np.array(list(itertools.permutations(range(m))));rewards=np.ones_like(orders,float);rewards[:,0]=0
  A,_,_=coefficients(orders,rewards,menu,2);out=solve_design(A,1.,floor=.0001)
  np.testing.assert_allclose(out['q'],pooled_allocation(exchangeable_coefficients(m,2),np.ones(2),1.),atol=2e-5)
  pair=[]
  for a in range(m):pair.append([a,a^1]+[x for x in range(m) if x not in [a,a^1]])
  pair=np.array(pair);reward=np.ones_like(pair,float);reward[:,0]=0
  B,_,_=coefficients(pair,reward,menu,2);v=solve_design(B,1.,floor=.0001)
  np.testing.assert_allclose(v['q'],[.5,.5],atol=2e-5)
  assert set(orders[:,0])==set(pair[:,0]);assert np.all(rewards[:,0]==0) and np.all(reward[:,0]==0)
 (ROOT/'results'/'twins.json').write_text(json.dumps(rows,indent=2));print('twins verified; max ratio',rows[-1]['wrong_design_ratio'])
if __name__=='__main__':main()
