"""Analytic grid and independent software-execution checks."""
import sys,json,asyncio,time,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from raceaudit import *
ROOT=Path(__file__).resolve().parents[1]
async def endpoint(a,delay,reward):
 await asyncio.sleep(delay)
 return a,reward,time.perf_counter()
async def race(delays,rewards,q,u):
 depth=sample_depth(q,u);tasks=[asyncio.create_task(endpoint(a,d,rewards[a])) for a,d in enumerate(delays)]
 observed=[];first=None
 try:
  for completed in asyncio.as_completed(tasks):
   a,y,t=await completed
   observed.append((a,y,t))
   if first is None:first=t # client response can be returned now via callback
   if len(observed)==depth:break
 finally:
  for task in tasks:
   if not task.done():task.cancel()
  await asyncio.gather(*tasks,return_exceptions=True)
 rec=Record(tuple(x[0] for x in observed),tuple(x[1] for x in observed),tuple(q[:depth]))
 return rec,first,observed[-1][2]
async def software():
 rng=np.random.default_rng(981);checks=0;timing=[]
 q=np.array([1.,.75,.5,.25]);us=[.9,.6,.4,.1]
 for z in range(30):
  order=rng.permutation(6);delays=np.zeros(6);delays[order]=.003*np.arange(1,7);rewards=rng.integers(0,2,6)
  for u in us:
   rec,first,last=await race(delays,rewards,q,u)
   assert rec.endpoint==tuple(order[:len(rec.endpoint)])
   for S in targets(6,3):
    # Compare to an independently enumerated full-table expectation formula.
    j=next(j for j,a in enumerate(order) if a not in S)
    expected=(rewards[order[j]]-rewards[order[0]])/q[j] if j<len(rec.endpoint) else 0
    assert abs(estimate_record(rec,S)-expected)<1e-12;checks+=1
   timing.append({'depth':len(rec.endpoint),'post_response_wait_seconds':last-first})
 return {'races':120,'policy_checks':checks,'timing':timing,'delay_model':'engineered asyncio sleeps, not measured provider inference'}
def main():
 grid=[]
 for m in [8,16,32,64]:
  for r in [1,2,4]:
   for b in [.1,.25,.5,1.]:
    if b>=r:continue
    p=exchangeable_coefficients(m,r);q=pooled_allocation(p,np.ones(r),b,.0);v=float(p@(1/q-1));unif=float(p.sum()*(r/b-1))
    grid.append({'m':m,'r':r,'b':b,'q':q.tolist(),'variance':v,'uniform_variance':unif,'improvement':1-v/unif,'number_targets':sum(__import__('math').comb(m,k) for k in range(1,r+1))})
 (ROOT/'results'/'analytic.json').write_text(json.dumps(grid,indent=2))
 (ROOT/'results'/'software.json').write_text(json.dumps(asyncio.run(software()),indent=2))
 # Fixed intervention counterexample: joint launch gives A=1ms,Y=0; B=2ms,Y=1.
 # Removing A before launch changes B's model mode to Y=0. Suppression still Y=1.
 (ROOT/'results'/'invariance_counterexample.json').write_text(json.dumps({'baseline_reward':0,'suppression_effect':1,'physical_removal_effect':0,'bias':1,'explanation':'Surviving endpoint reward changes under launch removal; no censoring correction identifies the changed mechanism.'},indent=2))
 print('analytic conditions',len(grid),'software checks complete')
if __name__=='__main__':main()
