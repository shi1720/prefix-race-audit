import unittest,itertools,math
from fractions import Fraction as F
import numpy as np
from scipy.optimize import minimize
from raceaudit import *
class ResearchTests(unittest.TestCase):
 def test_exhaustive_prefix_replay(self):
  count=0
  for m in range(2,7):
   r=min(3,m-1)
   for order in itertools.permutations(range(m)):
    reward=np.array([(a%2) for a in order])
    for S in targets(m,r):
     j,d=replay(order,reward,S);self.assertLessEqual(j,r)
     rec=Record(tuple(order[:r+1]),tuple(reward[:r+1]),tuple([1]*(r+1)))
     self.assertEqual(estimate_record(rec,S),d);count+=1
  print('exhaustive replay cases',count)
 def test_exact_rational_moments(self):
  q=[F(1),F(3,5),F(1,4),F(1,10)];prob=[q[k]-(q[k+1] if k+1<len(q) else 0) for k in range(len(q))];count=0
  for order in itertools.permutations(range(4)):
   for ys in itertools.product([0,1],repeat=4):
    for S in targets(4,3):
     j,d=replay(order,ys,S);d=F(int(d));mean=F(0);second=F(0)
     for k,p in enumerate(prob):
      val=d/q[j] if k>=j else F(0);mean+=p*val;second+=p*val*val
     self.assertEqual(mean,d);self.assertEqual(second-mean**2,d*d*(1/q[j]-1));count+=1
  print('exact rational moment cases',count)
 def test_exchangeable_witness(self):
  for m in range(2,7):
   for r in range(1,min(4,m)):
    cnt=np.zeros(r)
    for order in itertools.permutations(range(m)):
     j=next(j for j,a in enumerate(order) if a>=r)
     if j:cnt[j-1]+=1
    cnt/=math.factorial(m);np.testing.assert_allclose(cnt,exchangeable_coefficients(m,r),atol=1e-14)
 def test_independent_primal_certificate(self):
  rng=np.random.default_rng(75);worst=0
  for z in range(100):
   r=int(rng.integers(2,7));k=int(rng.integers(2,12));A=rng.uniform(0,1,(k,r));c=rng.uniform(.2,3,r);b=.4*c.sum();out=solve_design(A,b,c)
   cons=[{'type':'ineq','fun':lambda x:b-c@x[:-1]},{'type':'ineq','fun':lambda x:x[:-2]-x[1:-1]},{'type':'ineq','fun':lambda x:x[-1]-A@(1/x[:-1]-1)}]
   x0=np.r_[np.full(r,.4),max(A@np.full(r,1.5))]
   sol=minimize(lambda x:x[-1],x0,method='SLSQP',bounds=[(.005,1)]*r+[(0,None)],constraints=cons,options={'ftol':1e-10,'maxiter':2000})
   self.assertTrue(sol.success,sol.message);self.assertLessEqual(out['gap'],2e-6);self.assertAlmostEqual(out['value'],sol.fun,places=5);worst=max(worst,out['gap'])
  print('100 primal/dual checks, worst gap',worst)
 def test_nonmonotone_requires_pooling(self):
  q=pooled_allocation([1,9,2],[1,1,1],1.);self.assertAlmostEqual(q[0],q[1]);self.assertGreater(q[1],q[2]);self.assertAlmostEqual(q.sum(),1.)
 def test_depth_and_hidden_data(self):
  q=[1,.6,.3,.1];self.assertEqual(sample_depth(q,.8),1);self.assertEqual(sample_depth(q,.2),3)
  rec=Record((0,1),(0.,1.),(1.,.6));self.assertEqual(estimate_record(rec,{0,1}),0.);self.assertAlmostEqual(estimate_record(rec,{0}),1/.6);self.assertEqual(estimate_record(rec,{1}),0.)
 def test_invalid_design(self):
  with self.assertRaises(ValueError):sample_depth([1,.2,.3],.1)
  with self.assertRaises(ValueError):solve_design(np.ones((2,3)),.001)
  with self.assertRaises(ValueError):Record((1,1),(0,1),(1,.5))
if __name__=='__main__':unittest.main()

class RuntimeTests(unittest.IsolatedAsyncioTestCase):
 async def test_winner_released_before_audit(self):
  import asyncio
  from raceaudit.runtime import start_race
  gates=[asyncio.Event() for _ in range(5)];cancelled=[]
  async def call(a):
   try:
    await gates[a].wait();return a,float(a%2)
   except asyncio.CancelledError:
    cancelled.append(a);raise
  launch=asyncio.create_task(start_race([call(a) for a in range(5)],[1,.8,.5],.3))
  gates[0].set();winner,audit=await launch
  self.assertEqual(winner,(0,0.));self.assertFalse(audit.done())
  gates[1].set();await asyncio.sleep(0);await asyncio.sleep(0)
  gates[2].set();rec=await audit
  self.assertEqual(rec.endpoint,(0,1,2));self.assertEqual(sorted(cancelled),[3,4])
