"""Execute the prespecified trained-model and latency-replay experiment."""
import os,sys,json,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from sklearn.datasets import load_digits,load_breast_cancer,load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from raceaudit import targets,coefficients,solve_design,exchangeable_coefficients,pooled_allocation
ROOT=Path(__file__).resolve().parents[1]
def models(seed):
 return [GaussianNB(),make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500)),RandomForestClassifier(n_estimators=80,random_state=seed,n_jobs=1),ExtraTreesClassifier(n_estimators=80,random_state=seed,n_jobs=1),make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5)),make_pipeline(StandardScaler(),SVC())]
def traces(pred,y,env,rng):
 correct=(pred==y[:,None]).astype(float);n,m=pred.shape
 if env=='exponential':lat=rng.exponential(1,(n,m))
 else:
  base=np.array([.65,1.2,1.9,1.6,1.,1.35]);shared=rng.lognormal(0,.4,(n,1))
  lat=shared*base*rng.lognormal(0,.7,(n,m))
  if env=='correlated':lat*=.6+1.2*correct
 order=np.argsort(lat,axis=1,kind='stable')
 return order,np.take_along_axis(correct,order,axis=1),np.take_along_axis(lat,order,axis=1)
def geom(r,b,floor=.005):
 lo=0;hi=1
 for _ in range(100):
  z=(lo+hi)/2;q=np.maximum(floor,z**np.arange(1,r+1))
  if q.sum()<b:lo=z
  else:hi=z
 return np.maximum(floor,((lo+hi)/2)**np.arange(1,r+1))
def main():
 start=time.time();menu=targets(6,3);rows=[];mcs=[];fits=[]
 for dataset,loader in [('digits',load_digits),('breast_cancer',load_breast_cancer),('wine',load_wine)]:
  X,y=loader(return_X_y=True)
  for seed in range(100,110):
   train,hold=train_test_split(np.arange(len(y)),test_size=.5,stratify=y,random_state=seed)
   pilot,test=train_test_split(hold,test_size=.6,stratify=y[hold],random_state=seed+1000)
   pp=[];pt=[]
   for model in models(seed):
    model.fit(X[train],y[train]);pp.append(model.predict(X[pilot]));pt.append(model.predict(X[test]))
   pp=np.array(pp).T;pt=np.array(pt).T
   fits.append({'dataset':dataset,'seed':seed,'train':len(train),'pilot':len(pilot),'test':len(test),'accuracy':(pt==y[test,None]).mean(axis=0).tolist()})
   np.savez_compressed(ROOT/'data'/f'{dataset}_{seed}.npz',train=train,pilot=pilot,test=test,pilot_prediction=pp,test_prediction=pt,pilot_label=y[pilot],test_label=y[test])
   for e,env in enumerate(['exponential','heterogeneous','correlated']):
    op,rp,tp=traces(pp,y[pilot],env,np.random.default_rng(seed*100+e));ot,rt,tt=traces(pt,y[test],env,np.random.default_rng(seed*100+e+10))
    A,_,_=coefficients(op,rp,menu,3);B,J,D=coefficients(ot,rt,menu,3);n=len(test)
    np.savez_compressed(ROOT/'data'/f'{dataset}_{seed}_{env}_trace.npz',pilot_order=op,pilot_reward=rp,pilot_latency=tp,test_order=ot,test_reward=rt,test_latency=tt)
    for b in [.25,.5,1.,2.]:
     fitted=solve_design(A,b);oracle=solve_design(B,b)
     design={'uniform':np.full(3,b/3),'geometric':geom(3,b),'exchangeable':pooled_allocation(exchangeable_coefficients(6,3),np.ones(3),b,.005),'pilot':fitted['q'],'oracle':oracle['q']}
     for method,q in design.items():
      v=B@(1/q-1)/n
      rows.append({'dataset':dataset,'seed':seed,'environment':env,'budget':b,'method':method,'q':q.tolist(),'max_variance':float(v.max()),'mean_variance':float(v.mean()),'worst_target':sorted(menu[int(v.argmax())]),'pilot_gap':fitted['gap'],'oracle_gap':oracle['gap'],'baseline_accuracy':float(rt[:,0].mean()),'effect_min':float(D.mean(axis=0).min()),'effect_max':float(D.mean(axis=0).max())})
     if seed==100 and b==.5:
      u=np.random.default_rng(8800+e).random((500,n));theta=D.mean(axis=0)
      for method,q in design.items():
       prob=np.r_[1.,q][J];obs=u[:,:,None]<prob[None,:,:];Z=obs*D[None,:,:]/prob[None,:,:];est=Z.mean(axis=1)
       se=np.sqrt(np.sum(obs*D[None,:,:]**2*(1-prob)[None,:,:]/prob[None,:,:]**2,axis=1)/n**2)
       cover=((est-1.96*se<=theta)&(est+1.96*se>=theta)).mean(axis=0)
       complete=np.sum(obs*D[None,:,:],axis=1)/obs.sum(axis=1);err=est-theta;errcc=complete-theta
       mcs.append({'dataset':dataset,'environment':env,'method':method,'n':n,'q':q.tolist(),'max_abs_bias':float(np.abs(err.mean(axis=0)).max()),'max_rmse':float(np.sqrt((err**2).mean(axis=0)).max()),'max_exact_sd':float(np.sqrt((B@(1/q-1)/n).max())),'complete_case_max_abs_bias':float(np.abs(errcc.mean(axis=0)).max()),'coverage_min':float(cover.min()),'coverage_mean':float(cover.mean()),'realized_extra_mean':float((u[:,:,None]<q).sum(axis=2).mean())})
       np.savez_compressed(ROOT/'results'/f'mc_{dataset}_{env}_{method}.npz',estimate=est,complete_case=complete,truth=theta,se=se)
   print(dataset,seed,'done',flush=True)
 (ROOT/'results'/'experiment.json').write_text(json.dumps(rows,indent=2));(ROOT/'results'/'monte_carlo.json').write_text(json.dumps(mcs,indent=2));(ROOT/'results'/'training.json').write_text(json.dumps(fits,indent=2));(ROOT/'results'/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'elapsed_seconds':time.time()-start,'seed_range':[100,109]},indent=2));print('finished',time.time()-start)
if __name__=='__main__': main()
