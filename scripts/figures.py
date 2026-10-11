"""Publication figures and tables generated exclusively from saved results."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
COL=['#1d3557','#e76f51','#2a9d8f','#8a5a9b']
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)
def main():
 fig,ax=plt.subplots(figsize=(8.8,3.0));ax.axis('off')
 labels=['A: first response\nreturn to client','B: second response\nretain with probability q₂','C: third response\nretain with probability q₃','D: fourth response\nretain with probability q₄']
 for j,l in enumerate(labels):
  x=.02+j*.25
  ax.add_patch(FancyBboxPatch((x,.42),.21,.39,boxstyle='round,pad=.01',fc=COL[j],ec='none',transform=ax.transAxes))
  ax.text(x+.105,.615,l,ha='center',va='center',color='white',fontsize=9,transform=ax.transAxes)
  if j<3:ax.annotate('',xy=(x+.245,.615),xytext=(x+.215,.615),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#555'})
 ax.text(.5,.19,'Remove {A, B} → choose C.   One retained prefix serves every deletion set of size ≤ 3.',ha='center',transform=ax.transAxes,fontsize=10)
 ax.text(.5,.04,'Audit randomization changes observation; the response-suppression intervention changes selection.',ha='center',transform=ax.transAxes,fontsize=9,color='#555')
 save(fig,'prefix')
 twins=json.loads((ROOT/'results/twins.json').read_text());fig,axs=plt.subplots(1,2,figsize=(8.8,3.15))
 m=np.array([x['m'] for x in twins]);axs[0].plot(m,[x['q_exchangeable'][0] for x in twins],'-o',label='Exchangeable q₂',color=COL[0]);axs[0].plot(m,[x['q_exchangeable'][1] for x in twins],'-s',label='Exchangeable q₃',color=COL[1]);axs[0].axhline(.5,color=COL[2],ls='--',label='Paired q₂ = q₃')
 axs[0].set(xscale='log',xlabel='Number of endpoints m',ylabel='Retention probability',title='(a) Same winner logs; different allocations');axs[0].legend(fontsize=8)
 axs[1].loglog(m,[x['wrong_design_ratio'] for x in twins],'-o',color=COL[0],label='Wrong design / paired optimum');axs[1].loglog(m,[x['optimal_risk_ratio'] for x in twins],'-s',color=COL[1],label='Paired / exchangeable optimum');axs[1].set(xlabel='Number of endpoints m',ylabel='Audit variance ratio',title='(b) Exact information separation');axs[1].legend(fontsize=8)
 fig.tight_layout();save(fig,'twins')
 rows=json.loads((ROOT/'results/experiment.json').read_text());fig,axs=plt.subplots(1,3,figsize=(8.8,3.3),sharey=True)
 methods=['geometric','exchangeable','pilot','oracle'];rng=np.random.default_rng(1)
 for ax,e in zip(axs,['exponential','heterogeneous','correlated']):
  for j,method in enumerate(methods):
   vals=[]
   for x in rows:
    if x['environment']==e and x['method']==method and x['budget']==.5:
     base=next(t['max_variance'] for t in rows if t['method']=='uniform' and all(t[k]==x[k] for k in ['dataset','seed','environment','budget']))
     if base>0:vals.append(np.log2(x['max_variance']/base))
   ax.scatter(j+rng.uniform(-.14,.14,len(vals)),vals,s=13,color=COL[j],alpha=.65)
   ax.plot([j-.2,j+.2],[np.median(vals)]*2,color='black',lw=2)
  ax.axhline(0,color='#555',ls='--',lw=.8);ax.set_title(e.capitalize());ax.set_xticks(range(4),['Geom.','Exch.','Pilot','Oracle'],rotation=30);ax.grid(axis='y',alpha=.15)
 axs[0].set_ylabel('log₂ (worst audit variance / uniform)');fig.supxlabel('Completion budget b = 0.5; 30 dataset–split conditions per environment',fontsize=9);fig.tight_layout();save(fig,'empirical')
 fig,axs=plt.subplots(1,2,figsize=(8.8,3.1))
 for method,col in [('uniform',COL[0]),('pilot',COL[1])]:
  data=np.load(ROOT/'results'/f'mc_digits_heterogeneous_{method}.npz');k=np.argmax(np.abs(data['complete_case'].mean(axis=0)-data['truth']))
  for ax,key in zip(axs,['estimate','complete_case']):
   ax.hist(data[key][:,k]-data['truth'][k],bins=26,alpha=.45,color=col,label=method,density=True)
 for ax,title in zip(axs,['(a) Inverse-inclusion estimator','(b) Complete-case estimator']):
  ax.axvline(0,color='black',lw=1);ax.set(xlabel='Estimate minus finite-cohort effect',ylabel='Density',title=title);ax.legend(fontsize=8)
 fig.tight_layout();save(fig,'bias')
 # One table per dataset/environment, all 10 splits retained.
 lines=[];summary=[]
 for dataset in ['digits','breast_cancer','wine']:
  for env in ['exponential','heterogeneous','correlated']:
   row=[dataset.replace('_',' '),env];stats={}
   for method in ['exchangeable','pilot','oracle']:
    gains=[]
    for x in rows:
     if x['dataset']==dataset and x['environment']==env and x['method']==method and x['budget']==.5:
      base=next(t['max_variance'] for t in rows if t['method']=='uniform' and all(t[k]==x[k] for k in ['dataset','seed','environment','budget']))
      gains.append(100*(1-x['max_variance']/base) if base else 0.)
    stats[method]={'median_gain':float(np.median(gains)),'negative_count':int(sum(t<-1e-5 for t in gains)),'min_gain':float(min(gains)),'max_gain':float(max(gains))}
    row.append(f"{np.median(gains):.1f} ({sum(t<-1e-5 for t in gains)}/10)")
   lines.append(' & '.join(row)+r' \\');summary.append({'dataset':dataset,'environment':env,**stats})
 (ROOT/'paper'/'table_results.tex').write_text('\n'.join(lines))
 (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()
