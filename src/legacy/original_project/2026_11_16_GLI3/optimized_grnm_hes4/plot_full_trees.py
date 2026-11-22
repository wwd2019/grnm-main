#!/usr/bin/env python3
"""Plot all available GLI3 candidate pairs for three comparison methods."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
seed=set(pd.read_csv(ROOT/'inputs/target_scores.tsv',sep='\t').target)|{'HES4'}
universe=seed & set((ROOT/'scenic/extended/genes.txt').read_text().splitlines())
candidates=set(pd.read_csv(ROOT/'inputs/PRIORITY_with_strong_local_DA_peak.tsv',sep='\t').gene_symbol)&universe
assert len(universe)==1794 and len(candidates)==48
sc=pd.read_csv(HERE/'GLI3_all_scores.tsv',sep='\t').sort_values(['score','target'],ascending=[False,True])
assert set(sc.target)==universe
p=pd.read_csv(ROOT/'inputs/metacells.csv'); p=p[p.tf.eq('GLI3')&p.target.isin(universe)&p.padj.between(0,.05)].copy(); p['mag']=p.estimate.abs(); p=p.sort_values(['padj','mag','target'],ascending=[True,False,True]).drop_duplicates('target').rename(columns={'padj':'score'})
b=pd.read_csv(ROOT/'scenic/extended/run/adjacencies.tsv',sep='\t'); b=b[b.TF.eq('GLI3')&b.target.isin(universe)&b.importance.gt(0)].rename(columns={'importance':'score'}).sort_values(['score','target'],ascending=[False,True]).drop_duplicates('target')
sets=[('pando+GRN','All scored targets',sc,1794),('Pando metacells','Filtered external edges',p,50),('GRNBoost2 only','Before motif filtering',b,309)]
ink,teal,muted='#182b38','#087e76','#667785'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(14,7)); fig.subplots_adjust(left=.035,right=.965,top=.81,bottom=.22,wspace=.12)
fig.text(.04,.96,'GLI3 ranked candidate trees',fontsize=17,fontweight='bold',color=ink)
fig.text(.04,.918,'All available targets within the common evaluation universe',fontsize=10,color=muted)
exports=[]
for ax,(method,subtitle,frame,expected) in zip(axes,sets):
 d=frame[['target','score']].reset_index(drop=True).copy(); d.insert(0,'rank',np.arange(1,len(d)+1)); d.insert(0,'source','GLI3'); d.insert(0,'method',method); d['hit']=d.target.isin(candidates); assert len(d)==expected and d.target.is_unique; exports.append(d)
 ang=np.pi/2-np.arange(len(d))*2*np.pi/len(d); x=np.cos(ang); y=np.sin(ang); hits=d.hit.to_numpy();
 # Individual straight edges preserve the model's ranking list without assigning a sign.
 segments=np.stack([np.column_stack([.14*x,.14*y]),np.column_stack([.99*x,.99*y])],axis=1)
 ax.add_collection(LineCollection(segments[~hits],colors='#b9c7cd',linewidths=.30 if len(d)>500 else .55,alpha=.26 if len(d)>500 else .65,zorder=1))
 ax.add_collection(LineCollection(segments[hits],colors=teal,linewidths=.55 if len(d)>500 else 1.4,alpha=.72,zorder=2))
 ax.scatter(x[~hits],y[~hits],s=3 if len(d)>500 else (7 if len(d)>100 else 17),color='#a9b9c2',zorder=3)
 ax.scatter(x[hits],y[hits],s=14 if len(d)>500 else (23 if len(d)>100 else 34),color=teal,zorder=4)
 ax.scatter(0,0,s=1050,color=ink,zorder=5); ax.text(0,0,'GLI3',ha='center',va='center',color='white',weight='bold',fontsize=11,zorder=6)
 ax.set(xlim=(-1.3,1.3),ylim=(-1.23,1.23),aspect='equal'); ax.axis('off'); ax.set_title(method,fontsize=13,fontweight='bold',color=ink,pad=14)
 ax.text(0,-1.23,f'{len(d):,} targets · {int(d.hit.sum())} candidate-table matches',ha='center',va='center',fontsize=9,color=ink,weight='bold')
 ax.text(0,1.20,subtitle,ha='center',va='center',fontsize=9,color=muted)
 # A readable inset listing ranked genes avoids hiding the rest of a dense tree.
 names=[f'{r.rank}. {r.target}'+(' *' if r.hit else '') for r in d.head(10).itertuples(index=False)]
 ax.text(-1.18,-1.43,'Top 10: '+', '.join(names[:5])+'\n             '+', '.join(names[5:]),ha='left',va='top',fontsize=7.4,color=muted,linespacing=1.5,clip_on=False)
 important=d[d.target.isin(['HES4','HES5'])].set_index('target')['rank'].to_dict()
 ax.text(-1.18,-1.70,'HES4: '+(f"#{important['HES4']}" if 'HES4' in important else 'not listed')+'   HES5: '+(f"#{important['HES5']}" if 'HES5' in important else 'not listed'),ha='left',va='top',fontsize=8.5,color=teal,weight='bold',clip_on=False)
all_edges=pd.concat(exports,ignore_index=True); all_edges.to_csv(HERE/'gli3_full_trees_edges.tsv',sep='\t',index=False)
handles=[Line2D([],[],marker='o',linestyle='None',color=teal,label='Candidate-table match'),Line2D([],[],marker='o',linestyle='None',color='#a9b9c2',label='Other target')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),ncol=2,frameon=False,fontsize=9)
fig.text(.04,.045,'pando+GRN ranks all 1,794 genes: lines are candidate scores, not established edges. Pando shows retained edges.',fontsize=8,color=muted)
fig.text(.04,.022,'GRNBoost2 is before motif filtering; line color does not encode activation or repression. HES4 inclusion is exploratory.',fontsize=8,color=muted)
for extension in ('png','pdf','svg'): fig.savefig(HERE/f'gli3_full_three_trees.{extension}',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(fig)
for method,_,df,_ in sets: print(method,'rows',len(df),'candidate matches',int(df.target.isin(candidates).sum()))
