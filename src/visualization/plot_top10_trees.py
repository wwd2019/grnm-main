#!/usr/bin/env python3
"""Draw comparable GLI3 top-10 radial trees from archived ranking tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]; HERE=ROOT/'results/generated'; HERE.mkdir(parents=True,exist_ok=True)
METHODS={'grnm':('pando+GRN','pando+GRN'), 'pando':('Our Pando','Our Pando'),
         'external':('External Pando metacells','Pando baseline'),
         'grnboost':('GRNBoost2 only','SCENIC\n(before motif filtering)')}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--methods',nargs=3,choices=METHODS,default=['grnm','external','grnboost'])
parser.add_argument('--out',default='gli3_top10_three_trees')
args=parser.parse_args()
assert len(set(args.methods))==3,'Choose three distinct methods'
top=pd.read_csv(ROOT/'results/model/comparison_top10_four_methods.tsv',sep='\t')
ink,teal,muted,edge='#182b38','#087e76','#667785','#c9d3d8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(14,5.5)); fig.subplots_adjust(left=.035,right=.965,top=.81,bottom=.22,wspace=.18)
fig.text(.04,.955,'GLI3 top-10 candidate targets',fontsize=17,weight='bold',color=ink)
fig.text(.04,.905,'Each panel shows the ten highest-ranked targets for that method',fontsize=10,color=muted)
export=[]
for ax,key in zip(axes,args.methods):
 method,title=METHODS[key]; df=top[top.method.eq(method)].sort_values('rank').copy(); assert df['rank'].tolist()==list(range(1,11)); df.insert(0,'source','GLI3'); export.append(df)
 ax.set_aspect('equal'); ax.set_xlim(-1.48,1.48); ax.set_ylim(-1.35,1.35); ax.axis('off'); ax.set_title(title,fontsize=12,weight='bold',color=ink,pad=8)
 for row in df.itertuples(index=False):
  angle=np.pi/2-(row.rank-1)*2*np.pi/10; x,y=np.cos(angle),np.sin(angle); col=teal if row.hit else edge
  ax.plot([.15*x,.96*x],[.15*y,.96*y],color=col,lw=2 if row.hit else 1,zorder=1)
  ax.scatter(x,y,s=72 if row.hit else 50,facecolor=teal if row.hit else 'white',edgecolor=teal if row.hit else '#a4b3bc',linewidth=1.1,zorder=3)
  align='center' if abs(x)<.1 else ('left' if x>0 else 'right')
  ax.text(1.12*x,1.12*y,f'{row.target}  [{row.rank}]',ha=align,va='center',fontsize=9.5 if row.hit else 8.5,color=teal if row.hit else muted,weight='bold' if row.hit else 'normal')
 ax.scatter(0,0,s=1100,color=ink,zorder=4); ax.text(0,0,'GLI3',ha='center',va='center',color='white',weight='bold',fontsize=11,zorder=5)
 ax.text(0,-1.30,f'{int(df.hit.sum())}/10 candidate hits',ha='center',va='center',fontsize=10,weight='bold',color=ink)
pd.concat(export,ignore_index=True).to_csv(HERE/f'{args.out}_edges.tsv',sep='\t',index=False)
handles=[Line2D([],[],color=teal,lw=2,marker='o',markersize=6,label='Candidate-table hit'),Line2D([],[],color=edge,lw=1,marker='o',markerfacecolor='white',markeredgecolor='#a4b3bc',markersize=6,label='Other ranked target')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.105),ncol=2,frameon=False,fontsize=9)
fig.text(.04,.04,'Ranks use each method’s own scoring rule; lines do not indicate activation or repression. GRNBoost2 is before motif filtering.',fontsize=8,color=muted)
fig.text(.04,.015,'Exploratory comparison: HES4 was added to the target universe after its importance was known.',fontsize=8,color=muted)
for ext in ('pdf','png','svg'): fig.savefig(HERE/f'{args.out}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(fig)
print('methods:',', '.join(args.methods)); print('saved:',args.out)
