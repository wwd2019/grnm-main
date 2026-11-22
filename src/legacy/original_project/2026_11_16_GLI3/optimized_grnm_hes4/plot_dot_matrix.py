#!/usr/bin/env python3
"""Plot top-ten GLI3 targets with all gene labels; retain candidate-hit dots."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

here=Path(__file__).resolve().parent
top=pd.read_csv(here/'comparison_top10_four_methods.tsv',sep='\t')
order=['Our Pando','pando+GRN','External Pando metacells','GRNBoost2 only']
labels=['Our Pando','pando+GRN','Pando baseline','SCENIC\n(before motif filtering)']
assert top.groupby('method').size().loc[order].eq(10).all()
assert all(top.groupby('method')['rank'].apply(lambda x: sorted(x)==list(range(1,11))))
ink,teal,muted='#182b38','#087e76','#667785'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':ink,'axes.labelcolor':ink,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(16,6.8),facecolor='white')
ax=fig.add_axes([.16,.20,.78,.64])
fig.text(.035,.96,'Experimental candidate hits',size=15,weight='bold')
fig.text(.035,.915,'GLI3 targets in each method’s rank order',color=muted,size=10)
for row,method in enumerate(order):
 d=top[top.method.eq(method)].sort_values('rank'); hit=d.hit.astype(bool)
 ax.scatter(d.loc[~hit,'rank'],[row]*int((~hit).sum()),s=65,facecolor='white',edgecolor='#a8b7bf',linewidth=1.15,zorder=3)
 ax.scatter(d.loc[hit,'rank'],[row]*int(hit.sum()),s=72,color=teal,zorder=3)
 for item in d.itertuples(index=False):
  ax.annotate(item.target,(item.rank,row),xytext=(0,9),textcoords='offset points',rotation=35,rotation_mode='anchor',ha='left',va='bottom',size=8,color=teal if item.hit else ink,weight='bold' if item.hit else 'normal')
 ax.text(10.85,row,f'{int(hit.sum())}/10',va='center',weight='bold',color=ink)
ax.set(xlim=(.45,11.4),ylim=(len(order)-.4,-.70),xticks=range(1,11),yticks=range(len(order)),yticklabels=labels,xlabel='Rank')
ax.tick_params(axis='both',length=0,pad=8,labelcolor=ink)
for sp in ax.spines.values():sp.set_visible(False)
fig.legend(handles=[Line2D([],[],marker='o',linestyle='None',color=teal,label='Candidate-table gene'),Line2D([],[],marker='o',linestyle='None',markerfacecolor='white',markeredgecolor='#a8b7bf',label='Other ranked target')],loc='lower left',bbox_to_anchor=(.035,.075),ncol=2,frameon=False,fontsize=9)
fig.text(.035,.035,'Exploratory: HES4 was added after its importance was known. SCENIC uses GRNBoost2 before motif filtering.',size=8,color=muted)
for ext in ('pdf','png','svg'):fig.savefig(here/f'gli3_reviewer_dot_matrix.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(fig)
print('Saved one-panel top-ten dot matrix with all 40 gene labels.')
