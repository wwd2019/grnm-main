from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent; F=P.parent; S=F.parent
auc=pd.read_csv(F/'aucell_matrix.csv.gz',index_col=0); regulons=pd.read_csv(F/'regulon_targets.tsv.gz',sep='\t'); cells=pd.read_csv(S/'cells.csv'); groups=pd.read_csv(F/'aucell_by_group_mean.csv',index_col=0)
assert auc.index.is_unique and list(auc.index)==list(cells.cell) and auc.notna().all().all()
stats=pd.DataFrame({'regulon':auc.columns,'mean_auc':auc.mean().values,'sd_auc':auc.std().values,'nonzero_fraction':auc.gt(0).mean().values,'n_targets':auc.columns.map(regulons.groupby('regulon').target.nunique()).values})
stats.to_csv(P/'regulon_summary.tsv',sep='\t',index=False)
# Compare variation of group means in units of the within-dataset cell SD, rather than raw AUC scales.
scaling=auc.std().replace(0,np.nan); variability=(groups.std()/scaling).sort_values(ascending=False)
chosen=list(variability.head(10).index)
for g in ('HES5(+)','PAX6(+)'):
 if g in auc.columns and g not in chosen: chosen.append(g)
indices=sorted(groups.index,key=lambda s:(int(s.split('__',1)[0]),s.split('__',1)[1]))
g=groups.loc[indices,chosen]; z=(g-g.mean())/g.std(ddof=0).replace(0,np.nan)
counts=cells.group.value_counts().reindex(indices); table=z.T; table.to_csv(P/'selected_regulon_group_zscores.tsv',sep='\t')
fig,ax=plt.subplots(figsize=(15,6.5),layout='constrained'); im=ax.imshow(table.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
ax.set_yticks(range(len(table)),table.index)
for label in ax.get_yticklabels():
 if label.get_text().split('(')[0] in {'ZIC3','ZNF608','CUX1','HES1','JARID2','MAF','PBX1','SIX6','SOX11','EMX2','TOX','ZIC2','SIX3','SMAD3','SOX4','SALL4','DACH1','RAX','SOX6','ZNF521','PAX6','CREB5','DMRTA1','HES4','HES5'}:
  label.set_color('#C52F25'); label.set_fontweight('bold')
ax.set_xticks(range(len(indices)),[s.replace('__',' / ') for s in indices],rotation=70,ha='right',fontsize=7.5)
ax.tick_params(length=0); ax.set_xlabel('Time point / cell line');ax.set_title('SCENIC regulon activity across sampled groups',loc='left',weight='bold',pad=12)
fig.colorbar(im,ax=ax,label='Group mean AUCell z-score within regulon',shrink=.8)
fig.text(.01,-.035,'Exploratory: group means are standardized within each regulon; cell-type composition may differ. HES5/PAX6 regulons do not imply GLI3 regulation.',fontsize=8,color='#52616d')
for ext in ['png','pdf']: fig.savefig(P/f'aucell_group_heatmap.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
print('Regulons:',len(stats),'cells:',len(auc),'groups:',len(groups),'missing:',auc.isna().sum().sum())
print(stats[stats.regulon.isin(['HES5(+)','PAX6(+)'])].to_string(index=False))
print('Most group-varying:',', '.join(variability.head(10).index))
print('smallest group:',int(counts.min()),'largest:',int(counts.max()))
