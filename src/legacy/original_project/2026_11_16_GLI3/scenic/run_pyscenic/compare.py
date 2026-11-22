from pathlib import Path
import ast
import pandas as pd
p=Path(__file__).resolve().parent
parent=p.parent.parent
motifs=pd.read_csv(p/'motifs.csv',header=[0,1],index_col=[0,1])
annot=pd.read_csv(p.parent/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl',sep='\t')
direct=set(annot.loc[(annot.gene_name=='GLI3') & annot.orthologous_identity.eq(1) & annot.motif_similarity_qvalue.eq(0),'#motif_id'])
selected=motifs.loc[[idx for idx in motifs.index if idx[0]=='GLI3' and idx[1] in direct]]
targets=set()
for entry in selected[('Enrichment','TargetGenes')]:
    targets.update(gene for gene,weight in ast.literal_eval(entry))
adj=pd.read_csv(p/'adjacencies.tsv',sep='\t')
audit=pd.read_csv(parent/'inputs/target_scores.tsv',sep='\t').set_index('target')
scores=adj[(adj.TF=='GLI3') & adj.target.isin(targets) & adj.target.isin(audit.index) & (adj.target!='GLI3')].sort_values(['importance','target'],ascending=[False,True]).drop_duplicates('target')
scores['hit']=scores.target.map(audit.benchmark_positive)
scores.to_csv(p/'gli3_scenic.tsv',sep='\t',index=False)
top=scores.head(10).copy(); top.insert(0,'rank',range(1,len(top)+1)); top.to_csv(p/'scenic_top10.tsv',sep='\t',index=False)
summary=pd.read_csv(parent/'results/summary.tsv',sep='\t')
summary.loc[len(summary)]=['pySCENIC (GRNBoost2 + cisTarget)',len(scores),len(top),int(top.hit.sum()),pd.NA,';'.join(top.loc[top.hit,'target'])]
summary.to_csv(p/'comparison.tsv',sep='\t',index=False)
print(summary.to_string(index=False))
