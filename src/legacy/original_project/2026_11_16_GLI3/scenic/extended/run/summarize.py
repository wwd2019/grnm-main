from pathlib import Path
import pandas as pd
p=Path(__file__).resolve().parent
parent=p.parent.parent.parent
audit=pd.read_csv(parent/'inputs/target_scores.tsv',sep='\t').set_index('target')
adj=pd.read_csv(p/'adjacencies.tsv',sep='\t')
gli=adj[(adj.TF=='GLI3') & adj.target.isin(audit.index) & adj.target.ne('GLI3')].sort_values(['importance','target'],ascending=[False,True]).drop_duplicates('target').copy()
gli['hit']=gli.target.map(audit.benchmark_positive)
gli.head(10).assign(rank=range(1,min(10,len(gli))+1)).to_csv(p/'grnboost2_only_top10.tsv',sep='\t',index=False)
summary=pd.read_csv(parent/'results/summary.tsv',sep='\t')
summary.loc[len(summary)]=['pySCENIC: GLI3 regulon not detected',0,0,pd.NA,pd.NA,'']
summary.loc[len(summary)]=['GRNBoost2 only (without motif pruning)',len(gli),min(10,len(gli)),int(gli.head(10).hit.sum()),pd.NA,';'.join(gli.head(10).loc[gli.head(10).hit,'target'])]
summary.to_csv(p/'comparison.tsv',sep='\t',index=False)
print(summary.to_string(index=False)); print(gli.head(10).to_string(index=False))
