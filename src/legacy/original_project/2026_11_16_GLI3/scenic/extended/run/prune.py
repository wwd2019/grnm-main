from pathlib import Path
import ast
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from scipy.io import mmread
from ctxcore.rnkdb import opendb
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df
p=Path(__file__).resolve().parent
source=p.parent
parent=source.parent.parent
adj=pd.read_csv(p/'adjacencies.tsv',sep='\t')
genes=(source/'genes.txt').read_text().splitlines()
cells=pd.read_csv(source/'cells.csv').cell
expr=pd.DataFrame(mmread(source/'expression.mtx').T.toarray(),index=cells,columns=genes)
print('Creating modules using the full network',flush=True)
modules=modules_from_adjacencies(adj,expr)
modules=[m for m in modules if m.transcription_factor=='GLI3']
print('GLI3 modules:',len(modules),flush=True)
if not modules: raise RuntimeError('No GLI3 modules passed the minimum size')
dbpaths=sorted((source/'resources').glob('*.rankings.feather'))
assert len(dbpaths)==2
dbs=[opendb(fname=str(f),name=f.stem) for f in dbpaths]
print('Running cisTarget',flush=True)
df=prune2df(dbs,modules,str(source/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl'),rank_threshold=5000,client_or_address='custom_multiprocessing',num_workers=2,module_chunksize=1)
df.to_pickle(p/'gli3_motifs.pkl')
df.to_csv(p/'gli3_motifs.csv')
if df.empty:
    print('No GLI3-annotated motifs passed enrichment; no regulon can be ranked.', flush=True)
    import runpy
    runpy.run_path(str(p/'summarize.py'), run_name='__main__')
    raise SystemExit(0)
ann=pd.read_csv(source/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl',sep='\t',usecols=['#motif_id','gene_name','orthologous_identity','motif_similarity_qvalue'])
direct=set(ann.loc[(ann.gene_name=='GLI3') & ann.orthologous_identity.eq(1) & ann.motif_similarity_qvalue.eq(0),'#motif_id'])
selected=df.loc[[idx for idx in df.index if idx[0]=='GLI3' and idx[1] in direct]]
targets=set()
for value in selected[('Enrichment','TargetGenes')]:
    entries=ast.literal_eval(value) if isinstance(value,str) else value
    targets.update(gene for gene,weight in entries)
audit=pd.read_csv(parent/'inputs/target_scores.tsv',sep='\t').set_index('target')
scores=adj[(adj.TF=='GLI3') & adj.target.isin(targets) & adj.target.isin(audit.index) & (adj.target!='GLI3')].sort_values(['importance','target'],ascending=[False,True]).drop_duplicates('target').copy()
scores['hit']=scores.target.map(audit.benchmark_positive)
scores.to_csv(p/'gli3_scenic.tsv',sep='\t',index=False)
top=scores.head(10).copy(); top.insert(0,'rank',range(1,len(top)+1)); top.to_csv(p/'scenic_top10.tsv',sep='\t',index=False)
summary=pd.read_csv(parent/'results/summary.tsv',sep='\t')
summary.loc[len(summary)]=['pySCENIC (GRNBoost2 + cisTarget)',len(scores),len(top),int(top.hit.sum()),pd.NA,';'.join(top.loc[top.hit,'target'])]
summary.to_csv(p/'comparison.tsv',sep='\t',index=False)
print(summary.to_string(index=False),flush=True)
