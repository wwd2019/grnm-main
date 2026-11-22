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
parent=source.parent
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
df=prune2df(dbs,modules,str(source/"resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"),rank_threshold=5000,client_or_address="custom_multiprocessing",num_workers=2,module_chunksize=1,filter_for_annotation=False)
df.to_pickle(p/"gli3_unannotated_motifs.pkl")
df.to_csv(p/"gli3_unannotated_motifs.csv")
print("Enriched motifs without TF annotation restriction:",len(df),flush=True)
