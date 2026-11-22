from pathlib import Path
import os, sys, time
import numpy as np
import pandas as pd
from scipy.io import mmread
from joblib import Parallel, delayed
from arboreto.core import infer_partial_network, SGBM_KWARGS
root=Path(__file__).resolve().parents[2]
p=root/'results/scenic'
source=root/'dataset/scenic'
workers=int(os.getenv('SCENIC_WORKERS','8'))
limit=int(os.getenv('SCENIC_LIMIT','0'))
genes=(source/'genes.txt').read_text().splitlines()
tfs=set((source/'tfs.txt').read_text().splitlines())
x=mmread(source/'expression.mtx').T.tocsc().toarray().astype(np.float32)
tf_indices=[i for i,g in enumerate(genes) if g in tfs]
tf_names=[genes[i] for i in tf_indices]
tf_matrix=x[:,tf_indices]
targets=list(range(len(genes)))[:limit or None]
parts=p/'grnboost_parts'; parts.mkdir(exist_ok=True)
def fit(i):
    if np.ptp(x[:,i])==0:
        return pd.DataFrame(columns=['TF','target','importance'])
    return infer_partial_network('GBM',SGBM_KWARGS,tf_matrix,tf_names,genes[i],x[:,i],seed=42)
for start in range(0,len(targets),workers):
    ids=targets[start:start+workers]; output=parts/('%04d.tsv'%start)
    if output.exists(): continue
    t=time.monotonic()
    result=Parallel(n_jobs=workers,prefer='processes')(delayed(fit)(i) for i in ids)
    if any(v.empty and np.ptp(x[:,i])!=0 for i,v in zip(ids,result)):
        raise RuntimeError('Nonconstant target has no inferred links')
    pd.concat(result).to_csv(output,sep='\t',index=False)
    print(f'{min(start+workers,len(targets))}/{len(targets)} targets; batch {time.monotonic()-t:.1f}s',flush=True)
if not limit:
    files=[parts/('%04d.tsv'%i) for i in range(0,len(targets),workers)]
    if not all(f.exists() for f in files): raise RuntimeError('Missing result batches')
    pd.concat((pd.read_csv(f,sep='\t') for f in files),ignore_index=True).to_csv(p/'adjacencies.tsv',sep='\t',index=False)
    print('Complete',flush=True)
