from pathlib import Path
import pickle,json,time
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from scipy.io import mmread
from pyscenic.aucell import aucell,derive_auc_threshold
ROOT=Path(__file__).resolve().parents[2]; P=ROOT/'results/scenic/full'; SOURCE=ROOT/'dataset/scenic'
with open(P/'regulons.pkl','rb') as f: regs=pickle.load(f)
assert regs
rnames=[r.name for r in regs]; assert len(rnames)==len(set(rnames))
raw_genes=(SOURCE/'rna_all_genes.txt').read_text().splitlines(); cells=(SOURCE/'rna_cells.txt').read_text().splitlines(); x=mmread(SOURCE/'rna_all_genes.mtx').tocsr()
assert x.shape==(len(raw_genes),len(cells))==(33538,9327) and len(set(raw_genes))==len(raw_genes) and len(set(cells))==len(cells)
print('AUCell input:',x.shape,'regulons',len(regs),flush=True)
# Cells are chunked to bound memory. Each chunk uses the same gene shuffle seed,
# exactly preserving the tie-breaking gene permutation used for the full matrix.
step=250; partdir=P/'aucell_parts'; partdir.mkdir(exist_ok=True)
detected=np.asarray((x>0).sum(axis=0)).ravel(); print('Detected genes per cell:',np.quantile(detected,[.01,.05,.5]).tolist(),flush=True)
for start in range(0,len(cells),step):
 out=partdir/f'aucell_{start:05d}.pkl'
 if out.exists():
  chunk=pd.read_pickle(out)
  if list(chunk.index)==cells[start:start+step] and list(chunk.columns)==rnames:
   print('RESUME',start,flush=True); continue
  raise RuntimeError(f'Invalid checkpoint {out}')
 t=time.monotonic(); end=min(start+step,len(cells)); a=pd.DataFrame(x[:,start:end].T.toarray(),index=cells[start:end],columns=raw_genes)
 result=aucell(a,regs,auc_threshold=.05,noweights=True,normalize=False,seed=42,num_workers=4)
 result=result.reindex(index=cells[start:end],columns=rnames)
 if result.isna().any().any() or not np.isfinite(result.to_numpy()).all(): raise RuntimeError('AUCell has missing or invalid values')
 temp=out.with_suffix('.tmp'); result.to_pickle(temp); temp.replace(out)
 print(f'cells {end}/{len(cells)}; {time.monotonic()-t:.1f}s',flush=True)
parts=[pd.read_pickle(partdir/f'aucell_{i:05d}.pkl') for i in range(0,len(cells),step)]
a=pd.concat(parts); assert list(a.index)==cells and list(a.columns)==rnames
zero_cols=a.columns[(a==0).all(axis=0)].tolist()
if zero_cols: raise RuntimeError(f'AUCell returned all-zero regulons; check worker failure: {zero_cols}')
# gzip CSV has cells as rows and regulons as columns.
a.index.name='cell'; a.to_csv(P/'aucell_matrix.csv.gz',compression='gzip')
metadata=pd.read_csv(SOURCE/'cells.csv'); metadata.index=metadata.cell
mean=a.groupby(metadata.loc[a.index,'group']).mean(); mean.index.name='group'; mean.to_csv(P/'aucell_by_group_mean.csv')
(P/'aucell_summary.json').write_text(json.dumps({'cells':len(cells),'ranking_genes':len(raw_genes),'regulons':len(regs),'matrix_shape':list(a.shape),'AUCell_auc_threshold':.05,'seed':42,'weighted':False,'normalize':False,'GLI3_regulons':[r.name for r in regs if r.transcription_factor=='GLI3'],'detected_genes_quantiles':dict(zip(['1%','5%','50%'],map(float,np.quantile(detected,[.01,.05,.5]))))},indent=2))
print('AUCell complete:',a.shape,flush=True)
