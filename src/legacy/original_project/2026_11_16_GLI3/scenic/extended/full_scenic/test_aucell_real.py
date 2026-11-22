from pathlib import Path
import pickle
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from scipy.io import mmread
from pyscenic.transform import df2regulons
from pyscenic.aucell import aucell
p=Path(__file__).resolve().parent
with open(p/'parts/motifs_00000.pkl','rb') as f: d=pickle.load(f)
r=df2regulons(d); assert len(r)>0
with open(p/'test_real_regulons.pkl','wb') as f: pickle.dump(r[:3],f)
genes=(p/'rna_all_genes.txt').read_text().splitlines(); cells=(p/'rna_cells.txt').read_text().splitlines()
x=mmread(p/'rna_all_genes.mtx').tocsr(); a=pd.DataFrame(x[:,:5].T.toarray(),columns=genes,index=cells[:5])
s=aucell(a,r[:3],auc_threshold=.05,noweights=True,normalize=False,seed=42,num_workers=2)
assert s.shape==(5,3) and np.isfinite(s.to_numpy()).all(); s.to_csv(p/'test_real_aucell.csv')
print('Real regulon AUCell smoke passed:',s.shape,flush=True)
