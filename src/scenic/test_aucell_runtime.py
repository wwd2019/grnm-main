import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from ctxcore.genesig import GeneSignature
from pyscenic.aucell import aucell
rng=np.random.default_rng(42)
a=pd.DataFrame(rng.poisson(1,size=(30,100)),index=[f'c{i}' for i in range(30)],columns=[f'g{i}' for i in range(100)])
sig=[GeneSignature(name='demo',gene2weight={f'g{i}':1 for i in range(15)})]
kw=dict(auc_threshold=.05,noweights=True,normalize=False,seed=42,num_workers=2)
full=aucell(a,sig,**kw); chunks=pd.concat([aucell(a.iloc[:10],sig,**kw),aucell(a.iloc[10:],sig,**kw)])
assert np.allclose(full,chunks)
print('AUCell multiprocessing and chunk-equivalence passed:',full.shape)
