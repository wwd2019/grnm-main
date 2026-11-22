from pathlib import Path
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from scipy.io import mmread
from pyscenic.utils import modules_from_adjacencies
p=Path(__file__).resolve().parent
adj=pd.concat([pd.read_csv(f,sep='\t') for f in sorted((p/'parts').glob('*.tsv'))[:10]],ignore_index=True)
genes=(p.parent/'genes.txt').read_text().splitlines()
expr=pd.DataFrame(mmread(p.parent/'expression.mtx').T.toarray(),columns=genes)
m=modules_from_adjacencies(adj,expr)
print('module test:',len(m),'GLI3:',sum(x.transcription_factor=='GLI3' for x in m))
