from pathlib import Path
import pickle
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from scipy.io import mmread
from pyscenic.utils import modules_from_adjacencies
ROOT=Path(__file__).resolve().parents[2]; P=ROOT/'results/scenic/full'; SOURCE=ROOT/'dataset/scenic'
adj=pd.read_csv(ROOT/'results/scenic/adjacencies.tsv',sep='\t'); genes=(SOURCE/'genes.txt').read_text().splitlines(); cells=pd.read_csv(SOURCE/'cells.csv').cell
expr=pd.DataFrame(mmread(SOURCE/'expression.mtx').T.toarray(),columns=genes,index=cells)
assert expr.shape==(9327,6798)
mods=modules_from_adjacencies(adj,expr)
mods.sort(key=lambda m:(m.transcription_factor,m.name,sorted(m.context)))
with open(P/'modules.pkl','wb') as f: pickle.dump(mods,f)
pd.DataFrame([{'module_id':i,'TF':m.transcription_factor,'name':m.name,'genes':len(m),'context':';'.join(sorted(m.context))} for i,m in enumerate(mods)]).to_csv(P/'module_index.tsv',sep='\t',index=False)
print('Modules:',len(mods),'TFs:',len(set(m.transcription_factor for m in mods)),flush=True)
