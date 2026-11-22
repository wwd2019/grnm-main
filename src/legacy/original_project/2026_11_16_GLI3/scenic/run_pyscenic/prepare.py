from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
import loompy
here=Path(__file__).resolve().parent.parent
out=here/'run_pyscenic'
genes=np.array((here/'genes.txt').read_text().splitlines())
cells=pd.read_csv(here/'cells.csv')['cell'].to_numpy()
x=mmread(here/'expression.mtx').tocsr().astype('float32')
assert x.shape==(len(genes),len(cells))
loompy.create(str(out/'expression.loom'),x,{'Gene':genes},{'CellID':cells})
annot=pd.read_csv(here/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl',sep='\t',usecols=['gene_name'])
tfs=sorted(set(annot.gene_name.dropna()) & set(genes))
assert 'GLI3' in tfs
(out/'tfs.txt').write_text('\n'.join(tfs)+'\n')
print('expression',x.shape,'TFs',len(tfs),flush=True)
