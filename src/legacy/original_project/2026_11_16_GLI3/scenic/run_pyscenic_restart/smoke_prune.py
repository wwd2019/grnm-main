from pathlib import Path
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
from ctxcore.genesig import Regulon
from ctxcore.rnkdb import opendb
from pyscenic.prune import prune2df
p=Path(__file__).resolve().parent.parent
genes=(p/'genes.txt').read_text().splitlines()[:100]
mod=Regulon(name='GLI3(+)',gene2weight={g:1.0 for g in genes}, transcription_factor='GLI3',gene2occurrence={},context=frozenset(['activating']))
db=next((p/'resources').glob('*.rankings.feather'))
print('Testing cisTarget database and runtime',flush=True)
r=prune2df([opendb(fname=str(db),name=db.stem)],[mod],str(p/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl'),rank_threshold=5000,client_or_address='custom_multiprocessing',num_workers=1,module_chunksize=1)
print('cisTarget test finished; enriched motifs:',len(r),flush=True)
