from pathlib import Path
import pickle, json, time
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from ctxcore.rnkdb import opendb
from pyscenic.prune import prune2df
from pyscenic.transform import df2regulons
ROOT=Path(__file__).resolve().parents[2]; P=ROOT/'results/scenic/full'; SOURCE=ROOT/'dataset'
with open(P/'modules.pkl','rb') as f: mods=pickle.load(f)
dbpaths=sorted((SOURCE/'resources').glob('*.rankings.feather')); assert len(dbpaths)==2
dbs=[opendb(fname=str(f),name=f.stem) for f in dbpaths]
ann=str(SOURCE/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl'); step=120
for start in range(0,len(mods),step):
 out=P/'parts'/f'motifs_{start:05d}.pkl'
 if out.exists():
  with open(out,'rb') as f: pickle.load(f)
  print('RESUME',start,flush=True); continue
 t=time.monotonic(); batch=mods[start:start+step]
 # Default pySCENIC thresholds. Rank threshold CLI default is 5000.
 df=prune2df(dbs,batch,ann,rank_threshold=5000,auc_threshold=.05,nes_threshold=3.0,client_or_address='custom_multiprocessing',num_workers=4,module_chunksize=100)
 tmp=out.with_suffix('.tmp')
 with open(tmp,'wb') as f: pickle.dump(df,f)
 tmp.replace(out)
 print(f'modules {min(start+step,len(mods))}/{len(mods)}; {len(df)} motif rows; {time.monotonic()-t:.1f}s',flush=True)
files=[P/'parts'/f'motifs_{start:05d}.pkl' for start in range(0,len(mods),step)]
parts=[]
for f in files:
 with open(f,'rb') as h: parts.append(pickle.load(h))
nonempty=[p for p in parts if not p.empty]
if not nonempty: raise RuntimeError('No motif-enriched regulons found; AUCell cannot run')
df=pd.concat(nonempty); df.to_pickle(P/'motif_enrichment.pkl'); df.to_csv(P/'motif_enrichment.csv.gz',compression='gzip')
regs=df2regulons(df)
if not regs: raise RuntimeError('No regulons recovered')
with open(P/'regulons.pkl','wb') as f: pickle.dump(regs,f)
rows=[]
for reg in regs:
 for gene,weight in reg.gene2weight.items(): rows.append({'regulon':reg.name,'TF':reg.transcription_factor,'target':gene,'weight':weight})
pd.DataFrame(rows).to_csv(P/'regulon_targets.tsv.gz',sep='\t',index=False,compression='gzip')
(P/'pruning_summary.json').write_text(json.dumps({'modules':len(mods),'motif_rows':len(df),'regulons':len(regs),'regulon_TFs':len(set(r.transcription_factor for r in regs)),'GLI3_regulons':[r.name for r in regs if r.transcription_factor=='GLI3'],'NES_threshold':3.0,'rank_threshold':5000,'AUC_threshold':.05},indent=2))
print('Pruning complete',len(df),'motif rows',len(regs),'regulons',flush=True)
