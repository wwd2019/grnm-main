from pathlib import Path
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from scipy.io import mmread
from ctxcore.rnkdb import opendb
from ctxcore.recovery import aucs as calc_aucs
from pyscenic.utils import modules_from_adjacencies
p=Path(__file__).resolve().parent; source=p.parent
adj=pd.read_csv(p/'adjacencies.tsv',sep='\t'); genes=(source/'genes.txt').read_text().splitlines()
expr=pd.DataFrame(mmread(source/'expression.mtx').T.toarray(),columns=genes)
mods=[m for m in modules_from_adjacencies(adj,expr) if m.transcription_factor=='GLI3']
a=pd.read_csv(source/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl',sep='\t',usecols=['#motif_id','gene_name','motif_similarity_qvalue','orthologous_identity'])
a=a[(a.gene_name=='GLI3') & (a.motif_similarity_qvalue<=0.001)]; direct=set(a.loc[(a.motif_similarity_qvalue==0)&(a.orthologous_identity==1),'#motif_id']); allmotifs=set(a['#motif_id'])
out=[]
for path in sorted((source/'resources').glob('*.rankings.feather')):
 db=opendb(fname=str(path),name=path.stem)
 for i,m in enumerate(mods):
  df=db.load(m); values=calc_aucs(df,db.total_genes,np.ones(len(df.columns)),0.05); nes=(values-values.mean())/values.std()
  for motif,auc,n in zip(df.index,values,nes):
   if motif in allmotifs: out.append((path.name,i,len(m),len(df.columns),str(m.context),motif,motif in direct,float(auc),float(n)))
  print(path.name,'module',i,'mapped',len(df.columns),'of',len(m),flush=True)
res=pd.DataFrame(out,columns=['database','module','module_genes','mapped_genes','context','motif','direct','auc','NES']).sort_values('NES',ascending=False)
res.to_csv(p/'gli3_motif_scores_unfiltered.tsv',sep='\t',index=False)
print('max GLI3 annotated NES',res.NES.max(),'max direct NES',res.loc[res.direct,'NES'].max()); print(res.head(8).to_string(index=False))
