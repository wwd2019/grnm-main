from pathlib import Path
import ast
import numpy as np
np.object=object; np.bool=bool; np.int=int; np.float=float
import pandas as pd
from scipy.io import mmread
from ctxcore.rnkdb import opendb
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df
OUT=Path(__file__).resolve().parent; run=OUT.parent; source=run.parent; project=source.parent.parent
adj=pd.read_csv(run/'adjacencies.tsv',sep='\t'); genes=(source/'genes.txt').read_text().splitlines()
expr=pd.DataFrame(mmread(source/'expression.mtx').T.toarray(),columns=genes)
mods=[m for m in modules_from_adjacencies(adj,expr) if m.transcription_factor=='GLI3']
assert len(mods)==6
ann=pd.read_csv(source/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl',sep='\t',usecols=['#motif_id','gene_name','orthologous_identity','motif_similarity_qvalue'])
direct=set(ann.loc[ann.gene_name.eq('GLI3') & ann.orthologous_identity.eq(1) & ann.motif_similarity_qvalue.eq(0),'#motif_id'])
dbs=[opendb(fname=str(f),name=f.stem) for f in sorted((source/'resources').glob('*.rankings.feather'))]
print('Running GLI3 cisTarget enrichment at exploratory NES >= 1.9',flush=True)
df=prune2df(dbs,mods,str(source/'resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl'),rank_threshold=5000,nes_threshold=1.9,client_or_address='custom_multiprocessing',num_workers=2,module_chunksize=1)
df.to_pickle(OUT/'motifs_nes_ge_1.9.pkl'); df.to_csv(OUT/'motifs_nes_ge_1.9.csv')
audit=pd.read_csv(project/'inputs/target_scores.tsv',sep='\t'); universe=(set(audit.target)|{'HES4'}) & set(genes)
bench=set(pd.read_csv(project/'inputs/PRIORITY_with_strong_local_DA_peak.tsv',sep='\t').gene_symbol)&universe
print('Common universe',len(universe),'benchmark positives',len(bench),flush=True)
summary=[]
for cutoff,kind in [(3.0,'all_annotations'),(2.0,'all_annotations'),(1.9,'direct_only')]:
 tmp=df[df[('Enrichment','NES')].ge(cutoff)].copy()
 if kind=='direct_only': tmp=tmp.loc[[i for i in tmp.index if i[1] in direct]]
 targets=set()
 for item in tmp[('Enrichment','TargetGenes')]:
  val=ast.literal_eval(item) if isinstance(item,str) else item
  targets.update(g for g,w in val if g!='GLI3')
 # Save all motif-supported targets before restriction to comparison universe.
 pd.DataFrame({'target':sorted(targets),'in_comparison_universe':[g in universe for g in sorted(targets)]}).to_csv(OUT/f'targets_NES_{cutoff}_{kind}.tsv',sep='\t',index=False)
 scores=adj[adj.TF.eq('GLI3')&adj.target.isin(targets)&adj.target.isin(universe)&adj.importance.gt(0)].sort_values(['importance','target'],ascending=[False,True]).drop_duplicates('target').copy()
 scores['hit']=scores.target.isin(bench); scores.insert(0,'rank',range(1,len(scores)+1)); scores.to_csv(OUT/f'ranked_NES_{cutoff}_{kind}.tsv',sep='\t',index=False)
 top=scores.head(10); hits=int(top.hit.sum()); h=top.loc[top.hit,'target'].tolist(); ranks=dict(zip(scores.target,scores['rank']))
 summary.append({'NES_cutoff':cutoff,'annotation_filter':kind,'motif_module_rows':len(tmp),'motif_supported_targets_all':len(targets),'targets_in_universe':len(scores),'top_k':len(top),'hits':hits if len(top) else None,'hit_genes':';'.join(h),'HES4_rank':ranks.get('HES4'),'HES5_rank':ranks.get('HES5')})
result=pd.DataFrame(summary); result.to_csv(OUT/'threshold_comparison.tsv',sep='\t',index=False); print(result.to_string(index=False),flush=True)
