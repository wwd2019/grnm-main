from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent; root=P.parent
universe=(set(pd.read_csv(root/'inputs/target_scores.tsv',sep='\t').target)|{'HES4'}) & set((root/'scenic/extended/genes.txt').read_text().splitlines()); assert len(universe)==1794
candidate=set(pd.read_csv(root/'inputs/PRIORITY_with_strong_local_DA_peak.tsv',sep='\t').gene_symbol)&universe
assert {'HES4','HES5'} <= candidate
raw=pd.read_csv(root/'inputs/TF_gene_network_us.csv'); motif=pd.read_csv(root/'inputs/motif2tf_human.tsv',sep='\t'); pred=pd.read_csv(root/'frozen/human_GLI3_predictions.csv')
def pando(df,p):
 a=df[df.tf.eq('GLI3') & df.target.isin(universe) & df[p].between(0,.05)].copy(); a['mag']=a.estimate.abs(); return a.sort_values([p,'mag','target'],ascending=[True,False,True]).drop_duplicates('target').rename(columns={p:'score'})
first=pando(raw,'padj_min'); ext=pando(pd.read_csv(root/'inputs/metacells.csv'),'padj')
inputs=set(raw.loc[raw.tf.eq('GLI3'),'target'])|set(motif.loc[motif.motif_tf.eq('GLI3'),'tf'])
pred=pred[pred.target.isin(universe)].copy(); seed=pred.target.isin(first.target); novel=~pred.target.isin(inputs)&pred.pred_class.ne(0)&pred.weight.abs().ge(1.3)
old=pred[seed|novel].assign(score=lambda a:a.prob.where(a.pred_class.ne(0),0)).sort_values(['score','target'],ascending=[False,True])
b=pd.read_csv(root/'scenic/extended/run/adjacencies.tsv',sep='\t'); b=b[b.TF.eq('GLI3') & b.target.isin(universe)].rename(columns={'importance':'score'}).sort_values(['score','target'],ascending=[False,True]).drop_duplicates('target')
new=pd.read_csv(P/'GLI3_all_scores.tsv',sep='\t').sort_values(['score','target'],ascending=[False,True]); assert set(new.target)==universe
methods={'Our Pando':first,'Original GRNM':old,'External Pando metacells':ext,'GRNBoost2 only':b,'pando+GRN':new}
rows=[]; allrows=[]
for label,df in methods.items():
 df=df.reset_index(drop=True); df['rank']=df.index+1; df['hit']=df.target.isin(candidate); top=df.head(10).copy(); h=df[df.target.isin(['HES4','HES5'])]; ranks=dict(zip(h.target,h['rank'])); rows.append({'method':label,'edges':len(df),'top10_hits':int(top.hit.sum()),'hit_genes':';'.join(top.loc[top.hit,'target']),'HES4_rank':ranks.get('HES4'),'HES5_rank':ranks.get('HES5')}); top.insert(0,'method',label); allrows.append(top[['method','rank','target','score','hit']])
pd.DataFrame(rows).to_csv(P/'comparison_hes4_included.tsv',sep='\t',index=False); pd.concat(allrows).to_csv(P/'comparison_top10.tsv',sep='\t',index=False)
pd.DataFrame(rows).query("method != 'Original GRNM'").to_csv(P/'comparison_hes4_included_four_methods.tsv',sep='\t',index=False)
pd.concat(allrows).query("method != 'Original GRNM'").to_csv(P/'comparison_top10_four_methods.tsv',sep='\t',index=False)
print('universe',len(universe),'candidates',len(candidate)); print(pd.DataFrame(rows).to_string(index=False))
