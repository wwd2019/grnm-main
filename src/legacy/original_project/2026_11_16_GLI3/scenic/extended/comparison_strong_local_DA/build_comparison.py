from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
root=here.parents[2]
modeled=set((here.parent/'genes.txt').read_text().splitlines())
audit=pd.read_csv(root/'inputs/target_scores.tsv',sep='\t')
common=set(audit.target) & modeled
candidates=pd.read_csv(root/'inputs/PRIORITY_with_strong_local_DA_peak.tsv',sep='\t')
assert candidates.gene_symbol.notna().all() and candidates.gene_symbol.is_unique
positives=set(candidates.gene_symbol) & common
assert len(common)==1793
candidates.assign(in_common=candidates.gene_symbol.isin(common),in_original_universe=candidates.gene_symbol.isin(set(audit.target))).to_csv(here/'candidate_mapping.tsv',sep='\t',index=False)
print(f'Candidates: {len(candidates)}; in original universe: {candidates.gene_symbol.isin(set(audit.target)).sum()}; in common universe: {len(positives)}')
base=pd.read_csv(root/'results/top10_edges.tsv',sep='\t')
base=base[base.network.isin(['Our Pando','Our Pando + GRNM','Pando metacells'])].copy()
assert all(base.target.isin(common))
assert base.groupby('network').size().eq(10).all()
boost=pd.read_csv(here.parent/'run/grnboost2_only_top10.tsv',sep='\t')
boost=boost.rename(columns={'importance':'score'}).assign(network='GRNBoost2 only',stage='Before motif filtering')
top=pd.concat([base[['network','rank','target','score','stage','hit']],boost[['network','rank','target','score','stage','hit']]],ignore_index=True)
assert len(top)==40 and all(top.target.isin(common))
top['hit']=top.target.isin(positives)
order=['Our Pando','Our Pando + GRNM','Pando metacells','GRNBoost2 only']
summary=[]
for method in order:
 x=top[top.network==method].sort_values('rank'); n=len(x); hits=int(x.hit.sum())
 summary.append(dict(method=method,evaluated=n,hits=hits,hit_rate=hits/n,hit_genes=';'.join(x.loc[x.hit,'target'])))
summary=pd.DataFrame(summary)
summary.to_csv(here/'comparison_summary.tsv',sep='\t',index=False)
top.to_csv(here/'comparison_top10.tsv',sep='\t',index=False)
wide=top.assign(display=top.target+top.hit.map({True:' *',False:''})).pivot(index='rank',columns='network',values='display').reindex(columns=order)
wide.to_csv(here/'top10_side_by_side.tsv',sep='\t')
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
labels=['Our Pando','Our Pando + GRNM','External Pando\nmetacells','GRNBoost2 only\n(before motif filtering)']
bars=ax.bar(range(4),summary.hits,color=['#7593a2','#177663','#9aa3aa','#b77d45'],width=.62)
ax.bar_label(bars,labels=[f'{h}/10 ({h*10}%)' for h in summary.hits],padding=4)
ax.set_xticks(range(4),labels); ax.set_ylabel('Candidate hits in top 10'); ax.set_ylim(0,4); ax.set_yticks(range(5)); ax.spines[['top','right']].set_visible(False)
fig.savefig(here/'comparison_top10.png',dpi=200); fig.savefig(here/'comparison_top10.pdf'); plt.close(fig)
print(summary.to_string(index=False)); print(wide.to_string())
