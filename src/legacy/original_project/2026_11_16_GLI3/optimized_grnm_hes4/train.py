from pathlib import Path
import json, random
import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from threadpoolctl import threadpool_limits

P=Path(__file__).resolve().parent; ROOT=P.parent
SEED=42; rng=np.random.default_rng(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)
audit=pd.read_csv(ROOT/'inputs/target_scores.tsv',sep='\t')
names=sorted((set(audit.target) | {'GLI3', 'HES4'}) & set((ROOT/'scenic/extended/genes.txt').read_text().splitlines()))
idx={x:i for i,x in enumerate(names)}; n=len(names); universe=set(names)-{'GLI3'}
e=pd.read_csv(ROOT/'inputs/TF_gene_network_us.csv')
e=e[e.padj_min.between(0,.05)&e.tf.isin(idx)&e.target.isin(universe)&e.tf.ne(e.target)][['tf','target']].drop_duplicates()
# Group-wise split, GLI3 is excluded from every development split.
counts=e[e.tf.ne('GLI3')].groupby('tf').size(); eligible=counts[counts>=10].index.to_numpy(); train_tf, held=train_test_split(eligible,test_size=.4,random_state=SEED); val_tf,test_tf=train_test_split(held,test_size=.5,random_state=SEED)
train_tf=set(train_tf)|set(counts[counts<10].index); val_tf=set(val_tf); test_tf=set(test_tf)
train=e[e.tf.isin(train_tf)].copy(); val=e[e.tf.isin(val_tf)].copy(); test=e[e.tf.isin(test_tf)].copy()
assert not (train_tf & val_tf or train_tf & test_tf or val_tf & test_tf) and 'GLI3' not in train_tf|val_tf|test_tf
s=train.tf.map(idx).to_numpy(); t=train.target.map(idx).to_numpy(); indeg=np.bincount(t,minlength=n); source_in=indeg.copy()
# RNA association is unsupervised, using the same data as the GRNBoost2 baseline.
source=ROOT/'scenic/extended'; allgenes=(source/'genes.txt').read_text().splitlines(); col={g:i for i,g in enumerate(allgenes)}; x=mmread(source/'expression.mtx').tocsr()[[col[g] for g in names]].toarray().astype(np.float32); x-=x.mean(axis=1,keepdims=True); sd=np.sqrt((x*x).sum(axis=1)); x/=np.maximum(sd[:,None],1e-12)
with threadpool_limits(limits=2): corr=x@x.T
boost=pd.read_csv(source/'run/adjacencies.tsv',sep='\t'); boost=boost[boost.TF.isin(idx)&boost.target.isin(idx)]; B=np.zeros((n,n),dtype=np.float32); B[boost.TF.map(idx),boost.target.map(idx)]=boost.importance.to_numpy(dtype=np.float32)
motif=pd.read_csv(ROOT/'inputs/motif2tf_human.tsv',sep='\t'); motif=motif[motif.motif_tf.isin(idx)&motif.tf.isin(idx)]; M=np.zeros((n,n),dtype=np.float32)
for r in motif.itertuples(): M[idx[r.motif_tf],idx[r.tf]]=max(M[idx[r.motif_tf],idx[r.tf]],float(r.prc_similarity))
all_train=set(zip(s,t))
def features(src,tgt,subtract):
 src=np.asarray(src,dtype=int); tgt=np.asarray(tgt,dtype=int)
 correction=np.fromiter(((int(i),int(j)) in all_train for i,j in zip(src,tgt)),dtype=np.float32,count=len(src)) if subtract else 0
 return np.column_stack([np.log1p(np.maximum(indeg[tgt]-correction,0)),np.log1p(source_in[src]),np.log1p(source_in[tgt]),np.log1p(B[src,tgt]),corr[src,tgt],M[src,tgt]]).astype(np.float32)
def examples(frame):
 result=[]
 for tf,g in frame.groupby('tf'):
  positives=set(g.target); pool=np.array(sorted(universe-positives-{tf})); neg=rng.choice(pool,size=min(len(pool),4*len(positives)),replace=False)
  result.extend((idx[tf],idx[t],1) for t in sorted(positives)); result.extend((idx[tf],idx[t],0) for t in neg)
 return np.array(result,dtype=np.int64)
tr=examples(train); mu=features(tr[:,0],tr[:,1],True).mean(0); sd=features(tr[:,0],tr[:,1],True).std(0); sd[sd<1e-6]=1
X=torch.tensor((features(tr[:,0],tr[:,1],True)-mu)/sd); Y=torch.tensor(tr[:,2],dtype=torch.float32)
def full_group(frame):
 groups=[]
 for tf,g in frame.groupby('tf'):
  si=idx[tf]; targets=np.array([i for i in range(n) if i!=si and names[i]!='GLI3']); label=np.isin(targets,g.target.map(idx)).astype(int); groups.append((tf,targets,label,features(np.full(len(targets),si),targets,False)))
 return groups
vg=full_group(val); tg=full_group(test)
def stats(groups, predict):
 rows=[]
 for tf,targets,y,feat in groups:
  score=predict(feat); order=np.lexsort((np.array(names)[targets],-score)); rows.append((tf,int(y.sum()),average_precision_score(y,score),int(y[order[:10]].sum()),roc_auc_score(y,score)))
 return pd.DataFrame(rows,columns=['tf','positives','AP','hits10','AUROC'])
def predict_net(model):
 def f(a):
  with torch.no_grad(): return torch.sigmoid(model(torch.tensor((a-mu)/sd))).numpy().ravel()
 return f

def fit_model(width,drop,weight_decay):
 model=nn.Sequential(nn.Linear(6,width),nn.ReLU(),nn.Dropout(drop),nn.Linear(width,width//2),nn.ReLU(),nn.Linear(width//2,1))
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=weight_decay); lossfn=nn.BCEWithLogitsLoss(); gen=torch.Generator().manual_seed(SEED)
 loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y),batch_size=512,shuffle=True,generator=gen)
 best=-1; state=None; best_ep=0
 for ep in range(1,61):
  model.train()
  for xb,yb in loader:
   optimizer.zero_grad(); loss=lossfn(model(xb).squeeze(1),yb); loss.backward(); optimizer.step()
  if ep%5==0:
   model.eval(); value=stats(vg,predict_net(model)).AP.mean()
   if value>best: best=value; best_ep=ep; state={k:v.detach().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state); model.eval(); return model,best,best_ep
baselines={'target_degree':lambda f:f[:,0], 'GRNBoost2':lambda f:f[:,3]}
rows=[]; models={}
for name,f in baselines.items():
 for split,group in [('validation',vg),('test',tg)]:
  scores=stats(group,f); scores.assign(method=name,split=split).to_csv(P/f'{name}_{split}_per_tf.tsv',sep='\t',index=False); rows.append((name,split,len(scores),scores.AP.mean(),scores.hits10.mean(),scores.AUROC.mean()))
for width,drop,wd in [(32,.1,.01),(64,.2,.01),(64,.4,.1)]:
 label=f'GRNM_{width}_drop{drop}'; torch.manual_seed(SEED); model,best,epoch=fit_model(width,drop,wd); models[label]=(model,best,epoch,width,drop,wd)
 for split,group in [('validation',vg),('test',tg)]:
  z=stats(group,predict_net(model)); z.assign(method=label,split=split).to_csv(P/f'{label}_{split}_per_tf.tsv',sep='\t',index=False); rows.append((label,split,len(z),z.AP.mean(),z.hits10.mean(),z.AUROC.mean()))
 print(label,'val AP',best,'epoch',epoch,flush=True)
selected=max(models,key=lambda k:models[k][1]); model,best,epoch,width,drop,wd=models[selected]
metrics=pd.DataFrame(rows,columns=['method','split','TFs','mean_AP','mean_hits_top10','mean_AUROC']); metrics.to_csv(P/'validation_metrics.tsv',sep='\t',index=False)
# Candidate table enters only after the architecture and stopping epoch are fixed.
si=idx['GLI3']; targets=np.array([i for i in range(n) if i!=si and names[i]!='GLI3']); scores=predict_net(model)(features(np.full(len(targets),si),targets,False)); out=pd.DataFrame({'target':np.array(names)[targets],'score':scores}); out=out.sort_values(['score','target'],ascending=[False,True]); out.to_csv(P/'GLI3_all_scores.tsv',sep='\t',index=False)
benchmark=pd.read_csv(ROOT/'inputs/PRIORITY_with_strong_local_DA_peak.tsv',sep='\t'); positives=set(benchmark.gene_symbol)&universe
seed=set(e.loc[e.tf.eq('GLI3'),'target']); out['hit']=out.target.isin(positives); out['pando_seed']=out.target.isin(seed)
for label,df in [('all_targets',out),('pando_seed_only',out[out.pando_seed])]:
 top=df.head(10).copy(); top.insert(0,'rank',range(1,len(top)+1)); top.to_csv(P/f'GLI3_{label}_top10.tsv',sep='\t',index=False); print(label,'hits',int(top.hit.sum()),'genes',','.join(top.loc[top.hit,'target']),flush=True)
np.savez(P/'normalization.npz',mean=mu,std=sd); torch.save(model.state_dict(),P/'model.pt')
(P/'manifest.json').write_text(json.dumps({'seed':SEED,'common_genes':len(universe),'benchmark_positive':len(positives),'train_TFs':len(train_tf),'validation_TFs':len(val_tf),'test_TFs':len(test_tf),'train_edges':len(train),'selected_model':selected,'selected_epoch':epoch,'selection':'mean per-TF AP on validation TFs; architecture & epoch chosen without GLI3 candidate table','negative_sampling':'four random unlisted pairs per known Pando edge, not verified negatives','train_features':'target degree excluding queried edge; RNA correlation; GRNBoost2 importance; motif similarity; TF and target in-degree','heldout_features':'all outgoing edges of validation TFs, test TFs and GLI3 excluded before feature construction','seed_rank_note':'Pando seed only reranking, full scores have no significance or motif-pruning cutoff'},indent=2))
print(metrics.to_string(index=False)); print('selected:',selected,flush=True)
