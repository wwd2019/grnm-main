#!/usr/bin/env python3
"""Build three separate GLI3 circular two-layer trees using the Phase 4 R renderer."""
from pathlib import Path
import os, sys, subprocess, json, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn

OUT=Path(__file__).resolve().parent; MODEL_DIR=OUT.parent; ROOT=MODEL_DIR.parent
PROJECT=ROOT.parent/'grnm-0.3'; R=os.environ.get('RSCRIPT','/home/wwd/miniconda3/envs/seurat4_env_restored/bin/Rscript')
WATCHLIST=['ZIC3','ZNF608','CUX1','HES1','JARID2','MAF','PBX1','SIX6','SOX11','EMX2','TOX','ZIC2','SIX3','SMAD3','SOX4','SALL4','DACH1','RAX','SOX6','ZNF521','PAX6','CREB5','DMRTA1','HES4','HES5']
source=(MODEL_DIR/'train.py').read_text(); prelude=source.split('def examples(frame):')[0]
# Recreate exactly the feature matrix and TF split of the saved model, without fitting.
ctx={'__file__':str(MODEL_DIR/'train.py'),'__name__':'phase4_features'}
exec(compile(prelude,str(MODEL_DIR/'train.py'),'exec'),ctx)
idx=ctx['idx']; names=ctx['names']; features=ctx['features']; universe=ctx['universe']
norm=np.load(MODEL_DIR/'normalization.npz'); mu=norm['mean']; std=norm['std']
model=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
model.load_state_dict(torch.load(MODEL_DIR/'model.pt',map_location='cpu',weights_only=True)); model.eval()
boost=pd.read_csv(ROOT/'scenic/extended/run/adjacencies.tsv',sep='\t'); boost=boost[boost.TF.isin(idx)&boost.target.isin(universe)&boost.TF.ne(boost.target)&boost.importance.gt(0)].copy()
regulators=set(boost.TF)
external=pd.read_csv(ROOT/'inputs/metacells.csv'); external=external[external.tf.isin(idx)&external.target.isin(universe)&external.tf.ne(external.target)&external.padj.between(0,.05)].copy()
external['mag']=external.estimate.abs(); external=external.sort_values(['padj','mag','target'],ascending=[True,False,True]).drop_duplicates(['tf','target'])
root_scores=pd.read_csv(MODEL_DIR/'GLI3_all_scores.tsv',sep='\t').sort_values(['score','target'],ascending=[False,True])
# Root lists are chosen by each method's own ranking, with HES4/HES5 added when available.
def choose_root(df):
 return pd.concat([df.head(80),df[df.target.isin(['HES4','HES5'])]]).drop_duplicates('target').reset_index(drop=True)
def choose_hubs(root,possible):
 chosen=root[root.target.isin(possible)].target.head(6).tolist()
 for g in ['HES4','HES5']:
  if g in set(root.target) and g in possible and g not in chosen: chosen.append(g)
 return chosen
root_grnm=choose_root(root_scores); root_boost=choose_root(boost[boost.TF.eq('GLI3')].sort_values(['importance','target'],ascending=[False,True])); root_ext=choose_root(external[external.tf.eq('GLI3')])
all_saved=[]; summary=[]
for key,root,tfcol,metric,global_df,unsigned in [
 ('grnm',root_grnm,'tf','score',None,True),
 ('pando_metacells',root_ext,'tf','estimate',external,False),
 ('grnboost2',root_boost,'TF','importance',boost,True)]:
 possible=regulators if key!='pando_metacells' else set(external.tf)
 hubs=choose_hubs(root,possible); rootset=set(root.target)|{'GLI3'}
 rows=[]
 for rank,r in enumerate(root.itertuples(index=False),1):
  if key=='grnm': weight=float(r.score)
  elif key=='grnboost2': weight=float(r.importance)
  else: weight=float(r.estimate)
  rows.append({'tf':'GLI3','target':r.target,'estimate':weight,'layer':1,'rank':rank,'source_method':key})
 taken=set()
 for hub in hubs:
  if key=='grnm':
   targets=np.array([idx[g] for g in sorted(universe-rootset) if g!=hub],dtype=int)
   f=features(np.full(len(targets),idx[hub]),targets,False)
   with torch.no_grad(): scores=torch.sigmoid(model(torch.tensor((f-mu)/std))).numpy().ravel()
   cand=pd.DataFrame({'target':np.array(names)[targets],'weight':scores}).sort_values(['weight','target'],ascending=[False,True])
  else:
   group=global_df[(global_df[tfcol]==hub)&~global_df.target.isin(rootset)].copy()
   if key=='grnboost2': cand=group.rename(columns={'importance':'weight'}).sort_values(['weight','target'],ascending=[False,True])
   else: cand=group.rename(columns={'estimate':'weight'}).sort_values(['padj','mag','target'],ascending=[True,False,True])
  cand=cand[~cand.target.isin(taken)].head(5)
  for nr,r in enumerate(cand.itertuples(index=False),1): rows.append({'tf':hub,'target':r.target,'estimate':float(r.weight),'layer':2,'rank':nr,'source_method':key}); taken.add(r.target)
 edges=pd.DataFrame(rows); edges.to_csv(OUT/f'{key}_edges.csv',index=False); all_saved.append(edges)
 # Use the original renderer with two narrowly scoped adaptations: deduplicate edges and
 # show unsigned model scores neutrally rather than as activation.
 renderer=(PROJECT/'draw_grn_tree.R').read_text()
 renderer=renderer.replace('edges <- bind_rows(layer1, layer2, layer3)','edges <- bind_rows(layer1, layer2, layer3) %>% distinct(from, to, .keep_all = TRUE)')
 if unsigned:
  renderer=renderer.replace('if_else(estimate > 0, "positive", "negative")','if_else(!is.na(estimate), "unsigned", "unsigned")')
  renderer=renderer.replace('if_else(downstream_estimate > 0, "positive", "negative")','if_else(!is.na(downstream_estimate), "unsigned", "unsigned")')
  renderer=renderer.replace('values = c("positive" = "#6495ED", "negative" = "#FA8072"),\n      labels = c("positive" = "activation", "negative" = "repression")','values = c("unsigned" = "#8a9aa5"),\n      labels = c("unsigned" = "unsigned score")')
  renderer=renderer.replace('values = c("positive" = "#6495ED", "negative" = "#FA8072"),\n                            labels = c("positive" = "activation", "negative" = "repression")','values = c("unsigned" = "#8a9aa5"),\n                            labels = c("unsigned" = "unsigned score")')
 if not unsigned:
  renderer=renderer.replace('labels = c("positive" = "activation", "negative" = "repression")', 'labels = c("positive" = "positive coefficient", "negative" = "negative coefficient")')
 titles={'grnm':'pando+GRN','pando_metacells':'Pando metacells','grnboost2':'SCENIC (GRNBoost2; before motif filtering)'}
 renderer=renderer.replace('# ── 游离重点基因:', 'p <- p + labs(title = "'+titles[key]+'") + theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))\n\n# ── 游离重点基因:')
 path=OUT/f'{key}_renderer.R'; path.write_text(renderer)
 for ext in ['pdf','png']:
  out=OUT/f'{key}_tree.{ext}'
  cmd=[R,str(path),'--edges',str(OUT/f'{key}_edges.csv'),'--root','GLI3','--output',str(out),'--n1',str(len(root)),'--nhubs',str(len(hubs)),'--n2','5','--highlight',','.join(WATCHLIST),'--label-all']
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); (OUT/f'{key}_{ext}.log').write_text(r.stdout)
  if r.returncode or not out.exists() or out.stat().st_size<1000: raise RuntimeError(f'{key} {ext} failed (see log): {r.stdout[-1500:]}')
 summary.append({'method':key,'root_targets':len(root),'selected_hubs':hubs,'downstream_edges':int((edges.layer==2).sum()),'unsigned':unsigned})
 print(key,'root',len(root),'hubs',hubs,'downstream',int((edges.layer==2).sum()),flush=True)
pd.concat(all_saved,ignore_index=True).to_csv(OUT/'all_displayed_edges.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps(summary,indent=2))

# User-facing name retains the SCENIC stage and states that motif pruning was not passed.
for suffix in ('tree.pdf','tree.png','edges.csv'):
 shutil.copyfile(OUT/f'grnboost2_{suffix}', OUT/f'scenic_pre_motif_{suffix}')
