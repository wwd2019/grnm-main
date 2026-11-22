#!/usr/bin/env python3
"""Rank planarian target candidates for one SMED regulator using the saved edge model."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT/'src/legacy/original_project/grnm-0.3/scripts'
sys.path.insert(0, str(SCRIPTS))
from phase3_edge_predict import EdgeClassifier

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', default='SMED30006440', help='Planarian SMED regulator ID (default: Hes1-associated SMED30006440)')
parser.add_argument('--top', type=int, default=10)
parser.add_argument('--output', type=Path, default=ROOT/'results/generated/planarian_top_targets.tsv')
args = parser.parse_args()
if args.top < 1:
    parser.error('--top must be positive')
npz = np.load(ROOT/'results/planarian/phase2_output/all_gene_z.npz', allow_pickle=False)
z = np.asarray(npz['z'], dtype=np.float32)
genes = npz['gene_ids'].astype(str)
symbols = npz['symbols'].astype(str)
indices = np.flatnonzero(genes == args.source)
if len(indices) != 1:
    raise SystemExit(f'Source {args.source!r} was not found uniquely in the saved embedding.')
checkpoint = torch.load(ROOT/'results/planarian/phase3_output/edge_predictor.pt', map_location='cpu', weights_only=True)
model = EdgeClassifier(z_dim=int(checkpoint['z_dim']))
model.load_state_dict(checkpoint['model_state']); model.eval(); torch.set_num_threads(2)
embedding = torch.from_numpy(z)
source_z = embedding[int(indices[0])]
classes = np.array([-3,-2,-1,0,1,2,3], dtype=int)
nonzero_idx = np.array([0,1,2,4,5,6], dtype=int)
weights=[]; selected_classes=[]; selected_prob=[]; fallback=[]
with torch.no_grad():
    for start in range(0,len(genes),4096):
        target_z = embedding[start:start+4096]
        scores = torch.softmax(model(torch.cat([source_z.expand(len(target_z),-1), target_z],dim=1)),dim=1).numpy()
        indices_ = scores.argmax(axis=1)
        is_zero = indices_ == 3
        if is_zero.any():
            indices_[is_zero] = nonzero_idx[scores[is_zero][:,nonzero_idx].argmax(axis=1)]
        cls = classes[indices_]; prob = scores[np.arange(len(indices_)),indices_]
        weights.extend((cls*prob).tolist());selected_classes.extend(cls.tolist());selected_prob.extend(prob.tolist());fallback.extend(is_zero.tolist())
result=pd.DataFrame({'source':args.source,'target':genes,'target_symbol':symbols,'predicted_class':selected_classes,'class_probability':selected_prob,'signed_score':weights,'zero_class_fallback':fallback})
result=result[result.target.ne(args.source)].copy(); result['absolute_score']=result.signed_score.abs()
result=result.sort_values(['absolute_score','target'],ascending=[False,True]).head(args.top).reset_index(drop=True)
result.insert(0,'rank',np.arange(1,len(result)+1))
args.output.parent.mkdir(parents=True,exist_ok=True)
result.to_csv(args.output,sep='\t',index=False)
print(result.to_string(index=False));print('Saved:',args.output)
