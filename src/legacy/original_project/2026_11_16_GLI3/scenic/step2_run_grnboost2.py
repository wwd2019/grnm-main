from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
from arboreto.core import infer_partial_network, SGBM_KWARGS

here = Path(__file__).resolve().parent
genes = (here / "genes.txt").read_text().splitlines()
expr = mmread(here / "expression.mtx").T.tocsc()
assert expr.shape[1] == len(genes)
regulators = set(pd.read_csv(here.parent / "inputs/TF_gene_network_us.csv").tf)
tf_idx = [i for i,g in enumerate(genes) if g in regulators]
tf_names = [genes[i] for i in tf_idx]
assert "GLI3" in tf_names and len(tf_names)>1
tf_matrix = expr[:,tf_idx].toarray().astype(np.float32)
universe = set(pd.read_csv(here.parent / "inputs/target_scores.tsv", sep="\t").target)
parts = []
for i,target in enumerate(genes):
    if target == "GLI3" or target not in universe:
        continue
    y = expr[:,i].toarray().ravel()
    if np.ptp(y) == 0:
        continue
    links = infer_partial_network("GBM", SGBM_KWARGS, tf_matrix, tf_names, target, y, seed=42)
    parts.append(links[links.TF.eq("GLI3")])
    if i % 50 == 0:
        print(f"{i}/{len(genes)} targets",flush=True)
scores = pd.concat(parts,ignore_index=True).sort_values(["importance","target"],ascending=[False,True])
scores.to_csv(here/"gli3_grnboost2.tsv",sep="\t",index=False)

