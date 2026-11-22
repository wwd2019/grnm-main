from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

here = Path(__file__).resolve().parent
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)
project = Path(sys.argv[1])
sys.path.insert(0, str(project / "scripts"))
import new_human_results as pipeline

source = here / "inputs/TF_gene_network_us.csv"
glm = pd.read_csv(source).rename(columns={"padj_min": "padj"})
glm = glm.dropna(subset=["tf", "target", "estimate", "padj"])
adapted = out / "TF_gene_network_us.csv"
glm.to_csv(adapted, index=False)
pipeline.GLM_CSV = str(adapted)
pipeline.MOTIF_TSV = str(here / "inputs/motif2tf_human.tsv")
pipeline.P3_DIR = str(out)
pipeline.PNG_DIR = str(out)
classifier = pipeline.EdgeClassifier
pipeline.EdgeClassifier = lambda dim: classifier(dim, hidden=192, n_blocks=3, dropout=0.08)
torch.set_num_threads(2)

with threadpool_limits(limits=2):
    meta, blocks = pipeline.load_human_inputs()
    features = np.column_stack(list(blocks.values()))
    z = PCA(n_components=64, random_state=42).fit_transform(StandardScaler().fit_transform(features))
    z = StandardScaler().fit_transform(z).astype(np.float32)
    prediction, manifest = pipeline.run_phase3(
        meta, z, epochs=90, batch=384, lr=0.0005, warmup=8,
        no_edge_ratio=0.75, seed=42, device="cpu", min_abs_weight=1.8,
    )


