from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
import new_human_results as pipeline


def main():
    out = HERE / "grnm_rerun"
    out.mkdir(exist_ok=True)
    pipeline.P3_DIR = str(out)
    pipeline.PNG_DIR = str(out)
    torch.set_num_threads(2)
    meta, _ = pipeline.load_human_inputs()
    with np.load(PROJECT / "outputs/phase2_human_output/human_gene_z_phase2.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["gene_names"], meta["genes"]):
            raise ValueError("Cached embedding gene order does not match the training inputs")
        scores, _ = pipeline.run_phase3(meta, saved["z"], epochs=200, seed=42, device="cpu")
    if len(scores) != len(meta["genes"]) - 1 or scores.target.duplicated().any():
        raise ValueError("Incomplete GLI3 predictions")


if __name__ == "__main__":
    main()
