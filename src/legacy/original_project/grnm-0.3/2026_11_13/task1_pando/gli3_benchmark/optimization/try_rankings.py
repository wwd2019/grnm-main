from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[3]


def main():
    audit = pd.read_csv(OUT / "target_scores.tsv", sep="\t").set_index("target")
    baseline = audit["GRNM"]
    old = pd.read_csv(PROJECT / "outputs/phase3_human_output/human_GLI3_predictions.csv").set_index("target")
    old = old.reindex(audit.index)
    glm = pd.read_csv(PROJECT / "example_data/glm_network_coefs_human.csv")
    glm = glm[glm.padj.le(0.05)]
    direct = glm[glm.tf.eq("GLI3")].groupby("target").estimate.apply(lambda x: x.abs().max())
    with np.load(PROJECT / "outputs/phase2_human_output/human_gene_z_phase2.npz") as data:
        names = data["gene_names"]
        z1 = data["z_phase1"]
        z2 = data["z"]
    scores = pd.DataFrame(index=audit.index)
    scores["GRNM rerun"] = baseline
    scores["GRNM old"] = old.weight.abs()
    scores["Pando metacells"] = audit["Pando metacells"]
    scores["Rank consensus"] = (baseline.rank(pct=True) + old.weight.abs().rank(pct=True)) / 2
    scores["Rank consensus plus binary"] = (2 * scores["Rank consensus"] + audit.binary.rank(pct=True)) / 3
    scores["Input GLM only"] = direct
    for label, embedding in (("Phase1 cosine", z1), ("Phase2 cosine", z2)):
        centered = embedding - embedding.mean(axis=0)
        query = centered[np.flatnonzero(names == "GLI3")[0]]
        similarity = centered @ query / np.maximum(np.linalg.norm(centered, axis=1) * np.linalg.norm(query), 1e-12)
        scores[label] = pd.Series(similarity, index=names)
    rows = []
    for label, score in scores.items():
        for scope, mask in (("all", np.ones(len(audit), dtype=bool)), ("outside_direct_input", ~audit.direct_input)):
            ranked = score[mask].dropna().sort_index().sort_values(ascending=False, kind="stable")
            for k in (10, 20, 30, 40, 50):
                if len(ranked) >= k:
                    hits = audit.loc[ranked.index[:k], "benchmark_positive"]
                    rows.append((label, scope, k, int(hits.sum()), ";".join(hits.index[hits])))
    result = pd.DataFrame(rows, columns=["method", "scope", "k", "hits", "hit_symbols"])
    result.to_csv(OUT / "ranking_trials.tsv", sep="\t", index=False)
    scores.to_csv(OUT / "ranking_trial_scores.tsv", sep="\t")
    print(result[result.k.eq(30)].to_string(index=False))


if __name__ == "__main__":
    main()
