from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
BENCH = OUT.parent
PROJECT = BENCH.parents[2]


def features(z, source, target):
    left, right = z[source], z[target]
    return np.column_stack((left, right, left * right, abs(left - right)))


def main():
    with np.load(PROJECT / "outputs/phase2_human_output/human_gene_z_phase2.npz") as data:
        z, names = data["z"], data["gene_names"]
    ids = {name: i for i, name in enumerate(names)}
    glm = pd.read_csv(PROJECT / "example_data/glm_network_coefs_human.csv")
    glm = glm[glm.tf.isin(ids) & glm.target.isin(ids) & glm.tf.ne(glm.target)]
    significant = glm[glm.padj.le(0.05)]
    known = significant.groupby("tf").target.agg(set).to_dict()
    rng = np.random.default_rng(42)
    rows = []
    all_names = set(names)
    for tf, hits in sorted(known.items()):
        if tf == "GLI3":
            continue
        background = sorted(all_names - hits - {tf})
        sampled = rng.choice(background, size=min(len(background), len(hits) * 4), replace=False)
        rows.extend((ids[tf], ids[t], 1) for t in sorted(hits))
        rows.extend((ids[tf], ids[t], 0) for t in sampled)
    src, tgt, y = np.array(rows).T
    x = features(z, src, tgt)
    configs = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
        "boosted": lambda: HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=42),
    }
    cv = []
    with threadpool_limits(limits=1):
        for label, build in configs.items():
            for fold, (train, test) in enumerate(GroupKFold(n_splits=3).split(x, y, src), 1):
                model = build().fit(x[train], y[train])
                score = model.predict_proba(x[test])[:, 1]
                cv.append((label, fold, len(test), y[test].mean(), average_precision_score(y[test], score)))
        cv = pd.DataFrame(cv, columns=["model", "fold", "pairs", "positive_fraction", "reference_average_precision"])
        cv.to_csv(OUT / "grouped_cv.tsv", sep="\t", index=False)
        winner = cv.groupby("model").reference_average_precision.mean().idxmax()
        model = configs[winner]().fit(x, y)
        targets = np.flatnonzero(names != "GLI3")
        new_score = model.predict_proba(features(z, np.full(len(targets), ids["GLI3"]), targets))[:, 1]

    audit = pd.read_csv(BENCH / "comparison_results/target_scores.tsv", sep="\t").set_index("target")
    audit["binary"] = pd.Series(new_score, index=names[targets])
    motif = pd.read_csv(PROJECT / "example_data/motif2tf_human.tsv", sep="\t")
    prior = known.get("GLI3", set()) | set(motif.loc[motif.motif_tf.eq("GLI3"), "tf"])
    audit["direct_input"] = audit.index.isin(prior)
    audit.to_csv(OUT / "target_scores.tsv", sep="\t")
    result = []
    for scope, table in (("all", audit), ("outside_direct_input", audit[~audit.direct_input])):
        for column in ("GRNM", "binary", "Pando metacells"):
            ranking = table[column].dropna().sort_index().sort_values(ascending=False, kind="stable")
            for k in (10, 20, 30, 40, 50):
                if len(ranking) >= k:
                    hits = int(table.loc[ranking.index[:k], "benchmark_positive"].sum())
                    result.append((scope, column, len(ranking), k, hits, hits / k))
    metrics = pd.DataFrame(result, columns=["scope", "model", "scored", "k", "hits", "hit_fraction"])
    metrics.to_csv(OUT / "top_k.tsv", sep="\t", index=False)
    print(cv.to_string(index=False))
    print("Selected by grouped CV:", winner)
    print(metrics[metrics.k.isin([20, 30, 40])].to_string(index=False))


if __name__ == "__main__":
    main()

