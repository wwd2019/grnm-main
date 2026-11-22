"""One fitted link scorer per species. Evaluation TF edges are masked before feature construction."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]


def build_examples(reference, names, rng):
    rows = []
    for tf, targets in reference.groupby("tf").target:
        hits = sorted(set(targets))
        candidates = sorted(names - set(hits) - {tf})
        negative = rng.choice(candidates, size=min(4 * len(hits), len(candidates)), replace=False)
        rows.extend((tf, target, 1) for target in hits)
        rows.extend((tf, target, 0) for target in negative)
    return pd.DataFrame(rows, columns=["tf", "target", "label"])


def fit(train, names, eval_pairs, seed):
    ids = {g: i for i, g in enumerate(sorted(names))}
    source, target = train.tf.map(ids), train.target.map(ids)
    adj = sparse.csr_matrix((np.ones(len(train)), (source, target)), shape=(len(ids), len(ids)))
    # The factorization includes only training TFs' outgoing edges.
    incoming = TruncatedSVD(n_components=16, random_state=seed).fit_transform(adj.T)
    degree = np.asarray(adj.sum(axis=0)).ravel()
    outgoing = np.asarray(adj.sum(axis=1)).ravel()
    rng = np.random.default_rng(seed)
    samples = build_examples(train, names, rng)

    def features(frame):
        s, t = frame.tf.map(ids).to_numpy(), frame.target.map(ids).to_numpy()
        a, b = incoming[s], incoming[t]
        return np.column_stack((a, b, a*b, abs(a-b), np.log1p(degree[t]), np.log1p(outgoing[t])))

    model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=seed,
        early_stopping=False,
    )
    model.fit(features(samples), samples.label)
    return model.predict_proba(features(eval_pairs))[:, 1], eval_pairs.target.map(ids).map(dict(enumerate(degree)))


def human():
    bench = OUT.parent / "gli3_benchmark"
    audit = pd.read_csv(bench / "comparison_results/target_scores.tsv", sep="\t")
    names = set(audit.target) | {"GLI3"}
    ref = pd.read_csv(PROJECT / "example_data/glm_network_coefs_human.csv")
    ref = ref.loc[ref.padj.between(0, .05) & ref.tf.ne(ref.target), ["tf", "target"]].drop_duplicates()
    train = ref[ref.tf.ne("GLI3") & ref.tf.isin(names) & ref.target.isin(names)]
    audit["tf"] = "GLI3"
    audit["masked_network_score"], audit["target_degree"] = fit(train, names, audit, 42)
    audit.to_csv(OUT / "human_scores.tsv", sep="\t", index=False)
    rows = []
    for model in ("GRNM", "masked_network_score", "target_degree", "Pando metacells"):
        ranked = audit.dropna(subset=[model]).sort_values([model, "target"], ascending=[False, True])
        for k in (10, 20, 30, 40):
            top = ranked.head(k)
            rows.append((model, k, int(top.benchmark_positive.sum()), ";".join(top.loc[top.benchmark_positive, "target"])))
    return pd.DataFrame(rows, columns=["method", "k", "hits", "genes"])


def worm():
    mapping = pd.read_csv(PROJECT / "data/smed_smesg_mapping.csv").dropna().drop_duplicates()
    mapping = mapping[~mapping.smesg_id.duplicated(False) & ~mapping.smed_id.duplicated(False)]
    ids = mapping.set_index("smesg_id").smed_id
    folder = PROJECT / "outputs/phase5_output"
    names = set((folder / "combined_genes.txt").read_text(encoding="utf-8").splitlines()) & set(ids)
    ref = pd.read_csv(PROJECT / "2026_11_13/edge_sources/internal_model_predictions/edges.csv")
    # The legacy hand-added NFYC–IDH1 visualization edge is not a reference example.
    manual = ref.source.eq("SMESG000077821") & ref.target.eq("SMESG000035181")
    ref = ref.loc[~manual, ["source", "target"]].rename(columns={"source": "tf"}).apply(lambda c: c.map(ids)).dropna()
    ref = ref[ref.tf.ne(ref.target) & ref.tf.isin(names) & ref.target.isin(names)].drop_duplicates()
    tf = np.array(sorted(set(ref.tf)))
    rng = np.random.default_rng(42)
    holdout = set(rng.choice(tf, size=max(1, len(tf)//5), replace=False))
    train = ref[~ref.tf.isin(holdout)]
    # Use a matched universe for the existing expression baselines and the new scorer.
    scores = pd.read_csv(folder / "genie3_scores.csv", usecols=["tf", "target", "importance"])
    scores = scores[scores.tf.isin(holdout) & scores.target.isin(names) & scores.tf.ne(scores.target)]
    scores = scores.merge(ref[ref.tf.isin(holdout)].assign(label=1), how="left", on=["tf", "target"])
    scores["label"] = scores.label.fillna(0).astype(int)
    eval_parts = []
    for _, group in scores.groupby("tf"):
        pos = group[group.label.eq(1)]
        if len(pos):
            bg = group[group.label.eq(0)]
            sampled = bg.iloc[rng.choice(len(bg), size=min(len(bg), 4 * len(pos)), replace=False)]
            eval_parts.extend((pos, sampled))
    test = pd.concat(eval_parts, ignore_index=True)
    test["masked_network_score"], test["target_degree"] = fit(train, names, test, 42)
    coexp = pd.read_csv(folder / "coexp_scores.csv", usecols=["tf", "target", "score"])
    test = test.merge(coexp, how="left", on=["tf", "target"], validate="one_to_one")
    test = test.dropna(subset=["score"])
    test.to_csv(OUT / "worm_scores.tsv", sep="\t", index=False)
    rows = []
    for model in ("masked_network_score", "target_degree", "importance", "score"):
        value = test[model].abs() if model == "score" else test[model]
        rows.append((model, test.tf.nunique(), len(test), test.label.sum(),
                     average_precision_score(test.label, value), roc_auc_score(test.label, value)))
    return pd.DataFrame(rows, columns=["method", "heldout_tfs", "pairs", "reference_positives", "average_precision", "auroc"])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for species, run in (("human", human), ("worm", worm)):
            result = run()
            result.to_csv(OUT / f"{species}_metrics.tsv", sep="\t", index=False)
            print(species, result.to_string(index=False), flush=True)

