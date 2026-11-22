from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "comparison_results"
PROJECT = HERE.parents[2]
PANDO = HERE.parents[1] / "edge_sources/external_pando_reference"


def main():
    OUT.mkdir(exist_ok=True)
    benchmark = pd.read_csv(HERE / "benchmark_input/PRIORITY_strong_DE_top10pct_CUTTag_proxy.tsv", sep="\t")
    pairs = benchmark[["gene_id", "gene_symbol"]].dropna().drop_duplicates()
    ambiguous = set(pairs.loc[pairs.gene_symbol.duplicated(False), "gene_symbol"])
    symbols = set(pairs.gene_symbol) - ambiguous
    grnm = pd.read_csv(HERE / "grnm_rerun/human_GLI3_predictions.csv")
    if grnm.target.duplicated().any():
        raise ValueError("Duplicate GRNM targets")
    grnm = grnm.set_index("target")
    universe = set(grnm.index) - {"GLI3"} - ambiguous
    positives = symbols & universe
    scores = {"GRNM": grnm.weight.abs().dropna()}
    # Distinguish membership from ranking and keep the KO-derived network secondary.
    for label, pattern in (("Pando metacells", "*metacells*.csv"), ("Pando KO multiome", "*KO_multiome*.csv")):
        table = pd.read_csv(next(PANDO.glob(pattern)))
        subset = table[table.tf.eq("GLI3") & table.padj.between(0, 0.05)]
        scores[label] = subset.estimate.abs().groupby(subset.target).max().dropna()
    scenic_path = HERE / "benchmark_input/scenic_gli3_scores.tsv"
    if scenic_path.exists():
        scenic = pd.read_csv(scenic_path, sep="\t")
        if scenic.target.duplicated().any():
            raise ValueError("Duplicate SCENIC targets")
        scores["SCENIC"] = scenic.set_index("target").score.dropna()
    audit = pd.DataFrame({"target": sorted(universe)}).set_index("target")
    audit["benchmark_positive"] = audit.index.isin(positives)
    candidates = pairs.assign(
        ambiguous_symbol=pairs.gene_symbol.isin(ambiguous),
        in_common_universe=pairs.gene_symbol.isin(universe),
    )
    candidates.to_csv(OUT / "candidate_mapping.tsv", sep="\t", index=False)
    rows = []
    coverage = []
    for method, score in scores.items():
        ranking = score[score.index.isin(universe)].rename("score").rename_axis("target").reset_index()
        ranking = ranking.sort_values(["score", "target"], ascending=[False, True])
        ranking["hit"] = ranking.target.isin(positives)
        audit[method] = score
        coverage.append((method, len(ranking), int(ranking.hit.sum())))
        for k in (10, 20, 30, 40, 50):
            if len(ranking) >= k:
                hits = int(ranking.hit.iloc[:k].sum())
                cutoff = ranking.score.iloc[k - 1]
                tied = ranking[ranking.score.eq(cutoff)]
                above = ranking[ranking.score.gt(cutoff)]
                slots = k - len(above)
                low = int(above.hit.sum()) + max(0, slots - int((~tied.hit).sum()))
                high = int(above.hit.sum()) + min(slots, int(tied.hit.sum()))
                rows.append((method, k, hits, low, high, hits / k, hits / len(positives)))
    metrics = pd.DataFrame(rows, columns=[
        "method", "k", "hits", "min_hits_at_tie", "max_hits_at_tie", "hit_fraction", "benchmark_recall",
    ])
    metrics.to_csv(OUT / "top_k.tsv", sep="\t", index=False)
    pd.DataFrame(coverage, columns=["method", "ranked_targets", "benchmark_hits_in_ranked_targets"]).to_csv(
        OUT / "coverage.tsv", sep="\t", index=False,
    )
    audit.to_csv(OUT / "target_scores.tsv", sep="\t")
    print(f"Input rows {len(benchmark)}, unique symbols {pairs.gene_symbol.nunique()}, ambiguous {len(ambiguous)}, common positives {len(positives)}, universe {len(universe)}")
    print(metrics.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    for method, part in metrics.groupby("method", sort=False):
        ax.plot(part.k, part.hit_fraction, marker="o", label=method)
    ax.axhline(len(positives) / len(universe), color="0.4", linestyle="--", label="Uniform candidate baseline")
    ax.set(xlabel="Number of ranked targets", ylabel="Fraction on GLI3 candidate list", xticks=[10, 20, 30, 40, 50], ylim=(0, 1))
    ax.legend(frameon=False)
    fig.savefig(OUT / "top_k.pdf")
    fig.savefig(OUT / "top_k.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

