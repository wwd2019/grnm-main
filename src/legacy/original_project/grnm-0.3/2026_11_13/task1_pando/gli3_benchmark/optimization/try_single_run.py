from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
BENCH = OUT.parent


def main():
    audit = pd.read_csv(OUT / "target_scores.tsv", sep="\t").set_index("target")
    run = BENCH / "grnm_rerun"
    edges = pd.read_csv(run / "human_new_edge.txt", sep="\t")
    manifest = pd.read_json(run / "human_prediction_manifest.json", typ="series")
    sources = set(edges.source)
    if len(sources) != int(manifest.n_sources) or "GLI3" not in sources:
        raise ValueError("Unexpected exported source set")
    background = edges[edges.source.ne("GLI3")].copy()
    # Each retained edge contributes its magnitude above the export threshold.
    # Omitted edges contribute zero to this background measure.
    threshold = float(manifest.min_abs_weight)
    excess = (background.weight.abs() - threshold).clip(lower=0)
    n = pd.Series(len(sources) - 1, index=audit.index)
    source_is_target = audit.index.isin(sources)
    n.loc[source_is_target] -= 1
    background_mean = excess.groupby(background.target).sum().reindex(audit.index, fill_value=0) / n
    audit["GRNM adjusted single run"] = audit.GRNM - background_mean
    audit["background_mean_excess"] = background_mean
    rows = []
    for scope, subset in (("all", audit), ("outside_direct_input", audit[~audit.direct_input])):
        for name in ("GRNM", "GRNM adjusted single run", "Pando metacells"):
            ranked = subset[name].dropna().sort_index().sort_values(ascending=False, kind="stable")
            for k in (10, 20, 30, 40):
                if len(ranked) >= k:
                    hit = subset.loc[ranked.index[:k], "benchmark_positive"]
                    rows.append((scope, name, k, int(hit.sum()), ";".join(hit.index[hit])))
    metrics = pd.DataFrame(rows, columns=["scope", "method", "k", "hits", "hit_symbols"])
    audit[["GRNM", "GRNM adjusted single run", "background_mean_excess", "benchmark_positive", "direct_input"]].to_csv(
        OUT / "single_run_scores.tsv", sep="\t"
    )
    metrics.to_csv(OUT / "single_run_top_k.tsv", sep="\t", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
