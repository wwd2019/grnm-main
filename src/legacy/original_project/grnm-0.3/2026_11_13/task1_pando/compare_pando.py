from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
PANDO_REFERENCE_DIR = OUT.parent / "edge_sources" / "external_pando_reference"
PREDICTIONS = PROJECT / "outputs/phase3_human_output/human_new_edge.txt"
PROFILE = PROJECT / "outputs/phase2_human_output/human_gene_z_phase2.npz"
REFERENCES = {
    "Metacells": "zenodo5242913_RNA_ATAC_metacells_GRN_meta_format(1).csv",
    "GLI3 KO multiome": "zenodo5242913_GLI3_KO_multiome_18d__GRN_meta_format(2).csv",
}
PAIR = ["tf", "target"]


def significant(path):
    frame = pd.read_csv(path)
    frame = frame[frame.padj.between(0, 0.05) & frame.tf.ne(frame.target)].copy()
    frame["magnitude"] = frame.estimate.abs()
    return frame.sort_values("magnitude", ascending=False, kind="stable").drop_duplicates(PAIR)


def main():
    train = significant(PROJECT / "example_data/glm_network_coefs_human.csv")
    pred = pd.read_csv(PREDICTIONS, sep="\t").rename(columns={"source": "tf"})
    if pred.duplicated(PAIR).any():
        raise ValueError("Expected one exported score per directed pair")
    with np.load(PROFILE, allow_pickle=False) as archive:
        targets = set(archive["gene_names"])
    raw_glm = pd.read_csv(PROJECT / "example_data/glm_network_coefs_human.csv").dropna(subset=["tf", "target", "estimate"])
    motif = pd.read_csv(PROJECT / "example_data/motif2tf_human.tsv", sep="\t").dropna(subset=["tf", "motif_tf", "prc_similarity"])
    sources = (set(raw_glm.tf) | set(motif.motif_tf)) & targets
    if not set(pred.tf) <= sources or not set(pred.target) <= targets:
        raise ValueError("Prediction IDs outside the declared human candidate space")
    motif_pairs = motif.rename(columns={"tf": "target", "motif_tf": "tf"})
    train_index = pd.MultiIndex.from_frame(pd.concat([train[PAIR], motif_pairs[PAIR]]).drop_duplicates())
    density_by_tf = pred.groupby("tf").size() / (len(targets) - 1)
    metrics = []
    tf_metrics = []
    details = []
    for label, filename in REFERENCES.items():
        ref = significant(PANDO_REFERENCE_DIR / filename)
        ref["reference"] = label
        ref["eligible"] = ref.tf.isin(sources) & ref.target.isin(targets)
        ref["in_training_input"] = pd.MultiIndex.from_frame(ref[PAIR]).isin(train_index)
        ref = ref.merge(pred, how="left", on=PAIR, validate="one_to_one")
        ref["recovered"] = ref.weight.notna()
        ref["sign_agrees"] = (np.sign(ref.estimate) == np.sign(ref.weight)).where(ref.recovered)
        details.append(ref)
        for group, group_frame in (
            ("All TFs", ref),
            ("GLI3", ref[ref.tf.eq("GLI3")]),
        ):
            for scope, part in (
                ("All reference", group_frame),
                ("Eligible", group_frame[group_frame.eligible]),
                ("Eligible outside training input", group_frame[group_frame.eligible & ~group_frame.in_training_input]),
            ):
                n = len(part)
                recovered = int(part.recovered.sum())
                sign_match = int(part.sign_agrees.eq(True).sum())
                metrics.append((label, group, scope, n, recovered, recovered / n if n else np.nan,
                                sign_match, sign_match / recovered if recovered else np.nan,
                                part.tf.map(density_by_tf).fillna(0).mean()))
        for tf, part in ref[ref.eligible & ~ref.in_training_input].groupby("tf"):
            tf_metrics.append((label, tf, len(part), int(part.recovered.sum()), part.recovered.mean()))
    summary = pd.DataFrame(metrics, columns=[
        "reference", "group", "scope", "reference_edges", "recovered_edges", "recovery_fraction",
        "matching_sign", "sign_fraction_among_recovered", "tf_matched_expected_fraction",
    ])
    summary.to_csv(OUT / "pando_recovery.tsv", sep="\t", index=False)
    pd.concat(details).to_csv(OUT / "pando_edge_audit.tsv", sep="\t", index=False, columns=[
        "reference", *PAIR, "estimate", "padj", "eligible", "in_training_input",
        "weight", "recovered", "sign_agrees",
    ])
    tf_summary = pd.DataFrame(tf_metrics, columns=["reference", "tf", "eligible_edges", "recovered", "fraction"])
    tf_summary.to_csv(OUT / "pando_per_tf.tsv", sep="\t", index=False)

    candidate_n = len(sources) * (len(targets) - 1)
    density = len(pred) / candidate_n
    print(f"Candidate space: {len(sources)} TFs, {len(targets)} genes; {len(pred)}/{candidate_n} exported ({density:.2%})")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout="constrained")
    scopes = ["All reference", "Eligible", "Eligible outside training input"]
    for ax, group in zip(axes, ("All TFs", "GLI3")):
        for i, (label, color) in enumerate(zip(REFERENCES, ("#216778", "#d0833e"))):
            data = summary[(summary.group == group) & (summary.reference == label)].set_index("scope").loc[scopes]
            heights = data.recovery_fraction.fillna(0)
            bars = ax.bar(np.arange(3) + (i - 0.5) * 0.34, heights, width=0.34, label=label, color=color)
            ax.bar_label(bars, labels=[f"{hit:,}/{n:,}" if n else "n/a"
                                      for hit, n in zip(data.recovered_edges, data.reference_edges)], fontsize=8, padding=3)
            ax.scatter(np.arange(3) + (i - 0.5) * 0.34, data.tf_matched_expected_fraction,
                       marker="_", s=180, color="black", zorder=4,
                       label="TF-matched expectation" if i == 0 else None)
        ax.set(title=group, ylim=(0, 1.09), xticks=range(3),
               xticklabels=["All Pando", "Eligible", "Eligible outside\ntraining input"])
    axes[0].set_ylabel("Fraction of Pando edges recovered")
    axes[0].legend(loc="upper left", frameon=False)
    fig.savefig(OUT / "pando_recovery.pdf")
    fig.savefig(OUT / "pando_recovery.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, layout="constrained")
    for ax, label in zip(axes, REFERENCES):
        data = tf_summary[(tf_summary.reference == label) & (tf_summary.eligible_edges >= 5)]
        ax.scatter(data.eligible_edges, data.fraction, color="#216778", alpha=0.55, s=18)
        ax.axhline(density, color="#d0833e", linestyle="--", label="Overall export density")
        gli3 = data[data.tf.eq("GLI3")]
        if not gli3.empty:
            ax.scatter(gli3.eligible_edges, gli3.fraction, color="#a03843", label="GLI3", zorder=3)
        ax.set(title=label, xlabel="Eligible Pando edges outside training input", xscale="log", ylim=(-0.04, 1.04))
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Fraction recovered per TF")
    fig.savefig(OUT / "pando_per_tf.pdf")
    fig.savefig(OUT / "pando_per_tf.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()




