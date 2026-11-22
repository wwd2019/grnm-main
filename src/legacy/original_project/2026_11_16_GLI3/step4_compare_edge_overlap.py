from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from step2_evaluate_top10 import pando_rank

here = Path(__file__).resolve().parent
inputs = here / "inputs"
results = here / "results"
universe = set(pd.read_csv(inputs / "target_scores.tsv", sep="\t").target)
reference = pando_rank(inputs / "metacells.csv", "padj", universe)
first = pando_rank(inputs / "TF_gene_network_us.csv", "padj_min", universe)
final = pd.read_csv(results / "final_network.tsv", sep="\t")
summary = pd.read_csv(results / "summary.tsv", sep="\t")
assert list(summary.hits) == [1, 3, 2]
assert list(summary.evaluated) == [10, 10, 10]

audit = reference[["target", "padj", "estimate"]].copy()
audit["in_our_pando"] = audit.target.isin(first.target)
audit["in_final_grnm"] = audit.target.isin(final.target)
audit.to_csv(results / "metacells_overlap.tsv", sep="\t", index=False)
shared = int(audit.in_final_grnm.sum())
print(f"External edges recovered: {shared}/{len(audit)}")
print(audit.loc[audit.in_final_grnm, "target"].tolist())
print(f"First-stage overlap: {audit.in_our_pando.sum()}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
ax = axes[0]
bars = ax.bar(["Our Pando", "Our Pando\n+ GRNM", "External Pando\nmetacells"],
              summary.hits, color=["#577c91", "#117765", "#949da4"], width=.65)
ax.bar_label(bars, labels=[f"{x}/10" for x in summary.hits], padding=4)
ax.set(title="A  Experimental candidate hits", ylabel="Hits among top 10 GLI3 edges", ylim=(0, 4))
ax = axes[1]
sizes = [int(audit.in_our_pando.sum()), shared]
bars = ax.bar(["Our Pando", "Our Pando\n+ GRNM"], sizes, color=["#577c91", "#117765"], width=.6)
ax.bar_label(bars, labels=[f"{x}/{len(audit)}" for x in sizes], padding=4)
ax.set(title="B  Recovery of external Pando edges", ylabel="Shared GLI3 edges", ylim=(0, len(audit)))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(results / "reviewer_comparison.pdf")
fig.savefig(results / "reviewer_comparison.png", dpi=220)

