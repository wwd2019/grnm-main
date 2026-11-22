from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

results = Path(__file__).resolve().parent / "results"
top = pd.read_csv(results / "top10_edges.tsv", sep="\t")
overlap = pd.read_csv(results / "metacells_overlap.tsv", sep="\t")
ink, teal, muted = "#182b38", "#087e76", "#667785"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "text.color":ink, "axes.labelcolor":ink, "pdf.fonttype":42})
fig = plt.figure(figsize=(10, 4.1), facecolor="white")
left = fig.add_axes([.23,.22,.43,.51])
right = fig.add_axes([.73,.17,.23,.57])
fig.text(.035,.92,"A   Experimental candidate hits",size=13,weight="bold")
fig.text(.715,.92,"B   External edge overlap",size=13,weight="bold")
fig.text(.035,.84,"GLI3 direct edges in confidence rank order",color=muted)
fig.text(.715,.84,"One square per Pando metacells edge",color=muted,size=9)
names = ["Our Pando", "Our Pando + GRNM", "Pando metacells"]
for row, name in enumerate(names):
    subset = top[top.network.eq(name)]
    hit = subset.hit
    left.scatter(subset.loc[~hit,"rank"], [row]*int((~hit).sum()),
                 s=60, facecolor="white", edgecolor="#a8b7bf", linewidth=1.15)
    left.scatter(subset.loc[hit,"rank"], [row]*int(hit.sum()), s=68, color=teal)
    for item in subset.loc[hit].itertuples():
        left.annotate(item.target,(item.rank,row),xytext=(0,9),
                      textcoords="offset points",ha="center",size=7.5,color=teal)
    left.text(10.9,row,f"{int(hit.sum())}/10",va="center",weight="bold")
left.set(xlim=(.5,11.8),ylim=(2.55,-.55),xticks=range(1,11),yticks=range(3),
         yticklabels=names,xlabel="Rank")
left.tick_params(axis="both",length=0,pad=7,labelcolor=ink)
for spine in left.spines.values():
    spine.set_visible(False)
for index, row in enumerate(overlap.itertuples()):
    right.scatter(index%7, -(index//7), marker="s", s=112,
                  color=teal if row.in_final_grnm else "#dde6e9")
right.set_aspect("equal")
right.set(xlim=(-.6,6.6),ylim=(-6.6,.6))
right.axis("off")
fig.text(.83,.125,f"{int(overlap.in_final_grnm.sum())}/{len(overlap)} shared",ha="center",weight="bold")
fig.text(.83,.075,", ".join(sorted(overlap.loc[overlap.in_final_grnm,"target"])),
         ha="center",color=muted,size=9)
fig.legend(handles=[Line2D([],[],marker="o",linestyle="None",color=teal,label="Experimental candidate"),
                    Line2D([],[],marker="o",linestyle="None",markerfacecolor="white",
                           markeredgecolor="#a8b7bf",label="Other ranked target")],
           loc="lower left",bbox_to_anchor=(.028,.035),ncol=2,frameon=False,fontsize=9)
fig.savefig(results/"reviewer_comparison_dot_matrix.pdf")
fig.savefig(results/"reviewer_comparison_dot_matrix.png",dpi=240)
