from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

results = Path(__file__).resolve().parent / "results"
top = pd.read_csv(results / "top10_edges.tsv", sep="\t")
ink, teal, muted = "#182b38", "#087e76", "#667785"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "text.color":ink, "axes.labelcolor":ink, "pdf.fonttype":42})
fig = plt.figure(figsize=(9.2, 4.1), facecolor="white")
left = fig.add_axes([.24,.23,.72,.52])
fig.text(.035,.92,"Experimental candidate hits",size=13,weight="bold")
fig.text(.035,.84,"GLI3 direct edges in confidence rank order",color=muted)
names = ["Our Pando", "Our Pando + GRNM", "Pando metacells"]
for row, name in enumerate(names):
    subset = top[top.network.eq(name)]
    hit = subset.hit
    left.scatter(subset.loc[~hit,"rank"], [row]*int((~hit).sum()),
                 s=60, facecolor="white", edgecolor="#a8b7bf", linewidth=1.15)
    left.scatter(subset.loc[hit,"rank"], [row]*int(hit.sum()), s=68, color=teal)
    for item in subset.itertuples():
        left.annotate(item.target,(item.rank,row),xytext=(0,-10),
                      textcoords="offset points",ha="center",va="top",size=8,color=teal if item.hit else muted)
    left.text(10.9,row,f"{int(hit.sum())}/10",va="center",weight="bold")
left.set(xlim=(.5,11.8),ylim=(2.55,-.55),xticks=[],yticks=range(3),
         yticklabels=names)
left.tick_params(axis="both",length=0,pad=7,labelcolor=ink)
for spine in left.spines.values():
    spine.set_visible(False)
fig.legend(handles=[Line2D([],[],marker="o",linestyle="None",color=teal,label="Experimental candidate"),
                    Line2D([],[],marker="o",linestyle="None",markerfacecolor="white",
                           markeredgecolor="#a8b7bf",label="Other ranked target")],
           loc="lower left",bbox_to_anchor=(.028,.035),ncol=2,frameon=False,fontsize=9)
fig.savefig(results/"reviewer_panel_a.pdf")
fig.savefig(results/"reviewer_panel_a.png",dpi=240)



