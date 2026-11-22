from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

results = Path(__file__).resolve().parent / "results"
edges = pd.read_csv(results / "final_network.tsv", sep="\t")
assert edges.target.is_unique
ink, teal, orange, gray = "#182b38", "#087e76", "#b45c24", "#a4b3bc"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": ink})

fig, ax = plt.subplots(figsize=(8.4, 8.4))
ax.set_aspect("equal")
ax.set(xlim=(-1.48, 1.48), ylim=(-1.43, 1.41))
ax.axis("off")
fig.subplots_adjust(left=.045, right=.955, top=.875, bottom=.09)
fig.text(.06, .957, "GLI3 direct regulatory network", fontsize=17, weight="bold")
fig.text(.06, .922, f"Final Pando + GRNM network  |  {len(edges)} target genes", fontsize=10, color="#667785")

angles = np.pi/2 - np.arange(len(edges))*2*np.pi/len(edges)
for row, angle in sorted(zip(edges.itertuples(index=False), angles), key=lambda item: item[0].hit):
    x, y = np.cos(angle), np.sin(angle)
    novel = row.stage == "GRNM expansion"
    edge_color = teal if row.hit else (orange if novel else "#c9d3d8")
    line_width = 2.5 if row.hit else (1.35 if novel else .75)
    ax.annotate("", xy=(.96*x,.96*y), xytext=(.135*x,.135*y),
                arrowprops={"arrowstyle":"-|>", "color":edge_color,
                            "lw":line_width, "linestyle":"--" if novel else "-",
                            "mutation_scale":12 if row.hit else 7, "shrinkA":0, "shrinkB":0})
    ax.scatter(x, y, s=75 if row.hit else 38, facecolor=teal if row.hit else "white",
               edgecolor=teal if row.hit else gray, linewidth=1.05, zorder=3)
    align = "center" if abs(x)<.08 else ("left" if x>0 else "right")
    ax.text(1.08*x,1.08*y,row.target,ha=align,va="center",fontsize=10 if row.hit else 8,
            color=teal if row.hit else "#667785",weight="bold" if row.hit else "normal")
ax.scatter(0,0,s=1450,color=ink,zorder=4)
ax.text(0,0,"GLI3",ha="center",va="center",color="white",weight="bold",fontsize=12,zorder=5)

legend = [
    Line2D([],[],color=teal,lw=2.5,marker="o",label="Experimental candidate hit"),
    Line2D([],[],color="#c9d3d8",label="Other Pando input edge"),
    Line2D([],[],color=orange,linestyle="--",label="GRNM expansion edge"),

]
fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.04),ncol=3,frameon=False,fontsize=9)
fig.savefig(results / "gli3_final_tree.pdf")
fig.savefig(results / "gli3_final_tree.png", dpi=240)
plt.close(fig)

