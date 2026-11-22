#!/usr/bin/env python3
"""Draw the archived GLI3 top-10 comparison; run from any directory."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]; HERE = ROOT/'results/generated'; HERE.mkdir(parents=True,exist_ok=True)
source = ROOT / 'results/model/comparison_hes4_included.tsv'
data = pd.read_csv(source, sep='\t').set_index('method')
order = ['pando+GRN', 'Our Pando',
         'External Pando metacells', 'GRNBoost2 only']
labels = ['pando+GRN\n(all targets)', 'Our Pando',
          'External Pando\nmetacells', 'GRNBoost2 only\n(before motif filtering)']
data = data.loc[order]
assert data.top10_hits.between(0, 10).all()
colors = ['#117866', '#8ea6b1', '#9a9fa9', '#bc8245']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig = plt.figure(figsize=(11.5, 5.6), layout='constrained')
grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.18])
ax = fig.add_subplot(grid[0, 0]); positions = np.arange(len(data))
bars = ax.barh(positions, data.top10_hits, color=colors, height=.58)
ax.invert_yaxis(); ax.set_yticks(positions, labels); ax.set_xlim(0, 4.25)
ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xlabel('Candidate hits in top 10')
ax.set_title('A   Candidate recovery', loc='left', fontweight='bold', pad=14)
ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(axis='y', length=0)
for bar, (_, row) in zip(bars, data.iterrows()):
    ax.text(bar.get_width()+.08, bar.get_y()+bar.get_height()/2,
            f"{int(row.top10_hits)}/10   {row.hit_genes.replace(';', ', ')}",
            ha='left', va='center', fontsize=9)
ax.set_xlim(0, 5.4)

ax2 = fig.add_subplot(grid[0, 1]); rank = data[['HES4_rank','HES5_rank']].to_numpy(dtype=float)
# Color indicates whether the rank is in the top ten, not a comparable model score.
status = np.where(np.isnan(rank), 0, np.where(rank <= 10, 2, 1))
cmap = ListedColormap(['#eeeeee', '#e2ebef', '#b8e0d4'])
ax2.imshow(status, cmap=cmap, norm=BoundaryNorm([-.5,.5,1.5,2.5], cmap.N), aspect='auto')
ax2.set_xticks([0,1], ['HES4', 'HES5']); ax2.tick_params(axis='x', length=0, pad=8)
ax2.set_yticks(positions, labels); ax2.tick_params(axis='y', length=0, labelsize=9)
ax2.set_title('B   Rank of highlighted genes', loc='left', fontweight='bold', pad=14)
for i in range(len(data)):
    for j in range(2):
        value = rank[i,j]; label = 'Not listed' if np.isnan(value) else f'#{int(value)}'
        ax2.text(j, i, label, ha='center', va='center', fontsize=10,
                 fontweight='bold' if not np.isnan(value) and value <= 10 else 'normal', color='#173c35' if status[i,j]==2 else '#323a40')
ax2.set_xticks(np.arange(-.5,2,1), minor=True); ax2.set_yticks(np.arange(-.5,len(data),1), minor=True)
ax2.grid(which='minor', color='white', linewidth=3); ax2.tick_params(which='minor', bottom=False, left=False)
for spine in ax2.spines.values(): spine.set_visible(False)
ax2.legend(handles=[Patch(facecolor='#b8e0d4', label='In top 10'),
                    Patch(facecolor='#e2ebef', label='Outside top 10'),
                    Patch(facecolor='#eeeeee', label='Not listed')],
           loc='upper center', bbox_to_anchor=(.5,-.10), ncol=3, frameon=False, fontsize=8)
fig.suptitle('GLI3 candidate target comparison', x=.02, ha='left', fontsize=15, fontweight='bold')
fig.text(.02, -.045, 'Exploratory, HES4-inclusive universe (1,794 targets; 48 candidates). GRNBoost2 is shown before motif filtering.\npando+GRN ranks all targets; other methods rank their retained edges. Rank values and model scores are not directly comparable.',
         fontsize=8, color='#4b555b', ha='left', va='top')
for extension in ('png','pdf','svg'):
    fig.savefig(HERE/f'gli3_method_comparison.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved gli3_method_comparison.png, .pdf, and .svg')
