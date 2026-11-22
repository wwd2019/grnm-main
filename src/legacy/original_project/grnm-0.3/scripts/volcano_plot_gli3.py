import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 读取数据
# ============================================================
df = pd.read_csv("../raw_data/TF_gene_network.csv")

# 兼容不同版本的列名：新版用 padj_min / pval_min，旧版用 padj / pval
rename_map = {'padj_min': 'padj', 'pval_min': 'pval'}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# 只取 GLI3 作为 TF 的记录
gli3_df = df[df['tf'] == 'GLI3'].copy()
if gli3_df.empty:
    raise SystemExit("未找到 tf == 'GLI3' 的记录，请检查输入数据")
print(f"GLI3 target genes: {len(gli3_df)}")

# ---- 计算指标 ----
gli3_df['effect_size'] = gli3_df['estimate']

# padj 为 0 的替换为极小值，避免 log10(0) = -inf
min_padj = gli3_df.loc[gli3_df['padj'] > 0, 'padj'].min() / 10
gli3_df['padj_safe'] = gli3_df['padj'].replace(0, min_padj)
gli3_df['neg_log10_padj'] = -np.log10(gli3_df['padj_safe'])

# ---- 分组：按调控方向 ----
# direction 列取值 positive / negative
gli3_df['direction'] = gli3_df['direction'].astype(str).str.lower().str.strip()

# 发散配色（blue <-> red），已验证 CVD 分离度
DIR_STYLE = {
    'positive': ('#e34948', 'Activating (positive)'),
    'negative': ('#2a78d6', 'Repressive (negative)'),
}

print(f"positive: {(gli3_df['direction'] == 'positive').sum()}")
print(f"negative: {(gli3_df['direction'] == 'negative').sum()}")

# ============================================================
# 火山图: effect size vs -log10(padj)，颜色 = 调控方向
# ============================================================
SURFACE = '#fcfcfb'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# 先画非目标方向（若有），再画两组
for key in ['negative', 'positive']:
    grp = gli3_df[gli3_df['direction'] == key]
    if grp.empty:
        continue
    color, label = DIR_STYLE[key]
    ax.scatter(
        grp['effect_size'], grp['neg_log10_padj'],
        c=color, s=42, alpha=0.75,
        edgecolors=SURFACE, linewidths=0.8,   # 2px 表面环，分隔重叠点
        label=label, zorder=3,
    )

# 无法归类方向的点（数据异常时兜底）
other = gli3_df[~gli3_df['direction'].isin(DIR_STYLE)]
if not other.empty:
    ax.scatter(
        other['effect_size'], other['neg_log10_padj'],
        c='#b0b0b0', s=42, alpha=0.75,
        edgecolors=SURFACE, linewidths=0.8,
        label=f"Unclassified (n={len(other)})", zorder=2,
    )

# ---- 选择性直接标注：只标 effect size 绝对值和显著性最强的 ----
# 候选集合 = 显著性 top8 ∪ |effect| top5
cand = list(dict.fromkeys(
    list(gli3_df.nlargest(8, 'neg_log10_padj').index) +
    list(gli3_df.reindex(gli3_df['effect_size'].abs()
                         .sort_values(ascending=False).index).head(5).index)
))
# 按显著性降序，最显著的最先占用位置
cand.sort(key=lambda i: -gli3_df.loc[i, 'neg_log10_padj'])

# 候选偏移方向：先右后左，再上下，取第一个不与已放置标签重叠的
OFFSETS = [(6, 4), (-6, 4), (6, -10), (-6, -10), (6, 14), (-6, 14)]

fig.canvas.draw()                      # 取得 renderer 以便做真实包围盒碰撞检测
renderer = fig.canvas.get_renderer()
placed_boxes = []

for idx in cand:
    row = gli3_df.loc[idx]
    for dx, dy in OFFSETS:
        t = ax.annotate(
            row['target'],
            (row['effect_size'], row['neg_log10_padj']),
            fontsize=8, color='#52514e', fontstyle='italic',
            xytext=(dx, dy), textcoords='offset points', zorder=4,
        )
        bb = t.get_window_extent(renderer=renderer)
        if any(bb.overlaps(p) for p in placed_boxes):
            t.remove()
            continue
        placed_boxes.append(bb)
        break
    # 所有方向都冲突则放弃该标签（数值仍可从 CSV 表格视图查）

# ---- 参考线 ----
ax.axvline(x=0, color=BASELINE, linestyle='-', linewidth=1.0, alpha=0.9, zorder=1)

ax.set_xlabel('Effect size (GRN estimate)', fontsize=12, color='#0b0b0b')
ax.set_ylabel(r'$-\log_{10}(\mathrm{padj})$', fontsize=12, color='#0b0b0b')
ax.set_title('GLI3 target genes', fontsize=14, color='#0b0b0b', pad=12)

# ---- 网格与边框：细发丝线、退到背景 ----
ax.grid(True, which='major', color=GRID, linewidth=0.6, linestyle='-', zorder=0)
ax.set_axisbelow(True)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
for side in ['left', 'bottom']:
    ax.spines[side].set_color(BASELINE)
    ax.spines[side].set_linewidth(0.8)
ax.tick_params(colors=INK_MUTED, labelsize=10, length=3)

# ---- 图例：≥2 系列必须有 ----
ax.legend(
    loc='upper left', frameon=True, fontsize=10,
    facecolor=SURFACE, edgecolor=GRID, framealpha=1.0,
)

plt.tight_layout()
plt.savefig('../figures/GLI3_volcano_by_direction.png', dpi=150,
            bbox_inches='tight', facecolor=SURFACE)
plt.close()
print("Saved: ../figures/GLI3_volcano_by_direction.png")

# ---- 表格视图：图上每个点的数值都可查 ----
out = gli3_df[['target', 'effect_size', 'n_regions', 'padj', 'neg_log10_padj', 'direction']]
out = out.sort_values('neg_log10_padj', ascending=False)
out.to_csv('../figures/GLI3_volcano_by_direction.csv', index=False)
print("Saved: ../figures/GLI3_volcano_by_direction.csv")

print("\n=== Done ===")
