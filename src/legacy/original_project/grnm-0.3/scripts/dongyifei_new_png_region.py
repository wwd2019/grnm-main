"""
GRN可视化脚本 — 从 zenodo5242913_GLI3_KO_multiome_18d__GRN_meta_format(2).csv 读取数据，
按照 test_pngs/test1.png 的形式绘制 Gene Regulatory Network 图。

- 节点大小: PageRank
- 节点颜色: 伪时序 (regulatory cascade depth)
- 边颜色深浅: FDR (padj)
- 边形状: 弯曲 bezier 弧线
- 布局: 模块内紧凑 + 模块间力导向
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

# ============================================================
# 1. 读取数据
# ============================================================
csv_path = "../raw_data/zenodo5242913_GLI3_KO_multiome_18d__GRN_meta_format(2).csv"
df = pd.read_csv(csv_path, index_col=0)

print(f"Total edges in CSV: {len(df)}")
print(f"Unique TFs: {df['tf'].nunique()}")
print(f"Unique targets: {df['target'].nunique()}")

# ============================================================
# 2. 过滤显著调控关系
# ============================================================
df_sig = df[df['padj'] < 0.05].copy()
print(f"\nSignificant edges (padj<0.05): {len(df_sig)}")

if len(df_sig) > 5000:
    df_sig['abs_estimate'] = df_sig['estimate'].abs()
    df_sig = df_sig.nlargest(5000, 'abs_estimate')
    print(f"Filtered to top 5000 by |estimate|")

print(f"Edges used for GRN: {len(df_sig)}")

# ============================================================
# 3. 构建 NetworkX 有向图 + 存储边属性
# ============================================================
G = nx.Graph()
edge_padj_map = {}    # (u,v) -> padj

all_genes = set(df_sig['tf'].unique()) | set(df_sig['target'].unique())
print(f"Total unique genes (nodes): {len(all_genes)}")

for gene in all_genes:
    G.add_node(gene)

for _, row in df_sig.iterrows():
    tf = row['tf']
    target = row['target']
    weight = abs(row['estimate'])
    G.add_edge(tf, target, weight=weight)
    edge_padj_map[(tf, target)] = row['padj']
    edge_padj_map[(target, tf)] = row['padj']

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ============================================================
# 4. 最大连通分量 + Louvain 社区检测
# ============================================================
components = list(nx.connected_components(G))
largest_cc = max(components, key=len)
G_main = G.subgraph(largest_cc).copy()
print(f"Main component: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges")

partition = community_louvain.best_partition(G_main, weight='weight')
num_modules = len(set(partition.values()))
print(f"Number of modules (Louvain): {num_modules}")

# ============================================================
# 5. PageRank — 决定节点大小
# ============================================================
pr = nx.pagerank(G_main, weight='weight')
pr_values = np.array([pr[n] for n in G_main.nodes()])
pr_min, pr_max = pr_values.min(), pr_values.max()
print(f"PageRank range: [{pr_min:.6f}, {pr_max:.6f}]")

# 映射到节点大小 [3, 80]
def pr_to_size(pagerank_val):
    log_val = np.log1p(pagerank_val * 100)  # 对数缩放避免 hub 过大
    log_vals = np.log1p(np.array(list(pr.values())) * 100)
    lo, hi = log_vals.min(), log_vals.max()
    normalized = (log_val - lo) / (hi - lo) if hi > lo else 0.5
    return 30 + normalized * 500  # [30, 530]

node_size_dict = {n: pr_to_size(pr[n]) for n in G_main.nodes()}

# ============================================================
# 6. 伪时序 — 基于调控层级深度 (决定节点颜色)
# ============================================================
# 找到 source TFs: 在数据中仅作为 tf 出现，不作为 target 的基因
tf_set = set(df_sig['tf'].unique())
target_set = set(df_sig['target'].unique())
source_tfs = tf_set - target_set  # 纯粹的上游调控因子
source_tfs_in_graph = source_tfs & set(G_main.nodes())
source_tfs_list = list(source_tfs_in_graph)
print(f"Source TFs (never target): {len(source_tfs_in_graph)} in graph")

# 构建有向图实现更精确的调控层级
G_dir = nx.DiGraph()
for _, row in df_sig.iterrows():
    if row['tf'] in G_main.nodes() and row['target'] in G_main.nodes():
        G_dir.add_edge(row['tf'], row['target'], weight=abs(row['estimate']))

# 伪时序 = 各节点到所有source TF的平均最短路径 (连续值)
pseudotime = {}
for node in G_main.nodes():
    total_dist = 0.0
    reachable = 0
    for src in source_tfs_list:
        try:
            d = nx.shortest_path_length(G_dir, source=src, target=node)
            total_dist += d
            reachable += 1
        except nx.NetworkXNoPath:
            pass
    if reachable > 0:
        pseudotime[node] = total_dist / reachable  # 平均距离 → 连续伪时序
    else:
        # 从反向可达的 TF 出发 (即 target 本身可能是上游)
        for src in source_tfs_list:
            try:
                d = nx.shortest_path_length(G_dir, source=node, target=src)
                total_dist += d
                reachable += 1
            except nx.NetworkXNoPath:
                pass
        pseudotime[node] = -total_dist / max(reachable, 1)  # 负值表示反向

# 处理可能的 nan
for n in pseudotime:
    if np.isnan(pseudotime[n]) or np.isinf(pseudotime[n]):
        pseudotime[n] = 0.0

pt_values = np.array([pseudotime[n] for n in G_main.nodes()])
# 裁剪极端值
pt_low = np.percentile(pt_values, 2)
pt_high = np.percentile(pt_values, 98)
pt_values_clipped = np.clip(pt_values, pt_low, pt_high)
pt_min, pt_max = pt_values_clipped.min(), pt_values_clipped.max()
print(f"Pseudotime range: [{pt_min:.2f}, {pt_max:.2f}] ({len(set([round(v,2) for v in pt_values]))} unique values)")

# 将 clipped pseudotime 存回去用于绘图
for i, node in enumerate(G_main.nodes()):
    pseudotime[node] = pt_values_clipped[i]

# ============================================================
# 7. 布局 — 全局紧凑 + 温和模块收紧
# ============================================================
rng = np.random.RandomState(42)

# 按 module 分组
nd_list = {}
for node in G_main.nodes():
    mod = partition[node]
    if mod not in nd_list:
        nd_list[mod] = []
    nd_list[mod].append(node)

# 全局 spring_layout — 保证网络整体连通感
pos = nx.spring_layout(
    G_main,
    k=0.015,
    iterations=800,
    seed=42,
    weight='weight',
    scale=1.0,
    threshold=1e-5,
)

# 温和的模块收紧: pull_factor 控制紧凑度 (比之前温和很多)
centroids = {}
for mod, nodes in nd_list.items():
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    centroids[mod] = np.array([np.mean(xs), np.mean(ys)])

pull_factor = 0.30  # 温和收紧，保留模块间连通感
for node in G_main.nodes():
    mod = partition[node]
    pos[node] = pos[node] * (1 - pull_factor) + centroids[mod] * pull_factor

# Step 3: 伪时序吸引+排斥 — 相近 depth 聚拢，相差远的分开
pt_norm = Normalize(vmin=pt_min, vmax=pt_max)
node_pt = {n: pt_norm(pseudotime[n]) for n in G_main.nodes()}  # [0,1]

node_list = list(G_main.nodes())
k_near = 8        # 吸引最近的 k 个
k_far = 6         # 排斥最远的 k 个
attract_strength = 0.18
repel_strength = 0.10

# 预计算每个节点的伪时序近邻和远邻
# 所有节点按伪时序排序
sorted_by_pt = sorted(node_list, key=lambda n: node_pt[n])
n_nodes = len(sorted_by_pt)

for _ in range(3):  # 3 轮平滑
    new_pos = {}
    for i, node in enumerate(sorted_by_pt):
        pt_val = node_pt[node]
        cur = pos[node]

        # 吸引: 伪时序窗口内最近的节点
        window_start = max(0, i - 15)
        window_end = min(n_nodes, i + 15)
        window_nodes = sorted_by_pt[window_start:window_end]
        near = sorted(window_nodes, key=lambda n: abs(node_pt[n] - pt_val))[1:k_near+1]
        avg_near = np.mean([pos[n] for n in near], axis=0)

        # 排斥: 伪时序最远的节点
        far = sorted(node_list, key=lambda n: -abs(node_pt[n] - pt_val))[:k_far]
        repel_vec = np.zeros(2)
        for fn in far:
            delta = cur - pos[fn]
            dist = np.linalg.norm(delta)
            if dist > 1e-6:
                repel_vec += delta / dist * (0.02 / (dist + 0.01))
        if len(far) > 0:
            repel_vec /= len(far)

        # 合成
        target = cur * (1 - attract_strength) + avg_near * attract_strength
        target += repel_vec * repel_strength
        new_pos[node] = target
    pos = new_pos

print(f"Layout computed for {len(pos)} nodes")

# ============================================================
# 8. 边 FDR 映射 — 决定边颜色深浅
# ============================================================
# 构建 graph edge order 对应的 padj 列表
edge_list = list(G_main.edges())
edge_padjs = []
for u, v in edge_list:
    padj = edge_padj_map.get((u, v), 1.0)
    if padj == 0:
        padj = 1e-320
    edge_padjs.append(padj)

edge_padjs = np.array(edge_padjs)
edge_logp = -np.log10(edge_padjs)
# 裁剪到 2%-98% 避免极端值主导
el_low = np.percentile(edge_logp, 2)
el_high = np.percentile(edge_logp, 98)
edge_logp_clipped = np.clip(edge_logp, el_low, el_high)
el_min, el_max = edge_logp_clipped.min(), edge_logp_clipped.max()
print(f"Edge -log10(padj) range (clipped): [{el_min:.1f}, {el_max:.1f}]")

# 映射到灰色深浅: 低 FDR (高 -log10(padj)) → 深灰, 高 FDR → 浅灰
def fdr_to_color(logp_val):
    """返回 (r, g, b, alpha) — logp_val 已是 clipped"""
    # 归一化到 [0,1], 0=最不显著(浅), 1=最显著(深)
    if el_max > el_min:
        t = (logp_val - el_min) / (el_max - el_min)
    else:
        t = 0.5
    # 映射灰度: t=1 → #1a1a1a (很深), t=0 → #888 (中灰)
    gray = int(0x88 - t * (0x88 - 0x1a))
    gray = np.clip(gray, 0, 255) / 255.0
    alpha = 0.45 + t * 0.40  # 显著边更不透明
    return (gray, gray, gray, alpha)

# ============================================================
# 9. 绘制 GRN 图
# ============================================================
fig, ax = plt.subplots(figsize=(16, 16))
ax.set_facecolor('#fdfcf9')

# 9a. 绘制边 — 弯曲弧线，深浅由 FDR 决定
def curved_edge_segments(pos, edges, logp_vals, rad=0.28, n_pts=25):
    """生成弯曲边线段和对应的颜色"""
    segments = []
    colors = []
    for idx, (u, v) in enumerate(edges):
        p0 = np.array(pos[u])
        p1 = np.array(pos[v])
        mid = (p0 + p1) / 2
        d = p1 - p0
        perp = np.array([-d[1], d[0]])
        norm = np.linalg.norm(perp)
        if norm > 1e-8:
            perp = perp / norm * rad * np.linalg.norm(d)
        ctrl = mid + perp
        t = np.linspace(0, 1, n_pts)[:, np.newaxis]
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
        segments.append(pts)
        # FDR 颜色
        r, g, b, a = fdr_to_color(logp_vals[idx])
        colors.append((r, g, b, a))
    return segments, colors

edge_segs, edge_cols = curved_edge_segments(pos, edge_list, edge_logp_clipped, rad=0.16)
lc = LineCollection(edge_segs, colors=edge_cols, linewidths=0.25)
ax.add_collection(lc)

# 9b. 绘制节点 — 大小由 PageRank 决定，颜色由伪时序决定
# 伪时序 colormap: 早期→晚期 (黄→红→紫)
pseudotime_cmap = plt.cm.plasma

# 所有节点按伪时序排序绘制 (画出 depth 渐变)
all_nodes_list = list(G_main.nodes())
pt_arr = np.array([pseudotime[n] for n in all_nodes_list])
norm_pt = Normalize(vmin=pt_min, vmax=pt_max)

node_colors = [pseudotime_cmap(norm_pt(pseudotime[n])) for n in all_nodes_list]
node_sizes = [node_size_dict[n] for n in all_nodes_list]

nx.draw_networkx_nodes(
    G_main, pos,
    nodelist=all_nodes_list,
    node_size=node_sizes,
    node_color=node_colors,
    alpha=0.90,
    linewidths=0.3,
    edgecolors='#555555',
    ax=ax,
)

# 9c. 标注 top PageRank 基因
pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
top_pr_genes = pr_sorted[:15]
label_dict = {g: g for g, _ in top_pr_genes}

nx.draw_networkx_labels(
    G_main, pos,
    labels=label_dict,
    font_size=15,
    font_color='#222222',
    font_weight='bold',
    ax=ax,
)

# 9d. Colorbar — 伪时序
sm = ScalarMappable(norm=norm_pt, cmap=pseudotime_cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
cbar.set_label('Pseudotime (regulatory depth)', fontsize=9, color='#555555')
cbar.ax.tick_params(labelsize=7, colors='#555555')

# 9e. FDR 图例 (edge significance)
fdr_patches = [
    Patch(facecolor='#888888', edgecolor='#555555', label=f'Low sig (padj~{10**(-el_min):.0e})'),
    Patch(facecolor='#4a4a4a', edgecolor='#555555', label=f'Mid sig'),
    Patch(facecolor='#1a1a1a', edgecolor='#555555', label=f'High sig (padj~{10**(-el_max):.0e})'),
]
legend_fdr = ax.legend(handles=fdr_patches, loc='upper right', fontsize=8,
                        title='Edge FDR', title_fontsize=9, framealpha=0.85)

# 9f. 标题
ax.set_title(
    f'GRN: GLI3 KO multiome 18d\n'
    f'{G_main.number_of_nodes()} genes | {G_main.number_of_edges()} edges | '
    f'Nodes: PageRank size, Pseudotime color | Edges: FDR intensity',
    fontsize=10, fontweight='bold', color='#444444'
)
ax.axis('off')

plt.tight_layout()
output_path = '../figures/grn_gli3_ko.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#fdfcf9')
plt.close()
print(f"\nSaved GRN plot to: {output_path}")

# ============================================================
# 10. 统计信息
# ============================================================
print("\n===== GRN Statistics =====")
print(f"Nodes: {G_main.number_of_nodes()}  |  Edges: {G_main.number_of_edges()}  |  Modules: {num_modules}")
print(f"Pseudotime range: [{pt_min}, {pt_max}]")
print(f"Source TFs: {len(source_tfs_in_graph)}")

print("\nTop 15 genes by PageRank:")
for gene, val in top_pr_genes[:15]:
    is_tf = "TF" if gene in tf_set else "target"
    pt_val = pseudotime[gene]
    print(f"  {gene:12s} [{is_tf:6s}]  PR={val:.6f}  pseudotime={pt_val}")

print("\n===== Done =====")
