"""
GLI3.py — GLI3 调控图 (人类数据, 三种来源各一张)
================================================================
数据来源:
  1. example_data/glm_network_coefs_human.csv
     增强子-GLM 回归系数表 (tf, target, region, term, estimate, std_err,
     statistic, pval, padj, corr)。
     GLI3 作为 TF 共 302 个候选靶基因 / 2646 条 (靶, region) 记录。
  2. example_data/motif2tf_human.tsv
     Motif2TF 表 (tf, motif_tf, motif, name, prc_similarity, collection, ...)。
     语义: 基因 tf 的调控序列中含有 motif_tf 的 motif (即 motif_tf -> tf)。
     表中 GLI3 只以 motif 拥有者 (motif_tf) 身份出现, 对应 motif MA1491.1 (CORE)。
  3. raw_data/TF_gene_network.csv
     预过滤的 TF-靶基因网络 (tf, target, estimate, n_regions, regions,
     pval_min, padj_min, mixed_region_signs, n_genes, n_tfs, direction)。
     GLI3 作为 TF 共 134 个靶基因 (63 positive / 71 negative);
     另有 18 条「上游 TF -> GLI3」记录 (MECOM/SOX21/GSX2 等), 只有本来源有。
     注意: 该表列名与旧版不同 (padj_min/pval_min, 无 pval/padj), 载入时统一改名。

三份数据各画一张 GLI3 圆形调控树图, 画风与 scripts/phase4_plot_results.py 一致:
  - 边表统一为 temp_tree_data/*.csv (tf, target, estimate)
  - 调用 draw_grn_tree.R (igraph circular tree + ggraph) 出图
  - 同一配色: root 深蓝 #2C3E50, 重点基因红点 #E74C3C,
    边 activation=#6495ED / repression=#FA8072, ggrepel 外移重点基因标签
  - 同一分层策略: root -> 一级靶 (L1) -> hub TF 下游 (L2); 重点基因直连/两跳强制入选

用法: python GLI3.py
输出: figures/gli3_grn_glm.png
      figures/gli3_grn_motif.png
      figures/gli3_grn_tfnet.png
      temp_tree_data/gli3_glm_focus.csv
      temp_tree_data/gli3_motif_focus.csv
      temp_tree_data/gli3_tfnet_focus.csv
      outputs/GLI3_report.txt
"""

import os
import subprocess

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import grn_common as G

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = G.ROOT_DIR
PNG_DIR    = os.path.join(ROOT_DIR, 'figures')
TEMP_DIR   = os.path.join(ROOT_DIR, 'temp_tree_data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
R_SCRIPT   = os.path.join(ROOT_DIR, 'draw_grn_tree.R')
GLM_COEFS  = os.path.join(ROOT_DIR, 'example_data', 'glm_network_coefs_human.csv')
MOTIF2TF   = os.path.join(ROOT_DIR, 'example_data', 'motif2tf_human.tsv')
TF_NETWORK = os.path.join(ROOT_DIR, 'raw_data', 'TF_gene_network.csv')

for d in (PNG_DIR, TEMP_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

C1, C2, C3 = '#4C72B0', '#55A868', '#C44E52'
ROOT_GENE = 'GLI3'

# GLI3 重点展示基因 (人类符号; 与 phase4 的 GLI3_WATCHLIST 同一来源, 去掉涡虫专有项)
# 只有真正出现在边表里的基因会被红色高亮 (draw_grn_tree.R 自动过滤)
GLI3_WATCHLIST = [
    'ZIC3', 'ZNF608', 'CUX1', 'HES1', 'JARID2', 'MAF', 'PBX1', 'SIX6',
    'SOX11', 'EMX2', 'TOX', 'ZIC2', 'SIX3', 'SMAD3', 'SOX4', 'SALL4',
    'DACH1', 'RAX', 'SOX6', 'ZNF521', 'PAX6', 'CREB5', 'DMRTA1', 'HES4',
]


# ================================================================
# 公共: 边表写出 + Rscript 调用 (与 phase4_plot_results.py 一致)
# ================================================================

def _write_edges(edges, out_path):
    edges = edges[['tf', 'target', 'estimate']].drop_duplicates(['tf', 'target'])
    edges.to_csv(out_path, index=False)
    n_l1 = int((edges['tf'] == ROOT_GENE).sum())
    print(f'  边 CSV: {len(edges)} 条 (L1={n_l1}, 其余={len(edges) - n_l1}) -> {out_path}')
    return out_path


def _run_r_tree(edges_csv, root, output_png, n1=60, nhubs=6, n2=25,
                highlight=None, spread_labels=True, label_all=True):
    """调用 draw_grn_tree.R 生成圆形 GRN 树图 (参数与 phase4 完全一致)

    label_all=True: 图中所有基因都显示名字 (不只是重点基因)
    """
    cmd = ['Rscript', R_SCRIPT,
           '--edges', edges_csv, '--root', root, '--output', output_png,
           '--n1', str(n1), '--nhubs', str(nhubs), '--n2', str(n2)]
    if highlight:
        cmd += ['--highlight', ','.join(highlight)]
    if spread_labels:
        cmd += ['--spread-labels']
    if label_all:
        cmd += ['--label-all']
    print(f'  Rscript ... --root {root} -> {os.path.basename(output_png)}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f'  R STDERR: {result.stderr[:800]}')
        return False
    for line in result.stdout.strip().splitlines():
        print(f'    {line}')
    return True


def _best_per_edge(df, key=('tf', 'target')):
    """每个 (tf, target) 只保留 |estimate| 最大的一条 (跨 region 聚合)"""
    if df.empty:
        return df
    d = df.copy()
    d['abs_estimate'] = d['estimate'].abs()
    idx = d.groupby(list(key))['abs_estimate'].idxmax()
    return d.loc[idx].reset_index(drop=True)


# ================================================================
# 来源 1: GLM 网络系数 (glm_network_coefs_human.csv)
# ================================================================

def _load_glm():
    df = pd.read_csv(GLM_COEFS)
    need = {'tf', 'target', 'region', 'estimate', 'padj'}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f'错误: {GLM_COEFS} 缺少列 {sorted(miss)}')
    df = df.dropna(subset=['tf', 'target', 'estimate'])
    df['tf'] = df['tf'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    print(f'GLM 系数表: {len(df):,} 行, {df.tf.nunique()} 个 TF, '
          f'{df.target.nunique()} 个靶基因, GLI3 行数 {(df.tf == ROOT_GENE).sum():,}')
    return df


def _build_glm_tree(glm, top_n1=60, n_hubs=6, max_l2=25, padj_max=0.05, watchlist=None):
    """GLI3 (作为 TF) -> 靶基因 的树

      L1: padj<=padj_max 的靶按 |estimate| 取前 top_n1;
          重点基因若是直连靶 -> 强制入 L1 (直接调控);
          否则取 GLI3->中介TF->重点基因 中 path_score=min(|w1|,|w2|) 最高的两跳 (间接调控)
      L2: 一级靶中本身是 TF 的 hub, 各取 |estimate| 最大的 max_l2 个下游 (扣除已在 L1 的节点);
          两跳中介 TF 强制作为 hub
    """
    best = _best_per_edge(glm)
    gl = best[best.tf == ROOT_GENE].copy()
    if gl.empty:
        raise SystemExit(f'错误: {GLM_COEFS} 中没有 {ROOT_GENE} 作为 TF 的记录')

    gl = gl.sort_values('abs_estimate', ascending=False)
    gl_w = dict(zip(gl.target, gl.estimate))       # 每个靶取 |w| 最大那条的带符号权重
    tf_set = set(best.tf.unique())

    sig = gl[gl.padj <= padj_max]
    l1 = sig.head(top_n1)
    l1_set = set(l1.target)

    rows = [{'tf': ROOT_GENE, 'target': r.target, 'estimate': float(r.estimate)}
            for _, r in l1.iterrows()]

    # ── 重点基因: 直连优先, 否则最优两跳 ──
    status, forced_mids = {}, []
    for g in watchlist or []:
        if g in gl_w:                                    # 直接调控 (即使排在 top_n1 之外也强制入图)
            if g not in l1_set:
                rows.append({'tf': ROOT_GENE, 'target': g, 'estimate': float(gl_w[g])})
                l1_set.add(g)
            status[g] = ('direct', float(gl_w[g]))
            continue
        if g not in set(best.target):
            status[g] = ('absent', 'GLM 表中既不是 GLI3 靶, 也未被任何 TF 调控')
            continue
        bestp = None
        for _, r in best[best.target == g].iterrows():
            if r.tf in gl_w:                             # 中介必须是 GLI3 的直接靶
                score = min(abs(gl_w[r.tf]), abs(r.estimate))
                if bestp is None or score > bestp[0]:
                    bestp = (score, r.tf, float(gl_w[r.tf]), float(r.estimate))
        if bestp:
            _, mid, w1, w2 = bestp
            rows.append({'tf': ROOT_GENE, 'target': mid, 'estimate': w1})
            rows.append({'tf': mid, 'target': g, 'estimate': w2})
            l1_set.add(mid)
            forced_mids.append(mid)
            status[g] = ('indirect', mid, w1, w2)
        else:
            status[g] = ('orphan', 'GLM 表中存在, 但与 GLI3 无直连/两跳路径')

    # ── L2: hub TF -> 下游靶 ──
    l1_nodes = [r['target'] for r in rows if r['tf'] == ROOT_GENE]
    hubs = [t for t in dict.fromkeys(l1_nodes) if t in tf_set]
    hubs = sorted(hubs, key=lambda t: -abs(gl_w.get(t, 0.0)))[:n_hubs]
    hubs = list(dict.fromkeys(hubs + [m for m in forced_mids if m in tf_set]))

    n_l2_edges = 0
    for h in hubs:
        sub = best[(best.tf == h) & (~best.target.isin(l1_set)) & (best.target != h)]
        sub = sub.sort_values('abs_estimate', ascending=False).head(max_l2)
        for _, r in sub.iterrows():
            rows.append({'tf': h, 'target': r.target, 'estimate': float(r.estimate)})
            n_l2_edges += 1

    edges = pd.DataFrame(rows, columns=['tf', 'target', 'estimate'])
    meta = dict(n_records=int((glm.tf == ROOT_GENE).sum()), n_targets=len(gl),
                n_sig=int(len(sig)), n_l1=len(l1_set), n_hubs=len(hubs),
                n_l2=n_l2_edges, hubs=hubs)
    return edges, status, meta


# ================================================================
# 来源 2: Motif2TF (motif2tf_human.tsv)
# ================================================================

def _load_motif2tf():
    m = pd.read_csv(MOTIF2TF, sep='\t')
    need = {'tf', 'motif_tf', 'motif', 'prc_similarity'}
    miss = need - set(m.columns)
    if miss:
        raise SystemExit(f'错误: {MOTIF2TF} 缺少列 {sorted(miss)}')
    m = m.dropna(subset=['tf', 'motif_tf', 'prc_similarity'])
    m['tf'] = m['tf'].astype(str).str.strip()
    m['motif_tf'] = m['motif_tf'].astype(str).str.strip()
    print(f'Motif2TF 表: {len(m):,} 行, {m.motif_tf.nunique()} 个 motif 拥有者, '
          f'{m.tf.nunique()} 个基因, {m.motif.nunique()} 个 motif')
    return m


def _build_motif_tree(m2t, max_l2=10, watchlist=None):
    """GLI3 motif 的调控树

      L1: motif_tf == GLI3 的行 -> GLI3 motif (MA1491.1) 出现在哪些基因的调控序列中,
          权重 = prc_similarity
      L2: 若某个一级靶本身也是 motif 拥有者 (出现在 motif_tf 列), 继续展开其下游。
          motif2tf 表里 GLI3 的 5 个一级靶都不作为 motif 拥有者出现, 因此该来源
          对 GLI3 天然只有一层 (星形); 这里不做任何跨表补边。
    """
    m = m2t[m2t.motif_tf == ROOT_GENE].copy()
    if m.empty:
        raise SystemExit(f'错误: {MOTIF2TF} 中没有 {ROOT_GENE} 作为 motif 拥有者的记录')

    motif_names = sorted(m['motif'].astype(str).unique())
    coll = sorted(m['collection'].astype(str).unique()) if 'collection' in m.columns else []

    m = m.sort_values('prc_similarity', ascending=False).drop_duplicates('tf')
    rows = [{'tf': ROOT_GENE, 'target': r.tf, 'estimate': float(r.prc_similarity)}
            for _, r in m.iterrows()]
    l1 = [r['target'] for r in rows]

    n_l2_edges, down_owners = 0, []
    for t in l1:
        sub = m2t[m2t.motif_tf == t].sort_values('prc_similarity', ascending=False).head(max_l2)
        if len(sub):
            down_owners.append(t)
        for _, r in sub.iterrows():
            rows.append({'tf': t, 'target': r.tf, 'estimate': float(r.prc_similarity)})
            n_l2_edges += 1

    edges = pd.DataFrame(rows, columns=['tf', 'target', 'estimate'])
    meta = dict(n_targets=int(m.tf.nunique()), n_l1=len(l1), n_l2=n_l2_edges,
                motifs=motif_names, collections=coll, l1=l1, down_owners=down_owners)

    # 重点基因状态 (只做存在性陈述, 不补边)
    status = {}
    for g in watchlist or []:
        if g in l1:
            status[g] = ('direct', float(m.loc[m.tf == g, 'prc_similarity'].iloc[0]))
        elif (m2t.tf == g).any():
            status[g] = ('no_gli3_motif', '基因在表中, 但调控序列不含 GLI3 motif')
        else:
            status[g] = ('absent', 'Motif2TF 表中无此基因')
    return edges, status, meta


# ================================================================
# 来源 3: 预过滤 TF-靶基因网络 (raw_data/TF_gene_network.csv)
# ================================================================

def _load_tf_network():
    """载入 TF_gene_network.csv

    列名与另两个来源不同:
      pval_min / padj_min  (旧版文件为 pval / padj)
      estimate 为带符号权重, direction 列冗余记载其正负 (已验证 49764 行全部一致)
    """
    df = pd.read_csv(TF_NETWORK)

    # 列名归一化: 新版 padj_min/pval_min -> padj/pval, 旧版保持不变
    rename = {'padj_min': 'padj', 'pval_min': 'pval'}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    need = {'tf', 'target', 'estimate', 'padj'}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f'错误: {TF_NETWORK} 缺少列 {sorted(miss)}')

    df = df.dropna(subset=['tf', 'target', 'estimate'])
    df['tf'] = df['tf'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()

    # 自环剔除 (TF 调控自身, 树图里没有意义)
    n_self = int((df.tf == df.target).sum())
    if n_self:
        df = df[df.tf != df.target]
        print(f'  已剔除自环 {n_self} 条')

    n_down = int((df.tf == ROOT_GENE).sum())
    n_up = int((df.target == ROOT_GENE).sum())
    print(f'TF-靶网络: {len(df):,} 条边, {df.tf.nunique()} 个 TF, '
          f'{df.target.nunique()} 个靶基因')
    print(f'  GLI3 作为 TF (下游): {n_down} 条; GLI3 作为靶 (上游): {n_up} 条')
    return df


def _build_tfnet_tree(net, top_n1=60, n_hubs=6, max_l2=25, padj_max=0.05, watchlist=None):
    """GLI3 (作为 TF) -> 靶基因 的树, 数据来自 TF_gene_network.csv

      L1: padj<=padj_max 的靶按 |estimate| 取前 top_n1;
          重点基因直连则强制入 L1, 否则取 GLI3->中介TF->重点基因 中
          path_score=min(|w1|,|w2|) 最高的两跳 (间接调控)
      L2: 一级靶中本身是 TF 的 hub, 各取 |estimate| 最大的 max_l2 个下游

    注意: 与 GLM 来源不同, 本表已是预过滤结果 —— 134 个靶全部 padj<=0.0488,
    因此 padj_max=0.05 实际上不过滤任何东西, L1 完全由 |estimate| 决定。
    """
    net = net.drop_duplicates(['tf', 'target'])
    gl = net[net.tf == ROOT_GENE].copy()
    if gl.empty:
        raise SystemExit(f'错误: {TF_NETWORK} 中没有 {ROOT_GENE} 作为 TF 的记录')

    gl = gl.sort_values('estimate', key=lambda s: s.abs(), ascending=False)
    gl_w = dict(zip(gl.target, gl.estimate))       # 本表每个 (tf,target) 唯一, 直接取
    tf_set = set(net.tf.unique())

    sig = gl[gl.padj <= padj_max] if gl.padj.notna().any() else gl
    l1 = sig.head(top_n1)
    l1_set = set(l1.target)

    rows = [{'tf': ROOT_GENE, 'target': r.target, 'estimate': float(r.estimate)}
            for _, r in l1.iterrows()]

    # ── 重点基因: 直连优先, 否则最优两跳 ──
    status, forced_mids = {}, []
    for g in watchlist or []:
        if g in gl_w:                                    # 直接调控
            if g not in l1_set:
                rows.append({'tf': ROOT_GENE, 'target': g, 'estimate': float(gl_w[g])})
                l1_set.add(g)
            status[g] = ('direct', float(gl_w[g]))
            continue
        if g not in set(net.target):
            status[g] = ('absent', 'TF 网络中既不是 GLI3 靶, 也未被任何 TF 调控')
            continue
        # 中介 TF 必须是 GLI3 的直接靶 (gl_w 的键)
        cand = net[(net.target == g) & (net.tf.isin(gl_w.keys()))]
        bestp = None
        for _, r in cand.iterrows():
            score = min(abs(gl_w[r.tf]), abs(r.estimate))
            if bestp is None or score > bestp[0]:
                bestp = (score, r.tf, float(gl_w[r.tf]), float(r.estimate))
        if bestp:
            _, mid, w1, w2 = bestp
            rows.append({'tf': ROOT_GENE, 'target': mid, 'estimate': w1})
            rows.append({'tf': mid, 'target': g, 'estimate': w2})
            l1_set.add(mid)
            forced_mids.append(mid)
            status[g] = ('indirect', mid, w1, w2)
        else:
            status[g] = ('orphan', 'TF 网络中存在, 但与 GLI3 无直连/两跳路径')

    # ── L2: hub TF -> 下游靶 ──
    l1_nodes = [r['target'] for r in rows if r['tf'] == ROOT_GENE]
    hubs = [t for t in dict.fromkeys(l1_nodes) if t in tf_set]
    hubs = sorted(hubs, key=lambda t: -abs(gl_w.get(t, 0.0)))[:n_hubs]
    hubs = list(dict.fromkeys(hubs + [m for m in forced_mids if m in tf_set]))

    n_l2_edges = 0
    for h in hubs:
        sub = net[(net.tf == h) & (~net.target.isin(l1_set)) & (net.target != h)]
        sub = sub.reindex(sub.estimate.abs().sort_values(ascending=False).index).head(max_l2)
        for _, r in sub.iterrows():
            rows.append({'tf': h, 'target': r.target, 'estimate': float(r.estimate)})
            n_l2_edges += 1

    edges = pd.DataFrame(rows, columns=['tf', 'target', 'estimate'])
    meta = dict(n_records=int(len(gl)), n_targets=int(gl.target.nunique()),
                n_sig=int(len(sig)), n_l1=len(l1_set), n_hubs=len(hubs),
                n_l2=n_l2_edges, hubs=hubs,
                n_pos=int((gl.direction == 'positive').sum()) if 'direction' in gl else None,
                n_neg=int((gl.direction == 'negative').sum()) if 'direction' in gl else None,
                n_upstream=int((net.target == ROOT_GENE).sum()))
    return edges, status, meta


def _tfnet_upstream(net, top_n=20):
    """上游: 调控 GLI3 的 TF (GLI3 作为靶基因)。GLM / motif2tf 两个来源都没有这个方向。"""
    up = net[net.target == ROOT_GENE].copy()
    if up.empty:
        return up
    up = up.reindex(up.estimate.abs().sort_values(ascending=False).index)
    return up.head(top_n)




def _write_report(path, glm_meta, glm_status, motif_meta, motif_status,
                  tfnet_meta, tfnet_status, tfnet_upstream):
    lines = ['=' * 72,
             'GLI3 调控图报告 (人类) — 三种数据来源',
             '=' * 72,
             '',
             '一、GLM 网络系数 (glm_network_coefs_human.csv, GLI3 作为 TF)',
             f'  区域-靶记录 : {glm_meta["n_records"]:,}',
             f'  候选靶基因  : {glm_meta["n_targets"]}',
             f'  显著靶(padj<=0.05): {glm_meta["n_sig"]}',
             f'  图中 L1 节点: {glm_meta["n_l1"]}   L2 hubs: {glm_meta["n_hubs"]}   '
             f'L2 边: {glm_meta["n_l2"]}',
             f'  L2 hubs     : {", ".join(glm_meta["hubs"]) or "-"}',
             '',
             '  重点基因状态 (direct=GLI3 直连; indirect=GLI3->中介->基因):']
    for g, st in glm_status.items():
        if st[0] == 'direct':
            lines.append(f'    {g:<8} direct    GLI3 -> {g}  w={st[1]:+.4f}')
        elif st[0] == 'indirect':
            lines.append(f'    {g:<8} indirect  GLI3 -> {st[1]} (w={st[2]:+.4f}) -> {g} (w={st[3]:+.4f})')
        else:
            lines.append(f'    {g:<8} {st[0]:<9} {st[1]}')
    lines += ['',
              '二、Motif2TF (motif2tf_human.tsv, GLI3 作为 motif 拥有者)',
              f'  motif      : {", ".join(motif_meta["motifs"]) or "-"}',
              f'  collection : {", ".join(motif_meta["collections"]) or "-"}',
              f'  一级靶 (该 motif 命中的 TF): {motif_meta["n_l1"]} 个 — '
              f'{", ".join(motif_meta["l1"])}',
              f'  二级边     : {motif_meta["n_l2"]}']
    if motif_meta['n_l2'] == 0:
        lines.append('  说明: motif2tf 表中这些一级靶不作为 motif 拥有者出现, '
                     '该来源对 GLI3 只有一层 (星形), 未跨表补边。')
    lines += ['', '  重点基因状态:']
    for g, st in motif_status.items():
        if st[0] == 'direct':
            lines.append(f'    {g:<8} direct    GLI3 motif -> {g}  sim={st[1]:.4f}')
        else:
            lines.append(f'    {g:<8} {st[0]:<13} {st[1]}')

    # ── 三、TF 网络 (raw_data/TF_gene_network.csv) ──
    lines += ['',
              '三、TF-靶基因网络 (raw_data/TF_gene_network.csv, GLI3 作为 TF)',
              f'  GLI3 靶基因 : {tfnet_meta["n_targets"]}',
              f'  padj<=0.05  : {tfnet_meta["n_sig"]}']
    if tfnet_meta['n_pos'] is not None:
        lines.append(f'  调控方向    : positive={tfnet_meta["n_pos"]}  '
                     f'negative={tfnet_meta["n_neg"]}')
    lines += [f'  图中 L1 节点: {tfnet_meta["n_l1"]}   L2 hubs: {tfnet_meta["n_hubs"]}   '
              f'L2 边: {tfnet_meta["n_l2"]}',
              f'  L2 hubs     : {", ".join(tfnet_meta["hubs"]) or "-"}']
    all_sig = tfnet_meta['n_sig'] == tfnet_meta['n_targets'] and tfnet_meta['n_targets'] > 0
    if all_sig:
        lines.append('  说明: 该表为预过滤结果, 全部靶基因 padj<=0.05 (max 0.0488), '
                     '故 padj<=0.05 不构成筛选, L1 由 |estimate| 决定。')
    lines += ['', '  重点基因状态:']
    for g, st in tfnet_status.items():
        if st[0] == 'direct':
            lines.append(f'    {g:<8} direct    GLI3 -> {g}  w={st[1]:+.4f}')
        elif st[0] == 'indirect':
            lines.append(f'    {g:<8} indirect  GLI3 -> {st[1]} (w={st[2]:+.4f}) -> {g} (w={st[3]:+.4f})')
        else:
            lines.append(f'    {g:<8} {st[0]:<9} {st[1]}')

    lines += ['',
              f'  上游调控 (谁调控 GLI3): {tfnet_meta["n_upstream"]} 个 TF — '
              'GLM 与 motif2tf 两个来源均无此方向, 未画图, 仅列出:']
    if len(tfnet_upstream):
        for _, r in tfnet_upstream.iterrows():
            padj = r.get('padj', float('nan'))
            lines.append(f'    {r.tf:<10} w={r.estimate:+.4f}  padj={padj:.3g}  '
                         f'{r.get("direction", "")}')
    else:
        lines.append('    (无)')

    report = '\n'.join(lines)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(report)


# ================================================================
# 主函数
# ================================================================

def main():
    # ====== Figure 1: GLM 网络系数 -> GLI3 圆形调控树 ======
    print('\n' + '=' * 72)
    print('Figure 1: GLM network coefs (human) — GLI3')
    print('=' * 72)
    glm = _load_glm()
    glm_edges, glm_status, glm_meta = _build_glm_tree(
        glm, top_n1=60, n_hubs=6, max_l2=25, padj_max=0.05, watchlist=GLI3_WATCHLIST)
    print(f'  一级靶 {glm_meta["n_l1"]} 个, hub {glm_meta["n_hubs"]} 个 '
          f'({", ".join(glm_meta["hubs"]) or "-"}), L2 边 {glm_meta["n_l2"]} 条')
    glm_csv = _write_edges(glm_edges, os.path.join(TEMP_DIR, 'gli3_glm_focus.csv'))
    glm_png = os.path.join(PNG_DIR, 'gli3_grn_glm.png')
    ok1 = _run_r_tree(glm_csv, ROOT_GENE, glm_png, n1=60, nhubs=6, n2=25,
                      highlight=GLI3_WATCHLIST, spread_labels=True)
    print(f'  Figure 1: {"OK" if ok1 else "FAILED"} -> {glm_png}')

    # ====== Figure 2: Motif2TF -> GLI3 motif 调控树 ======
    print('\n' + '=' * 72)
    print('Figure 2: Motif2TF (human) — GLI3 motif')
    print('=' * 72)
    m2t = _load_motif2tf()
    motif_edges, motif_status, motif_meta = _build_motif_tree(
        m2t, max_l2=10, watchlist=GLI3_WATCHLIST)
    print(f'  motif {", ".join(motif_meta["motifs"])}, 一级靶 {motif_meta["n_l1"]} 个, '
          f'二级边 {motif_meta["n_l2"]} 条')
    motif_csv = _write_edges(motif_edges, os.path.join(TEMP_DIR, 'gli3_motif_focus.csv'))
    motif_png = os.path.join(PNG_DIR, 'gli3_grn_motif.png')
    ok2 = _run_r_tree(motif_csv, ROOT_GENE, motif_png, n1=20, nhubs=3, n2=10,
                      highlight=motif_meta['l1'], spread_labels=True)
    print(f'  Figure 2: {"OK" if ok2 else "FAILED"} -> {motif_png}')

    # ====== Figure 3: TF_gene_network -> GLI3 圆形调控树 ======
    print('\n' + '=' * 72)
    print('Figure 3: TF-gene network (raw_data/TF_gene_network.csv) — GLI3')
    print('=' * 72)
    net = _load_tf_network()
    tfnet_edges, tfnet_status, tfnet_meta = _build_tfnet_tree(
        net, top_n1=60, n_hubs=6, max_l2=25, padj_max=0.05, watchlist=GLI3_WATCHLIST)
    tfnet_upstream = _tfnet_upstream(net, top_n=20)
    print(f'  一级靶 {tfnet_meta["n_l1"]} 个, hub {tfnet_meta["n_hubs"]} 个 '
          f'({", ".join(tfnet_meta["hubs"]) or "-"}), L2 边 {tfnet_meta["n_l2"]} 条')
    tfnet_csv = _write_edges(tfnet_edges, os.path.join(TEMP_DIR, 'gli3_tfnet_focus.csv'))
    tfnet_png = os.path.join(PNG_DIR, 'gli3_grn_tfnet.png')
    ok3 = _run_r_tree(tfnet_csv, ROOT_GENE, tfnet_png, n1=60, nhubs=6, n2=25,
                      highlight=GLI3_WATCHLIST, spread_labels=True)
    print(f'  Figure 3: {"OK" if ok3 else "FAILED"} -> {tfnet_png}')

    # ====== 报告 ======
    report = os.path.join(OUTPUT_DIR, 'GLI3_report.txt')
    _write_report(report, glm_meta, glm_status, motif_meta, motif_status,
                  tfnet_meta, tfnet_status, tfnet_upstream)
    print(f'\nDone.\n  {glm_png}\n  {motif_png}\n  {tfnet_png}\n  {report}')

    if not (ok1 and ok2 and ok3):
        raise SystemExit('至少一张图生成失败, 请查看上面的 R STDERR')


if __name__ == '__main__':
    main()
