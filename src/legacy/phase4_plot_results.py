"""
phase4_plot_results.py — 全流程结果可视化
================================================================
绘制 Phase 1~3 的分析图表 + GLI3 GRN 树图 (R ggraph 管道)

用法: python phase4_plot_results.py
"""

import os, pickle, re, json, subprocess, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import grn_common as G

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = G.ROOT_DIR
PHASE1_DIR = G.PHASE1_DIR
PHASE2_DIR = G.PHASE2_DIR
PHASE3_DIR = G.PHASE3_DIR
PNG_DIR     = os.path.join(ROOT_DIR, 'figures')
RAW_DIR     = G.RAW_DIR
MAP_DIR     = G.MAP_DIR
TEMP_DIR    = os.path.join(ROOT_DIR, 'temp_tree_data')
R_SCRIPT    = os.path.join(ROOT_DIR, 'draw_grn_tree.R')
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

C1, C2, C3 = '#4C72B0', '#55A868', '#C44E52'

# GLI3 树图重点展示基因 (单一来源: grn_common.DEFAULT_WATCHLIST, 已去重)
GLI3_WATCHLIST = list(G.DEFAULT_WATCHLIST)


def _load_prediction_manifest():
    """读取 phase3 completeness manifest; 不存在或 PARTIAL 时给出显式警告。"""
    mp = os.path.join(PHASE3_DIR, 'prediction_manifest.json')
    if not os.path.exists(mp):
        print('⚠ 未找到 prediction_manifest.json — new_edge.txt 无法验证完整性 '
              '(可能是旧版 phase3 生成的)。建议重跑 phase3 以获得可审计结果。')
        return None
    with open(mp, encoding='utf-8') as f:
        man = json.load(f)
    if man.get('status') != 'COMPLETE':
        raise SystemExit(
            f"错误: phase3 结果为 {man.get('status')} "
            f"(completed {man.get('completed_sources')}/{man.get('expected_sources')} sources)。"
            f"\nnew_edge.txt 不完整 (可能被 --max-pairs 或中断截断), 绘图只会误导。"
            f"\n请重跑: python phase3_edge_predict.py  (不要带 --max-pairs)")
    print(f"phase3 manifest: COMPLETE — {man.get('completed_sources')} sources, "
          f"{man.get('written_rows'):,} 行")
    return man


# ================================================================
# Phase 1-3 概览图表 (保持不变)
# ================================================================

def plot_phase3_edge_stats(ax):
    edge_file = os.path.join(PHASE3_DIR, 'new_edge.txt')
    if not os.path.exists(edge_file):
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes); return
    df = pd.read_csv(edge_file, sep='\t', nrows=200000)
    w = df['weight']
    lo, hi = np.percentile(w, 1), np.percentile(w, 99)
    w_clip = w[(w >= lo) & (w <= hi)]
    ax.hist(w_clip[w_clip > 0], bins=40, alpha=0.7, color=C3, label=f'pos({(w>0).sum():,})', edgecolor='white')
    ax.hist(w_clip[w_clip < 0], bins=40, alpha=0.7, color=C1, label=f'neg({(w<0).sum():,})', edgecolor='white')
    ax.axvline(0, color='black', ls='--', lw=0.8)
    ax.set_xlabel('weight'); ax.set_ylabel('count')
    ax.set_title('Phase3: new_edge weight distribution (first 200k)')
    ax.legend(fontsize=8)


def plot_phase3_edge_stats_full(ax):
    edge_file = os.path.join(PHASE3_DIR, 'new_edge.txt')
    if not os.path.exists(edge_file):
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes); return
    try:    df = pd.read_csv(edge_file, sep='\t')
    except: df = pd.read_csv(edge_file, sep='\t', nrows=500000)
    pos_n, neg_n = (df['weight'] > 0).sum(), (df['weight'] < 0).sum()
    ax.pie([pos_n, neg_n], labels=[f'positive\n{pos_n:,}', f'negative\n{neg_n:,}'],
           colors=[C3, C1], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title(f'Phase3: new_edge ({len(df):,} edges)')


def plot_phase2_predictions(ax):
    p1 = np.load(os.path.join(PHASE1_DIR, 'gene_z.npz'), allow_pickle=True)
    p2 = np.load(os.path.join(PHASE2_DIR, 'all_gene_z.npz'), allow_pickle=True)
    p1_genes = list(p1['gene_names']); p2_smed = list(p2['gene_ids'])
    with open(os.path.join(PHASE1_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    smesg2smed = scaler.get('smesg2smed', {})
    common_smed = sorted(set(smesg2smed[g] for g in p1_genes if g in smesg2smed))
    p2_idx = {s: i for i, s in enumerate(p2_smed)}
    idx1, idx2 = [], []
    for smed in common_smed:
        if smed in p2_idx:
            i1 = p1_genes.index(smesg2smed.get(smed, smed)) if smed in smesg2smed else None
            if i1 is not None: idx1.append(i1); idx2.append(p2_idx[smed])
    valid = [(i, j) for i, j in zip(idx1, idx2) if i < len(p1['z'])]
    if len(valid) < 2: ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes); return
    z_true = p1['z'][[v[0] for v in valid]]; z_pred = p2['z'][[v[1] for v in valid]]
    n = min(len(z_true), 300)
    ax.scatter(z_true[:n, 0], z_pred[:n, 0], alpha=0.5, s=15, c=C1, edgecolors='none')
    vmin, vmax = min(z_true[:n,0].min(), z_pred[:n,0].min()), max(z_true[:n,0].max(), z_pred[:n,0].max())
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1, alpha=0.7)
    try: from sklearn.metrics import r2_score; r2_val = r2_score(z_true, z_pred)
    except: r2_val = 0
    cos_all = [np.dot(z_true[i], z_pred[i])/(np.linalg.norm(z_true[i])*np.linalg.norm(z_pred[i])+1e-8) for i in range(len(z_true))]
    avg_cos = np.mean(cos_all) if cos_all else 0
    ax.set_xlabel('True z[0]'); ax.set_ylabel('Pred z[0]')
    ax.set_title(f'Phase2: True vs Pred z[0]\nR2={r2_val:.3f}  CosSim={avg_cos:.3f}  genes={len(valid)}')


# ================================================================
# ID 映射
# ================================================================

def _load_symbol_maps():
    """读取 adata_var_data.csv 的 ID→基因名映射

    返回 (ad, smed2sym, smesg2sym, alt_map, _map_smesg, _map_target)
    - _map_smesg: SMESG 目标 ID → 基因名
    - _map_target: 目标 ID → 基因名 (SMED/SMESG 都支持) —
      new_edge.txt 的 target 两种 ID 都有, 必须先试 SMED 再试 SMESG
    - new_edge 的 "SMESG000036216.1" 对应表内 "SMESG000036216-1"
      (转录本后缀写法不同), 未命中时按 base / base-1 回退。
    - 表内 gene_symbol 为 SMED/SMESG 占位符的行视为未映射。
    """
    ad = pd.read_csv(os.path.join(MAP_DIR, 'adata_var_data.csv'))
    smesg2sym = {k: v for k, v in zip(ad['smesg'].astype(str), ad['gene_symbol'])
                 if isinstance(v, str) and not v.startswith(('SMED', 'SMESG'))}
    alt_map = {}
    for k, v in smesg2sym.items():
        m = re.match(r'^(SMESG\d+)[-]\d+$', k)
        if m:
            alt_map.setdefault(m.group(1), v)
    smed2sym = {k: v for k, v in zip(ad['smed'].astype(str), ad['gene_symbol'])
                if isinstance(v, str)}
    # 合并映射: SMED 与 SMESG 前缀不冲突
    id2sym = dict(smed2sym)
    id2sym.update(smesg2sym)

    def _map_smesg(ids):
        sym = ids.map(smesg2sym)
        # 仍未映射的试 base-1 / base-2 转录本
        miss = sym.isna()
        base = ids.astype(str).str.replace(r'\.[0-9]+$', '', regex=True)
        sym = sym.where(~miss, base.map(alt_map))
        return sym.fillna(ids)

    def _map_target(ids):
        sym = ids.map(id2sym)
        miss = sym.isna()
        base = ids.astype(str).str.replace(r'\.[0-9]+$', '', regex=True)
        sym = sym.where(~miss, base.map(alt_map))
        return sym.fillna(ids)

    return ad, smed2sym, smesg2sym, alt_map, _map_smesg, _map_target


# ================================================================
# Phase 3 边 CSV 生成: new_edge.txt → tf,target,estimate
# ================================================================

def _prepare_phase3_csv(out_path, top_k=200, watchlist=None):
    """new_edge.txt → tf,target,estimate (内存安全版)

    - 每个 TF 用小根堆只保留 |weight| 最高的 top_k 条最可信边 (默认 200,
      覆盖 R 树图所需: GLI3 一级 top-n1 + hub 二级 top-n2 + 游离子图 top-10)
    - watchlist (重点基因): 凡涉及这些基因的边全部保留 (direct/间接证据)
    - 流式写出 CSV, 不构建巨型 rows 列表/DataFrame (717M 行文件下 OOM 的根因)

    注意: 不能做全局 top-N — 会饿死排序靠后的 TF。必须按 TF 分组各取前 top_k。
    """
    _, smed2sym, _, _, _, _map_target = _load_symbol_maps()
    watch_set = set(watchlist) if watchlist else set()
    import heapq

    per_tf_best = {}   # tf -> 小根堆 [(abs_w, w, target)]  ≤ top_k
    watch_rows = []    # 涉及重点基因的边 (全部保留, 不截断)
    n_read = 0
    for chunk in pd.read_csv(os.path.join(PHASE3_DIR, 'new_edge.txt'),
                             sep='\t', chunksize=1_000_000,
                             dtype={'source': str, 'target': str}):
        chunk['tgt_clean'] = chunk['target'].str.replace(r'\.[0-9]+$', '', regex=True)
        chunk['tf']     = chunk['source'].map(smed2sym).fillna(chunk['source'])
        chunk['target'] = _map_target(chunk['tgt_clean'])
        if watch_set:
            wm = chunk['tf'].isin(watch_set) | chunk['target'].isin(watch_set)
            watch_rows.append(chunk.loc[wm, ['tf', 'target', 'weight']].copy())
            rest = chunk.loc[~wm]
        else:
            rest = chunk
        for tf, w, tgt in zip(rest['tf'], rest['weight'], rest['target']):
            entry = (abs(w), w, tgt)
            lst = per_tf_best.setdefault(tf, [])
            if len(lst) < top_k:
                heapq.heappush(lst, entry)
            elif entry[0] > lst[0][0]:  # 比当前第 k 大的最小者更可信
                heapq.heapreplace(lst, entry)
        n_read += len(chunk)
        if n_read % 50_000_000 == 0:
            print(f'    已处理 {n_read:,} 行, 当前 TF {len(per_tf_best):,} ...')

    # 流式写出 (避免把全部边攒成 list[dict] 再 DataFrame → OOM)
    with open(out_path, 'w', newline='') as f:
        f.write('tf,target,estimate\n')
        n_watch = 0
        for df in watch_rows:
            n_watch += len(df)
            df[['tf', 'target']].assign(estimate=df['weight']).to_csv(
                f, index=False, header=False)
        for tf, lst in per_tf_best.items():
            for _, w, tgt in lst:
                f.write(f'{tf},{tgt},{w:.6f}\n')
    n_tfs = len(per_tf_best)
    print(f'  Phase3 edge CSV: {n_tfs} 个TF (每个TF top {top_k}'
          f' + 重点基因边 {n_watch}) -> {out_path}')


# ================================================================
# 重点基因报告: 存在性统计 + 新GRN 调控边分数
# ================================================================

def _watchlist_report(watchlist, out_path):
    """输出重点基因在各类表里的存在数量 + 新GRN (new_edge.txt) 中这些基因的调控边分数

    报告内容:
      一、存在性统计: 映射表 / phase1 GRN / 新GRN / zenodo GLI3-KO
      二、新GRN 调控边分数:
          - GLI3 直连边分数 (直接调控)
          - 无直连时最可信的两跳路径分数 GLI3->中介->基因 (间接调控)
          - 基因作为 TF / 作为靶 的边数与 top-5 |weight| 分数
    """
    watch = list(dict.fromkeys(watchlist))  # 去重保序
    ad, smed2sym, _, _, _map_smesg, _map_target = _load_symbol_maps()

    # ── phase2 身份状态 (区分 rna_missing / unmapped / predicted) ──
    _manifest_ok = os.path.exists(os.path.join(PHASE3_DIR, 'prediction_manifest.json'))
    try:
        p2_ids = set(map(str, np.load(os.path.join(PHASE2_DIR, 'all_gene_z.npz'),
                                      allow_pickle=True)['gene_ids']))
    except FileNotFoundError:
        p2_ids = set()
    ident = G.resolve_watchlist(watch, G.load_id_maps(), p2_ids)

    # ── 存在性: 映射表 / phase1 GRN / zenodo ──
    mapped_set = set(ad['gene_symbol'].dropna())
    zen = pd.read_csv(os.path.join(RAW_DIR, 'zenodo5242913_GLI3_KO_multiome_18d__GRN_meta_format(2).csv'))
    zen_tf, zen_tg = set(zen['tf'].dropna()), set(zen['target'].dropna())
    ed = pd.read_csv(os.path.join(ROOT_DIR, 'example_data', 'edges.csv'), index_col=0)
    p1_tf = set(_map_smesg(ed['source'].astype(str).str.replace(r'\.[0-9]+$', '', regex=True)))
    p1_tg = set(_map_smesg(ed['target'].astype(str).str.replace(r'\.[0-9]+$', '', regex=True)))

    # ── 新GRN 流式扫描 ──
    gli3_smed = ad.loc[ad['gene_symbol'] == 'GLI3', 'smed'].dropna()
    gli3_smed = gli3_smed.iloc[0] if len(gli3_smed) else None
    gli3_rows = []                      # GLI3 全部边 (target, weight)
    tf_best = {g: [] for g in watch}    # g 作为 TF: top-5 [(abs, w, target)]
    tg_best = {g: [] for g in watch}    # g 作为靶: top-5 [(abs, w, source_tf)]
    tg_all  = {g: [] for g in watch}    # g 作为靶: 全部 (source_tf, w) — 用于两跳
    tf_n = {g: 0 for g in watch}
    tg_n = {g: 0 for g in watch}
    seen_tf, seen_tg = set(), set()
    for chunk in pd.read_csv(os.path.join(PHASE3_DIR, 'new_edge.txt'),
                             sep='\t', chunksize=1000000,
                             dtype={'source': str, 'target': str}):
        chunk['tgt_clean'] = chunk['target'].str.replace(r'\.[0-9]+$', '', regex=True)
        chunk['tf']     = chunk['source'].map(smed2sym).fillna(chunk['source'])
        chunk['target'] = _map_target(chunk['tgt_clean'])
        seen_tf.update(chunk['tf'].unique())
        seen_tg.update(chunk['target'].unique())
        if gli3_smed:
            g3 = chunk.loc[chunk['source'] == gli3_smed, ['target', 'weight']]
            gli3_rows.extend(zip(g3['target'], g3['weight']))
        for g in watch:
            m = chunk['tf'] == g
            if m.any():
                tf_n[g] += int(m.sum())
                lst = tf_best[g]
                for t, w in zip(chunk.loc[m, 'target'], chunk.loc[m, 'weight']):
                    lst.append((abs(w), w, t))
                lst.sort(reverse=True)
                del lst[5:]
            m = chunk['target'] == g
            if m.any():
                tg_n[g] += int(m.sum())
                lst = tg_best[g]
                for s, w in zip(chunk.loc[m, 'tf'], chunk.loc[m, 'weight']):
                    lst.append((abs(w), w, s))
                    tg_all[g].append((s, w))
                lst.sort(reverse=True)
                del lst[5:]

    # GLI3 直连边 (每个靶取 |w| 最大) + 两跳路径
    gli3_map = {}
    for t, w in gli3_rows:
        if t not in gli3_map or abs(w) > abs(gli3_map[t]):
            gli3_map[t] = w
    direct, twohop = {}, {}
    for g in watch:
        if g in gli3_map:
            direct[g] = gli3_map[g]
        else:
            best = None
            for s, w2 in tg_all[g]:
                w1 = gli3_map.get(s)
                if w1 is not None:
                    score = min(abs(w1), abs(w2))
                    if best is None or score > best[0]:
                        best = (score, s, w1, w2)
            if best:
                twohop[g] = best

    # ── 组装报告 ──
    lines = []
    lines.append('=' * 72)
    lines.append(f'重点基因 watchlist 报告 (输入 {len(watchlist)} 个, 去重后 {len(watch)} 个)')
    lines.append(f'  {", ".join(watch)}')
    lines.append('=' * 72)

    lines.append('')
    lines.append('〇、phase2 身份状态 (决定该基因能否进入新GRN)')
    lines.append('  rna_missing = 映射表中没有对应SMED/SMESG, 永远不可能出现在预测里')
    lines.append('  unmapped    = 有ID但不在phase2 z空间 (本次RNA数据缺失)')
    lines.append('  predicted   = 有phase2 z, 可作为source/target参与新GRN')
    lines.append('')
    for g in watch:
        w = ident[g]
        lines.append(f'  {g:<8} {w["status"]:<12} smed={w["smeds"] or "-"}  in_z={w["in_z"] or "-"}')

    lines.append('')
    lines.append('一、存在性统计 (基因在表中出现, 作为 TF 或靶基因)')
    lines.append('')
    lines.append(f'{"基因":<10} {"phase2z":<8} {"映射表":<8} {"phase1":<8} {"新GRN":<8} {"zenodo":<8}')
    lines.append('-' * 50)
    n_mapz = n_mapped = n_p1 = n_new = n_zen = 0
    for g in watch:
        in_z   = ident[g]['status'] == 'predicted'
        in_map = g in mapped_set
        in_p1  = g in p1_tf or g in p1_tg
        in_new = g in seen_tf or g in seen_tg
        in_zen = g in zen_tf or g in zen_tg
        n_mapz += in_z; n_mapped += in_map; n_p1 += in_p1; n_new += in_new; n_zen += in_zen
        lines.append(f'{g:<10} {"✓" if in_z else "-":<8} {"✓" if in_map else "-":<8} '
                     f'{"✓" if in_p1 else "-":<8} {"✓" if in_new else "-":<8} '
                     f'{"✓" if in_zen else "-":<8}')
    lines.append('-' * 50)
    lines.append(f'{"合计":<10} {n_mapz}/{len(watch):<6} {n_mapped}/{len(watch):<6} '
                 f'{n_p1}/{len(watch):<6} {n_new}/{len(watch):<6} {n_zen}/{len(watch):<6}')

    lines.append('')
    lines.append('二、新GRN (new_edge.txt) 调控边分数')
    if not _manifest_ok:
        lines.append('  ⚠ 无 prediction_manifest.json: 无法确认 new_edge.txt 是否完整生成, '
                     '若某基因 as-TF 为 0 可能只是预测被截断, 建议重跑 phase3')
    lines.append('')
    for g in watch:
        w = ident[g]
        lines.append(f'● {g}  [phase2: {w["status"]}]')
        if w['status'] == 'rna_missing':
            lines.append(f'   映射表中不存在该基因符号 (SMED/SMESG 均无), 无法参与建模与预测;')
            lines.append(f'   请核对基因名拼写或补充映射表 (此前显示"在新GRN中不存在"即此原因)。')
            lines.append('')
            continue
        if g in direct:
            lines.append(f'   GLI3 直连: GLI3 -> {g}   weight = {direct[g]:+.4f}  (直接调控)')
        elif g in twohop:
            score, mid, w1, w2 = twohop[g]
            lines.append(f'   GLI3 直连: 无')
            lines.append(f'   最可信两跳 (间接调控): GLI3 -> {mid} (w={w1:+.4f}) -> {g} (w={w2:+.4f})   path = {score:.4f}')
        else:
            lines.append(f'   GLI3 直连: 无')
            if tf_n[g] or tg_n[g]:
                lines.append(f'   两跳路径: 无 (基因在新GRN中存在, 但与 GLI3 不可达 — orphan)')
            else:
                lines.append(f'   两跳路径: 无')
        if tf_n[g]:
            top = ', '.join(f'{t}({w_:+.3f})' for _, w_, t in tf_best[g])
            lines.append(f'   作为TF: {tf_n[g]:,} 条边, |weight| top5: {top}')
        if tg_n[g]:
            top = ', '.join(f'{s}({w_:+.3f})' for _, w_, s in tg_best[g])
            lines.append(f'   作为靶: {tg_n[g]:,} 条边, |weight| top5: {top}')
        if not tf_n[g] and not tg_n[g]:
            if w['status'] == 'predicted':
                lines.append(f'   在新GRN中无任何边: 若 manifest 缺失/PARTIAL, 可能因预测被截断; '
                             f'请重跑完整 phase3 后再判断')
            else:
                lines.append(f'   无 phase2 z 向量, 无法参与预测 (phase2 数据缺失)')
        lines.append('')

    # 机器可读的状态表 (direct/indirect/orphan/rna_missing/unmapped), 供下游与复现
    status_rows = []
    for g in watch:
        w = ident[g]
        if w['status'] != 'predicted':
            st = w['status']
        elif g in direct:
            st = 'direct'
        elif g in twohop:
            st = 'indirect'
        elif tf_n[g] or tg_n[g]:
            st = 'orphan'
        else:
            st = 'no_edges'
        status_rows.append({'gene': g, 'status': st,
                            'smeds': ';'.join(w['smeds']), 'in_z_smeds': ';'.join(w['in_z']),
                            'gli3_direct_weight': direct.get(g, ''),
                            'gli3_twohop': ('GLI3->%s->%s' % (twohop[g][1], g)) if g in twohop else '',
                            'n_as_tf': tf_n[g], 'n_as_target': tg_n[g]})
    pd.DataFrame(status_rows).to_csv(
        os.path.join(os.path.dirname(out_path), 'watchlist_status.csv'), index=False)

    report = '\n'.join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(report)
    return report


# ================================================================
# 聚焦模式: 只提取 root(GLI3) ↔ 重点基因 的调控边
# ================================================================

def _extract_focus_from_phase3(root_smed, watch_syms, watch_smeds, map_target):
    """流式扫描 new_edge.txt, 只取两类边:

      1. root → X            (X 为任意基因, 作为"中介"候选 / 直接靶)
      2. X    → 重点基因      (X 为中介候选)

    返回 (root_edges, hop_edges):
      root_edges: {target_symbol: (abs_w, w)}      root 的全部边 (每个靶取 |w| 最大)
      hop_edges : {(tf_symbol, watch_sym): (abs_w, w)} 指向重点基因的边
    """
    _, smed2sym, _, _, _, _map_tgt = _load_symbol_maps()
    root_edges, hop_edges = {}, {}
    n_read = 0
    for chunk in pd.read_csv(os.path.join(PHASE3_DIR, 'new_edge.txt'), sep='\t',
                             chunksize=2_000_000,
                             dtype={'source': str, 'target': str, 'weight': float}):
        src = chunk['source'].values
        w = chunk['weight'].values
        # 只保留 source 是 root, 或 target 可能是重点基因的行
        tgt_sym = map_target(chunk['target'].str.replace(r'\.[0-9]+$', '', regex=True))
        is_root = (src == root_smed)
        is_watch = tgt_sym.isin(watch_syms).values | \
                   chunk['target'].str.replace(r'\.[0-9]+$', '', regex=True).isin(watch_smeds).values
        for i in np.where(is_root)[0]:
            t = tgt_sym.iloc[i]
            cur = root_edges.get(t)
            if cur is None or abs(w[i]) > cur[0]:
                root_edges[t] = (abs(w[i]), w[i])
        for i in np.where(is_watch)[0]:
            tf_sym = smed2sym.get(src[i], src[i])
            wsym = tgt_sym.iloc[i]
            if wsym not in watch_syms:          # 映射失败时回退到 ID 匹配
                ids = [g for g, ss in G.load_id_maps()['sym2smed'].items()
                       if chunk['target'].str.replace(r'\.[0-9]+$', '', regex=True).iloc[i] in ss]
                wsym = ids[0] if ids else wsym
            key = (tf_sym, wsym)
            cur = hop_edges.get(key)
            if cur is None or abs(w[i]) > cur[0]:
                hop_edges[key] = (abs(w[i]), w[i])
        n_read += len(chunk)
        if n_read % 50_000_000 == 0:
            print(f'      已扫描 {n_read:,} 行 ...')
    print(f'   root 边 {len(root_edges):,} 条, 指向重点基因的边 {len(hop_edges):,} 条')
    return root_edges, hop_edges


def _prepare_focus_csv(out_path, watchlist, root_symbol='GLI3', n_bg=25):
    """生成"GLI3 ↔ 重点基因 + 背景基因"的聚焦边 CSV。

    边分两类:
      ① GLI3 → 重点基因          (直接调控, 红高亮)
      ② GLI3 → 中介TF → 重点基因  (间接调控, 红高亮)
      ③ 背景: GLI3 → top-n_bg |w| 靶基因 (edges.csv 中已知的 GLI3 靶优先, 其余用模型权重排序)
         中介TF → 该 TF 的 top 下游靶 (让树图有第二层)
         一级靶中仍是 TF 的 → 再下游 (第三层, 若存在)

    返回 (status_dict, root_w_dict)。root_w: {symbol: weight} GLI3 全部预测边。
    """
    maps = G.load_id_maps()
    root_smeds = maps['sym2smed'].get(root_symbol, [])
    root_smed = root_smeds[0] if root_smeds else None
    watch_syms = set(watchlist)
    watch_smeds = {s for g in watchlist for s in maps['sym2smed'].get(g, [])}
    if root_smed is None:
        print(f'   ⚠ 映射表中找不到 {root_symbol}, 无法生成聚焦图')
        return {}, {}

    _, _, _, _, _, _map_tgt = _load_symbol_maps()
    root_edges, hop_edges = _extract_focus_from_phase3(
        root_smed, watch_syms, watch_smeds, _map_tgt)

    rows, status = [], {}
    for g in watchlist:
        if not maps['sym2smed'].get(g):
            status[g] = ('none', 'rna_missing: 映射表无此基因符号')
            continue
        if g in root_edges:                       # ① 直接调控
            _, w = root_edges[g]
            rows.append({'tf': root_symbol, 'target': g, 'estimate': w})
            status[g] = ('direct', w)
            continue
        # ② 间接: GLI3→中介→g 中 path_score 最高
        best = None
        for (tf_sym, wsym), (abs2, w2) in hop_edges.items():
            if wsym != g or tf_sym not in root_edges:
                continue
            abs1, w1 = root_edges[tf_sym]
            score = min(abs1, abs2)
            if best is None or score > best[0]:
                best = (score, tf_sym, w1, w2)
        if best:
            _, mid, w1, w2 = best
            rows.append({'tf': root_symbol, 'target': mid, 'estimate': w1})
            rows.append({'tf': mid, 'target': g, 'estimate': w2})
            status[g] = ('indirect', mid, w1, w2)
        else:
            status[g] = ('none', '新GRN中无 GLI3 直接边, 也无可达的两跳中介')

    # ③ 背景基因: edges.csv 中已知的 GLI3 靶优先 + 其余按模型权重排序
    _, _, _, _, _map_smesg, _ = _load_symbol_maps()
    ed = pd.read_csv(os.path.join(ROOT_DIR, 'example_data', 'edges.csv'), index_col=0)
    ed['tf'] = _map_smesg(ed['source'].astype(str).str.replace(r'[.-]\d+$', '', regex=True))
    ed['target'] = _map_smesg(ed['target'].astype(str).str.replace(r'[.-]\d+$', '', regex=True))
    known_gli3_t = set(ed[ed['tf'] == root_symbol]['target'].dropna())

    # 优先取已知靶, 再从 root_edges 按 |w| 补足 n_bg
    bg = [t for t in known_gli3_t if t in root_edges and t not in watch_syms]
    rest = [t for t in sorted(root_edges, key=lambda t: -root_edges[t][0])
            if t not in watch_syms and t not in bg]
    bg = (bg + rest)[:n_bg]
    for t in bg:
        _, w = root_edges[t]
        rows.append({'tf': root_symbol, 'target': t, 'estimate': w})
    print(f'  背景基因 (GLI3 top-n_bg 靶, 已知优先): {len(bg)} 个')

    # ③b 第二层: 一级节点 (重点中介 + 背景) 中仍是 TF 的 → 各自下游 top-5
    # 这保证树图有第 2 层 (不止根+叶子)
    l1_nodes = [r['target'] for r in rows if r['tf'] == root_symbol]
    # 仅对重点基因的中介 + 已知是 TF 的背景节点拉第二层 (最多 6 个 hub × 5 下游)
    l1_tfs = [t for t in l1_nodes if t in set(ed['tf']) or
              t in {x[0] for x in hop_edges if x[0] in root_edges}]
    l1_tfs = l1_tfs[:6]
    n_l2 = 0
    for hub in l1_tfs:
        downstream = {t: w for (tf, t), (aw, w) in hop_edges.items()
                      if tf == hub and t not in l1_nodes and t not in watch_syms}
        for t, w in sorted(downstream.items(), key=lambda x: -x[1])[:5]:
            rows.append({'tf': hub, 'target': t, 'estimate': w})
            n_l2 += 1
    print(f'  第二层边 (一级TF→下游): {n_l2} 条')

    pd.DataFrame(rows, columns=['tf', 'target', 'estimate']).to_csv(out_path, index=False)
    print(f'  聚焦边 CSV: {len(rows)} 条边 -> {out_path}')
    for g, st in status.items():
        if st[0] == 'direct':
            print(f'    {g:8s} 直接   GLI3 -> {g}  w={st[1]:+.4f}')
        elif st[0] == 'indirect':
            print(f'    {g:8s} 间接   GLI3 -> {st[1]} (w={st[2]:+.4f}) -> {g} (w={st[3]:+.4f})')
        else:
            print(f'    {g:8s} 无     {st[1]}')
    return status, {g: v[1] for g, v in root_edges.items()}


def _prepare_focus_csv_phase1(out_path, watchlist, root_symbol='GLI3', n_bg=25):
    """phase1 (edges.csv) 的聚焦版本 — GLI3 ↔ 重点基因 + 背景基因 + 一级TF第二层。

    注: GLI3 的 28 个直接靶中没有任何一个是 edges.csv 中的 TF, 故严格意义上 phase1
    从 GLI3 出发只有一层。这里把"背景基因"(其余已知 GLI3 靶)全部放进一级环, 并保留
    二级扩展尝试 (若某背景靶是 TF 则其下游为第二层)。
    """
    _, _, _, _, _map_smesg, _ = _load_symbol_maps()
    ed = pd.read_csv(os.path.join(ROOT_DIR, 'example_data', 'edges.csv'), index_col=0)
    ed['tf'] = _map_smesg(ed['source'].astype(str).str.replace(r'[.-]\d+$', '', regex=True))
    ed['target'] = _map_smesg(ed['target'].astype(str).str.replace(r'[.-]\d+$', '', regex=True))
    root_edges = ed[ed['tf'] == root_symbol]
    rows, status = [], {}
    for g in watchlist:
        hit = root_edges[root_edges['target'] == g]
        if len(hit):
            w = float(hit['weight'].iloc[0])
            rows.append({'tf': root_symbol, 'target': g, 'estimate': w})
            status[g] = ('direct', w)
        else:
            meds = set(root_edges['target'])
            hop = ed[(ed['tf'].isin(meds)) & (ed['target'] == g)]
            if len(hop):
                r = hop.iloc[0]
                w1 = float(root_edges[root_edges['target'] == r['tf']]['weight'].iloc[0])
                w2 = float(r['weight'])
                rows.append({'tf': root_symbol, 'target': r['tf'], 'estimate': w1})
                rows.append({'tf': r['tf'], 'target': g, 'estimate': w2})
                status[g] = ('indirect', r['tf'], w1, w2)
            else:
                status[g] = ('none', 'phase1 edges.csv 中不存在该基因')

    # 背景: GLI3 的其余已知靶 (全部, 因只有 28 个)
    l1_nodes = [r['target'] for r in rows if r['tf'] == root_symbol]
    for _, r in root_edges.iterrows():
        if r['target'] not in l1_nodes and r['target'] not in watchlist:
            rows.append({'tf': root_symbol, 'target': r['target'], 'estimate': float(r['weight'])})
    # 第二层: 一级节点中也是 TF 的 → 各自下游 (若有)
    l2_n = 0
    for t in l1_nodes:
        dn = ed[(ed['tf'] == t) & (~ed['target'].isin(l1_nodes))]
        for _, r in dn.head(5).iterrows():
            rows.append({'tf': t, 'target': r['target'], 'estimate': float(r['weight'])})
            l2_n += 1
    pd.DataFrame(rows, columns=['tf', 'target', 'estimate']).to_csv(out_path, index=False)
    print(f'  聚焦边 CSV (phase1): {len(rows)} 条边 ({l2_n} 条二级) -> {out_path}')
    for g, st in status.items():
        print(f'    {g:8s} ' + (f'直接 w={st[1]:+.4f}' if st[0] == 'direct' else
                                f'间接 via {st[1]}' if st[0] == 'indirect' else f'无 ({st[1]})'))
    return status


# ================================================================
# Rscript 调用
# ================================================================

def _run_r_tree(edges_csv, root, output_png, n1=80, nhubs=3, n2=50, highlight=None,
                spread_labels=False):
    """调用 draw_grn_tree.R 生成圆形 GRN 树图

    highlight: 重点展示基因列表 (逗号分隔传给 R, 仅高亮树中存在的基因)
    """
    cmd = ['Rscript', R_SCRIPT,
           '--edges', edges_csv, '--root', root, '--output', output_png,
           '--n1', str(n1), '--nhubs', str(nhubs), '--n2', str(n2)]
    if highlight:
        cmd += ['--highlight', ','.join(highlight)]
    if spread_labels:
        cmd += ['--spread-labels']
    print(f'  Rscript ... --root {root} -> {os.path.basename(output_png)}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f'  R STDERR: {result.stderr[:500]}')
        return False
    print(result.stdout.strip()[-80:])  # last line
    return True


# ================================================================
# 主函数
# ================================================================

def main():
    # ── 先验证 phase3 结果完整性, 拒绝用截断文件绘图 ──
    _load_prediction_manifest()

    # ====== Figure 1: 原有概览图 ======
    fig1 = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 3, figure=fig1, hspace=0.35, wspace=0.3)

    ax1 = fig1.add_subplot(gs[0, 0])
    p1 = np.load(os.path.join(PHASE1_DIR, 'gene_z.npz'), allow_pickle=True); z = p1['z']
    ax1.hist(z.flatten(), bins=60, color=C1, alpha=0.8, edgecolor='white')
    ax1.set_xlabel('z value'); ax1.set_ylabel('frequency')
    ax1.set_title(f'Phase1: z distribution\n({z.shape[0]} genes x {z.shape[1]} dim)')

    ax2 = fig1.add_subplot(gs[0, 1])
    sc = ax2.scatter(z[:,0], z[:,1], s=8, c=np.arange(len(z)), cmap='viridis', alpha=0.6, edgecolors='none')
    ax2.set_xlabel('z[0]'); ax2.set_ylabel('z[1]')
    ax2.set_title(f'Phase1: z[0] vs z[1]\n({len(z)} genes)')
    plt.colorbar(sc, ax=ax2, label='gene index')

    ax3 = fig1.add_subplot(gs[0, 2]); plot_phase2_predictions(ax3)
    ax4 = fig1.add_subplot(gs[1, 0]); plot_phase3_edge_stats(ax4)
    ax5 = fig1.add_subplot(gs[1, 1]); plot_phase3_edge_stats_full(ax5)

    ax6 = fig1.add_subplot(gs[1, 2]); ax6.axis('off')
    summary = [f"Phase 1: GRNM", f"  genes: {z.shape[0]}, z dim: {z.shape[1]}"]
    p2 = np.load(os.path.join(PHASE2_DIR, 'all_gene_z.npz'), allow_pickle=True)
    summary.append(f"\nPhase 2: RNA->z"); summary.append(f"  all genes: {p2['z'].shape[0]}, z dim: {p2['z'].shape[1]}")
    ef = os.path.join(PHASE3_DIR, 'new_edge.txt')
    if os.path.exists(ef):
        try:
            dfe = pd.read_csv(ef, sep='\t')
            summary.append(f"\nPhase 3: new edges"); summary.append(f"  edges: {len(dfe):,}")
            summary.append(f"  pos: {(dfe['weight']>0).sum():,}, neg: {(dfe['weight']<0).sum():,}")
        except: summary.append(f"\nPhase 3: new edges\n  (large file)")
    summary.append(f"\nHighlight genes (GLI3):\n  " + ', '.join(GLI3_WATCHLIST))
    ax6.text(0.05, 0.95, '\n'.join(summary), transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig1.suptitle('GRN New Edge Discovery - Full Pipeline Results', fontsize=16, fontweight='bold', y=0.98)

    # GLI3 重点基因列表以图底脚注展示 (面板空间不足, 树图内会红色高亮存在的基因)
    gli3_str = 'GLI3 tree highlight genes (shown if present): ' + ', '.join(GLI3_WATCHLIST)
    for i, line in enumerate(textwrap.wrap(gli3_str, width=95)):
        fig1.text(0.5, 0.03 - i * 0.014, line, ha='center', va='bottom',
                  fontsize=8, family='monospace', color='#444444')

    out1 = os.path.join(PNG_DIR, 'pipeline_summary.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight'); plt.close(fig1)
    print(f'Done -> {out1}')

    # ====== 重点基因报告: 存在性统计 + 新GRN 调控边分数 ======
    report_file = os.path.join(ROOT_DIR, 'outputs', 'watchlist_report.txt')
    _watchlist_report(GLI3_WATCHLIST, report_file)
    print(f'Done -> {report_file}')

    # ====== GLI3 GRN 树图 (聚焦模式: GLI3 ↔ 重点基因 + edges.csv 已知基因) ======
    # 在聚焦边基础上, 追加 phase1 已知 GRN (edges.csv) 中 GLI3 的直接靶基因
    # 作为"背景基因" — 它们构成 GLI3 调控的真实上下文, 让树图不单调。
    # (重点基因保持红色高亮, 背景基因灰色小点)
    _, _, _, _, _map_smesg, _ = _load_symbol_maps()

    # 已知 GRN (edges.csv) 中 GLI3 的直接靶 (映射为 symbol)
    ed_all = pd.read_csv(os.path.join(ROOT_DIR, 'example_data', 'edges.csv'), index_col=0)
    ed_all['tf'] = _map_smesg(ed_all['source'].astype(str).str.replace(r'[.-]\d+$', '', regex=True))
    ed_all['target'] = _map_smesg(ed_all['target'].astype(str).str.replace(r'[.-]\d+$', '', regex=True))
    gli3_known_targets = ed_all[ed_all['tf'] == 'GLI3']['target'].dropna().unique().tolist()
    print(f'  edges.csv 中 GLI3 的直接靶背景基因: {len(gli3_known_targets)} 个')

    # Phase 1: edges.csv 聚焦边 + GLI3 背景靶基因 (聚焦函数已含背景; GLI3 靶中无 TF → 数据本质只有1层)
    p1_focus = os.path.join(TEMP_DIR, 'phase1_gli3_focus.csv')
    _prepare_focus_csv_phase1(p1_focus, GLI3_WATCHLIST, 'GLI3')
    out_p1 = os.path.join(PNG_DIR, 'gli3_grn_phase1.png')
    ok1 = _run_r_tree(p1_focus, 'GLI3', out_p1, n1=80, nhubs=4, n2=25,
                      highlight=GLI3_WATCHLIST, spread_labels=True)
    print(f'  Phase1 GLI3 GRN (聚焦+背景, edges中 GLI3 靶均非TF故只有1层): '
          f'{"OK" if ok1 else "FAILED"} -> {out_p1}')

    # Phase 3: 模型预测 聚焦 + 背景 (有二级/三级调控, 不止1层)
    p3_focus = os.path.join(TEMP_DIR, 'phase3_gli3_focus.csv')
    st_p3, root_w = _prepare_focus_csv(p3_focus, GLI3_WATCHLIST, 'GLI3')
    out_p3 = os.path.join(PNG_DIR, 'gli3_grn_phase3.png')
    ok3 = _run_r_tree(p3_focus, 'GLI3', out_p3, n1=80, nhubs=6, n2=30,
                      highlight=GLI3_WATCHLIST, spread_labels=True)
    print(f'  Phase3 GLI3 GRN (模型预测, 聚焦+背景+多层): {"OK" if ok3 else "FAILED"} -> {out_p3}')


if __name__ == '__main__':
    main()
