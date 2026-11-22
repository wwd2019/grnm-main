"""
new_human_results.py — 人类数据的三阶段训练流程 + 扩展后的 GLI3 GRN 图
================================================================================
原流程 (phase1/2/3) 面向涡虫 (SMED/SMESG)：RNA+ATAC -> z -> 边分类。
本脚本用 example_data/ 下两份人类数据复现同一套三阶段训练思路。

数据来源 (只用这两份, 不引入任何外部数据):
  1. example_data/glm_network_coefs_human.csv
     增强子-GLM 回归系数 (tf, target, region, estimate, padj, ...)
     -> 提供"谁调控谁、调控多强", 以及每个基因的调控谱 (in / out 两个特征块)
  2. example_data/motif2tf_human.tsv
     Motif2TF 表 (tf, motif_tf, motif, prc_similarity, collection, ...)
     -> 提供 motif 层面的调控证据, 以及每个基因的 motif 谱特征块

Phase 1  GRNM 图嵌入 (z)
  - 节点 = GLM 全部基因 ∪ motif2tf 全部基因
  - 两个特征块 (对应原流程的 RNA / ATAC 双模态):
      rna-like  : GLM 出边系数谱 (该基因作为 TF 对 304 个靶的系数)
      atac-like : GLM 入边系数谱 (该基因作为靶被 1282 个 TF 调控的系数)
                  + motif 相似度谱 (220 个 motif 拥有者) + motif 拥有者 one-hot
  - 边 = 显著 GLM 对 (padj<=0.05, 每个 (tf,target) 取 |estimate| 最大)
         + motif2tf 边 (motif_tf -> tf, 权重 = prc_similarity)
  - 图自编码器 (grnm2.Grnm, 与原流程同一实现) -> 每个基因 64 维 z

Phase 2  调控谱 -> z 的残差网络映射
  - 输入 = GLM 入边系数谱 (1282 维); 目标 = Phase1 z
  - 按基因划分 train/val, val 早停, 报告 R² / CosSim
  - 对全部有调控输入谱的基因重新预测 z; 无输入谱的基因沿用 Phase1 z (记入 manifest)

Phase 3  配对 z -> 7 类调控关系分类
  - 标签 = 显著 GLM 对 (正/负各按 |estimate| 33%/66% 分位切 3 档) + 随机采样 0 (无调控)
  - 对 (GLM 中的 TF ∪ motif 拥有者) × 全部基因 全量预测, 流式写出 human_new_edge.txt

最终  GLI3 扩展调控图 (三层证据合并)
  A 实测: GLI3 在 GLM 中的显著直连靶 (按 |estimate|)
  B 扩展: Phase3 预测出的 GLI3 -> 基因 (这些基因在 GLM 里没有 GLI3 系数 = 模型外推的新边)
  C motif: GLI3 作为 motif 拥有者命中的基因 (motif2tf MA1491.1)
  画风复用 draw_grn_tree.R, 与 phase4_plot_results.py 完全一致

用法:
  python scripts/new_human_results.py                       # 全流程 (默认参数)
  python scripts/new_human_results.py --phase1-epochs 60 --phase3-epochs 100
  python scripts/new_human_results.py --skip-phase1 --skip-phase2 --skip-phase3   # 只重画图

输出:
  outputs/phase1_human_output/         human_gene_z.npz, human_feature_blocks.npz, human_inputs.pkl
  outputs/phase2_human_output/         human_rna2z_model.pt, human_gene_z_phase2.npz, metrics.json
  outputs/phase3_human_output/         human_new_edge.txt, human_prediction_manifest.json
  outputs/human_GLI3_report.txt
  figures/human_phase1_embedding.png, human_phase2_results.png, human_phase3_results.png,
  figures/human_gli3_grn_expanded.png
  temp_tree_data/human_gli3_expanded_edges.csv, human_gli3_edge_scores.csv
"""

import os
import sys
import io
import json
import math
import time
import pickle
import random
import argparse
import subprocess
import contextlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, ROOT_DIR)

from phase2_rna2z import RNA2Z, lr_warmup_cosine    # 原流程的残差网络 + 学习率调度
from phase3_edge_predict import EdgeClassifier      # 原流程的 7 类分类器

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
# 图内中文标签需要 CJK 字体 (优先用系统里第一个可用的中文无衬线字体)
for _font in ('Microsoft YaHei', 'SimHei', 'DengXian', 'Noto Sans CJK SC', 'DejaVu Sans'):
    if any(_f.name == _font for _f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [_font] + list(plt.rcParams['font.sans-serif'])
        break

# ---- 路径 ----
GLM_CSV = os.path.join(ROOT_DIR, 'example_data', 'glm_network_coefs_human.csv')
MOTIF_TSV = os.path.join(ROOT_DIR, 'example_data', 'motif2tf_human.tsv')
P1_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase1_human_output')
P2_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase2_human_output')
P3_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase3_human_output')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
PNG_DIR = os.path.join(ROOT_DIR, 'figures')
TEMP_DIR = os.path.join(ROOT_DIR, 'temp_tree_data')
R_SCRIPT = os.path.join(ROOT_DIR, 'draw_grn_tree.R')
for _d in (P1_DIR, P2_DIR, P3_DIR, OUTPUT_DIR, PNG_DIR, TEMP_DIR):
    os.makedirs(_d, exist_ok=True)

ROOT_GENE = 'GLI3'
PADJ_MAX = 0.05
Z_DIM = 64
C1, C2, C3 = '#4C72B0', '#55A868', '#C44E52'

# 重点展示基因 (与 phase4 的 GLI3_WATCHLIST 同一来源, 人类符号)
GLI3_WATCHLIST = [
    'ZIC3', 'ZNF608', 'CUX1', 'HES1', 'JARID2', 'MAF', 'PBX1', 'SIX6',
    'SOX11', 'EMX2', 'TOX', 'ZIC2', 'SIX3', 'SMAD3', 'SOX4', 'SALL4',
    'DACH1', 'RAX', 'SOX6', 'ZNF521', 'PAX6', 'CREB5', 'DMRTA1', 'HES4',
]


# ================================================================
# 输入装配: 两个数据源 -> 统一基因空间 + 特征块 + 边表
# ================================================================

def load_human_inputs():
    """读取并规范化两份人类数据, 返回 (meta, blocks)。

    blocks 的每一块都与 meta['genes'] 顺序一致:
      blk_out  GLM 出边系数谱 (基因作为 TF)          dim = #GLM TF
      blk_in   GLM 入边系数谱 (基因作为靶)           dim = #GLM target
      blk_mot  motif 相似度谱 (基因 × motif 拥有者)  dim = #motif 拥有者
      blk_own  motif 拥有者 one-hot                  dim = #motif 拥有者
    """
    glm = pd.read_csv(GLM_CSV)
    m2t = pd.read_csv(MOTIF_TSV, sep='\t')
    for col in ('tf', 'target'):
        glm[col] = glm[col].astype(str).str.strip()
    for col in ('tf', 'motif_tf'):
        m2t[col] = m2t[col].astype(str).str.strip()
    glm = glm.dropna(subset=['tf', 'target', 'estimate'])
    m2t = m2t.dropna(subset=['tf', 'motif_tf', 'prc_similarity'])

    # 显著 GLM 对: 同一 (tf, target) 跨 region 只保留 |estimate| 最大的一条
    sig = glm[glm['padj'] <= PADJ_MAX].copy()
    sig['abs_est'] = sig['estimate'].abs()
    if sig.empty:
        raise SystemExit(f'错误: {GLM_CSV} 中没有 padj<={PADJ_MAX} 的记录')
    pick = sig.groupby(['tf', 'target'])['abs_est'].idxmax()
    pairs = sig.loc[pick, ['tf', 'target', 'estimate', 'padj']].reset_index(drop=True)

    tfs = sorted(glm['tf'].unique())
    tgts = sorted(glm['target'].unique())
    owners = sorted(m2t['motif_tf'].unique())
    genes = sorted(set(tfs) | set(tgts) | set(m2t['tf']) | set(owners))
    gidx = {g: i for i, g in enumerate(genes)}
    tidx = {t: i for i, t in enumerate(tfs)}
    uidx = {t: i for i, t in enumerate(tgts)}
    oidx = {o: i for i, o in enumerate(owners)}

    n = len(genes)
    blk_out = np.zeros((n, len(tfs)), dtype=np.float64)
    blk_in = np.zeros((n, len(tgts)), dtype=np.float64)
    blk_mot = np.zeros((n, len(owners)), dtype=np.float64)
    blk_own = np.zeros((n, len(owners)), dtype=np.float64)

    for tf, tgt, est in zip(pairs['tf'], pairs['target'], pairs['estimate']):
        blk_out[gidx[tf], tidx[tf]] = est          # 作为 TF 的调控输出强度
        blk_in[gidx[tgt], uidx[tgt]] = est         # 作为靶收到的调控强度
    for tf, owner, sim in zip(m2t['tf'], m2t['motif_tf'], m2t['prc_similarity']):
        j = oidx.get(owner)
        if j is not None:
            i = gidx[tf]
            blk_mot[i, j] = max(blk_mot[i, j], float(sim))
    for owner in owners:
        blk_own[gidx[owner], oidx[owner]] = 1.0

    # ---- 边表 (Phase1 的图结构) ----
    glm_edges = pairs.rename(columns={'tf': 'source'})[['source', 'target', 'estimate']].copy()
    glm_edges['weight'] = glm_edges['estimate'] * 100.0   # 与原流程同量级 (edges 权重 ×100)

    m2t_best = (m2t.sort_values('prc_similarity', ascending=False)
                   .drop_duplicates(['motif_tf', 'tf']))
    mot_edges = pd.DataFrame({
        'source': m2t_best['motif_tf'].values,
        'target': m2t_best['tf'].values,
        'weight': m2t_best['prc_similarity'].values.astype(float) * 100.0,
    })
    mot_edges = mot_edges[mot_edges['source'].isin(gidx) & mot_edges['target'].isin(gidx)]

    edges = pd.concat([glm_edges[['source', 'target', 'weight']],
                       mot_edges[['source', 'target', 'weight']]], ignore_index=True)

    meta = dict(genes=genes, tfs=tfs, tgts=tgts, owners=owners,
                pairs=pairs, edges=edges, glm=glm, m2t=m2t,
                n_sig_pairs=len(pairs), n_glm_edges=len(glm_edges),
                n_motif_edges=len(mot_edges), n_motif_edges_raw=len(m2t_best))
    blocks = dict(blk_out=blk_out, blk_in=blk_in, blk_mot=blk_mot, blk_own=blk_own)
    print(f'基因空间: {n} 个 (GLM TF {len(tfs)} / GLM 靶 {len(tgts)} / motif 拥有者 {len(owners)})')
    print(f'显著 GLM 对: {len(pairs)} 条; motif2tf 边: {len(mot_edges)} 条 '
          f'(原始 {len(m2t_best)}, 两端不在基因空间的 {len(m2t_best) - len(mot_edges)} 条已排除)')
    return meta, blocks


# ================================================================
# Phase 1: GRNM 图嵌入 (z)
# ================================================================

def run_phase1(meta, blocks, d1=60, d2=8, epochs=150, alpha=0.01):
    import grnm2

    genes = meta['genes']
    rna_like = pd.DataFrame(blocks['blk_out'], index=genes)
    atac_like = pd.DataFrame(
        np.hstack([blocks['blk_in'], blocks['blk_mot'], blocks['blk_own']]), index=genes)

    tf_df = pd.DataFrame({'tfname': meta['tfs'], 'smesg': meta['tfs']}, index=meta['tfs'])
    edges = meta['edges'][['source', 'target', 'weight']]

    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):     # Grnm 会打印整张 gene_name_dict, 这里屏蔽
        grn = grnm2.Grnm(rna_like, atac_like, edges, tf_df)
        grn.pca(d1=d1)
        grn.graph()
        grn.gae(d2=d2, epoch_n=epochs, alpha_n=alpha)
    z = np.asarray(grn.z, dtype=np.float32)
    print(f'[Phase1] 图嵌入: z={z.shape}, 节点={len(genes)}, 边={len(edges)}, '
          f'{epochs} epochs, {time.time() - t0:.1f}s')

    np.savez(os.path.join(P1_DIR, 'human_gene_z.npz'), z=z, gene_names=np.array(genes))
    np.savez(os.path.join(P1_DIR, 'human_feature_blocks.npz'),
             genes=np.array(genes), **blocks)
    with open(os.path.join(P1_DIR, 'human_inputs.pkl'), 'wb') as f:
        pickle.dump({'tfs': meta['tfs'], 'tgts': meta['tgts'], 'owners': meta['owners'],
                     'pairs': meta['pairs'], 'edges': meta['edges']}, f)

    plot_phase1_embedding(meta, z, n_edges=len(edges))
    return z


def plot_phase1_embedding(meta, z, n_edges=None):
    """Phase1 嵌入散点图 (可脱离训练单独重绘, 训练后与 --skip-phase1 均调用)。"""
    genes = meta['genes']
    edges_n = len(meta['edges']) if n_edges is None else n_edges
    from sklearn.decomposition import PCA
    pcs = PCA(n_components=2, random_state=0).fit_transform(z)
    is_tf = np.isin(genes, meta['tfs'])
    is_owner = np.isin(genes, meta['owners'])
    only_owner = is_owner & ~is_tf
    rest = ~is_tf & ~is_owner
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(pcs[rest, 0], pcs[rest, 1], s=8, c='#BEBEBE', alpha=0.6,
               edgecolors='none', label=f'其他基因 ({int(rest.sum())})')
    ax.scatter(pcs[only_owner, 0], pcs[only_owner, 1], s=18, c=C2, alpha=0.85,
               edgecolors='none', label=f'motif 拥有者 ({int(only_owner.sum())})')
    ax.scatter(pcs[is_tf, 0], pcs[is_tf, 1], s=26, c=C3, alpha=0.9,
               edgecolors='none', label=f'GLM 中的 TF ({int(is_tf.sum())})')
    if ROOT_GENE in genes:
        k = genes.index(ROOT_GENE)
        ax.scatter([pcs[k, 0]], [pcs[k, 1]], s=200, facecolors='none',
                   edgecolors='#2C3E50', linewidths=1.6)
        ax.annotate(ROOT_GENE, (pcs[k, 0], pcs[k, 1]), xytext=(6, 6),
                    textcoords='offset points', fontsize=11,
                    fontweight='bold', color='#1a1a2e')
    ax.set_xlabel('z PC1'); ax.set_ylabel('z PC2')
    ax.set_title(f'Phase1 (human): GRNM embedding\n{len(genes)} genes × {z.shape[1]}-dim z, '
                 f'{edges_n} edges (GLM + motif)', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    out = os.path.join(PNG_DIR, 'human_phase1_embedding.png')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'[Phase1] 图 -> {out}')
    return z


# ================================================================
# Phase 2: 调控输入谱 -> z (残差网络)
# ================================================================

def run_phase2(meta, blocks, z1, epochs=300, batch=64, lr=3e-4, warmup=30,
               hidden=256, bottleneck=64, n_blocks=3, dropout=0.1,
               val_split=0.15, patience=60, seed=42, device='cpu'):
    genes = meta['genes']
    X_raw = blocks['blk_in']                       # (n, #GLM target) 调控输入谱
    has_profile = np.abs(X_raw).sum(axis=1) > 0
    print(f'[Phase2] 有调控输入谱的基因: {int(has_profile.sum())}/{len(genes)}')

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    idx_all = np.where(has_profile)[0]
    tr_idx, va_idx = train_test_split(idx_all, test_size=val_split, random_state=seed)
    x_mean = X_raw[tr_idx].mean(axis=0)
    x_std = X_raw[tr_idx].std(axis=0); x_std[x_std == 0] = 1.0
    X_norm = (X_raw - x_mean) / x_std

    z_mean = z1[tr_idx].mean(axis=0)
    z_std = z1[tr_idx].std(axis=0); z_std[z_std == 0] = 1.0
    Z_norm = (z1 - z_mean) / z_std

    X_tr = torch.FloatTensor(X_norm[tr_idx]); y_tr = torch.FloatTensor(Z_norm[tr_idx])
    X_va = torch.FloatTensor(X_norm[va_idx]); y_va = torch.FloatTensor(Z_norm[va_idx])
    tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    va_ld = DataLoader(TensorDataset(X_va, y_va), batch_size=batch * 4)

    model = RNA2Z(X_raw.shape[1], Z_DIM, hidden=hidden, bottleneck=bottleneck,
                  n_blocks=n_blocks, dropout=dropout).to(device)
    print(f'[Phase2] 残差网络参数: {sum(p.numel() for p in model.parameters()):,}')

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = LambdaLR(opt, lambda ep: lr_warmup_cosine(ep, warmup, epochs, 1.0))
    crit = nn.MSELoss()

    best_val, best_st, best_ep, bad = float('inf'), None, 0, 0
    hist = {'train': [], 'val': []}
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tl = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item() * xb.size(0)
        tl /= len(tr_idx)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                vl += crit(model(xb), yb).item() * xb.size(0)
        vl /= len(va_idx)
        sched.step()
        hist['train'].append(tl); hist['val'].append(vl)

        if vl < best_val:
            best_val, best_ep, bad = vl, ep, 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 50 == 0 or ep == epochs - 1 or bad >= patience:
            print(f'  Ep {ep:4d} | Train {tl:.5f} | Val {vl:.5f} | best {best_val:.5f} (ep{best_ep})')
        if bad >= patience:
            print(f'  早停 @ epoch {ep}'); break
    model.load_state_dict(best_st)
    print(f'[Phase2] 最优 Val Loss {best_val:.5f} (ep {best_ep}), {time.time() - t0:.1f}s')

    def _metrics(sel, pred_norm):
        zt = Z_norm[sel] * z_std + z_mean
        zp = pred_norm * z_std + z_mean
        cos = []
        for i in range(len(zt)):
            a, b = zt[i], zp[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-8 and nb > 1e-8:
                cos.append(float(np.dot(a, b) / (na * nb)))
        return (float(mean_squared_error(zt, zp)), float(r2_score(zt, zp)),
                float(np.mean(cos) if cos else 0.0), cos)

    model.eval()
    with torch.no_grad():
        p_tr = model(X_tr.to(device)).cpu().numpy()
        p_va = model(X_va.to(device)).cpu().numpy()
    tr_mse, tr_r2, tr_cos, _ = _metrics(tr_idx, p_tr)
    va_mse, va_r2, va_cos, va_cos_list = _metrics(va_idx, p_va)
    print(f'[Phase2] held-out Val: MSE={va_mse:.4f} R2={va_r2:.4f} CosSim={va_cos:.4f}')
    print(f'[Phase2] train:       MSE={tr_mse:.4f} R2={tr_r2:.4f} CosSim={tr_cos:.4f}')

    # 全基因映射: 有输入谱 -> phase2 z; 否则沿用 phase1 z
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_norm), 4096):
            xb = torch.FloatTensor(X_norm[i:i + 4096]).to(device)
            preds.append(model(xb).cpu().numpy())
    z2_pred = np.vstack(preds) * z_std + z_mean
    z_all = z1.copy()
    z_all[has_profile] = z2_pred[has_profile]
    n_fallback = int((~has_profile).sum())
    print(f'[Phase2] 最终 z: {z_all.shape} (phase2 覆盖 {int(has_profile.sum())}, '
          f'phase1 fallback {n_fallback})')

    np.savez(os.path.join(P2_DIR, 'human_gene_z_phase2.npz'),
             z=z_all, z_phase1=z1, z_phase2=z2_pred,
             gene_names=np.array(genes), has_profile=has_profile)
    torch.save({'model_state': best_st, 'in_dim': int(X_raw.shape[1]), 'out_dim': Z_DIM,
                'x_mean': x_mean, 'x_std': x_std, 'z_mean': z_mean, 'z_std': z_std},
               os.path.join(P2_DIR, 'human_rna2z_model.pt'))
    metrics = dict(val_mse=va_mse, val_r2=va_r2, val_cossim=va_cos,
                   train_mse=tr_mse, train_r2=tr_r2, train_cossim=tr_cos,
                   best_epoch=best_ep, best_val_loss=float(best_val),
                   n_train=int(len(tr_idx)), n_val=int(len(va_idx)),
                   n_with_profile=int(has_profile.sum()), n_fallback=n_fallback,
                   in_dim=int(X_raw.shape[1]), z_dim=Z_DIM)
    with open(os.path.join(P2_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist['train'], label='Train'); axes[0].plot(hist['val'], label='Val')
    axes[0].axvline(best_ep, color='red', ls='--', label=f'best ep{best_ep}')
    axes[0].set_title('Phase2 (human): loss'); axes[0].set_xlabel('epoch'); axes[0].legend()
    pv = (p_va * z_std + z_mean)[:, 0]
    tv = (Z_norm[va_idx] * z_std + z_mean)[:, 0]
    axes[1].scatter(tv, pv, s=8, alpha=0.5, c=C1, edgecolors='none')
    lim = [min(tv.min(), pv.min()), max(tv.max(), pv.max())]
    axes[1].plot(lim, lim, 'r--', lw=1)
    axes[1].set_xlabel('z[0] (phase1)'); axes[1].set_ylabel('z[0] (phase2 预测)')
    axes[1].set_title(f'Val z[0]  R2={va_r2:.3f}')
    axes[2].hist(va_cos_list, bins=30, color='steelblue')
    axes[2].axvline(va_cos, color='red', ls='--', label=f'mean={va_cos:.3f}')
    axes[2].set_xlabel('cosine similarity (val)'); axes[2].legend()
    axes[2].set_title('Phase2: per-gene CosSim')
    fig.suptitle(f'Phase2 (human): 调控输入谱 -> z  (train {len(tr_idx)} / val {len(va_idx)} genes)',
                 fontsize=12, fontweight='bold')
    out = os.path.join(PNG_DIR, 'human_phase2_results.png')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'[Phase2] 图 -> {out}')
    return z_all, metrics


# ================================================================
# Phase 3: 配对 z -> 7 类调控关系
# ================================================================

def _class_thresholds(glm_sig):
    """正/负调控各按 |estimate| 33%/66% 分位切成 3 档。"""
    pos = glm_sig.loc[glm_sig['estimate'] > 0, 'estimate']
    neg = glm_sig.loc[glm_sig['estimate'] < 0, 'estimate']
    q = {}
    if len(pos) >= 3:
        q['pos'] = tuple(float(x) for x in np.percentile(pos, [33.33, 66.67]))
    if len(neg) >= 3:
        q['neg'] = tuple(float(x) for x in np.percentile(neg, [33.33, 66.67]))
    return q


def _to_class(est, q):
    if est > 0:
        if 'pos' not in q:
            return 1
        q1, q2 = q['pos']
        return 1 if est <= q1 else (2 if est <= q2 else 3)
    if 'neg' not in q:
        return -1
    q1, q2 = q['neg']
    return -3 if est <= q1 else (-2 if est <= q2 else -1)


def run_phase3(meta, z, epochs=200, batch=128, lr=1e-4, warmup=10, no_edge_ratio=1.0,
               test_ratio=0.1, seed=42, device='cpu', min_abs_weight=0.5):
    """训练 7 类边分类器, 全量预测并流式写出 human_new_edge.txt。

    返回 (gli3_pred_df, manifest)。gli3_pred_df 是 GLI3 作为 source 的**全部**预测
    (不做阈值抽稀), 供扩展图区分"实测边"与"模型外推边"。
    """
    genes = meta['genes']
    z = np.asarray(z, dtype=np.float32)
    gidx = {g: i for i, g in enumerate(genes)}
    glm_sig = meta['pairs']
    q = _class_thresholds(glm_sig)
    print(f'[Phase3] 类别阈值: 正 {q.get("pos")}, 负 {q.get("neg")}')

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ---- 训练标签: 显著 GLM 对 -> 类别 ----
    pos_pairs, neg_pairs = [], []
    for tf, tgt, est in zip(glm_sig['tf'], glm_sig['target'], glm_sig['estimate']):
        if tf == tgt or tf not in gidx or tgt not in gidx:
            continue
        if est > 0:
            pos_pairs.append((tf, tgt, _to_class(est, q)))
        else:
            neg_pairs.append((tf, tgt, _to_class(est, q)))
    all_pairs = pos_pairs + neg_pairs
    known = {(s, t) for s, t, _ in all_pairs}

    # 无调控负采样 (类别 0), 与其余各类等量
    per_class = max(1, len(all_pairs) // 6)
    n_no_edge = int(per_class * no_edge_ratio)
    no_edge, attempts = [], 0
    while len(no_edge) < n_no_edge and attempts < n_no_edge * 200:
        s = random.choice(genes); t = random.choice(genes)
        attempts += 1
        if s != t and (s, t) not in known and (s, t) not in set(no_edge):
            no_edge.append((s, t))
    all_pairs = all_pairs + [(s, t, 0) for s, t in no_edge]

    labels = np.array([c for _, _, c in all_pairs], dtype=np.int64)
    X = np.zeros((len(all_pairs), Z_DIM * 2), dtype=np.float32)
    for i, (s, t, _) in enumerate(all_pairs):
        X[i] = np.concatenate([z[gidx[s]], z[gidx[t]]])

    print(f'[Phase3] 样本 {len(all_pairs):,}: 正 {len(pos_pairs):,} / 负 {len(neg_pairs):,} / '
          f'无调控 {len(no_edge):,}')
    for lbl in (-3, -2, -1, 0, 1, 2, 3):
        print(f'  类别 {lbl:>2}: {(labels == lbl).sum():,} 对')

    y = labels + 3
    tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=test_ratio,
                                      random_state=seed, stratify=y)
    print(f'[Phase3] 训练 {len(tr_idx):,} / 测试 {len(te_idx):,} (按类别分层)')

    X_tr = torch.FloatTensor(X[tr_idx]); y_tr = torch.LongTensor(y[tr_idx])
    X_te = torch.FloatTensor(X[te_idx]); y_te = torch.LongTensor(y[te_idx])
    tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True)

    counts = np.bincount(y[tr_idx], minlength=7)
    cw = torch.FloatTensor(len(tr_idx) / (7.0 * np.maximum(counts, 1))).to(device)

    model = EdgeClassifier(Z_DIM).to(device)
    print(f'[Phase3] 分类器参数: {sum(p.numel() for p in model.parameters()):,}')

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    warm = warmup if warmup > 0 else max(1, epochs // 20)
    sched = LambdaLR(opt, lambda ep: min(1.0, (ep + 1) / warm) if ep < warm else
                     0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, epochs - warm))))
    crit = nn.CrossEntropyLoss(weight=cw)

    best_loss, best_st, best_ep = float('inf'), None, 0
    hist = {'loss': []}
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tl = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item() * xb.size(0)
        tl /= len(tr_idx)
        sched.step()
        hist['loss'].append(tl)
        if tl < best_loss:
            best_loss, best_ep = tl, ep
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc_tr = (model(X_tr.to(device)).argmax(1) == y_tr.to(device)).float().mean().item()
                acc_te = (model(X_te.to(device)).argmax(1) == y_te.to(device)).float().mean().item()
            print(f'  Ep {ep:4d} | CE {tl:.4f} | best {best_loss:.4f} (ep{best_ep}) | '
                  f'Train Acc {acc_tr:.3f} | Test Acc {acc_te:.3f}')
    model.load_state_dict(best_st)
    print(f'[Phase3] 训练完成 {time.time() - t0:.1f}s, best CE {best_loss:.4f} (ep {best_ep})')

    # ---- 测试集评估 ----
    model.eval()
    with torch.no_grad():
        probs_te = torch.softmax(model(X_te.to(device)), dim=1).cpu().numpy()
        probs_tr = torch.softmax(model(X_tr.to(device)), dim=1).cpu().numpy()
    pred_te = probs_te.argmax(1); pred_tr = probs_tr.argmax(1)
    te_acc = float(np.mean(pred_te == y[te_idx])); tr_acc = float(np.mean(pred_tr == y[tr_idx]))
    names = {-3: '强负', -2: '中负', -1: '弱负', 0: '无调控', 1: '弱正', 2: '中正', 3: '强正'}
    recalls = {}
    print(f'[Phase3] 训练集准确率 {tr_acc:.4f}, 测试集准确率 {te_acc:.4f}')
    for lbl in (-3, -2, -1, 0, 1, 2, 3):
        m = y[te_idx] == lbl + 3
        if m.sum():
            recalls[lbl] = float(np.mean(pred_te[m] == lbl + 3))
            print(f'  {names[lbl]}({lbl}): recall={recalls[lbl]:.4f} ({int(m.sum())} 样本)')

    # ---- 全量预测 ----
    sources = sorted((set(meta['tfs']) | set(meta['owners'])) & set(genes))
    targets = genes
    print(f'[Phase3] 全量预测: source {len(sources)} × target {len(targets)} '
          f'(排除自配对) = {len(sources) * (len(targets) - 1):,} 对')

    z_t = torch.FloatTensor(z).to(device)
    NZ_CLASSES = np.array([-3., -2., -1., 1., 2., 3.], dtype=np.float32)
    NZ_COLS = np.array([0, 1, 2, 4, 5, 6])

    def _predict(src_names, tgt_names):
        """对给定 source/target 批量预测, 返回 [(src, tgt, class, prob, signed_weight)]。"""
        out = []
        for src in src_names:
            src_z = z_t[gidx[src]]
            for j in range(0, len(tgt_names), 20000):
                tb = tgt_names[j:j + 20000]
                bi = [gidx[t] for t in tb]
                z_pairs = torch.cat([src_z.unsqueeze(0).expand(len(bi), -1), z_t[bi]], dim=1)
                with torch.no_grad():
                    probs = torch.softmax(model(z_pairs), dim=1).cpu().numpy()
                pred = probs.argmax(1)
                for k, t in enumerate(tb):
                    if t == src:
                        continue
                    c = int(pred[k]) - 3
                    p = float(probs[k, pred[k]])
                    w = c * p
                    if c == 0:                      # argmax=无调控 → 最强非零类别
                        sub = probs[k, NZ_COLS]
                        b = int(sub.argmax())
                        c2 = int(NZ_CLASSES[b]); p2 = float(sub[b])
                        w = c2 * p2
                        out.append((src, t, 0, p, w))
                    else:
                        out.append((src, t, c, p, w))
        return out

    out_path = os.path.join(P3_DIR, 'human_new_edge.txt')
    tmp_path = out_path + '.tmp'
    n_rows = n_pos = n_neg = 0
    samp_w = []
    with open(tmp_path, 'w', newline='') as f:
        f.write('source\ttarget\tweight\n')
        for si, src in enumerate(sources):
            recs = _predict([src], targets)
            keep = [(t, w) for _, t, _, _, w in recs if abs(w) >= min_abs_weight]
            if keep:
                f.write('\n'.join(f'{src}\t{t}\t{w:.6f}' for t, w in keep) + '\n')
                n_rows += len(keep)
                n_pos += sum(1 for _, w in keep if w > 0)
                n_neg += sum(1 for _, w in keep if w < 0)
                if len(samp_w) < 200000:
                    samp_w.extend(w for _, w in keep[:200])
            if si % max(1, len(sources) // 20) == 0:
                print(f'  source {si + 1}/{len(sources)}, 已写 {n_rows:,} 行')
    os.replace(tmp_path, out_path)
    print(f'[Phase3] 写出 {n_rows:,} 行 (正 {n_pos:,} / 负 {n_neg:,}) -> {out_path}')

    # GLI3 全部预测 (不抽稀), 用于"模型外推新边"判定
    gli3_df = pd.DataFrame()
    if ROOT_GENE in gidx:
        recs = _predict([ROOT_GENE], targets)
        gli3_df = pd.DataFrame(recs, columns=['source', 'target', 'pred_class', 'prob', 'weight'])
        gli3_df = gli3_df[['target', 'pred_class', 'prob', 'weight']]
        gli3_df.to_csv(os.path.join(P3_DIR, 'human_GLI3_predictions.csv'), index=False)
        print(f'[Phase3] GLI3 全量预测 {len(gli3_df)} 条 -> human_GLI3_predictions.csv')

    manifest = {
        'generated_at': pd.Timestamp.now().isoformat(timespec='seconds'),
        'status': 'COMPLETE',
        'data_sources': [os.path.basename(GLM_CSV), os.path.basename(MOTIF_TSV)],
        'n_genes': len(genes), 'n_sources': len(sources), 'n_targets': len(targets),
        'written_rows': n_rows, 'pos': n_pos, 'neg': n_neg,
        'min_abs_weight': min_abs_weight,
        'glm_pairs_used': int(len(glm_sig)),
        'motif_edges_used': int(meta['n_motif_edges']),
        'class_thresholds': {k: list(v) for k, v in q.items()},
        'train_acc': tr_acc, 'test_acc': te_acc, 'recall': recalls,
        'best_epoch': best_ep, 'best_ce': float(best_loss),
        'note': ('weight = 类别 × 置信度; argmax=无调控 的配对用最强非零类别 × 其概率作弱权重; '
                 '按 min_abs_weight 抽稀。全部预测由本仓库两份人类数据训练得到。'),
    }
    with open(os.path.join(P3_DIR, 'human_prediction_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 图
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(hist['loss'], label='Train CE')
    axes[0].set_title('Phase3 (human): training loss'); axes[0].set_xlabel('epoch'); axes[0].legend()
    cm = confusion_matrix(y[te_idx], pred_te)
    im = axes[1].imshow(cm, cmap='Blues')
    tick = ['-3', '-2', '-1', '0', '1', '2', '3']
    axes[1].set_xticks(range(7)); axes[1].set_yticks(range(7))
    axes[1].set_xticklabels(tick); axes[1].set_yticklabels(tick)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    axes[1].set_title(f'Test confusion (acc={te_acc:.3f})')
    for a in range(7):
        for b in range(7):
            axes[1].text(b, a, cm[a, b], ha='center', va='center', fontsize=6)
    plt.colorbar(im, ax=axes[1])
    axes[2].hist(np.asarray(samp_w), bins=50, color='steelblue', range=(-3, 3))
    axes[2].set_xlabel('weight')
    axes[2].set_title(f'New edges (n={n_rows:,}, sampled {len(samp_w):,})')
    fig.suptitle('Phase3 (human): z pair -> 7-class regulatory relation', fontsize=12,
                 fontweight='bold')
    out = os.path.join(PNG_DIR, 'human_phase3_results.png')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'[Phase3] 图 -> {out}')
    return gli3_df, manifest


# ================================================================
# 扩展 GLI3 调控图
# ================================================================

def build_gli3_expanded(meta, gli3_pred_df, top_glm=60, top_new=40, n_hubs=6, n_l2=25,
                        new_min_abs=0.3, watchlist=None):
    """合并三层证据, 生成绘图边表与打分表。

      A 实测   GLM 中 GLI3 的显著直连靶 (按 |estimate| 取前 top_glm 作一级环)
      B 扩展   Phase3 预测的 GLI3->基因, 且该基因在 GLM 中**没有** GLI3 系数 (= 模型外推)
      C motif  GLI3 作为 motif 拥有者命中的基因
      D 两跳   重点基因无直连时: GLI3 -> 中介 -> 基因 (至少一跳来自实测或预测)

    返回 (edges, tag, status, info)
    """
    genes = meta['genes']
    gidx = {g: i for i, g in enumerate(genes)}
    glm_sig = meta['pairs']
    watch = list(dict.fromkeys(watchlist or []))

    # ── A 实测 ──
    gl = glm_sig[glm_sig['tf'] == ROOT_GENE].copy()
    gl['abs'] = gl['estimate'].abs()
    gl = gl.sort_values('abs', ascending=False).reset_index(drop=True)
    measured = {r.target: float(r.estimate) for _, r in gl.iterrows()}
    measured_rank = {r.target: i + 1 for i, (_, r) in enumerate(gl.iterrows())}

    # ── B 预测 (GLI3 作为 source) ──
    pred = {}
    if gli3_pred_df is not None and len(gli3_pred_df):
        pred = {r.target: float(r.weight) for _, r in gli3_pred_df.iterrows()}

    # ── C motif ──
    m2t = meta['m2t']
    motif_rows = m2t[m2t['motif_tf'] == ROOT_GENE]
    motif_hits = (motif_rows.groupby('tf')['prc_similarity'].max()
                  .sort_values(ascending=False))
    motif_hits = {k: float(v) for k, v in motif_hits.items()}
    motif_names = sorted(motif_rows['motif'].astype(str).unique())

    rows, tag = [], {}

    def _add(t, w, kind):
        rows.append({'tf': ROOT_GENE, 'target': t, 'estimate': float(w)})
        tag[t] = kind

    l1_set = set()
    for _, r in gl.head(top_glm).iterrows():          # A
        _add(r.target, r.estimate, 'measured'); l1_set.add(r.target)

    new_pred = sorted(((t, w) for t, w in pred.items()
                       if t not in measured and t not in l1_set and t != ROOT_GENE
                       and t in gidx and abs(w) >= new_min_abs),
                      key=lambda x: -abs(x[1]))
    for t, w in new_pred[:top_new]:                   # B
        _add(t, w, 'predicted'); l1_set.add(t)

    for t, s in motif_hits.items():                   # C
        if t in l1_set or t == ROOT_GENE or t not in gidx:
            continue
        _add(t, s, 'motif'); l1_set.add(t)

    # ── D 两跳 (重点基因) ──
    root_edges = dict(measured)
    for t, w in pred.items():                         # GLI3 侧统一用"绝对权重更大者"
        if t in gidx:
            old = root_edges.get(t)
            if old is None or abs(w) > abs(old):
                root_edges[t] = w
    # 中介 -> 基因 的实测边表 (GLM)
    mid_to_g = {}
    for tf, tgt, est in zip(glm_sig['tf'], glm_sig['target'], glm_sig['estimate']):
        mid_to_g.setdefault(tgt, []).append((tf, float(est)))

    status, forced = {}, []
    for g in watch:
        if g == ROOT_GENE:
            continue
        if g in root_edges:
            w = float(root_edges[g])
            kind = 'measured' if g in measured else 'predicted'
            if g not in l1_set:
                _add(g, w, kind); l1_set.add(g)
            status[g] = (kind + '_direct', w)
            continue
        best = None
        for mid, w2 in mid_to_g.get(g, []):
            w1 = root_edges.get(mid)
            if w1 is None:
                continue
            score = min(abs(w1), abs(w2))
            if best is None or score > best[0]:
                best = (score, mid, w1, w2)
        if best:
            _, mid, w1, w2 = best
            rows.append({'tf': ROOT_GENE, 'target': mid, 'estimate': float(w1)})
            rows.append({'tf': mid, 'target': g, 'estimate': float(w2)})
            if mid not in tag:
                tag[mid] = 'measured' if mid in measured else 'predicted'
            l1_set.add(mid); forced.append(mid)
            status[g] = ('two_hop', mid, float(w1), float(w2))
        else:
            status[g] = ('absent', 'GLM 与预测中均无 GLI3 直连, 也无两跳路径')

    # ── 第二层: hub 的 GLM 下游 ──
    tf_set = set(meta['tfs'])
    hubs = [t for t in dict.fromkeys(r['target'] for r in rows if r['tf'] == ROOT_GENE)
            if t in tf_set]
    hubs = hubs[:n_hubs]
    hubs = list(dict.fromkeys(hubs + [m for m in forced if m in tf_set]))
    for h in hubs:
        sub = glm_sig[(glm_sig['tf'] == h) & (~glm_sig['target'].isin(l1_set))]
        sub = sub.reindex(sub['estimate'].abs().sort_values(ascending=False).index).head(n_l2)
        for _, r in sub.iterrows():
            rows.append({'tf': h, 'target': r.target, 'estimate': float(r.estimate)})

    edges = (pd.DataFrame(rows, columns=['tf', 'target', 'estimate'])
             .drop_duplicates(['tf', 'target'], keep='first'))

    info = dict(measured=measured, measured_rank=measured_rank, pred=pred,
                motif_hits=motif_hits, motif_names=motif_names,
                n_measured=len(measured), n_pred_total=len(pred),
                n_new=min(len(new_pred), top_new), n_motif=sum(1 for v in tag.values() if v == 'motif'),
                hubs=hubs, n_l1=len(l1_set))
    return edges, tag, status, info


def _run_r_tree(edges_csv, root, output_png, n1=60, nhubs=6, n2=25,
                highlight=None, spread_labels=True):
    """调用 draw_grn_tree.R (与 phase4_plot_results.py 相同的调用方式)。"""
    cmd = ['Rscript', R_SCRIPT, '--edges', edges_csv, '--root', root,
           '--output', output_png, '--n1', str(n1), '--nhubs', str(nhubs), '--n2', str(n2)]
    if highlight:
        cmd += ['--highlight', ','.join(highlight)]
    if spread_labels:
        cmd += ['--spread-labels']
    print(f'[Tree] Rscript --root {root} -> {os.path.basename(output_png)}')
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        print(f'[Tree] R STDERR: {res.stderr[:1000]}')
        return False
    for line in res.stdout.strip().splitlines():
        print(f'   {line}')
    return True


def plot_gli3_evidence(info, edges, tag, status, out_png):
    """证据分层图: 实测 / 模型外推 / motif 三类边的权重分布 + 直方图。"""
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    meas = pd.Series(info['measured']).sort_values(key=abs, ascending=False)
    ax1.bar(range(min(40, len(meas))), meas.abs().values[:40], color=C3, alpha=0.85)
    ax1.set_xlabel('GLI3 measured target rank (|estimate|)'); ax1.set_ylabel('|estimate|')
    ax1.set_title(f'A 实测: GLM 中 GLI3 显著靶 {len(meas)} 个\n(展示前 40)', fontsize=10)

    ax2 = fig.add_subplot(1, 3, 2)
    new_pred = pd.Series({t: w for t, w in info['pred'].items() if t not in info['measured']})
    new_pred = new_pred.reindex(new_pred.abs().sort_values(ascending=False).index)
    k = min(40, len(new_pred))
    ax2.bar(range(k), new_pred.abs().values[:k], color=C1, alpha=0.85)
    ax2.set_xlabel('GLI3 predicted target rank (|weight|, 无 GLM 系数)')
    ax2.set_ylabel('|weight|')
    ax2.set_title(f'B 扩展: Phase3 外推新边 {len(new_pred)} 条\n(展示前 40)', fontsize=10)

    ax3 = fig.add_subplot(1, 3, 3)
    vals = [list(pd.Series(info['measured']).abs()), list(pd.Series(info['pred']).abs()) or [0.0],
            list(pd.Series(info['motif_hits']).abs()) or [0.0]]
    parts = ax3.boxplot(vals, labels=['A 实测\n(GLM |estimate|)', 'B 扩展\n(预测 |weight|)',
                                      'C motif\n(prc_similarity)'], patch_artist=True,
                        showfliers=False, widths=0.55)
    for patch, c in zip(parts['boxes'], [C3, C1, C2]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax3.set_title('三类证据的强度分布', fontsize=10)
    ax3.tick_params(axis='x', labelsize=8)

    fig.suptitle('GLI3 扩展调控图 — 证据分层 (全部来自 example_data 两份人类数据)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f'[Figure] 证据分层图 -> {out_png}')


def write_report(path, meta, info, status, p1_meta=None, p2_meta=None, p3_meta=None):
    lines = ['=' * 78,
             'GLI3 扩展调控图报告 (人类数据三阶段流程)',
             '=' * 78, '',
             '数据来源 (仅此两份, 未引入任何外部数据):',
             f'  1. {os.path.relpath(GLM_CSV, ROOT_DIR)} — 增强子 GLM 回归系数',
             f'  2. {os.path.relpath(MOTIF_TSV, ROOT_DIR)} — Motif2TF 注释',
             '',
             '一、数据装配',
             f'  基因空间            : {len(meta["genes"])}',
             f'  GLM TF / 靶          : {len(meta["tfs"])} / {len(meta["tgts"])}',
             f'  motif 拥有者         : {len(meta["owners"])}',
             f'  显著 GLM 对 (padj<={PADJ_MAX}): {meta["n_sig_pairs"]}',
             f'  motif2tf 边 (两端在基因空间): {meta["n_motif_edges"]}',
             '']
    if p1_meta:
        lines += ['二、Phase1 GRNM 图嵌入', ] + [
            f'  {k:<22}: {v}' for k, v in p1_meta.items()] + ['']
    if p2_meta:
        lines += ['三、Phase2 调控谱 → z 残差网络', ] + [
            f'  {k:<22}: {v}' for k, v in p2_meta.items()] + ['']
    if p3_meta:
        lines += ['四、Phase3 7 类调控关系分类', ] + [
            f'  {k:<22}: {v}' for k, v in p3_meta.items()] + ['']
    lines += ['五、GLI3 扩展图构成',
              f'  A 实测直连 (GLM)     : {info["n_measured"]} 个靶 (图上取 |estimate| 前 60)',
              f'  B 模型外推 (Phase3)  : {info["n_new"]} 条 (GLM 中无 GLI3 系数的基因)',
              f'  C motif 命中         : {info["n_motif"]} 个 (motif {", ".join(info["motif_names"]) or "-"})',
              f'  图中一级节点合计     : {info["n_l1"]}',
              f'  第二层 hubs          : {", ".join(info["hubs"]) or "-"}',
              '',
              '六、重点基因状态 (direct=直连; two_hop=两跳; absent=不可达)']
    for g, st in status.items():
        if st[0].endswith('_direct'):
            lines.append(f'  {g:<8} {st[0]:<16} weight={st[1]:+.4f}')
        elif st[0] == 'two_hop':
            lines.append(f'  {g:<8} {"two_hop":<16} GLI3 -> {st[1]} (w={st[2]:+.4f}) -> {g} (w={st[3]:+.4f})')
        else:
            lines.append(f'  {g:<8} {st[0]:<16} {st[1]}')
    lines += ['',
              '说明: B 类边是模型在"没有 GLM 系数的基因"上给出的预测, 属于外推结果, '
              '不等同于实验验证; 图上与 A/C 混排只是为了呈现完整候选网络。',
              '逐条分数见 temp_tree_data/human_gli3_edge_scores.csv。']
    txt = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(txt)


# ================================================================
# 主函数
# ================================================================

def main():
    ap = argparse.ArgumentParser(description='人类数据三阶段 GRN 流程 + 扩展 GLI3 调控图')
    ap.add_argument('--phase1-epochs', type=int, default=150)
    ap.add_argument('--phase1-d1', type=int, default=60, help='PCA 维数')
    ap.add_argument('--phase1-d2', type=int, default=8, help='GAE 嵌入维数 (z 输出固定 64)')
    ap.add_argument('--phase2-epochs', type=int, default=300)
    ap.add_argument('--phase2-patience', type=int, default=60)
    ap.add_argument('--phase3-epochs', type=int, default=200)
    ap.add_argument('--min-abs-weight', type=float, default=0.5,
                    help='Phase3 输出边的 |weight| 抽稀阈值')
    ap.add_argument('--new-min-abs', type=float, default=0.3,
                    help='扩展图中"模型外推边"的最小 |weight|')
    ap.add_argument('--top-glm', type=int, default=60, help='扩展图 GLI3 实测直连展示条数')
    ap.add_argument('--top-new', type=int, default=40, help='扩展图 GLI3 外推新边展示条数')
    ap.add_argument('--n-hubs', type=int, default=6)
    ap.add_argument('--n-l2', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--skip-phase1', action='store_true')
    ap.add_argument('--skip-phase2', action='store_true')
    ap.add_argument('--skip-phase3', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print('=' * 78)
    print('人类数据三阶段流程 (phase1 GRNM → phase2 调控谱→z → phase3 边分类)')
    print('=' * 78)

    meta, blocks = load_human_inputs()
    genes = meta['genes']

    # ── Phase 1 ──
    z1_path = os.path.join(P1_DIR, 'human_gene_z.npz')
    p1_meta = None
    if args.skip_phase1 and os.path.exists(z1_path):
        z1 = np.load(z1_path, allow_pickle=True)['z']
        print(f'[Phase1] 跳过训练, 复用 {z1_path} z={z1.shape}')
        plot_phase1_embedding(meta, z1)     # 复用缓存时也刷新图/字体
        p1_meta = {'z_shape': f'{z1.shape[0]} genes × {z1.shape[1]} dim',
                   'nodes': len(genes), 'edges': len(meta['edges']),
                   'feature_blocks': (f'out {blocks["blk_out"].shape[1]} + '
                                      f'in {blocks["blk_in"].shape[1]} + '
                                      f'motif {blocks["blk_mot"].shape[1]} + '
                                      f'owner {blocks["blk_own"].shape[1]}'),
                   'epochs': '(reused from cache)'}
    else:
        z1 = run_phase1(meta, blocks, d1=args.phase1_d1, d2=args.phase1_d2,
                        epochs=args.phase1_epochs)
        p1_meta = {'z_shape': f'{z1.shape[0]} genes × {z1.shape[1]} dim',
                   'nodes': len(genes), 'edges': len(meta['edges']),
                   'feature_blocks': (f'out {blocks["blk_out"].shape[1]} + '
                                      f'in {blocks["blk_in"].shape[1]} + '
                                      f'motif {blocks["blk_mot"].shape[1]} + '
                                      f'owner {blocks["blk_own"].shape[1]}'),
                   'epochs': args.phase1_epochs}

    # ── Phase 2 ──
    z2_path = os.path.join(P2_DIR, 'human_gene_z_phase2.npz')
    p2_meta = None
    if args.skip_phase2 and os.path.exists(z2_path):
        z = np.load(z2_path, allow_pickle=True)['z']
        print(f'[Phase2] 跳过训练, 复用 {z2_path} z={z.shape}')
        mp = os.path.join(P2_DIR, 'metrics.json')
        if os.path.exists(mp):
            p2_meta = json.load(open(mp, encoding='utf-8'))
    else:
        z, p2_meta = run_phase2(meta, blocks, z1, epochs=args.phase2_epochs,
                                patience=args.phase2_patience, seed=args.seed,
                                device=args.device)

    # ── Phase 3 ──
    pred_csv = os.path.join(P3_DIR, 'human_GLI3_predictions.csv')
    p3_meta = None
    if args.skip_phase3 and os.path.exists(pred_csv):
        gli3_pred = pd.read_csv(pred_csv)
        print(f'[Phase3] 跳过训练, 复用 {pred_csv} ({len(gli3_pred)} 条 GLI3 预测)')
        mpath = os.path.join(P3_DIR, 'human_prediction_manifest.json')
        if os.path.exists(mpath):
            p3_meta = json.load(open(mpath, encoding='utf-8'))
    else:
        gli3_pred, p3_meta = run_phase3(meta, z, epochs=args.phase3_epochs, seed=args.seed,
                                        device=args.device,
                                        min_abs_weight=args.min_abs_weight)

    # ── 扩展 GLI3 调控图 ──
    print('=' * 78)
    print('扩展 GLI3 调控图')
    print('=' * 78)
    edges, tag, status, info = build_gli3_expanded(
        meta, gli3_pred, top_glm=args.top_glm, top_new=args.top_new,
        n_hubs=args.n_hubs, n_l2=args.n_l2, new_min_abs=args.new_min_abs,
        watchlist=GLI3_WATCHLIST)

    csv_path = os.path.join(TEMP_DIR, 'human_gli3_expanded_edges.csv')
    edges.to_csv(csv_path, index=False)
    n_l1 = int((edges['tf'] == ROOT_GENE).sum())
    print(f'边表: {len(edges)} 条 (GLI3 一级 {n_l1}, 其余 {len(edges) - n_l1}) -> {csv_path}')

    # 打分明细
    score_rows = []
    for t, kind in tag.items():
        score_rows.append({
            'target': t, 'evidence': kind,
            'glm_estimate': info['measured'].get(t, ''),
            'glm_rank': info['measured_rank'].get(t, ''),
            'phase3_weight': info['pred'].get(t, ''),
            'motif_similarity': info['motif_hits'].get(t, ''),
        })
    scores = pd.DataFrame(score_rows).sort_values(
        ['evidence', 'glm_rank'] if 'glm_rank' in score_rows[0] else ['evidence'])
    scores_path = os.path.join(TEMP_DIR, 'human_gli3_edge_scores.csv')
    scores.to_csv(scores_path, index=False)
    print(f'打分明细: {len(scores)} 行 -> {scores_path}')

    out_png = os.path.join(PNG_DIR, 'human_gli3_grn_expanded.png')
    ok = _run_r_tree(csv_path, ROOT_GENE, out_png, n1=args.top_glm + args.top_new + 10,
                     nhubs=args.n_hubs, n2=args.n_l2,
                     highlight=GLI3_WATCHLIST, spread_labels=True)
    print(f'主树图: {"OK" if ok else "FAILED"} -> {out_png}')

    plot_gli3_evidence(info, edges, tag, status,
                       os.path.join(PNG_DIR, 'human_gli3_evidence.png'))

    report = os.path.join(OUTPUT_DIR, 'human_GLI3_report.txt')
    write_report(report, meta, info, status, p1_meta=p1_meta, p2_meta=p2_meta, p3_meta=p3_meta)
    print(f'\n报告 -> {report}')

    if not ok:
        raise SystemExit('主树图生成失败, 请检查上面的 R STDERR')


if __name__ == '__main__':
    main()
