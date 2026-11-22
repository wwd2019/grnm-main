"""
step1_grn_embedding.py — 第一步：GRNM 训练 & 中间向量提取
================================================================
流程:
  1. 加载新 scRNA PCA (SMED) + 旧 scATAC PCA (SMESG)
  2. 通过 SMED→SMESG 匹配表取交集基因
  3. 标准化 → PCA → Graph → GAE → 中间向量 z
  4. 保存 z 向量、基因列表、标准化参数

输出:
  models/
    gene_database.npz     ← 交集基因的 z 向量 + 基因名
    rna_scaler.pkl        ← RNA 标准化参数
    grn_checkpoint.pkl    ← GRNM 相关数据 (PCA_r, PCA_a 等)

用法:
  python step1_grn_embedding.py
"""

import os
import pickle
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import grnm2

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'example_data')
NEW_DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 可调参数 ──
RNA_PCA_FILE = os.path.join(NEW_DATA_DIR, 'rna_pca_new.csv')   # 新 scRNA PCA
ATAC_PCA_FILE = os.path.join(DATA_DIR, 'atac_pca.csv')         # 旧 scATAC PCA
MAPPING_FILE = os.path.join(NEW_DATA_DIR, 'smed_smesg_mapping.csv')
EDGES_FILE = os.path.join(DATA_DIR, 'edges.csv')
TF_MOTIF_FILE = os.path.join(DATA_DIR, 'tf_gene_motif.csv')

GRNM_D1 = 60       # GRNM 内部 PCA 维度
GRNM_D2 = 8        # GAE 分类头维度
GRNM_EPOCHS = 100  # GAE 训练轮数
GRNM_ALPHA = 0.01  # 联合损失权重
SEED = 42          # 固定随机种子, 保证 z 可复现


def set_seed(seed=SEED):
    """固定所有随机源。GAE/GCN 的权重初始化是随机的,
    不固定的话每次重跑 z 都不同, 下游 step2 的邻居结果也会漂移。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    set_seed()
    print('=' * 60)
    print('Step 1: GRNM 训练 & 中间向量 z 提取')
    print('=' * 60)

    # ──────────────────────────────────────────────
    # 1. 加载数据
    # ──────────────────────────────────────────────
    print('\n--- 加载数据 ---')

    # RNA (SMED)
    rna = pd.read_csv(RNA_PCA_FILE)
    gene_col = rna.columns[0]
    rna = rna.rename(columns={gene_col: 'smed_id'}).set_index('smed_id')
    rna.columns = [str(c).replace('PC', '') for c in rna.columns]
    print(f'RNA: {rna.shape}')

    # ATAC (SMESG)
    atac = pd.read_csv(ATAC_PCA_FILE, index_col=0)
    atac = atac[~atac.index.str.contains('Unknown', na=False)]
    print(f'ATAC: {atac.shape}')

    # SMED → SMESG 映射
    mapping = pd.read_csv(MAPPING_FILE)
    smed_to_smesg = dict(zip(mapping['smed_id'], mapping['smesg_id']))
    print(f'映射表: {len(smed_to_smesg)} SMED → SMESG')

    # ──────────────────────────────────────────────
    # 2. 取交集
    # ──────────────────────────────────────────────
    print('\n--- 取交集 ---')
    rna_smed = set(rna.index)
    atac_smesg = set(atac.index)

    # 在 RNA 中且能映射的 SMED
    mapped_smed = rna_smed & set(smed_to_smesg.keys())
    # SMESG → SMED (只保留 RNA 中存在的)
    # 注意: 用 sorted 保证多对一映射的选择可复现 (set 迭代顺序随 PYTHONHASHSEED 变化)
    smesg_to_smed = {}
    for s in sorted(mapped_smed):
        smesg = smed_to_smesg[s]
        if smesg not in smesg_to_smed:
            smesg_to_smed[smesg] = s

    # 与 ATAC 交集 (SMESG)
    common_smesg = sorted(set(smesg_to_smed.keys()) & atac_smesg)
    common_smed = [smesg_to_smed[s] for s in common_smesg]
    print(f'SMED 可映射: {len(mapped_smed)}')
    print(f'映射后 SMESG ∩ ATAC: {len(common_smesg)}')

    # 提取特征
    rna_common = rna.loc[common_smed].copy()
    rna_common.index = common_smesg
    atac_common = atac.loc[common_smesg].copy()
    print(f'交集 RNA: {rna_common.shape}')
    print(f'交集 ATAC: {atac_common.shape}')

    # ──────────────────────────────────────────────
    # 3. 标准化 & 过滤 edges
    # ──────────────────────────────────────────────
    print('\n--- 标准化 ---')
    rna_mean = rna_common.mean(axis=0)
    rna_std = rna_common.std(axis=0).replace(0, 1.0)
    rna_norm = (rna_common - rna_mean) / rna_std

    atac_mean = atac_common.mean(axis=0)
    atac_std = atac_common.std(axis=0).replace(0, 1.0)
    atac_norm = (atac_common - atac_mean) / atac_std

    # Edges & TF motif
    edges = pd.read_csv(EDGES_FILE, index_col=0).dropna(subset=['source', 'target'])
    tf_motif = pd.read_csv(TF_MOTIF_FILE, index_col=0)

    # 只留两端都在交集基因中的边
    edges_filt = edges[
        edges['source'].isin(common_smesg) &
        edges['target'].isin(common_smesg)
    ].copy()
    edges_filt.loc[:, 'weight'] = edges_filt['weight'] * 100

    # 排序：在 edges 中出现的基因排前面
    edge_genes = set(edges_filt['source']) | set(edges_filt['target'])
    ordered = [g for g in common_smesg if g in edge_genes]
    rna_norm = rna_norm.loc[ordered]
    atac_norm = atac_norm.loc[ordered]

    print(f'Edges 过滤: {len(edges_filt)} 条边')
    print(f'最终基因数: {len(ordered)}')

    # ──────────────────────────────────────────────
    # 4. GRNM 流程
    # ──────────────────────────────────────────────
    print('\n--- GRNM 流程 ---')
    grn = grnm2.Grnm(rna_norm, atac_norm, edges_filt, tf_motif)

    print(f'PCA: d1={GRNM_D1}')
    grn.pca(d1=GRNM_D1)
    print(f'  PCA_r: {grn.PCA_r_data.shape}')
    print(f'  PCA_a: {grn.PCA_a_data.shape}')

    print('Graph...')
    grn.graph()
    print(f'  {grn.g}')

    print(f'GAE: epochs={GRNM_EPOCHS}, alpha={GRNM_ALPHA}')
    grn.gae(d2=GRNM_D2, epoch_n=GRNM_EPOCHS, alpha_n=GRNM_ALPHA)

    z = grn.z  # (n_genes, 64)
    print(f'中间向量 z: {z.shape}')

    # ──────────────────────────────────────────────
    # 5. 保存
    # ──────────────────────────────────────────────
    print('\n--- 保存 ---')

    # 基因数据库: z + 基因名
    np.savez(
        os.path.join(MODEL_DIR, 'gene_database.npz'),
        z=z,
        gene_names=np.array(ordered),
    )
    print(f'  gene_database.npz')

    # RNA 标准化参数
    with open(os.path.join(MODEL_DIR, 'rna_scaler.pkl'), 'wb') as f:
        pickle.dump({'mean': rna_mean, 'std': rna_std}, f)
    print(f'  rna_scaler.pkl')

    # GRN checkpoint (PCA 数据、图等)
    with open(os.path.join(MODEL_DIR, 'grn_checkpoint.pkl'), 'wb') as f:
        pickle.dump({
            'pca_r_data': grn.PCA_r_data,
            'pca_a_data': grn.PCA_a_data,
            'gene_names': ordered,
            'smed_to_smesg': smed_to_smesg,
            'smesg_to_smed': smesg_to_smed,
            'common_smesg': common_smesg,
            'rna_mean': rna_mean,
            'rna_std': rna_std,
            'atac_mean': atac_mean,
            'atac_std': atac_std,
        }, f)
    print(f'  grn_checkpoint.pkl')

    print(f'\nDone! 基因数={len(ordered)}, z_dim={z.shape[1]}')
    print(f'输出目录: {MODEL_DIR}/')


if __name__ == '__main__':
    main()
