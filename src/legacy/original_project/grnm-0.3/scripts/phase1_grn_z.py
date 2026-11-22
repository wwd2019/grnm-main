"""
phase1_grn_z.py — 新RNA + 旧ATAC → GRNM → 中间向量 z
================================================================
输入: 新 scRNA PCA (SMED, 64 PCs) + 旧 scATAC PCA (SMESG, 1300 PCs)
      通过 SMED→SMESG 映射取交集基因
输出: 每个交集基因的 64 维中间向量 z

用法: python phase1_grn_z.py
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import grnm2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'example_data')
NEW_DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase1_output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 参数 ──
D1, D2, EPOCHS, ALPHA = 60, 8, 100, 0.01


def main():
    # 1. 加载新 RNA (SMED)
    rna = pd.read_csv(os.path.join(NEW_DATA_DIR, 'rna_pca_new.csv'))
    gc = rna.columns[0]
    rna = rna.rename(columns={gc: 'smed_id'}).set_index('smed_id')
    rna.columns = [str(c).replace('PC', '') for c in rna.columns]
    print(f'新RNA: {rna.shape}')

    # 2. 加载旧 ATAC (SMESG)
    atac = pd.read_csv(os.path.join(DATA_DIR, 'atac_pca.csv'), index_col=0)
    atac = atac[~atac.index.str.contains('Unknown', na=False)]
    print(f'旧ATAC: {atac.shape}')

    # 3. 加载映射 → 取交集
    mapping = pd.read_csv(os.path.join(NEW_DATA_DIR, 'smed_smesg_mapping.csv'))
    smed2smesg = dict(zip(mapping['smed_id'], mapping['smesg_id']))
    # SMESG → SMED (只保留RNA中有的)
    smesg2smed = {}
    for smed, smesg in smed2smesg.items():
        if smed in rna.index and smesg not in smesg2smed:
            smesg2smed[smesg] = smed

    common_smesg = sorted(set(atac.index) & set(smesg2smed.keys()))
    common_smed = [smesg2smed[s] for s in common_smesg]
    print(f'映射交集: {len(common_smesg)}')

    rna_c = rna.loc[common_smed]; rna_c.index = common_smesg
    atac_c = atac.loc[common_smesg]
    print(f'交集 RNA: {rna_c.shape}, ATAC: {atac_c.shape}')

    # 4. 标准化
    rna_mean, rna_std = rna_c.mean(axis=0), rna_c.std(axis=0).replace(0, 1)
    atac_mean, atac_std = atac_c.mean(axis=0), atac_c.std(axis=0).replace(0, 1)
    rna_n = (rna_c - rna_mean) / rna_std
    atac_n = (atac_c - atac_mean) / atac_std

    # 5. Edges & TF motif
    edges = pd.read_csv(os.path.join(DATA_DIR, 'edges.csv'), index_col=0).dropna(subset=['source', 'target'])
    tf_motif = pd.read_csv(os.path.join(DATA_DIR, 'tf_gene_motif.csv'), index_col=0)
    edges_f = edges[edges['source'].isin(common_smesg) & edges['target'].isin(common_smesg)].copy()
    edges_f.loc[:, 'weight'] = edges_f['weight'] * 100

    # 排序: 在 edges 中的基因排前面
    eg = set(edges_f['source']) | set(edges_f['target'])
    ordered = [g for g in common_smesg if g in eg]
    rna_n = rna_n.loc[ordered]; atac_n = atac_n.loc[ordered]
    print(f'Edges: {len(edges_f)}, 最终基因: {len(ordered)}')

    # 6. GRNM
    grn = grnm2.Grnm(rna_n, atac_n, edges_f, tf_motif)
    grn.pca(d1=D1)
    grn.graph()
    grn.gae(d2=D2, epoch_n=EPOCHS, alpha_n=ALPHA)
    z = grn.z
    print(f'z: {z.shape}')

    # 7. 保存
    np.savez(os.path.join(OUT_DIR, 'gene_z.npz'),
             z=z, gene_names=np.array(ordered))
    with open(os.path.join(OUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump({'rna_mean': rna_mean, 'rna_std': rna_std,
                     'atac_mean': atac_mean, 'atac_std': atac_std,
                     'gene_names': ordered,
                     'smed2smesg': smed2smesg, 'smesg2smed': smesg2smed}, f)
    with open(os.path.join(OUT_DIR, 'grn_obj.pkl'), 'wb') as f:
        pickle.dump({'pca_r': grn.PCA_r_data, 'pca_a': grn.PCA_a_data,
                     'gene_names': ordered}, f)

    # 8. 保存交集基因的 RNA 特征 (供 Phase 2 训练用)
    rna_n.index.name = 'smesg_id'
    rna_n.to_csv(os.path.join(OUT_DIR, 'rna_train.csv'))
    print(f'Done → {OUT_DIR}/')
    print(f'基因数={len(ordered)}, z_dim={z.shape[1]}')


if __name__ == '__main__':
    main()
