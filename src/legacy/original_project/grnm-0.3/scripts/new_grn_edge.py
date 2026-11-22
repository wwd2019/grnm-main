"""
new_grn_edge.py — 新基因调控关系发现模型
================================================================
两阶段流程:

  Phase 1 — 训练:
    1. 加载新 RNA 数据 (data/rna_pca_new.csv, SMED 格式, 50106 genes × 64 PCs)
    2. 通过 SMED→SMESG 映射表与 ATAC (example_data/atac_pca.csv) 取交集
    3. 用交集基因运行完整 GRNM 流程 (PCA → Graph → GAE) → 得到中间向量 z
    4. 训练神经网络: 标准化 RNA (64 PCs) → 预测 z (64 dim)

  Phase 2 — 发现新调控关系:
    5. 对未在交集中的新基因，用训练好的模型预测其 z 向量
    6. 在已知基因的 z 空间中做最近邻检索
    7. 输出候选的新调控关系

使用方法:
    python new_grn_edge.py              # Phase 1: 训练模型
    python new_grn_edge.py --discover   # Phase 2: 发现新调控关系
"""

import os
import sys
import argparse
import pickle
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

# ⚠️ matplotlib 后端必须在导入 pyplot 之前设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ⚠️ torch/dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# 导入 GRNM 模型（内部会导入 torch/dgl）
import grnm2

# ────────────────────────────────────────────────────────────────
# 路径配置
# ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'example_data')
NEW_DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
PNG_DIR = os.path.join(ROOT_DIR, 'figures')
RESULT_DIR = os.path.join(ROOT_DIR, 'outputs', 'new_edge_results')

for d in [MODEL_DIR, PNG_DIR, RESULT_DIR, NEW_DATA_DIR]:
    os.makedirs(d, exist_ok=True)


# ================================================================
# 1. 神经网络模型: RNA → z
# ================================================================
class ResidualBlock(nn.Module):
    """带 BatchNorm 和 Dropout 的残差块"""

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class RNA2LatentPredictor(nn.Module):
    """
    从 RNA 特征预测 GRNM 中间向量 z 的神经网络。

    Input:  标准化后的 RNA PCA 特征 (n_genes, rna_dim)
    Output: 预测的 z 向量 (n_genes, z_dim=64)

    学习从 RNA 单一模态映射到 RNA+ATAC 联合隐空间。
    """

    def __init__(self, rna_dim, z_dim=64, hidden_dim=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)


# ================================================================
# 2. 数据加载: 新 RNA (SMED) + 旧 ATAC (SMESG) + SMED→SMESG 映射
# ================================================================
def load_and_align_data():
    """
    加载并对齐数据：
      - 新 RNA: data/rna_pca_new.csv (SMED IDs, 50106 genes × 64 PCs)
      - 旧 ATAC: example_data/atac_pca.csv (SMESG IDs, 1828 genes × 1300 PCs)
      - SMED→SMESG 映射: data/smed_smesg_mapping.csv
      - Edges & TF motif: example_data/

    返回:
        rna_common:     交集基因的 RNA 特征 (SMESG index, 标准化后)
        atac_common:    交集基因的 ATAC 特征 (SMESG index, 标准化后)
        filtered_edges: 过滤后的边
        tf_gene_motif:  TF motif 表
        gene_names:     交集基因名列表 (SMESG)
        smed_to_smesg:  SMED→SMESG 映射 dict
        raw_rna:        原始 RNA 数据 (SMED index, 未过滤)
        rna_mean, rna_std: RNA 标准化的均值和标准差
    """
    print('=' * 60)
    print('Step 1: 加载并对齐数据')
    print('=' * 60)

    # --- 1a. 加载新 RNA 数据 (SMED IDs) ---
    rna_new = pd.read_csv(os.path.join(NEW_DATA_DIR, 'rna_pca_new.csv'))
    # 第一列是 SMED gene ID
    gene_id_col = rna_new.columns[0]
    rna_new = rna_new.rename(columns={gene_id_col: 'smed_id'})
    rna_new = rna_new.set_index('smed_id')
    # 列名 PC1..PC64 → 0..63
    rna_new.columns = [str(c).replace('PC', '') for c in rna_new.columns]
    print(f'  新 RNA: {rna_new.shape} (SMED IDs)')
    print(f'    示例 ID: {list(rna_new.index[:5])}')

    # --- 1b. 加载旧 ATAC 数据 (SMESG IDs) ---
    atac = pd.read_csv(os.path.join(DATA_DIR, 'atac_pca.csv'), index_col=0)
    atac = atac[~atac.index.str.contains('Unknown', na=False)]
    print(f'  旧 ATAC: {atac.shape} (SMESG IDs)')
    print(f'    示例 ID: {list(atac.index[:5])}')

    # --- 1c. 加载 SMED→SMESG 映射 ---
    mapping = pd.read_csv(os.path.join(NEW_DATA_DIR, 'smed_smesg_mapping.csv'))
    smed_to_smesg = dict(zip(mapping['smed_id'], mapping['smesg_id']))
    print(f'  映射表: {len(smed_to_smesg)} SMED → SMESG')

    # --- 1d. 取交集 ---
    # RNA (SMED) → 映射 → SMESG → 与 ATAC (SMESG) 取交集
    rna_smed_set = set(rna_new.index)
    atac_smesg_set = set(atac.index)
    # 在 RNA 中且能映射的 SMED
    mapped_smed_in_rna = rna_smed_set & set(smed_to_smesg.keys())
    # 构建正向: SMED(在RNA中) → SMESG
    smed_to_smesg_in_rna = {s: smed_to_smesg[s] for s in mapped_smed_in_rna}
    # 反向: SMESG → SMED(在RNA中), 处理多对一 (取第一个)
    smesg_to_smed_in_rna = {}
    for smed, smesg in smed_to_smesg_in_rna.items():
        if smesg not in smesg_to_smed_in_rna:
            smesg_to_smed_in_rna[smesg] = smed
    # SMESG 交集
    mapped_smesg = set(smed_to_smesg_in_rna.values())
    common_smesg = sorted(mapped_smesg & atac_smesg_set)
    print(f'  RNA SMED 总数: {len(rna_smed_set)}')
    print(f'  可映射的 SMED: {len(mapped_smed_in_rna)}')
    print(f'  映射后 SMESG ∩ ATAC 交集: {len(common_smesg)}')

    # --- 1e. 提取交集基因的特征 ---
    # 对每个 SMESG 找到它在 RNA 中对应的 SMED
    common_smed = [smesg_to_smed_in_rna[s] for s in common_smesg]

    rna_common_raw = rna_new.loc[common_smed].copy()
    rna_common_raw.index = common_smesg  # 改用 SMESG 作为 index
    print(f'  交集 RNA: {rna_common_raw.shape}')

    atac_common_raw = atac.loc[common_smesg].copy()
    print(f'  交集 ATAC: {atac_common_raw.shape}')

    # --- 1f. Z-score 标准化 ---
    rna_mean = rna_common_raw.mean(axis=0)
    rna_std = rna_common_raw.std(axis=0).replace(0, 1.0)
    rna_common = (rna_common_raw - rna_mean) / rna_std

    atac_mean = atac_common_raw.mean(axis=0)
    atac_std = atac_common_raw.std(axis=0).replace(0, 1.0)
    atac_common = (atac_common_raw - atac_mean) / atac_std
    print('  已执行 Z-score 标准化')

    # --- 1g. 加载 Edges & TF motif ---
    edges = pd.read_csv(os.path.join(DATA_DIR, 'edges.csv'), index_col=0)
    edges = edges.dropna(subset=['source', 'target'])
    filtered_edges = edges[
        edges['source'].isin(common_smesg) &
        edges['target'].isin(common_smesg)
    ].copy()
    filtered_edges.loc[:, 'weight'] = filtered_edges['weight'] * 100
    print(f'  Edges (交集内): {len(filtered_edges)}')

    tf_gene_motif = pd.read_csv(os.path.join(DATA_DIR, 'tf_gene_motif.csv'), index_col=0)
    print(f'  TF motif: {len(tf_gene_motif)}')

    # --- 1h. 排序（保持与 example 一致）---
    edge_genes_set = set(filtered_edges['source']).union(set(filtered_edges['target']))
    ordered_genes = [g for g in common_smesg if g in edge_genes_set]

    rna_common = rna_common.loc[ordered_genes]
    atac_common = atac_common.loc[ordered_genes]

    print(f'  最终使用基因数: {len(ordered_genes)}')
    print(f'    (在 edges 中出现的交集基因)')

    return (rna_common, atac_common, filtered_edges, tf_gene_motif,
            ordered_genes, rna_new, smed_to_smesg, smesg_to_smed_in_rna,
            rna_mean, rna_std, rna_common_raw, common_smed)


# ================================================================
# 3. 运行 GRNM 流程获取中间向量 z (ground truth)
# ================================================================
def run_grnm_pipeline(rna_common, atac_common, filtered_edges, tf_gene_motif,
                      d1=60, d2=8, epoch_n=100, alpha_n=0.01):
    """
    运行完整 GRNM 流程: PCA → Graph → GAE → 提取中间向量 z

    Returns:
        z:             GAE 中间向量 (n_genes, 64)
        pca_r_data:    RNA PCA 降维后 (n_genes, d1)
        pca_a_data:    ATAC PCA 降维后 (n_genes, d1)
        all_features:  拼接特征 (n_genes, d1*2)
        grn_obj:       Grnm 对象
    """
    print('\n' + '=' * 60)
    print('Step 2: 运行 GRNM 流程提取中间向量 z')
    print('=' * 60)

    grn_obj = grnm2.Grnm(rna_common, atac_common, filtered_edges, tf_gene_motif)

    print(f'  PCA: d1={d1}')
    grn_obj.pca(d1=d1)
    print(f'  PCA_r_data: {grn_obj.PCA_r_data.shape}')
    print(f'  PCA_a_data: {grn_obj.PCA_a_data.shape}')

    all_features = np.hstack([grn_obj.PCA_r_data, grn_obj.PCA_a_data])
    print(f'  拼接特征 (all): {all_features.shape}')

    print('  构建图网络...')
    grn_obj.graph()
    print(f'  图: {grn_obj.g}')

    print(f'  GAE: d2={d2}, epochs={epoch_n}, alpha={alpha_n}')
    grn_obj.gae(d2=d2, epoch_n=epoch_n, alpha_n=alpha_n)

    z = grn_obj.z  # (n_genes, 64)
    print(f'  中间向量 z: {z.shape}')

    return z, grn_obj.PCA_r_data, grn_obj.PCA_a_data, all_features, grn_obj


# ================================================================
# 4. 训练 RNA → z 预测模型
# ================================================================
def train_predictor(rna_features, z_target, gene_names,
                    z_dim=64, hidden_dim=256, n_blocks=3,
                    batch_size=64, epochs=300, lr=1e-3,
                    val_split=0.15, device='cpu'):
    """
    训练 RNA → z 的 MLP 回归模型。

    rna_features: 标准化后的 RNA 特征 (n_genes, rna_dim)
    z_target:     GAE 中间向量 (n_genes, z_dim)
    """
    print('\n' + '=' * 60)
    print('Step 3: 训练 RNA → z 预测模型')
    print('=' * 60)

    n_genes, rna_dim = rna_features.shape
    print(f'  样本数: {n_genes}, RNA 特征维度: {rna_dim}, z 维度: {z_dim}')

    # 输入标准化（在 Z-score 基础上再做一次 min-max 或保持）
    rna_values = rna_features.values if hasattr(rna_features, 'values') else rna_features

    # 划分训练/验证集
    indices = np.arange(n_genes)
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    print(f'  训练集: {len(train_idx)}, 验证集: {len(val_idx)}')

    X_train = torch.FloatTensor(rna_values[train_idx])
    y_train = torch.FloatTensor(z_target[train_idx])
    X_val = torch.FloatTensor(rna_values[val_idx])
    y_val = torch.FloatTensor(z_target[val_idx])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型
    model = RNA2LatentPredictor(
        rna_dim=rna_dim, z_dim=z_dim,
        hidden_dim=hidden_dim, n_blocks=n_blocks
    ).to(device)
    print(f'  模型参数量: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=25, min_lr=1e-6, verbose=True)
    criterion = nn.MSELoss()

    # 训练循环
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 60

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_dataset)

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 30 == 0 or epoch == epochs - 1 or patience_counter >= early_stop_patience:
            print(f'  Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')

        if patience_counter >= early_stop_patience:
            print(f'  早停于 epoch {epoch}')
            break

    model.load_state_dict(best_model_state)
    print(f'  最佳 Val Loss: {best_val_loss:.6f}')

    return model, history, (train_idx, val_idx)


# ================================================================
# 5. 模型评估
# ================================================================
def evaluate_model(model, rna_features, z_target, train_idx, val_idx,
                   gene_names, device='cpu'):
    """全面评估模型性能"""
    print('\n' + '=' * 60)
    print('Step 4: 模型评估')
    print('=' * 60)

    rna_values = rna_features.values if hasattr(rna_features, 'values') else rna_features
    X_all = torch.FloatTensor(rna_values).to(device)

    model.eval()
    with torch.no_grad():
        z_pred_all = model(X_all).cpu().numpy()

    mse = mean_squared_error(z_target, z_pred_all)
    rmse = np.sqrt(mse)
    per_sample_mse = np.mean((z_target - z_pred_all) ** 2, axis=1)

    dim_pearsons = []
    for d in range(z_target.shape[1]):
        if np.std(z_target[:, d]) > 1e-8 and np.std(z_pred_all[:, d]) > 1e-8:
            r, _ = pearsonr(z_target[:, d], z_pred_all[:, d])
            dim_pearsons.append(r)
    avg_pearson = np.mean(dim_pearsons) if dim_pearsons else 0.0

    cos_sims = []
    for i in range(len(z_target)):
        dot = np.dot(z_target[i], z_pred_all[i])
        norm_t = np.linalg.norm(z_target[i])
        norm_p = np.linalg.norm(z_pred_all[i])
        if norm_t > 1e-8 and norm_p > 1e-8:
            cos_sims.append(dot / (norm_t * norm_p))
    avg_cos_sim = np.mean(cos_sims) if cos_sims else 0.0

    r2 = r2_score(z_target, z_pred_all)

    z_pred_train = z_pred_all[train_idx]
    z_pred_val = z_pred_all[val_idx]
    val_mse = mean_squared_error(z_target[val_idx], z_pred_val)
    train_mse = mean_squared_error(z_target[train_idx], z_pred_train)

    print(f'  整体 MSE:        {mse:.6f}')
    print(f'  整体 RMSE:       {rmse:.6f}')
    print(f'  训练集 MSE:      {train_mse:.6f}')
    print(f'  验证集 MSE:      {val_mse:.6f}')
    print(f'  平均 Pearson r:  {avg_pearson:.4f}')
    print(f'  平均余弦相似度:  {avg_cos_sim:.4f}')
    print(f'  R² Score:        {r2:.4f}')

    best_indices = np.argsort(per_sample_mse)[:10]
    worst_indices = np.argsort(per_sample_mse)[-10:][::-1]

    print(f'\n  预测最好的 5 个基因:')
    for i in best_indices[:5]:
        print(f'    {gene_names[i]}: MSE={per_sample_mse[i]:.6f}, CosSim={cos_sims[i]:.4f}')
    print(f'  预测最差的 5 个基因:')
    for i in worst_indices[:5]:
        print(f'    {gene_names[i]}: MSE={per_sample_mse[i]:.6f}, CosSim={cos_sims[i]:.4f}')

    return {
        'mse': mse, 'rmse': rmse, 'train_mse': train_mse, 'val_mse': val_mse,
        'avg_pearson': avg_pearson, 'avg_cos_sim': avg_cos_sim, 'r2': r2,
        'per_sample_mse': per_sample_mse, 'z_pred': z_pred_all,
        'cos_sims': cos_sims,
    }


# ================================================================
# 6. 可视化
# ================================================================
def plot_training_results(history, eval_results, z_target, gene_names):
    """生成训练曲线和评估图表"""
    print('\n' + '=' * 60)
    print('Step 5: 生成可视化')
    print('=' * 60)

    z_pred = eval_results['z_pred']
    cos_sims = eval_results['cos_sims']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) 训练曲线
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    ax.plot(history['val_loss'], label='Val Loss', alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (b) z[0] 预测 vs 真实
    ax = axes[0, 1]
    ax.scatter(z_target[:, 0], z_pred[:, 0], alpha=0.3, s=8, c='steelblue')
    vmin, vmax = z_target[:, 0].min(), z_target[:, 0].max()
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.7)
    ax.set_xlabel('True z[0]'); ax.set_ylabel('Predicted z[0]')
    ax.set_title(f'z[0] Prediction (R={eval_results["avg_pearson"]:.3f})')
    ax.grid(True, alpha=0.3)

    # (c) z[1] 预测 vs 真实
    ax = axes[0, 2]
    ax.scatter(z_target[:, 1], z_pred[:, 1], alpha=0.3, s=8, c='seagreen')
    vmin, vmax = z_target[:, 1].min(), z_target[:, 1].max()
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.7)
    ax.set_xlabel('True z[1]'); ax.set_ylabel('Predicted z[1]')
    ax.set_title('z[1] Prediction')
    ax.grid(True, alpha=0.3)

    # (d) 每基因 MSE 分布
    ax = axes[1, 0]
    ax.hist(eval_results['per_sample_mse'], bins=50, color='steelblue',
            edgecolor='white', alpha=0.8)
    ax.axvline(eval_results['mse'], color='red', linestyle='--',
               label=f'Mean MSE = {eval_results["mse"]:.4f}')
    ax.set_xlabel('Per-gene MSE'); ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Per-Gene Prediction Error')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (e) 每维度 Pearson r
    ax = axes[1, 1]
    dim_pearsons = []
    for d in range(z_target.shape[1]):
        if np.std(z_target[:, d]) > 1e-8 and np.std(z_pred[:, d]) > 1e-8:
            r, _ = pearsonr(z_target[:, d], z_pred[:, d])
            dim_pearsons.append(r)
    ax.bar(range(len(dim_pearsons)), dim_pearsons, color='steelblue', alpha=0.8)
    ax.axhline(y=np.mean(dim_pearsons), color='red', linestyle='--',
               label=f'Mean = {np.mean(dim_pearsons):.3f}')
    ax.set_xlabel('z Dimension'); ax.set_ylabel('Pearson r')
    ax.set_title('Per-Dimension Prediction Correlation')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (f) 余弦相似度分布
    ax = axes[1, 2]
    ax.hist(cos_sims, bins=40, color='seagreen', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(cos_sims), color='red', linestyle='--',
               label=f'Mean = {np.mean(cos_sims):.4f}')
    ax.set_xlabel('Cosine Similarity'); ax.set_ylabel('Frequency')
    ax.set_title('Cosine Similarity: Predicted vs True z')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(PNG_DIR, 'new_edge_training_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  图表已保存: {fig_path}')


# ================================================================
# 7. 保存模型和训练产物
# ================================================================
def save_model_and_artifacts(model, rna_mean, rna_std, eval_results,
                             gene_names, z_target, grn_obj, pca_r_data):
    """保存训练好的模型、标准化参数和基因数据库"""
    print('\n' + '=' * 60)
    print('Step 6: 保存模型和训练产物')
    print('=' * 60)

    # 7a. PyTorch 模型
    model_path = os.path.join(MODEL_DIR, 'rna2latent_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'rna_dim': len(rna_mean),
            'z_dim': z_target.shape[1],
            'hidden_dim': 256,
            'n_blocks': 3,
            'dropout': 0.2,
        },
        'rna_mean': rna_mean.values,
        'rna_std': rna_std.values,
        'rna_columns': list(rna_mean.index),
        'eval_metrics': {
            'mse': float(eval_results['mse']),
            'rmse': float(eval_results['rmse']),
            'avg_pearson': float(eval_results['avg_pearson']),
            'avg_cos_sim': float(eval_results['avg_cos_sim']),
            'r2': float(eval_results['r2']),
        },
        'gene_names': gene_names,
        'timestamp': datetime.now().isoformat(),
    }, model_path)
    print(f'  模型: {model_path}')

    # 7b. 标准化参数
    scaler_path = os.path.join(MODEL_DIR, 'rna_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'mean': rna_mean, 'std': rna_std}, f)
    print(f'  标准化参数: {scaler_path}')

    # 7c. 基因 z 向量数据库
    np.savez(
        os.path.join(MODEL_DIR, 'gene_database.npz'),
        z=z_target,
        gene_names=np.array(gene_names),
        z_pred=eval_results['z_pred'],
    )
    print(f'  基因数据库: {os.path.join(MODEL_DIR, "gene_database.npz")}')

    # 7d. PCA_r 参考数据
    np.save(os.path.join(MODEL_DIR, 'pca_r_data.npy'), pca_r_data)
    print(f'  PCA_r 参考: {os.path.join(MODEL_DIR, "pca_r_data.npy")}')


# ================================================================
# 8. 相位二: 发现新调控关系
# ================================================================
def discover_new_edges():
    """
    Phase 2 — 新调控关系发现:
    对未在 GRNM 交集中的新基因（只有 RNA 数据），
    用训练好的 RNA→z 模型预测其在联合隐空间中的位置，
    通过最近邻检索发现潜在的新调控关系。
    """
    print('=' * 60)
    print('Phase 2: 新调控关系发现')
    print(f'启动时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    # --- 加载模型和标准化参数 ---
    model_path = os.path.join(MODEL_DIR, 'rna2latent_model.pt')
    if not os.path.exists(model_path):
        print('错误: 模型文件不存在，请先运行训练 (python new_grn_edge.py)')
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    rna_columns = checkpoint['rna_columns']
    rna_mean = pd.Series(checkpoint['rna_mean'], index=rna_columns)
    rna_std = pd.Series(checkpoint['rna_std'], index=rna_columns)
    train_gene_names = checkpoint['gene_names']

    model = RNA2LatentPredictor(
        rna_dim=model_config['rna_dim'],
        z_dim=model_config['z_dim'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'  模型已加载: {model_path}')

    # --- 加载基因数据库 ---
    db = np.load(os.path.join(MODEL_DIR, 'gene_database.npz'), allow_pickle=True)
    z_db = db['z']           # 已知基因的 z (GAE ground truth)
    gene_names_db = db['gene_names']
    print(f'  已知基因数据库: {len(gene_names_db)} 个基因')

    # --- 加载所有 RNA 数据和映射 ---
    rna_all = pd.read_csv(os.path.join(NEW_DATA_DIR, 'rna_pca_new.csv'))
    gene_id_col = rna_all.columns[0]
    rna_all = rna_all.rename(columns={gene_id_col: 'smed_id'}).set_index('smed_id')
    rna_all.columns = [str(c).replace('PC', '') for c in rna_all.columns]

    mapping = pd.read_csv(os.path.join(NEW_DATA_DIR, 'smed_smesg_mapping.csv'))
    smed_to_smesg = dict(zip(mapping['smed_id'], mapping['smesg_id']))
    # SMESG → SMED: 只包含 RNA 中存在的 SMED
    rna_smed_set_all = set(rna_all.index)
    smesg_to_smed = {}
    for smed, smesg in smed_to_smesg.items():
        if smed in rna_smed_set_all and smesg not in smesg_to_smed:
            smesg_to_smed[smesg] = smed

    # 找出未在训练集中的基因
    train_smed_set = {smesg_to_smed.get(g, None) for g in train_gene_names}
    train_smed_set.discard(None)
    all_smed = set(rna_all.index)
    novel_smed = sorted(all_smed - train_smed_set)
    print(f'  所有 SMED 基因: {len(all_smed)}')
    print(f'  训练集基因 (SMED): {len(train_smed_set)}')
    print(f'  候选新基因 (SMED): {len(novel_smed)}')

    if len(novel_smed) == 0:
        print('  没有可预测的新基因')
        return

    # --- 筛选: 可映射到 SMESG 且有 motif 信息的新基因 ---
    tf_motif = pd.read_csv(os.path.join(DATA_DIR, 'tf_gene_motif.csv'), index_col=0)
    tf_smesg_set = set(tf_motif['smesg'].dropna())

    novel_with_mapping = []
    for smed in novel_smed:
        smesg = smed_to_smesg.get(smed)
        if smesg and smesg in tf_smesg_set:
            novel_with_mapping.append(smed)

    print(f'  可映射且为 TF 的新基因: {len(novel_with_mapping)}')

    if len(novel_with_mapping) == 0:
        print('  没有同时满足映射和TF条件的新基因，放宽条件到所有可映射基因...')
        novel_with_mapping = [s for s in novel_smed if s in smed_to_smesg]
        print(f'  可映射的新基因: {len(novel_with_mapping)}')

    # --- 预测新基因的 z 向量 ---
    novel_rna = rna_all.loc[novel_with_mapping]
    # 对齐列
    novel_rna = novel_rna.reindex(columns=rna_columns, fill_value=0.0)
    novel_rna_norm = (novel_rna - rna_mean) / rna_std
    X_novel = torch.FloatTensor(novel_rna_norm.values)

    with torch.no_grad():
        z_novel = model(X_novel).cpu().numpy()
    print(f'  预测完成: z_novel shape = {z_novel.shape}')

    # --- 最近邻检索: 在已知基因的 z 空间中找最近邻 ---
    sim_matrix = cosine_similarity(z_novel, z_db)
    top_k = 10

    novel_smesg_ids = [smed_to_smesg.get(s, s) for s in novel_with_mapping]

    results = []
    for i, smed_id in enumerate(novel_with_mapping):
        smesg_id = novel_smesg_ids[i]
        top_idx = np.argsort(sim_matrix[i])[-top_k:][::-1]
        neighbors = []
        for idx in top_idx:
            neighbors.append({
                'neighbor_smesg': gene_names_db[idx],
                'cosine_sim': float(sim_matrix[i, idx]),
            })
        results.append({
            'smed_id': smed_id,
            'smesg_id': smesg_id,
            'predicted_z': z_novel[i],
            'neighbors': neighbors,
            'max_similarity': neighbors[0]['cosine_sim'],
        })

    # 按最大相似度排序
    results.sort(key=lambda r: r['max_similarity'], reverse=True)

    # --- 输出结果 ---
    print(f'\n{"="*70}')
    print(f'新调控关系发现结果 — Top 30 候选')
    print(f'{"="*70}')
    print(f'{"排名":<6} {"SMED ID":<18} {"SMESG ID":<18} {"最高相似度":<12} {"最近邻基因 (top 3)":<40}')
    print(f'{"─"*70}')

    for rank, r in enumerate(results[:30], 1):
        top3 = ', '.join([n['neighbor_smesg'] for n in r['neighbors'][:3]])
        print(f'{rank:<6} {r["smed_id"]:<18} {r["smesg_id"]:<18} '
              f'{r["max_similarity"]:<12.4f} {top3:<40}')

    # --- 保存完整结果 ---
    # CSV: 每个新基因的 top-k 近邻
    rows = []
    for r in results:
        for j, nb in enumerate(r['neighbors']):
            rows.append({
                'rank': j + 1,
                'smed_id': r['smed_id'],
                'smesg_id': r['smesg_id'],
                'neighbor_smesg': nb['neighbor_smesg'],
                'cosine_similarity': nb['cosine_sim'],
            })

    result_df = pd.DataFrame(rows)
    result_path = os.path.join(RESULT_DIR, 'novel_gene_neighbors.csv')
    result_df.to_csv(result_path, index=False)
    print(f'\n  近邻结果: {result_path}')

    # NPZ: 预测的 z 向量
    np.savez(
        os.path.join(RESULT_DIR, 'novel_z_predictions.npz'),
        smed_ids=np.array(novel_with_mapping),
        smesg_ids=np.array(novel_smesg_ids),
        z_pred=z_novel,
    )
    print(f'  z 预测: {os.path.join(RESULT_DIR, "novel_z_predictions.npz")}')

    # --- 可视化: 新基因 vs 已知基因的相似度分布 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    max_sims = [r['max_similarity'] for r in results]
    ax.hist(max_sims, bins=40, color='coral', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(max_sims), color='blue', linestyle='--',
               label=f'Mean = {np.mean(max_sims):.4f}')
    ax.set_xlabel('Max Cosine Similarity to Known Genes')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Novel Genes: Max Similarity Distribution\n({len(results)} genes)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    # 高置信度 (>0.8) vs 低置信度的基因数
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    counts = [sum(1 for s in max_sims if s >= t) for t in thresholds]
    ax.bar([f'≥{t}' for t in thresholds], counts, color='steelblue', alpha=0.8)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(c), ha='center', fontsize=10)
    ax.set_ylabel('Number of Novel Genes')
    ax.set_title('Genes Above Similarity Threshold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(PNG_DIR, 'novel_gene_discovery.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  图表: {fig_path}')

    print('\n' + '=' * 60)
    print('新调控关系发现完成！')
    print('=' * 60)


# ================================================================
# 主流程
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='new_grn_edge — 新基因调控关系发现'
    )
    parser.add_argument('--discover', action='store_true',
                        help='Phase 2: 使用已训练模型发现新调控关系')
    parser.add_argument('--epochs', type=int, default=100,
                        help='GAE 训练轮数 (default: 100)')
    parser.add_argument('--pred-epochs', type=int, default=300,
                        help='RNA→z 预测器训练轮数 (default: 300)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # ── Phase 2: 发现新调控关系 ──
    if args.discover:
        discover_new_edges()
        return

    # ── Phase 1: 训练 ──
    print('=' * 60)
    print('Phase 1: new_grn_edge — 训练 RNA → z 模型')
    print(f'启动时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    device = args.device

    # Step 1: 加载并对齐数据
    (rna_common, atac_common, filtered_edges, tf_gene_motif,
     gene_names, rna_all, smed_to_smesg, _smesg_to_smed_in_rna,
     rna_mean, rna_std, rna_common_raw, common_smed) = load_and_align_data()

    # Step 2: 运行 GRNM → 获取 z
    z_target, pca_r, pca_a, all_features, grn_obj = run_grnm_pipeline(
        rna_common, atac_common, filtered_edges, tf_gene_motif,
        epoch_n=args.epochs,
    )

    # Step 3: 训练 RNA → z 预测器
    # 使用标准化后的 RNA (64 PCs) 作为输入
    rna_input = rna_common  # 已标准化
    print(f'\n  预测器输入 (标准化RNA): {rna_input.shape}')

    model, history, (train_idx, val_idx) = train_predictor(
        rna_input, z_target, gene_names,
        epochs=args.pred_epochs, batch_size=args.batch_size,
        lr=args.lr, device=device,
    )

    # Step 4: 评估
    eval_results = evaluate_model(
        model, rna_input, z_target, train_idx, val_idx,
        gene_names, device,
    )

    # Step 5: 可视化
    plot_training_results(history, eval_results, z_target, gene_names)

    # Step 6: 保存
    save_model_and_artifacts(
        model, rna_mean, rna_std, eval_results,
        gene_names, z_target, grn_obj, pca_r,
    )

    print('\n' + '=' * 60)
    print('Phase 1 训练完成！')
    print(f'模型: {MODEL_DIR}/')
    print(f'图表: {PNG_DIR}/')
    print()
    print('运行 Phase 2 发现新调控关系:')
    print('  python new_grn_edge.py --discover')
    print('=' * 60)


if __name__ == '__main__':
    main()
