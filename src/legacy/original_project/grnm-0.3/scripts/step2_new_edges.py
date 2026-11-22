"""
step2_new_edges.py — 第二步: RNA→z 预测 & 新调控关系发现
================================================================
流程:
  1. 加载 step1 产出的 z 向量和标准化参数
  2. 训练 MLP: 标准化 RNA → z
  3. 对未在 GRN 中的新基因预测 z
  4. 在已知基因 z 空间中最近邻检索 → 候选调控关系
  5. 输出结果

依赖: step1_grn_embedding.py 先运行

用法:
  python step2_new_edges.py              # 训练 + 发现
  python step2_new_edges.py --skip-train  # 仅发现 (使用已有模型)
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, '..'))   # 规范化, 去掉 '..'
DATA_DIR = os.path.join(ROOT_DIR, 'example_data')
NEW_DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
RESULT_DIR = os.path.join(ROOT_DIR, 'outputs', 'new_edge_results')
PNG_DIR = os.path.join(ROOT_DIR, 'figures')
for d in [MODEL_DIR, RESULT_DIR, PNG_DIR]:
    os.makedirs(d, exist_ok=True)


# ================================================================
# 模型
# ================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class RNA2Latent(nn.Module):
    def __init__(self, rna_dim, z_dim=64, hidden_dim=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)


# ================================================================
# 训练
# ================================================================
def train_model(rna_features, z_target, gene_names, device='cpu',
                epochs=300, batch_size=64, lr=1e-3, val_split=0.15):
    n_genes, rna_dim = rna_features.shape
    z_dim = z_target.shape[1]
    print(f'  样本: {n_genes}, RNA dim: {rna_dim}, z dim: {z_dim}')

    rna_vals = rna_features.values if hasattr(rna_features, 'values') else rna_features
    indices = np.arange(n_genes)
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    print(f'  训练: {len(train_idx)}, 验证: {len(val_idx)}')

    X_tr = torch.FloatTensor(rna_vals[train_idx]); y_tr = torch.FloatTensor(z_target[train_idx])
    X_val = torch.FloatTensor(rna_vals[val_idx]); y_val = torch.FloatTensor(z_target[val_idx])
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = RNA2Latent(rna_dim, z_dim).to(device)
    print(f'  参数量: {sum(p.numel() for p in model.parameters()):,}')

    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=25, min_lr=1e-6)
    crit = nn.MSELoss()

    best_val = float('inf'); best_state = None; patience = 0
    history = {'train': [], 'val': []}

    for ep in range(epochs):
        model.train(); tr_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(X_tr)

        model.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb), yb).item() * xb.size(0)
        val_loss /= len(X_val)

        sched.step(val_loss)
        history['train'].append(tr_loss); history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if ep % 50 == 0 or patience >= 60 or ep == epochs - 1:
            print(f'  Epoch {ep:4d} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}')
        if patience >= 60:
            print(f'  早停 @ epoch {ep}'); break

    model.load_state_dict(best_state)
    print(f'  最佳 Val Loss: {best_val:.4f}')
    return model, history, (train_idx, val_idx)


# ================================================================
# 评估
# ================================================================
def evaluate(model, rna_features, z_target, train_idx, val_idx, gene_names, device='cpu'):
    rna_vals = rna_features.values if hasattr(rna_features, 'values') else rna_features
    X = torch.FloatTensor(rna_vals).to(device)
    model.eval()
    with torch.no_grad():
        zp = model(X).cpu().numpy()

    mse = mean_squared_error(z_target, zp)
    cos_sims = []
    for i in range(len(z_target)):
        d = np.dot(z_target[i], zp[i])
        nt, np_ = np.linalg.norm(z_target[i]), np.linalg.norm(zp[i])
        if nt > 1e-8 and np_ > 1e-8: cos_sims.append(d / (nt * np_))
    avg_cos = np.mean(cos_sims) if cos_sims else 0
    r2 = r2_score(z_target, zp)
    val_mse = mean_squared_error(z_target[val_idx], zp[val_idx])

    print(f'  整体 MSE: {mse:.4f} | 验证 MSE: {val_mse:.4f} | R²: {r2:.4f} | 平均 CosSim: {avg_cos:.4f}')
    return {'mse': mse, 'val_mse': val_mse, 'r2': r2, 'avg_cos_sim': avg_cos}, zp, cos_sims


# ================================================================
# 新基因发现
# ================================================================
def discover(model, rna_mean, rna_std, z_db, gene_names_db,
             rna_all, smed_to_smesg, smesg_to_smed=None, device='cpu'):
    print(f'\n--- 新基因发现 ---')
    print(f'已知基因: {len(gene_names_db)}')

    # 训练集中的 SMED (与主流程使用同一份映射, 保证一致)
    if smesg_to_smed is None:
        smesg_to_smed = {}
        for s in set(rna_all.index) & set(smed_to_smesg.keys()):
            sg = smed_to_smesg[s]
            if sg not in smesg_to_smed:
                smesg_to_smed[sg] = s
    train_smed = set()
    for sg in gene_names_db:
        if sg in smesg_to_smed:
            train_smed.add(smesg_to_smed[sg])

    # 候选新基因
    novel = sorted(set(rna_all.index) - train_smed)
    # 筛选: 能映射到 SMESG 且是 TF
    tf_smesg = set(pd.read_csv(os.path.join(DATA_DIR, 'tf_gene_motif.csv'), index_col=0)['smesg'].dropna())
    novel_tf = []
    for s in novel:
        sg = smed_to_smesg.get(s)
        if sg and sg in tf_smesg:
            novel_tf.append(s)
    print(f'候选新 SMED: {len(novel):,}')
    print(f'其中可映射 TF: {len(novel_tf)}')

    if len(novel_tf) == 0:
        print('无候选新基因'); return

    # 预测 z
    novel_rna = rna_all.loc[novel_tf].reindex(columns=rna_mean.index, fill_value=0)
    novel_rna_norm = (novel_rna - rna_mean) / rna_std
    X = torch.FloatTensor(novel_rna_norm.values).to(device)
    model.eval()
    with torch.no_grad():
        z_novel = model(X).cpu().numpy()

    # 最近邻
    sim = cosine_similarity(z_novel, z_db)
    top_k = 10

    results = []
    novel_smesg = [smed_to_smesg.get(s, s) for s in novel_tf]
    for i, smed in enumerate(novel_tf):
        top = np.argsort(sim[i])[-top_k:][::-1]
        nbrs = [(gene_names_db[j], float(sim[i, j])) for j in top]
        results.append({'smed': smed, 'smesg': novel_smesg[i], 'neighbors': nbrs, 'max_sim': nbrs[0][1]})
    results.sort(key=lambda r: r['max_sim'], reverse=True)

    # 打印 Top 20
    print(f'\n{"="*80}')
    print(f'Top 20 候选新调控关系')
    print(f'{"="*80}')
    for rk, r in enumerate(results[:20], 1):
        top3 = ', '.join([f'{n}({s:.3f})' for n, s in r['neighbors'][:3]])
        print(f'{rk:3} {r["smed"]:16} {r["smesg"]:18} sim={r["max_sim"]:.4f}  ← {top3}')

    # 保存
    rows = []
    for r in results:
        for j, (n, s) in enumerate(r['neighbors']):
            rows.append({'rank': j+1, 'smed_id': r['smed'], 'smesg_id': r['smesg'],
                         'neighbor_smesg': n, 'cosine_similarity': s})
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, 'novel_gene_neighbors.csv'), index=False)
    np.savez(os.path.join(RESULT_DIR, 'novel_z_predictions.npz'),
             smed_ids=np.array([r['smed'] for r in results]),
             smesg_ids=np.array([r['smesg'] for r in results]),
             z_pred=z_novel)
    print(f'\n结果已保存: {RESULT_DIR}/')

    # 图
    fig, ax = plt.subplots(figsize=(8, 4))
    sims = [r['max_sim'] for r in results]
    ax.hist(sims, bins=20, color='steelblue', edgecolor='white')
    ax.axvline(np.mean(sims), color='red', linestyle='--', label=f'Mean={np.mean(sims):.3f}')
    ax.set_xlabel('Max Cosine Similarity'); ax.set_ylabel('Count')
    ax.set_title(f'Novel TF Genes (n={len(results)})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PNG_DIR, 'step2_novel_genes.png'), dpi=150)
    plt.close()


# ================================================================
# 主流程
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='Step 2: RNA→z 预测 & 新调控关系发现')
    parser.add_argument('--skip-train', action='store_true', help='跳过训练，直接用已有模型')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print('=' * 60)
    print('Step 2: RNA → z 预测 & 新调控关系发现')
    print(f'启动: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    # ── 加载 step1 产物 ──
    print('\n--- 加载 step1 产物 ---')
    db = np.load(os.path.join(MODEL_DIR, 'gene_database.npz'), allow_pickle=True)
    z_target = db['z']
    gene_names_db = list(db['gene_names'])
    print(f'z: {z_target.shape}, 基因: {len(gene_names_db)}')

    with open(os.path.join(MODEL_DIR, 'rna_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    rna_mean, rna_std = scaler['mean'], scaler['std']

    ckpt_path = os.path.join(MODEL_DIR, 'grn_checkpoint.pkl')
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        smed_to_smesg = ckpt['smed_to_smesg']
        smesg_to_smed = ckpt['smesg_to_smed']
        print(f'  已加载 grn_checkpoint.pkl')
    else:
        # step1 未运行过: 从映射表重建 (等价于 step1 的逻辑)
        print('  [警告] 未找到 grn_checkpoint.pkl, 从 smed_smesg_mapping.csv 重建映射表')
        print('         (若需完整 checkpoint, 请先运行 step1_grn_embedding.py)')
        mapping = pd.read_csv(os.path.join(NEW_DATA_DIR, 'smed_smesg_mapping.csv'))
        smed_to_smesg = dict(zip(mapping['smed_id'], mapping['smesg_id']))
        rna_smed_set = set(
            pd.read_csv(os.path.join(NEW_DATA_DIR, 'rna_pca_new.csv'), usecols=[0]).iloc[:, 0]
        )
        smesg_to_smed = {}
        # 与 step1 相同: 只用 dedup 后的 smed_to_smesg; sorted 保证可复现
        for s in sorted(rna_smed_set & set(smed_to_smesg.keys())):
            sg = smed_to_smesg[s]
            if sg not in smesg_to_smed:
                smesg_to_smed[sg] = s
    print(f'  映射表: SMED→SMESG {len(smed_to_smesg)}, SMESG→SMED {len(smesg_to_smed)}')

    # ── 加载全部 RNA ──
    print('\n--- 加载 RNA 数据 ---')
    rna_all = pd.read_csv(os.path.join(NEW_DATA_DIR, 'rna_pca_new.csv'))
    gc = rna_all.columns[0]
    rna_all = rna_all.rename(columns={gc: 'smed_id'}).set_index('smed_id')
    rna_all.columns = [str(c).replace('PC', '') for c in rna_all.columns]
    print(f'RNA 全部: {rna_all.shape}')

    # 准备训练数据：用已知基因的 RNA（标准化后）
    missing = [g for g in gene_names_db if g not in smesg_to_smed]
    if missing:
        print(f'  [警告] {len(missing)} 个已知基因无法映射到 RNA 中的 SMED, 已跳过: {missing[:5]}')
    pair = [(g, smesg_to_smed[g]) for g in gene_names_db if g in smesg_to_smed]
    keep_genes = [g for g, _ in pair]
    train_smed = [s for _, s in pair]
    z_target = z_target[[i for i, g in enumerate(gene_names_db) if g in smesg_to_smed]]
    gene_names_db = keep_genes
    rna_train = rna_all.loc[train_smed].reindex(columns=rna_mean.index, fill_value=0)
    rna_train_norm = (rna_train - rna_mean) / rna_std
    print(f'训练用 RNA: {rna_train_norm.shape}')

    device = args.device

    if not args.skip_train:
        # ── 训练 ──
        print('\n--- 训练 RNA → z 模型 ---')
        model, history, (tr_idx, val_idx) = train_model(
            rna_train_norm, z_target, gene_names_db, device=device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )

        # ── 评估 ──
        print('\n--- 评估 ---')
        metrics, z_pred, cos_sims = evaluate(
            model, rna_train_norm, z_target, tr_idx, val_idx, gene_names_db, device,
        )

        # ── 保存模型 ──
        print('\n--- 保存模型 ---')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {'rna_dim': len(rna_mean), 'z_dim': z_target.shape[1]},
            'rna_mean': rna_mean.values, 'rna_std': rna_std.values,
            'rna_columns': list(rna_mean.index),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'gene_names': gene_names_db,
            'timestamp': datetime.now().isoformat(),
        }, os.path.join(MODEL_DIR, 'rna2latent_model.pt'))
        print(f'  模型已保存: rna2latent_model.pt')

        # 训练曲线
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history['train'], label='Train'); axes[0].plot(history['val'], label='Val')
        axes[0].set_title('Loss'); axes[0].legend()
        axes[1].scatter(z_target[:, 0], z_pred[:, 0], s=5, alpha=0.3)
        axes[1].set_xlabel('True z[0]'); axes[1].set_ylabel('Pred z[0]'); axes[1].set_title('z[0]')
        axes[2].hist(cos_sims, bins=30, color='steelblue')
        axes[2].axvline(np.mean(cos_sims), color='red', linestyle='--')
        axes[2].set_xlabel('Cosine Similarity'); axes[2].set_title(f'CosSim (mean={np.mean(cos_sims):.3f})')
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, 'step2_training.png'), dpi=150)
        plt.close()
        print(f'  图表: step2_training.png')
    else:
        # 加载已有模型
        ckpt_m = torch.load(os.path.join(MODEL_DIR, 'rna2latent_model.pt'),
                            map_location=device, weights_only=False)
        model = RNA2Latent(ckpt_m['model_config']['rna_dim'], ckpt_m['model_config']['z_dim'])
        model.load_state_dict(ckpt_m['model_state_dict'])
        model.to(device); model.eval()
        print('  已加载已有模型')

    # ── 新基因发现 ──
    discover(model, rna_mean, rna_std, z_target, gene_names_db,
             rna_all, smed_to_smesg, smesg_to_smed=smesg_to_smed, device=device)

    print('\n全部完成!')


if __name__ == '__main__':
    main()
