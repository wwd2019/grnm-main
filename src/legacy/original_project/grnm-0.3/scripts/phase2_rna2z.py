"""
phase2_rna2z.py — 新 RNA → z 拟合 & 全基因中间向量生成
================================================================
输入:
  - Phase 1 产出的 z 向量 (目标, 旧基因 SMESG)
  - 新 scRNA PCA (SMED, 50K genes)
  - SMED→SMESG 匹配表

流程:
  1. 取交集基因 (新RNA ∩ 匹配表 → SMESG ∩ Phase1基因)
  2. 切 train/val (seeded), 标准化统计量只从 train 计算
  3. 训练残差网络 (ResNet): 标准化新RNA → z, 按 val loss 早停并选最优
  4. held-out 评估 (val 为 headline)
  5. 对所有新RNA基因预测 z
  6. 保存全部预测 z 向量

用法: python phase2_rna2z.py
"""

import os, sys, math, pickle, argparse, random
import grn_common as G
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'example_data')
NEW_DATA_DIR = os.path.join(ROOT_DIR, 'data')
PHASE1_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase1_output')
OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase2_output')
PNG_DIR = os.path.join(ROOT_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)


# ============================================================
# 模型: 现代残差网络 — 预激活 + 门控注意力 (ResNet v2 + SE 风格)
# ============================================================
class GatedPreActResBlock(nn.Module):
    """预激活门控残差块

    y = LN(x + gate(x) ⊙ F(x))
      F:    Linear(dim→bn) → GELU → Dropout → Linear(bn→dim)   (瓶颈结构)
      gate: Linear(dim→dim/16) → GELU → Linear(dim/16→dim) → Sigmoid  (通道注意力)
    """
    def __init__(self, dim, bottleneck=None, dropout=0.1):
        super().__init__()
        bn = bottleneck or max(dim // 4, 32)
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bn)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bn, dim)
        # 门控分支 (SENet 风格通道注意力)
        gate_dim = max(dim // 16, 16)
        self.gate = nn.Sequential(
            nn.Linear(dim, gate_dim), nn.GELU(),
            nn.Linear(gate_dim, dim), nn.Sigmoid())
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        h = self.norm1(x)
        gate = self.gate(h)                      # (B, dim) 门控权重
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return self.norm2(identity + gate * h)


class RNA2Z(nn.Module):
    """RNA → z 现代残差网络 (PreAct + SE gating)

    结构:
      stem: Linear(in→hidden) → LayerNorm → GELU
      body: [GatedPreActResBlock × n_blocks]
      head: LayerNorm → Linear(hidden→out)
    """
    def __init__(self, in_dim, out_dim=64, hidden=512, bottleneck=128,
                 n_blocks=4, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.ModuleList(
            [GatedPreActResBlock(hidden, bottleneck, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))
        self.n_blocks = n_blocks

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def lr_warmup_cosine(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    """线性 warmup + 余弦退火学习率调度"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=50, help='warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--bottleneck', type=int, default=64)
    parser.add_argument('--n-blocks', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='验证集比例 (统计量与早停均基于 val)')
    parser.add_argument('--patience', type=int, default=80,
                        help='val loss 连续无改善的早停轮数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('='*60)
    print('Phase 2: 新 RNA → z 拟合 (train/val 划分 + val 早停)')
    print('='*60)

    # ── 1. 加载 Phase1 的 z ──
    p1 = np.load(os.path.join(PHASE1_DIR, 'gene_z.npz'), allow_pickle=True)
    z_target = p1['z']                    # (n_old, 64)
    p1_genes = list(p1['gene_names'])     # SMESG
    print(f'Phase1 z: {z_target.shape}, 基因: {len(p1_genes)}')

    # ── 2. 加载新 RNA ──
    rna = pd.read_csv(os.path.join(NEW_DATA_DIR, 'rna_pca_new.csv'))
    gc = rna.columns[0]
    rna = rna.rename(columns={gc: 'smed_id'}).set_index('smed_id')
    rna.columns = [str(c).replace('PC', '') for c in rna.columns]
    print(f'新RNA: {rna.shape}')

    # ── 3. 加载映射 → 取交集 ──
    mapping = pd.read_csv(os.path.join(NEW_DATA_DIR, 'smed_smesg_mapping.csv'))
    smed2smesg = dict(zip(mapping['smed_id'], mapping['smesg_id']))
    # 反向: SMESG → SMED (只保留RNA中有的)
    smesg2smed = {}
    for smed, smesg in smed2smesg.items():
        if smed in rna.index and smesg not in smesg2smed:
            smesg2smed[smesg] = smed

    # 交集: Phase1 的 SMESG ∩ 能映射到新RNA的SMESG
    common_smesg = sorted(set(p1_genes) & set(smesg2smed.keys()))
    common_smed = [smesg2smed[s] for s in common_smesg]
    print(f'交集基因: {len(common_smesg)}')

    # 划分 train/val (seeded), 标准化统计量只从 train 计算
    n_common = len(common_smesg)
    tr_idx, va_idx = train_test_split(np.arange(n_common), test_size=args.val_split,
                                      random_state=args.seed)
    print(f'train/val 划分: {len(tr_idx)} / {len(va_idx)} (val={args.val_split})')

    rna_all_raw = rna.loc[common_smed]
    rna_mean = rna_all_raw.iloc[tr_idx].mean(axis=0)
    rna_std = rna_all_raw.iloc[tr_idx].std(axis=0).replace(0, 1)
    rna_norm = (rna_all_raw - rna_mean) / rna_std

    # z 目标对应排列
    p1_gene_to_idx = {g: i for i, g in enumerate(p1_genes)}
    z_idx = [p1_gene_to_idx[s] for s in common_smesg]
    z_raw = z_target[z_idx]
    # z 标准化 (每维度 mean=0, std=1, 统计量只从 train 计算)
    z_mean = z_raw[tr_idx].mean(axis=0)
    z_std = z_raw[tr_idx].std(axis=0); z_std[z_std == 0] = 1.0
    z_norm = (z_raw - z_mean) / z_std
    print(f'交集基因: RNA={rna_all_raw.shape}, z={z_raw.shape}')
    print(f'标准化统计量 (仅train): z mean={z_mean.mean():.3f}, std={z_std.mean():.3f}')

    # ── 4. 训练 (train/val 划分 + val 早停) ──
    device = args.device
    n_tr = len(tr_idx)
    print(f'训练样本: {n_tr}, 验证样本: {len(va_idx)} (按 val loss 早停, patience={args.patience})')

    X_tr_t = torch.FloatTensor(rna_norm.iloc[tr_idx].values); y_tr_t = torch.FloatTensor(z_norm[tr_idx])
    X_va_t = torch.FloatTensor(rna_norm.iloc[va_idx].values); y_va_t = torch.FloatTensor(z_norm[va_idx])
    tr_ld = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=args.batch, shuffle=True)
    va_ld = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=args.batch * 4)

    model = RNA2Z(rna_all_raw.shape[1], 64, hidden=args.hidden, bottleneck=args.bottleneck,
                  n_blocks=args.n_blocks, dropout=args.dropout).to(device)
    print(f'残差网络: {model.n_blocks} 个 GatedPreActResBlock '
          f'(hidden={args.hidden}, bn={args.bottleneck}), 参数量 {sum(p.numel() for p in model.parameters()):,}')

    # AdamW + warmup + 余弦退火
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = LambdaLR(opt, lambda ep: lr_warmup_cosine(ep, args.warmup, args.epochs, 1.0))
    crit = nn.MSELoss()

    best_val, best_st, best_ep, patience_cnt = float('inf'), None, 0, 0
    hist = {'train': [], 'val': []}

    for ep in range(args.epochs):
        model.train(); tl = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = crit(model(xb), yb); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += l.item()*xb.size(0)
        tl /= n_tr

        model.eval(); vl = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                vl += crit(model(xb), yb).item() * xb.size(0)
        vl /= len(va_idx)
        sched.step()
        hist['train'].append(tl); hist['val'].append(vl)

        # 按验证 loss 保存最优
        if vl < best_val:
            best_val, best_st, best_ep = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, ep
            patience_cnt = 0
        else:
            patience_cnt += 1

        if ep % 100 == 0 or ep == args.epochs-1 or patience_cnt >= args.patience:
            print(f'  Ep {ep:4d} | Train {tl:.6f} | Val {vl:.6f} | best {best_val:.6f} (ep{best_ep}) | LR {opt.param_groups[0]["lr"]:.2e}')
        if patience_cnt >= args.patience:
            print(f'  早停 @ epoch {ep} (val loss 连续 {args.patience} 轮无改善)'); break

    model.load_state_dict(best_st)
    print(f'最优 Val Loss: {best_val:.6f} (ep {best_ep})')

    # ── 5. 评估 (反标准化后计算, held-out val 为 headline) ──
    def _metrics(z_true_norm, z_pred_norm):
        zt = z_true_norm * z_std + z_mean
        zp = z_pred_norm * z_std + z_mean
        cos = []
        for i in range(len(zt)):
            d = np.dot(zt[i], zp[i])
            nt, np_ = np.linalg.norm(zt[i]), np.linalg.norm(zp[i])
            if nt > 1e-8 and np_ > 1e-8: cos.append(d/(nt*np_))
        return (mean_squared_error(zt, zp), r2_score(zt, zp),
                float(np.mean(cos)) if cos else 0.0, cos)

    model.eval()
    with torch.no_grad():
        zp_tr = model(X_tr_t.to(device)).cpu().numpy()
        zp_va = model(X_va_t.to(device)).cpu().numpy()
    tr_mse, tr_r2, tr_cos, tr_cos_list = _metrics(z_norm[tr_idx], zp_tr)
    va_mse, va_r2, va_cos, va_cos_list = _metrics(z_norm[va_idx], zp_va)
    print(f'评估 (held-out): Val MSE={va_mse:.4f}, R²={va_r2:.4f}, CosSim={va_cos:.4f}')
    print(f'评估 (训练集):   MSE={tr_mse:.4f}, R²={tr_r2:.4f}, CosSim={tr_cos:.4f}')

    # ── 6. 全基因预测 ──
    print(f'\n全基因预测...')
    rna_all_norm = (rna.reindex(columns=rna_mean.index, fill_value=0) - rna_mean) / rna_std
    batch_size = 4096
    z_all_norm = []
    X_all = torch.FloatTensor(rna_all_norm.values)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_all), batch_size):
            xb = X_all[i:i+batch_size].to(device)
            z_all_norm.append(model(xb).cpu().numpy())
    z_all_norm = np.vstack(z_all_norm)
    z_all = z_all_norm * z_std + z_mean  # 反标准化回原始 z 空间
    print(f'全部 z: {z_all.shape}')

    # ── 7. 保存 ──
    # 为每个 z 行补齐多身份 (SMED / SMESG / symbol), 下游不再靠前缀猜身份
    maps = G.load_id_maps()
    gene_ids = np.array(rna.index).astype(str)
    smesg_ids = np.array([maps['smed2smesg'].get(s, '') for s in gene_ids])
    symbols = np.array([maps['smed2sym'].get(s, s) for s in gene_ids])

    np.savez(os.path.join(OUT_DIR, 'all_gene_z.npz'),
             z=z_all, gene_ids=gene_ids,
             smesg_ids=smesg_ids, symbols=symbols)

    # gene_manifest.csv: 全量基因身份 + phase2 覆盖状态 + 重点基因标记
    z_smed_set = set(gene_ids.tolist())
    watch = G.resolve_watchlist(G.DEFAULT_WATCHLIST, maps, z_smed_set)
    watch_smeds = {s for w in watch.values() for s in w['smeds']}
    manifest = pd.DataFrame({
        'smed_id': gene_ids,
        'smesg_id': smesg_ids,
        'gene_symbol': symbols,
        'has_phase2_z': True,           # all_gene_z 覆盖的每一行都有 z
        'in_trainable_common': np.isin(gene_ids, np.array(common_smed)),
        'is_watchlist': np.isin(gene_ids, np.array(sorted(watch_smeds))),
    })
    manifest.to_csv(os.path.join(OUT_DIR, 'gene_manifest.csv'), index=False)

    # 重点基因身份报告 (解释 HES4/DMRTA1/CREB5 为何不会出现)
    watch_lines = ['重点基因 phase2 身份状态', '=' * 48]
    for g, w in watch.items():
        watch_lines.append(f'  {g:<8} status={w["status"]:<12} '
                           f'smeds={w["smeds"] or "-"}  in_z={w["in_z"] or "-"}')
    with open(os.path.join(OUT_DIR, 'watchlist_identity.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(watch_lines) + '\n')
    print('\n'.join(watch_lines))

    torch.save({'model_state': best_st,
                'in_dim': rna_all_raw.shape[1], 'out_dim': 64,
                'rna_mean': rna_mean.values, 'rna_std': rna_std.values,
                'z_mean': z_mean, 'z_std': z_std,
                'rna_columns': list(rna_mean.index)},
               os.path.join(OUT_DIR, 'rna2z_model.pt'))
    # 也保存交集基因的z (用于Phase3训练, 全部交集基因: train+val)
    with torch.no_grad():
        zp_common = model(torch.FloatTensor(rna_norm.values).to(device)).cpu().numpy()
    zp_common = zp_common * z_std + z_mean
    np.savez(os.path.join(OUT_DIR, 'common_gene_z.npz'),
             z=zp_common, smesg_ids=np.array(common_smesg), smed_ids=np.array(common_smed))
    print(f'Done → {OUT_DIR}/')

    # ── 8. 图表 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist['train'], label='Train'); axes[0].plot(hist['val'], label='Val')
    axes[0].axvline(best_ep, color='red', linestyle='--', label=f'best ep{best_ep}')
    axes[0].set_title('Loss (val early stop)'); axes[0].legend()
    axes[1].scatter((z_norm[va_idx]*z_std+z_mean)[:, 0], (zp_va*z_std+z_mean)[:, 0], s=5, alpha=0.3)
    axes[1].set_xlabel('True z[0]'); axes[1].set_ylabel('Pred z[0]')
    axes[1].set_title(f'Val z[0] (R²={va_r2:.3f})')
    axes[2].hist(va_cos_list, bins=30, color='steelblue')
    axes[2].axvline(va_cos, color='red', linestyle='--', label=f'mean={va_cos:.3f}')
    axes[2].set_xlabel('Cosine Similarity (val)'); axes[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(PNG_DIR, 'phase2_results.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
