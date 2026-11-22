"""
phase3_edge_predict.py — 配对 z → 7 类调控关系分类 → new_edge.txt
================================================================
输入:
  - Phase 2 产出的全部基因 z 向量
  - example_data/edges.csv (训练标签: 已知正/负调控及权重)
  - 随机负采样 (无调控配对)
  - SMED→SMESG 匹配表

类别定义 (按 edges 权重范围分位划分, 各类数量平衡):
  -3: 强负调控, -2: 中负调控, -1: 弱负调控
   0: 无调控 (随机采样)
   1: 弱正调控,  2: 中正调控,  3: 强正调控

流程:
  1. 按权重 33%/66% 分位点将正/负调控各切为 3 档
  2. 按类别分层划分训练集/测试集 (默认 8:2)
  3. 训练残差网络 7 类分类模型 (训练集, 只报告 loss)
  4. 测试集评估: 准确率 + 各类 recall + 混淆矩阵
  5. 对全部基因 × 全部基因组合 (排除自配对) 预测类别, 流式写出
  6. 流式写出 new_edge.txt
     weight = 类别 × 置信度概率 (有符号置信度, 绝对值越大越可信);
     argmax=无调控 的配对用最强非零类别×概率作为弱权重;
     默认 --min-abs-weight 0.5 抽稀 (设为 0 输出全部配对)

用法: python phase3_edge_predict.py
"""

import os, sys, math, argparse, random, json, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import grn_common as G

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'example_data')
NEW_DATA_DIR = os.path.join(ROOT_DIR, 'data')
PHASE2_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase2_output')
OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase3_output')
PNG_DIR = os.path.join(ROOT_DIR, 'figures')


# ============================================================
# 残差网络 — 配对 z → 7 类调控关系分类
# ============================================================
class ResBlock(nn.Module):
    """残差块: y = ReLU(x + F(x))"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.act = nn.ReLU()
    def forward(self, x): return self.act(x + self.net(x))


class EdgeClassifier(nn.Module):
    """输入: concat(z_source, z_target) (128 dim)
    输出: 7 类 logits — [-3,-2,-1,0,1,2,3]
      -3: 强负调控, -2: 中负调控, -1: 弱负调控
       0: 无调控
       1: 弱正调控,  2: 中正调控,  3: 强正调控
    """
    def __init__(self, z_dim=64, hidden=512, n_blocks=5, dropout=0.05, n_classes=7):
        super().__init__()
        in_dim = z_dim * 2  # 128
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, n_classes)
        self.class_labels = [-3, -2, -1, 0, 1, 2, 3]
    def forward(self, z_pairs):
        h = self.proj(z_pairs)
        for b in self.blocks: h = b(h)
        return self.out(h)  # (B, 7) logits


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup', type=int, default=0, help='warmup epochs (0=自动 epochs//20)')
    parser.add_argument('--no-edge-ratio', type=float, default=1.0,
                        help='无调控负样本与已知边数的比例 (默认1:1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例 (按类别分层划分)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--min-abs-weight', type=float, default=0.5,
                        help='输出边的最小 |weight| 抽稀阈值 (0=全配对输出, 默认0.5)')
    parser.add_argument('--max-pairs', type=int, default=0,
                        help='预测最大配对数 (0=全部基因×基因; 非0结果会标记 PARTIAL, 不可用于最终绘图)')
    parser.add_argument('--tf-source-only', action='store_true',
                        help='source 只取映射表中的已知 TF + 重点基因 (约数千个而非5万, '
                             '全基因×全基因约25亿对, 该选项可在保证重点基因覆盖下完整跑完)')
    args = parser.parse_args()

    print('='*60)
    print('Phase 3: 配对 z → 调控关系分类 (正调控/负调控/无调控)')
    print('='*60)

    device = args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 1. 加载 Phase2 全部基因 z (多身份: SMED + SMESG + symbol) ──
    p2 = np.load(os.path.join(PHASE2_DIR, 'all_gene_z.npz'), allow_pickle=True)
    z_all = p2['z']           # (N_all, 64)
    genes_all = list(map(str, p2['gene_ids']))     # SMED
    z_smed_set = set(genes_all)
    gene2idx = {g: i for i, g in enumerate(genes_all)}
    print(f'全部基因 z: {z_all.shape}')
    if 'symbols' in p2.files:
        gene2sym = dict(zip(genes_all, map(str, p2['symbols'])))
        gene2smesg = dict(zip(genes_all, map(str, p2.get('smesg_ids', np.array([])))))
    else:
        maps = G.load_id_maps()
        gene2sym = {g: G.map_symbol(g, maps) for g in genes_all}
        gene2smesg = {g: maps['smed2smesg'].get(g, '') for g in genes_all}
    watch_status = G.resolve_watchlist(G.DEFAULT_WATCHLIST, G.load_id_maps(), z_smed_set)
    watch_smeds = {s for w in watch_status.values() for s in w['in_z']}
    print(f'重点基因 phase2 覆盖: '
          + ', '.join(f'{g}={w["status"]}' for g, w in watch_status.items()))

    # ── 2. 加载 edges → 三类标签 ──
    mapping = pd.read_csv(os.path.join(NEW_DATA_DIR, 'smed_smesg_mapping.csv'))
    smed2smesg = dict(zip(mapping['smed_id'], mapping['smesg_id']))
    smesg2smed = {}
    for smed, smesg in smed2smesg.items():
        if smed in gene2idx and smesg not in smesg2smed:
            smesg2smed[smesg] = smed

    edges = pd.read_csv(os.path.join(DATA_DIR, 'edges.csv'), index_col=0).dropna(subset=['source', 'target'])
    print(f'原始 edges: {len(edges)}')

    # 收集能映射的配对及权重; 无法映射到 z 空间的端点单独记录, 不静默丢弃
    pos_records, neg_records = [], []  # (smed_s, smed_t, weight)
    unmapped_rows = []                # 记录丢弃原因
    for _, row in edges.iterrows():
        s_smed = smesg2smed.get(row['source'])
        t_smed = smesg2smed.get(row['target'])
        if s_smed and t_smed:
            if row['weight'] > 0: pos_records.append((s_smed, t_smed, row['weight']))
            else:                 neg_records.append((s_smed, t_smed, row['weight']))
        else:
            why = []
            if not s_smed: why.append(f'source={row["source"]} 无法映射到phase2 z')
            if not t_smed: why.append(f'target={row["target"]} 无法映射到phase2 z')
            unmapped_rows.append({'source': row['source'], 'target': row['target'],
                                  'weight': row['weight'], 'reason': '; '.join(why)})
    if unmapped_rows:
        udf = pd.DataFrame(unmapped_rows)
        udf.to_csv(os.path.join(OUT_DIR, 'unmapped_training_edges.csv'), index=False)
        print(f'无法映射被排除的训练边: {len(udf)} 条 -> unmapped_training_edges.csv')

    n_pos, n_neg = len(pos_records), len(neg_records)
    print(f'正调控配对: {n_pos}, 负调控配对: {n_neg}')

    # ── 3. 构建训练数据 (7 类, 按权重分位平衡划分) ──
    # 正调控: 按权重 33%/66% 分位点切为 3 档 → 类别 1/2/3
    pos_ws = np.array(sorted(r[2] for r in pos_records))
    q1_pos = np.percentile(pos_ws, 33.33)
    q2_pos = np.percentile(pos_ws, 66.67)
    print(f'正调控权重分位: q33={q1_pos:.3f}, q66={q2_pos:.3f}')

    # 负调控: 按权重分位切为 3 档 → 类别 -1/-2/-3
    # (越负 = 抑制越强 = -3; 越接近 0 = 抑制越弱 = -1)
    neg_ws = np.array(sorted(r[2] for r in neg_records))  # 升序: 最负在前
    q1_neg = np.percentile(neg_ws, 33.33)
    q2_neg = np.percentile(neg_ws, 66.67)
    print(f'负调控权重分位: q33={q1_neg:.3f}, q66={q2_neg:.3f}')

    def pos_cls(w):
        """正权重 → 类别 1/2/3"""
        if w <= q1_pos: return 1
        if w <= q2_pos: return 2
        return 3

    def neg_cls(w):
        """负权重 → 类别 -1/-2/-3 (越负档位越低)"""
        if w <= q1_neg: return -3   # 最负 = 强抑制
        if w <= q2_neg: return -2
        return -1                   # 接近 0 = 弱抑制

    all_pairs, labels = [], []
    for s, t, w in pos_records:
        all_pairs.append((s, t)); labels.append(pos_cls(w))
    for s, t, w in neg_records:
        all_pairs.append((s, t)); labels.append(neg_cls(w))

    # 无调控负采样 (类别 0): 数量与其他类平衡 (取平均每类数量)
    known_pairs = set((s, t) for s, t, _ in pos_records) | set((s, t) for s, t, _ in neg_records)
    avg_cls_size = (n_pos + n_neg) / 6  # 正3档 + 负3档的平均每档数量
    n_no_edge = int(avg_cls_size * args.no_edge_ratio)
    if args.no_edge_ratio == 1.0:
        n_no_edge = int(avg_cls_size)  # 与其余各类等量

    gene_pool = [g for g in genes_all if g in smesg2smed.values() or g in smed2smesg]
    if len(gene_pool) == 0:
        gene_pool = genes_all

    no_edge_pairs = []
    attempts = 0
    while len(no_edge_pairs) < n_no_edge and attempts < n_no_edge * 200:
        s = random.choice(gene_pool)
        t = random.choice(gene_pool)
        attempts += 1
        if s != t and (s, t) not in known_pairs and (s, t) not in no_edge_pairs:
            no_edge_pairs.append((s, t))

    all_pairs += no_edge_pairs
    labels += [0] * len(no_edge_pairs)
    print(f'无调控采样: {len(no_edge_pairs)} (目标与其他类平衡)')

    z_dim = z_all.shape[1]
    X_data = np.zeros((len(all_pairs), z_dim*2), dtype=np.float32)
    # 类别标签 -3..3 → 索引 0..6 (CE loss 需要 0 起始)
    y_labels = np.array(labels, dtype=np.int64)      # 原始类别 -3..3
    y_data = y_labels + 3                             # 索引 0..6
    for i, (s, t) in enumerate(all_pairs):
        X_data[i] = np.concatenate([z_all[gene2idx[s]], z_all[gene2idx[t]]])

    n = len(X_data)
    print(f'样本总数: {n}')
    for lbl in [-3, -2, -1, 0, 1, 2, 3]:
        print(f'  类别 {lbl:>2}: {(y_labels == lbl).sum():,} 对')

    # ── 4. 划分训练集 / 测试集 (按类别分层) ──
    from sklearn.model_selection import train_test_split
    tr_idx, te_idx = train_test_split(np.arange(n), test_size=args.test_ratio,
                                      random_state=args.seed, stratify=y_data)
    print(f'\n训练集: {len(tr_idx)}, 测试集: {len(te_idx)} (按类别分层)')

    # ── 5. 训练分类模型 (训练集训练, 只报告 loss) ──
    # 类别权重按训练集分布计算
    class_counts = np.bincount(y_data[tr_idx], minlength=7)
    class_weights = torch.FloatTensor(len(tr_idx) / (7.0 * class_counts)).to(device)
    print(f'类别权重: {[f"{w:.2f}" for w in class_weights.tolist()]}')

    X_tr_t = torch.FloatTensor(X_data[tr_idx]); y_tr_t = torch.LongTensor(y_data[tr_idx])
    X_te_t = torch.FloatTensor(X_data[te_idx])
    tr_ld = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=args.batch, shuffle=True)

    model = EdgeClassifier(z_dim).to(device)
    print(f'参数: {sum(p.numel() for p in model.parameters()):,}')

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    warmup_ep = args.warmup if args.warmup > 0 else max(1, args.epochs // 20)
    sched = LambdaLR(opt, lambda ep: min(1.0, (ep+1)/warmup_ep) if ep < warmup_ep
                     else 0.5 * (1 + math.cos(math.pi * (ep - warmup_ep) / max(1, args.epochs - warmup_ep))))
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_loss, best_st, best_ep = float('inf'), None, 0
    best_test_acc = 0.0
    hist = {'loss': [], 'train_acc': [], 'test_acc': []}

    @torch.no_grad()
    def compute_acc():
        """训练集 + 测试集正确率 (仅打印间隔调用)"""
        model.eval()
        logits_tr = model(X_tr_t.to(device)); acc_tr = (logits_tr.argmax(1) == y_tr_t.to(device)).float().mean().item()
        logits_te = model(X_te_t.to(device)); acc_te = (logits_te.argmax(1) == y_te_t.to(device)).float().mean().item()
        return acc_tr, acc_te

    y_te_t = torch.LongTensor(y_data[te_idx])

    for ep in range(args.epochs):
        model.train(); tl = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = crit(model(xb), yb)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += l.item()*xb.size(0)
        tl /= len(tr_idx)
        sched.step()
        hist['loss'].append(tl)

        if tl < best_loss:
            best_loss, best_st, best_ep = tl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, ep

        if ep % 50 == 0 or ep == args.epochs-1:
            acc_tr, acc_te = compute_acc()
            hist['train_acc'].append(acc_tr); hist['test_acc'].append(acc_te)
            best_test_acc = max(best_test_acc, acc_te)
            print(f'  Ep {ep:4d} | Loss {tl:.4f} | best {best_loss:.4f} (ep{best_ep}) | '
                  f'Train Acc {acc_tr:.3f} | Test Acc {acc_te:.3f} | LR {opt.param_groups[0]["lr"]:.2e}')

    model.load_state_dict(best_st)
    print(f'最终训练 Loss: {best_loss:.4f} (ep {best_ep}), 历史最高测试正确率: {best_test_acc:.4f}')

    # ── 6. 测试集评估 ──
    model.eval()
    with torch.no_grad():
        logits_te = model(X_te_t.to(device))
        probs_te = torch.softmax(logits_te, dim=1).cpu().numpy()
        pred_idx_te = probs_te.argmax(axis=1)
        # 训练集也评估, 用于对照
        logits_tr = model(X_tr_t.to(device))
        probs_tr = torch.softmax(logits_tr, dim=1).cpu().numpy()
        pred_idx_tr = probs_tr.argmax(axis=1)

    y_te = y_data[te_idx]; y_tr = y_data[tr_idx]
    te_acc = np.mean(pred_idx_te == y_te)
    tr_acc = np.mean(pred_idx_tr == y_tr)
    print(f'\n训练集准确率: {tr_acc:.4f} ({len(tr_idx)} 样本)')
    print(f'测试集准确率: {te_acc:.4f} ({len(te_idx)} 样本)')

    class_names = {-3: '强负调控', -2: '中负调控', -1: '弱负调控',
                   0: '无调控', 1: '弱正调控', 2: '中正调控', 3: '强正调控'}
    print(f'测试集各类 recall:')
    for lbl in [-3, -2, -1, 0, 1, 2, 3]:
        mask = y_te == lbl + 3
        if mask.sum() > 0:
            print(f'  {class_names[lbl]}({lbl}): recall={np.mean(pred_idx_te[mask] == lbl+3):.4f} ({mask.sum()} 样本)')

    # ── 7. 全量预测: 全部基因 × 全部基因 (排除自配对) → 流式写出 new_edge.txt ──
    print(f'\n--- 全量预测 (基因×基因, 排除自配对) ---')

    target_smed = sorted(set(genes_all))
    if args.tf_source_only:
        # source 限定为已知 TF (smesg_tf 列) + 重点基因, 否则 5万×5万≈25亿对无法跑完;
        # 重点基因即使在 TF 表之外也强制作为 source (用户要求: 只要存在就必须可展示)
        maps0 = G.load_id_maps()
        import pandas as _pd
        ad0 = _pd.read_csv(os.path.join(G.MAP_DIR, 'adata_var_data.csv'))
        tf_smesg = set(map(str, ad0['smesg_tf'].dropna()))
        tf_smeds = {smed for smed, sg in maps0['smed2smesg'].items()
                    if sg in tf_smesg and smed in z_smed_set}
        source_genes = sorted(tf_smeds | (watch_smeds & z_smed_set))
        print(f'Source 模式: --tf-source-only → 已知TF∩z + 重点基因 = {len(source_genes)}')
    else:
        source_genes = sorted(set(genes_all))
    print(f'Source: {len(source_genes)}, Target: {len(target_smed)}')
    total_pairs = len(source_genes) * len(target_smed) - len(source_genes)  # 排除自配对
    print(f'总配对数: {total_pairs:,} (排除自配对)')

    if args.max_pairs > 0:
        total_pairs = min(total_pairs, args.max_pairs)
        print(f'  限制到: {total_pairs:,}')

    out_path = os.path.join(OUT_DIR, 'new_edge.txt')
    tmp_path = out_path + '.tmp'          # 先写临时文件, 成功后原子替换, 避免残缺文件被误当完整结果
    z_tensor = torch.FloatTensor(z_all).to(device)
    model.eval()

    pred_batch = 10000
    # 非零类别的类标签及其在 probs 中的列 (列0=类别-3, 列1=-2, 列2=-1, 列4=1, 列5=2, 列6=3)
    NZ_CLASSES = np.array([-3., -2., -1., 1., 2., 3.], dtype=np.float32)
    NZ_COLS = np.array([0, 1, 2, 4, 5, 6])
    # 权重策略: argmax≠0 → class×prob[argmax];
    #           argmax=0 → 最强非零类别×其概率 (弱有符号权重, 默认按阈值抽稀)
    sample_stride = max(1, total_pairs // 200000)
    n_rows = n_pos = n_neg = n_fallback = 0
    row_no = 0
    samp_w = []

    f_out = open(tmp_path, 'w', newline='')
    f_out.write('source\ttarget\tweight\n')

    pbar_every = max(1, len(source_genes) // 20)
    done = 0
    completed_sources = 0
    source_as_src = set()                 # 实际写出过边的 source (供完整性审计)
    watch_src_seen, watch_tgt_seen = set(), set()
    partial = False
    for i, src in enumerate(source_genes):
        src_z = z_tensor[gene2idx[src]]
        src_is_watch = src in watch_smeds
        src_had_row = False
        for j in range(0, len(target_smed), pred_batch):
            # Exclude self-pairs before enforcing the remaining prediction budget.
            t_batch = [target for target in target_smed[j:j+pred_batch] if target != src]
            if args.max_pairs > 0:
                t_batch = t_batch[:max(0, args.max_pairs - done)]
            if not t_batch:
                continue
            t_idx = [gene2idx[t] for t in t_batch]
            z_pairs = torch.cat([src_z.unsqueeze(0).expand(len(t_batch), -1), z_tensor[t_idx]], dim=1)
            with torch.no_grad():
                probs = torch.softmax(model(z_pairs), dim=1).cpu().numpy()  # (B, 7)
            B = len(t_batch)
            pred_idx = probs.argmax(axis=1)
            w = np.zeros(B, dtype=np.float32)
            # Class indices map to [-3, -2, -1, 0, 1, 2, 3]; zero is index 3.
            nz = pred_idx != 3
            if nz.any():
                rows = np.where(nz)[0]
                w[rows] = (pred_idx[rows] - 3) * probs[rows, pred_idx[rows]]
            zr = np.where(~nz)[0]
            if len(zr):
                sub = probs[np.ix_(zr, NZ_COLS)]          # (K, 6) 非零类概率
                best = sub.argmax(axis=1)
                w[zr] = NZ_CLASSES[best] * sub[np.arange(len(zr)), best]
                n_fallback += len(zr)
            keep = np.array(t_batch) != src               # 排除自配对
            tgt_is_watch = np.isin(t_batch, list(watch_smeds)) if watch_smeds else np.zeros(B, bool)
            if args.min_abs_weight > 0:
                # 重点基因 (作为 source 或 target) 的边永不被阈值丢弃
                keep &= (np.abs(w) >= args.min_abs_weight) | tgt_is_watch | src_is_watch
            write_idx = np.where(keep)[0]
            if len(write_idx):
                lines = [f'{src}\t{t_batch[k]}\t{w[k]:.6f}' for k in write_idx]
                f_out.write('\n'.join(lines) + '\n')
                src_had_row = True
                if tgt_is_watch[write_idx].any():
                    watch_tgt_seen.update(t_batch[k] for k in write_idx if tgt_is_watch[k])
            n_pos += int((w[write_idx] > 0).sum())
            n_neg += int((w[write_idx] < 0).sum())
            # 等距抽样供直方图 (文件不排序, 顺序抽样即近似均匀)
            for k in write_idx:
                if row_no % sample_stride == 0:
                    samp_w.append(float(w[k]))
                row_no += 1
            n_rows += len(write_idx)
            done += B
            if args.max_pairs > 0 and done >= args.max_pairs:
                break
        if src_had_row:
            source_as_src.add(src)
            if src_is_watch:
                watch_src_seen.add(src)
        completed_sources += 1
        if i % pbar_every == 0:
            print(f'  Source {i+1}/{len(source_genes)}, 累计 {n_rows:,} 行 (正 {n_pos:,} / 负 {n_neg:,} / 无调控fallback {n_fallback:,})')
        if args.max_pairs > 0 and done >= args.max_pairs:
            partial = True
            print(f'  [PARTIAL] 达到 --max-pairs {args.max_pairs:,}, 提前停止; 此结果不完整, 不可用于最终绘图')
            break

    f_out.close()
    # 原子替换: 只有跑到这里才认为完整生成, 残缺的 .tmp 不会覆盖旧的 new_edge.txt
    os.replace(tmp_path, out_path)

    # ── 完整性审计: 每个有有效 z 的 source 都应被遍历到 ──
    expected_sources = len(source_genes)
    status = 'PARTIAL' if partial else 'COMPLETE'
    watch_src_expected = {s for s in watch_smeds if s in set(source_genes)}
    audit = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'status': status,
        'z_input_genes': len(genes_all),
        'n_source_genes': len(source_genes),
        'n_target_genes': len(target_smed),
        'expected_sources': expected_sources,
        'completed_sources': completed_sources,
        'sources_with_written_edges': len(source_as_src),
        'total_pairs_evaluated': done,
        'written_rows': n_rows,
        'min_abs_weight': args.min_abs_weight,
        'max_pairs': args.max_pairs,
        'watchlist': {g: {'status': w['status'], 'in_z_smeds': w['in_z']}
                      for g, w in watch_status.items()},
        'watch_source_edges_written': sorted(watch_src_seen),
        'watch_target_edges_written': sorted(watch_tgt_seen),
        'watch_source_expected_missing': sorted(watch_src_expected - watch_src_seen),
        'note': ('sources_with_written_edges < n_source_genes 通常是因为部分 source 的全部预测都低于 '
                 'min_abs_weight 阈值 (0边) — 这些基因的预测确实评估过, 不代表被截断; '
                 '若 status=PARTIAL 则结果不完整, 请勿用于最终绘图。'),
    }
    with open(os.path.join(OUT_DIR, 'prediction_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    if status == 'PARTIAL':
        print(f'!!! 本次为 PARTIAL 运行 (completed {completed_sources}/{expected_sources} sources), '
              f'new_edge.txt 不完整, 请重跑完整预测 !!!')
    else:
        print(f'完整性校验: 已遍历 {completed_sources}/{expected_sources} 个 source -> COMPLETE')
    print(f'重点基因边覆盖: 作为source写出={sorted(watch_src_seen) or "无"}; 作为target写出={sorted(watch_tgt_seen) or "无"}')
    print(f'coverage manifest -> {os.path.join(OUT_DIR, "prediction_manifest.json")}')
    print(f'输出行数: {n_rows:,} / {done:,} 配对')
    print(f'  正调控: {n_pos:,}, 负调控: {n_neg:,}, argmax=无调控 fallback: {n_fallback:,}')
    if samp_w:
        print(f'  weight 范围: [{min(samp_w):.4f}, {max(samp_w):.4f}] (抽样 {len(samp_w):,})')
    print(f'\nnew_edge.txt → {out_path}')

    # ── 8. 保存模型 ──
    torch.save({'model_state': best_st, 'z_dim': z_dim},
               os.path.join(OUT_DIR, 'edge_predictor.pt'))

    # ── 9. 图表 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(hist['loss'], label='Train CE Loss')
    axes[0].set_title('Training Loss'); axes[0].legend()
    # 测试集 7×7 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, pred_idx_te)
    im = axes[1].imshow(cm, cmap='Blues')
    tick_labels = ['-3', '-2', '-1', '0', '1', '2', '3']
    axes[1].set_xticks(range(7)); axes[1].set_yticks(range(7))
    axes[1].set_xticklabels(tick_labels); axes[1].set_yticklabels(tick_labels)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    axes[1].set_title(f'Test Confusion (acc={te_acc:.3f})')
    for a in range(7):
        for b in range(7):
            axes[1].text(b, a, cm[a, b], ha='center', va='center', fontsize=6)
    plt.colorbar(im, ax=axes[1])
    axes[2].hist(np.asarray(samp_w), bins=50, color='steelblue', range=(-3, 3))
    axes[2].set_xlabel('weight'); axes[2].set_title(f'New edges (n={n_rows:,}, 抽样 {len(samp_w):,})')
    plt.tight_layout(); plt.savefig(os.path.join(PNG_DIR, 'phase3_results.png'), dpi=150)
    plt.close(fig)

    print('Done!')


if __name__ == '__main__':
    main()