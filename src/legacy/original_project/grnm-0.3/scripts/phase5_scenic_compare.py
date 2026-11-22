# phase5_scenic_compare.py — SCENIC 方法 (GENIE3) 与 GRNM 方法的对比
# ================================================================
# 数据: GSE262682 原始 scRNA (涡虫再生 6 个 10x 样本, 用户提供)
# SCENIC 工作流:
#   1. GRN 推断: GENIE3 (随机森林, sklearn 实现, 100 棵树 —
#      计算量所限低于标准 1000 棵, 如实声明)
#   2. 共表达基线: log1p 归一化后的 Pearson 相关
#   3. 对比对象: phase3 new_edge.txt (GRNM 方法)
#   4. 金标准: example_data/edges.csv
# 注: 涡虫无 cisTarget motif 数据库, motif 富集步骤无法执行, 如实省略。
#
# 用法:
#   python scripts/phase5_scenic_compare.py                 # 全流程(含 GENIE3, 较慢)
#   python scripts/phase5_scenic_compare.py --skip-genie3   # 跳过 GENIE3, 只用共表达
# ================================================================

import gzip, os, re, sys, argparse, heapq
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
RAW_SCRN_DIR = r'C:\Users\25439\Desktop\GSE262682_RAW\scRNA'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EDGE_DIR = os.path.join(ROOT_DIR, 'example_data')
MAP_DIR  = os.path.join(ROOT_DIR, 'map_data')
PHASE3_DIR = os.path.join(ROOT_DIR, 'outputs', 'phase3_output')
OUT_DIR  = os.path.join(ROOT_DIR, 'outputs', 'phase5_output')
PNG_DIR  = os.path.join(ROOT_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

SAMPLES = [
    ('0hpa',  'GSM8173695_0hpa'), ('6hpa', 'GSM8173696_6hpa'),
    ('36hpa', 'GSM8173697_36hpa'), ('72hpa', 'GSM8173698_72hpa'),
    ('EGFP_RNAi', 'GSM8173699_EGFP.RNAi'), ('ZNF311_RNAi', 'GSM8173700_ZNF311.RNAi'),
]
GLI3_SMED = 'SMED30015613'
WATCH = {'PAX6': ['SMED30024586', 'SMED30017839'], 'HES5': ['SMED30005790']}

N_CELLS_SUB = 3000        # GENIE3/共表达用的细胞子集 (分层抽样)
N_VAR_GENES = 3000        # 高变基因数 (评估 universe 的一部分)
TOP_K = 50                # Jaccard 用 top-k
N_TREES = 100             # GENIE3 随机森林树数 (标准为 1000, 计算量妥协)


# ────────────────────────────────────────────────────────────────
# 1. 构建合并计数矩阵 (6 个样本, SMESG/SMED 统一为 SMED)
# ────────────────────────────────────────────────────────────────
def build_matrix():
    print('1. 构建合并计数矩阵 ...')
    mapping = pd.read_csv(os.path.join(DATA_DIR, 'smed_smesg_mapping.csv'))
    smesg2smed = dict(zip(mapping['smesg_id'], mapping['smed_id']))
    # phase2 基因空间 (与 GRNM 方法对齐)
    p2_genes = set(pd.read_csv(os.path.join(DATA_DIR, 'rna_pca_new.csv')).iloc[:, 0].astype(str))

    mats, cell_meta = [], []
    for label, gsm in SAMPLES:
        feat_path = os.path.join(RAW_SCRN_DIR, f'{gsm}_features.tsv.gz')
        bar_path  = os.path.join(RAW_SCRN_DIR, f'{gsm}_barcodes.tsv.gz')
        mtx_path  = os.path.join(RAW_SCRN_DIR, f'{gsm}_matrix.mtx.gz')
        with gzip.open(feat_path, 'rt') as f:
            feats = [ln.split('\t')[0] for ln in f.read().splitlines()]
        with gzip.open(bar_path, 'rt') as f:
            bars = [f'{label}_{b}' for b in f.read().splitlines()]
        with gzip.open(mtx_path, 'rb') as f:
            m = scipy.io.mmread(f).tocsr()   # genes × cells
        # 基因 ID 归一: SMESGxxxxx.N / SMESGxxxxx → SMED
        gids = []
        for g in feats:
            g = re.sub(r'\.[0-9]+$', '', g)
            gids.append(smesg2smed.get(g, g))
        gids = np.array(gids)
        keep = np.array([g in p2_genes for g in gids])
        m = m[keep, :]
        gids = gids[keep]
        print(f'   {label}: {m.shape[1]} 细胞 × {m.shape[0]} 基因 (原始 {len(feats)})')
        mats.append((gids, m))
        cell_meta.append(pd.DataFrame({'cell': bars, 'sample': label}))
        del m

    all_genes = sorted(set(np.concatenate([g for g, _ in mats])))
    g2i = {g: i for i, g in enumerate(all_genes)}
    cols = []
    data, row, col = [], [], []
    ncol = 0
    for (gids, m), meta in zip(mats, cell_meta):
        coo = m.tocoo()
        data.append(coo.data); row.append([g2i[g] for g in gids[coo.row]])
        col.append(coo.col + ncol)
        ncol += m.shape[1]
    X = sp.csr_matrix((np.concatenate(data),
                       (np.concatenate(row), np.concatenate(col))),
                      shape=(len(all_genes), ncol))
    meta = pd.concat(cell_meta, ignore_index=True)
    sp.save_npz(os.path.join(OUT_DIR, 'combined_counts.npz'), X)
    with open(os.path.join(OUT_DIR, 'combined_genes.txt'), 'w') as f:
        f.write('\n'.join(all_genes))
    meta.to_csv(os.path.join(OUT_DIR, 'combined_cells.csv'), index=False)
    print(f'   合并矩阵: {X.shape[0]} 基因 × {X.shape[1]} 细胞 -> '
          f'outputs/phase5_output/combined_counts.npz')
    return X, all_genes, meta


def load_or_build():
    npz = os.path.join(OUT_DIR, 'combined_counts.npz')
    if os.path.exists(npz):
        X = sp.load_npz(npz)
        with open(os.path.join(OUT_DIR, 'combined_genes.txt')) as f:
            genes = f.read().splitlines()
        meta = pd.read_csv(os.path.join(OUT_DIR, 'combined_cells.csv'))
        print(f'1. 复用已有合并矩阵: {X.shape[0]} × {X.shape[1]}')
        return X, genes, meta
    return build_matrix()


# ────────────────────────────────────────────────────────────────
# 2. log 归一化 + 细胞分层子集
# ────────────────────────────────────────────────────────────────
def normalize_and_subset(X, meta):
    # log1p(CPM)
    sizes = np.asarray(X.sum(axis=0)).ravel()
    X = X.multiply(1e4 / sizes).tocsr()
    X.data = np.log1p(X.data)
    # 每样本取等量细胞
    cells = []
    for s in SAMPLES:
        sub = meta.index[meta['sample'] == s[0]].tolist()
        k = max(1, N_CELLS_SUB // len(SAMPLES))
        cells += list(np.random.default_rng(42).choice(sub, size=min(k, len(sub)), replace=False))
    return X[:, cells], meta.iloc[cells]


# ────────────────────────────────────────────────────────────────
# 3. GENIE3 式随机森林 GRN 推断 (TF → 靶基因重要性)
# ────────────────────────────────────────────────────────────────
def run_genie3(X, genes, tf_smed, targets):
    print('3. GENIE3 随机森林推断 ...')
    g2i = {g: i for i, g in enumerate(genes)}
    tf_rows = np.array([g2i[t] for t in tf_smed])
    tgt_rows = np.array([g2i[t] for t in targets])
    # TF 表达 (dense, 细胞 × TF)
    Xtf = X[tf_rows, :].toarray().T
    Xt = X[tgt_rows, :].toarray()

    def fit_one(k):
        t = targets[k]
        rf = RandomForestRegressor(n_estimators=N_TREES, max_features='sqrt',
                                   n_jobs=1, random_state=42)
        rf.fit(Xtf, Xt[k])
        return t, rf.feature_importances_

    res = Parallel(n_jobs=8, verbose=5)(delayed(fit_one)(k) for k in range(len(targets)))
    rows = [(tf, t, imp) for t, imps in res for tf, imp in zip(tf_smed, imps)]
    df = pd.DataFrame(rows, columns=['tf', 'target', 'importance'])
    df.to_csv(os.path.join(OUT_DIR, 'genie3_scores.csv'), index=False)
    print(f'   GENIE3 分数: {len(df):,} 条 -> outputs/phase5_output/genie3_scores.csv')
    return df


# ────────────────────────────────────────────────────────────────
# 4. 共表达基线 (Pearson, log 归一化矩阵)
# ────────────────────────────────────────────────────────────────
def run_coexp(X, genes, tf_smed, targets):
    print('4. 共表达基线 (Pearson) ...')
    g2i = {g: i for i, g in enumerate(genes)}
    tf_rows = np.array([g2i[t] for t in tf_smed])
    tgt_rows = np.array([g2i[t] for t in targets])
    sub = X[np.concatenate([tf_rows, tgt_rows]), :].toarray()
    n = sub.shape[1]
    mu = sub.mean(axis=1, keepdims=True)
    sd = sub.std(axis=1, keepdims=True) + 1e-12
    sub = (sub - mu) / sd
    corr = sub[:len(tf_smed), :] @ sub[len(tf_smed):, :].T / (n - 1)
    rows = [(tf, t, float(corr[i, j]))
            for i, tf in enumerate(tf_smed) for j, t in enumerate(targets)]
    df = pd.DataFrame(rows, columns=['tf', 'target', 'score'])
    df.to_csv(os.path.join(OUT_DIR, 'coexp_scores.csv'), index=False)
    print(f'   共表达分数: {len(df):,} 条 -> outputs/phase5_output/coexp_scores.csv')
    return df


# ────────────────────────────────────────────────────────────────
# 5. 金标准 + phase3 分数加载
# ────────────────────────────────────────────────────────────────
def load_gold(tf_smed, genes):
    """金标准正/负配对 (负采样用 set 查找, 避免 apply 慢)。

    注: 重点基因边 (GLI3→PAX6/HES5) 不注入金标准 — 那是本项目关注的"发现",
    把它加进金标准会循环论证 (拿预测验证预测)。金标准只用 edges.csv。
    """
    mapping = pd.read_csv(os.path.join(DATA_DIR, 'smed_smesg_mapping.csv'))
    smesg2smed = dict(zip(mapping['smesg_id'], mapping['smed_id']))
    ed = pd.read_csv(os.path.join(EDGE_DIR, 'edges.csv'), index_col=0)
    s = ed['source'].map(smesg2smed)
    t = ed['target'].map(smesg2smed)
    pp = pd.DataFrame({'tf': s, 'target': t}).dropna()
    gene_set = set(genes)
    pp = pp[pp['tf'].isin(tf_smed) & pp['target'].isin(gene_set)].drop_duplicates()
    rng = np.random.default_rng(42)
    pos_keys = set(zip(pp['tf'], pp['target']))
    neg = pd.DataFrame({'tf': rng.choice(tf_smed, len(pp)),
                        'target': rng.choice(list(gene_set), len(pp))})
    mask = [k not in pos_keys for k in zip(neg['tf'], neg['target'])]
    neg = neg[mask].drop_duplicates().head(len(pp)).reset_index(drop=True)
    print(f'5. 金标准: 正 {len(pp)} / 负 {len(neg)}')
    return pp, neg


def load_phase3_scores(tf_set, universe_set, top_m_per_tf=1000):
    """流式读取 new_edge.txt, 每个 (tf, target) 保留 max|weight|;
    每 TF 只保留 top_m_per_tf 条最高分边 (默认1000, 远大于评估所需 top50/top100)。
    score_map 上限 ≈ |tf_set|×1000 (几百万条), 避免 1.7 亿配对 OOM。
    评估 universe 内的 (tf,target) 始终保留, 不受 top-M 截断影响。"""
    print(f'   读取 new_edge.txt (GRNM 方法, 每 TF 保留 top {top_m_per_tf} 条) ...')
    mapping = pd.read_csv(os.path.join(DATA_DIR, 'smed_smesg_mapping.csv'))
    smesg2smed = dict(zip(mapping['smesg_id'], mapping['smed_id']))
    per_tf = {}    # tf -> {target: max|w|}
    n_total = 0
    for chunk in pd.read_csv(os.path.join(PHASE3_DIR, 'new_edge.txt'), sep='\t',
                             chunksize=2_000_000,
                             dtype={'source': str, 'target': str, 'weight': float}):
        src = chunk['source'].values
        tgt = chunk['target'].str.replace(r'\.[0-9]+$', '', regex=True).values
        w = chunk['weight'].abs().values
        keep = np.array([s in tf_set for s in src])
        if not keep.any():
            n_total += len(chunk)
            continue
        src_k = src[keep]; tgt_k = tgt[keep]; w_k = w[keep]
        # 批内 SMESG→SMED 归一 (仅 keep 子集, 走 pandas map 更快)
        tgt_s = pd.Series(tgt_k).map(smesg2smed).fillna(pd.Series(tgt_k)).values
        for tf_i, tgt_i, wi in zip(src_k, tgt_s, w_k):
            d = per_tf.setdefault(tf_i, {})
            if d.get(tgt_i, -1) < wi:
                d[tgt_i] = float(wi)
        n_total += len(chunk)
        if n_total % 10_000_000 == 0:
            print(f'     已处理 {n_total:,} 行, TF数 {len(per_tf):,}')
    # 最终: 每 TF 裁剪到 top-M (universe 内配对强制保留)
    score_map = {}
    for tf_i, d in per_tf.items():
        if len(d) > top_m_per_tf:
            uni = {t: w for t, w in d.items() if t in universe_set}
            rest = {t: w for t, w in d.items() if t not in universe_set}
            keep = dict(heapq.nlargest(top_m_per_tf - len(uni), rest.items(),
                                       key=lambda kv: kv[1])) if len(rest) else {}
            d = {**uni, **keep}
        for tgt_i, wi in d.items():
            score_map[(tf_i, tgt_i)] = wi
    print(f'   GRNM 唯一配对: {len(score_map):,} (从 {n_total:,} 行, 每 TF ≤{top_m_per_tf})')
    return score_map


# ────────────────────────────────────────────────────────────────
# 6. 评估与出图
# ────────────────────────────────────────────────────────────────
def evaluate(scores, name, pp, neg):
    def s_of(pairs):
        return np.array([scores.get((a, b), 0.0) for a, b in zip(pairs['tf'], pairs['target'])])
    sp_, sn = s_of(pp), s_of(neg)
    lab = np.r_[np.ones(len(sp_)), np.zeros(len(sn))]
    s = np.r_[sp_, sn]
    auroc = roc_auc_score(lab, s)
    auprc = average_precision_score(lab, s)
    print(f'   {name}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}')
    return auroc, auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-genie3', action='store_true', help='跳过 GENIE3 (只用共表达)')
    args = parser.parse_args()

    X, genes, meta = load_or_build()
    X, meta = normalize_and_subset(X, meta)
    print(f'2. 细胞子集: {X.shape[1]} 个 (log1p CPM 归一化)')

    # TF 列表 (与 GRNM 相同来源)
    tf_tab = pd.read_csv(os.path.join(EDGE_DIR, 'tf_gene_motif.csv'))
    mapping = pd.read_csv(os.path.join(DATA_DIR, 'smed_smesg_mapping.csv'))
    smesg2smed = dict(zip(mapping['smesg_id'], mapping['smed_id']))
    tf_smed = [smesg2smed.get(s) for s in tf_tab['smesg']]
    tf_smed = sorted({t for t in tf_smed if t in set(genes)})
    print(f'   TF: {len(tf_smed)} 个')

    # 金标准配对
    pp, neg = load_gold(tf_smed, genes)
    # 评估 universe: 金标准基因 ∪ 高变基因 ∪ 重点基因
    var = np.asarray(X.power(2).mean(axis=1)).ravel() - np.asarray(X.mean(axis=1)).ravel() ** 2
    top_var = set(np.array(genes)[np.argsort(-var)[:N_VAR_GENES]])
    universe = sorted(set(pp['target']) | set(neg['target']) | top_var |
                      {g for ids in WATCH.values() for g in ids})
    universe = [g for g in universe if g in set(genes)]
    print(f'   评估 universe: {len(universe)} 个基因')

    # GENIE3 (SCENIC GRN 推断)
    if args.skip_genie3:
        print('3. 跳过 GENIE3 (--skip-genie3)')
        genie3 = None
    else:
        genie3 = run_genie3(X, genes, tf_smed, universe)

    # 共表达基线
    coexp = run_coexp(X, genes, tf_smed, universe)

    # phase3 分数 (每 TF 仅保留 top-1000 |weight|, 约 500 万条, 避免 1.7 亿配对 OOM)
    p3 = load_phase3_scores(set(tf_smed), set(universe), top_m_per_tf=1000)

    # 评估
    print('6. 金标准评估 ...')
    met = {}
    if genie3 is not None:
        met['GENIE3'] = evaluate(dict(zip(zip(genie3['tf'], genie3['target']),
                                          genie3['importance'])), 'GENIE3 (SCENIC)', pp, neg)
    met['共表达'] = evaluate(dict(zip(zip(coexp['tf'], coexp['target']), coexp['score'])),
                             '共表达 (Pearson)', pp, neg)
    met['GRNM'] = evaluate(p3, 'GRNM (phase3)', pp, neg)

    # top-50 Jaccard (universe 上)
    print('   计算 top-50 Jaccard ...')
    jacs = {}
    score_sets = [('共表达', coexp, 'score')]
    if genie3 is not None:
        score_sets.insert(0, ('GENIE3', genie3, 'importance'))
    for nm, df, k in score_sets:
        jl = []
        for tf in tf_smed:
            top = df[df['tf'] == tf].nlargest(TOP_K, k)['target'].tolist()
            tgt_p3 = pd.Series({t: p3.get((tf, t), 0.0) for t in universe})
            top_p3 = tgt_p3.nlargest(TOP_K).index.tolist()
            jl.append(len(set(top) & set(top_p3)) / max(1, len(set(top) | set(top_p3))))
        jacs[nm] = float(np.mean(jl))
        print(f'   {nm} vs GRNM top-{TOP_K} Jaccard: {jacs[nm]:.4f}')

    # GLI3 regulon
    print('   GLI3 regulon 对比 ...')
    gt_gli3 = set(pp['target'][pp['tf'] == GLI3_SMED])
    report_lines = []
    if GLI3_SMED in tf_smed:
        for nm, df, k in score_sets:
            g3 = df[df['tf'] == GLI3_SMED].nlargest(100, k)
            ov = len(set(g3['target']) & gt_gli3)
            print(f'   {nm} GLI3 top100 ∩ 金标准靶: {ov} (金标准 {len(gt_gli3)})')
            r_all = df[df['tf'] == GLI3_SMED].sort_values(k, ascending=False)
            ranks = {}
            for gname, ids in WATCH.items():
                pos = [np.where(r_all['target'].values == i)[0] for i in ids
                       if i in set(r_all['target'])]
                ranks[gname] = int(min(p[0] for p in pos) + 1) if pos else np.nan
            print(f'   {nm} 重点基因排名: ' +
                  ', '.join(f'{g}:{int(r) if not np.isnan(r) else "未出现"}'
                            for g, r in ranks.items()))
            report_lines.append(f'{nm}: ' +
                                ', '.join(f'{g}={int(r) if not np.isnan(r) else "NA"}'
                                          for g, r in ranks.items()))
        g3p = pd.Series({(a, b): w for (a, b), w in p3.items() if a == GLI3_SMED})
        ovp = len(set(g3p.nlargest(100).index) & gt_gli3)
        print(f'   GRNM GLI3 top100 ∩ 金标准靶: {ovp}')
        g3p_sorted = g3p.sort_values(ascending=False)
        p3ranks = {}
        for g, ids in WATCH.items():
            pos = [np.where(g3p_sorted.index == i)[0] for i in ids if i in g3p.index]
            p3ranks[g] = int(min(p[0] for p in pos) + 1) if pos else np.nan
        print('   GRNM 重点基因排名: ' +
              ', '.join(f'{g}:{int(r) if not np.isnan(r) else "未出现"}'
                        for g, r in p3ranks.items()))
        report_lines.append('GRNM: ' +
                            ', '.join(f'{g}={int(r) if not np.isnan(r) else "NA"}'
                                      for g, r in p3ranks.items()))
    else:
        print('   GLI3 不在 TF 列表, 跳过')

    # 汇总
    lines = ['=' * 64,
             'Phase5 SCENIC (GENIE3) 方法对比报告 (GSE262682 原始 scRNA)',
             f'  scRNA: 6 样本合并 {X.shape[1]} 细胞子集, log1p CPM; 评估 universe {len(universe)} 基因',
             f'  GENIE3: sklearn 随机森林 {N_TREES} 棵树 (标准 SCENIC 为 1000, 计算量妥协)',
             '  注: 涡虫无 cisTarget motif 数据库, motif 富集步骤省略',
             '',
             '1. 金标准 edges.csv 评估:']
    for nm, (a, p) in met.items():
        lines.append(f'   {nm}: AUROC={a:.3f}  AUPRC={p:.3f}')
    lines.append(f'2. top-{TOP_K} Jaccard: ' + '; '.join(f'{nm}={v:.4f}' for nm, v in jacs.items()))
    lines.append('3. GLI3 regulon 重点基因排名 (越靠前越可信):')
    lines += [f'   {l}' for l in report_lines]
    with open(os.path.join(OUT_DIR, 'phase5_scenic_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('   汇总 -> outputs/phase5_output/phase5_scenic_summary.txt')
    print('Done!')


if __name__ == '__main__':
    main()
