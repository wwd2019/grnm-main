# 鼠标 GRN 基因调控网络分析脚本
# 读取 example_data 下的数据，进行 PCA → Graph → GAE → UMAP → Louvain 聚类
# 并生成调控网络可视化图
import os
import random
import pickle

# ⚠️ 必须在所有 matplotlib 导入之前设置后端
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，直接保存图片不弹窗

# ⚠️ torch/dgl 必须在 pandas 之前导入，否则 DLL 冲突导致 segfault
import grnm2  # grnm2 内部会导入 torch/dgl，必须最先导入

import pandas as pd

# 基于脚本所在目录解析数据路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'example_data')
PNG_DIR = os.path.join(ROOT_DIR, 'figures')
os.makedirs(PNG_DIR, exist_ok=True)


def optimize_louvain_resolution(grn_obj, domain_module_gene_path,
                                q_min=0.2, q_max=1.2, q_step=0.1):
    """扫描 Louvain resolution，以之前成果的全基因模块划分为 ground truth

    使用 domain_module_gene.txt 中 1805 个基因 → 8 个模块的映射
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import numpy as np

    # --- 加载先前成果：SMESG → previous_module ---
    prev_gene_mod = pd.read_csv(domain_module_gene_path, sep='\t')
    # 列名 M1~M8 → 模块编号 1~8
    smesg_to_prev_mod = {}
    for col in prev_gene_mod.columns:
        mod_num = int(col[1:])  # M1 → 1
        for smesg in prev_gene_mod[col].dropna():
            smesg_id = smesg.strip()
            if smesg_id and smesg_id not in smesg_to_prev_mod:
                smesg_to_prev_mod[smesg_id] = mod_num

    prev_genes = set(smesg_to_prev_mod.keys())
    all_prev_modules = sorted(set(smesg_to_prev_mod.values()))
    print(f"先前成果: {len(prev_genes)} 个基因 → {len(all_prev_modules)} 个模块")

    # 加载 domain 映射（用于后续标注）
    tf_link_path = domain_module_gene_path.rsplit('module_split', 1)[0] + 'tf_link_modules.csv'
    tf_link_df = pd.read_csv(tf_link_path)
    prev_mod_to_domain = {}
    for mod in tf_link_df['tf_module'].unique():
        top_domain = tf_link_df[tf_link_df['tf_module'] == mod]['domain'].value_counts().index[0]
        prev_mod_to_domain[mod + 1] = top_domain  # tf_link 用 0-7, domain_module_gene 用 1-8
    print(f"  先前 Module → Domain: {prev_mod_to_domain}")

    # 当前数据中能匹配到的基因
    current_gene_names = set(grn_obj.gene_name)
    common_genes = prev_genes & current_gene_names
    print(f"  与当前数据交集: {len(common_genes)} 个基因")

    q_values = list(np.arange(q_min, q_max + q_step/2, q_step))
    q_values = [round(q, 2) for q in q_values]

    print(f"\n{'='*75}")
    print(f"扫描 Louvain resolution (q={q_min}~{q_max}, step={q_step})")
    print(f"评分: ARI vs 先前模块划分 (交集基因={len(common_genes)})")
    print(f"{'='*75}")
    print(f"{'q':<8} {'模块':<6} {'ARI':<8} {'NMI':<8} {'可对比基因':<10} {'评分':<10}")
    print(f"{'─'*55}")

    results = []
    for q in q_values:
        grn_obj.louvain(q=q)

        # 当前 partition: gene_index → module
        # 找出 common_genes 在当前模块中的分配
        labels_true = []  # 先前模块编号
        labels_pred = []  # 当前模块编号
        for smesg_id in common_genes:
            gene_idx = grn_obj.gene_idx.get(smesg_id)
            if gene_idx is not None and gene_idx in grn_obj.partition:
                labels_true.append(smesg_to_prev_mod[smesg_id])
                labels_pred.append(grn_obj.partition[gene_idx])

        n_matched = len(labels_true)
        n_mod = len(set(grn_obj.partition.values()))

        if n_matched < 10 or n_mod < 2:
            results.append((q, n_mod, 0, 0, n_matched, 0))
            print(f"{q:<8} {n_mod:<6} {'N/A':<8} {'N/A':<8} {n_matched:<10} {'N/A'}")
            continue

        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        score = ari  # 直接用 ARI 评分

        print(f"{q:<8} {n_mod:<6} {ari:<8.4f} {nmi:<8.4f} {n_matched:<10} {score:<10.4f}")
        results.append((q, n_mod, ari, nmi, n_matched, score))

    results.sort(key=lambda r: r[5], reverse=True)
    best_q, best_n, best_ari, best_nmi, best_matched, best_score = results[0]

    print(f"{'─'*55}")
    print(f"最优: q={best_q}, 模块数={best_n}, ARI={best_ari:.4f}, NMI={best_nmi:.4f}")
    print(f"Top 5:")
    for r in results[:5]:
        print(f"  q={r[0]:<6} 模块={r[1]:<4} ARI={r[2]:.4f} NMI={r[3]:.4f} 匹配={r[4]} 得分={r[5]:.4f}")

    return best_q


def compare_lineage_enrichment(grn_obj, prev_results_dir,
                               pvalue_threshold=0.05):
    """对比当前 Louvain 分群与之前成果的基因模块划分

    使用 domain_module_gene.txt（全基因→模块）和 tf_link_modules.csv（模块→domain）
    """
    import numpy as np
    from collections import Counter

    # --- 1. 加载先前成果 ---
    gene_mod_path = prev_results_dir + r'\module_split&GO\domain_module_gene.txt'
    tf_link_path = prev_results_dir + r'\tf_link_modules.csv'

    prev_gene_mod = pd.read_csv(gene_mod_path, sep='\t')
    tf_link = pd.read_csv(tf_link_path)

    # SMESG → previous_module (1-8)
    smesg_to_prev_mod = {}
    for col in prev_gene_mod.columns:
        mod_num = int(col[1:])  # M1 → 1
        for smesg in prev_gene_mod[col].dropna():
            sid = smesg.strip()
            if sid and sid not in smesg_to_prev_mod:
                smesg_to_prev_mod[sid] = mod_num

    # previous module → domain
    prev_mod_to_domain = {}
    for mod in tf_link['tf_module'].unique():
        top_domain = tf_link[tf_link['tf_module'] == mod]['domain'].value_counts().index[0]
        prev_mod_to_domain[mod + 1] = top_domain

    prev_genes = set(smesg_to_prev_mod.keys())

    # --- 2. 当前模块的基因分配 ---
    current_module_genes = {}  # mod → [SMESG IDs]
    for gene_idx, mod in grn_obj.partition.items():
        smesg_id = grn_obj.index_togene.get(gene_idx, str(gene_idx))
        if mod not in current_module_genes:
            current_module_genes[mod] = []
        current_module_genes[mod].append(smesg_id)

    # --- 3. 只取同时在两个结果中的基因 ---
    current_all_genes = set()
    for genes in current_module_genes.values():
        current_all_genes.update(genes)
    common_genes = prev_genes & current_all_genes

    print(f"\n{'='*70}")
    print(f"对比: 当前分群 vs 先前成果")
    print(f"  数据来源:")
    print(f"    先前成果: {gene_mod_path}")
    print(f"    先前 Domain: {tf_link_path}")
    print(f"    当前分群: grn_obj.partition (Louvain 聚类, 来自 example_data/*.csv)")
    print(f"  先前: {len(prev_genes)} 基因 → {len(set(smesg_to_prev_mod.values()))} 模块")
    print(f"  当前: {len(current_all_genes)} 基因 → {len(current_module_genes)} 模块")
    print(f"  交集: {len(common_genes)} 个基因")
    print(f"{'='*70}")

    # --- 4. 建立交叉表：当前模块 × 先前模块 ---
    prev_mods = sorted(set(smesg_to_prev_mod.values()))
    curr_mods = sorted(current_module_genes.keys())

    # 交叉表
    cross = {}
    for cm in curr_mods:
        cross[cm] = {}
        for pm in prev_mods:
            cross[cm][pm] = 0

    for smesg_id in common_genes:
        gene_idx = grn_obj.gene_idx.get(smesg_id)
        if gene_idx is not None and gene_idx in grn_obj.partition:
            cm = grn_obj.partition[gene_idx]
            pm = smesg_to_prev_mod[smesg_id]
            cross[cm][pm] += 1

    # --- 5. 输出 ---
    print(f"\n{'模块':<6} {'基因数':<8} {'最优匹配先前模块':<25} {'重叠':<8} {'Domain':<20}")
    print(f"{'─'*70}")

    for cm in curr_mods:
        c_total = sum(cross[cm].values())
        best_pm = max(cross[cm], key=cross[cm].get)
        best_overlap = cross[cm][best_pm]
        best_ratio = best_overlap / max(c_total, 1)
        best_domain = prev_mod_to_domain.get(best_pm, '?')

        print(f"{cm:<6} {c_total:<8} M{best_pm} (重叠={best_overlap}, {best_ratio:.1%}){'':<8} "
              f"{best_overlap:<8} {best_domain:<20}")

    # --- 6. 详细交叉表 ---
    print(f"\n{'='*70}")
    print(f"交叉表: 当前模块 (行) × 先前模块 (列)")
    print(f"{'='*70}")
    header_name = "cur\\prev"
    header = f"{header_name:<10}"
    for pm in prev_mods:
        domain = prev_mod_to_domain.get(pm, '?')
        header += f" {'M'+str(pm):>6}"
    print(header)
    print(f"{'─'*(10 + 7*len(prev_mods))}")
    for cm in curr_mods:
        row = f"模块{cm:<7}"
        for pm in prev_mods:
            row += f" {cross[cm][pm]:>6}"
        print(row)

    # Domain 对照
    print(f"\n先前模块 → Domain 对照:")
    for pm in prev_mods:
        print(f"  M{pm} → {prev_mod_to_domain.get(pm, '?')}")
    print(f"\nDomain 说明:")
    print(f"  epi_l=表皮  blasterma_nfy=芽基(NFY)  neu=神经  gut_i=肠道")
    print(f"  new_domain1=新domain1  new_domain2=新domain2  epi_ap=表皮顶  pharynx=咽部")


def new_edge():
    # ===================================================================
    # 1. 读取数据
    # ===================================================================
    # example_data 中 CSV 第一列（无列名）为基因 ID，使用 index_col=0
    rna_features = pd.read_csv(os.path.join(DATA_DIR, 'rna_pca.csv'), index_col=0)
    print('=== RNA features ===')
    print(rna_features.head())

    # 保留 index 中不包含 'Unknown' 的行
    filtered_rna_features = rna_features[~rna_features.index.str.contains('Unknown', na=False)]
    print(f'RNA: {len(filtered_rna_features)} genes (filtered from {len(rna_features)})')

    atac_features = pd.read_csv(os.path.join(DATA_DIR, 'atac_pca.csv'), index_col=0)
    print('=== ATAC features ===')
    print(atac_features.head())

    # 保留 index 中不包含 'Unknown' 的行
    filtered_atac_features = atac_features[~atac_features.index.str.contains('Unknown', na=False)]
    print(f'ATAC: {len(filtered_atac_features)} genes (filtered from {len(atac_features)})')

    # ===================================================================
    # 2. 标准化
    # ===================================================================
    # 对 RNA 特征按行标准化
    normalized_rna = (filtered_rna_features.T - filtered_rna_features.mean(axis=1)) / filtered_rna_features.std(axis=1)
    filtered_rna_features = normalized_rna.T

    # 对 ATAC 特征按行标准化
    normalized_atac = (filtered_atac_features.T - filtered_atac_features.mean(axis=1)) / filtered_atac_features.std(axis=1)
    filtered_atac_features = normalized_atac.T

    # ===================================================================
    # 3. 读取边和 TF motif 数据
    # ===================================================================
    edges = pd.read_csv(os.path.join(DATA_DIR, 'edges.csv'), index_col=0)
    print('=== Edges ===')
    print(edges.head())
    edges = edges.dropna(subset=['source', 'target'])
    print(f'Edges: {len(edges)} after dropping NaN')

    # ===================================================================
    # 4. 读取 cell marker 基因字典（cell_marker.pkl）
    # ===================================================================
    with open(os.path.join(DATA_DIR, 'cell_marker.pkl'), 'rb') as f:
        gene_marker_dict = pickle.load(f)
    print(f'Cell types in marker dict: {list(gene_marker_dict.keys())}')
    for k, v in gene_marker_dict.items():
        print(f'  {k}: {len(v)} marker genes')

    # ===================================================================
    # 5. 数据来源统计：解释为什么当前 GRN 用 ~1800 个基因，而 Excel 有 ~1600 个 TF
    # ===================================================================
    print('\n>>> Step 4b: 数据来源统计')
    # --- 5a. 当前 GRN 数据流 ---
    print(f'\n--- 当前 GRN 数据流 ---')
    print(f'  RNA 原始基因数:    {len(rna_features)}')
    print(f'  RNA 过滤 Unknown 后: {len(filtered_rna_features)}')
    print(f'  ATAC 原始基因数:   {len(atac_features)}')
    print(f'  ATAC 过滤 Unknown 后:{len(filtered_atac_features)}')
    common_genes = sorted(filtered_atac_features.index.intersection(filtered_rna_features.index))
    print(f'  RNA ∩ ATAC 交集:   {len(common_genes)}  ← GRN 实际使用的基因')

    # --- 5b. TF motif 统计 ---
    tf_motif = pd.read_csv(os.path.join(DATA_DIR, 'tf_gene_motif.csv'), index_col=0)
    motif_smesg = set(tf_motif['smesg'].dropna())
    common_tf = set(common_genes) & motif_smesg
    common_target = set(common_genes) - motif_smesg
    print(f'\n--- 1828 个基因的类型构成 ---')
    print(f'  转录因子 (TF, 在 tf_gene_motif 中): {len(common_tf)}')
    print(f'  靶基因 (非 TF):                    {len(common_target)}')

    # --- 5c. Excel 文件统计 ---
    xlsx_path = r'F:\DongYiFei\涡虫GRN\target gene lists\table2.Lineage enriched motifs_TFs.xlsx'
    try:
        xl = pd.ExcelFile(xlsx_path)
        excel_tf_all = set()
        for sheet in xl.sheet_names:
            df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet)
            excel_tf_all.update(df_sheet['TF'].dropna().unique())
        print(f'\n--- Excel 文件 (Lineage enriched motifs) ---')
        print(f'  Sheet 数: {len(xl.sheet_names)}')
        print(f'  每 sheet 约 1646 行（TF 富集分数表）')
        print(f'  全部唯一 TF (基因符号名): {len(excel_tf_all)}')

        # TF 符号 → SMESG 的映射
        tf_name_to_smesg = dict(zip(tf_motif.index, tf_motif['smesg']))
        excel_matched = excel_tf_all & set(tf_name_to_smesg.keys())
        excel_smesg = set(tf_name_to_smesg[n] for n in excel_matched if n in tf_name_to_smesg)
        excel_in_grn = excel_smesg & set(common_genes)
        print(f'  匹配到 SMESG 的 TF: {len(excel_smesg)}')
        print(f'  其中在当前 GRN 1828 基因中: {len(excel_in_grn)}')
        print(f'\n  → Excel 的 ~1600 个 TF (符号名) 与 GRN 的 1828 个基因 (SMESG ID)')
        print(f'  → 是不同层面的数据：TF vs 全部基因，不能直接比较数量')
        print(f'  → TF 是 GRN 基因的一个子集（{len(common_tf)}/{len(common_genes)}），其余为靶基因')
    except Exception as e:
        print(f'\n--- Excel 文件读取失败: {e} ---')

    # ===================================================================
    # 6. 取交集 + 排序
    # ===================================================================
    print(f'\nCommon genes (intersection): {len(common_genes)}')

    # Step 2: 保留 source 和 target 都在 common_genes 中的行
    filtered_edges = edges[
        edges['source'].isin(common_genes) &
        edges['target'].isin(common_genes)
    ]
    filtered_edges = filtered_edges.copy()
    filtered_edges.loc[:, 'weight'] = filtered_edges['weight'] * 100
    print(f'Filtered edges (both ends in common_genes): {len(filtered_edges)}')

    # Step 3: 获取 edges 中第一个出现的基因（source 或 target）
    first_edge_gene = None
    first_row = filtered_edges.iloc[0]
    if first_row['source'] in common_genes:
        first_edge_gene = first_row['source']
    elif first_row['target'] in common_genes:
        first_edge_gene = first_row['target']

    # Step 4: 剩余基因（去除 first_edge_gene）
    rest_genes = [g for g in common_genes if g != first_edge_gene]

    # Step 5: 筛选 rest_genes 中实际在 edges 中出现的基因，随机选最后一个基因
    edge_genes = set(filtered_edges['source']).union(set(filtered_edges['target']))
    valid_rest_genes = [g for g in rest_genes if g in edge_genes]
    last_gene = random.choice(valid_rest_genes)

    # Step 6: 构建新顺序
    middle_genes = [g for g in rest_genes if g != last_gene]
    ordered_genes = [first_edge_gene] + middle_genes + [last_gene]

    # Step 7: 根据新顺序提取特征
    filtered_atac_common = filtered_atac_features.loc[common_genes].loc[ordered_genes]
    filtered_rna_common = filtered_rna_features.loc[common_genes].loc[ordered_genes]

    # ===================================================================
    # 6. 读取 TF motif 表
    # ===================================================================
    gene_csv = pd.read_csv(os.path.join(DATA_DIR, 'tf_gene_motif.csv'), index_col=0)
    tf_gene_motif = gene_csv
    print('=== TF gene motif ===')
    print(tf_gene_motif.head())
    print(f'Total TFs: {len(tf_gene_motif)}')

    # ===================================================================
    # 7. 创建 Grnm 对象并运行分析流程
    # ===================================================================
    grnexample = grnm2.Grnm(filtered_rna_common, filtered_atac_common, filtered_edges, tf_gene_motif)

    # PCA 降维
    print('\n>>> Step 1: PCA')
    grnexample.pca(d1=60)
    print(f'PCA_r_data shape: {grnexample.PCA_r_data.shape}')
    print(f'PCA_a_data shape: {grnexample.PCA_a_data.shape}')
    print(f'gene_idx size: {len(grnexample.gene_idx)}')

    # 构建图网络
    print('\n>>> Step 2: Graph')
    grnexample.graph()
    print(grnexample.g)

    # 图自编码器训练
    print('\n>>> Step 3: GAE')
    grnexample.gae(d2=8, epoch_n=100, alpha_n=0.01)
    print(f'Embedding z shape: {grnexample.z.shape}')

    # UMAP 降维可视化
    print('\n>>> Step 4: UMAP')
    grnexample.umap()
    print(f'UMAP data shape: {grnexample.umap_data.shape}')

    # Louvain 社区检测
    print('\n>>> Step 5: Louvain')
    domain_gene_txt = r'F:\涡虫GRN\GRN生信图\4_GRN\match_domain\module_split&GO\domain_module_gene.txt'

    # 扫描 resolution 参数，以先前成果的全基因模块划分为 ground truth
    best_q = optimize_louvain_resolution(grnexample, domain_gene_txt)
    print(f'最优 resolution: {best_q}')

    # 使用最优参数运行 Louvain
    grnexample.louvain(q=best_q)
    partion = grnexample.partition
    num_modules = len(set(partion.values()))
    print(f'Number of modules: {num_modules}')

    # 展示每个社区的基因 ID（每个模块最多显示 20 个）
    grnexample.show_community_genes(top_n=20)

    # ===================================================================
    # 6. 与之前成果的全基因模块划分对比
    # ===================================================================
    print('\n>>> Step 5b: 与先前成果对比')
    prev_dir = r'F:\涡虫GRN\GRN生信图\4_GRN\match_domain'
    compare_lineage_enrichment(grnexample, prev_dir)

    # ===================================================================
    # 8. Marker 基因热图
    # ===================================================================
    print('\n>>> Step 6: Match Heatmap')
    grnexample.match_heat(celltypes_marker=gene_marker_dict)
    print(f'Heatmap saved to {os.path.join(PNG_DIR, "test.png")}')

    # ===================================================================
    # 9. 聚类评估
    # ===================================================================
    print('\n>>> Step 7: DB & CH Score')
    grnexample.DB_CH_score()

    # ===================================================================
    # 10. GRN 网络可视化
    # ===================================================================
    print('\n>>> Step 8: GRN Plot')
    # 选取一些 TF 基因来标注（注意：example_data 使用 SMESG 命名，ENS 基因可能不在数据中）
    # 如果 ENS 基因不在数据中，会自动忽略
    highlight_genes = ['SMESG000000077', 'SMESG000000110', 'SMESG000000959']
    print(f'Highlight genes: {highlight_genes}')
    grnexample.grn_plot('y', highlight_genes)
    print(f'GRN network saved to {os.path.join(PNG_DIR, "grn_network.png")}')

    # ===================================================================
    # 11. 所有图片已保存到 png/ 目录
    # ===================================================================
    print('\n>>> 所有图片已保存完成！')
    print(f'    - 热图: {os.path.join(PNG_DIR, "test.png")}')
    print(f'    - GRN 网络: {os.path.join(PNG_DIR, "grn_network.png")}')


if __name__ == '__main__':
    new_edge()
