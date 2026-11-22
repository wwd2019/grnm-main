# 用于产生图片
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import grnm
rna_features = pd.read_csv('../example_data/rna_pca.csv', index_col=0)
atac_features = pd.read_csv('../example_data/atac_pca.csv', index_col=0)
edges = pd.read_csv('../example_data/edges.csv', index_col=0)
tf_gene_motif = pd.read_csv('../example_data/tf_gene_motif.csv', index_col=0)
print(tf_gene_motif.head())
import pickle
with open('../example_data/cell_marker.pkl', 'rb') as file:
    # 从文件中加载序列化的对象
    marker_gene = pickle.load(file)
# print(marker_gene) # 各个种群的marker基因

grnexample = grnm.Grnm(rna_features, atac_features, edges, tf_gene_motif)
grnexample.pca(d1=60)
grnexample.graph()
grnexample.gae(d2=8, epoch_n=500, alpha_n=0.01)

# The parameter 'q' controls the resolution of the community detection
q = 0.5
grnexample.louvain(q)
grnexample.umap()

# 查看模块划分
partition = grnexample.partition
print("Partition (first 10 items):", dict(list(partition.items())[:10]))

# DB/CH 聚类评估指标
grnexample.DB_CH_score()

# 模块-细胞类型 marker 热图（需要用 marker 数据时取消注释）
# grnexample.match_heat(celltypes_marker=marker_gene)

# GRN 网络可视化
gene_list = ['SMESG000000077', 'SMESG000000110', 'SMESG000000959',
             'SMESG000001687', 'SMESG000002697', 'SMESG000002774']
grnexample.grn_plot('y', gene_list)

# 单基因 GRN 可视化（需要用的时候取消注释）
# grnexample.single_grn_plot(
#     'SMESG000001287', 'y',
#     ['SMESG000001287', 'SMESG000000077', 'SMESG000000110', 'SMESG000000959',
#      'SMESG000001202', 'SMESG000001687', 'SMESG000002697', 'SMESG000002774',
#      'SMESG000002907']
# )
#
#
#
#


