# 用于产生图片
import pandas as pd
from grnm import Grnm
rna_features = pd.read_csv('../example_data/rna_pca.csv', index_col=0)
atac_features = pd.read_csv('../example_data/atac_pca.csv', index_col=0)
edges = pd.read_csv('../example_data/edges.csv', index_col=0)
print('rna',len(rna_features))
print('atac',len(atac_features))
tf_gene_motif = pd.read_csv('../example_data/tf_gene_motif.csv', index_col=0)
print(tf_gene_motif.head())
print('gene',len(tf_gene_motif))
import pickle
with open('../example_data/cell_marker.pkl', 'rb') as file:
    # 从文件中加载序列化的对象
    marker_gene = pickle.load(file)
# print(marker_gene) # 各个种群的marker基因
grnexample = Grnm(rna_features, atac_features, edges, tf_gene_motif)
grnexample.pca(d1=60)
grnexample.graph()
grnexample.gae(d2=8,epoch_n=500,alpha_n=0.01)
# # The parameter 'q' controls the resolution of the community detection
# def louvain_test(q):
#     grnexample.louvain(q)
#     #You can get the division of the module through this method
#     partion = grnexample.partition
#     grnexample.umap()
#     # grnexample.match_heat(celltypes_marker=marker_gene)
#     genes_list = []
#     grnexample.grn_plot('y',['SMESG000000077', 'SMESG000000110', 'SMESG000000959', 'SMESG000001687', 'SMESG000002697', 'SMESG000002774'])
#    # grnexample.single_grn_plot('SMESG000001287','y',['SMESG000001287','SMESG000000077', 'SMESG000000110', 'SMESG000000959', 'SMESG000001202', 'SMESG000001687', 'SMESG000002697', 'SMESG000002774', 'SMESG000002907'])
# for q in range(1,10,1):
#     louvain_test(q)
#     # print(q)
grnexample.louvain(q=2)
# You can get the division of the module through this method
partion = grnexample.partition
grnexample.umap()
# grnexample.match_heat(celltypes_marker=marker_gene)
genes_list = []
gene_name_dict = dict(zip(tf_gene_motif.index ,tf_gene_motif['smesg']))
print(gene_name_dict)
import pandas as pd

# 转换成 DataFrame
df = pd.DataFrame(list(gene_name_dict.items()), columns=["tfname", "smesg"])

# 保存成 CSV
df.to_csv("../example_data/gene_name_dict.csv", index=False)

print("已保存到 gene_name_dict.csv")

gene_name = ['NFYC','MDH1','MDH2','IDH2']
smesg_id = ['SMESG000077821','SMESG000070444','SMESG000008924','SMESG000049154']
smesg_id = [gene_name_dict[gene] for gene in gene_name if gene in gene_name_dict]
print(smesg_id)
grnexample.grn_plot('y', smesg_id)
# grnexample.match_heat(celltypes_marker=marker_gene)
# grnexample.tf_target_scatterplot(tf_list=tf_list_,tf_module=tf_module)
#
#
#
#


