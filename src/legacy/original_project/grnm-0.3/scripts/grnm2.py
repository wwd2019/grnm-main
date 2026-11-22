import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

from dimensionality_reduction.pca_preprocessing import PCA
from dimensionality_reduction.umap_preprocessing import UMAPReduction
from dimensionality_reduction.graph_preprocessing import GraphGeneration
from dimensionality_reduction.graph_autoencoder import GraphAutoencoderTrainer
from clustering.louvain_clustering import *
from plotting.generate_heatmap import *
from plotting.grn_visualization import grn_net
from plotting.module_target_visualization import tf_moudle
import numpy as np
class Grnm():
    def __init__(self, rnadata, atacdata, edges, tf_gene_motif):
        # Initialize with input data
        self.rnadata = rnadata
        self.atacdata = atacdata
        self.edges = edges
        self.tf_gene_motif = tf_gene_motif
        # print(self.tf_gene_motif)
        self.gene_name_dict = dict(zip(self.tf_gene_motif['smesg'], self.tf_gene_motif.index))
        print(self.gene_name_dict)
        # Get gene names and create gene index dictionary
        self.gene_name = rnadata.index.to_list()
        self.gene_idx = {self.gene_name[i]: i for i in range(len(self.gene_name))}

        self.index_togene =  {i:self.gene_name[i] for i in range(len(self.gene_name))}
    def pca(self, d1):
        # Perform PCA on RNA and ATAC data
        PCA_r = PCA(n_components=d1)
        PCA_r.fit_transform(self.rnadata)
        self.PCA_r_data = PCA_r.PCA_data
        PCA_a = PCA(n_components=d1)
        PCA_a.fit_transform(self.atacdata)
        self.PCA_a_data = PCA_a.PCA_data

    def graph(self):
        # Generate graph using RNA and ATAC data, edges, and gene index
        print('gene_names','gene_idx',len(self.gene_name), len(self.gene_idx))
        gn = GraphGeneration(self.PCA_r_data, self.PCA_a_data, self.edges, self.gene_idx, self.tf_gene_motif)
        gn.graph_generation()
        self.g = gn.g

    def gae(self, d2=8, epoch_n=1000, alpha_n=0.01):
        # Perform GAE on graph
        gae_ = GraphAutoencoderTrainer(self.g, epoch_n, alpha_n)
        gae_.train(n_components=d2)
        self.z = gae_.z

    def umap(self):
        # Perform UMAP on GAE embedding
        UMAP = UMAPReduction()
        UMAP.fit_transform(self.z)
        self.umap_data = UMAP.umapdata

    def louvain(self,q):
        # Apply Louvain algorithm for community detection

        Louvain_ = Louvain(self.g, q)
        Louvain_.run_louvain()
        self.partition = Louvain_.get_partition()
        num_modules = len(set(self.partition.values()))
        print(f"模块数量: {num_modules}")

    def show_community_genes(self, top_n=None):
        """展示每个社区（模块）包含的基因 ID"""
        # 按模块分组基因
        module_genes = {}
        for gene_idx, mod in self.partition.items():
            if mod not in module_genes:
                module_genes[mod] = []
            gene_id = self.index_togene.get(gene_idx, str(gene_idx))
            module_genes[mod].append(gene_id)

        print(f"\n{'='*60}")
        print(f"Louvain 社区检测结果: 共 {len(module_genes)} 个模块")
        print(f"{'='*60}")
        for mod in sorted(module_genes.keys()):
            genes = module_genes[mod]
            print(f"\n--- 模块 {mod}: {len(genes)} 个基因 ---")
            if top_n and len(genes) > top_n:
                print(f"  {', '.join(genes[:top_n])} ... (仅显示前 {top_n} 个)")
            else:
                print(f"  {', '.join(genes)}")

    def DB_CH_score(self):
        """
        Compute Davies–Bouldin Index and Calinski–Harabasz Score based on UMAP-reduced embeddings and Louvain clustering.

        Returns:
            db_score (float): Davies–Bouldin Index (lower is better).
            ch_score (float): Calinski–Harabasz Score (higher is better).

        """
        # 只选择 partition 有定义的节点
        partition_keys = sorted(self.partition.keys())  # 6899 个节点
        umap_sub = self.umap_data[partition_keys]  # 对应的 UMAP 向量
        labels = np.array([self.partition[i] for i in partition_keys])  # partition 的聚类标签
        pca_sub_a = self.PCA_a_data[partition_keys]
        pca_sub_r = self.PCA_r_data[partition_keys]
        # 打印检查
        print(f"用于评估的节点数: {len(labels)}")

        # 计算指标
        db_score = davies_bouldin_score(umap_sub, labels)
        ch_score = calinski_harabasz_score(umap_sub, labels)

        print(f"Davies–Bouldin 指数（越小越好）: {db_score:.4f}")
        print(f"Calinski–Harabasz 分数（越大越好）: {ch_score:.4f}")
        # for background
        db_score_a = davies_bouldin_score(pca_sub_a, labels)
        ch_score_a = calinski_harabasz_score(pca_sub_a, labels)
        print(f"atac Davies–Bouldin 指数（越小越好）: {db_score_a:.4f}")
        print(f"atac Calinski–Harabasz 分数（越大越好）: {ch_score_a:.4f}")

        db_score_r = davies_bouldin_score(pca_sub_r, labels)
        ch_score_r = calinski_harabasz_score(pca_sub_r, labels)
        print(f"rna Davies–Bouldin 指数（越小越好）: {db_score_r:.4f}")
        print(f"rna Calinski–Harabasz 分数（越大越好）: {ch_score_r:.4f}")


        self.db_score = db_score
        self.ch_score = ch_score
        return db_score, ch_score

    def match_heat(self, celltypes_marker):
        # Calculate overlap between modules and cell types markers
        moudels_dict = {}
        for moudel in self.partition.items():
            if moudel[1] in moudels_dict:
                moudels_dict[moudel[1]].append(self.index_togene[moudel[0]])
            else:
                moudels_dict[moudel[1]] = [self.index_togene[moudel[0]]]
        self.moudles = moudels_dict

        match = HeatmapGenerator(moudels_dict, celltypes_marker)
        print(match)
        match.generate_overlap_heatmap()

    def grn_plot(self,label_dis='y',gene_list_dis=[]):
        # Visualize gene regulatory network
        df = pd.DataFrame(self.umap_data, columns=['x', 'y'])
        df['tf_name'] = self.gene_name
        print(len(self.gene_name)) # 18145
        # print(self.gene_name)
        df['module'] = self.partition
        print(df)
        print(len(self.partition)) # 6899
        grnm_p = grn_net(df, self.edges, self.gene_idx, self.gene_name_dict)

        df['gene_id'] = self.gene_name
        df['gene_name'] = [self.gene_name_dict.get(id, id) for id in self.gene_name]
        # 提取 source 和 target 列的集合
        sources = set(self.edges['source'])
        targets = set(self.edges['target'])

        def get_role(gene_id):
            is_source = gene_id in sources
            is_target = gene_id in targets
            if is_source and is_target:
                return 'both'
            elif is_source:
                return 'source'
            elif is_target:
                return 'target'
            else:
                return 'none'

        # 添加角色列
        df['grn_role'] = df['gene_id'].apply(get_role)
        df.to_csv("../raw_data/gene_table.csv", index=False)
        self.edges.to_csv("../raw_data/edge_table.csv", index=False)
        print('save end')
        grnm_p.grn_plot(label_dis, gene_list_dis)



    def single_grn_plot(self, nodes,label_dis='y',gene_list_dis=[]):
        # Visualize single gene regulatory network
        df = pd.DataFrame(self.umap_data, columns=['x', 'y'])
        df['tf_name'] = self.gene_name
        df['module'] = self.partition
        grnm_sp = grn_net(df, self.edges, self.gene_idx, self.gene_name_dict)
        grnm_sp.s_grn_net_plot(nodes,label_dis,gene_list_dis)

    def tf_target_scatterplot(self,tf_list,tf_module):
        # Plot scatterplot for TF target genes
        print(tf_list)
        tf_ratio = []
        for tf in tf_list:
            num = 0
            num_all = 0
            tf_id = self.gene_idx[tf]
            connected_nodes = self.g[tf_id]
            for node in connected_nodes:
                par_value = self.partition[node]
                if par_value is not None:
                    if (self.partition[tf_id] == par_value):
                        num += 1
                num_all += 1
            tf_ratio.append(num / num_all)
        tf_m = tf_moudle(tf_list, tf_module, tf_ratio)
        tf_m.plot_scatter()
# # #
# import pandas as pd
# time = '72'
# rna_features = pd.read_csv('D:/work/lmy/rna_pca_'+time+'.csv', index_col=0)
# atac_features = pd.read_csv('D:/work/lmy/atac_pca_'+time+'.csv', index_col=0)
# edges = pd.read_csv('D:/work/lmy/edges_'+time+'.csv', index_col=0)
#
# tf_gene_motif = pd.read_csv('D:/work/lmy/tf_gene_motif.csv', index_col=0)
#
# gcn = Grnm(rna_features,atac_features,edges,tf_gene_motif)
# gcn.g_pca(10)
# gcn.g_graph()
# gcn.g_gae(8,100,0.01)
# #
# gcn.g_louvain(0.5)
# gcn.g_umap()
# # print(gcn.partition)
#
# gcn.g_match_heat(celltype_markers)
# gcn.g_grn_plot('y',['SMESG000000077', 'SMESG000000110', 'SMESG000000959', 'SMESG000001687', 'SMESG000002697', 'SMESG000002774'])
# gcn.g_single_grn_plot('SMESG000001287','y',['SMESG000001287','SMESG000000077', 'SMESG000000110', 'SMESG000000959', 'SMESG000001202', 'SMESG000001687', 'SMESG000002697', 'SMESG000002774', 'SMESG000002907'])
# # print(gcn.partition)
# gcn.g_tf_target_scatterplot()





